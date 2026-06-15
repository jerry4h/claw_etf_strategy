import pandas as pd
import numpy as np

df = pd.read_csv('meta_data/all_etfs_nav_2013_2026_merged.csv', index_col=0, parse_dates=True)
df_weekly = df.resample('W-MON').last().dropna(how='all').ffill().bfill()

pe_pct = pd.read_csv('/mnt/agents/pe_percentile_weekly.csv', index_col=0, parse_dates=True)
pe_weekly = pe_pct.reindex(df_weekly.index, method='ffill').bfill().values.flatten()

w_prices = df_weekly.values
w_index = df_weekly.index
n_weeks = len(w_index)
w_rets = np.diff(w_prices, axis=0) / w_prices[:-1]

w_mom4 = np.full((n_weeks, 5), np.nan)
for i in range(4, n_weeks):
    w_mom4[i] = np.prod(1 + w_rets[i-4:i], axis=0) - 1

w_vol20 = np.full((n_weeks, 5), np.nan)
for i in range(20, n_weeks):
    w_vol20[i] = w_rets[i-20:i].std(axis=0) * np.sqrt(52)

FEE_RATE = 0.00005

def run_once(params):
    mom_w, vol_w, val_w, def_alloc, step_low, step_high, max_def, hongli_ratio = params
    start_idx = 20
    nav = 1.0
    peak = 1.0
    last_alloc = np.zeros(5)
    max_dd = 0.0
    in_stop_loss = False
    stop_loss_weeks = 0
    
    for i in range(start_idx, n_weeks - 1):
        scores = np.zeros(5)
        for j in [0, 2, 3]:
            if not np.isnan(w_mom4[i, j]) and not np.isnan(w_vol20[i, j]):
                scores[j] = mom_w * w_mom4[i, j] - vol_w * w_vol20[i, j] + val_w * (0.5 - pe_weekly[i] / 100.0)
        
        off_scores = [(scores[j], j) for j in [0, 2, 3]]
        off_scores.sort(reverse=True)
        selected_off = [j for _, j in off_scores[:2]]
        
        nasdaq_vol = w_vol20[i, 0]
        if np.isnan(nasdaq_vol):
            def_ratio = def_alloc
        elif nasdaq_vol < step_low:
            def_ratio = def_alloc
        elif nasdaq_vol > step_high:
            def_ratio = max_def
        else:
            def_ratio = def_alloc + (max_def - def_alloc) * (nasdaq_vol - step_low) / (step_high - step_low)
        
        if not in_stop_loss and nav < peak * 0.92:
            in_stop_loss = True
            stop_loss_weeks = 0
        
        if in_stop_loss:
            def_ratio = max(def_ratio, 0.95)
            stop_loss_weeks += 1
            if stop_loss_weeks >= 4:
                in_stop_loss = False
        
        alloc = np.zeros(5)
        alloc[1] = def_ratio * hongli_ratio
        alloc[4] = def_ratio * (1 - hongli_ratio)
        for j in selected_off:
            alloc[j] = (1 - def_ratio) / len(selected_off)
        
        turnover = np.sum(np.abs(alloc - last_alloc))
        fee_cost = turnover * FEE_RATE
        
        wret = sum(alloc[j] * w_rets[i, j] for j in range(5) if not np.isnan(w_rets[i, j]))
        nav *= (1 + wret - fee_cost)
        peak = max(peak, nav)
        
        dd = (peak - nav) / peak
        if dd > max_dd:
            max_dd = dd
        
        last_alloc = alloc
    
    total_ret = nav - 1
    n_weeks_used = n_weeks - 1 - start_idx
    annual_ret = (1 + total_ret) ** (52 / n_weeks_used) - 1
    return annual_ret, max_dd

print("Searching for ret>15%, DD<10%...")
results = []

# Focused parameter grid around known sweet spots
for mom_w in [0.40, 0.45, 0.50, 0.55]:
    for vol_w in [0.15, 0.20, 0.25, 0.30]:
        for val_w in [0.05, 0.10, 0.15]:
            if mom_w + vol_w + val_w > 1.0:
                continue
            for def_alloc in [0.15, 0.20, 0.25]:
                for step_low in [0.15, 0.20, 0.25, 0.30]:
                    for step_high in [0.25, 0.30, 0.35, 0.40, 0.50]:
                        if step_high <= step_low:
                            continue
                        for max_def in [0.70, 0.75, 0.80, 0.85, 0.90]:
                            for hongli in [0.0, 0.25, 0.50, 0.75]:
                                ann, dd = run_once((mom_w, vol_w, val_w, def_alloc, step_low, step_high, max_def, hongli))
                                if ann > 0.15 and dd < 0.10:
                                    results.append({
                                        'mom_w': mom_w, 'vol_w': vol_w, 'val_w': val_w,
                                        'def_alloc': def_alloc, 'step_low': step_low,
                                        'step_high': step_high, 'max_def': max_def, 'hongli': hongli,
                                        'annual_return': ann, 'max_drawdown': dd,
                                        'sharpe': ann / dd
                                    })

print(f"Found {len(results)} combinations with ret>15%, DD<10%")
if len(results) > 0:
    results_df = pd.DataFrame(results)
    print("\nTop 10 by Sharpe:")
    print(results_df.nlargest(10, 'sharpe').to_string(index=False))
    print("\nTop 10 by Return:")
    print(results_df.nlargest(10, 'annual_return').to_string(index=False))
else:
    print("\nNo exact matches. Running broader search for closest configs...")
    all_results = []
    for mom_w in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        for vol_w in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
            for val_w in [0.05, 0.10, 0.15, 0.20]:
                if mom_w + vol_w + val_w > 1.0:
                    continue
                for def_alloc in [0.10, 0.15, 0.20, 0.25, 0.30]:
                    for step_low in [0.10, 0.15, 0.20, 0.25, 0.30]:
                        for step_high in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
                            if step_high <= step_low:
                                continue
                            for max_def in [0.60, 0.70, 0.80, 0.90, 1.00]:
                                for hongli in [0.0, 0.25, 0.50, 0.75, 1.00]:
                                    ann, dd = run_once((mom_w, vol_w, val_w, def_alloc, step_low, step_high, max_def, hongli))
                                    all_results.append({
                                        'mom_w': mom_w, 'vol_w': vol_w, 'val_w': val_w,
                                        'def_alloc': def_alloc, 'step_low': step_low,
                                        'step_high': step_high, 'max_def': max_def, 'hongli': hongli,
                                        'annual_return': ann, 'max_drawdown': dd
                                    })
    all_df = pd.DataFrame(all_results)
    all_df['score'] = all_df['annual_return'] - 0.5 * all_df['max_drawdown']
    print("\nTop 15 by risk-adjusted score (return - 0.5*DD):")
    print(all_df.nlargest(15, 'score')[['mom_w','vol_w','val_w','def_alloc','step_low','step_high','max_def','hongli','annual_return','max_drawdown']].to_string(index=False))
    
    # Show configs closest to 15%/10%
    all_df['distance'] = np.sqrt((0.15 - all_df['annual_return'])**2 + (0.10 - all_df['max_drawdown'])**2)
    print("\nClosest 15 to target (15% ret, 10% DD):")
    print(all_df.nsmallest(15, 'distance')[['mom_w','vol_w','val_w','def_alloc','step_low','step_high','max_def','hongli','annual_return','max_drawdown','distance']].to_string(index=False))
