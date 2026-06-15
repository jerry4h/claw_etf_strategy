#!/usr/bin/env python3
"""
虾池ETF轮动策略 - 统一引擎 + 完整网格搜索

引擎定稿设计:
- 周频调仓 (W-MON)
- 4周动量 (累积收益率)
- 20周波动率 (年化)
- PE分位数估值 (5年滚动窗口)
- vol三段式防御 + 8%净值止损兜底
- 红利低波+国债防御层
- 纳指+沪深300+黄金进攻层
- 调仓费 0.05% (FEE=0.0005)

运行方式: python run_unified_search.py > search_log.txt 2>&1
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from multiprocessing import Pool, cpu_count

# ============ 数据加载 ============
print("[{}] Loading data...".format(datetime.now().strftime('%H:%M:%S')))

df = pd.read_csv('meta_data/all_etfs_nav_2013_2026_merged.csv', index_col=0, parse_dates=True)
df_weekly = df.resample('W-MON').last().dropna(how='all').ffill().bfill()

pe_pct = pd.read_csv('/mnt/agents/pe_percentile_weekly.csv', index_col=0, parse_dates=True)
pe_weekly = pe_pct.reindex(df_weekly.index, method='ffill').bfill().values.flatten()

w_prices = df_weekly.values
w_index = df_weekly.index
n_weeks = len(w_index)
w_rets = np.diff(w_prices, axis=0) / w_prices[:-1]

ETFS = ['纳指ETF', '红利低波ETF', '沪深300ETF', '黄金ETF', '国债ETF']
OFFENSIVE = [0, 2, 3]  # 纳指, 沪深300, 黄金
DEFENSIVE = [1, 4]     # 红利低波, 国债

# Precompute factors
w_mom4 = np.full((n_weeks, 5), np.nan)
for i in range(4, n_weeks):
    w_mom4[i] = np.prod(1 + w_rets[i-4:i], axis=0) - 1

w_vol20 = np.full((n_weeks, 5), np.nan)
for i in range(20, n_weeks):
    w_vol20[i] = w_rets[i-20:i].std(axis=0) * np.sqrt(52)

# Defense layer dynamic selection: 红利低波 only available 2019+
def get_defensive_etfs(date_idx):
    """返回可用的防御ETF索引"""
    if w_index[date_idx].year >= 2019:
        return [1, 4]  # 红利低波 + 国债
    else:
        return [4]     # 仅国债

print("[{}] Data ready: {} weeks".format(datetime.now().strftime('%H:%M:%S'), n_weeks))


# ============ 回测引擎 ============
FEE_RATE = 0.00005  # 0.005% turnover fee

def run_backtest(params):
    """
    统一回测引擎
    params = (mom_w, vol_w, val_w, top_n, def_alloc, step_low, step_high, max_def, hongli_ratio)
    """
    mom_w, vol_w, val_w, top_n, def_alloc, step_low, step_high, max_def, hongli_ratio = params
    
    start_idx = 20  # First valid vol20
    nav = 1.0
    peak = 1.0
    last_alloc = np.zeros(5)
    max_dd = 0.0
    
    # Defense mode state (止损兜底)
    in_stop_loss = False
    stop_loss_weeks = 0
    
    # Weekly returns for Sharpe
    weekly_returns = []
    
    for i in range(start_idx, n_weeks - 1):
        # ---- Compute scores for offensive ETFs ----
        scores = np.zeros(5)
        for j in OFFENSIVE:
            if not np.isnan(w_mom4[i, j]) and not np.isnan(w_vol20[i, j]):
                scores[j] = (
                    mom_w * w_mom4[i, j] 
                    - vol_w * w_vol20[i, j] 
                    + val_w * (0.5 - pe_weekly[i] / 100.0)
                )
        
        # Select top_n offensive
        off_scores = [(scores[j], j) for j in OFFENSIVE]
        off_scores.sort(reverse=True)
        selected_off = [j for _, j in off_scores[:top_n]]
        
        # ---- Get available defensive ETFs ----
        avail_def = get_defensive_etfs(i)
        n_def = len(avail_def)
        
        # ---- Defense allocation ----
        # Layer 1: vol三段式防御
        nasdaq_vol = w_vol20[i, 0]
        if np.isnan(nasdaq_vol):
            def_ratio = def_alloc
        elif nasdaq_vol < step_low:
            def_ratio = def_alloc
        elif nasdaq_vol > step_high:
            def_ratio = max_def
        else:
            def_ratio = def_alloc + (max_def - def_alloc) * (nasdaq_vol - step_low) / (step_high - step_low)
        
        # Layer 2: 8%止损兜底 (当vol预警不足时)
        if not in_stop_loss and nav < peak * 0.92:
            in_stop_loss = True
            stop_loss_weeks = 0
        
        if in_stop_loss:
            def_ratio = max(def_ratio, 0.95)  # 至少95%防御
            stop_loss_weeks += 1
            if stop_loss_weeks >= 4:  # 观察4周后恢复
                in_stop_loss = False
        
        # Build allocation
        alloc = np.zeros(5)
        
        # Defensive layer with hongli_ratio
        if n_def == 2:
            # 红利低波 + 国债
            alloc[avail_def[0]] = def_ratio * hongli_ratio    # 红利低波
            alloc[avail_def[1]] = def_ratio * (1 - hongli_ratio)  # 国债
        else:
            # 仅国债
            alloc[avail_def[0]] = def_ratio
        
        # Offensive layer
        for j in selected_off:
            alloc[j] = (1 - def_ratio) / len(selected_off)
        
        # Turnover fee
        turnover = np.sum(np.abs(alloc - last_alloc))
        fee_cost = turnover * FEE_RATE
        
        # Compute return
        wret = sum(alloc[j] * w_rets[i, j] for j in range(5) if not np.isnan(w_rets[i, j]))
        nav *= (1 + wret - fee_cost)
        peak = max(peak, nav)
        
        dd = (peak - nav) / peak
        if dd > max_dd:
            max_dd = dd
        
        weekly_returns.append(wret - fee_cost)
        last_alloc = alloc
    
    # Metrics
    total_ret = nav - 1
    n_weeks_used = n_weeks - 1 - start_idx
    annual_ret = (1 + total_ret) ** (52 / n_weeks_used) - 1
    
    returns = np.array(weekly_returns)
    sharpe = returns.mean() / returns.std() * np.sqrt(52) if returns.std() > 0 else 0
    
    return {
        'mom_w': mom_w, 'vol_w': vol_w, 'val_w': val_w,
        'top_n': top_n, 'def_alloc': def_alloc,
        'step_low': step_low, 'step_high': step_high,
        'max_def': max_def, 'hongli_ratio': hongli_ratio,
        'annual_return': annual_ret,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'final_nav': nav
    }


# ============ 网格搜索 ============
print("\n[{}] Starting grid search...".format(datetime.now().strftime('%H:%M:%S')))

# Phase 1: Search weight combinations
print("\n=== Phase 1: Weight Search ===")
weight_results = []
weight_params = []

for mom_w in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    for vol_w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        for val_w in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
            if mom_w + vol_w + val_w > 1.0:
                continue
            for top_n in [1, 2]:
                # Fixed defense params for phase 1
                weight_params.append((
                    mom_w, vol_w, val_w, top_n,
                    0.30, 0.15, 0.30, 0.90, 0.20
                ))

print(f"Phase 1 combinations: {len(weight_params)}")

# Run phase 1
if len(weight_params) > 50:
    n_workers = min(max(1, int(cpu_count() * 0.6)), 8)
    with Pool(n_workers) as pool:
        weight_results = pool.map(run_backtest, weight_params)
else:
    weight_results = [run_backtest(p) for p in weight_params]

weight_df = pd.DataFrame(weight_results)
print(f"Phase 1 complete. Valid results: {len(weight_df)}")

# Filter: return > 5%, DD < 25%
valid_weights = weight_df[
    (weight_df['annual_return'] > 0.05) & 
    (weight_df['max_drawdown'] < 0.25)
].copy()

# Sort by Sharpe
top_weights = valid_weights.nlargest(20, 'sharpe')
print("\nTop 10 weight combos by Sharpe:")
print(top_weights[['mom_w', 'vol_w', 'val_w', 'top_n', 'annual_return', 'max_drawdown', 'sharpe']].head(10).to_string(index=False))

# Use top 3 weight combos for phase 2
top_3_weights = top_weights.head(3)

# Phase 2: Search defense parameters (REDUCED)
print("\n=== Phase 2: Defense Parameter Search ===")
defense_results = []
defense_params = []

for _, w_row in top_3_weights.iterrows():
    mom_w = w_row['mom_w']
    vol_w = w_row['vol_w']
    val_w = w_row['val_w']
    top_n = int(w_row['top_n'])
    
    for def_alloc in [0.25, 0.30, 0.35, 0.40, 0.45]:
        for step_low in [0.10, 0.15, 0.20, 0.25, 0.30]:
            for step_high in [0.20, 0.25, 0.30, 0.35, 0.40]:
                if step_high <= step_low:
                    continue
                for max_def in [0.75, 0.80, 0.85, 0.90, 0.95]:
                    for hongli in [0.0, 0.25, 0.50, 0.75, 1.0]:
                        defense_params.append((
                            mom_w, vol_w, val_w, top_n,
                            def_alloc, step_low, step_high, max_def, hongli
                        ))

print(f"Phase 2 combinations: {len(defense_params)}")

# Run phase 2
n_workers = min(max(1, int(cpu_count() * 0.6)), 8)
print(f"Using {n_workers} workers...")

with Pool(n_workers) as pool:
    defense_results = pool.map(run_backtest, defense_params)

defense_df = pd.DataFrame(defense_results)
print(f"Phase 2 complete. Valid results: {len(defense_df)}")

# Filter
valid_defense = defense_df[
    (defense_df['annual_return'] > 0.05) & 
    (defense_df['max_drawdown'] < 0.25)
].copy()

# Phase 3: Fine-tune around best defense params (REDUCED)
print("\n=== Phase 3: Fine-tuning ===")
best_from_p2 = valid_defense.nlargest(1, 'sharpe').iloc[0]
print("Best from Phase 2:")
for col in ['mom_w', 'vol_w', 'val_w', 'top_n', 'def_alloc', 'step_low', 'step_high', 'max_def', 'hongli_ratio']:
    print("  {}: {}".format(col, best_from_p2[col]))
print("  Annual: {:.2f}%, DD: {:.2f}%, Sharpe: {:.3f}".format(
    best_from_p2['annual_return']*100, best_from_p2['max_drawdown']*100, best_from_p2['sharpe']))

fine_results = []
fine_params = []

base_mom = best_from_p2['mom_w']
base_vol = best_from_p2['vol_w']
base_val = best_from_p2['val_w']
base_top = int(best_from_p2['top_n'])
base_def = best_from_p2['def_alloc']
base_sl = best_from_p2['step_low']
base_sh = best_from_p2['step_high']
base_max = best_from_p2['max_def']
base_hl = best_from_p2['hongli_ratio']

for mom_w in [base_mom - 0.05, base_mom, base_mom + 0.05]:
    for vol_w in [base_vol - 0.05, base_vol, base_vol + 0.05]:
        for val_w in [base_val - 0.05, base_val, base_val + 0.05]:
            if mom_w + vol_w + val_w > 1.0 or any(v < 0 for v in [mom_w, vol_w, val_w]):
                continue
            for def_alloc in [base_def - 0.05, base_def, base_def + 0.05]:
                if def_alloc < 0.15 or def_alloc > 0.60:
                    continue
                for step_low in [base_sl - 0.02, base_sl, base_sl + 0.02]:
                    for step_high in [base_sh - 0.03, base_sh, base_sh + 0.03]:
                        if step_high <= step_low or step_low < 0.05 or step_high > 0.60:
                            continue
                        for max_def in [base_max - 0.05, base_max, base_max + 0.05]:
                            if max_def < def_alloc or max_def > 1.0:
                                continue
                            for hongli in [base_hl - 0.10, base_hl, base_hl + 0.10]:
                                if hongli < 0 or hongli > 1:
                                    continue
                                fine_params.append((
                                    round(mom_w, 2), round(vol_w, 2), round(val_w, 2), base_top,
                                    round(def_alloc, 2), round(step_low, 2), round(step_high, 2),
                                    round(max_def, 2), round(hongli, 2)
                                ))

print(f"Phase 3 combinations: {len(fine_params)}")

with Pool(n_workers) as pool:
    fine_results = pool.map(run_backtest, fine_params)

fine_df = pd.DataFrame(fine_results)

# ============ 最终汇总 ============
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

all_results = pd.concat([weight_df, defense_df, fine_df], ignore_index=True)
all_valid = all_results[
    (all_results['annual_return'] > 0.05) & 
    (all_results['max_drawdown'] < 0.25)
].copy()

# Top by different metrics
print("\n--- Top 10 by Sharpe ---")
top_sharpe = all_valid.nlargest(10, 'sharpe')
print(top_sharpe[['mom_w', 'vol_w', 'val_w', 'top_n', 'def_alloc', 'step_low', 'step_high', 'max_def', 'hongli_ratio', 'annual_return', 'max_drawdown', 'sharpe']].to_string(index=False))

print("\n--- Top 10 by Return (DD < 15%) ---")
top_ret = all_valid[all_valid['max_drawdown'] < 0.15].nlargest(10, 'annual_return')
print(top_ret[['mom_w', 'vol_w', 'val_w', 'top_n', 'def_alloc', 'step_low', 'step_high', 'max_def', 'hongli_ratio', 'annual_return', 'max_drawdown', 'sharpe']].to_string(index=False))

print("\n--- Top 10 by Return/DD Ratio ---")
all_valid['ret_dd'] = all_valid['annual_return'] / all_valid['max_drawdown']
top_ratio = all_valid.nlargest(10, 'ret_dd')
print(top_ratio[['mom_w', 'vol_w', 'val_w', 'top_n', 'def_alloc', 'step_low', 'step_high', 'max_def', 'hongli_ratio', 'annual_return', 'max_drawdown', 'ret_dd']].to_string(index=False))

# Save results
all_valid.to_csv('output/unified_search_results.csv', index=False)
print("\n[{}] Results saved to output/unified_search_results.csv".format(datetime.now().strftime('%H:%M:%S')))
print(f"Total combinations tested: {len(all_results)}")
print(f"Valid combinations: {len(all_valid)}")

# Best overall
best = all_valid.nlargest(1, 'sharpe').iloc[0]
print("\n=== BEST PARAMETER SET (by Sharpe) ===")
print(f"  mom_w={best['mom_w']}, vol_w={best['vol_w']}, val_w={best['val_w']}")
print(f"  top_n={best['top_n']}, def_alloc={best['def_alloc']}")
print(f"  step_low={best['step_low']}, step_high={best['step_high']}, max_def={best['max_def']}")
print(f"  hongli_ratio={best['hongli_ratio']}")
print(f"  Annual Return: {best['annual_return']*100:.2f}%")
print(f"  Max Drawdown: {best['max_drawdown']*100:.2f}%")
print(f"  Sharpe: {best['sharpe']:.3f}")

# Save best
with open('output/best_params.json', 'w') as f:
    json.dump({
        'mom_w': best['mom_w'], 'vol_w': best['vol_w'], 'val_w': best['val_w'],
        'top_n': int(best['top_n']), 'def_alloc': best['def_alloc'],
        'step_low': best['step_low'], 'step_high': best['step_high'],
        'max_def': best['max_def'], 'hongli_ratio': best['hongli_ratio'],
        'annual_return': best['annual_return'], 'max_drawdown': best['max_drawdown'],
        'sharpe': best['sharpe']
    }, f, indent=2)

print("\n[{}] Done!".format(datetime.now().strftime('%H:%M:%S')))
