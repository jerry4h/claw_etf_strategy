import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from data_loader import load_nav_data, ETFS, DEFENSIVE
from factors import calculate_all_factors
from strategy import calculate_composite_score, select_top_offensive
from backtest import FEE_RATE


def run_backtest_monthly(mom_w=0.4, vol_w=0.4, val_w=0.0,
                          top_n=2, defensive_allocation=0.40,
                          target_volatility=0.12, enable_vol_adj=True,
                          progressive_defense=False):
    """月频回测（每月最后一个交易日调仓）"""
    nav_df = load_nav_data()
    ret_df = nav_df.pct_change().dropna()
    factors = calculate_all_factors(nav_df, ret_df, 20, 20, 60)

    # 月频：每月最后一个交易日
    monthly_dates = nav_df.resample("M").indices
    monthly_dates = sorted(monthly_dates.keys())

    portfolio_value = 1.0
    peak_value = 1.0
    current_allocation = {}
    portfolio_history = []

    in_defensive_mode = False
    defensive_mode_weeks = 0
    stop_loss_triggered_value = None

    for i, date in enumerate(monthly_dates[:-1]):
        scores = calculate_composite_score(
            factors["momentum"].loc[:date],
            factors["volatility"].loc[:date],
            factors["valuation"].loc[:date],
            ETFS, mom_w, vol_w, val_w
        )

        if scores.empty:
            continue

        # 止损检查
        if portfolio_value < peak_value * (1 - 0.08):
            if not in_defensive_mode:
                in_defensive_mode = True
                defensive_mode_weeks = 0
                stop_loss_triggered_value = peak_value * (1 - 0.08)
        else:
            if in_defensive_mode:
                defensive_mode_weeks += 1

        n_defensive = len(DEFENSIVE)

        if in_defensive_mode:
            # 月频：恢复观察期改为2个月（约8-9周）
            recovery_months = 2
            if defensive_mode_weeks < recovery_months:
                if progressive_defense:
                    step = (1.0 - defensive_allocation) / recovery_months
                    prog_alloc = min(1.0, defensive_allocation + (defensive_mode_weeks + 1) * step)
                    def_weight = prog_alloc / n_defensive
                    new_allocation = {etf: def_weight for etf in DEFENSIVE}
                    if prog_alloc < 1.0:
                        selected = select_top_offensive(scores, 1)
                        if selected:
                            off_weight = (1 - prog_alloc) / len(selected)
                            for etf in selected:
                                new_allocation[etf] = new_allocation.get(etf, 0) + off_weight
                else:
                    def_weight = 1.0 / n_defensive
                    new_allocation = {etf: def_weight for etf in DEFENSIVE}
            else:
                in_defensive_mode = False
                selected = select_top_offensive(scores, top_n)
                def_weight = defensive_allocation / n_defensive
                new_allocation = {etf: def_weight for etf in DEFENSIVE}
                if selected:
                    off_weight = (1 - defensive_allocation) / len(selected)
                    for etf in selected:
                        new_allocation[etf] = new_allocation.get(etf, 0) + off_weight
        else:
            selected = select_top_offensive(scores, top_n)
            def_weight = defensive_allocation / n_defensive
            new_allocation = {etf: def_weight for etf in DEFENSIVE}
            if selected:
                off_weight = (1 - defensive_allocation) / len(selected)
                for etf in selected:
                    new_allocation[etf] = new_allocation.get(etf, 0) + off_weight

        # 波动率调整
        if enable_vol_adj and new_allocation and date in factors["volatility"].index:
            vol_row = factors["volatility"].loc[date]
            portfolio_vol = 0
            for etf, w in new_allocation.items():
                if etf in vol_row.index and not pd.isna(vol_row[etf]):
                    portfolio_vol += w * vol_row[etf]
            portfolio_vol /= 100.0
            if portfolio_vol > 0 and target_volatility > 0:
                vol_multiplier = min(1.0, target_volatility / portfolio_vol)
                vol_multiplier = max(0.5, vol_multiplier)
                new_allocation = {etf: w * vol_multiplier for etf, w in new_allocation.items()}

        next_date = monthly_dates[i + 1]
        next_nav = nav_df.loc[next_date] if next_date in nav_df.index else nav_df.iloc[nav_df.index.get_indexer([next_date], method="ffill")[0]]

        monthly_return = 0
        for etf, weight in current_allocation.items():
            if etf in next_nav.index and not pd.isna(next_nav[etf]) and etf in nav_df.columns:
                prev_nav = nav_df.loc[date][etf] if date in nav_df.index and etf in nav_df.loc[date].index else nav_df[etf].iloc[nav_df.index.get_indexer([date], method="ffill")[0]]
                if not pd.isna(prev_nav) and prev_nav != 0:
                    ret = (next_nav[etf] - prev_nav) / prev_nav
                    monthly_return += weight * ret

        if i > 0:
            turnover = sum(
                abs(current_allocation.get(etf, 0) - new_allocation.get(etf, 0))
                for etf in set(current_allocation) | set(new_allocation)
            )
            monthly_return -= turnover * FEE_RATE

        portfolio_value *= (1 + monthly_return)
        peak_value = max(peak_value, portfolio_value)
        current_allocation = new_allocation

        allocation_record = {"date": next_date}
        for etf in ETFS:
            allocation_record[f"weight_{etf}"] = new_allocation.get(etf, 0.0)

        portfolio_history.append({
            "date": next_date,
            "portfolio_value": portfolio_value,
            "peak_value": peak_value,
            "monthly_return": monthly_return,
            "in_defensive": in_defensive_mode,
            **allocation_record
        })

    return pd.DataFrame(portfolio_history)


def calc_metrics(df, periods_per_year=12):
    if df.empty:
        return None
    final_val = df.iloc[-1]["portfolio_value"]
    total_return = final_val - 1
    annual_return = (1 + total_return) ** (periods_per_year / len(df)) - 1
    max_dd = ((df["peak_value"] - df["portfolio_value"]) / df["peak_value"]).max()

    turnovers = []
    trade_count = 0
    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]
        change = sum(abs(curr[f"weight_{e}"] - prev[f"weight_{e}"]) for e in ETFS)
        turnovers.append(change / 2)
        if change > 0.001:
            trade_count += 1

    avg_turnover = np.mean(turnovers)
    annual_turnover = avg_turnover * periods_per_year
    total_fee = sum(turnovers) * 2 * FEE_RATE

    return {
        "annual_return": annual_return,
        "max_drawdown": -max_dd,
        "total_return": total_return,
        "trade_count": trade_count,
        "annual_turnover": annual_turnover,
        "total_fee_impact": total_fee,
    }


# 对比：周频/双周/月频
from backtest import run_backtest

params_list = [
    dict(mom_w=0.15, vol_w=0.2, val_w=0.0, top_n=2, defensive_allocation=0.40, label="保守型"),
    dict(mom_w=0.30, vol_w=0.4, val_w=0.0, top_n=2, defensive_allocation=0.30, label="进取型"),
]

print("=" * 110)
print("三频率对比：周频 vs 双周 vs 月频")
print("=" * 110)

results = []

for p in params_list:
    base_label = p.pop("label")
    print(f"\n{'='*60}")
    print(f"参数: {base_label} (mom={p['mom_w']}, vol={p['vol_w']}, def={p['defensive_allocation']})")
    print(f"{'='*60}")

    # 周频
    r_w = run_backtest(freq="W-MON", **p)
    m_w = calc_metrics(r_w, 52)
    m_w["方案"] = "周频"
    m_w["参数"] = base_label
    results.append(m_w)
    calmar = m_w['annual_return'] / (-m_w['max_drawdown'])
    print(f"  周频:   年化={m_w['annual_return']*100:.2f}% 回撤={m_w['max_drawdown']*100:.2f}% Calmar={calmar:.3f} "
          f"调仓={m_w['trade_count']}次 换手={m_w['annual_turnover']*100:.0f}% 手续费={m_w['total_fee_impact']*100:.3f}%")

    # 双周
    r_2w = run_backtest(freq="2W-MON", **p)
    m_2w = calc_metrics(r_2w, 26)
    m_2w["方案"] = "双周"
    m_2w["参数"] = base_label
    results.append(m_2w)
    calmar = m_2w['annual_return'] / (-m_2w['max_drawdown'])
    print(f"  双周:   年化={m_2w['annual_return']*100:.2f}% 回撤={m_2w['max_drawdown']*100:.2f}% Calmar={calmar:.3f} "
          f"调仓={m_2w['trade_count']}次 换手={m_2w['annual_turnover']*100:.0f}% 手续费={m_2w['total_fee_impact']*100:.3f}%")

    # 月频-原始
    r_m = run_backtest_monthly(progressive_defense=False, **p)
    m_m = calc_metrics(r_m, 12)
    m_m["方案"] = "月频"
    m_m["参数"] = base_label
    results.append(m_m)
    calmar = m_m['annual_return'] / (-m_m['max_drawdown'])
    print(f"  月频:   年化={m_m['annual_return']*100:.2f}% 回撤={m_m['max_drawdown']*100:.2f}% Calmar={calmar:.3f} "
          f"调仓={m_m['trade_count']}次 换手={m_m['annual_turnover']*100:.0f}% 手续费={m_m['total_fee_impact']*100:.3f}%")

    # 月频-渐进防御
    r_mp = run_backtest_monthly(progressive_defense=True, **p)
    m_mp = calc_metrics(r_mp, 12)
    m_mp["方案"] = "月频+渐进"
    m_mp["参数"] = base_label
    results.append(m_mp)
    calmar = m_mp['annual_return'] / (-m_mp['max_drawdown'])
    print(f"  月频+渐进: 年化={m_mp['annual_return']*100:.2f}% 回撤={m_mp['max_drawdown']*100:.2f}% Calmar={calmar:.3f} "
          f"调仓={m_mp['trade_count']}次 换手={m_mp['annual_turnover']*100:.0f}% 手续费={m_mp['total_fee_impact']*100:.3f}%")

# 汇总
print("\n" + "=" * 110)
print("汇总对比")
print("=" * 110)
df = pd.DataFrame(results)
df['Calmar'] = df['annual_return'] / (-df['max_drawdown'])
df['annual_pct'] = (df['annual_return']*100).round(2).astype(str)+'%'
df['dd_pct'] = (df['max_drawdown']*100).round(2).astype(str)+'%'
df['turnover_pct'] = (df['annual_turnover']*100).round(0).astype(str)+'%'
df['fee_pct'] = (df['total_fee_impact']*100).round(3).astype(str)+'%'
print(df[['方案', '参数', 'annual_pct', 'dd_pct', 'Calmar', 'trade_count', 'turnover_pct', 'fee_pct']].to_string(index=False))
