import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from data_loader import load_nav_data, ETFS, DEFENSIVE, OFFENSIVE
from factors import calculate_all_factors
from strategy import calculate_composite_score, select_top_offensive
from backtest import FEE_RATE


def run_backtest_improved(start_date=None, end_date=None,
                          mom_w=0.4, vol_w=0.4, val_w=0.0,
                          top_n=2, stop_loss_threshold=0.08,
                          recovery_weeks=4,
                          defensive_allocation=0.40,
                          target_volatility=0.12,
                          enable_vol_adj=True,
                          freq="W-MON",
                          progressive_defense=False):
    """
    改进版回测引擎:
    - progressive_defense: 渐进式防御，防御比例随防御周数逐步增加
    """
    nav_df = load_nav_data()

    if start_date:
        nav_df = nav_df[nav_df.index >= pd.to_datetime(start_date)]
    if end_date:
        nav_df = nav_df[nav_df.index <= pd.to_datetime(end_date)]

    ret_df = nav_df.pct_change().dropna()
    factors = calculate_all_factors(nav_df, ret_df, 20, 20, 60)

    weekly_dates = nav_df.resample(freq).indices
    weekly_dates = sorted(weekly_dates.keys())

    portfolio_value = 1.0
    peak_value = 1.0
    current_allocation = {}
    portfolio_history = []

    in_defensive_mode = False
    defensive_mode_weeks = 0
    stop_loss_triggered_value = None

    for i, date in enumerate(weekly_dates[:-1]):
        scores = calculate_composite_score(
            factors["momentum"].loc[:date],
            factors["volatility"].loc[:date],
            factors["valuation"].loc[:date],
            ETFS, mom_w, vol_w, val_w
        )

        if scores.empty:
            continue

        # 止损检查
        if portfolio_value < peak_value * (1 - stop_loss_threshold):
            if not in_defensive_mode:
                in_defensive_mode = True
                defensive_mode_weeks = 0
                stop_loss_triggered_value = peak_value * (1 - stop_loss_threshold)
        else:
            if in_defensive_mode:
                defensive_mode_weeks += 1

        # 防御层
        available_defensive = DEFENSIVE
        n_defensive = len(available_defensive)

        if in_defensive_mode:
            if defensive_mode_weeks < recovery_weeks:
                # 渐进式防御逻辑
                if progressive_defense:
                    # 从 base_def_alloc 逐步增加到 1.0
                    step = (1.0 - defensive_allocation) / recovery_weeks
                    prog_alloc = min(1.0, defensive_allocation + (defensive_mode_weeks + 1) * step)
                    def_weight = prog_alloc / n_defensive
                    new_allocation = {etf: def_weight for etf in available_defensive}

                    # 剩余仓位给进攻层（选1个最强的）
                    if prog_alloc < 1.0:
                        selected = select_top_offensive(scores, 1)
                        if selected:
                            off_weight = (1 - prog_alloc) / len(selected)
                            for etf in selected:
                                new_allocation[etf] = new_allocation.get(etf, 0) + off_weight
                else:
                    # 原始逻辑：100%防御
                    def_weight = 1.0 / n_defensive
                    new_allocation = {etf: def_weight for etf in available_defensive}
            else:
                # 观察期结束，恢复进攻
                in_defensive_mode = False
                selected = select_top_offensive(scores, top_n)
                def_weight = defensive_allocation / n_defensive
                new_allocation = {etf: def_weight for etf in available_defensive}
                if selected:
                    off_weight = (1 - defensive_allocation) / len(selected)
                    for etf in selected:
                        new_allocation[etf] = new_allocation.get(etf, 0) + off_weight
        else:
            # 正常模式
            selected = select_top_offensive(scores, top_n)
            def_weight = defensive_allocation / n_defensive
            new_allocation = {etf: def_weight for etf in available_defensive}
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

        next_date = weekly_dates[i + 1]
        next_nav = nav_df.loc[next_date] if next_date in nav_df.index else nav_df.iloc[nav_df.index.get_indexer([next_date], method="ffill")[0]]

        weekly_return = 0
        for etf, weight in current_allocation.items():
            if etf in next_nav.index and not pd.isna(next_nav[etf]) and etf in nav_df.columns:
                prev_nav = nav_df.loc[date][etf] if date in nav_df.index and etf in nav_df.loc[date].index else nav_df[etf].iloc[nav_df.index.get_indexer([date], method="ffill")[0]]
                if not pd.isna(prev_nav) and prev_nav != 0:
                    ret = (next_nav[etf] - prev_nav) / prev_nav
                    weekly_return += weight * ret

        if i > 0:
            turnover = sum(
                abs(current_allocation.get(etf, 0) - new_allocation.get(etf, 0))
                for etf in set(current_allocation) | set(new_allocation)
            )
            weekly_return -= turnover * FEE_RATE

        portfolio_value *= (1 + weekly_return)
        peak_value = max(peak_value, portfolio_value)
        current_allocation = new_allocation

        allocation_record = {"date": next_date}
        for etf in ETFS:
            allocation_record[f"weight_{etf}"] = new_allocation.get(etf, 0.0)

        portfolio_history.append({
            "date": next_date,
            "portfolio_value": portfolio_value,
            "peak_value": peak_value,
            "weekly_return": weekly_return,
            "in_defensive": in_defensive_mode,
            **allocation_record
        })

    return pd.DataFrame(portfolio_history)


def calc_metrics(result_df, periods_per_year=52):
    if result_df.empty:
        return None
    final_val = result_df.iloc[-1]["portfolio_value"]
    total_return = final_val - 1
    annual_return = (1 + total_return) ** (periods_per_year / len(result_df)) - 1
    max_dd = ((result_df["peak_value"] - result_df["portfolio_value"]) / result_df["peak_value"]).max()

    turnovers = []
    trade_count = 0
    for i in range(1, len(result_df)):
        prev = result_df.iloc[i-1]
        curr = result_df.iloc[i]
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


# ============ 测试 ============
from backtest import run_backtest

params_list = [
    dict(mom_w=0.15, vol_w=0.2, val_w=0.0, top_n=2, defensive_allocation=0.40, label="保守型"),
    dict(mom_w=0.30, vol_w=0.4, val_w=0.0, top_n=2, defensive_allocation=0.30, label="进取型"),
]

frequencies = [
    ("W-MON", "周频", 52),
    ("2W-MON", "双周", 26),
]

print("=" * 100)
print("改进方案测试：双周调仓 + 渐进式防御")
print("=" * 100)

results = []

for p in params_list:
    base_label = p.pop("label")
    print(f"\n{'='*60}")
    print(f"参数: {base_label} (mom={p['mom_w']}, vol={p['vol_w']}, def={p['defensive_allocation']})")
    print(f"{'='*60}")

    for freq, freq_label, periods in frequencies:
        # 基准（原始逻辑）
        r_base = run_backtest(freq=freq, **p)
        m_base = calc_metrics(r_base, periods)
        m_base["方案"] = f"{freq_label}-原始"
        m_base["参数"] = base_label
        m_base["freq"] = freq_label
        results.append(m_base)
        calmar = m_base['annual_return'] / (-m_base['max_drawdown'])
        print(f"\n  {freq_label}-原始: 年化={m_base['annual_return']*100:.2f}% 回撤={m_base['max_drawdown']*100:.2f}% "
              f"Calmar={calmar:.3f} 调仓={m_base['trade_count']}次 换手={m_base['annual_turnover']*100:.0f}%")

        # 渐进式防御
        r_prog = run_backtest_improved(freq=freq, progressive_defense=True, **p)
        m_prog = calc_metrics(r_prog, periods)
        m_prog["方案"] = f"{freq_label}-渐进防御"
        m_prog["参数"] = base_label
        m_prog["freq"] = freq_label
        results.append(m_prog)
        calmar = m_prog['annual_return'] / (-m_prog['max_drawdown'])
        print(f"  {freq_label}-渐进防御: 年化={m_prog['annual_return']*100:.2f}% 回撤={m_prog['max_drawdown']*100:.2f}% "
              f"Calmar={calmar:.3f} 调仓={m_prog['trade_count']}次 换手={m_prog['annual_turnover']*100:.0f}%")

        # 差异
        ret_diff = (m_prog['annual_return'] - m_base['annual_return']) * 100
        dd_diff = (m_prog['max_drawdown'] - m_base['max_drawdown']) * 100
        print(f"  → 差异: 年化{ret_diff:+.2f}% 回撤{dd_diff:+.2f}%")

# 汇总表
print("\n" + "=" * 100)
print("汇总对比")
print("=" * 100)
df = pd.DataFrame(results)
df['Calmar'] = df['annual_return'] / (-df['max_drawdown'])
df['annual_pct'] = (df['annual_return']*100).round(2).astype(str)+'%'
df['dd_pct'] = (df['max_drawdown']*100).round(2).astype(str)+'%'
df['turnover_pct'] = (df['annual_turnover']*100).round(0).astype(str)+'%'
print(df[['方案', '参数', 'annual_pct', 'dd_pct', 'Calmar', 'trade_count', 'turnover_pct']].to_string(index=False))
