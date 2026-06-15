import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from data_loader import load_nav_data, ETFS, DEFENSIVE, OFFENSIVE
from factors import calculate_all_factors
from strategy import calculate_composite_score
import backtest

FEE_RATE = backtest.FEE_RATE
DEFENSIVE_POOL = backtest.DEFENSIVE_POOL


def _allocate_discrete(scores, defensive_allocation, top_n):
    """原始3选2离散选择"""
    from strategy import select_top_offensive, get_allocation
    selected = select_top_offensive(scores, top_n)
    return get_allocation(selected, defensive_allocation)


def _allocate_proportional(scores, defensive_allocation, tau):
    """按评分比例分配进攻层权重（softmax）"""
    off_scores = scores[OFFENSIVE].iloc[-1].dropna() if not scores.empty else pd.Series()

    def_weight = defensive_allocation / len(DEFENSIVE_POOL)
    allocation = {etf: def_weight for etf in DEFENSIVE_POOL}

    if off_scores.empty or off_scores.max() == 0:
        off_weight = (1 - defensive_allocation) / len(OFFENSIVE)
        for etf in OFFENSIVE:
            allocation[etf] = off_weight
        return allocation

    # z-score标准化 + softmax
    mean_score = off_scores.mean()
    std_score = off_scores.std() if off_scores.std() > 0 else 1
    normalized = (off_scores - mean_score) / std_score
    exp_scores = np.exp(normalized / tau)
    weights = exp_scores / exp_scores.sum()

    for etf in OFFENSIVE:
        allocation[etf] = weights.get(etf, 0) * (1 - defensive_allocation)

    return allocation


def run_backtest_v2(start_date=None, end_date=None,
                    mom_w=0.4, vol_w=0.4, val_w=0.2,
                    mom_window=20, vol_window=20, val_window=60,
                    top_n=2, stop_loss_threshold=0.08,
                    recovery_weeks=4,
                    defensive_allocation=0.55,
                    target_volatility=0.12,
                    enable_vol_adj=True,
                    stop_loss_offensive_n=2,
                    freq="W-MON",
                    use_proportional=False,
                    tau=1.0):
    """
    与原始backtest.py完全一致的逻辑，仅分配方式可切换
    """
    nav_df = load_nav_data()

    if start_date:
        nav_df = nav_df[nav_df.index >= pd.to_datetime(start_date)]
    if end_date:
        nav_df = nav_df[nav_df.index <= pd.to_datetime(end_date)]

    ret_df = nav_df.pct_change().dropna()
    factors = calculate_all_factors(nav_df, ret_df, mom_window, vol_window, val_window)

    # 与原始完全一致：用resample生成调仓日
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

        available_defensive = DEFENSIVE_POOL
        n_defensive = len(available_defensive)

        if in_defensive_mode:
            if stop_loss_offensive_n == 1:
                def_alloc = 0.70
                def_weight = def_alloc / n_defensive
                new_allocation = {etf: def_weight for etf in available_defensive}
                # goal模式逻辑...省略，当前模式不用
            else:
                if defensive_mode_weeks < recovery_weeks:
                    def_weight = 1.0 / n_defensive
                    new_allocation = {etf: def_weight for etf in available_defensive}
                else:
                    in_defensive_mode = False
                    if use_proportional:
                        new_allocation = _allocate_proportional(scores, defensive_allocation, tau)
                    else:
                        new_allocation = _allocate_discrete(scores, defensive_allocation, top_n)
        else:
            if use_proportional:
                new_allocation = _allocate_proportional(scores, defensive_allocation, tau)
            else:
                new_allocation = _allocate_discrete(scores, defensive_allocation, top_n)

        # 波动率调整（与原始完全一致）
        if enable_vol_adj and new_allocation and date in factors["volatility"].index:
            vol_row = factors["volatility"].loc[date]
            portfolio_vol = 0
            vol_count = 0
            for etf, w in new_allocation.items():
                if etf in vol_row.index and not pd.isna(vol_row[etf]):
                    portfolio_vol += w * vol_row[etf]
                    vol_count += 1
            portfolio_vol /= 100.0
            if portfolio_vol > 0 and target_volatility > 0:
                vol_multiplier = min(1.0, target_volatility / portfolio_vol)
                vol_multiplier = max(0.5, vol_multiplier)
                new_allocation = {etf: w * vol_multiplier for etf, w in new_allocation.items()}

        next_date = weekly_dates[i + 1]
        next_nav = nav_df.loc[next_date] if next_date in nav_df.index else nav_df.iloc[nav_df.index.get_indexer([next_date], method="ffill")[0]]

        # 用current_allocation（上周持仓）计算收益（与原始一致）
        weekly_return = 0
        for etf, weight in current_allocation.items():
            if etf in next_nav.index and not pd.isna(next_nav[etf]) and etf in nav_df.columns:
                prev_nav = nav_df.loc[date][etf] if date in nav_df.index and etf in nav_df.loc[date].index else nav_df[etf].iloc[nav_df.index.get_indexer([date], method="ffill")[0]]
                if not pd.isna(prev_nav) and prev_nav != 0:
                    ret = (next_nav[etf] - prev_nav) / prev_nav
                    weekly_return += weight * ret

        # 手续费（i>0时计算，与原始一致）
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


def calc_metrics(result_df):
    if result_df.empty:
        return None
    final_val = result_df.iloc[-1]["portfolio_value"]
    total_return = final_val - 1
    annual_return = (1 + total_return) ** (52 / len(result_df)) - 1
    max_dd = ((result_df["peak_value"] - result_df["portfolio_value"]) / result_df["peak_value"]).max()
    defensive_days = result_df["in_defensive"].sum()

    turnovers = []
    for i in range(1, len(result_df)):
        prev = result_df.iloc[i-1]
        curr = result_df.iloc[i]
        change = sum(abs(curr[f"weight_{e}"] - prev[f"weight_{e}"]) for e in ETFS)
        turnovers.append(change / 2)
    avg_turnover = np.mean(turnovers)
    annual_turnover = avg_turnover * 52
    total_fee = sum(turnovers) * 2 * FEE_RATE

    return {
        "annual_return": annual_return,
        "max_drawdown": -max_dd,
        "total_return": total_return,
        "avg_weekly_turnover": avg_turnover,
        "annual_turnover": annual_turnover,
        "total_fee_impact": total_fee,
        "defensive_weeks": int(defensive_days),
    }


# ============ 测试 ============
params_list = [
    dict(mom_w=0.15, vol_w=0.2, defensive_allocation=0.40),
    dict(mom_w=0.30, vol_w=0.4, defensive_allocation=0.30),
]

tau_values = [0.5, 1.0, 2.0, 5.0]

print("=" * 85)
print("进攻层分配方式对比测试（v2 - 与原始backtest完全一致）")
print("=" * 85)

results = []

for p in params_list:
    label = f"mom={p['mom_w']},vol={p['vol_w']},def={p['defensive_allocation']}"

    # 验证：原始3选2
    r = run_backtest_v2(use_proportional=False, **p)
    m = calc_metrics(r)
    m["方案"] = "3选2(离散)"
    m["参数"] = label
    m["tau"] = "-"
    results.append(m)
    calmar = m['annual_return'] / (-m['max_drawdown'])
    print(f"\n{label} | 3选2(离散):")
    print(f"  年化: {m['annual_return']*100:.2f}%  回撤: {m['max_drawdown']*100:.2f}%  Calmar: {calmar:.3f}")
    print(f"  年化换手: {m['annual_turnover']*100:.0f}%  手续费: {m['total_fee_impact']*100:.3f}%")

    # softmax 各种tau
    for tau in tau_values:
        r = run_backtest_v2(use_proportional=True, tau=tau, **p)
        m = calc_metrics(r)
        m["方案"] = f"比例(tau={tau})"
        m["参数"] = label
        m["tau"] = tau
        results.append(m)
        calmar = m['annual_return'] / (-m['max_drawdown'])
        print(f"\n{label} | 比例分配(tau={tau}):")
        print(f"  年化: {m['annual_return']*100:.2f}%  回撤: {m['max_drawdown']*100:.2f}%  Calmar: {calmar:.3f}")
        print(f"  年化换手: {m['annual_turnover']*100:.0f}%  手续费: {m['total_fee_impact']*100:.3f}%")

# 汇总
print("\n" + "=" * 85)
print("汇总对比")
print("=" * 85)
df = pd.DataFrame(results)
df['Calmar'] = df['annual_return'] / (-df['max_drawdown'])
df['annual_pct'] = (df['annual_return']*100).round(2).astype(str)+'%'
df['dd_pct'] = (df['max_drawdown']*100).round(2).astype(str)+'%'
df['turnover_pct'] = (df['annual_turnover']*100).round(0).astype(str)+'%'
df['fee_pct'] = (df['total_fee_impact']*100).round(3).astype(str)+'%'
print(df[['方案', '参数', 'annual_pct', 'dd_pct', 'Calmar', 'turnover_pct', 'fee_pct']].to_string(index=False))
