import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from data_loader import load_nav_data, ETFS
from factors import calculate_all_factors
from strategy import calculate_composite_score, check_stop_loss
from backtest import DEFENSIVE_POOL

FEE_RATE = 0.00005
OFFENSIVE = ["纳指ETF", "沪深300ETF", "黄金ETF"]

def run_backtest_proportional(mom_w=0.4, vol_w=0.4, val_w=0.0,
                               top_n=2, defensive_allocation=0.55,
                               target_volatility=0.12, enable_vol_adj=True,
                               freq="W-MON", use_softmax=True, tau=1.0):
    """
    回测引擎 - 进攻层按评分比例分配（非离散选择）
    use_softmax=True: 用softmax按评分比例分配进攻层权重
    use_softmax=False: 用原始3选2逻辑
    """
    nav_df = load_nav_data()
    ret_df = nav_df.pct_change().dropna()
    factors = calculate_all_factors(nav_df, ret_df, mom_window=20, vol_window=20, val_window=60)

    scores = calculate_composite_score(
        factors["momentum"], factors["volatility"], factors["valuation"],
        ETFS, mom_w=mom_w, vol_w=vol_w, val_w=val_w
    )

    weekly_dates = pd.date_range(start=nav_df.index.min(), end=nav_df.index.max(), freq=freq)
    weekly_dates = weekly_dates[weekly_dates.isin(nav_df.index)]

    portfolio_value = 1.0
    peak_value = 1.0
    portfolio_history = []
    in_defensive_mode = False
    defensive_mode_weeks = 0
    stop_loss_triggered_value = None

    for i, date in enumerate(weekly_dates[:-1]):
        # 当前净值
        if date in nav_df.index:
            current_nav = nav_df.loc[date]
        else:
            current_nav = nav_df.iloc[nav_df.index.get_indexer([date], method="ffill")[0]]

        portfolio_value = portfolio_value
        if portfolio_value > peak_value:
            peak_value = portfolio_value

        # 止损检查
        if portfolio_value < peak_value * (1 - 0.08):
            if not in_defensive_mode:
                in_defensive_mode = True
                defensive_mode_weeks = 0
                stop_loss_triggered_value = peak_value * (1 - 0.08)
        else:
            if in_defensive_mode:
                defensive_mode_weeks += 1

        # 计算当前分数
        valid_scores = scores.loc[scores.index <= date] if not scores.empty else pd.DataFrame()
        if valid_scores.empty or len(valid_scores) == 0:
            score_row = pd.Series(0, index=OFFENSIVE)
        else:
            score_row = valid_scores.iloc[-1]

        # 防御层
        available_defensive = DEFENSIVE_POOL
        n_defensive = len(available_defensive)

        if in_defensive_mode:
            if defensive_mode_weeks < 4:
                def_weight = 1.0 / n_defensive
                new_allocation = {etf: def_weight for etf in available_defensive}
            else:
                in_defensive_mode = False
                if use_softmax:
                    new_allocation = _allocate_proportional(score_row, defensive_allocation, tau)
                else:
                    new_allocation = _allocate_discrete(score_row, defensive_allocation, top_n)
        else:
            if use_softmax:
                new_allocation = _allocate_proportional(score_row, defensive_allocation, tau)
            else:
                new_allocation = _allocate_discrete(score_row, defensive_allocation, top_n)

        # 波动率调整
        if enable_vol_adj and new_allocation:
            portfolio_vol = 0
            vol_count = 0
            for etf, w in new_allocation.items():
                if etf in ret_df.columns:
                    vol = ret_df[etf].iloc[max(0, nav_df.index.get_loc(date)-20):nav_df.index.get_loc(date)+1].std() * np.sqrt(252)
                    portfolio_vol += w * vol
                    vol_count += 1
            portfolio_vol /= 100.0
            if portfolio_vol > 0 and target_volatility > 0:
                vol_multiplier = min(1.0, target_volatility / portfolio_vol)
                vol_multiplier = max(0.5, vol_multiplier)
                new_allocation = {etf: w * vol_multiplier for etf, w in new_allocation.items()}

        # 计算下周收益
        next_date = weekly_dates[i + 1]
        if next_date in nav_df.index:
            next_nav = nav_df.loc[next_date]
        else:
            next_nav = nav_df.iloc[nav_df.index.get_indexer([next_date], method="ffill")[0]]

        weekly_return = 0
        for etf, weight in new_allocation.items():
            if etf in current_nav.index and etf in next_nav.index and current_nav[etf] > 0:
                etf_return = (next_nav[etf] - current_nav[etf]) / current_nav[etf]
                weekly_return += weight * etf_return

        # 记录当前分配
        allocation_record = {}
        for etf in ETFS:
            allocation_record[f"weight_{etf}"] = new_allocation.get(etf, 0.0)

        portfolio_history.append({
            "date": next_date,
            "portfolio_value": portfolio_value * (1 + weekly_return),
            "peak_value": peak_value,
            "weekly_return": weekly_return,
            "in_defensive": in_defensive_mode,
            **allocation_record
        })

        portfolio_value = portfolio_value * (1 + weekly_return)

    return pd.DataFrame(portfolio_history)


def _allocate_discrete(score_row, defensive_allocation, top_n):
    """原始3选2离散选择"""
    off_scores = score_row[OFFENSIVE].dropna()
    top = off_scores.nlargest(top_n).index.tolist() if not off_scores.empty else []

    def_weight = defensive_allocation / len(DEFENSIVE_POOL)
    allocation = {etf: def_weight for etf in DEFENSIVE_POOL}

    if top:
        off_weight = (1 - defensive_allocation) / len(top)
        for etf in top:
            allocation[etf] = off_weight
    else:
        extra = (1 - defensive_allocation) / len(DEFENSIVE_POOL)
        for etf in DEFENSIVE_POOL:
            allocation[etf] += extra

    return allocation


def _allocate_proportional(score_row, defensive_allocation, tau):
    """按评分比例分配进攻层权重（softmax）"""
    off_scores = score_row[OFFENSIVE].dropna()

    def_weight = defensive_allocation / len(DEFENSIVE_POOL)
    allocation = {etf: def_weight for etf in DEFENSIVE_POOL}

    if off_scores.empty or off_scores.max() == 0:
        # 无有效评分，平均分配给进攻层
        off_weight = (1 - defensive_allocation) / len(OFFENSIVE)
        for etf in OFFENSIVE:
            allocation[etf] = off_weight
        return allocation

    # softmax 变换
    # 先做 z-score 标准化，避免数值过大/过小导致softmax极端
    mean_score = off_scores.mean()
    std_score = off_scores.std() if off_scores.std() > 0 else 1
    normalized = (off_scores - mean_score) / std_score

    exp_scores = np.exp(normalized / tau)
    weights = exp_scores / exp_scores.sum()

    for etf in OFFENSIVE:
        allocation[etf] = weights.get(etf, 0) * (1 - defensive_allocation)

    return allocation


def calc_metrics(result_df):
    if result_df.empty:
        return None
    final_val = result_df.iloc[-1]["portfolio_value"]
    total_return = final_val - 1
    annual_return = (1 + total_return) ** (52 / len(result_df)) - 1
    max_dd = ((result_df["peak_value"] - result_df["portfolio_value"]) / result_df["peak_value"]).max()
    defensive_days = result_df["in_defensive"].sum()

    # 计算换手率
    turnovers = []
    for i in range(1, len(result_df)):
        prev = result_df.iloc[i-1]
        curr = result_df.iloc[i]
        change = sum(abs(curr[f"weight_{e}"] - prev[f"weight_{e}"]) for e in ETFS)
        turnovers.append(change / 2)
    avg_turnover = np.mean(turnovers)
    annual_turnover = avg_turnover * 52

    # 手续费估算（假设每次调仓产生换手×万0.5的费用）
    total_fee = sum(turnovers) * 2 * FEE_RATE  # ×2因为买入+卖出

    return {
        "annual_return": annual_return,
        "max_drawdown": -max_dd,
        "total_return": total_return,
        "avg_weekly_turnover": avg_turnover,
        "annual_turnover": annual_turnover,
        "total_fee_impact": total_fee,
        "defensive_weeks": int(defensive_days),
    }


# 测试参数
params_list = [
    dict(mom_w=0.15, vol_w=0.2, defensive_allocation=0.40),
    dict(mom_w=0.30, vol_w=0.4, defensive_allocation=0.30),
]

tau_values = [0.5, 1.0, 2.0]

print("=" * 80)
print("进攻层分配方式对比测试")
print("=" * 80)

results = []

for p in params_list:
    label = f"mom={p['mom_w']},vol={p['vol_w']},def={p['defensive_allocation']}"

    # 离散3选2
    r = run_backtest_proportional(use_softmax=False, **p)
    m = calc_metrics(r)
    m["方案"] = "3选2(离散)"
    m["参数"] = label
    m["tau"] = "-"
    results.append(m)
    print(f"\n{label} | 3选2(离散):")
    print(f"  年化: {m['annual_return']*100:.2f}%  回撤: {m['max_drawdown']*100:.2f}%  Calmar: {m['annual_return']/(-m['max_drawdown']):.3f}")
    print(f"  年化换手: {m['annual_turnover']*100:.0f}%  手续费损耗: {m['total_fee_impact']*100:.3f}%")

    # softmax 各种tau
    for tau in tau_values:
        r = run_backtest_proportional(use_softmax=True, tau=tau, **p)
        m = calc_metrics(r)
        m["方案"] = f"比例(tau={tau})"
        m["参数"] = label
        m["tau"] = tau
        results.append(m)
        print(f"\n{label} | 比例分配(tau={tau}):")
        print(f"  年化: {m['annual_return']*100:.2f}%  回撤: {m['max_drawdown']*100:.2f}%  Calmar: {m['annual_return']/(-m['max_drawdown']):.3f}")
        print(f"  年化换手: {m['annual_turnover']*100:.0f}%  手续费损耗: {m['total_fee_impact']*100:.3f}%")

# 汇总表
print("\n" + "=" * 80)
print("汇总对比")
print("=" * 80)
df = pd.DataFrame(results)
df['Calmar'] = df['annual_return'] / (-df['max_drawdown'])
df['annual_pct'] = (df['annual_return']*100).round(2).astype(str)+'%'
df['dd_pct'] = (df['max_drawdown']*100).round(2).astype(str)+'%'
df['turnover_pct'] = (df['annual_turnover']*100).round(0).astype(str)+'%'
df['fee_pct'] = (df['total_fee_impact']*100).round(3).astype(str)+'%'
print(df[['方案', '参数', 'annual_pct', 'dd_pct', 'Calmar', 'turnover_pct', 'fee_pct']].to_string(index=False))
