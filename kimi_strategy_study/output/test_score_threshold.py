import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from data_loader import load_nav_data, ETFS
from factors import calculate_all_factors
from strategy import calculate_composite_score
from backtest import DEFENSIVE_POOL, FEE_RATE

def run_backtest_score_threshold(mom_w=0.4, vol_w=0.4, val_w=0.0,
                                  top_n=2, defensive_allocation=0.40,
                                  score_threshold=0.0,
                                  target_volatility=0.12, enable_vol_adj=True,
                                  freq="W-MON"):
    """
    回测引擎 - 评分优势阈值版
    score_threshold: 只有当新标的评分 >= 当前持仓最低评分 * (1+threshold) 时才切换
    """
    nav_df = load_nav_data()
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

    # 记录当前进攻层持仓（用于评分优势判断）
    current_offensive = []

    for i, date in enumerate(weekly_dates[:-1]):
        scores = calculate_composite_score(
            factors["momentum"].loc[:date],
            factors["volatility"].loc[:date],
            factors["valuation"].loc[:date],
            ETFS, mom_w, vol_w, val_w
        )

        if scores.empty:
            continue

        score_row = scores.iloc[-1]

        # 止损检查
        if portfolio_value < peak_value * (1 - 0.08):
            if not in_defensive_mode:
                in_defensive_mode = True
                defensive_mode_weeks = 0
                stop_loss_triggered_value = peak_value * (1 - 0.08)
        else:
            if in_defensive_mode:
                defensive_mode_weeks += 1

        available_defensive = DEFENSIVE_POOL
        n_defensive = len(available_defensive)

        if in_defensive_mode:
            if defensive_mode_weeks < 4:
                def_weight = 1.0 / n_defensive
                new_allocation = {etf: def_weight for etf in available_defensive}
                current_offensive = []
            else:
                in_defensive_mode = False
                new_allocation, current_offensive = _allocate_with_score_threshold(
                    score_row, defensive_allocation, top_n, current_offensive, score_threshold)
        else:
            new_allocation, current_offensive = _allocate_with_score_threshold(
                score_row, defensive_allocation, top_n, current_offensive, score_threshold)

        # 波动率调整（与原始一致）
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


def _allocate_with_score_threshold(score_row, defensive_allocation, top_n, current_offensive, score_threshold):
    """带评分优势阈值的分配逻辑"""
    OFFENSIVE = ["纳指ETF", "沪深300ETF", "黄金ETF"]
    off_scores = score_row[OFFENSIVE].dropna().sort_values(ascending=False)

    def_weight = defensive_allocation / len(DEFENSIVE_POOL)
    allocation = {etf: def_weight for etf in DEFENSIVE_POOL}

    if off_scores.empty or off_scores.max() == 0:
        off_weight = (1 - defensive_allocation) / len(OFFENSIVE)
        for etf in OFFENSIVE:
            allocation[etf] = off_weight
        return allocation, []

    # 计算新的top_n
    new_top = off_scores.nlargest(top_n).index.tolist()

    if score_threshold > 0 and current_offensive:
        # 评分优势阈值逻辑（基于绝对差值，正确处理负数评分）
        current_min_score = min(score_row.get(etf, -999) for etf in current_offensive if etf in score_row.index)
        new_entries = [etf for etf in new_top if etf not in current_offensive]

        should_switch = True
        min_advantage = abs(current_min_score) * score_threshold
        for new_etf in new_entries:
            new_score = score_row.get(new_etf, -999)
            # 新评分必须比当前最低评分高出至少 min_advantage
            if new_score < current_min_score + min_advantage:
                should_switch = False
                break

        if should_switch:
            current_offensive = new_top
        else:
            new_top = current_offensive.copy()
    else:
        current_offensive = new_top

    # 如果没有有效进攻持仓，使用默认top
    if not new_top:
        new_top = off_scores.nlargest(top_n).index.tolist()
        current_offensive = new_top

    off_weight = (1 - defensive_allocation) / len(new_top)
    for etf in new_top:
        allocation[etf] = off_weight

    return allocation, current_offensive


def calc_metrics(result_df):
    if result_df.empty:
        return None
    final_val = result_df.iloc[-1]["portfolio_value"]
    total_return = final_val - 1
    annual_return = (1 + total_return) ** (52 / len(result_df)) - 1
    max_dd = ((result_df["peak_value"] - result_df["portfolio_value"]) / result_df["peak_value"]).max()

    turnovers = []
    trade_count = 0
    offensive_switches = 0
    OFFENSIVE = ["纳指ETF", "沪深300ETF", "黄金ETF"]

    for i in range(1, len(result_df)):
        prev = result_df.iloc[i-1]
        curr = result_df.iloc[i]
        change = sum(abs(curr[f"weight_{e}"] - prev[f"weight_{e}"]) for e in ETFS)
        turnovers.append(change / 2)

        if change > 0.001:
            trade_count += 1

        for etf in OFFENSIVE:
            pw = prev[f"weight_{etf}"]
            cw = curr[f"weight_{etf}"]
            if (pw == 0 and cw > 0) or (pw > 0 and cw == 0):
                offensive_switches += 1

    avg_turnover = np.mean(turnovers)
    annual_turnover = avg_turnover * 52
    total_fee = sum(turnovers) * 2 * FEE_RATE

    return {
        "annual_return": annual_return,
        "max_drawdown": -max_dd,
        "total_return": total_return,
        "trade_weeks": trade_count,
        "offensive_switches": offensive_switches,
        "annual_turnover": annual_turnover,
        "total_fee_impact": total_fee,
    }


# 测试
params_list = [
    dict(mom_w=0.15, vol_w=0.2, defensive_allocation=0.40),
    dict(mom_w=0.30, vol_w=0.4, defensive_allocation=0.30),
]

threshold_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

print("=" * 100)
print("评分优势阈值测试")
print("=" * 100)
print("逻辑: 只有当新标的评分 >= 当前持仓最低评分 × (1+阈值) 时，才允许切换")
print()

results = []

for p in params_list:
    label = f"mom={p['mom_w']},vol={p['vol_w']},def={p['defensive_allocation']}"

    for threshold in threshold_values:
        r = run_backtest_score_threshold(score_threshold=threshold, **p)
        m = calc_metrics(r)
        m["方案"] = f"优势阈值={threshold:.0%}" if threshold > 0 else "无阈值(当前)"
        m["参数"] = label
        m["threshold"] = threshold
        results.append(m)
        calmar = m['annual_return'] / (-m['max_drawdown'])
        print(f"{label} | 优势阈值={threshold:.0%}: "
              f"年化={m['annual_return']*100:.2f}% 回撤={m['max_drawdown']*100:.2f}% Calmar={calmar:.3f} "
              f"调仓周数={m['trade_weeks']} 进攻切换={m['offensive_switches']} 换手={m['annual_turnover']*100:.0f}%")

# 汇总
print("\n" + "=" * 100)
print("汇总对比")
print("=" * 100)
df = pd.DataFrame(results)
df['Calmar'] = df['annual_return'] / (-df['max_drawdown'])
df['annual_pct'] = (df['annual_return']*100).round(2).astype(str)+'%'
df['dd_pct'] = (df['max_drawdown']*100).round(2).astype(str)+'%'
df['turnover_pct'] = (df['annual_turnover']*100).round(0).astype(str)+'%'
df['fee_pct'] = (df['total_fee_impact']*100).round(3).astype(str)+'%'
print(df[['方案', '参数', 'annual_pct', 'dd_pct', 'Calmar', 'offensive_switches', 'turnover_pct', 'fee_pct']].to_string(index=False))
