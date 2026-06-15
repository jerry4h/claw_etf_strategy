import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from data_loader import load_nav_data, ETFS, DEFENSIVE
from factors import calculate_all_factors
from strategy import calculate_composite_score, select_top_offensive
from backtest import FEE_RATE


def run_backtest_dynamic(mom_w=0.4, vol_w=0.4, val_w=0.0,
                          top_n=2, base_defensive_allocation=0.40,
                          dynamic_mode="off",  # off, score_spread, max_score, volatility
                          dynamic_range=0.10,
                          target_volatility=0.12, enable_vol_adj=True,
                          freq="2W-MON"):
    """
    动态防御回测:
    - dynamic_mode="off": 固定防御比例
    - dynamic_mode="score_spread": 基于进攻层评分离散度（标准差）
    - dynamic_mode="max_score": 基于进攻层最高评分
    - dynamic_mode="volatility": 基于组合波动率
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

    # 历史评分统计（用于标准化）
    all_scores = calculate_composite_score(
        factors["momentum"], factors["volatility"], factors["valuation"],
        ETFS, mom_w, vol_w, val_w
    )
    off_scores_hist = all_scores[["纳指ETF", "沪深300ETF", "黄金ETF"]]
    score_mean = off_scores_hist.mean().mean()
    score_std = off_scores_hist.std().mean()

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
        off_scores = score_row[["纳指ETF", "沪深300ETF", "黄金ETF"]].dropna()

        # 动态防御比例计算
        if dynamic_mode == "off" or in_defensive_mode:
            defensive_allocation = base_defensive_allocation
        elif dynamic_mode == "score_spread":
            # 评分离散度：标准差大=信号混乱=增加防御
            spread = off_scores.std() if len(off_scores) > 1 else 0
            spread_norm = min(1.0, max(0, (spread - 0.05) / 0.15))  # 归一化到0-1
            defensive_allocation = base_defensive_allocation + (spread_norm - 0.5) * 2 * dynamic_range
            defensive_allocation = max(0.15, min(0.70, defensive_allocation))
        elif dynamic_mode == "max_score":
            # 最高评分：高分=强趋势=减少防御
            max_s = off_scores.max() if len(off_scores) > 0 else 0
            max_norm = min(1.0, max(0, (max_s - score_mean) / (2 * score_std)))
            defensive_allocation = base_defensive_allocation + (0.5 - max_norm) * 2 * dynamic_range
            defensive_allocation = max(0.15, min(0.70, defensive_allocation))
        elif dynamic_mode == "score_zscore":
            # 最高评分的z-score
            max_s = off_scores.max() if len(off_scores) > 0 else 0
            zscore = (max_s - score_mean) / score_std if score_std > 0 else 0
            defensive_allocation = base_defensive_allocation - zscore * dynamic_range
            defensive_allocation = max(0.15, min(0.70, defensive_allocation))
        elif dynamic_mode == "momentum_count":
            # 正向动量标的数量：越多=趋势越强=减少防御
            positive_count = (off_scores > score_mean).sum()
            defensive_allocation = base_defensive_allocation + (1.5 - positive_count) / 3 * dynamic_range * 2
            defensive_allocation = max(0.15, min(0.70, defensive_allocation))
        else:
            defensive_allocation = base_defensive_allocation

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
            recovery_weeks = 4 if freq.startswith("W") or freq.startswith("2W") else 2
            if defensive_mode_weeks < recovery_weeks:
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

        allocation_record = {"date": next_date, "dynamic_def": defensive_allocation}
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


def calc_metrics(df, periods_per_year=26):
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

    # 动态防御的平均比例
    avg_def = df["dynamic_def"].mean() if "dynamic_def" in df.columns else 0

    return {
        "annual_return": annual_return,
        "max_drawdown": -max_dd,
        "total_return": total_return,
        "trade_count": trade_count,
        "annual_turnover": annual_turnover,
        "total_fee_impact": total_fee,
        "avg_def_alloc": avg_def,
    }


# ============ 测试 ============
params_list = [
    dict(mom_w=0.15, vol_w=0.2, val_w=0.0, top_n=2, base_defensive_allocation=0.40, label="保守型"),
    dict(mom_w=0.30, vol_w=0.4, val_w=0.0, top_n=2, base_defensive_allocation=0.30, label="进取型"),
]

modes = [
    ("off", "固定防御"),
    ("score_spread", "评分离散度"),
    ("max_score", "最高评分"),
    ("score_zscore", "评分z-score"),
    ("momentum_count", "正向动量数量"),
]

print("=" * 100)
print("动态防御测试（双周调仓）")
print("=" * 100)
print("逻辑: 根据进攻层信号质量动态调整防御比例")
print()

results = []

for p in params_list:
    base_label = p.pop("label")
    print(f"\n{'='*60}")
    print(f"参数: {base_label} (base_def={p['base_defensive_allocation']})")
    print(f"{'='*60}")

    for mode, mode_label in modes:
        r = run_backtest_dynamic(dynamic_mode=mode, freq="2W-MON", **p)
        m = calc_metrics(r, 26)
        m["方案"] = mode_label
        m["参数"] = base_label
        results.append(m)
        calmar = m['annual_return'] / (-m['max_drawdown'])
        print(f"  {mode_label}: 年化={m['annual_return']*100:.2f}% 回撤={m['max_drawdown']*100:.2f}% "
              f"Calmar={calmar:.3f} 调仓={m['trade_count']}次 平均防御={m['avg_def_alloc']*100:.1f}%")

# 汇总
print("\n" + "=" * 100)
print("汇总对比")
print("=" * 100)
df = pd.DataFrame(results)
df['Calmar'] = df['annual_return'] / (-df['max_drawdown'])
df['annual_pct'] = (df['annual_return']*100).round(2).astype(str)+'%'
df['dd_pct'] = (df['max_drawdown']*100).round(2).astype(str)+'%'
df['def_pct'] = (df['avg_def_alloc']*100).round(1).astype(str)+'%'
df['turnover_pct'] = (df['annual_turnover']*100).round(0).astype(str)+'%'
print(df[['方案', '参数', 'annual_pct', 'dd_pct', 'Calmar', 'def_pct', 'trade_count', 'turnover_pct']].to_string(index=False))
