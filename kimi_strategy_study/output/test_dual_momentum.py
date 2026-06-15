import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from data_loader import load_nav_data, ETFS, DEFENSIVE
from factors import calculate_all_factors, calculate_momentum, calculate_volatility
from strategy import calculate_composite_score, select_top_offensive
from backtest import FEE_RATE


def run_backtest_dual_momentum(mom_w=0.4, vol_w=0.4, val_w=0.0,
                                mom_short_window=20, mom_long_window=60,
                                mom_blend_ratio=0.7,  # 短周期权重
                                top_n=2, defensive_allocation=0.40,
                                target_volatility=0.12, enable_vol_adj=True,
                                freq="2W-MON",
                                progressive_defense=False,
                                dual_confirm=False):  # 是否双动量确认模式
    """
    双动量回测:
    - mom_blend_ratio: 短周期动量权重（如0.7 = 70% 20日 + 30% 60日）
    - dual_confirm: True时，只有当短周期和长周期动量都排名前N才选入
    """
    nav_df = load_nav_data()
    ret_df = nav_df.pct_change().dropna()

    # 计算双动量
    mom_short = calculate_momentum(ret_df, mom_short_window)
    mom_long = calculate_momentum(ret_df, mom_long_window)

    # 混合动量
    mom_blended = mom_blend_ratio * mom_short + (1 - mom_blend_ratio) * mom_long

    # 其他因子
    vol = calculate_volatility(ret_df, 20)
    from factors import calculate_valuation_percentile
    val = calculate_valuation_percentile(nav_df, 60)

    factors = {
        "momentum": mom_blended.shift(1),
        "volatility": vol.shift(1),
        "valuation": val.shift(1),
    }

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
        # 计算综合评分
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
            recovery_weeks = 4 if freq.startswith("W") or freq.startswith("2W") else 2
            if defensive_mode_weeks < recovery_weeks:
                if progressive_defense:
                    step = (1.0 - defensive_allocation) / recovery_weeks
                    prog_alloc = min(1.0, defensive_allocation + (defensive_mode_weeks + 1) * step)
                    def_weight = prog_alloc / n_defensive
                    new_allocation = {etf: def_weight for etf in DEFENSIVE}
                    if prog_alloc < 1.0:
                        selected = _select_top_dual(scores, mom_short.loc[:date], mom_long.loc[:date], 1, dual_confirm)
                        if selected:
                            off_weight = (1 - prog_alloc) / len(selected)
                            for etf in selected:
                                new_allocation[etf] = new_allocation.get(etf, 0) + off_weight
                else:
                    def_weight = 1.0 / n_defensive
                    new_allocation = {etf: def_weight for etf in DEFENSIVE}
            else:
                in_defensive_mode = False
                selected = _select_top_dual(scores, mom_short.loc[:date], mom_long.loc[:date], top_n, dual_confirm)
                def_weight = defensive_allocation / n_defensive
                new_allocation = {etf: def_weight for etf in DEFENSIVE}
                if selected:
                    off_weight = (1 - defensive_allocation) / len(selected)
                    for etf in selected:
                        new_allocation[etf] = new_allocation.get(etf, 0) + off_weight
        else:
            selected = _select_top_dual(scores, mom_short.loc[:date], mom_long.loc[:date], top_n, dual_confirm)
            def_weight = defensive_allocation / n_defensive
            new_allocation = {etf: def_weight for etf in DEFENSIVE}
            if selected:
                off_weight = (1 - defensive_allocation) / len(selected)
                for etf in selected:
                    new_allocation[etf] = new_allocation.get(etf, 0) + off_weight

        # 波动率调整
        if enable_vol_adj and new_allocation and date in vol.index:
            vol_row = vol.loc[date]
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


def _select_top_dual(scores, mom_short, mom_long, top_n, dual_confirm):
    """双动量选股"""
    if scores.empty:
        return []

    OFFENSIVE = ["纳指ETF", "沪深300ETF", "黄金ETF"]
    offensive_scores = scores[OFFENSIVE].dropna(axis=1, how="all")
    if offensive_scores.empty or offensive_scores.iloc[-1].max() == 0:
        return []

    if dual_confirm:
        # 双动量确认：短周期和长周期都必须排名前N+1
        short_scores = mom_short[OFFENSIVE].iloc[-1].dropna() if not mom_short.empty else pd.Series()
        long_scores = mom_long[OFFENSIVE].iloc[-1].dropna() if not mom_long.empty else pd.Series()

        if short_scores.empty or long_scores.empty:
            return offensive_scores.iloc[-1].nlargest(top_n).index.tolist()

        short_top = set(short_scores.nlargest(top_n + 1).index)
        long_top = set(long_scores.nlargest(top_n + 1).index)

        # 交集：两个周期都排名靠前的
        consensus = list(short_top & long_top)

        # 如果交集不足top_n个，从综合评分中补足
        if len(consensus) < top_n:
            all_top = offensive_scores.iloc[-1].nlargest(top_n + 2).index.tolist()
            for etf in all_top:
                if etf not in consensus and len(consensus) < top_n:
                    consensus.append(etf)

        return consensus[:top_n]
    else:
        # 混合动量模式：直接用综合评分
        return offensive_scores.iloc[-1].nlargest(top_n).index.tolist()


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

blend_ratios = [1.0, 0.8, 0.7, 0.6, 0.5]  # 短周期权重，1.0=纯20日，0.5=50/50

print("=" * 110)
print("双动量测试：20日+60日动量混合（双周调仓）")
print("=" * 110)
print("blend_ratio: 短周期(20日)权重，1-blend_ratio=长周期(60日)权重")
print()

results = []

for p in params_list:
    base_label = p.pop("label")
    print(f"\n{'='*60}")
    print(f"参数: {base_label} (mom={p['mom_w']}, vol={p['vol_w']}, def={p['defensive_allocation']})")
    print(f"{'='*60}")

    # 基准：纯20日动量
    r_base = run_backtest(freq="2W-MON", **p)
    m_base = calc_metrics(r_base, 26)
    m_base["方案"] = "纯20日动量"
    m_base["参数"] = base_label
    m_base["blend"] = "1.0"
    results.append(m_base)
    calmar = m_base['annual_return'] / (-m_base['max_drawdown'])
    print(f"  纯20日动量: 年化={m_base['annual_return']*100:.2f}% 回撤={m_base['max_drawdown']*100:.2f}% "
          f"Calmar={calmar:.3f} 调仓={m_base['trade_count']}次")

    # 双动量混合
    for blend in blend_ratios[1:]:
        r = run_backtest_dual_momentum(mom_blend_ratio=blend, freq="2W-MON", **p)
        m = calc_metrics(r, 26)
        m["方案"] = f"混合{blend:.0%}"
        m["参数"] = base_label
        m["blend"] = str(blend)
        results.append(m)
        calmar = m['annual_return'] / (-m['max_drawdown'])
        ret_diff = (m['annual_return'] - m_base['annual_return']) * 100
        print(f"  混合{blend:.0%}(20日): 年化={m['annual_return']*100:.2f}% 回撤={m['max_drawdown']*100:.2f}% "
              f"Calmar={calmar:.3f} 调仓={m['trade_count']}次 差异={ret_diff:+.2f}%")

    # 双动量确认模式
    r_confirm = run_backtest_dual_momentum(mom_blend_ratio=0.7, dual_confirm=True, freq="2W-MON", **p)
    m_confirm = calc_metrics(r_confirm, 26)
    m_confirm["方案"] = "双动量确认"
    m_confirm["参数"] = base_label
    m_confirm["blend"] = "0.7+确认"
    results.append(m_confirm)
    calmar = m_confirm['annual_return'] / (-m_confirm['max_drawdown'])
    ret_diff = (m_confirm['annual_return'] - m_base['annual_return']) * 100
    print(f"  双动量确认: 年化={m_confirm['annual_return']*100:.2f}% 回撤={m_confirm['max_drawdown']*100:.2f}% "
          f"Calmar={calmar:.3f} 调仓={m_confirm['trade_count']}次 差异={ret_diff:+.2f}%")

# 汇总
print("\n" + "=" * 110)
print("汇总对比")
print("=" * 110)
df = pd.DataFrame(results)
df['Calmar'] = df['annual_return'] / (-df['max_drawdown'])
df['annual_pct'] = (df['annual_return']*100).round(2).astype(str)+'%'
df['dd_pct'] = (df['max_drawdown']*100).round(2).astype(str)+'%'
df['turnover_pct'] = (df['annual_turnover']*100).round(0).astype(str)+'%'
print(df[['方案', '参数', 'annual_pct', 'dd_pct', 'Calmar', 'trade_count', 'turnover_pct']].to_string(index=False))
