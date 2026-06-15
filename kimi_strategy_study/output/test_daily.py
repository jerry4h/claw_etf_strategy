import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from data_loader import load_nav_data, ETFS, DEFENSIVE, OFFENSIVE
from factors import calculate_all_factors
from strategy import calculate_composite_score

FEE_RATE = 0.00005


def run_backtest_daily(mom_w=0.4, vol_w=0.4, val_w=0.0,
                        top_n=2, defensive_allocation=0.40,
                        target_volatility=0.12, enable_vol_adj=True):
    """日频回测（每个交易日调仓）"""
    nav_df = load_nav_data()
    ret_df = nav_df.pct_change().dropna()
    factors = calculate_all_factors(nav_df, ret_df, 20, 20, 60)

    dates = nav_df.index.tolist()

    portfolio_value = 1.0
    peak_value = 1.0
    current_allocation = {}
    portfolio_history = []

    in_defensive_mode = False
    defensive_mode_weeks = 0
    stop_loss_triggered_value = None

    for i, date in enumerate(dates[:-1]):
        scores = calculate_composite_score(
            factors["momentum"].loc[:date],
            factors["volatility"].loc[:date],
            factors["valuation"].loc[:date],
            ETFS, mom_w, vol_w, val_w
        )

        if scores.empty:
            continue

        # 止损检查（日频：用20日累计回撤替代单周回撤）
        # 为了公平对比，保持8%阈值，但按日检查
        if portfolio_value < peak_value * (1 - 0.08):
            if not in_defensive_mode:
                in_defensive_mode = True
                defensive_mode_weeks = 0
                stop_loss_triggered_value = peak_value * (1 - 0.08)
        else:
            if in_defensive_mode:
                defensive_mode_weeks += 1

        # 防御层
        n_defensive = len(DEFENSIVE)

        if in_defensive_mode:
            # 日频简化：用20个交易日（约4周）作为恢复观察期
            if defensive_mode_weeks < 20:
                def_weight = 1.0 / n_defensive
                new_allocation = {etf: def_weight for etf in DEFENSIVE}
            else:
                in_defensive_mode = False
                selected = _select_top(scores, top_n)
                def_weight = defensive_allocation / n_defensive
                new_allocation = {etf: def_weight for etf in DEFENSIVE}
                if selected:
                    off_weight = (1 - defensive_allocation) / len(selected)
                    for etf in selected:
                        new_allocation[etf] = new_allocation.get(etf, 0) + off_weight
        else:
            selected = _select_top(scores, top_n)
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

        next_date = dates[i + 1]
        next_nav = nav_df.loc[next_date]

        daily_return = 0
        for etf, weight in current_allocation.items():
            if etf in next_nav.index and not pd.isna(next_nav[etf]) and etf in nav_df.columns:
                prev_nav = nav_df.loc[date][etf] if date in nav_df.index and etf in nav_df.loc[date].index else nav_df[etf].iloc[nav_df.index.get_indexer([date], method="ffill")[0]]
                if not pd.isna(prev_nav) and prev_nav != 0:
                    ret = (next_nav[etf] - prev_nav) / prev_nav
                    daily_return += weight * ret

        if i > 0:
            turnover = sum(
                abs(current_allocation.get(etf, 0) - new_allocation.get(etf, 0))
                for etf in set(current_allocation) | set(new_allocation)
            )
            daily_return -= turnover * FEE_RATE

        portfolio_value *= (1 + daily_return)
        peak_value = max(peak_value, portfolio_value)
        current_allocation = new_allocation

        allocation_record = {"date": next_date}
        for etf in ETFS:
            allocation_record[f"weight_{etf}"] = new_allocation.get(etf, 0.0)

        portfolio_history.append({
            "date": next_date,
            "portfolio_value": portfolio_value,
            "peak_value": peak_value,
            "daily_return": daily_return,
            "in_defensive": in_defensive_mode,
            **allocation_record
        })

    return pd.DataFrame(portfolio_history)


def _select_top(scores, top_n):
    if scores.empty:
        return []
    offensive_scores = scores[["纳指ETF", "沪深300ETF", "黄金ETF"]].dropna(axis=1, how="all")
    if offensive_scores.empty or offensive_scores.iloc[-1].max() == 0:
        return []
    return offensive_scores.iloc[-1].nlargest(top_n).index.tolist()


def calc_metrics(df, is_daily=True):
    if df.empty:
        return None
    final_val = df.iloc[-1]["portfolio_value"]
    total_return = final_val - 1
    periods_per_year = 252 if is_daily else 52
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
        "trade_days": trade_count,
        "annual_turnover": annual_turnover,
        "total_fee_impact": total_fee,
    }


# 对比测试
from backtest import run_backtest

params_list = [
    dict(mom_w=0.15, vol_w=0.2, val_w=0.0, top_n=2, defensive_allocation=0.40),
    dict(mom_w=0.30, vol_w=0.4, val_w=0.0, top_n=2, defensive_allocation=0.30),
]

print("=" * 90)
print("日频 vs 周频调仓对比")
print("=" * 90)

for p in params_list:
    label = f"mom={p['mom_w']},vol={p['vol_w']},def={p['defensive_allocation']}"
    print(f"\n{label}")
    print("-" * 60)

    # 周频
    r_w = run_backtest(**p)
    m_w = calc_metrics(r_w, is_daily=False)
    print(f"  周频: 年化={m_w['annual_return']*100:.2f}% 回撤={m_w['max_drawdown']*100:.2f}% "
          f"Calmar={m_w['annual_return']/(-m_w['max_drawdown']):.3f} "
          f"调仓={m_w['trade_days']}次 换手={m_w['annual_turnover']*100:.0f}% 手续费={m_w['total_fee_impact']*100:.3f}%")

    # 日频
    r_d = run_backtest_daily(**p)
    m_d = calc_metrics(r_d, is_daily=True)
    print(f"  日频: 年化={m_d['annual_return']*100:.2f}% 回撤={m_d['max_drawdown']*100:.2f}% "
          f"Calmar={m_d['annual_return']/(-m_d['max_drawdown']):.3f} "
          f"调仓={m_d['trade_days']}次 换手={m_d['annual_turnover']*100:.0f}% 手续费={m_d['total_fee_impact']*100:.3f}%")

    # 差异
    ret_diff = (m_d['annual_return'] - m_w['annual_return']) * 100
    dd_diff = (m_d['max_drawdown'] - m_w['max_drawdown']) * 100
    fee_diff = (m_d['total_fee_impact'] - m_w['total_fee_impact']) * 100
    print(f"  差异: 年化{ret_diff:+.2f}% 回撤{dd_diff:+.2f}% 手续费+{fee_diff:.3f}%")
