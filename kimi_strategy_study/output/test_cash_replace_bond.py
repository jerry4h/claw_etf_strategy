import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from data_loader import load_nav_data, ETFS, DEFENSIVE
from factors import calculate_all_factors
from strategy import calculate_composite_score, select_top_offensive
from backtest import FEE_RATE


def run_backtest_cash(mom_w=0.4, vol_w=0.4, val_w=0.0,
                       top_n=2, defensive_allocation=0.40,
                       cash_yield=0.02,  # 现金年化收益率（如货币基金）
                       target_volatility=0.12, enable_vol_adj=True,
                       freq="2W-MON",
                       progressive_defense=False):
    """
    用现金替代国债的回测:
    - 防御层 = 红利低波ETF + 现金
    - 现金收益率 = cash_yield（默认2%年化，日收益=cash_yield/252）
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

    # 现金日收益率
    cash_daily_return = cash_yield / 252

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
        if portfolio_value < peak_value * (1 - 0.08):
            if not in_defensive_mode:
                in_defensive_mode = True
                defensive_mode_weeks = 0
                stop_loss_triggered_value = peak_value * (1 - 0.08)
        else:
            if in_defensive_mode:
                defensive_mode_weeks += 1

        # 防御层：红利低波 + 现金（各半）
        def_etf = "红利低波ETF"

        if in_defensive_mode:
            recovery_weeks = 4 if freq.startswith("W") or freq.startswith("2W") else 2
            if defensive_mode_weeks < recovery_weeks:
                if progressive_defense:
                    step = (1.0 - defensive_allocation) / recovery_weeks
                    prog_alloc = min(1.0, defensive_allocation + (defensive_mode_weeks + 1) * step)
                    def_etf_weight = prog_alloc / 2
                    cash_weight = prog_alloc / 2
                    new_allocation = {def_etf: def_etf_weight, "现金": cash_weight}
                    if prog_alloc < 1.0:
                        selected = select_top_offensive(scores, 1)
                        if selected:
                            off_weight = (1 - prog_alloc) / len(selected)
                            for etf in selected:
                                new_allocation[etf] = new_allocation.get(etf, 0) + off_weight
                else:
                    def_etf_weight = 0.5
                    cash_weight = 0.5
                    new_allocation = {def_etf: def_etf_weight, "现金": cash_weight}
            else:
                in_defensive_mode = False
                selected = select_top_offensive(scores, top_n)
                def_etf_weight = defensive_allocation / 2
                cash_weight = defensive_allocation / 2
                new_allocation = {def_etf: def_etf_weight, "现金": cash_weight}
                if selected:
                    off_weight = (1 - defensive_allocation) / len(selected)
                    for etf in selected:
                        new_allocation[etf] = new_allocation.get(etf, 0) + off_weight
        else:
            selected = select_top_offensive(scores, top_n)
            def_etf_weight = defensive_allocation / 2
            cash_weight = defensive_allocation / 2
            new_allocation = {def_etf: def_etf_weight, "现金": cash_weight}
            if selected:
                off_weight = (1 - defensive_allocation) / len(selected)
                for etf in selected:
                    new_allocation[etf] = new_allocation.get(etf, 0) + off_weight

        # 波动率调整（现金波动率为0）
        if enable_vol_adj and new_allocation and date in factors["volatility"].index:
            vol_row = factors["volatility"].loc[date]
            portfolio_vol = 0
            for etf, w in new_allocation.items():
                if etf == "现金":
                    continue  # 现金波动率为0
                if etf in vol_row.index and not pd.isna(vol_row[etf]):
                    portfolio_vol += w * vol_row[etf]
            portfolio_vol /= 100.0
            if portfolio_vol > 0 and target_volatility > 0:
                vol_multiplier = min(1.0, target_volatility / portfolio_vol)
                vol_multiplier = max(0.5, vol_multiplier)
                new_allocation = {etf: w * vol_multiplier for etf, w in new_allocation.items()}

        next_date = weekly_dates[i + 1]
        next_nav = nav_df.loc[next_date] if next_date in nav_df.index else nav_df.iloc[nav_df.index.get_indexer([next_date], method="ffill")[0]]

        # 计算期间交易日数（用于现金收益）
        if date in nav_df.index and next_date in nav_df.index:
            date_idx = nav_df.index.get_loc(date)
            next_idx = nav_df.index.get_loc(next_date)
            trading_days = next_idx - date_idx
        else:
            trading_days = 10  # 默认值（双周约10个交易日）

        weekly_return = 0
        for etf, weight in current_allocation.items():
            if etf == "现金":
                # 现金收益
                weekly_return += weight * cash_daily_return * trading_days
            elif etf in next_nav.index and not pd.isna(next_nav[etf]) and etf in nav_df.columns:
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
        allocation_record["weight_现金"] = new_allocation.get("现金", 0.0)

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
        change = sum(abs(curr.get(f"weight_{e}", 0) - prev.get(f"weight_{e}", 0)) for e in list(ETFS) + ["现金"])
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


# 对比测试
from backtest import run_backtest

params_list = [
    dict(mom_w=0.15, vol_w=0.2, val_w=0.0, top_n=2, defensive_allocation=0.40, label="保守型"),
    dict(mom_w=0.30, vol_w=0.4, val_w=0.0, top_n=2, defensive_allocation=0.30, label="进取型"),
]

print("=" * 100)
print("现金替代国债测试（双周调仓）")
print("=" * 100)
print("假设现金年化收益2%（货币基金水平），防御层=红利低波50%+现金50%")
print()

results = []

for p in params_list:
    base_label = p.pop("label")
    print(f"\n{'='*60}")
    print(f"参数: {base_label} (def={p['defensive_allocation']})")
    print(f"{'='*60}")

    # 基准：有国债（双周）
    r_base = run_backtest(freq="2W-MON", **p)
    m_base = calc_metrics(r_base, 26)
    m_base["方案"] = "有国债"
    m_base["参数"] = base_label
    results.append(m_base)
    calmar = m_base['annual_return'] / (-m_base['max_drawdown'])
    print(f"  有国债:   年化={m_base['annual_return']*100:.2f}% 回撤={m_base['max_drawdown']*100:.2f}% "
          f"Calmar={calmar:.3f} 调仓={m_base['trade_count']}次")

    # 现金替代（收益率2%）
    r_cash2 = run_backtest_cash(cash_yield=0.02, freq="2W-MON", **p)
    m_cash2 = calc_metrics(r_cash2, 26)
    m_cash2["方案"] = "现金替代(2%)"
    m_cash2["参数"] = base_label
    results.append(m_cash2)
    calmar = m_cash2['annual_return'] / (-m_cash2['max_drawdown'])
    ret_diff = (m_cash2['annual_return'] - m_base['annual_return']) * 100
    print(f"  现金(2%): 年化={m_cash2['annual_return']*100:.2f}% 回撤={m_cash2['max_drawdown']*100:.2f}% "
          f"Calmar={calmar:.3f} 调仓={m_cash2['trade_count']}次 差异={ret_diff:+.2f}%")

    # 现金替代（收益率3%）
    r_cash3 = run_backtest_cash(cash_yield=0.03, freq="2W-MON", **p)
    m_cash3 = calc_metrics(r_cash3, 26)
    m_cash3["方案"] = "现金替代(3%)"
    m_cash3["参数"] = base_label
    results.append(m_cash3)
    calmar = m_cash3['annual_return'] / (-m_cash3['max_drawdown'])
    ret_diff = (m_cash3['annual_return'] - m_base['annual_return']) * 100
    print(f"  现金(3%): 年化={m_cash3['annual_return']*100:.2f}% 回撤={m_cash3['max_drawdown']*100:.2f}% "
          f"Calmar={calmar:.3f} 调仓={m_cash3['trade_count']}次 差异={ret_diff:+.2f}%")

    # 现金替代（收益率0%）
    r_cash0 = run_backtest_cash(cash_yield=0.00, freq="2W-MON", **p)
    m_cash0 = calc_metrics(r_cash0, 26)
    m_cash0["方案"] = "现金替代(0%)"
    m_cash0["参数"] = base_label
    results.append(m_cash0)
    calmar = m_cash0['annual_return'] / (-m_cash0['max_drawdown'])
    ret_diff = (m_cash0['annual_return'] - m_base['annual_return']) * 100
    print(f"  现金(0%): 年化={m_cash0['annual_return']*100:.2f}% 回撤={m_cash0['max_drawdown']*100:.2f}% "
          f"Calmar={calmar:.3f} 调仓={m_cash0['trade_count']}次 差异={ret_diff:+.2f}%")

    # 降低防御比例（用更少现金）
    p_lower = dict(p, defensive_allocation=p['defensive_allocation'] - 0.10)
    r_lower = run_backtest_cash(cash_yield=0.02, freq="2W-MON", **p_lower)
    m_lower = calc_metrics(r_lower, 26)
    m_lower["方案"] = f"现金(2%)+降防御{p_lower['defensive_allocation']}"
    m_lower["参数"] = base_label
    results.append(m_lower)
    calmar = m_lower['annual_return'] / (-m_lower['max_drawdown'])
    ret_diff = (m_lower['annual_return'] - m_base['annual_return']) * 100
    print(f"  现金(2%)+def={p_lower['defensive_allocation']}: 年化={m_lower['annual_return']*100:.2f}% "
          f"回撤={m_lower['max_drawdown']*100:.2f}% Calmar={calmar:.3f} 差异={ret_diff:+.2f}%")

# 汇总
print("\n" + "=" * 100)
print("汇总对比")
print("=" * 100)
df = pd.DataFrame(results)
df['Calmar'] = df['annual_return'] / (-df['max_drawdown'])
df['annual_pct'] = (df['annual_return']*100).round(2).astype(str)+'%'
df['dd_pct'] = (df['max_drawdown']*100).round(2).astype(str)+'%'
df['turnover_pct'] = (df['annual_turnover']*100).round(0).astype(str)+'%'
print(df[['方案', '参数', 'annual_pct', 'dd_pct', 'Calmar', 'trade_count', 'turnover_pct']].to_string(index=False))
