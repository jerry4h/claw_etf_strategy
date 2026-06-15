import pandas as pd
import numpy as np
from data_loader import load_nav_data, calculate_returns, ETFS, DEFENSIVE, OFFENSIVE
from factors import calculate_all_factors
from strategy import calculate_composite_score, select_top_offensive, get_allocation

FEE_RATE = 0.00005
DEFENSIVE_POOL = DEFENSIVE  # ["红利低波ETF", "国债ETF"]
OFFENSIVE_POOL = OFFENSIVE  # ["纳指ETF", "沪深300ETF", "黄金ETF"]

def run_backtest(start_date=None, end_date=None,
                 mom_w=0.4, vol_w=0.4, val_w=0.2,
                 mom_window=20, vol_window=20, val_window=60,
                 top_n=2, stop_loss_threshold=0.08,
                 recovery_weeks=4,
                 defensive_allocation=0.55,
                 target_volatility=0.12,
                 enable_vol_adj=True,
                 stop_loss_offensive_n=2,
                 freq="W-MON"):
    """
    回测引擎:
    - 防御模式可持续多周，触发后观察 recovery_weeks 周才允许恢复
    - 防御层ETF根据时间动态选择（红利低波仅2019+可用）
    - P3: 可选波动率仓位调整 — 组合波动率超过 target_volatility 时整体缩减持仓
    - stop_loss_offensive_n: 止损触发后进攻层选几个（goal模式=1，当前模式=2）
    """
    nav_df = load_nav_data()

    if start_date:
        nav_df = nav_df[nav_df.index >= pd.to_datetime(start_date)]
    if end_date:
        nav_df = nav_df[nav_df.index <= pd.to_datetime(end_date)]

    ret_df = calculate_returns(nav_df)
    factors = calculate_all_factors(nav_df, ret_df, mom_window, vol_window, val_window)

    weekly_dates = nav_df.resample(freq).indices
    weekly_dates = sorted(weekly_dates.keys())

    portfolio_value = 1.0
    peak_value = 1.0
    current_allocation = {}
    portfolio_history = []

    # ---- 防御模式持久化 ----
    in_defensive_mode = False
    defensive_mode_weeks = 0  # 防御模式已持续周数
    stop_loss_triggered_value = None  # 记录触发止损时的净值

    for i, date in enumerate(weekly_dates[:-1]):
        scores = calculate_composite_score(
            factors["momentum"].loc[:date],
            factors["volatility"].loc[:date],
            factors["valuation"].loc[:date],
            ETFS, mom_w, vol_w, val_w
        )

        if scores.empty:
            continue

        # ---- 防御模式状态更新 ----
        if portfolio_value < peak_value * (1 - stop_loss_threshold):
            if not in_defensive_mode:
                in_defensive_mode = True
                defensive_mode_weeks = 0
                stop_loss_triggered_value = peak_value * (1 - stop_loss_threshold)
        else:
            if in_defensive_mode:
                defensive_mode_weeks += 1

        # ---- 防御层: 固定2个ETF（红利低波 + 国债）----
        available_defensive = DEFENSIVE_POOL
        n_defensive = len(available_defensive)

        if in_defensive_mode:
            if stop_loss_offensive_n == 1:
                # goal模式：止损触发后 70%防御 + 30%进攻仅选1个
                def_alloc = 0.70
                def_weight = def_alloc / n_defensive
                new_allocation = {etf: def_weight for etf in available_defensive}

                selected = select_top_offensive(scores, 1)  # 只选1个
                if selected:
                    off_alloc = 0.30
                    off_weight = off_alloc / len(selected)
                    for etf in selected:
                        new_allocation[etf] = new_allocation.get(etf, 0) + off_weight
                # 恢复条件：净值超过止损触发时的水平
                if portfolio_value >= stop_loss_triggered_value:
                    in_defensive_mode = False
                    stop_loss_triggered_value = None
            else:
                # 当前模式（stop_loss_offensive_n=2）：100%防御，观察recovery_weeks周后恢复
                if defensive_mode_weeks < recovery_weeks:
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

        # ---- P3: 波动率仓位调整 ----
        if enable_vol_adj and new_allocation and date in factors["volatility"].index:
            vol_row = factors["volatility"].loc[date]
            portfolio_vol = 0
            vol_count = 0
            for etf, w in new_allocation.items():
                if etf in vol_row.index and not pd.isna(vol_row[etf]):
                    portfolio_vol += w * vol_row[etf]
                    vol_count += 1
            portfolio_vol /= 100.0  # 转为小数
            if portfolio_vol > 0 and target_volatility > 0:
                vol_multiplier = min(1.0, target_volatility / portfolio_vol)
                vol_multiplier = max(0.5, vol_multiplier)  # 最多砍到50%仓位
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

        # 记录持仓明细
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

def _run_single_backtest(params):
    """单次回测的参数封装，用于多进程"""
    mom_w, vol_w, val_w, top_n, def_alloc = params
    result = run_backtest(
        mom_w=mom_w, vol_w=vol_w, val_w=val_w,
        top_n=top_n, defensive_allocation=def_alloc
    )
    if result.empty:
        return None
    final_val = result.iloc[-1]["portfolio_value"]
    total_return = final_val - 1
    annual_return = (1 + total_return) ** (52 / len(result)) - 1
    max_dd = ((result["peak_value"] - result["portfolio_value"]) / result["peak_value"]).max()
    defensive_days = result["in_defensive"].sum()
    total_days = len(result)
    return {
        "mom_w": mom_w,
        "vol_w": vol_w,
        "val_w": val_w,
        "top_n": top_n,
        "defensive_allocation": def_alloc,
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": -max_dd,
        "defensive_weeks": int(defensive_days),
        "total_weeks": total_days
    }

def grid_search(use_multiprocessing=True):
    """
    单阶段全量网格搜索:
    - 覆盖完整参数空间（跳过 top_n=3，因历史数据显示全部不满足回撤约束）
    - 使用多进程并行加速
    """
    import multiprocessing

    print("=" * 60)
    print("全量网格搜索（单阶段，跳过 top_n=3）")
    print("=" * 60)

    all_params = [
        (mom_w, vol_w, val_w, top_n, def_alloc)
        for mom_w in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        for vol_w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for val_w in [0.0, 0.1, 0.2, 0.3, 0.4]
        for top_n in [1, 2]
        for def_alloc in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    ]
    all_params = list(set(all_params))  # 去重
    print(f"参数空间: {len(all_params)} 种组合")

    n_workers = max(1, int(multiprocessing.cpu_count() * 0.5))
    print(f"使用 {n_workers} 个进程 (总CPU核心: {multiprocessing.cpu_count()})")

    results = []
    try:
        with multiprocessing.Pool(n_workers) as pool:
            from tqdm import tqdm
            results = list(tqdm(
                pool.imap(_run_single_backtest, all_params),
                total=len(all_params),
                desc="Grid Search"
            ))
        results = [r for r in results if r is not None]
    except Exception as e:
        print(f"多进程失败: {e}，切换为串行")
        for params in all_params:
            r = _run_single_backtest(params)
            if r is not None:
                results.append(r)

    all_results = pd.DataFrame(results)
    all_results = all_results.drop_duplicates(subset=['mom_w', 'vol_w', 'val_w', 'top_n', 'defensive_allocation'])

    print(f"\n搜索完成，有效结果: {len(all_results)}")

    # 只保留满足约束的结果（max_drawdown存储为负数，如-0.12表示12%回撤）
    valid_results = all_results[(all_results['annual_return'] > 0.10) & (all_results['max_drawdown'] > -0.15)]
    print(f"满足约束(年化>10%, 回撤<15%)的结果: {len(valid_results)} 个")

    # 最终Top10
    print("\n" + "=" * 60)
    print("年化收益 Top 10 参数:")
    print("=" * 60)
    top10 = all_results.nlargest(10, 'annual_return')
    print(top10[['mom_w', 'vol_w', 'val_w', 'top_n', 'defensive_allocation', 'annual_return', 'max_drawdown']].to_string(index=False))

    return all_results
