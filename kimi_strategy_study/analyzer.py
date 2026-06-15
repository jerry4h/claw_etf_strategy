import pandas as pd
import numpy as np
from pathlib import Path
from data_loader import load_nav_data, calculate_returns, ETFS
from backtest import run_backtest

def analyze_annual_performance():
    nav_df = load_nav_data()
    ret_df = calculate_returns(nav_df)

    years = range(2013, 2027)
    results = []

    for year in years:
        year_start = f"{year}-01-01"
        year_end = f"{year}-12-31"

        year_nav = nav_df[(nav_df.index >= year_start) & (nav_df.index <= year_end)]
        if year_nav.empty:
            continue

        year_ret = ret_df[(ret_df.index >= year_start) & (ret_df.index <= year_end)]

        benchmark_return = year_ret["沪深300ETF"].mean() * len(year_ret) if "沪深300ETF" in year_ret.columns else 0

        for mom_w, vol_w, top_n in [(0.4, 0.4, 2)]:
            backtest_result = run_backtest(
                start_date=year_start, end_date=year_end,
                mom_w=mom_w, vol_w=vol_w, top_n=top_n
            )

            if not backtest_result.empty:
                strat_return = backtest_result.iloc[-1]["portfolio_value"] - 1

                results.append({
                    "year": year,
                    "strategy_return": strat_return,
                    "benchmark_return": benchmark_return,
                    "excess": strat_return - benchmark_return
                })

    return pd.DataFrame(results)

def analyze_weekday_effect():
    """
    回答"一周哪一天操作对收益影响大"。
    通过重新运行回测，比较周一/周二/周三/周四/周五调仓的最终绩效差异。
    """
    from backtest import run_backtest

    mom_w, vol_w, val_w = 0.40, 0.60, 0.0
    top_n = 2
    defensive_allocation = 0.30
    stop_loss_threshold = 0.08

    weekday_names = ["周一", "周二", "周三", "周四", "周五"]
    freqs = ["W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI"]

    results = []
    for name, freq in zip(weekday_names, freqs):
        result = run_backtest(
            mom_w=mom_w, vol_w=vol_w, val_w=val_w,
            top_n=top_n, defensive_allocation=defensive_allocation,
            stop_loss_threshold=stop_loss_threshold,
            freq=freq
        )
        if not result.empty:
            total_ret = result.iloc[-1]["portfolio_value"] - 1
            annual_ret = (1 + total_ret) ** (52 / len(result)) - 1
            max_dd = ((result["peak_value"] - result["portfolio_value"]) / result["peak_value"]).max()
            win_rate = (result["weekly_return"] > 0).mean()
            results.append({
                "weekday": name,
                "freq": freq,
                "annual_return": annual_ret,
                "max_drawdown": max_dd,
                "win_rate": win_rate,
                "total_weeks": len(result)
            })

    return pd.DataFrame(results)

def calculate_metrics(portfolio_history):
    if portfolio_history.empty:
        return {}

    total_return = portfolio_history.iloc[-1]["portfolio_value"] - 1
    annual_return = (1 + total_return) ** (52 / len(portfolio_history)) - 1

    portfolio_history["drawdown"] = (portfolio_history["peak_value"] - portfolio_history["portfolio_value"]) / portfolio_history["peak_value"]
    max_drawdown = portfolio_history["drawdown"].max()

    returns = portfolio_history["weekly_return"]
    sharpe = returns.mean() / returns.std() * np.sqrt(52) if returns.std() > 0 else 0
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar
    }
