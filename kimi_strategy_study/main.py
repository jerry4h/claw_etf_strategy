import pandas as pd
from pathlib import Path
from backtest import run_backtest, grid_search
from analyzer import analyze_weekday_effect, calculate_metrics
from analysis import analyze_contribution
from data_loader import load_nav_data

def run_comparison():
    """对比 goal模式（止损后进攻仅选1个）vs 当前模式（止损后进攻选2个）"""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # 固定参数（基于修复未来函数后的临时最优）
    # 基于全量11,000组搜索结果的最优参数 (2026-04-19)
    # Top 1: mom=0.40, vol=0.60, val=0.0, top_n=2, def_alloc=0.30 → 14.07% / -9.97%
    mom_w, vol_w, val_w = 0.40, 0.60, 0.0
    top_n = 2
    defensive_allocation = 0.30
    stop_loss_threshold = 0.08

    print("=" * 60)
    print("策略对比回测: goal模式 vs 当前模式")
    print("=" * 60)

    # ---- 模式1: goal模式（止损后进攻仅选1个）----
    print("\n[1/2] 运行 goal模式 (stop_loss_offensive_n=1)...")
    goal_result = run_backtest(
        mom_w=mom_w, vol_w=vol_w, val_w=val_w,
        top_n=top_n, defensive_allocation=defensive_allocation,
        stop_loss_threshold=stop_loss_threshold,
        stop_loss_offensive_n=1
    )
    goal_result.to_csv(output_dir / "nav_history_goal_mode.csv", index=False)
    print(f"  保存至 nav_history_goal_mode.csv ({len(goal_result)}周)")

    # ---- 模式2: 当前模式（止损后进攻选2个）----
    print("\n[2/2] 运行 当前模式 (stop_loss_offensive_n=2)...")
    current_result = run_backtest(
        mom_w=mom_w, vol_w=vol_w, val_w=val_w,
        top_n=top_n, defensive_allocation=defensive_allocation,
        stop_loss_threshold=stop_loss_threshold,
        stop_loss_offensive_n=2
    )
    current_result.to_csv(output_dir / "nav_history_current_mode.csv", index=False)
    print(f"  保存至 nav_history_current_mode.csv ({len(current_result)}周)")

    # ---- 打印对比表 ----
    print("\n" + "=" * 60)
    print("绩效对比")
    print("=" * 60)

    def calc_metrics(result):
        total_ret = result.iloc[-1]["portfolio_value"] - 1
        annual_ret = (1 + total_ret) ** (52 / len(result)) - 1
        dd = (result["peak_value"] - result["portfolio_value"]) / result["peak_value"]
        max_dd = dd.max()
        sharpe = result["weekly_return"].mean() / result["weekly_return"].std() * np.sqrt(52)
        win_rate = (result["weekly_return"] > 0).mean()
        defensive_weeks = int(result["in_defensive"].sum())
        return {
            "年化收益": annual_ret,
            "总收益": total_ret,
            "最大回撤": max_dd,
            "夏普比率": sharpe,
            "胜率(周)": win_rate,
            "防御周数": defensive_weeks
        }

    import numpy as np
    goal_metrics = calc_metrics(goal_result)
    current_metrics = calc_metrics(current_result)

    print(f"{'指标':<12} {'goal模式':>12} {'当前模式':>12} {'差异':>12}")
    print("-" * 50)
    for key in goal_metrics:
        g = goal_metrics[key]
        c = current_metrics[key]
        diff = g - c
        if isinstance(g, float):
            if "收益" in key or "回撤" in key or "胜率" in key:
                print(f"{key:<12} {g:>11.2%} {c:>11.2%} {diff:>+11.2%}")
            else:
                print(f"{key:<12} {g:>11.2f} {c:>11.2f} {diff:>+11.2f}")
        else:
            print(f"{key:<12} {g:>12} {c:>12} {diff:>+12}")

    print("\n结论: ", end="")
    if goal_metrics["年化收益"] > current_metrics["年化收益"]:
        print(f"goal模式年化收益更高 (+{(goal_metrics['年化收益']-current_metrics['年化收益']):.2%})")
    else:
        print(f"当前模式年化收益更高 (+{(current_metrics['年化收益']-goal_metrics['年化收益']):.2%})")

    if goal_metrics["最大回撤"] < current_metrics["最大回撤"]:
        print(f"       goal模式回撤更小 ({(current_metrics['最大回撤']-goal_metrics['最大回撤']):.2%})")
    else:
        print(f"       当前模式回撤更小 ({(goal_metrics['最大回撤']-current_metrics['最大回撤']):.2%})")

    # 保存默认的 nav_history.csv（当前模式）用于后续分析
    current_result.to_csv(output_dir / "nav_history.csv", index=False)
    print(f"\n默认 nav_history.csv 已更新为当前模式结果")

    return goal_result, current_result

def main():
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # 运行对比
    goal_result, current_result = run_comparison()

    print("\n" + "=" * 60)
    print("网格搜索 (多进程)")
    print("=" * 60)
    print("\nRunning grid search with multiprocessing...")
    grid_result = grid_search(use_multiprocessing=True)
    if not grid_result.empty:
        grid_result = grid_result.sort_values("annual_return", ascending=False)
        grid_result.to_csv(output_dir / "param_grid_search.csv", index=False)
        print(f"\nSaved to param_grid_search.csv ({len(grid_result)} combinations)")
        print("\nTop 5 parameters:")
        print(grid_result.head().to_string())

    print("\n" + "=" * 60)
    print(" weekday效应分析")
    print("=" * 60)
    weekday_result = analyze_weekday_effect()
    weekday_result.to_csv(output_dir / "weekday_effect.csv", index=False)
    print(f"Saved to weekday_effect.csv")

    print("\nDone!")

if __name__ == "__main__":
    main()
