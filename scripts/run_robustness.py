#!/usr/bin/env python3
"""鲁棒性评估 CLI 入口 — 支持单策略和多策略对比评估 (v2)。

用法:
  # 单策略评估
  python scripts/run_robustness.py \
      --config config/strategy_v2_3_cap040.yaml \
      --output output/robustness_v23_baseline/

  # 双策略对比评估
  python scripts/run_robustness.py \
      --configs config/strategy_v2_3_cap040.yaml,config/strategy_v2_3_cap040_D4_tuned.yaml \
      --labels "v2.3 基线","v2.3+cap040+D4 tuned" \
      --output output/robustness_comparison/ \
      --n-mc 400 \
      --perturbation 0.15 \
      --n-wf 9

  # 含 OAT 完整评估
  python scripts/run_robustness.py \
      --configs config/strategy_v2_3_cap040.yaml,config/strategy_v2_3_cap040_D4_tuned.yaml \
      --labels "v2.3 基线","v2.3+cap040+D4 tuned" \
      --output output/robustness_v2/ \
      --n-mc 400 \
      --perturbation 0.15 \
      --oat \
      --n-wf 9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.robustness import (
    evaluate_robustness,
    generate_robustness_report,
)


def main():
    parser = argparse.ArgumentParser(
        description='鲁棒性评估 — 五指标简化版 v4 (DSR / PSS / MC 生存率 / 基准相对胜率 / SPS)'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='单策略配置 YAML 路径'
    )
    parser.add_argument(
        '--configs', type=str, default=None,
        help='多策略配置 YAML 路径，逗号分隔 (如 config/a.yaml,config/b.yaml)'
    )
    parser.add_argument(
        '--labels', type=str, default=None,
        help='策略标签，逗号分隔 (如 "v2.3 基线","v2.3+cap040+D4 tuned")'
    )
    parser.add_argument(
        '--output', type=str, default='output/robustness_evaluation/',
        help='输出目录 (默认: output/robustness_evaluation/)'
    )
    parser.add_argument(
        '--n-mc', type=int, default=400,
        help='Monte Carlo 运行次数 (默认: 400, v2)'
    )
    parser.add_argument(
        '--n-wf', type=int, default=9,
        help='Walk-Forward 窗口数 (默认: 9)'
    )
    parser.add_argument(
        '--n-trials', type=int, default=52,
        help='DSR 多重测试矫正的变体数 (默认: 52 = 基线1 + D4_1 + 中间50)'
    )
    parser.add_argument(
        '--n-jobs', type=int, default=-1,
        help='并行进程数 (默认: -1 = 全部 CPU)'
    )
    parser.add_argument(
        '--perturbation', type=float, default=0.15,
        help='MC 扰动幅度 (默认: 0.15 = ±15%%, v2)'
    )
    parser.add_argument(
        '--oat', action='store_true', default=False,
        help='启用 OAT 多级敏感度分析 (v2 新增, 每策略额外 49 次回测)'
    )

    parser.add_argument(
        '--sps', action='store_true', default=False,
        help='启用 SPS 起点敏感性分析 (v3 新增)'
    )
    parser.add_argument(
        '--sps-horizon', type=int, default=3,
        help='SPS 投资期限 (默认: 3 年)'
    )
    parser.add_argument(
        '--full-grid', action='store_true', default=False,
        help='启用 Phase 6 全参数 7 级网格评估 (v4 新增)'
    )
    parser.add_argument(
        '--n-local-mc', type=int, default=50,
        help='Phase 6 局部 MC 运行次数 (默认: 50)'
    )
    args = parser.parse_args()

    # Determine configs
    if args.configs:
        config_list = [c.strip() for c in args.configs.split(',')]
    elif args.config:
        config_list = [args.config.strip()]
    else:
        parser.error("必须指定 --config 或 --configs")

    # Determine labels
    if args.labels:
        labels = [l.strip() for l in args.labels.split(',')]
    else:
        labels = config_list  # fallback to config paths

    if len(labels) != len(config_list):
        print(f"[ERROR] labels 数量 ({len(labels)}) 与 configs 数量 ({len(config_list)}) 不匹配")
        sys.exit(1)

    print("=" * 70)
    print("鲁棒性评估 — 五指标简化版 v4")
    print(f"项目: {PROJECT_ROOT}")
    print(f"策略数: {len(config_list)}")
    print(f"MC 次数: {args.n_mc}")
    print(f"MC 扰动幅度: ±{args.perturbation*100:.0f}%")
    print(f"WF 窗口: {args.n_wf}")
    print(f"DSR trials: {args.n_trials}")
    print(f"OAT 敏感度: {'启用' if args.oat else '关闭'}")
    print(f"PSS: 始终运行（MC 数据零成本计算）")
    print(f"SPS: {'启用' if args.sps else '关闭'}")
    print(f"Phase 6 全网格: {'启用' if args.full_grid else '关闭'}")
    if args.full_grid:
        print(f"局部 MC 次数: {args.n_local_mc}")
    print("=" * 70)

    results = []
    for i, cfg_path in enumerate(config_list):
        label = labels[i]
        print(f"\n{'=' * 70}")
        print(f"策略 {i+1}/{len(config_list)}: {label}")
        print(f"配置: {cfg_path}")
        print(f"{'=' * 70}")

        try:
            result = evaluate_robustness(
                config_path=cfg_path,
                n_mc=args.n_mc,
                n_wf_windows=args.n_wf,
                n_trials=args.n_trials,
                n_jobs=args.n_jobs,
                perturbation=args.perturbation,
                oat=args.oat,
                sps=args.sps,
                sps_horizon=args.sps_horizon,
                full_grid=args.full_grid,
                n_local_mc=args.n_local_mc,
            )
            results.append(result)

            print(f"\n  ① DSR:                     {result.dsr:.4f}")
            print(f"  ② MC 生存率 (v2):          {result.mc_survival_rate*100:.1f}%")
            print(f"  ③ 基准相对胜率:            {result.benchmark_relative_win_rate*100:.1f}%")
            print(f"  基准 Sharpe:               {result.strategy_metrics['sharpe_ratio']:.4f}")
            print(f"  基准年化收益:              {result.strategy_metrics['annual_return']*100:.2f}%")
            print(f"  基准最大回撤:              {result.strategy_metrics['max_drawdown']*100:.2f}%")
            if args.oat and result.oat_sensitivity:
                print(f"  ④ OAT 敏感度:              已分析 {len(result.oat_sensitivity)} 个参数")

            # PSS always shown (computed from MC data, zero cost)
            if result.pss is not None:
                pss_m = result.pss
                pss_label = 'PSS 参数稳定性'
                print(f"  {'④' if not args.oat else '⑤'} PSS 年化 P10/P50/P90:      {pss_m['return_p10']*100:.2f}% / {pss_m['return_p50']*100:.2f}% / {pss_m['return_p90']*100:.2f}%")
                print(f"     PSS DD P10/P50/P90:       {pss_m['dd_p10']*100:.2f}% / {pss_m['dd_p50']*100:.2f}% / {pss_m['dd_p90']*100:.2f}%")
                print(f"     PSS Sharpe P10/P50/P90:   {pss_m['sharpe_p10']:.4f} / {pss_m['sharpe_p50']:.4f} / {pss_m['sharpe_p90']:.4f}")
                print(f"     PSS CV (ret/dd/shp):      {pss_m['return_cv']:.2f} / {pss_m['dd_cv']:.2f} / {pss_m['sharpe_cv']:.2f}")
            if args.sps and result.starting_point_sensitivity is not None:
                sps_m = result.starting_point_sensitivity
                sps_idx = '⑤' if not args.oat else '⑥'
                print(f"  {sps_idx} SPS 最差起点:           {sps_m['worst_annual_return']*100:.2f}% ({sps_m['worst_start_date']})")
            if args.full_grid and result.full_grid is not None:
                n_params = len(result.full_grid)
                n_points = sum(len(v) for v in result.full_grid.values())
                print(f"  P6 全参数网格:              {n_params} 参数, {n_points} 格点已评估")

        except Exception as e:
            print(f"\n  [ERROR] 策略 {label} 评估失败: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("\n[FATAL] 所有策略评估均失败，终止。")
        sys.exit(1)

    # 生成报告
    output_dir = PROJECT_ROOT / args.output
    report = generate_robustness_report(
        results=results,
        output_dir=str(output_dir),
        labels=labels,
    )

    report_path = output_dir / 'ROBUSTNESS_COMPARISON_REPORT.md'
    json_path = output_dir / 'robustness_results.json'
    print(f"\n{'=' * 70}")
    print(f"报告已生成: {report_path}")
    print(f"JSON 已生成: {json_path}")
    print("=" * 70)

    # Print report summary
    print(report)


if __name__ == '__main__':
    main()