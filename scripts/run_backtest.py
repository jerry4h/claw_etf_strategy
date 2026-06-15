#!/usr/bin/env python3
"""
虾池ETF轮动策略 — 单次回测 CLI 入口

用法：
    python scripts/run_backtest.py
    python scripts/run_backtest.py --config config/strategy_v2_3.yaml
    python scripts/run_backtest.py --start 2020-01-01 --end 2025-12-31
"""

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.strategy import load_config
from src.backtest import run_backtest
from src.report import generate_full_report


def main():
    parser = argparse.ArgumentParser(
        description='虾池ETF轮动策略 — 单次回测'
    )
    parser.add_argument(
        '--config', type=str, default='config/strategy_v2_3.yaml',
        help='策略配置文件路径（默认: config/strategy_v2_3.yaml）'
    )
    parser.add_argument(
        '--start', type=str, default=None,
        help='回测起始日期（YYYY-MM-DD）'
    )
    parser.add_argument(
        '--end', type=str, default=None,
        help='回测结束日期（YYYY-MM-DD）'
    )
    parser.add_argument(
        '--output', type=str, default='output/report_v2_3.md',
        help='报告输出路径（默认: output/report_v2_3.md）'
    )
    parser.add_argument(
        '--no-charts', action='store_true',
        help='不生成图表'
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f'错误: 配置文件不存在: {config_path}')
        sys.exit(1)

    print(f'加载配置: {config_path}')
    config = load_config(config_path)
    print(f'策略: {config.name} v{config.version}')
    print(f'参数: mom_w={config.mom_w}, vol_w={config.vol_w}, top_n={config.top_n}')
    print(f'防御: def_alloc={config.def_alloc}, step_low={config.step_low}, step_high={config.step_high}')
    print()

    print('运行回测...')
    result = run_backtest(
        config,
        start_date=args.start,
        end_date=args.end
    )

    m = result.metrics
    print()
    print('=' * 60)
    print(f'回测结果')
    print('=' * 60)
    print(f'  累计收益:      {m["total_return"]*100:.1f}%')
    print(f'  年化收益:      {m["annual_return"]*100:.2f}%')
    print(f'  最大回撤:      {m["max_drawdown"]*100:.2f}%')
    print(f'  标准夏普:      {m["sharpe_ratio"]:.3f}')
    print(f'  简化夏普:      {m["simple_sharpe"]:.3f}')
    print(f'  卡尔马:        {m["calmar_ratio"]:.2f}')
    print(f'  年化波动率:    {m["annual_volatility"]*100:.2f}%')
    print(f'  周胜率:        {m["win_rate"]*100:.1f}%')
    print(f'  回测周数:      {m["total_weeks"]}')
    print(f'  防御周数:      {m["defensive_weeks"]}')
    print()

    # 生成报告
    output_path = PROJECT_ROOT / args.output
    report = generate_full_report(
        result,
        output_path,
        include_charts=not args.no_charts
    )
    print(f'报告已生成: {output_path}')

    # 打印年度分解
    from src.report import generate_annual_breakdown
    print()
    print(generate_annual_breakdown(result))


if __name__ == '__main__':
    main()
