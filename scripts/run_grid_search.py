#!/usr/bin/env python3
"""
虾池ETF轮动策略 — 网格搜索 CLI 入口

用法：
    python scripts/run_grid_search.py
    python scripts/run_grid_search.py --config config/strategy_v2_3.yaml --jobs 4
"""

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.strategy import load_config
from src.backtest import grid_search


def main():
    parser = argparse.ArgumentParser(
        description='虾池ETF轮动策略 — 网格搜索'
    )
    parser.add_argument(
        '--config', type=str, default='config/strategy_v2_3.yaml',
        help='基准策略配置文件路径'
    )
    parser.add_argument(
        '--jobs', type=int, default=-1,
        help='并行进程数（默认: CPU 数的一半）'
    )
    parser.add_argument(
        '--output', type=str, default='output/grid_search_results.csv',
        help='结果输出路径'
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f'错误: 配置文件不存在: {config_path}')
        sys.exit(1)

    # 默认参数空间（量化工程自验用轻量空间）
    param_space = {
        'mom_w': [0.30, 0.35, 0.40],
        'vol_w': [0.20, 0.25, 0.30, 0.35],
        'top_n': [2],
        'def_alloc': [0.20, 0.25, 0.30],
        'step_high': [0.30, 0.35, 0.40, 0.45],
    }

    print(f'基准配置: {config_path}')
    print(f'参数空间: {param_space}')
    print()

    results = grid_search(
        param_space=param_space,
        base_config_path=config_path,
        n_jobs=args.jobs,
    )

    if results.empty:
        print('无有效结果')
        sys.exit(1)

    # 保存结果
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f'结果已保存: {output_path}')

    # 打印 Top 10
    print()
    print('=' * 80)
    print('Top 10 参数组合（按年化收益排序）')
    print('=' * 80)
    cols = ['mom_w', 'vol_w', 'top_n', 'def_alloc', 'step_high',
            'annual_return', 'max_drawdown', 'sharpe_ratio', 'calmar_ratio']
    print(results[cols].head(10).to_string(index=False))


if __name__ == '__main__':
    main()
