#!/usr/bin/env python3
"""交易成本敏感性分析 — 测试不同费率假设下策略的表现衰减。

用法:
    python scripts/cost_sensitivity.py
    python scripts/cost_sensitivity.py --fees 0.0001,0.0005,0.001,0.002

输出: 各费率下的 Sharpe、年化收益、最大回撤对比表。
"""
import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.strategy import load_config, StrategyConfig
from src.backtest import run_backtest


def main():
    parser = argparse.ArgumentParser(description='交易成本敏感性分析')
    parser.add_argument(
        '--fees', type=str, default='0.00005,0.0001,0.0005,0.001,0.0015,0.002',
        help='逗号分隔的费率列表（单边，默认: 0.5bp~20bp）'
    )
    parser.add_argument(
        '--config', type=str, default='config/strategy_v3_0_final.yaml',
        help='策略配置文件路径'
    )
    args = parser.parse_args()

    fee_rates = [float(f) for f in args.fees.split(',')]
    config_path = PROJECT / args.config
    base_cfg = load_config(config_path)

    print('=' * 70)
    print(' 交易成本敏感性分析')
    print('=' * 70)
    print(f' 策略: {base_cfg.name}')
    print(f' 基准费率: {base_cfg.fee_rate*10000:.1f}bp (单边)')
    print()
    print(f' {"费率(bp)":>10s} {"Sharpe":>8s} {"年化收益":>10s} {"最大回撤":>10s} {"Calmar":>8s} {"vs基准":>8s}')
    print(f' {"-"*10:>10s} {"-"*8:>8s} {"-"*10:>10s} {"-"*10:>10s} {"-"*8:>8s} {"-"*8:>8s}')

    baseline_sharpe = None
    results = []

    for fee in fee_rates:
        cfg = StrategyConfig(
            name=base_cfg.name,
            version=base_cfg.version,
            mom_w=base_cfg.mom_w,
            vol_w=base_cfg.vol_w,
            top_n=base_cfg.top_n,
            score_margin=base_cfg.score_margin,
            mom_window=base_cfg.mom_window,
            vol_window=base_cfg.vol_window,
            pe_window_years=base_cfg.pe_window_years,
            def_alloc=base_cfg.def_alloc,
            step_low=base_cfg.step_low,
            step_high=base_cfg.step_high,
            max_def=base_cfg.max_def,
            hongli_ratio=base_cfg.hongli_ratio,
            rebalance_threshold=base_cfg.rebalance_threshold,
            fee_rate=fee,
            anchor=base_cfg.anchor,
            stop_loss=base_cfg.stop_loss,
            recovery_weeks=base_cfg.recovery_weeks,
            max_single_alloc=base_cfg.max_single_alloc,
            overflow_to_defense_only=base_cfg.overflow_to_defense_only,
            inv_vol_enabled=base_cfg.inv_vol_enabled,
            inv_vol_window=base_cfg.inv_vol_window,
            nav_path=base_cfg.nav_path,
            pe_path=base_cfg.pe_path,
            start_date=base_cfg.start_date,
            end_date=base_cfg.end_date,
            risk_free_rate=base_cfg.risk_free_rate,
        )
        result = run_backtest(cfg)
        m = result.metrics

        if baseline_sharpe is None:
            baseline_sharpe = m['sharpe_ratio']

        delta = m['sharpe_ratio'] - baseline_sharpe
        fee_bp = fee * 10000

        print(f' {fee_bp:>10.1f} {m["sharpe_ratio"]:>8.3f} {m["annual_return"]*100:>9.2f}% {m["max_drawdown"]*100:>9.2f}% {m["calmar_ratio"]:>8.2f} {delta:>+7.3f}')
        results.append({'fee_bp': fee_bp, **m})

    print()
    print(' 说明: 1bp = 0.01%, 10bp = 0.10%')
    print(' ETF 实际交易成本参考:')
    print('   - 佣金: ~2.5bp (单边)')
    print('   - 买卖价差: 5~15bp (流动性较差的 ETF)')
    print('   - QDII 溢价: 0~200bp (纳指 ETF 偶发)')
    print('   - 综合单边成本估计: 10~20bp')
    print()

    # 找到 Sharpe 降至 1.0 以下的临界费率
    for r in results:
        if r['sharpe_ratio'] < 1.0:
            print(f' 注意: 费率 >= {r["fee_bp"]:.1f}bp 时 Sharpe < 1.0')
            break
    else:
        print(f' 所有测试费率下 Sharpe 均 >= 1.0')


if __name__ == '__main__':
    main()
