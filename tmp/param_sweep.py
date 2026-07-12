#!/usr/bin/env python3
"""中证500ETF参数扫描 — 单参数遍历找大方向"""
import sys, json
sys.path.insert(0, '/home/ubuntu/claw_etf_strategy')
_V = '/home/ubuntu/claw_etf_strategy/.venv/lib/python3.12/site-packages'
import pathlib; pathlib.Path(_V).exists() and sys.path.insert(0, str(_V))

import os
os.environ['MPLBACKEND'] = 'Agg'  # suppress figure generation

from src.backtest import run_backtest
from src.strategy import load_config
from pathlib import Path

ROOT = Path('/home/ubuntu/claw_etf_strategy')
cfg = load_config(ROOT / 'config/strategy_v3_0_final.yaml')

def run(param_name, param_value):
    old = getattr(cfg, param_name, None)
    setattr(cfg, param_name, param_value)
    try:
        r = run_backtest(cfg)
        m = r.metrics
        return {
            'param': param_value,
            'sharpe': m['sharpe_ratio'],
            'ann_ret': m['annual_return'] * 100,
            'dd': m['max_drawdown'] * 100,
            'vol': m['annual_volatility'] * 100,
            'win_rate': m['win_rate'] * 100,
            'def_weeks': m['defensive_weeks'],
            'total_weeks': m['total_weeks'],
            'final': r.nav_series['nav'].iloc[-1],
        }
    finally:
        setattr(cfg, param_name, old)

print("=" * 60)
print("中证500ETF 参数扫描 (默认: vol_w=1.05 invvol=12 step_low=0.15 mom=4 vol_w=20 thresh=0.06)")
print("=" * 60)

scans = [
    ('vol_w', 'vol_w (动量vs波动率)', [0.80, 0.90, 1.00, 1.05, 1.10, 1.15, 1.20]),
    ('inv_vol_window', 'inv_vol_window', [8, 10, 12, 14, 16, 20]),
    ('step_low', 'step_low (防御下限)', [0.10, 0.12, 0.15, 0.18, 0.20]),
    ('mom_window', 'mom_window (动量窗口)', [4, 5, 6, 8, 10]),
    ('vol_window', 'vol_window (波动率窗口)', [10, 13, 16, 20, 25]),
    ('rebalance_threshold', 'rebalance_threshold', [0.03, 0.05, 0.06, 0.07, 0.10]),
]

for param_name, label, values in scans:
    print(f"\n--- {label} ---")
    print(f"  {'值':>6s} {'Sharpe':>8s} {'年化':>7s} {'DD':>7s} {'波动率':>7s} {'胜率':>6s} {'防御':>5s} {'终值':>7s}")
    print(f"  {'-'*55}")

    best_for_param = {'sharpe': -99}

    for v in values:
        res = run(param_name, v)
        marker = ' *' if res['sharpe'] >= max(best_for_param.get('sharpe', -99), -99) else '  '
        print(f"  {v:>6.2f} {res['sharpe']:>8.3f} {res['ann_ret']:>6.2f}% {res['dd']:>6.2f}% "
              f"{res['vol']:>6.2f}% {res['win_rate']:>5.1f}% {res['def_weeks']:>3d}/{res['total_weeks']:<3d} "
              f"{res['final']:>5.2f}x{marker}")
        best_for_param = res if res['sharpe'] > best_for_param.get('sharpe', -99) else best_for_param

    print(f"  --- 最优: {label}={best_for_param['param']} "
          f"Sharpe={best_for_param['sharpe']:.3f} DD={best_for_param['dd']:.2f}%")

print(f"\n{'='*60}")
print("扫描完成")
print(f"{'='*60}")