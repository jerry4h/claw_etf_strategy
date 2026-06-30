#!/usr/bin/env python3
"""
虾池ETF轮动 v3.0 — 新数据超参数优化
======================================

扫描 vol_w (0.70-1.10) × inv_vol_window (4-12)
使用 data/all_etfs_nav_2013_20260626.csv (QFQ 前复权)

用法: python scripts/optimize_v3_0_newdata.py
"""
import sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.backtest import run_backtest
from src.strategy import load_config, StrategyConfig
from dataclasses import replace
import itertools, json
from collections import OrderedDict

CONFIG = PROJECT / 'config/strategy_v3_0_invvol_newdata.yaml'
CSV = 'data/all_etfs_nav_2013_20260626.csv'


def run_one(mom_w, vol_w, invvol_w):
    cfg = load_config(CONFIG)
    cfg.nav_path = CSV
    cfg.end_date = None
    cfg.start_date = None
    cfg.mom_w = mom_w
    cfg.vol_w = vol_w
    cfg.inv_vol_window = invvol_w
    r = run_backtest(cfg)
    m = r.metrics
    return {
        'mom_w': mom_w,
        'vol_w': vol_w,
        'invvol_w': invvol_w,
        'sharpe': m['sharpe_ratio'],
        'ann_ret': m['annual_return'] * 100,
        'dd': m['max_drawdown'] * 100,
        'ann_vol': m['annual_volatility'] * 100,
        'calmar': m.get('calmar_ratio', 0),
        'win_rate': m['win_rate'] * 100,
        'def_weeks': m.get('defensive_weeks', 0),
    }


def main():
    # vol_w = mom4 和 vol20 的相对权重
    # mom_w 固定为 1.0，扫描 vol_w
    vol_w_grid = [round(x, 2) for x in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10]]
    invvol_grid = [4, 6, 8, 10, 12]

    print("=" * 90)
    print(" v3.0 inv-vol8 — 新数据超参数扫描")
    print("=" * 90)
    print(f" 数据: {CSV}")
    print(f" mom_w 固定 = 1.0")
    print(f" vol_w 范围 = {vol_w_grid}")
    print(f" invvol 范围 = {invvol_grid}")
    total = len(vol_w_grid) * len(invvol_grid)
    print(f" 总组合 = {total}")
    print()

    results = []
    for i, (vw, ivw) in enumerate(itertools.product(vol_w_grid, invvol_grid)):
        r = run_one(1.0, vw, ivw)
        results.append(r)
        print(f"  [{i+1}/{total}] vol_w={vw:.2f}  invvol={ivw}  →  Sharpe={r['sharpe']:.3f}  年化={r['ann_ret']:.1f}%  DD={r['dd']:.1f}%")

    # 排名
    print()
    print("=" * 90)
    print(" Top 15 (按 Sharpe 排序)")
    print("=" * 90)
    print(f"  {'vol_w':>6} {'invvol':>6} {'Sharpe':>8} {'年化%':>7} {'DD%':>6} {'波动%':>7} {'Calmar':>7} {'胜率%':>6} {'防御周':>6}")
    print(f"  {'-'*59}")
    top = sorted(results, key=lambda x: x['sharpe'], reverse=True)[:15]
    for r in top:
        print(f"  {r['vol_w']:>6.2f} {r['invvol_w']:>6d} {r['sharpe']:>8.3f} {r['ann_ret']:>6.1f}  {r['dd']:>5.1f}  {r['ann_vol']:>6.1f}  {r['calmar']:>6.2f} {r['win_rate']:>5.1f}  {r['def_weeks']:>5d}")

    # 保存
    import csv
    path = PROJECT / 'output/hyperparam_newdata.csv'
    path.parent.mkdir(exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\n完整结果保存: {path}")


if __name__ == '__main__':
    main()
