1|#!/usr/bin/env python3
2|"""
3|虾池ETF轮动 v3.0 — 新数据超参数优化
4|======================================
5|
6|扫描 vol_w (0.70-1.10) × inv_vol_window (4-12)
7|使用 data/all_etfs_nav_2013_20260703.csv (QFQ 前复权)
8|
9|用法: python scripts/optimize_v3_0_newdata.py
10|"""
11|import sys
12|from pathlib import Path
13|PROJECT = Path(__file__).resolve().parent.parent
14|sys.path.insert(0, str(PROJECT))
15|
16|from src.backtest import run_backtest
17|from src.strategy import load_config, StrategyConfig
18|from dataclasses import replace
19|import itertools, json
20|from collections import OrderedDict
21|
22|CONFIG = PROJECT / 'config/strategy_v3_0_invvol_newdata.yaml'
23|
24|
25|def run_one(mom_w, vol_w, invvol_w):
26|    cfg = load_config(CONFIG)
27|    csv_path = cfg.nav_path  # 从 YAML 读取，避免硬编码
28|    cfg.end_date = None
29|    cfg.start_date = None
30|    cfg.mom_w = mom_w
31|    cfg.vol_w = vol_w
32|    cfg.inv_vol_window = invvol_w
33|    r = run_backtest(cfg)
34|    m = r.metrics
35|    return {
36|        'mom_w': mom_w,
37|        'vol_w': vol_w,
38|        'invvol_w': invvol_w,
39|        'sharpe': m['sharpe_ratio'],
40|        'ann_ret': m['annual_return'] * 100,
41|        'dd': m['max_drawdown'] * 100,
42|        'ann_vol': m['annual_volatility'] * 100,
43|        'calmar': m.get('calmar_ratio', 0),
44|        'win_rate': m['win_rate'] * 100,
45|        'def_weeks': m.get('defensive_weeks', 0),
46|    }
47|
48|
49|def main():
50|    # vol_w = mom4 和 vol20 的相对权重
51|    # mom_w 固定为 1.0，扫描 vol_w
52|    vol_w_grid = [round(x, 2) for x in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10]]
53|    invvol_grid = [4, 6, 8, 10, 12]
54|
55|    print("=" * 90)
56|    print(" v3.0 inv-vol8 — 新数据超参数扫描")
57|    print("=" * 90)
58|    print(f" 数据: {load_config(CONFIG).nav_path}")
59|    print(f" mom_w 固定 = 1.0")
60|    print(f" vol_w 范围 = {vol_w_grid}")
61|    print(f" invvol 范围 = {invvol_grid}")
62|    total = len(vol_w_grid) * len(invvol_grid)
63|    print(f" 总组合 = {total}")
64|    print()
65|
66|    results = []
67|    for i, (vw, ivw) in enumerate(itertools.product(vol_w_grid, invvol_grid)):
68|        r = run_one(1.0, vw, ivw)
69|        results.append(r)
70|        print(f"  [{i+1}/{total}] vol_w={vw:.2f}  invvol={ivw}  →  Sharpe={r['sharpe']:.3f}  年化={r['ann_ret']:.1f}%  DD={r['dd']:.1f}%")
71|
72|    # 排名
73|    print()
74|    print("=" * 90)
75|    print(" Top 15 (按 Sharpe 排序)")
76|    print("=" * 90)
77|    print(f"  {'vol_w':>6} {'invvol':>6} {'Sharpe':>8} {'年化%':>7} {'DD%':>6} {'波动%':>7} {'Calmar':>7} {'胜率%':>6} {'防御周':>6}")
78|    print(f"  {'-'*59}")
79|    top = sorted(results, key=lambda x: x['sharpe'], reverse=True)[:15]
80|    for r in top:
81|        print(f"  {r['vol_w']:>6.2f} {r['invvol_w']:>6d} {r['sharpe']:>8.3f} {r['ann_ret']:>6.1f}  {r['dd']:>5.1f}  {r['ann_vol']:>6.1f}  {r['calmar']:>6.2f} {r['win_rate']:>5.1f}  {r['def_weeks']:>5d}")
82|
83|    # 保存
84|    import csv
85|    path = PROJECT / 'output/hyperparam_newdata.csv'
86|    path.parent.mkdir(exist_ok=True)
87|    with open(path, 'w', newline='') as f:
88|        w = csv.DictWriter(f, fieldnames=results[0].keys())
89|        w.writeheader()
90|        w.writerows(results)
91|    print(f"\n完整结果保存: {path}")
92|
93|
94|if __name__ == '__main__':
95|    main()
96|