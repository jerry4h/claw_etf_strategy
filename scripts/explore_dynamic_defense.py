#!/usr/bin/env python3
"""探索: 根据红利低波回撤动态调整防御层红/国配比"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, pandas as pd

CSV = 'data/all_etfs_nav_2013_20260626.csv'

# 加载数据
nav = pd.read_csv(CSV, parse_dates=['日期']).set_index('日期').sort_index()
etfs = ['纳指ETF','红利低波ETF','沪深300ETF','黄金ETF','国债ETF']
offensive = ['纳指ETF','沪深300ETF','黄金ETF']
defensive = ['红利低波ETF','国债ETF']
for c in etfs:
    nav[c] = pd.to_numeric(nav[c], errors='coerce')
nav = nav[etfs].ffill().dropna()

# 因子
prices = nav.values
w_rets = np.diff(prices, axis=0) / prices[:-1]
n_w = nav.shape[0]

# momentum (4w)
mom = np.full((n_w, len(etfs)), np.nan)
for i in range(4, n_w):
    mom[i] = np.prod(1 + w_rets[i-4:i], axis=0) - 1

# volatility (20w, ddof=0)
vol = np.full((n_w, len(etfs)), np.nan)
for i in range(20, n_w):
    vol[i] = np.std(w_rets[i-20:i], axis=0, ddof=0) * np.sqrt(52)

# 红利低波 52周滚动回撤
hl_price = nav['红利低波ETF'].values
hl_peak = pd.Series(hl_price).rolling(52, min_periods=20).max().values
hl_dd = (hl_peak - hl_price) / hl_peak
hl_dd[np.isnan(hl_dd)] = 0.0

# 固定参数
MOM_W = 1.0
VOL_W = 1.05
TOP_N = 2
INV_VOL_W = 12
MAX_SINGLE = 0.40
DEF_ALLOC = 0.25
STEP_LOW = 0.15
STEP_HIGH = 0.35
MAX_DEF = 0.95

print("=" * 80)
print("防御层动态权重探索 — 红利低波回撤驱动")
print("=" * 80)
print(f"\n基线: hongli_ratio=0.50 (固定50/50)")

# 基线
from src.backtest import run_backtest
from src.strategy import load_config
cfg = load_config('config/strategy_v3_0_final.yaml')
cfg.nav_path = CSV; cfg.end_date = None; cfg.start_date = None
from src.backtest import run_backtest
r0 = run_backtest(cfg)
m0 = r0.metrics
print(f"  基线: Sharpe={m0['sharpe_ratio']:.3f} 年化={m0['annual_return']*100:.1f}% DD={m0['max_drawdown']*100:.1f}%")

# 扫描动态方案
print(f"\n{'='*80}")
print("扫描: 动态 hongli_ratio = clip(min_r + (max_r-min_r)*dd/max_dd, min_r, max_r)")
print(f"{'='*80}")

from dataclasses import replace

results = []
for min_r in [0.10, 0.15, 0.20]:
    for max_r in [0.70, 0.80, 0.90]:
        for max_dd in [0.08, 0.10, 0.12, 0.15]:
            # 手动回测循环
            n = len(nav)
            cv, pv = 1.0, 1.0
            dd_max, prev = 0.0, {}
            rets_list = []
            def_weeks = 0
            
            for i in range(20, n):
                # 动态 hongli_ratio
                dd_val = hl_dd[i] if not np.isnan(hl_dd[i]) else 0.0
                hr = min_r + (max_r - min_r) * min(dd_val / max_dd, 1.0)
                hr = np.clip(hr, min_r, max_r)
                
                # scoring
                sc = {}
                for j, e in enumerate(offensive):
                    ei = etfs.index(e)
                    mv, vv = mom[i, ei], vol[i, ei]
                    if np.isnan(mv) or np.isnan(vv):
                        sc[e] = -np.inf
                    else:
                        sc[e] = MOM_W * mv - VOL_W * vv
                
                ranked = sorted(sc, key=lambda e: sc[e], reverse=True)
                sel = ranked[:TOP_N]
                
                # defense ratio
                vi = etfs.index('纳指ETF')
                vn = vol[i, vi]
                if np.isnan(vn): dr = DEF_ALLOC
                elif vn < STEP_LOW: dr = DEF_ALLOC
                elif vn > STEP_HIGH: dr = MAX_DEF
                else: dr = DEF_ALLOC + (vn - STEP_LOW) / (STEP_HIGH - STEP_LOW) * (MAX_DEF - DEF_ALLOC)
                
                # invvol weights
                iv = {}
                for e in sel:
                    ei = etfs.index(e)
                    s = w_rets[max(0, i - INV_VOL_W):i, ei]
                    s = s[~np.isnan(s)]
                    v_s = np.std(s, ddof=0) * np.sqrt(52) if len(s) >= 3 else 0.20
                    iv[e] = 1.0 / max(v_s, 0.05)
                t = sum(iv.values())
                wts = {e: w / t for e, w in iv.items()} if t > 0 else {e: 1.0 / len(sel) for e in sel}
                
                # allocate
                alloc = {}
                alloc['红利低波ETF'] = dr * hr
                alloc['国债ETF'] = dr * (1 - hr)
                off_t = 1.0 - dr
                for e, w in wts.items():
                    alloc[e] = alloc.get(e, 0) + w * off_t
                for e in alloc:
                    alloc[e] = min(alloc[e], MAX_SINGLE)
                tot = sum(alloc.values())
                if tot < 1.0:
                    def_total = sum(alloc.get(e, 0) for e in defensive)
                    if def_total > 0:
                        excess = 1.0 - tot
                        for e in defensive:
                            alloc[e] += excess * alloc[e] / def_total
                
                # track
                if i < n - 1:
                    nxt = nav.iloc[i+1]
                    cur = nav.iloc[i]
                    wr = sum(alloc.get(e, 0) * (nxt[e] / cur[e] - 1)
                              for e in alloc if cur[e] > 0)
                    cv *= (1 + wr)
                    pv = max(pv, cv)
                    dd = (pv - cv) / pv
                    dd_max = max(dd_max, dd)
                    rets_list.append(wr)
                    if sum(alloc.get(e,0) for e in defensive) > sum(alloc.get(e,0) for e in offensive):
                        def_weeks += 1
            
            sharpe = (np.mean(rets_list) / max(np.std(rets_list, ddof=0), 1e-10)) * np.sqrt(52)
            ann_ret = (cv ** (52 / max(len(rets_list), 1))) - 1
            
            label = f"min={min_r:.2f},max={max_r:.2f},th={max_dd:.2f}"
            results.append({
                'label': label, 'min_r': min_r, 'max_r': max_r, 'max_dd': max_dd,
                'sharpe': sharpe, 'ann_ret': ann_ret*100, 'dd': dd_max*100,
                'def_weeks': def_weeks, 'final_nav': cv
            })
            print(f"  {label:>35s}  Sharpe={sharpe:.3f}  年化={ann_ret*100:.1f}%  DD={dd_max*100:.1f}%  终值={cv:.4f}")

print(f"\n{'='*80}")
print("Top 6 动态方案 vs 基线")
print(f"{'='*80}")
print(f"{'方案':>35s} {'Sharpe':>8} {'年化%':>7} {'DD%':>6} {'终值':>8} {'防御周':>6}")
print('-' * 70)

# 添加基线
results.append({
    'label': 'BASELINE(fixed50/50)',
    'min_r': 0.50, 'max_r': 0.50, 'max_dd': 0,
    'sharpe': m0['sharpe_ratio'],
    'ann_ret': m0['annual_return']*100,
    'dd': m0['max_drawdown']*100,
    'def_weeks': m0['defensive_weeks'],
    'final_nav': r0.nav_series['nav'].iloc[-1]
})

top = sorted(results, key=lambda x: x['sharpe'], reverse=True)[:6]
for r in top:
    flag = ' ← 基线' if 'BASELINE' in r['label'] else ''
    print(f"{r['label']:>35s} {r['sharpe']:>8.3f} {r['ann_ret']:>6.1f}  {r['dd']:>5.1f}  {r['final_nav']:>8.4f} {r['def_weeks']:>5d}{flag}")

print(f"\n{'='*80}")
print("结论:")
best = max(results, key=lambda x: x['sharpe'])
print(f"  最优动态方案: {best['label']}")
print(f"    Sharpe +{(best['sharpe'] - m0['sharpe_ratio'])/m0['sharpe_ratio']*100:+.1f}% vs 基线")
print(f"    年化 {best['ann_ret']:.1f}% vs 基线 {m0['annual_return']*100:.1f}%")
print(f"    DD {best['dd']:.1f}% vs 基线 {m0['max_drawdown']*100:.1f}%")
print(f"    终值 {best['final_nav']:.4f} vs 基线 {r0.nav_series['nav'].iloc[-1]:.4f}")
