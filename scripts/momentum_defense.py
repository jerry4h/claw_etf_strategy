#!/usr/bin/env python3
"""防御层动量趋势分配 — 红利上升加仓/下降减仓"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, pandas as pd
from src.factors import calculate_momentum, calculate_volatility
from src.data_loader import ETFS, OFFENSIVE, DEFENSIVE
from src.backtest import run_backtest
from src.strategy import load_config

CSV = 'data/all_etfs_nav_2013_20260626.csv'

# 加载
nav = pd.read_csv(CSV, parse_dates=['日期']).set_index('日期').sort_index()
for c in ETFS: nav[c] = pd.to_numeric(nav[c], errors='coerce')
nav = nav[ETFS].ffill().dropna()

# 因子
m4 = calculate_momentum(nav, window=4)
v20 = calculate_volatility(nav, window=20)
prices = nav.values
w_rets = np.diff(prices, axis=0) / prices[:-1]
n_w = nav.shape[0]

# 红利低波 4w/12w momentum
hl_mom4 = m4['红利低波ETF'].values  # 4w momentum
hl_mom12 = np.full(n_w, np.nan)    # 12w momentum
for i in range(12, n_w):
    hl_mom12[i] = np.prod(1 + w_rets[i-12:i, ETFS.index('红利低波ETF')]) - 1

MOM_W, VOL_W, TOP_N = 1.0, 1.05, 2
INV_VOL_W = 12
MAX_SINGLE = 0.40
DEF_ALLOC, STEP_LOW, STEP_HIGH, MAX_DEF = 0.25, 0.15, 0.35, 0.95

# 基线（引擎）
cfg = load_config('config/strategy_v3_0_final.yaml')
cfg.nav_path = CSV; cfg.end_date = None; cfg.start_date = None
r0 = run_backtest(cfg)
m0 = r0.metrics

print('===== 防御层动量趋势分配 =====')
print(f'基线(引擎含阈值): Sharpe={m0["sharpe_ratio"]:.3f} 年化={m0["annual_return"]*100:.1f}% DD={m0["max_drawdown"]*100:.1f}%')
print(f'\n原则: 红利 momentum ↑ → 加仓红利, momentum ↓ → 加仓国债')
print(f'      hongli_ratio = clip(base + mom_signal * range, min_r, max_r)')

from dataclasses import replace

for mom_name, mom_arr in [('hl_mom4', hl_mom4), ('hl_mom12', hl_mom12)]:
    print(f'\n=== 信号: {mom_name} ===')
    for base_r in [0.40, 0.50]:
        for adj in [0.10, 0.15, 0.20]:
            min_r = base_r - adj
            max_r = base_r + adj
            
            cv, pv = 1.0, 1.0
            dd_max = 0.0
            rets_list = []
            def_weeks = 0
            
            for i in range(20, n_w):
                # 动量信号 → 分配比例偏移
                mom_val = mom_arr[i] if not np.isnan(mom_arr[i]) else 0.0
                hr = base_r + mom_val * adj  # positive mom → more hongli
                hr = np.clip(hr, max(min_r, 0.05), min(max_r, 0.95))
                
                # scoring
                sc = {}
                for e in OFFENSIVE:
                    mv = m4[e].iloc[i]; vv = v20[e].iloc[i]
                    if not (np.isnan(mv) or np.isnan(vv)):
                        sc[e] = MOM_W*mv - VOL_W*vv
                sel = sorted(sc, key=lambda e: sc[e], reverse=True)[:TOP_N]
                
                # defense ratio
                vn = v20['纳指ETF'].iloc[i]
                if np.isnan(vn): dr = DEF_ALLOC
                elif vn < STEP_LOW: dr = DEF_ALLOC
                elif vn > STEP_HIGH: dr = MAX_DEF
                else: dr = DEF_ALLOC + (vn-STEP_LOW)/(STEP_HIGH-STEP_LOW)*(MAX_DEF-DEF_ALLOC)
                
                # invvol
                iv = {}
                for e in sel:
                    s = w_rets[max(0, i-INV_VOL_W):i, ETFS.index(e)]
                    s = s[~np.isnan(s)]
                    v = np.std(s, ddof=0)*np.sqrt(52) if len(s)>=3 else 0.20
                    iv[e] = 1.0/max(v, 0.05)
                t = sum(iv.values())
                wts = {e: w/t for e,w in iv.items()} if t>0 else {e: 1.0/len(sel) for e in sel}
                
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
                    def_t = sum(alloc.get(e,0) for e in DEFENSIVE)
                    if def_t > 0:
                        alloc[e] += (1.0-tot)*alloc[e]/def_t
                
                # track return
                if i < n_w - 1:
                    nxt = nav.iloc[i+1]; cur = nav.iloc[i]
                    wr = sum(alloc.get(e,0)*(nxt[e]/cur[e]-1) for e in alloc if cur[e]>0)
                    cv *= (1+wr); pv = max(pv, cv)
                    dd_max = max(dd_max, (pv-cv)/pv)
                    rets_list.append(wr)
                    def_weeks += 1 if sum(alloc.get(e,0) for e in DEFENSIVE) > 0.5 else 0
            
            sharpe = (np.mean(rets_list)/max(np.std(rets_list,ddof=0),1e-10))*np.sqrt(52)
            ann_ret = cv**(52/max(len(rets_list),1)) - 1
            print(f'  base={base_r:.2f},adj={adj:.2f} [{min_r:.2f},{max_r:.2f}]: Sharpe={sharpe:.3f} 年化={ann_ret*100:.1f}% DD={dd_max*100:.1f}% 终值={cv:.4f} 防周={def_weeks}')

print(f'\n=== 对比基线(手动无阈值, 50/50) ===')
# 手动复现静态50/50
cv, pv, dd_max, rets_list, def_weeks = 1.0, 1.0, 0.0, [], 0
for i in range(20, n_w):
    sc = {}
    for e in OFFENSIVE:
        mv = m4[e].iloc[i]; vv = v20[e].iloc[i]
        if not (np.isnan(mv) or np.isnan(vv)): sc[e] = MOM_W*mv - VOL_W*vv
    sel = sorted(sc, key=lambda e: sc[e], reverse=True)[:TOP_N]
    vn = v20['纳指ETF'].iloc[i]
    if np.isnan(vn): dr = DEF_ALLOC
    elif vn < STEP_LOW: dr = DEF_ALLOC
    elif vn > STEP_HIGH: dr = MAX_DEF
    else: dr = DEF_ALLOC + (vn-STEP_LOW)/(STEP_HIGH-STEP_LOW)*(MAX_DEF-DEF_ALLOC)
    iv = {}
    for e in sel:
        s = w_rets[max(0,i-INV_VOL_W):i, ETFS.index(e)]; s = s[~np.isnan(s)]
        v = np.std(s,ddof=0)*np.sqrt(52) if len(s)>=3 else 0.20
        iv[e] = 1.0/max(v,0.05)
    t = sum(iv.values())
    wts = {e: w/t for e,w in iv.items()} if t>0 else {e: 1.0/len(sel) for e in sel}
    alloc = {}
    alloc['红利低波ETF'] = dr * 0.5
    alloc['国债ETF'] = dr * 0.5
    off_t = 1.0 - dr
    for e,w in wts.items(): alloc[e] = alloc.get(e,0) + w*off_t
    for e in alloc: alloc[e] = min(alloc[e], MAX_SINGLE)
    tot = sum(alloc.values())
    if tot < 1.0:
        for e in DEFENSIVE: alloc[e] += (1.0-tot) * alloc[e] / sum(alloc.get(d,0) for d in DEFENSIVE)
    if i < n_w - 1:
        nxt = nav.iloc[i+1]; cur = nav.iloc[i]
        wr = sum(alloc.get(e,0)*(nxt[e]/cur[e]-1) for e in alloc if cur[e]>0)
        cv *= (1+wr); pv = max(pv, cv)
        dd_max = max(dd_max, (pv-cv)/pv)
        rets_list.append(wr)
        def_weeks += 1 if sum(alloc.get(e,0) for e in DEFENSIVE) > 0.5 else 0
sharpe = (np.mean(rets_list)/max(np.std(rets_list,ddof=0),1e-10))*np.sqrt(52)
ann_ret = cv**(52/max(len(rets_list),1)) - 1
print(f'静态50/50: Sharpe={sharpe:.3f} 年化={ann_ret*100:.1f}% DD={dd_max*100:.1f}% 终值={cv:.4f} 防周={def_weeks}')
