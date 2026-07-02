#!/usr/bin/env python3
"""红利低波动态分配 — 基于回撤调整防守层内权重

一次性预计算全部调仓方案，再对比静/动态终值"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, pandas as pd
from src.factors import calculate_momentum, calculate_volatility
from src.data_loader import ETFS, OFFENSIVE, DEFENSIVE

CSV = Path(__file__).resolve().parent.parent / 'data/all_etfs_nav_2013_20260626.csv'

# ── 参数 ──
MOM_W, VOL_W, TOP_N = 1.0, 1.05, 2
INV_VOL = 12
DEF_ALLOC, STEP_LOW, STEP_HIGH, MAX_DEF = 0.25, 0.15, 0.35, 0.95
MAX_SINGLE = 0.40

def precompute(nav):
    """预计算所有周的调仓方案"""
    n = len(nav)
    m4 = calculate_momentum(nav, window=4)
    v20 = calculate_volatility(nav, window=20)
    prices = nav.values
    wr = pd.DataFrame(np.diff(prices, axis=0)/prices[:-1], index=nav.index[1:], columns=ETFS)

    allocs = []
    for i in range(20, n-1):
        sc = {}
        for e in OFFENSIVE:
            mv = m4[e].iloc[i]; vv = v20[e].iloc[i]
            if not (pd.isna(mv) or pd.isna(vv)):
                sc[e] = MOM_W*mv - VOL_W*vv
        sel = sorted(sc, key=lambda e: sc[e], reverse=True)[:TOP_N]

        vn = v20['纳指ETF'].iloc[i]
        if pd.isna(vn): dr = DEF_ALLOC
        elif vn < STEP_LOW: dr = DEF_ALLOC
        elif vn > STEP_HIGH: dr = MAX_DEF
        else: dr = DEF_ALLOC + (vn-STEP_LOW)/(STEP_HIGH-STEP_LOW)*(MAX_DEF-DEF_ALLOC)

        iv = {}
        for e in sel:
            s = wr[e].iloc[max(0, i-1-INV_VOL+1):i].dropna()
            v = np.std(s.values, ddof=0)*np.sqrt(52) if len(s)>=3 else 0.20
            iv[e] = 1.0/max(v, 0.05)
        t = sum(iv.values())
        wts = {e: w/t for e,w in iv.items()} if t>0 else {e: 1.0/len(sel) for e in sel}

        al = {e: dr/len(DEFENSIVE) for e in DEFENSIVE}
        off_t = 1.0 - dr
        for e,w in wts.items():
            al[e] = al.get(e,0) + w*off_t
        for e in al:
            al[e] = min(al[e], MAX_SINGLE)
        tot = sum(al.values())
        if tot < 1.0:
            def_t = sum(al.get(e,0) for e in DEFENSIVE)
            if def_t > 0:
                for e in DEFENSIVE:
                    al[e] += (1.0-tot)*al[e]/def_t
        allocs.append(al)
    return allocs

def run(allocs, nav, dynamic=False, dd_low=0.05, dd_high=0.15, min_r=0.50, max_r=0.80):
    h_px = nav['红利低波ETF'].values[20:]  # align with allocs
    peak = np.maximum.accumulate(h_px)
    dd = (peak - h_px) / np.maximum(peak, 1e-10)

    n_val = 1.0; p_n = 1.0; mx_d = 0.0; w_r = []
    for t, al in enumerate(allocs):
        idx = t + 20
        cur = nav.iloc[idx]; nxt = nav.iloc[idx+1]

        if dynamic:
            hr = min_r
            if dd[t] > dd_low:
                if dd[t] >= dd_high:
                    hr = max_r
                else:
                    hr = min_r + (dd[t]-dd_low)/(dd_high-dd_low)*(max_r-min_r)
            td = al.get('红利低波ETF',0) + al.get('国债ETF',0)
            if td > 0:
                al = al.copy()
                al['红利低波ETF'] = td * hr
                al['国债ETF'] = td * (1 - hr)

        ret = sum(al.get(e,0)*(nxt[e]/cur[e]-1) for e in al if cur[e]>0)
        n_val *= (1+ret); p_n = max(p_n, n_val)
        mx_d = max(mx_d, (p_n-n_val)/p_n)
        w_r.append(ret)

    s = np.mean(w_r)/max(np.std(w_r,ddof=0),1e-10)*np.sqrt(52)
    a = n_val**(52/max(len(w_r),1))-1
    d_w = sum(1 for al in allocs if sum(al.get(e,0) for e in DEFENSIVE)>0.50)/len(allocs)*100
    return s, a, mx_d, n_val, d_w

# ── 加载 + 预计算 ──
print('加载数据...'); df = pd.read_csv(CSV)
df['日期'] = pd.to_datetime(df['日期']); df = df.set_index('日期').sort_index()
for c in ETFS: df[c] = pd.to_numeric(df[c], errors='coerce')
nav = df[ETFS].ffill().dropna()
print(f'预计算调仓 ({len(nav)-21} 周)...')
allocs = precompute(nav)
print(f'完成: {len(allocs)} 个调仓方案')

# ── 对比 ──
s_s, a_s, d_s, n_s, dw_s = run(allocs, nav, dynamic=False)
s_d, a_d, d_d, n_d, dw_d = run(allocs, nav, dynamic=True,
                                dd_low=0.05, dd_high=0.15, min_r=0.50, max_r=0.80)

print()
print('===== 红利低波动态分配 =====')
print('参数: DD_LOW=0.05, DD_HIGH=0.15, MIN_R=0.50, MAX_R=0.80')
print()
print(f'{"指标":<10}  {"静态50/50":>10}  {"动态":>10}  {"差异":>10}')
print('-'*44)
print(f'{"Sharpe":<10}  {s_s:>10.3f}  {s_d:>10.3f}  {s_d-s_s:>+10.3f}')
print(f'{"年化%":<10}  {a_s*100:>10.2f}  {a_d*100:>10.2f}  {(a_d-a_s)*100:>+10.2f}')
print(f'{"DD%":<10}  {d_s*100:>10.2f}  {d_d*100:>10.2f}  {(d_d-d_s)*100:>+10.2f}')
print(f'{"终值":<10}  {n_s:>10.4f}  {n_d:>10.4f}  {n_d-n_s:>+10.4f}')
print(f'{"防周%":<10}  {dw_s:>10.1f}  {dw_d:>10.1f}  {dw_d-dw_s:>+10.1f}')

# ── 扫描 ──
print()
print('=== DD_LOW 扫描 ===')
print(f'{"DD_LOW":>7}  {"Sharpe":>8}  {"年化%":>7}  {"DD%":>6}')
for dl in [0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.12]:
    s,a,d,n,_ = run(allocs, nav, True, dd_low=dl)
    print(f'{dl:>7.2f}  {s:>8.3f}  {a*100:>6.2f}  {d*100:>5.2f}')

print()
print('=== MAX_R 扫描 ===')
print(f'{"MAX_R":>7}  {"Sharpe":>8}  {"年化%":>7}  {"DD%":>6}')
for mr in [0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]:
    s,a,d,n,_ = run(allocs, nav, True, max_r=mr)
    print(f'{mr:>7.2f}  {s:>8.3f}  {a*100:>6.2f}  {d*100:>5.2f}')

print()
print('=== DD_HIGH 扫描 ===')
print(f'{"DD_HIGH":>7}  {"Sharpe":>8}  {"年化%":>7}  {"DD%":>6}')
for dh in [0.10,0.12,0.15,0.18,0.20,0.25]:
    s,a,d,n,_ = run(allocs, nav, True, dd_high=dh)
    print(f'{dh:>7.2f}  {s:>8.3f}  {a*100:>6.2f}  {d*100:>5.2f}')