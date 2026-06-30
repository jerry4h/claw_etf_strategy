#!/usr/bin/env python3
"""对比 Equal vs InvVol8 vs InvVol12 — Layer 2 提升量化"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from src.backtest import run_backtest
from src.strategy import load_config

CSV = 'data/all_etfs_nav_2013_20260626.csv'

def run(mom_w, vol_w, iv_en, iv_w):
    cfg = load_config('config/strategy_v3_0_final.yaml')
    cfg.nav_path = CSV; cfg.end_date = None; cfg.start_date = None
    cfg.inv_vol_enabled = iv_en
    cfg.inv_vol_window = iv_w
    return run_backtest(cfg)

r_eq = run(1.0, 1.05, False, 8)
r_iv8 = run(1.0, 1.05, True, 8)
r_iv12 = run(1.0, 1.05, True, 12)

def fmt(m):
    return (m['sharpe_ratio'], m['annual_return']*100, m['max_drawdown']*100,
            m['annual_volatility']*100, m.get('calmar_ratio',0), m['win_rate']*100)

print('='*90)
print('Layer 2 对比: Equal 分配 vs InvVol8 vs InvVol12 (Layer1+Layer3 相同)')
print('='*90)
print('  {:>12} {:>8} {:>8} {:>7} {:>7} {:>7} {:>7} {:>10}'.format(
    '配置', 'Sharpe', '年化%', 'DD%', '波动%', 'Calmar', '胜率%', '累计终值'))
print('  ' + '-'*66)

for lbl, r in [('Equal分配', r_eq), ('InvVol8', r_iv8), ('InvVol12', r_iv12)]:
    s, ar, dd, av, cal, wr = fmt(r.metrics)
    fv = r.nav_series['nav'].iloc[-1]
    print('  {:>12} {:>8.3f} {:>7.2f} {:>6.2f} {:>6.2f} {:>6.2f} {:>6.1f} {:>9.2f}'.format(
        lbl, s, ar, dd, av, cal, wr, fv))

print()
print('超额收益 (vs Equal):')
for lbl, r in [('InvVol8', r_iv8), ('InvVol12', r_iv12)]:
    eq_nav = r_eq.nav_series['nav'].values[-1]
    iv_nav = r.nav_series['nav'].values[-1]
    print('  {}: 累计超额 {:+.2f}%  (Equal={:.2f} → {})'.format(
        lbl, (iv_nav/eq_nav-1)*100, eq_nav, lbl, iv_nav))

print()
print('逐年对比:')
print('  {:>6} {:>12} {:>10} {:>10} {:>10} {:>9} {:>8}'.format(
    '年份', 'Equal年化', 'IV12年化', 'EqualDD%', 'IV12DD%', 'EqualCalm', 'IV12Calm'))
print('  ' + '-'*65)

for yr in sorted(set(r_eq.nav_series.index.year)):
    eq_yr = r_eq.nav_series[r_eq.nav_series.index.year == yr]
    iv_yr = r_iv12.nav_series[r_iv12.nav_series.index.year == yr]
    if len(eq_yr) < 4: continue
    eq_n = eq_yr['nav'].values; iv_n = iv_yr['nav'].values
    e_ret = (eq_n[-1]/eq_n[0]-1)*100; i_ret = (iv_n[-1]/iv_n[0]-1)*100
    e_dd = np.max((np.maximum.accumulate(eq_n)-eq_n)/np.maximum.accumulate(eq_n))*100
    i_dd = np.max((np.maximum.accumulate(iv_n)-iv_n)/np.maximum.accumulate(iv_n))*100
    e_cal = e_ret / e_dd if e_dd > 0 else 0
    i_cal = i_ret / i_dd if i_dd > 0 else 0
    print('  {:>6} {:>11.2f} {:>9.2f} {:>9.2f} {:>8.2f} {:>8.2f} {:>7.2f}'.format(
        yr, e_ret, i_ret, e_dd, i_dd, e_cal, i_cal))
