#!/usr/bin/env python3
"""快速绩效计算 — 策略 vs 等权持有 (5 ETF 各 20%)
输出: 当年收益、近1年收益、当前回撤及起始日

用法:
  python scripts/calc_performance.py              # 全部输出
  python scripts/calc_performance.py --ytd        # 仅当年
  python scripts/calc_performance.py --json       # JSON 格式 (供 web 使用)
"""

import argparse, json, math, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
# Add project venv (has pandas/numpy/matplotlib)
_VENV = PROJECT / '.venv' / 'lib' / 'python3.12' / 'site-packages'
if _VENV.exists():
    sys.path.insert(0, str(_VENV))
sys.path.insert(0, str(PROJECT))

from src.data_loader import ETFS, OFFENSIVE, DEFENSIVE
from src.factors import calculate_momentum, calculate_volatility
from src.strategy import load_config

cfg = load_config(PROJECT / 'config/strategy_v3_0_final.yaml')

# ── helpers ──

def defense_ratio(v):
    if pd.isna(v): return cfg.def_alloc
    if v < cfg.step_low: return cfg.def_alloc
    if v > cfg.step_high: return cfg.max_def
    return cfg.def_alloc + (v - cfg.step_low) / (cfg.step_high - cfg.step_low) * (cfg.max_def - cfg.def_alloc)

def invvol_weights(selected, wr, i):
    iv = {}
    for e in selected:
        s = wr[e].iloc[max(0, i-1-cfg.inv_vol_window+1):i].dropna()
        v = np.std(s.values, ddof=0) * math.sqrt(52) if len(s) >= 3 else 0.20
        iv[e] = 1.0 / max(v, 0.05)
    t = sum(iv.values()) or 1
    return {e: w/t for e, w in iv.items()}

# ── compute both nav series ──

def compute_navs():
    df = pd.read_csv(cfg.nav_path)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.set_index('日期').sort_index().ffill().dropna()
    nav_df = df[ETFS].copy()

    # Strategy NAV
    m4 = calculate_momentum(nav_df, window=4)
    v20 = calculate_volatility(nav_df, window=20)
    prices = nav_df.values
    wr = pd.DataFrame(np.diff(prices, axis=0) / prices[:-1], index=nav_df.index[1:], columns=ETFS)

    strat_nav_arr = np.ones(len(nav_df))
    bench_nav_arr = np.ones(len(nav_df))
    last_alloc = np.array([0.2]*5)

    for i in range(cfg.vol_window, len(nav_df)):
        # ── Strategy ──
        scores = {}
        for j, e in enumerate(ETFS):
            if e not in OFFENSIVE: continue
            mv, vv = m4[e].iloc[i], v20[e].iloc[i]
            if pd.notna(mv) and pd.notna(vv):
                scores[e] = cfg.mom_w * mv - cfg.vol_w * vv
        ranked = sorted(scores, key=lambda k: scores[k], reverse=True)
        sel = ranked[:cfg.top_n]

        alloc = np.zeros(5)
        dr = defense_ratio(v20['纳指ETF'].iloc[i])
        # defense
        alloc[DEFENSIVE.index('红利低波ETF')] = dr * 0.5
        alloc[DEFENSIVE.index('国债ETF')] = dr * 0.5
        # offense
        off_share = 1.0 - dr
        if sel:
            wts = invvol_weights(sel, wr, i)
            for e, w in wts.items():
                j = ETFS.index(e)
                alloc[j] += w * off_share
                alloc[j] = min(alloc[j], cfg.max_single_alloc)
        tot = alloc.sum()
        if tot < 1.0:
            overflow = 1.0 - tot
            for j in [ETFS.index(e) for e in DEFENSIVE]:
                alloc[j] += overflow * alloc[j] / alloc[[ETFS.index(e) for e in DEFENSIVE]].sum()

        # rebalance threshold check
        diff = np.max(np.abs(alloc - last_alloc))
        if diff < cfg.rebalance_threshold:
            alloc = last_alloc.copy()

        fee = np.sum(np.abs(alloc - last_alloc)) * cfg.fee_rate
        last_alloc = alloc.copy()

        strat_wret = sum(alloc[j] * (nav_df.iloc[i, j] / nav_df.iloc[i-1, j] - 1)
                         for j in range(5) if nav_df.iloc[i-1, j] > 0)
        strat_nav_arr[i] = strat_nav_arr[i-1] * (1 + strat_wret - fee)

        # ── Benchmark: equal weight 5 ETFs ──
        bench_wret = sum(0.2 * (nav_df.iloc[i, j] / nav_df.iloc[i-1, j] - 1)
                         for j in range(5) if nav_df.iloc[i-1, j] > 0)
        bench_nav_arr[i] = bench_nav_arr[i-1] * (1 + bench_wret)

    strat_nav = pd.Series(strat_nav_arr, index=nav_df.index)
    bench_nav = pd.Series(bench_nav_arr, index=nav_df.index)
    return strat_nav, bench_nav

# ── metrics ──

def current_year_start(nav_idx):
    """Find first week of current year in the index"""
    cur_year = nav_idx[-1].year
    return nav_idx[nav_idx.year == cur_year][0]

def compute_perf(strat_nav, bench_nav):
    idx = strat_nav.index
    last = idx[-1]
    cur_year = last.year

    # 今年起点
    ytd_start = idx[idx.year == cur_year][0]
    # 1年前起点 (52 weeks ago)
    oney_start = idx[-52] if len(idx) > 52 else idx[0]

    def _metrics(nav, label):
        # 当年收益
        ytd_ret = nav.loc[last] / nav.loc[ytd_start] - 1
        # 近1年收益
        oney_ret = nav.loc[last] / nav.loc[oney_start] - 1
        # 当前回撤
        peak = nav.loc[ytd_start:last].cummax()
        dd = (peak - nav.loc[ytd_start:last]) / peak
        current_dd = dd.iloc[-1]
        # 回撤起始日
        dd_start = None
        if current_dd > 0.001:
            # Find the peak before current drawdown
            peak_idx = dd.idxmax()  # when dd was 0 = peak date
            # Actually the most recent peak
            recent_peak = nav.loc[ytd_start:last].cummax()
            # When was the peak attained
            peak_dates = recent_peak[recent_peak == recent_peak.iloc[-1]].index
            dd_start = peak_dates[0]
        return ytd_ret, oney_ret, current_dd, dd_start

    s_ytd, s_1y, s_dd, s_dd_start = _metrics(strat_nav, '策略')
    b_ytd, b_1y, b_dd, b_dd_start = _metrics(bench_nav, '等权')

    return {
        'strategy': {'ytd': s_ytd, '1y': s_1y, 'dd': s_dd, 'dd_start': s_dd_start},
        'benchmark': {'ytd': b_ytd, '1y': b_1y, 'dd': b_dd, 'dd_start': b_dd_start},
        'last_date': last.strftime('%Y-%m-%d'),
        'ytd_start': ytd_start.strftime('%Y-%m-%d'),
        'oney_start': oney_start.strftime('%Y-%m-%d'),
    }

def fmt_table(perf):
    lines = []
    lines.append(f"📊 绩效对比: 策略 vs 等权持有 (5 ETF 各 20%)")
    lines.append(f"  数据截至 {perf['last_date']}")
    lines.append(f"{'─'*50}")
    lines.append(f"  {'':>22s} {'策略':>10s} {'等权持有':>10s}")
    lines.append(f"{'─'*50}")
    s, b = perf['strategy'], perf['benchmark']
    lines.append(f"  今年收益(2026)      {s['ytd']*100:>+8.2f}% {b['ytd']*100:>+9.2f}%")
    lines.append(f"  近1年收益           {s['1y']*100:>+8.2f}% {b['1y']*100:>+9.2f}%")
    lines.append(f"  当前回撤            {s['dd']*100:>8.2f}% {b['dd']*100:>9.2f}%")
    if s['dd_start']:
        lines.append(f"  回撤起始            {str(s['dd_start'])[:10]:>8s} {str(b['dd_start'])[:10]:>9s}")
    else:
        lines.append(f"  回撤起始            {'-':>8s} {'-':>9s}")
    lines.append(f"{'─'*50}")
    return '\n'.join(lines)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='快速绩效对比')
    p.add_argument('--json', action='store_true', help='JSON 输出')
    p.add_argument('--ytd', action='store_true', help='仅当年')
    args = p.parse_args()

    strat_nav, bench_nav = compute_navs()
    perf = compute_perf(strat_nav, bench_nav)

    if args.json:
        print(json.dumps(perf, ensure_ascii=False, indent=2, default=str))
    else:
        print(fmt_table(perf))