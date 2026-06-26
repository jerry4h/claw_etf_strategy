#!/usr/bin/env python3
"""
实时调仓计算脚本 — 虾池ETF轮动 v3.0
=====================================
用法:
  python scripts/rebalance_live.py                      # 最新数据 → 下周一调仓
  python scripts/rebalance_live.py --verify             # 全量回测 vs 引擎
  python scripts/rebalance_live.py --week 2026-06-22    # 查看特定周

策略: v3.0 inv-vol8 (零门控, 零阈值)
  Layer 1 (买什么):  score = mom4 − 0.857×vol20, top_n=2
  Layer 2 (买多少):  inv-vol8 权重
  Layer 3 (防多少):  nasdaq vol 三段式线性插值 [25%, 95%]
"""

from __future__ import annotations
import argparse, math, sys
from pathlib import Path
import numpy as np, pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.data_loader import ETFS, OFFENSIVE, DEFENSIVE
from src.utils import compute_sharpe, annualize_return

# ── v3.0 参数 ──
MOM_W, VOL_W, TOP_N = 1.0, 0.857, 2
INV_VOL_W = 8
DEF_ALLOC, STEP_LOW, STEP_HIGH, MAX_DEF, HONGLI = 0.25, 0.20, 0.35, 0.95, 0.50
MAX_SINGLE = 0.40
REBAL_THRESH, FEE = 0.07, 0.00005
RISK_FREE = 0.025
DEFAULT_CSV = 'data/all_etfs_nav_2013_20260622_scaled.csv'


def load(csv):
    df = pd.read_csv(csv)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.set_index('日期').sort_index()
    for c in ETFS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df[ETFS].ffill().dropna()


def factors(nav):
    wr = nav[ETFS].pct_change()
    m4 = (1 + wr).rolling(4).apply(np.prod, raw=True) - 1
    v20 = wr.rolling(20).std() * math.sqrt(52)
    return wr, m4, v20


def defense(v_nasdaq):
    """vol 三段式: 纯线性插值, 零门控"""
    if pd.isna(v_nasdaq):
        return DEF_ALLOC
    if v_nasdaq < STEP_LOW:
        return DEF_ALLOC
    if v_nasdaq > STEP_HIGH:
        return MAX_DEF
    return DEF_ALLOC + (v_nasdaq - STEP_LOW) / (STEP_HIGH - STEP_LOW) * (MAX_DEF - DEF_ALLOC)


def invvol_weights(selected, wr, i):
    iv = {}
    for e in selected:
        s = wr[e].iloc[max(0, i - INV_VOL_W + 1):i + 1].dropna()
        v = s.std() * math.sqrt(52) if len(s) >= 3 else 0.20
        iv[e] = 1.0 / max(v, 0.05)
    t = sum(iv.values())
    return {e: w / t for e, w in iv.items()} if t > 0 else {e: 1.0 / len(selected) for e in selected}


def compute(nav, i):
    wr, m4, v20 = factors(nav)
    sc = {}
    for e in OFFENSIVE:
        mv = m4[e].iloc[i]; vv = v20[e].iloc[i]
        if pd.notna(mv) and pd.notna(vv):
            sc[e] = MOM_W * mv - VOL_W * vv
    ranked = sorted(sc, key=lambda e: sc[e], reverse=True)
    sel = ranked[:TOP_N]

    def_r = defense(v20['纳指ETF'].iloc[i])
    wts = invvol_weights(sel, wr, i)

    alloc = {}
    for e in DEFENSIVE:
        alloc[e] = def_r / len(DEFENSIVE)
    off_t = 1.0 - def_r
    for e, w in wts.items():
        alloc[e] = alloc.get(e, 0) + w * off_t

    for e in alloc:
        alloc[e] = min(alloc[e], MAX_SINGLE)
    tot = sum(alloc.values())
    if tot < 1.0:
        df_total = sum(alloc.get(e, 0) for e in DEFENSIVE)
        if df_total > 0:
            excess = 1.0 - tot
            for e in DEFENSIVE:
                alloc[e] += excess * alloc[e] / df_total

    return alloc, sc, wr, m4, v20


def fmt_alloc(alloc, amount=500000):
    lines = []
    for e in ETFS:
        w = alloc.get(e, 0)
        if w > 0.001:
            lines.append(f"  {e:<10s} {w*100:>5.1f}%  ≈ {w*amount:>8,.0f}元")
    lines.append(f"  {'合计':<10s} {sum(alloc.values())*100:>5.1f}%")
    return '\n'.join(lines)


def verify():
    from src.backtest import run_backtest
    from src.strategy import load_config

    cfg = load_config(PROJECT / 'config/strategy_v3_0_invvol.yaml')
    cfg.nav_path = DEFAULT_CSV
    r = run_backtest(cfg)

    df = load(PROJECT / DEFAULT_CSV)
    n = len(df)
    nav_val, peak = 1.0, 1.0
    dd_max = 0.0
    prev_al = {}
    weekly_rets = []

    for i in range(20, n - 1):
        al, _, _, _, _ = compute(df, i)
        if not al: continue

        if prev_al:
            mc = max(abs(al.get(e, 0) - prev_al.get(e, 0))
                     for e in set(al) | set(prev_al))
            if mc < REBAL_THRESH:
                al = prev_al

        nxt = df.iloc[i + 1]; cur = df.iloc[i]
        wr = sum(al.get(e, 0) * (nxt[e] / cur[e] - 1)
                 for e in al if e in df.columns and pd.notna(cur[e]) and cur[e] > 0)
        nav_val *= (1 + wr)
        peak = max(peak, nav_val)
        dd = (peak - nav_val) / peak
        dd_max = max(dd_max, dd)
        weekly_rets.append(wr)
        prev_al = al

    eng_s = r.metrics['sharpe_ratio']; eng_r = r.metrics['annual_return']; eng_d = r.metrics['max_drawdown']
    scr_s = compute_sharpe(pd.Series(weekly_rets), RISK_FREE)
    scr_r = annualize_return(nav_val - 1, len(weekly_rets))
    scr_d = dd_max

    print(f"\n{'='*60}")
    print(f" 验证: 实时脚本 vs 引擎回测")
    print(f"{'='*60}")
    print(f" 指标     引擎         脚本         差异")
    print(f" Sharpe   {eng_s:.4f}       {scr_s:.4f}       {abs(eng_s-scr_s):.4f}")
    print(f" 年化     {eng_r*100:.2f}%      {scr_r*100:.2f}%       {abs(eng_r-scr_r)*100:.2f}pp")
    print(f" DD       {eng_d*100:.2f}%      {scr_d*100:.2f}%       {abs(eng_d-scr_d)*100:.2f}pp")
    if abs(eng_s - scr_s) < 0.02:
        print(f"\n ✅ 通过 (Sharpe差 < 0.02)")
    else:
        print(f"\n ⚠️ 偏差较大, 需排查")


def main():
    p = argparse.ArgumentParser(description='虾池ETF轮动 v3.0 实时调仓')
    p.add_argument('csv', nargs='?', default=DEFAULT_CSV, help='CSV路径')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--week', type=str, default=None)
    p.add_argument('--amount', type=float, default=500000)
    a = p.parse_args()

    if a.verify:
        verify(); return

    df = load(PROJECT / a.csv)
    if a.week:
        td = pd.to_datetime(a.week)
        idx = df.index.get_indexer([td])[0]
    else:
        idx = len(df) - 1

    if idx < 20:
        print(f"[ERROR] 数据不足. 最早可用: {df.index[20].date()}"); return

    alloc, sc, wr, m4, v20 = compute(df, idx)
    if not alloc:
        print("[ERROR] 无法计算"); return

    print("=" * 70)
    print(" 虾池ETF轮动 v3.0  实时调仓计算")
    print("=" * 70)
    print(f" 数据: {a.csv}")
    print(f" 基准日: {df.index[idx].date()}  (本周净值)")
    print(f" 调仓日: 下周一")
    print(f" 范围: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)}周)")
    print(f" mom_w=1.0  vol_w={VOL_W}  top_n={TOP_N}  invvol{INV_VOL_W}")

    # 上周持仓对比
    prev_al = {}
    if idx > 20:
        prev_al, _, _, _, _ = compute(df, idx - 1)
        mc = max(abs(alloc.get(e, 0) - prev_al.get(e, 0))
                 for e in set(alloc) | set(prev_al))
        if mc < REBAL_THRESH:
            print(f"\n  调仓阈值{REBAL_THRESH*100:.0f}%: 最大仓位变化{mc*100:.1f}% → 不调仓")
            prev_al = dict(alloc)
        else:
            print(f"\n  调仓阈值{REBAL_THRESH*100:.0f}%: 最大仓位变化{mc*100:.1f}% → 调仓!")

    # Layer 1
    print(f"\n  Layer 1 (买什么): score = mom4 − {VOL_W}×vol20")
    print(f"  {'ETF':<10s} {'mom4':>8s} {'vol20':>8s} {'score':>8s} {'rank':>6s}")
    print(f"  {'-'*42}")
    for e, s in sorted(sc.items(), key=lambda x: x[1], reverse=True):
        mv = m4[e].iloc[idx]; vv = v20[e].iloc[idx]
        rk = 'TOP' if e in sorted(sc, key=lambda x: sc[x], reverse=True)[:TOP_N] else ''
        print(f"  {e:<10s} {mv*100:>7.2f}% {vv*100:>7.1f}% {s:>8.4f} {rk:>6s}")

    # Layer 3
    vn = v20['纳指ETF'].iloc[idx]
    d_val = defense(vn)
    if pd.isna(vn):
        d_tag = "数据缺失 → 基准防御"
    elif vn < STEP_LOW:
        d_tag = f"<{STEP_LOW*100:.0f}% → 基准防御"
    elif vn > STEP_HIGH:
        d_tag = f">{STEP_HIGH*100:.0f}% → 极限防御"
    else:
        d_tag = f"线性插值 [{STEP_LOW*100:.0f}%, {STEP_HIGH*100:.0f}%]"
    print(f"\n  Layer 3 (防多少): 纳指vol20 = {vn*100:5.1f}% → 防御 {d_val*100:5.0f}%  ({d_tag})")

    # Layer 2
    print(f"\n  Layer 2 (买多少): inv-vol{INV_VOL_W} 权重")
    print(f"\n  ── 下周持仓 ──")
    print(fmt_alloc(alloc, a.amount))

    if prev_al and prev_al != alloc:
        print(f"\n  ── 调仓操作 ──")
        print(f"  {'ETF':<10s} {'上周':>7s} {'本周':>7s} {'变化':>7s} {'操作':>8s}")
        print(f"  {'-'*42}")
        for e in alloc:
            pw = prev_al.get(e, 0) * 100; cw = alloc[e] * 100; dw = cw - pw
            act = '买入' if dw > 0.5 else ('卖出' if dw < -0.5 else '—')
            print(f"  {e:<10s} {pw:>6.1f}% {cw:>6.1f}% {dw:>+6.1f}% {act:>8s}")
        for e in prev_al:
            if e not in alloc and prev_al[e] > 0.001:
                pw = prev_al[e] * 100
                print(f"  {e:<10s} {pw:>6.1f}% {'0.0':>6}% {-pw:>+6.1f}% {'卖出':>8s}")

    print(f"\n  {'='*70}")
    print(f"  ✅ 下周一按此比例调仓")


if __name__ == '__main__':
    main()