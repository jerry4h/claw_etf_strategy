#!/usr/bin/env python3
"""
虾池ETF轮动 v3.0 — 实时调仓计算
=================================
用法:
  python scripts/rebalance_live.py                     # 最新数据 → 下周一调仓
  python scripts/rebalance_live.py --verify            # 全量回测 vs 引擎验证
  python scripts/rebalance_live.py --week 2026-06-26   # 查看特定周
  python scripts/rebalance_live.py --save-state        # 确认调仓并保存状态

策略:
  Layer 1: score = mom4 - 1.10*vol11, top_n=2
  Layer 2: inv-vol10 weights
  Layer 3: nasdaq vol 3-tier [25%, 95%]
  零门控, 零阈值 (除 cap040)

所有因子计算通过 src/factors.py (ddof=0)，杜绝重复实现。
阈值基于上一次实际调仓的仓位（通过状态文件 data/.last_alloc.json 维护），
非上周的理论计算仓位。运行 --save-state 确认调仓后自动更新状态文件。
CSV格式: 日期,纳指ETF,红利低波ETF,中证500ETF,黄金ETF,国债ETF
"""

from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.data_loader import ETFS, OFFENSIVE, DEFENSIVE
from src.utils import compute_sharpe, annualize_return
from src.factors import calculate_momentum, calculate_volatility
from src.strategy import load_config

cfg = load_config(PROJECT / 'config/strategy_v3_0_final.yaml')
MOM_W = cfg.mom_w
VOL_W = cfg.vol_w
TOP_N = cfg.top_n
INV_VOL_W = cfg.inv_vol_window
MOM_WINDOW = cfg.mom_window
VOL_WINDOW = cfg.vol_window
DEF_ALLOC = cfg.def_alloc
STEP_LOW = cfg.step_low
STEP_HIGH = cfg.step_high
MAX_DEF = cfg.max_def
MAX_SINGLE = cfg.max_single_alloc
REBAL_THRESH = cfg.rebalance_threshold
FEE = cfg.fee_rate
RISK_FREE = cfg.risk_free_rate
SCORE_MARGIN = cfg.score_margin

STATE_FILE = PROJECT / 'data' / '.last_alloc.json'

def load_state() -> dict | None:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, KeyError):
            return None
    return None

def save_state(alloc: dict):
    STATE_FILE.write_text(json.dumps(alloc, ensure_ascii=False, indent=2))
    print(f"  已保存调仓状态到 {STATE_FILE}")

def load(csv):
    df = pd.read_csv(csv)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.set_index('日期').sort_index()
    for c in ETFS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df[ETFS].ffill().dropna()

def engine_factors(nav):
    m4 = calculate_momentum(nav, window=MOM_WINDOW)
    v20 = calculate_volatility(nav, window=VOL_WINDOW)
    prices = nav[ETFS].values
    wr_df = pd.DataFrame(
        np.diff(prices, axis=0) / prices[:-1],
        index=nav.index[1:], columns=ETFS
    )
    return wr_df, m4, v20

def score_etf(etf, m4, v20, i):
    mv, vv = m4[etf].iloc[i], v20[etf].iloc[i]
    if pd.isna(mv) or pd.isna(vv):
        return None
    return MOM_W * mv - VOL_W * vv

def defense_ratio(v_nasdaq):
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
        start = max(0, i - 1 - INV_VOL_W + 1)
        end = i
        s = wr[e].iloc[start:end].dropna()
        v = np.std(s.values, ddof=0) * math.sqrt(52) if len(s) >= 3 else 0.20
        iv[e] = 1.0 / max(v, 0.05)
    t = sum(iv.values())
    if t <= 0:
        return {e: 1.0 / max(len(selected), 1) for e in selected}
    return {e: w / t for e, w in iv.items()}

def compute(nav, i, prev_sel=None):
    wr, m4, v20 = engine_factors(nav)
    sc = {e: s for e in OFFENSIVE if (s := score_etf(e, m4, v20, i)) is not None}
    ranked = sorted(sc, key=lambda e: sc[e], reverse=True)

    # --- Score Margin: 防噪声换仓 ---
    if SCORE_MARGIN > 0 and prev_sel is not None and len(ranked) > TOP_N:
        gap = sc[ranked[TOP_N - 1]] - sc[ranked[TOP_N]]
        if gap < SCORE_MARGIN:
            valid_prev = [e for e in prev_sel if e in sc]
            if len(valid_prev) == TOP_N:
                ranked = valid_prev

    sel = ranked[:TOP_N]
    def_r = defense_ratio(v20['纳指ETF'].iloc[i])
    wts = invvol_weights(sel, wr, i)
    alloc = {e: def_r / len(DEFENSIVE) for e in DEFENSIVE}
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

def should_rebalance(curr, prev):
    if not prev:
        return True, 0.0
    max_chg = max(abs(curr.get(e, 0) - prev.get(e, 0))
                  for e in set(curr) | set(prev))
    return max_chg >= REBAL_THRESH, max_chg

def fmt_alloc(alloc, amount=500000):
    lines = []
    for e in ETFS:
        w = alloc.get(e, 0)
        if w > 0.001:
            lines.append(f"  {e:<10s} {w*100:>5.1f}%  ~ {w*amount:>8,.0f}元")
    lines.append(f"  {'合计':<10s} {sum(alloc.values())*100:>5.1f}%")
    return '\n'.join(lines)

def print_scores(sc, m4, v20, idx):
    print(f"\nLayer 1 (买什么)  scoring = mom{MOM_WINDOW} - {VOL_W}*vol{VOL_WINDOW}")
    print(f"  {'ETF':<10s} {'mom{MOM_WINDOW}':>10s} {'vol{VOL_WINDOW}':>10s} {'score':>9s} {'rank':>6s}")
    print(f"  {'-'*45}")
    sel = sorted(sc, key=lambda e: sc[e], reverse=True)[:TOP_N]
    for e in sorted(sc, key=lambda e: sc[e], reverse=True):
        mv = m4[e].iloc[idx]
        vv = v20[e].iloc[idx]
        rk = '<- TOP' if e in sel else ''
        print(f"  {e:<10s} {mv*100:>7.2f}% {vv*100:>7.1f}% {sc[e]:>9.4f} {rk:>6s}")

def print_rebalance(prev_al, curr_al):
    print(f"\n  -- 调仓操作 --")
    print(f"  {'ETF':<10s} {'上次':>7s} {'本周':>7s} {'变化':>7s} {'操作':>8s}")
    print(f"  {'-'*42}")
    for e in curr_al:
        pw = prev_al.get(e, 0) * 100
        cw = curr_al[e] * 100
        dw = cw - pw
        act = '买入' if dw > 0.5 else ('卖出' if dw < -0.5 else '-')
        print(f"  {e:<10s} {pw:>6.1f}% {cw:>6.1f}% {dw:>+6.1f}% {act:>8s}")
    for e in prev_al:
        if e not in curr_al and prev_al[e] > 0.001:
            pw = prev_al[e] * 100
            print(f"  {e:<10s} {pw:>6.1f}% {'0.0':>6}% {-pw:>+6.1f}% {'卖出':>8s}")

def main():
    p = argparse.ArgumentParser(description='虾池ETF轮动 v3.0 实时调仓')
    p.add_argument('csv', nargs='?', default=cfg.nav_path, help='CSV路径')
    p.add_argument('--verify', action='store_true', help='全量回测 vs 引擎验证')
    p.add_argument('--week', type=str, default=None, help='指定日期 YYYY-MM-DD')
    p.add_argument('--amount', type=float, default=500000, help='总资金(元)')
    p.add_argument('--save-state', action='store_true', help='确认调仓并保存状态')
    a = p.parse_args()

    if a.verify:
        from src.backtest import run_backtest
        r = run_backtest(cfg)
        eng = r.metrics
        df = load(PROJECT / a.csv)
        n = len(df); nav, peak = 1.0, 1.0; dd_max = 0.0
        prev_al = {}; prev_sel = None; wrets = []
        for i in range(max(MOM_WINDOW, VOL_WINDOW), n - 1):
            al, sc, _, _, _ = compute(df, i, prev_sel=prev_sel)
            if not al:
                continue
            do, mc = should_rebalance(al, prev_al)
            if not do:
                al = prev_al
            nxt, cur = df.iloc[i + 1], df.iloc[i]
            wr = sum(al.get(e, 0) * (nxt[e] / cur[e] - 1)
                     for e in al if e in df.columns and pd.notna(cur[e]) and cur[e] > 0)
            nav *= (1 + wr)
            peak = max(peak, nav)
            dd = (peak - nav) / peak
            dd_max = max(dd_max, dd)
            wrets.append(wr)
            prev_al = al
            prev_sel = sorted(sc, key=lambda e: sc[e], reverse=True)[:TOP_N]
        scr_s = compute_sharpe(pd.Series(wrets), RISK_FREE)
        scr_r = annualize_return(nav - 1, len(wrets))
        scr_d = dd_max
        print(f"\n{'='*60}")
        print(" 验证: 实时脚本 vs 引擎回测")
        print(f"{'='*60}")
        print(f" 指标     引擎         脚本         差异")
        print(f" Sharpe   {eng['sharpe_ratio']:.4f}       {scr_s:.4f}       {abs(eng['sharpe_ratio']-scr_s):.4f}")
        print(f" 年化     {eng['annual_return']*100:.2f}%      {scr_r*100:.2f}%       {abs(eng['annual_return']-scr_r)*100:.2f}pp")
        print(f" DD       {eng['max_drawdown']*100:.2f}%      {scr_d*100:.2f}%       {abs(eng['max_drawdown']-scr_d)*100:.2f}pp")
        ok = abs(eng['sharpe_ratio'] - scr_s) < 0.02
        print(f"\n {'✅ 通过' if ok else '⚠️ 偏差较大, 需排查'}")
        return

    df = load(PROJECT / a.csv)
    idx = (len(df) - 1 if not a.week
           else df.index.get_indexer([pd.to_datetime(a.week)])[0])
    if idx < max(MOM_WINDOW, VOL_WINDOW):
        print(f"[ERROR] 数据不足. 最早: {df.index[max(MOM_WINDOW, VOL_WINDOW)].date()}")
        return

    # 计算上次选中的进攻ETF（用于score_margin）
    prev_sel = None
    if idx > max(MOM_WINDOW, VOL_WINDOW):
        prev_sc = compute(df, idx - 1)[1]  # sc is index 1
        prev_sel = sorted(prev_sc, key=lambda e: prev_sc[e], reverse=True)[:TOP_N]

    alloc, sc, wr, m4, v20 = compute(df, idx, prev_sel=prev_sel)
    if not alloc:
        print("[ERROR] 无法计算")
        return

    print("=" * 70)
    print(f" 虾池ETF轮动 v3.0  实时调仓")
    print("=" * 70)
    print(f" 数据: {a.csv} | 基准: {df.index[idx].date()} | 调仓: 下周一")
    print(f" 范围: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)}周)")
    print(f" mom_w={MOM_W}  vol_w={VOL_W}  top_n={TOP_N}  invvol{INV_VOL_W}  "
          f"mom_w={MOM_WINDOW}  vol_w={VOL_WINDOW}  "
          f"step_low={STEP_LOW}  thresh={REBAL_THRESH}")

    last_state = load_state()
    if last_state is not None:
        prev_al = last_state
        ref_label = "上次实仓"
    elif idx > max(MOM_WINDOW, VOL_WINDOW):
        prev_al, _, _, _, _ = compute(df, idx - 1)
        ref_label = "上周理论"
    else:
        prev_al = {}
        ref_label = "无"

    if prev_al:
        do_reb, max_chg = should_rebalance(alloc, prev_al)
        print(f"\n调仓阈值 {REBAL_THRESH*100:.0f}%: 参考{ref_label} 最大变化 {max_chg*100:.1f}% "
              f"→ {'调仓!' if do_reb else '不调仓'}")
    else:
        do_reb = True

    print_scores(sc, m4, v20, idx)

    vn = v20['纳指ETF'].iloc[idx]
    dr = defense_ratio(vn)
    print(f"\nLayer 3 (防多少): 纳指vol{VOL_WINDOW}={vn*100:5.1f}% "
          f"→ {'max_def' if vn > STEP_HIGH else '基准' if vn < STEP_LOW else f'线性: {dr*100:.0f}%'}")

    print(f"\nLayer 2 (买多少): inv-vol{INV_VOL_W} 权重")
    print(f"\n-- 下周一持仓 --")
    print(fmt_alloc(alloc, a.amount))

    if prev_al:
        _, cur_mc = should_rebalance(alloc, prev_al)
        if cur_mc >= REBAL_THRESH:
            print_rebalance(prev_al, alloc)

    if a.save_state:
        save_state(alloc)

    print(f"\n{'='*70}")
    if a.save_state:
        print(f" 已保存调仓状态 - 下次运行将基于本次仓位做阈值判断")
    else:
        print(f" 提示: 确认调仓后请加 --save-state 保存仓位状态，下次阈值判断更准")

if __name__ == '__main__':
    main()