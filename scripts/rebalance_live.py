#!/usr/bin/env python3
"""
虾池ETF轮动 v3.0 — 实时调仓计算
=================================
用法:
  python scripts/rebalance_live.py                     # 最新数据 → 下周一调仓
  python scripts/rebalance_live.py --verify            # 全量回测 vs 引擎验证
  python scripts/rebalance_live.py --week 2026-06-26   # 查看特定周

策略:
  Layer 1: score = mom4 − 0.857×vol20, top_n=2
  Layer 2: inv-vol8 weights
  Layer 3: nasdaq vol 3-tier [25%, 95%]
  零门控, 零阈值 (除 cap040)

⚠️ 所有因子计算通过 src/factors.py (ddof=0)，杜绝重复实现。
CSV格式: 日期,纳指ETF,红利低波ETF,沪深300ETF,黄金ETF,国债ETF
"""

from __future__ import annotations
import argparse, math, sys
from pathlib import Path
import numpy as np, pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.data_loader import ETFS, OFFENSIVE, DEFENSIVE
from src.utils import compute_sharpe, annualize_return
from src.factors import calculate_momentum, calculate_volatility

# ── v3.0 参数 (与 config/strategy_v3_0_invvol_final.yaml 保持一致) ──
MOM_W, VOL_W, TOP_N            = 1.0, 1.05, 2
INV_VOL_W                      = 12
DEF_ALLOC, STEP_LOW, STEP_HIGH, MAX_DEF = 0.25, 0.15, 0.35, 0.95
HONGLI_RATIO, MAX_SINGLE       = 0.50, 0.40
REBAL_THRESH, FEE, RISK_FREE   = 0.06, 0.00005, 0.025
DEFAULT_CSV                    = 'data/all_etfs_nav_latest.csv'


# ══════════════════════════════════════════════════════════════════════
# 核心计算 (统一使用 src/factors.py 引擎, ddof=0)
# ══════════════════════════════════════════════════════════════════════

def load(csv):
    df = pd.read_csv(csv)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.set_index('日期').sort_index()
    for c in ETFS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df[ETFS].ffill().dropna()


def engine_factors(nav):
    """通过引擎标准因子模块计算, ddof=0.
    返回 (wr_df, m4, v20) — 与 backtest.py 完全对齐.
    """
    m4 = calculate_momentum(nav, window=4)
    v20 = calculate_volatility(nav, window=20)
    # 手动计算 w_rets (引擎格式: diff returns), 用于 invvol
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
    """vol 三段式防御 (与 src/strategy.py::calculate_defense_ratio 完全一致)"""
    if pd.isna(v_nasdaq):
        return DEF_ALLOC
    if v_nasdaq < STEP_LOW:
        return DEF_ALLOC
    if v_nasdaq > STEP_HIGH:
        return MAX_DEF
    return DEF_ALLOC + (v_nasdaq - STEP_LOW) / (STEP_HIGH - STEP_LOW) * (MAX_DEF - DEF_ALLOC)


def invvol_weights(selected, wr, i):
    """inv-vol 权重分配 (ddof=0, 与引擎对齐).
    
    wr: 引擎格式的 diff returns (shape n_weeks-1, columns=ETFS)
    取 wr[i-INV_VOL_W:i] 窗口 (i 是 nav 索引, wr 少一行)
    """
    iv = {}
    for e in selected:
        # wr 少一行 (diff), 用 i-1 对应第 i 周的 return
        start = max(0, i - 1 - INV_VOL_W + 1)
        end = i  # exclusive for wr, so range = wr[start:end]
        s = wr[e].iloc[start:end].dropna()
        v = np.std(s.values, ddof=0) * math.sqrt(52) if len(s) >= 3 else 0.20
        iv[e] = 1.0 / max(v, 0.05)
    t = sum(iv.values())
    if t <= 0:
        return {e: 1.0 / max(len(selected), 1) for e in selected}
    return {e: w / t for e, w in iv.items()}


def compute(nav, i):
    """给定周索引 i, 返回 (alloc, scores, factors)
    
    注意: wr (diff returns) 比 nav 少一行; score_etf 使用 m4/v20 (与 nav 等长).
    """
    wr, m4, v20 = engine_factors(nav)
    # scoring: m4/v20 与 nav 等长, 用 i 直接索引
    sc = {e: s for e in OFFENSIVE if (s := score_etf(e, m4, v20, i)) is not None}
    ranked = sorted(sc, key=lambda e: sc[e], reverse=True)
    sel = ranked[:TOP_N]
    def_r = defense_ratio(v20['纳指ETF'].iloc[i])
    # invvol: wr 少一行, 用 i-1 对应第 i 周的 return 数据
    wts = invvol_weights(sel, wr, i)
    # 组装分配
    alloc = {e: def_r / len(DEFENSIVE) for e in DEFENSIVE}
    off_t = 1.0 - def_r
    for e, w in wts.items():
        alloc[e] = alloc.get(e, 0) + w * off_t
    for e in alloc:
        alloc[e] = min(alloc[e], MAX_SINGLE)
    # 溢出到防御层
    tot = sum(alloc.values())
    if tot < 1.0:
        df_total = sum(alloc.get(e, 0) for e in DEFENSIVE)
        if df_total > 0:
            excess = 1.0 - tot
            for e in DEFENSIVE:
                alloc[e] += excess * alloc[e] / df_total
    return alloc, sc, wr, m4, v20


def should_rebalance(curr, prev, weekly=False):
    """检查是否需要调仓: 单只ETF权重变化最大值 vs 阈值"""
    if not prev:
        return True, 0.0
    max_chg = max(abs(curr.get(e, 0) - prev.get(e, 0))
                  for e in set(curr) | set(prev))
    return max_chg >= REBAL_THRESH, max_chg


# ══════════════════════════════════════════════════════════════════════
# 输出格式化
# ══════════════════════════════════════════════════════════════════════

def fmt_alloc(alloc, amount=500000):
    lines = []
    for e in ETFS:
        w = alloc.get(e, 0)
        if w > 0.001:
            lines.append(f"  {e:<10s} {w*100:>5.1f}%  ≈ {w*amount:>8,.0f}元")
    lines.append(f"  {'合计':<10s} {sum(alloc.values())*100:>5.1f}%")
    return '\n'.join(lines)


def print_scores(sc, m4, v20, idx):
    """打印 Layer 1 scoring 详情"""
    print(f"\nLayer 1 (买什么) — scoring = mom4 − {VOL_W}×vol20")
    print(f"  {'ETF':<10s} {'mom4':>8s} {'vol20':>8s} {'score':>9s} {'rank':>6s}")
    print(f"  {'-'*45}")
    sel = sorted(sc, key=lambda e: sc[e], reverse=True)[:TOP_N]
    for e in sorted(sc, key=lambda e: sc[e], reverse=True):
        mv = m4[e].iloc[idx]
        vv = v20[e].iloc[idx]
        rk = '← TOP' if e in sel else ''
        print(f"  {e:<10s} {mv*100:>7.2f}% {vv*100:>7.1f}% {sc[e]:>9.4f} {rk:>6s}")


def print_rebalance(prev_al, curr_al):
    """打印调仓对比"""
    print(f"\n  ── 调仓操作 ──")
    print(f"  {'ETF':<10s} {'上周':>7s} {'本周':>7s} {'变化':>7s} {'操作':>8s}")
    print(f"  {'-'*42}")
    for e in curr_al:
        pw = prev_al.get(e, 0) * 100
        cw = curr_al[e] * 100
        dw = cw - pw
        act = '买入' if dw > 0.5 else ('卖出' if dw < -0.5 else '—')
        print(f"  {e:<10s} {pw:>6.1f}% {cw:>6.1f}% {dw:>+6.1f}% {act:>8s}")
    for e in prev_al:
        if e not in curr_al and prev_al[e] > 0.001:
            pw = prev_al[e] * 100
            print(f"  {e:<10s} {pw:>6.1f}% {'0.0':>6}% {-pw:>+6.1f}% {'卖出':>8s}")


# ══════════════════════════════════════════════════════════════════════
# 验证
# ══════════════════════════════════════════════════════════════════════

def verify():
    from src.backtest import run_backtest
    from src.strategy import load_config
    cfg = load_config(PROJECT / 'config/strategy_v3_0_invvol.yaml')
    cfg.nav_path = DEFAULT_CSV
    r = run_backtest(cfg)
    eng = r.metrics

    df = load(PROJECT / DEFAULT_CSV)
    n = len(df); nav, peak = 1.0, 1.0; dd_max = 0.0; prev_al = {}; wrets = []
    for i in range(20, n - 1):
        al, _, _, _, _ = compute(df, i)
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


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description='虾池ETF轮动 v3.0 实时调仓')
    p.add_argument('csv', nargs='?', default=DEFAULT_CSV, help='CSV路径')
    p.add_argument('--verify', action='store_true', help='全量验证')
    p.add_argument('--week', type=str, default=None, help='指定日期 YYYY-MM-DD')
    p.add_argument('--amount', type=float, default=500000, help='总资金(元)')
    a = p.parse_args()

    if a.verify:
        verify()
        return

    df = load(PROJECT / a.csv)
    idx = (len(df) - 1 if not a.week
           else df.index.get_indexer([pd.to_datetime(a.week)])[0])
    if idx < 20:
        print(f"[ERROR] 数据不足. 最早: {df.index[20].date()}")
        return

    alloc, sc, wr, m4, v20 = compute(df, idx)
    if not alloc:
        print("[ERROR] 无法计算")
        return

    print("=" * 70)
    print(f" 虾池ETF轮动 v3.0  实时调仓")
    print("=" * 70)
    print(f" 数据: {a.csv} | 基准: {df.index[idx].date()} | 调仓: 下周一")
    print(f" 范围: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)}周)")
    print(f" mom_w=1.0  vol_w={VOL_W}  top_n={TOP_N}  invvol{INV_VOL_W}")

    # 上周对比
    prev_al = {}
    if idx > 20:
        prev_al, _, _, _, _ = compute(df, idx - 1)
        do_reb, max_chg = should_rebalance(alloc, prev_al, weekly=True)
        print(f"\n调仓阈值 {REBAL_THRESH*100:.0f}%: 周最大变化 {max_chg*100:.1f}% "
              f"→ {'⚡ 调仓!' if do_reb else '— 不调仓'}")

    # Layer 1
    print_scores(sc, m4, v20, idx)

    # Layer 3
    vn = v20['纳指ETF'].iloc[idx]
    dr = defense_ratio(vn)
    print(f"\nLayer 3 (防多少): 纳指vol20={vn*100:5.1f}% "
          f"→ {'max_def' if vn > STEP_HIGH else '基准' if vn < STEP_LOW else f'线性: {dr*100:.0f}%'}")

    # Layer 2
    print(f"\nLayer 2 (买多少): inv-vol{INV_VOL_W} 权重")
    print(f"\n── 下周一持仓 ──")
    print(fmt_alloc(alloc, a.amount))

    if prev_al:
        _, cur_mc = should_rebalance(alloc, prev_al, weekly=True)
        if cur_mc >= REBAL_THRESH:
            print_rebalance(prev_al, alloc)

    print(f"\n{'='*70}")
    print(f" 下周一按此比例调仓")


if __name__ == '__main__':
    main()