#!/usr/bin/env python3
"""
虾池ETF轮动 v3.0 — 周频调仓计算脚本

注意: 所有因子计算通过 src/factors.py 引擎 (ddof=0), 杜绝重复实现。

用法:
  python scripts/weekly_rebalance.py --data data/latest_nav.csv

输入 CSV 格式 (最近 20+ 周净值, 列名=ETF名, 按日期升序):
  日期,纳指ETF,红利低波ETF,沪深300ETF,黄金ETF,国债ETF
  2026-06-08,0.2345,1.5678,...

输出: 本周目标仓位分配表
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.factors import calculate_momentum, calculate_volatility

# ── 策略参数 (v3.0 inv-vol8) ──
MOM_W = 1.0          # 固定
VOL_W = 0.857        # 最优 vol/mom 比
TOP_N = 2            # 选几只进攻 ETF
MOM_WINDOW = 4       # 4 周动量
VOL_WINDOW = 20      # 20 周波动率
INV_VOL_WINDOW = 8   # inv-vol8 窗口
DEF_ALLOC = 0.25     # 基准防御
STEP_LOW = 0.20      # vol 三段式下限
STEP_HIGH = 0.35     # vol 三段式上限
MAX_DEF = 0.95       # 极限防御
HONGLI_RATIO = 0.50  # 防御层中红利低波占比
MAX_SINGLE = 0.40    # 单 ETF 上限

# 标的池
OFFENSIVE = ["纳指ETF", "沪深300ETF", "黄金ETF"]
DEFENSIVE = ["红利低波ETF", "国债ETF"]
ALL_ETFS = OFFENSIVE + DEFENSIVE
NASDAQ = "纳指ETF"


def load_data(csv_path: str) -> pd.DataFrame:
    """加载净值 CSV, 返回 DataFrame (index=日期, columns=ETF名)."""
    df = pd.read_csv(csv_path, parse_dates=["日期"])
    df = df.set_index("日期").sort_index()
    for e in ALL_ETFS:
        if e not in df.columns:
            raise ValueError(f"缺少列: {e}")
    return df


def compute_weekly(nav_df: pd.DataFrame) -> pd.DataFrame:
    """确保数据为周频 (W-MON resample). 已经是周频则原样返回."""
    w = nav_df.resample("W-MON").last().dropna(how="all")
    return w


def compute_momentum(weekly_nav: pd.Series | pd.DataFrame, window: int) -> float:
    """4 周动量 (引擎对齐: ddof=0, diff-based)."""
    if isinstance(weekly_nav, pd.Series):
        frame = weekly_nav.to_frame()
    else:
        frame = weekly_nav
    full = calculate_momentum(frame, window=window)
    return float(full.iloc[-1].iloc[0])


def compute_vol(weekly_nav: pd.Series | pd.DataFrame, window: int) -> float:
    """20 周年化波动率 (引擎对齐: ddof=0)."""
    if isinstance(weekly_nav, pd.Series):
        frame = weekly_nav.to_frame()
    else:
        frame = weekly_nav
    full = calculate_volatility(frame, window=window)
    return float(full.iloc[-1].iloc[0])


def compute_inv_vol8_weights(selected: list[str], weekly_nav: pd.DataFrame) -> dict[str, float]:
    """inv-vol8 权重: 每只 ETF 的 1/vol8 归一化 (ddof=0, 引擎对齐)."""
    prices = weekly_nav.values
    n_weeks = prices.shape[0]
    # 引擎方式: diff returns
    wr = np.diff(prices, axis=0) / prices[:-1]
    rets = pd.DataFrame(wr, index=weekly_nav.index[1:], columns=weekly_nav.columns)
    n = min(INV_VOL_WINDOW, len(rets))
    vols = {}
    for e in selected:
        if e in rets.columns:
            v = np.std(rets[e].iloc[-n:].values, ddof=0) * np.sqrt(52)
            vols[e] = v if not np.isnan(v) and v > 0 else 1.0
    if not vols:
        return {e: 1.0 / len(selected) for e in selected}
    invs = {e: 1.0 / max(v, 0.05) for e, v in vols.items()}
    total = sum(invs.values())
    if total == 0:
        return {e: 1.0 / len(selected) for e in selected}
    return {e: inv / total for e, inv in invs.items()}


def compute_allocation(nav_path: str) -> dict[str, float]:
    """完整计算本周目标仓位."""
    nav_df = load_data(nav_path)
    weekly = compute_weekly(nav_df)

    if len(weekly) < VOL_WINDOW:
        raise ValueError(f"数据不足: 需要至少 {VOL_WINDOW} 周, 当前 {len(weekly)} 周")

    # ── Layer 1: 评分选 TOP2 ──
    scores = {}
    for etf in OFFENSIVE:
        mom = compute_momentum(weekly[etf], MOM_WINDOW)
        vol = compute_vol(weekly[etf], VOL_WINDOW)
        if np.isnan(mom) or np.isnan(vol):
            scores[etf] = -np.inf
        else:
            scores[etf] = MOM_W * mom - VOL_W * vol

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [e for e, s in ranked[:TOP_N] if s > -np.inf]
    if not selected:
        raise RuntimeError("无有效进攻 ETF")
    if len(selected) < TOP_N:
        remaining = [e for e in OFFENSIVE if e not in selected]
        selected += remaining[:TOP_N - len(selected)]

    # ── Layer 2: inv-vol8 权重 ──
    iv_weights = compute_inv_vol8_weights(selected, weekly)

    # ── Layer 3: vol 三段式防御 (引擎 ddof=0) ──
    nasdaq_vol = compute_vol(weekly[NASDAQ], VOL_WINDOW)
    if np.isnan(nasdaq_vol):
        def_ratio = DEF_ALLOC
    elif nasdaq_vol < STEP_LOW:
        def_ratio = DEF_ALLOC
    elif nasdaq_vol > STEP_HIGH:
        def_ratio = MAX_DEF
    else:
        slope = (nasdaq_vol - STEP_LOW) / (STEP_HIGH - STEP_LOW)
        def_ratio = DEF_ALLOC + (MAX_DEF - DEF_ALLOC) * slope

    # ── 组装最终仓位 ──
    alloc = {}
    alloc[DEFENSIVE[0]] = def_ratio * HONGLI_RATIO      # 红利低波
    alloc[DEFENSIVE[1]] = def_ratio * (1 - HONGLI_RATIO)  # 国债
    total_off = 1.0 - def_ratio
    for etf, w in iv_weights.items():
        alloc[etf] = min(total_off * w, MAX_SINGLE)
    total = sum(alloc.values())
    if abs(total - 1.0) > 1e-6:
        for e in alloc:
            alloc[e] /= total
    return alloc


def main():
    parser = argparse.ArgumentParser(description="虾池ETF轮动 v3.0 周频调仓计算")
    parser.add_argument("--data", required=True, help="净值 CSV 路径 (需 20+ 周数据)")
    args = parser.parse_args()

    try:
        alloc = compute_allocation(args.data)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print()
    print("=" * 60)
    print("  虾池ETF轮动 v3.0 — 本周目标仓位")
    print("=" * 60)
    print(f"\n{'ETF':<14} {'仓位':>10} {'金额(万元/50万)':>16}")
    print("-" * 42)
    for etf in ALL_ETFS:
        w = alloc.get(etf, 0.0)
        print(f"  {etf:<12} {w:>10.2%} {w * 50:>16.2f}")
    print("-" * 42)
    total_w = sum(alloc.get(e, 0) for e in ALL_ETFS)
    print(f"  {'合计':<12} {total_w:>10.2%} {total_w * 50:>16.2f}")

    # Layer 明细
    nav_df = load_data(args.data)
    weekly = compute_weekly(nav_df)
    nasdaq_vol = compute_vol(weekly[NASDAQ], VOL_WINDOW)

    # 计算 offensive scorings
    scores_map = {}
    mom_map = {}
    vol_map = {}
    for etf in OFFENSIVE:
        mom = compute_momentum(weekly[etf], MOM_WINDOW)
        vol = compute_vol(weekly[etf], VOL_WINDOW)
        mom_map[etf] = mom
        vol_map[etf] = vol
        scores_map[etf] = mom - VOL_W * vol

    top_etfs = sorted(scores_map, key=lambda e: scores_map[e], reverse=True)[:TOP_N]

    print(f"\n{'='*60}")
    print("  Layer 1 (评分): score = mom4 − 0.857×vol20")
    print(f"{'='*60}")
    for etf in OFFENSIVE:
        flag = " ← TOP" if etf in top_etfs else ""
        print(f"  {etf:<12} mom4={mom_map[etf]:+.4f}  vol20={vol_map[etf]*100:.1f}%  score={scores_map[etf]:+.6f}{flag}")

    print(f"\n  Layer 2 (inv-vol8, ddof=0, 引擎对齐):")
    iv = compute_inv_vol8_weights(top_etfs, weekly)
    for etf, w in iv.items():
        print(f"    {etf:<10} weight={w:.2%}")

    print(f"\n  Layer 3 (vol defense):")
    print(f"    纳指20周年化vol(ddof=0) = {nasdaq_vol*100:.1f}%")
    if np.isnan(nasdaq_vol):
        def_r = DEF_ALLOC
        zone = "N/A"
    elif nasdaq_vol < STEP_LOW:
        def_r = DEF_ALLOC
        zone = "低波区 (基准防御)"
    elif nasdaq_vol > STEP_HIGH:
        def_r = MAX_DEF
        zone = "高波区 (极限防御)"
    else:
        slope = (nasdaq_vol - STEP_LOW) / (STEP_HIGH - STEP_LOW)
        def_r = DEF_ALLOC + (MAX_DEF - DEF_ALLOC) * slope
        zone = "中波区 (线性插值)"
    print(f"    防御比例 = {def_r:.1%} ({zone})")

    print()