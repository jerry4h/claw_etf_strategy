#!/usr/bin/env python3
"""
虾池ETF轮动 实时调仓计算 (版本跟随 config.version, 当前默认 v4.5-pvd 生产 config)
=================================
用法:
  python scripts/rebalance_live.py                     # 最新数据 → 下周一调仓
  python scripts/rebalance_live.py --verify            # 全量回测 vs 引擎验证
  python scripts/rebalance_live.py --week 2026-06-26   # 查看特定周
  python scripts/rebalance_live.py --save-state        # 确认调仓并保存状态

策略:
  Layer 1: score = mom_w * mom - vol_w * vol, top_n=2
  Layer 2: inv-vol weights (shared engine_core)
  Layer 3: nasdaq vol 3-tier [def_alloc, max_def]
  Layer 3.5: crisis correlation convergence (shared engine_core)
  DefAlloc: hl_ratio = clip(intercept - coeff*vol_hongli, 0, intercept)

所有因子计算通过 src/factors.py (ddof=0)。
核心决策逻辑通过 src/engine_core.py 共享，与回测引擎保持一致。
阈值基于上一次实际调仓的仓位（通过状态文件 data/.last_alloc.json 维护），
非上周的理论计算仓位。运行 --save-state 确认调仓后自动更新状态文件。
CSV格式: 日期,纳指ETF,红利低波ETF,中证500ETF,黄金ETF,国债ETF
"""

from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
from typing import NamedTuple
import numpy as np, pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.data_loader import ETFS, OFFENSIVE, DEFENSIVE, load_nav_data, load_weekly_volume_from_cache, load_pe_percentile
from src.utils import compute_sharpe, annualize_return
from src.factors import (calculate_momentum, calculate_volatility, calculate_momentum_ewma,
                         calculate_volatility_ewma, calculate_volatility_tapered,
                         compute_pvd_factor, calculate_pe_percentile)
from src.strategy import load_config, calculate_defense_ratio, check_stop_loss
from src.engine_core import (
    compute_crisis_boost, compute_crisis_boost_directed, compute_dynamic_hongli,
    compute_inv_vol_weights, compute_score_margin,
    apply_trend_confirmation, compute_ashare_vol_boost,
    compute_pvd_vol_gates,
)

def _apply_cfg(c):
    """把 config 派生为模块级常量 (module import 时用默认 config 调一次; main() --config 时重调切换)。"""
    global cfg, MOM_W, VOL_W, TOP_N, INV_VOL_W, MOM_WINDOW, VOL_WINDOW, DEF_ALLOC
    global STEP_LOW, STEP_HIGH, MAX_DEF, MAX_SINGLE, REBAL_THRESH, FEE, RISK_FREE
    global SCORE_MARGIN, TREND_CONFIRM, DM_SENS, DM_WIN, HONGLI_RATIO, _START_IDX
    cfg = c
    MOM_W = c.mom_w
    VOL_W = c.vol_w
    TOP_N = c.top_n
    INV_VOL_W = c.inv_vol_window
    MOM_WINDOW = c.mom_window
    VOL_WINDOW = c.vol_window
    DEF_ALLOC = c.def_alloc
    STEP_LOW = c.step_low
    STEP_HIGH = c.step_high
    MAX_DEF = c.max_def
    MAX_SINGLE = c.max_single_alloc
    REBAL_THRESH = c.rebalance_threshold
    FEE = c.fee_rate
    RISK_FREE = c.risk_free_rate
    SCORE_MARGIN = c.score_margin
    TREND_CONFIRM = getattr(c, 'trend_confirm_weeks', 0) or 0
    DM_SENS = getattr(c, 'dynamic_margin_sensitivity', 0.0) or 0.0
    DM_WIN = getattr(c, 'dynamic_margin_window', 4)
    HONGLI_RATIO = c.hongli_ratio
    # 回测/replay 起始预热 (须与 src/backtest.py start_idx 口径一致)
    if c.ewma_factors_enabled:
        _START_IDX = max(c.ewma_mom_halflife * 2, c.ewma_vol_halflife * 2, MOM_WINDOW, VOL_WINDOW)
    elif c.vol_taper_enabled:
        _START_IDX = max(c.vol_taper_window, MOM_WINDOW)   # P0-2: taper 需 vol_taper_window 预热
    else:
        _START_IDX = max(MOM_WINDOW, VOL_WINDOW)


# 默认生产 config = v4.5-pvd (PVD 条件激活, 全门禁通过); main() 可用 --config 切换
# (回退前代 v4.3: config/strategy_v4_3.yaml)
_apply_cfg(load_config(PROJECT / 'config/strategy_v4_5_pvd.yaml'))

STATE_FILE = PROJECT / 'data' / '.last_alloc.json'

def load_state() -> dict | None:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, KeyError):
            return None
    return None

def save_state(alloc: dict):
    """Atomic write: write to temp file then rename to prevent corruption."""
    import tempfile, os
    tmp_fd, tmp_path = tempfile.mkstemp(dir=STATE_FILE.parent, suffix='.tmp')
    try:
        with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
            f.write(json.dumps(alloc, ensure_ascii=False, indent=2))
        os.replace(tmp_path, STATE_FILE)
    except Exception:
        os.unlink(tmp_path)
        raise
    print(f"  已保存调仓状态到 {STATE_FILE}")

def load(csv):
    """Load NAV data using shared load_nav_data() for consistent cleaning."""
    return load_nav_data(csv)

# --- PVD 模块级缓存（engine_factors 首次加载后复用）---
_pvd_cache = {"loaded": False, "pvd_df": None, "pvd_active": False,
              "gate_lo": None, "gate_hi": None}

# --- v4.6 PE 防御调制模块级缓存（同 PVD 模式: 首次加载后复用, 失败降级）---
_pe_def_cache = {"loaded": False, "values": None, "active": False}


def engine_factors(nav):
    if cfg.ewma_factors_enabled:
        m4 = calculate_momentum_ewma(nav, halflife=cfg.ewma_mom_halflife)
        v20 = calculate_volatility_ewma(nav, halflife=cfg.ewma_vol_halflife)
    elif cfg.vol_taper_enabled:
        # P0-1 修: taper 模式与引擎一致 —— 复用 calculate_volatility_tapered, 不自己重写
        m4 = calculate_momentum(nav, window=MOM_WINDOW)
        v20 = calculate_volatility_tapered(nav, window=cfg.vol_taper_window, taper=cfg.vol_taper_len)
    else:
        m4 = calculate_momentum(nav, window=MOM_WINDOW)
        v20 = calculate_volatility(nav, window=VOL_WINDOW)
    prices = nav[ETFS].values
    wr_df = pd.DataFrame(
        np.diff(prices, axis=0) / prices[:-1],
        index=nav.index[1:], columns=ETFS
    )
    # Also return numpy w_rets for shared functions
    wr_np = wr_df.values

    # --- PVD: 加载周频成交额并计算 PVD（pvd_enabled 时首次加载，后续复用）---
    if getattr(cfg, 'pvd_enabled', False) and not _pvd_cache["loaded"]:
        _cache_dir = PROJECT / 'data' / 'experiments' / 'tushare_cache'
        if _cache_dir.exists():
            try:
                weekly_vol = load_weekly_volume_from_cache(_cache_dir, nav.index, list(nav.columns))
                pvd_df = compute_pvd_factor(nav, weekly_vol,
                                           window=cfg.pvd_window, min_periods=cfg.pvd_min_periods)
                _pvd_cache["pvd_df"] = pvd_df
                _pvd_cache["pvd_active"] = True
                # expanding 无前视门限 (共享 engine_core.compute_pvd_vol_gates,
                # 与 backtest.py 同口径: 第 i 周仅用截至 i 的历史 vol)
                gate_lo, gate_hi = compute_pvd_vol_gates(
                    v20['纳指ETF'].values, cfg.pvd_vol_pct_range)
                _pvd_cache["gate_lo"] = gate_lo
                _pvd_cache["gate_hi"] = gate_hi
            except Exception as e:
                import warnings as _w
                _w.warn(f"PVD 降级: 成交额缓存加载失败({e}), pvd_enabled 降级为 False")
                _pvd_cache["pvd_active"] = False
        else:
            import warnings as _w
            _w.warn("PVD 降级: 成交额缓存目录不存在, pvd_enabled 降级为 False")
            _pvd_cache["pvd_active"] = False
        _pvd_cache["loaded"] = True

    # --- v4.6: PE 防御调制序列加载（pe_defense_enabled 时首次加载; 口径同 backtest.py:
    #     5年滚动分位 + shift(1) 防前视 + ffill asof 对齐周频标签）---
    if getattr(cfg, 'pe_defense_enabled', False) and not _pe_def_cache["loaded"]:
        try:
            _pe_path = PROJECT / cfg.pe_path if not Path(cfg.pe_path).is_absolute() else Path(cfg.pe_path)
            _pe_raw = load_pe_percentile(_pe_path)
            _pe_pct = calculate_pe_percentile(_pe_raw, window_years=cfg.pe_window_years)
            _pe_pct = _pe_pct.shift(1).reindex(nav.index, method="ffill")
            _pe_def_cache["values"] = _pe_pct.values[:, 0]
            _pe_def_cache["active"] = True
        except Exception as e:
            import warnings as _w
            _w.warn(f"PE 防御调制降级: PE 数据加载失败({e}), pe_defense 降级为 False")
            _pe_def_cache["active"] = False
        _pe_def_cache["loaded"] = True

    return wr_df, wr_np, m4, v20

def score_etf(etf, m4, v20, i):
    mv, vv = m4[etf].iloc[i], v20[etf].iloc[i]
    if pd.isna(mv) or pd.isna(vv):
        return None
    return MOM_W * mv - VOL_W * vv

def compute_defense_with_crisis(v_nasdaq, wr_np, i, v20=None):
    """Layer 3 + Layer 3.5 defense ratio using shared functions."""
    # Layer 3: vol-based defense ratio (uses shared strategy.calculate_defense_ratio)
    dr = calculate_defense_ratio(v_nasdaq, cfg)

    # Layer 3.5: crisis correlation convergence boost (shared engine_core)
    # off_idx for live script: OFFENSIVE = ['纳指ETF', '中证500ETF', '黄金ETF']
    # ETFS = ['纳指ETF', '红利低波ETF', '中证500ETF', '黄金ETF', '国债ETF']
    off_idx = [ETFS.index(e) for e in OFFENSIVE]
    # v4.6 定向 boost 分级应用 (同 backtest.py 口径, 共享 engine_core 函数):
    #   corr_level > corr_split → 显性危机满额防御; 否则灰区定向降进攻
    if getattr(cfg, 'directed_boost_enabled', False):
        boost, corr_lvl = compute_crisis_boost_directed(wr_np, i, off_idx, cfg)
        if boost > 0:
            if corr_lvl > cfg.directed_boost_corr_split:
                dr = min(dr + boost, 1.0)
            else:
                dr = min(dr + boost * (1.0 - dr), 1.0)
    else:
        boost = compute_crisis_boost(wr_np, i, off_idx, cfg)
        if boost > 0:
            dr = min(dr + boost, 1.0)

    # M3: 中证500 vol 危机加成（与引擎 backtest.py:356 一致；默认关，审计 M-1）
    if v20 is not None:
        vol_values = v20[ETFS].values
        ashare_idx = ETFS.index('中证500ETF')
        ab = compute_ashare_vol_boost(vol_values, i, ashare_idx, cfg)
        if ab > 0:
            dr = min(dr + ab, 1.0)

    # v4.6: PE 估值防御调制（同 backtest.py 口径; 高估值期抬升防御下限）
    # cfg 双重检查: 防 --config 切换后模块级缓存残留 (如 v4.6→v4.3 回退)
    if getattr(cfg, 'pe_defense_enabled', False) and _pe_def_cache["active"] \
            and _pe_def_cache["values"] is not None \
            and i < len(_pe_def_cache["values"]):
        _pe_v = _pe_def_cache["values"][i]
        if not np.isnan(_pe_v) and _pe_v > cfg.pe_defense_pct_threshold:
            dr = min(dr + cfg.pe_defense_delta, cfg.max_def)

    return dr


class ComputeResult(NamedTuple):
    """Named return type for compute() — self-documenting tuple access."""
    alloc: dict            # ETF -> weight allocation
    scores: dict           # ETF -> offensive score
    weekly_rets: pd.DataFrame  # weekly returns DataFrame
    momentum: pd.DataFrame  # momentum factor
    volatility: pd.DataFrame  # volatility factor
    selected: list         # selected offensive ETF names
    pending: frozenset | None  # trend confirmation pending set
    pending_count: int     # pending weeks counter
    gap_history: list      # dynamic margin gap history


def compute(nav, i, prev_sel=None, prev_pending=None, prev_pending_count=0, gap_history=None,
            force_def_floor=None):
    wr_df, wr_np, m4, v20 = engine_factors(nav)
    sc = {e: s for e in OFFENSIVE if (s := score_etf(e, m4, v20, i)) is not None}

    # --- PVD 条件激活 (v4.5): nasdaq vol 在中位且 top-2 momentum gap < 阈值时注入 ---
    if _pvd_cache["pvd_active"]:
        _nv = v20['纳指ETF'].iloc[i] if not pd.isna(v20['纳指ETF'].iloc[i]) else None
        if _nv is not None and _pvd_cache["gate_lo"][i] <= _nv <= _pvd_cache["gate_hi"][i]:
            # 检查 top-2 momentum gap（遍历全部 ETF，与 backtest.py 口径一致）
            _mom_vals = [(m4[e].iloc[i], e) for e in ETFS
                         if not pd.isna(m4[e].iloc[i])]
            if len(_mom_vals) >= 2:
                _mom_vals.sort(key=lambda x: x[0], reverse=True)
                _gap = _mom_vals[0][0] - _mom_vals[1][0]
                if _gap < cfg.pvd_score_gap_threshold:
                    pvd_df = _pvd_cache["pvd_df"]
                    for e in list(sc.keys()):
                        pvd_val = pvd_df[e].iloc[i] if e in pvd_df.columns else np.nan
                        if not pd.isna(pvd_val):
                            sc[e] += cfg.pvd_w * pvd_val

    ranked = sorted(sc, key=lambda e: sc[e], reverse=True)

    # --- Score Margin: 防噪声换仓 (使用共享 engine_core.compute_score_margin) ---
    if gap_history is None:
        gap_history = []
    if prev_sel is not None and len(ranked) > TOP_N:
        gap = sc[ranked[TOP_N - 1]] - sc[ranked[TOP_N]]
        eff_margin, gap_history = compute_score_margin(gap, gap_history, cfg)
        if SCORE_MARGIN > 0 or DM_SENS > 0:
            if gap < eff_margin:
                valid_prev = [e for e in prev_sel if e in sc]
                if len(valid_prev) == TOP_N:
                    ranked = valid_prev

    candidate_sel = ranked[:TOP_N]

    # --- Trend Confirmation: 趋势确认 (使用共享 engine_core.apply_trend_confirmation) ---
    pending = prev_pending
    pending_cnt = prev_pending_count
    if TREND_CONFIRM > 0 and prev_sel is not None:
        # Convert to index-based for shared function, then back to names
        off_idx_map = {e: idx for idx, e in enumerate(OFFENSIVE)}
        cand_idx = [off_idx_map[e] for e in candidate_sel if e in off_idx_map]
        last_idx = [off_idx_map[e] for e in prev_sel if e in off_idx_map]
        result_idx, pending, pending_cnt = apply_trend_confirmation(
            cand_idx, last_idx, pending, pending_cnt, cfg
        )
        candidate_sel = [OFFENSIVE[idx] for idx in result_idx]

    sel = candidate_sel

    # Layer 3 + 3.5: defense ratio with crisis correlation boost
    v_nasdaq = v20['纳指ETF'].iloc[i]
    def_r = compute_defense_with_crisis(v_nasdaq, wr_np, i, v20)
    # 止损期：把防御下限抬到 force_def_floor(=max_def)，仍走完整分配保留进攻端敞口，
    # 与引擎 backtest.py:434 `def_ratio = max(def_ratio, max_def)` 完全一致（审计 M-2）
    if force_def_floor is not None:
        def_r = max(def_r, force_def_floor)

    # Layer 2: inv-vol weights (shared engine_core)
    off_indices = [ETFS.index(e) for e in sel]
    inv_w = compute_inv_vol_weights(wr_np, off_indices, i, INV_VOL_W)
    wts = {e: w for e, w in zip(sel, inv_w)}

    # 动态hongli_ratio (shared engine_core.compute_dynamic_hongli)
    if len(DEFENSIVE) >= 2:
        hl_vol = v20['红利低波ETF'].iloc[i]
        eff_hl_ratio = compute_dynamic_hongli(hl_vol, cfg)
    else:
        eff_hl_ratio = HONGLI_RATIO

    alloc = {DEFENSIVE[0]: def_r * eff_hl_ratio} if len(DEFENSIVE) > 0 else {}
    if len(DEFENSIVE) > 1:
        alloc[DEFENSIVE[1]] = def_r * (1 - eff_hl_ratio)
    off_t = 1.0 - def_r
    for e, w in wts.items():
        alloc[e] = alloc.get(e, 0) + w * off_t
    # Cap only offensive ETFs, overflow -> defense
    overflow = 0.0
    for e in OFFENSIVE:
        if e in alloc and alloc[e] > MAX_SINGLE:
            overflow += alloc[e] - MAX_SINGLE
            alloc[e] = MAX_SINGLE
    if overflow > 0:
        def_total = sum(alloc.get(e, 0) for e in DEFENSIVE)
        if def_total > 0:
            for e in DEFENSIVE:
                alloc[e] += overflow * alloc[e] / def_total
        else:
            for e in DEFENSIVE:
                alloc[e] += overflow / len(DEFENSIVE)
    tot = sum(alloc.values())
    if tot < 1.0:
        df_total = sum(alloc.get(e, 0) for e in DEFENSIVE)
        if df_total > 0:
            excess = 1.0 - tot
            for e in DEFENSIVE:
                alloc[e] += excess * alloc[e] / df_total
    return ComputeResult(alloc, sc, wr_df, m4, v20, sel, pending, pending_cnt, gap_history)

def should_rebalance(curr, prev):
    if not prev:
        return True, 0.0
    max_chg = max(abs(curr.get(e, 0) - prev.get(e, 0))
                  for e in set(curr) | set(prev))
    return max_chg >= REBAL_THRESH, max_chg

def replay_stop_loss_state(df, upto_idx):
    """从数据起点 replay 策略净值 + 单层止损状态机到 upto_idx，判定该周是否应止损。

    与回测引擎 (src/backtest.py 的 check_stop_loss + recovery_weeks 单层止损) 同口径，
    也与 --verify 分支的止损镜像一致。用于让实盘主路径的"下周一持仓"在回撤触及
    stop_loss 阈值时强制进入 max_def 防御，避免实盘推荐遗漏核心风控 (审计 H2)。

    返回: {'should_stop', 'in_recovery', 'nav', 'peak', 'drawdown', 'triggers'}
    """
    start = _START_IDX
    nav, peak = 1.0, 1.0
    prev_al = {}
    prev_sel = None
    prev_pending = None
    prev_pending_count = 0
    gap_hist = []
    in_sl = False
    sl_weeks = 0
    triggers = 0
    for i in range(start, min(upto_idx, len(df) - 1)):
        result = compute(df, i, prev_sel=prev_sel, prev_pending=prev_pending,
                         prev_pending_count=prev_pending_count, gap_history=gap_hist)
        al = result.alloc
        if not al:
            continue
        if not in_sl and check_stop_loss(nav, peak, cfg.stop_loss):
            in_sl = True
            sl_weeks = 0
            triggers += 1
        if in_sl:
            # 止损期走完整分配(force_def_floor=max_def)，保留 1-max_def 进攻敞口，与引擎一致(审计 M-2)
            al = compute(df, i, prev_sel=prev_sel, prev_pending=prev_pending,
                         prev_pending_count=prev_pending_count, gap_history=gap_hist,
                         force_def_floor=MAX_DEF).alloc
            sl_weeks += 1
            if sl_weeks >= cfg.recovery_weeks:
                in_sl = False
        do, _ = should_rebalance(al, prev_al)
        if not do:
            al = prev_al
        nxt, cur = df.iloc[i + 1], df.iloc[i]
        wr = sum(al.get(e, 0) * (nxt[e] / cur[e] - 1)
                 for e in al if e in df.columns and pd.notna(cur[e]) and cur[e] > 0)
        nav *= (1 + wr)
        peak = max(peak, nav)
        prev_al = al
        prev_sel = result.selected
        prev_pending = result.pending
        prev_pending_count = result.pending_count
        gap_hist = result.gap_history
    dd = (peak - nav) / peak if peak > 0 else 0.0
    # upto_idx 周(下周一持仓)的判定：仍在止损期 或 当前回撤触发阈值
    should_stop = bool(in_sl or check_stop_loss(nav, peak, cfg.stop_loss))
    return {'should_stop': should_stop, 'in_recovery': in_sl, 'nav': nav,
            'peak': peak, 'drawdown': dd, 'triggers': triggers}

def fmt_alloc(alloc, amount=500000):
    lines = []
    for e in ETFS:
        w = alloc.get(e, 0)
        if w > 0.001:
            lines.append(f"  {e:<10s} {w*100:>5.1f}%  ~ {w*amount:>8,.0f}元")
    lines.append(f"  {'合计':<10s} {sum(alloc.values())*100:>5.1f}%")
    return '\n'.join(lines)

def print_scores(sc, m4, v20, idx, actual_sel=None):
    print(f"\nLayer 1 (买什么)  scoring = mom{MOM_WINDOW} - {VOL_W}*vol{VOL_WINDOW}")
    mom_label = f"mom{MOM_WINDOW}"
    vol_label = f"vol{VOL_WINDOW}"
    print(f"  {'ETF':<10s} {mom_label:>10s} {vol_label:>10s} {'score':>9s} {'rank':>6s}")
    print(f"  {'-'*45}")
    sel = actual_sel if actual_sel is not None else sorted(sc, key=lambda e: sc[e], reverse=True)[:TOP_N]
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
    p = argparse.ArgumentParser(description=f'{cfg.name} 实时调仓')
    p.add_argument('csv', nargs='?', default=None, help='CSV路径(默认取 config.nav_path)')
    p.add_argument('--config', default='config/strategy_v4_5_pvd.yaml',
                   help='策略配置(默认 v4.5-pvd 生产; 回退 v4.3 用 config/strategy_v4_3.yaml)')
    p.add_argument('--verify', action='store_true', help='全量回测 vs 引擎验证')
    p.add_argument('--week', type=str, default=None, help='指定日期 YYYY-MM-DD')
    p.add_argument('--amount', type=float, default=500000, help='总资金(元)')
    p.add_argument('--save-state', action='store_true', help='确认调仓并保存状态')
    p.add_argument('--premium-check', action='store_true',
                   help='调仓日溢价哨兵: 拉取QDII最新溢价并提示(只提示不自动切换, 任务22)')
    a = p.parse_args()

    # P0-3: 按 --config 切换生产配置 (重派生模块常量 + taper-aware 预热)
    if a.config and a.config != 'config/strategy_v4_5_pvd.yaml':
        _apply_cfg(load_config(PROJECT / a.config))
    csv = a.csv or cfg.nav_path

    if a.verify:
        from src.backtest import run_backtest
        r = run_backtest(cfg)
        eng = r.metrics
        df = load(PROJECT / csv)
        n = len(df); nav, peak = 1.0, 1.0; dd_max = 0.0
        prev_al = {}; prev_sel = None; wrets = []
        prev_pending = None; prev_pending_count = 0; gap_hist = []
        # Stop-loss state (mirrors backtest engine)
        in_stop_loss = False; stop_loss_weeks = 0; stop_loss_count = 0
        for i in range(_START_IDX, n - 1):
            result = compute(df, i, prev_sel=prev_sel, prev_pending=prev_pending,
                             prev_pending_count=prev_pending_count, gap_history=gap_hist)
            al = result.alloc
            if not al:
                continue
            # --- Stop-loss check (P1-4: mirrors engine logic) ---
            if not in_stop_loss and check_stop_loss(nav, peak, cfg.stop_loss):
                in_stop_loss = True
                stop_loss_weeks = 0
                stop_loss_count += 1
            if in_stop_loss:
                # 止损期走完整分配(force_def_floor=max_def)，保留进攻敞口，与引擎/主路径一致(审计 M-2)
                al = compute(df, i, prev_sel=prev_sel, prev_pending=prev_pending,
                             prev_pending_count=prev_pending_count, gap_history=gap_hist,
                             force_def_floor=MAX_DEF).alloc
                stop_loss_weeks += 1
                if stop_loss_weeks >= cfg.recovery_weeks:
                    in_stop_loss = False
            do, mc = should_rebalance(al, prev_al)
            if not do:
                al = prev_al
            nxt, cur = df.iloc[i + 1], df.iloc[i]
            wr = sum(al.get(e, 0) * (nxt[e] / cur[e] - 1)
                     for e in al if e in df.columns and pd.notna(cur[e]) and cur[e] > 0)
            # 审查修复(verify 漏算交易费): 与引擎同口径扣减换手费
            # 首周 prev_al={} → turnover=1.0 全额建仓费, 与引擎 last_alloc=zeros 一致
            turnover = sum(abs(al.get(e, 0) - prev_al.get(e, 0))
                           for e in set(al) | set(prev_al))
            fee_cost = turnover * FEE
            nav *= (1 + wr - fee_cost)
            peak = max(peak, nav)
            dd = (peak - nav) / peak
            dd_max = max(dd_max, dd)
            wrets.append(wr - fee_cost)
            prev_al = al
            prev_sel = result.selected
            prev_pending = result.pending
            prev_pending_count = result.pending_count
            gap_hist = result.gap_history
        scr_s = compute_sharpe(pd.Series(wrets), RISK_FREE)
        scr_r = annualize_return(nav - 1, len(wrets))
        scr_d = dd_max
        print(f"\n{'='*60}")
        print(" 验证: 实时脚本 vs 引擎回测")
        print(f"{'='*60}")
        print(f" 指标       引擎         脚本         差异")
        print(f" Sharpe     {eng['sharpe_ratio']:.4f}       {scr_s:.4f}       {abs(eng['sharpe_ratio']-scr_s):.4f}")
        print(f" 年化       {eng['annual_return']*100:.2f}%      {scr_r*100:.2f}%       {abs(eng['annual_return']-scr_r)*100:.2f}pp")
        print(f" DD         {eng['max_drawdown']*100:.2f}%      {scr_d*100:.2f}%       {abs(eng['max_drawdown']-scr_d)*100:.2f}pp")
        print(f" 止损触发   -            {stop_loss_count}次")
        ok = abs(eng['sharpe_ratio'] - scr_s) < 0.02
        print(f"\n {'✅ 通过' if ok else '⚠️ 偏差较大, 需排查'}")
        return

    df = load(PROJECT / csv)
    idx = (len(df) - 1 if not a.week
           else df.index.get_indexer([pd.to_datetime(a.week)])[0])
    if idx < _START_IDX:
        print(f"[ERROR] 数据不足. 最早: {df.index[_START_IDX].date()}")
        return

    # 计算上次选中的进攻ETF（用于score_margin + trend confirmation）
    prev_sel = None
    prev_gap_hist = []
    prev_pending = None
    prev_pending_count = 0
    if idx > _START_IDX:
        # Build pending state by replaying recent weeks
        lookback = max(TREND_CONFIRM + 2, 3)
        start_replay = _START_IDX   # P0-2: taper-aware 预热起点
        replay_from = max(start_replay, idx - lookback)
        _prev_sel = None
        _prev_pending = None
        _prev_pending_count = 0
        gap_hist_replay = []
        for _ri in range(replay_from, idx):
            _result = compute(
                df, _ri, prev_sel=_prev_sel, prev_pending=_prev_pending, prev_pending_count=_prev_pending_count, gap_history=gap_hist_replay
            )
            _sel = _result.selected
            _prev_pending = _result.pending
            _prev_pending_count = _result.pending_count
            gap_hist_replay = _result.gap_history
            _prev_sel = _sel
        prev_sel = _sel
        prev_pending = _prev_pending
        prev_pending_count = _prev_pending_count
        prev_gap_hist = gap_hist_replay

    _main_result = compute(
        df, idx, prev_sel=prev_sel, prev_pending=prev_pending, prev_pending_count=prev_pending_count, gap_history=prev_gap_hist
    )
    alloc, sc, wr, m4, v20, actual_sel = (
        _main_result.alloc, _main_result.scores, _main_result.weekly_rets,
        _main_result.momentum, _main_result.volatility, _main_result.selected
    )
    if not alloc:
        print("[ERROR] 无法计算")
        return

    # --- 单层止损检查（审计 H2：与回测引擎一致，防止实盘推荐遗漏核心风控）---
    sl_state = replay_stop_loss_state(df, idx)
    stop_loss_active = sl_state['should_stop']
    if stop_loss_active:
        # 止损期走完整分配(force_def_floor=max_def)，保留进攻敞口，与引擎一致(审计 M-2)
        alloc = compute(df, idx, prev_sel=prev_sel, prev_pending=prev_pending,
                        prev_pending_count=prev_pending_count, gap_history=prev_gap_hist,
                        force_def_floor=MAX_DEF).alloc

    _vol_desc = (f"tapered_vol{cfg.vol_taper_window}+{cfg.vol_taper_len}" if cfg.vol_taper_enabled
                 else (f"ewma_vol(hl={cfg.ewma_vol_halflife})" if cfg.ewma_factors_enabled
                       else f"vol{VOL_WINDOW}"))
    print("=" * 70)
    print(f" {cfg.name}  实时调仓")
    print("=" * 70)
    print(f" 数据: {csv} | 基准: {df.index[idx].date()} | 调仓: 下周一")
    print(f" 范围: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)}周)")
    print(f" mom_w={MOM_W}  vol_w={VOL_W}  top_n={TOP_N}  invvol{INV_VOL_W}  "
          f"mom_window={MOM_WINDOW}  {_vol_desc}  "
          f"step_low={STEP_LOW}  thresh={REBAL_THRESH}")

    last_state = load_state()
    if last_state is not None:
        prev_al = last_state
        ref_label = "上次实仓"
    elif idx > _START_IDX:
        _prev_result = compute(df, idx - 1, prev_sel=prev_sel, prev_pending=prev_pending, prev_pending_count=max(0, prev_pending_count - 1))
        prev_al = _prev_result.alloc
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

    print_scores(sc, m4, v20, idx, actual_sel=actual_sel)

    vn = v20['纳指ETF'].iloc[idx]
    dr = compute_defense_with_crisis(vn, wr.values if hasattr(wr, 'values') else wr, idx, v20)
    print(f"\nLayer 3 (防多少): 纳指vol{VOL_WINDOW}={vn*100:5.1f}% "
          f"→ {'max_def' if vn > STEP_HIGH else '基准' if vn < STEP_LOW else f'线性: {dr*100:.0f}%'}")

    print(f"\nLayer 2 (买多少): inv-vol{INV_VOL_W} 权重")
    if stop_loss_active:
        print(f"\n  🛑 止损触发: 当前回撤 {sl_state['drawdown']*100:.1f}% ≥ 阈值 {cfg.stop_loss*100:.0f}% "
              f"→ 持仓强制 {MAX_DEF*100:.0f}% 防御(红利低波+国债)，与回测引擎一致")
    print(f"\n-- 下周一持仓 --")
    print(fmt_alloc(alloc, a.amount))

    if prev_al:
        _, cur_mc = should_rebalance(alloc, prev_al)
        if cur_mc >= REBAL_THRESH:
            print_rebalance(prev_al, alloc)


    # QDII 溢价检查提示
    if alloc.get('纳指ETF', 0) > 0.01:
        print(f"\n  ⚠️  QDII 溢价提醒: 纳指ETF(513100) 目标仓位 {alloc['纳指ETF']*100:.1f}%")
        print(f"     请在交易前检查实时溢价率（天天基金/集思录）:")
        print(f"     - 溢价 < 1%: 正常买入")
        print(f"     - 溢价 1~2%: 谨慎，可分批买入")
        print(f"     - 溢价 > 2%: 建议延迟买入或减少仓位")
        print(f"     溢价买入的收益需覆盖溢价回落风险。")

    # 调仓日溢价哨兵 (--premium-check): 惰性导入, 默认路径/模块导入期零网络依赖
    if a.premium_check:
        try:
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location('premium_sentinel', Path(__file__).parent / 'premium_sentinel.py')
            _ps = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_ps)
            print('\n' + _ps.advise(_ps.fetch_premiums()))
            print(_ps.collapse_report())  # v2 回落防线 (任务28), 自身兜底不抛
        except Exception as _e:
            print(f"\n  ⚠️ 溢价哨兵降级: 获取/判定失败({str(_e)[:80]}), 不影响以上调仓建议")

    if a.save_state:
        save_state(alloc)

    print(f"\n{'='*70}")
    if a.save_state:
        print(f" 已保存调仓状态 - 下次运行将基于本次仓位做阈值判断")
    else:
        print(f" 提示: 确认调仓后请加 --save-state 保存仓位状态，下次阈值判断更准")

if __name__ == '__main__':
    main()
