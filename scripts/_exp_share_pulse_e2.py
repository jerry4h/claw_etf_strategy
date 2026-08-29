#!/usr/bin/env python3
"""ETF 份额脉冲因子 —— E0 + E1'(事件研究) + E2 双轨集成回测.

用户诉求(原话要点):
  "不仅是理论分析相关性, 而且要把当前份额的增长作为一个因子集成到我们的策略中尝试。
   最终给出结论, 而不是只做了相关性分析就给出不可行的判断。"
  "份额这样的因子可能长期都不生效...但有那么一些节点它会突然给信号...更适合作为
   一个阈值、一个观测点, 提供脉冲信号。有少部分的周报会出现异常波动。"

为什么必须重做(而非沿用既有结论):
  份额因子此前被两次判否, 但**两次都止步于相关性, E2 回测从未真正运行**:
    - exp_share_flow.json        gate.verdict = NO-GO        (横截面 rank IC)
    - exp_etf_flow_style.json    proceed_to_e2 = False       (降级为观察工具)
    - exp_etf_flow_style_e2.json 读到 proceed_to_e2=False -> **拒绝执行**
  既有 E1 门禁本身的反 p-hacking 逻辑是正当的, 但它测的是**横截面连续 IC**:
  隐含"因子每周都线性有效"的假设。稀疏脉冲若只在 5% 极端周携带信息、其余 95%
  为噪声, 连续 IC 必被稀释到不显著 —— 此时"不过门禁"反映的是**框架盲区**而非
  因子失败。故本脚本换范式(事件研究 + 真回测)并**无条件实跑 E2**。

怎么在实跑 E2 的同时仍然防 p-hacking(不能两头都松):
  1. 参数网格与判定标准**先定后测**, 写死在常量里, 不因结果调整;
  2. 报告**全网格**, 不挑最优;
  3. 要求**稳健区域**(邻域一致 + 多数格子改善), 孤立最优点判为参数噪声;
  4. 预注册主假说只有一条(理论驱动), 其余组合标注为探索性且不做校正后宣称。

数据口径全部复用 _exp_share_flow_study.py 的既有实测结论, 不重新发明:
  - 源: data/national_team/fund_share/{code}.csv 的 fd_share
  - **零滞后**: 实测 tushare fund_share 周五当天即可得(初版"滞后 7 天"是误判并已
    更正), 故 lag_steps=0, 周五信号周一执行, 与策略 rebalance 时序一致;
  - 拆分/折算做后向复权(精确整数倍跳变指纹), 不引入前视;
  - 健康子集由 E0 实测决定, 不硬编码。

用法:
    python scripts/_exp_share_pulse_e2.py
    python scripts/_exp_share_pulse_e2.py --render-only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# 复用既有实测口径: 份额加载/复权/对齐(sf) 与脉冲机件/事件研究(fh)。
# 两者的门禁与统计口径逐位一致, 结论可与股指期货线横向比较。
sf = _load("sf", "scripts/_exp_share_flow_study.py")
fh = _load("fh", "scripts/_exp_futures_holding_study.py")

OUT_DIR = PROJECT / "output" / "experiments"
OUT_MD = OUT_DIR / "exp_share_pulse_e2.md"
OUT_JSON = OUT_DIR / "exp_share_pulse_e2.json"
OUT_PNG = OUT_DIR / "share_pulse_e2.png"
CFG_PATH = PROJECT / "config" / "strategy_v4_6.yaml"

# ---- 先定后测的常量(不因结果调整) ----
LAG_STEPS = 0          # 0=周五当日份额, 已实测零滞后
ROLL_WIN = 104         # 滚动分位窗口(周). 份额规模长期增长 -> 非平稳, 禁用 expanding
MIN_HIST = 104
MIN_EVENTS = 15        # 份额脉冲比期货稀疏, 下限略放宽但仍需可统计
MIN_YEAR_SPAN = 4      # 触发年份跳数下限
Q_PULSE = 0.95         # 极端分位
TRACK_A_W = (0.05, 0.10, 0.15)      # tiebreaker 权重(上限 0.15, 沿用既有设定)
TRACK_B_DELTA = (0.05, 0.10, 0.15)  # 防御比例增减幅度
HOLD_GRID = (1, 2, 4)               # 脉冲持续周数
SCORE_GAP_THRESHOLD = 0.02          # 仿 PVD 条件激活, 与既有 E2 脚本一致
CAGR_TOL_PP = -0.5                  # CAGR 恶化容忍上限
SEED = 20260829

# ---- E3 验证常量(同样先定后测) ----
E3_SPLIT = "2019-01-01"             # 分期切点: 2019 年是策略池扩张与风格切换分界
E3_FEE_BP = (0.5, 2.0, 5.0, 10.0)   # 单边费率(bp), 0.5bp 为生产基准
E3_PLACEBO_N = 50                   # 安慰剂重排次数(每次一整轮回测)
E3_BOOT_N = 500                     # 成对块 bootstrap 次数
E3_BLOCK_W = 8                      # 块长(周) >= 2x max(hold), 保留重叠窗口自相关
E3_PLACEBO_MAX_PCT = 10.0           # 安慰剂中优于真信号的比例上限


# --------------------------------------------------------------------------
# E0: 份额数据与脉冲信号
# --------------------------------------------------------------------------
def load_share_weekly(weekly_index) -> tuple[pd.DataFrame, dict, list]:
    """份额周频面板 + E0 体检 + 健康子集.

    健康判据直接采用既有 sf.e0_quality 的**日频**指标, 而非自制周频口径:
      (1) 2020 年后日频同值停滞率 < 50%
      (2) 无 >= 120 天的连续冻结段
    周频重采样会把停滞稀释(纳指 55.8% -> 43.1%), 用周频阈值会错放坏数据。
    实测结果与先前研究的 HEALTHY 名单一致(纳指 55.8%/国债 51.3% 超阈被剔),
    且纳指另有 187 天冻结段(2025-09-11~2026-06-25)延续至近期, 对实盘信号致命。

    注: 不用覆盖率作判据 —— 红利低波覆盖 57.8% 是**晚上市**(首值 2018-12-21)
    造成的, 上市后空洞率为 0%, 用覆盖率剔它是误判。
    """
    raw = sf.load_share_raw()
    nav_last = pd.Timestamp(weekly_index.max())
    e0 = sf.e0_quality(raw, nav_last)
    adj = sf.adjust_splits(raw, e0)
    panel = sf.build_share_panel(adj, weekly_index, LAG_STEPS)

    health, healthy = {}, []
    for name in panel.columns:
        rec = e0.get(name, {})
        # e0_quality 里该键是小数(如 0.5578)而非百分数, 乘 100 统一口径
        stag_d = float(rec.get("unchanged_rate_2020plus", 1.0)) * 100.0
        runs = rec.get("stagnation_runs", []) or []
        longest = max([int(r["n_days"]) for r in runs], default=0)
        s = panel[name]
        fv = s.first_valid_index()
        post_gap = (float(s[s.index >= fv].isna().mean())
                    if fv is not None else 1.0)
        ok = bool(stag_d < 50.0 and longest < 120)
        health[name] = {
            "first_valid": str(fv.date()) if fv is not None else None,
            "post_listing_gap_pct": round(post_gap * 100, 2),
            "daily_stagnation_2020plus_pct": round(stag_d, 2),
            "longest_frozen_days": longest,
            "n_exact_ratio_jumps": len(rec.get("exact_ratio_jumps", []) or []),
            "healthy": ok,
            "reject_reason": (None if ok else
                              (f"停滞率 {stag_d:.1f}%>=50%" if stag_d >= 50
                               else f"冻结段 {longest}天>=120")),
        }
        if ok:
            healthy.append(name)
    return panel, health, healthy


def build_pulse_signals(panel: pd.DataFrame, nav: pd.DataFrame,
                        healthy: list) -> dict:
    """构造两类份额脉冲信号.

    A) 个体信号 per_etf_growth: 每只健康 ETF 自身的 1 周份额增长率
       -> 用于轨 A 横截面 tiebreaker(份额流入的那只加分)
    B) 聚合信号 pool_net_flow: 健康池加总**净申赎额**占总规模比
       -> 用于轨 B 市场级防御(整体资金撤离=风险信号)

    净申赎额口径遵循项目既有 ADR: NAV_{t-1} x (S_t - S_{t-1}), 只取申赎贡献,
    完全剔除价格效应 —— 否则"涨出来的规模"会被误读成"买出来的份额"。
    """
    ph = panel[healthy]
    growth = ph / ph.shift(1) - 1.0

    nv = nav.reindex(ph.index)[healthy]
    d_share = ph.diff()
    net_flow = (nv.shift(1) * d_share).sum(axis=1, min_count=1)
    aum = (nv * ph).sum(axis=1, min_count=1)
    pool_net_flow = net_flow / aum.shift(1)

    return {
        "per_etf_growth": growth,
        "pool_net_flow": pool_net_flow,
        "pool_net_flow_4w": pool_net_flow.rolling(4, min_periods=4).sum(),
    }


def pulse_and_coverage(s: pd.Series, side: str) -> tuple[pd.Series, dict]:
    """脉冲标记 + 触发覆盖度. 复用 fh 的实现(rolling 分位, shift(1) 无前视)."""
    q = Q_PULSE if side == "high" else round(1.0 - Q_PULSE, 2)
    flag = fh.pulse_flags(s, q, side, min_hist=MIN_HIST, mode="roll")
    cov = fh.trigger_coverage(flag)
    cov["coverage_ok"] = bool(cov["year_span"] >= MIN_YEAR_SPAN
                              and cov["n"] >= MIN_EVENTS)
    return flag, cov


def hold_expand(flag: pd.Series, hold: int) -> pd.Series:
    """把脉冲展开成持续 hold 周的激活窗口(含触发周本身)."""
    a = flag.to_numpy().astype(bool)
    out = np.zeros(len(a), dtype=bool)
    for t in np.where(a)[0]:
        out[t:min(t + hold, len(a))] = True
    return pd.Series(out, index=flag.index)


# --------------------------------------------------------------------------
# E2 通用: 基线与周日期序列
# --------------------------------------------------------------------------
def week_dates() -> list:
    """回测周循环的日期序列, 用于给不接收周下标的注入点做计数器对齐."""
    from _exp_huijin_backstop_signal_study import _backtest_week_dates
    return _backtest_week_dates(CFG_PATH)


def load_cfg(fee_rate: float | None = None):
    """配置加载. fee_rate 非空时只改费率(dataclass replace), 其余字段逐位不变."""
    import dataclasses

    from src.strategy import load_config
    cfg = load_config(CFG_PATH)
    if fee_rate is not None:
        cfg = dataclasses.replace(cfg, fee_rate=float(fee_rate))
    return cfg


def run_baseline(fee_rate: float | None = None):
    from src.backtest import run_backtest
    return run_backtest(load_cfg(fee_rate))


# --------------------------------------------------------------------------
# E2 轨 A: 脉冲 -> 横截面 tiebreaker (patch sbt.compute_all_factors)
# --------------------------------------------------------------------------
def run_track_a(w: float, hold: int, pulse_by_etf: dict):
    """份额流入脉冲的那只 ETF, 在 momentum 上获得 tiebreaker 加分.

    注入 momentum 而非 score: score 在 backtest 内部按
    `eff_mom_w * mom - eff_vol_w * vol` 组装, patch momentum 是不改 src/ 的前提下
    唯一能影响横截面打分的入口(沿用 _exp_etf_flow_style_e2.py 的已验证做法)。
    加分量按 SCORE_GAP_THRESHOLD 缩放, 保持 tiebreaker 语义: 只在打分接近时才起作用,
    不足以单独改变排序 —— 否则就不是 tiebreaker 而是新的主因子了。
    """
    import src.backtest as sbt
    from src.backtest import run_backtest

    cfg = load_cfg()
    orig = sbt.compute_all_factors
    st = {"calls": 0, "applied": 0, "cols_missing": 0}

    def patched(*args, **kwargs):
        f = orig(*args, **kwargs)
        st["calls"] += 1
        mom = f["momentum"].copy()
        cols = list(mom.columns)
        vals = mom.values.copy()
        adj = w * SCORE_GAP_THRESHOLD
        for name, fl in pulse_by_etf.items():
            if name not in cols:
                st["cols_missing"] += 1
                continue
            j = cols.index(name)
            act = hold_expand(fl, hold).reindex(mom.index).fillna(False)
            hit = act.to_numpy()
            for i in np.where(hit)[0]:
                if not np.isnan(vals[i, j]):
                    vals[i, j] += adj
                    st["applied"] += 1
        mom.iloc[:, :] = vals
        f["momentum"] = mom
        return f

    sbt.compute_all_factors = patched
    try:
        res = run_backtest(cfg)
    finally:
        sbt.compute_all_factors = orig
    audit = {
        "point": "sbt.compute_all_factors", "w": w, "hold": hold,
        "n_calls": st["calls"], "n_cells_applied": st["applied"],
        "cols_missing": st["cols_missing"],
        "never_applied": bool(st["applied"] == 0),
        "aligned": bool(st["applied"] > 0),
    }
    return res, audit


# --------------------------------------------------------------------------
# E2 轨 B: 脉冲 -> 市场级防御 (patch sbt.calculate_defense_ratio)
# --------------------------------------------------------------------------
def run_track_b(delta: float, hold: int, flag_out: pd.Series,
                flag_in: pd.Series, dates: list,
                fee_rate: float | None = None):
    """聚合净申赎脉冲 -> 双向调防御.

    极端净流出(资金撤离) => 多防御 +delta; 极端净流入(风险偏好) => 少防御 -delta。

    **注入点必须是 calculate_defense_ratio, 不能是 compute_ashare_vol_boost**:
    后者在 v4.6 下恒返 0, "少防御"分支会因 max(0, 0-delta)=0 静默失效, 使双向测试
    退化成单边(既有 _exp_etf_flow_style_e2.py 已记录过这个 latent bug)。
    该函数不接收周下标, 故用调用计数器对齐 week_dates(); 未对齐即报错。
    """
    import src.backtest as sbt
    from src.backtest import run_backtest

    cfg = load_cfg(fee_rate)
    orig = sbt.calculate_defense_ratio
    a_out = hold_expand(flag_out, hold)
    a_in = hold_expand(flag_in, hold)
    st = {"k": 0, "more": 0, "less": 0, "over": 0, "miss": 0}

    def patched(nasdaq_vol, config, base_def_alloc=None):
        base = orig(nasdaq_vol, config, base_def_alloc=base_def_alloc)
        k = st["k"]
        st["k"] = k + 1
        if k >= len(dates):
            st["over"] += 1
            return base
        d = dates[k]
        try:
            o = bool(a_out.asof(d)) if len(a_out) else False
            i_ = bool(a_in.asof(d)) if len(a_in) else False
        except (KeyError, TypeError):
            st["miss"] += 1
            return base
        if o:
            st["more"] += 1
            return float(min(1.0, base + delta))
        if i_:
            st["less"] += 1
            return float(max(0.0, base - delta))
        return base

    sbt.calculate_defense_ratio = patched
    try:
        res = run_backtest(cfg)
    finally:
        sbt.calculate_defense_ratio = orig
    audit = {
        "point": "sbt.calculate_defense_ratio", "delta": delta, "hold": hold,
        "n_calls": int(st["k"]), "n_expected_weeks": len(dates),
        "n_more_defense": int(st["more"]), "n_less_defense": int(st["less"]),
        "n_beyond_dates": int(st["over"]), "n_missing": int(st["miss"]),
        "never_applied": bool(st["more"] + st["less"] == 0),
        "aligned": bool(st["k"] == len(dates) and st["over"] == 0),
    }
    if not audit["aligned"]:
        raise RuntimeError(f"轨B delta={delta} hold={hold} 注入未对齐: {audit}")
    return res, audit


# --------------------------------------------------------------------------
# E1' 事件研究
# --------------------------------------------------------------------------
# 预注册主假说(理论驱动, 唯一一条): 池内资金极端净流出 -> 后续风险上升。
PREREG = ("pool_net_flow", "low")


def run_e1(sigs: dict, tg: pd.DataFrame, healthy: list) -> dict:
    """事件研究扫描. 主判据为预注册假说, 其余标注探索性。"""
    combos = [("pool_net_flow", "low"), ("pool_net_flow", "high"),
              ("pool_net_flow_4w", "low"), ("pool_net_flow_4w", "high")]
    for name in healthy:
        combos.append((f"growth::{name}", "high"))
        combos.append((f"growth::{name}", "low"))

    out = {}
    for key, side in combos:
        if key.startswith("growth::"):
            s = sigs["per_etf_growth"][key.split("::", 1)[1]]
        else:
            s = sigs[key]
        flag, cov = pulse_and_coverage(s, side)
        ev = fh.event_study(flag, tg)
        ev["coverage"] = cov
        out[f"{key}|{side}"] = ev
    return out


# --------------------------------------------------------------------------
# E2 网格
# --------------------------------------------------------------------------
# 两轨主指标**按经济含义分别先定**, 不事后择优:
#   轨 A 选股 -> 目标是选得更准, 主指标 Sharpe(风险调整收益)
#   轨 B 防御 -> 目标是降回撤, 主指标 Ulcer index(对全程回撤敏感)
TRACK_PRIMARY = {"A": "d_sharpe", "B": "d_ulcer_pp"}


def e2_run_all(sigs: dict, healthy: list) -> dict:
    base = run_baseline()
    bm = fh.metrics_of(base)
    print(f"  baseline: CAGR={bm['cagr_pct']}% MaxDD={bm['maxdd_pct']}% "
          f"Ulcer={bm['ulcer_pct']}% Sharpe={bm['sharpe']}", flush=True)

    growth = sigs["per_etf_growth"]
    pulse_by_etf, cov_a = {}, {}
    for name in healthy:
        fl, cov = pulse_and_coverage(growth[name], "high")
        pulse_by_etf[name] = fl
        cov_a[name] = cov

    fl_out, cov_out = pulse_and_coverage(sigs["pool_net_flow"], "low")
    fl_in, cov_in = pulse_and_coverage(sigs["pool_net_flow"], "high")
    dates = week_dates()

    rows, audits = [], []
    for w in TRACK_A_W:
        for hold in HOLD_GRID:
            res, au = run_track_a(w, hold, pulse_by_etf)
            m = fh.metrics_of(res)
            rows.append({"track": "A", "param": w, "hold": hold, **m,
                         **_deltas(m, bm),
                         "n_applied": au["n_cells_applied"]})
            audits.append(au)
        print(f"  轨A w={w}: 3 组 hold 完成", flush=True)

    for delta in TRACK_B_DELTA:
        for hold in HOLD_GRID:
            res, au = run_track_b(delta, hold, fl_out, fl_in, dates)
            m = fh.metrics_of(res)
            cdd = fh.conditional_dd(res, fl_out)
            rows.append({"track": "B", "param": delta, "hold": hold, **m,
                         **_deltas(m, bm),
                         "n_more_defense": au["n_more_defense"],
                         "n_less_defense": au["n_less_defense"],
                         "cond_dd_mean_pct": cdd.get("mean_fwd_dd_pct")})
            audits.append(au)
        print(f"  轨B delta={delta}: 3 组 hold 完成", flush=True)

    return {
        "baseline": bm,
        "grid": rows,
        "coverage_track_a": cov_a,
        "coverage_pool_out": cov_out,
        "coverage_pool_in": cov_in,
        "n_runs": len(rows),
        "audit_all_aligned": bool(all(a["aligned"] for a in audits)),
        "audit_none_never_applied": bool(
            all(not a["never_applied"] for a in audits)),
        "audit_sample_a": audits[0],
        "audit_sample_b": next((a for a in audits
                                if a["point"].endswith("defense_ratio")), None),
    }


def _deltas(m: dict, bm: dict) -> dict:
    return {
        "d_cagr_pp": round(m["cagr_pct"] - bm["cagr_pct"], 4),
        "d_maxdd_pp": round(m["maxdd_pct"] - bm["maxdd_pct"], 4),
        "d_ulcer_pp": round(m["ulcer_pct"] - bm["ulcer_pct"], 4),
        "d_avg_dd_pp": round(m["avg_dd_pct"] - bm["avg_dd_pct"], 4),
        "d_sharpe": round(m["sharpe"] - bm["sharpe"], 4),
        "d_calmar": round(m["calmar"] - bm["calmar"], 4),
        "d_turnover": int(m["rebalance_count"] - bm["rebalance_count"]),
    }


def e2_robust(grid: list) -> dict:
    """按轨判定稳健区域: 主指标改善 + 多数格子 + 邻域一致 + CAGR 约束."""
    df = pd.DataFrame(grid)
    out = {}
    for track, g in df.groupby("track"):
        col = TRACK_PRIMARY[track]
        better_is_higher = col == "d_sharpe"
        good = g[col] > 0 if better_is_higher else g[col] < 0
        piv = g.pivot(index="param", columns="hold", values=col)
        best = (g.loc[g[col].idxmax()] if better_is_higher
                else g.loc[g[col].idxmin()])
        nb = g[((g.param == best.param) | (g.hold == best.hold))
               & ~((g.param == best.param) & (g.hold == best.hold))]
        nbg = ((nb[col] > 0).sum() if better_is_higher else (nb[col] < 0).sum())
        out[track] = {
            "primary_metric": col,
            "n_cells": int(piv.size), "n_improve": int(good.sum()),
            "improve_share_pct": round(100.0 * good.sum() / piv.size, 1),
            "best_primary": round(float(best[col]), 4),
            "best_param": float(best.param), "best_hold": int(best.hold),
            "best_d_cagr_pp": round(float(best.d_cagr_pp), 4),
            "best_d_sharpe": round(float(best.d_sharpe), 4),
            "best_d_ulcer_pp": round(float(best.d_ulcer_pp), 4),
            "best_d_maxdd_pp": round(float(best.d_maxdd_pp), 4),
            "best_d_turnover": int(best.d_turnover),
            "neighbors_total": int(len(nb)), "neighbors_improving": int(nbg),
            "isolated_optimum": bool(nbg < max(len(nb) // 2, 1)),
            "majority_improve": bool(good.sum() > piv.size / 2),
            "cagr_constraint_ok": bool(float(best.d_cagr_pp) > CAGR_TOL_PP),
            "maxdd_insensitive": bool(g.d_maxdd_pp.abs().max() < 1e-9),
        }
    return out


# --------------------------------------------------------------------------
# E3: 分期 OOS / 安慰剂噪声带 / 块 bootstrap / 成本敏感性
# --------------------------------------------------------------------------
def track_a_noop_check(grid: list) -> dict:
    """轨 A 是否根本没改变过任何东西(区分"无效"与"未测出结论").

    tiebreaker 加分量 = w x SCORE_GAP_THRESHOLD = 0.001~0.003, 而 momentum 本身
    量级在 0.1~0.3 。若绝大多数格子的所有 delta 恰好为 0, 说明加分从未翻转过
    任何一次横截面排序 —— 此时正确的描述是"本参数范围内未测出结论", **不能写成
    "份额选股无效"** —— 后者需要把加分量放到能真正改变持仓的尺度才能声称。
    """
    cols = ("d_sharpe", "d_cagr_pp", "d_ulcer_pp", "d_maxdd_pp", "d_turnover")
    g = [r for r in grid if r.get("track") == "A"]
    zero = [r for r in g
            if all(abs(float(r.get(c, 0.0))) < 1e-9 for c in cols)]
    return {
        "n_cells": len(g),
        "n_cells_all_zero": len(zero),
        "n_applied_min": min([int(r.get("n_applied", 0)) for r in g] or [0]),
        "n_applied_max": max([int(r.get("n_applied", 0)) for r in g] or [0]),
        "adj_magnitude_range": [round(min(TRACK_A_W) * SCORE_GAP_THRESHOLD, 5),
                                round(max(TRACK_A_W) * SCORE_GAP_THRESHOLD, 5)],
        "is_noop": bool(g and len(zero) >= len(g) * 0.5),
        "interpretation": (
            "加分量相对 momentum 量级过小, 从未翻转横截面排序 → 轨 A 属"
            "**未测出结论**(no-op), 不得解读为份额选股被证否"
            if (g and len(zero) >= len(g) * 0.5) else
            "轨 A 存在实质差异, 结果可解读"),
    }


def _sub_metrics(nav_df: pd.DataFrame, lo=None, hi=None) -> dict:
    """子区间络效. 回撤在区间起点重置——基线与变体同口径, 差值仍可比."""
    d = nav_df
    if lo is not None:
        d = d[d.index >= pd.Timestamp(lo)]
    if hi is not None:
        d = d[d.index < pd.Timestamp(hi)]
    r = d["weekly_return"].astype(float).dropna()
    if len(r) < 52:
        return {"insufficient": True, "n_weeks": int(len(r))}
    nav = (1.0 + r).cumprod()
    dd = 1.0 - nav / nav.cummax()
    yrs = len(r) / 52.0
    sd = float(r.std(ddof=1))
    return {
        "n_weeks": int(len(r)),
        "start": str(d.index.min().date()), "end": str(d.index.max().date()),
        "cagr_pct": round((float(nav.iloc[-1]) ** (1.0 / yrs) - 1.0) * 100, 4),
        "maxdd_pct": round(float(dd.max()) * 100, 4),
        "ulcer_pct": round(float(np.sqrt((dd ** 2).mean())) * 100, 4),
        "sharpe": (round(float(r.mean()) / sd * np.sqrt(52.0), 4)
                   if sd > 0 else None),
    }


def e3_split_oos(base_res, var_res) -> dict:
    """分期 OOS: 改善是否只来自单一时期.

    判据是**符号一致性**而非各期各自显著: 半样本仅 ~300 周, 要求每期都达到
    全样本级的显著性等于要求效应量翻倍, 不现实; 但若某期方向相反, 则改善来自
    特定时期而非稳定机制 —— 这正是股指期货线上把 E2 退化成"2019-2020 单年实验"
    的那类陷阱。
    """
    segs = {"full": (None, None), "pre_2019": (None, E3_SPLIT),
            "post_2019": (E3_SPLIT, None)}
    out = {}
    for k, (lo, hi) in segs.items():
        b = _sub_metrics(base_res.nav_series, lo, hi)
        v = _sub_metrics(var_res.nav_series, lo, hi)
        if b.get("insufficient") or v.get("insufficient"):
            out[k] = {"insufficient": True, "n_weeks": b.get("n_weeks")}
            continue
        out[k] = {
            "n_weeks": b["n_weeks"], "start": b["start"], "end": b["end"],
            "base_ulcer_pct": b["ulcer_pct"], "var_ulcer_pct": v["ulcer_pct"],
            "d_ulcer_pp": round(v["ulcer_pct"] - b["ulcer_pct"], 4),
            "d_maxdd_pp": round(v["maxdd_pct"] - b["maxdd_pct"], 4),
            "d_cagr_pp": round(v["cagr_pct"] - b["cagr_pct"], 4),
            "d_sharpe": (round(v["sharpe"] - b["sharpe"], 4)
                         if (b["sharpe"] is not None
                             and v["sharpe"] is not None) else None),
        }
    halves = [out[k] for k in ("pre_2019", "post_2019")
              if not out[k].get("insufficient")]
    out["n_halves"] = len(halves)
    out["sign_consistent"] = bool(len(halves) == 2
                                  and all(h["d_ulcer_pp"] < 0 for h in halves))
    out["worse_half"] = (max(halves, key=lambda h: h["d_ulcer_pp"])["d_ulcer_pp"]
                         if halves else None)
    return out


def e3_placebo(delta: int | float, hold: int, n_out: int, n_in: int,
               index: pd.DatetimeIndex, dates: list, base_ulcer: float,
               observed_d_ulcer: float, n_iter: int = E3_PLACEBO_N) -> dict:
    """安慰剂噪声带: 保持触发**次数与 hold 不变**, 只把触发日期随机重排, 跑整轮回测.

    这直接回答 block bootstrap 回答不了的混淆: 改善是来自**份额信号本身**, 还是
    来自"偶尔多加点防御/少加点防御"这个动作本身(在上行市中降仓本身就会降回撤)。
    随机日期只从 MIN_HIST 之后取, 与真信号的可用区间完全一致。
    """
    rng = np.random.default_rng(SEED)
    pool = np.arange(MIN_HIST, len(index))
    if len(pool) < n_out + n_in + 1:
        return {"insufficient": True}
    vals = []
    for it in range(n_iter):
        po = rng.choice(pool, size=n_out, replace=False)
        pi = rng.choice(np.setdiff1d(pool, po), size=n_in, replace=False)
        fo = pd.Series(False, index=index)
        fi = pd.Series(False, index=index)
        fo.iloc[po] = True
        fi.iloc[pi] = True
        res, _au = run_track_b(delta, hold, fo, fi, dates)
        vals.append(fh.metrics_of(res)["ulcer_pct"] - base_ulcer)
        if (it + 1) % 10 == 0:
            print(f"    安慰剂 {it + 1}/{n_iter}", flush=True)
    a = np.asarray(vals, dtype=float)
    better = int((a <= observed_d_ulcer).sum())
    return {
        "n_iter": int(len(a)), "n_out": int(n_out), "n_in": int(n_in),
        "observed_d_ulcer_pp": round(float(observed_d_ulcer), 4),
        "placebo_mean_pp": round(float(a.mean()), 4),
        "placebo_sd_pp": round(float(a.std(ddof=1)), 4),
        "placebo_p05_pp": round(float(np.percentile(a, 5)), 4),
        "placebo_median_pp": round(float(np.median(a)), 4),
        "placebo_min_pp": round(float(a.min()), 4),
        "n_placebo_better": better,
        "pct_placebo_better": round(100.0 * better / len(a), 2),
        "z_vs_placebo": (round(float((observed_d_ulcer - a.mean())
                                     / a.std(ddof=1)), 3)
                         if a.std(ddof=1) > 0 else None),
        "passed": bool(100.0 * better / len(a) <= E3_PLACEBO_MAX_PCT),
        "samples_pp": [round(float(x), 4) for x in a],
    }


def e3_bootstrap(base_res, var_res, n_iter: int = E3_BOOT_N,
                 block: int = E3_BLOCK_W) -> dict:
    """成对循环块 bootstrap: 对**同一组时间下标**重采样基线与变体的周收益.

    保留两者的配对关系(否则差值全是两条无关路径的噪声)与块内自相关
    (block=8 周 >= 2x max(hold)=8, 足以覆盖 4 周前瞻窗口的重叠)。
    判据: ΔUlcer 的 97.5% 分位 < 0, 即整条 95% CI 落在改善侧。
    """
    b = base_res.nav_series["weekly_return"].astype(float)
    v = var_res.nav_series["weekly_return"].astype(float)
    idx = b.index.intersection(v.index)
    rb = b.reindex(idx).to_numpy()
    rv = v.reindex(idx).to_numpy()
    n = len(rb)
    if n < block * 4:
        return {"insufficient": True, "n": n}
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(SEED)

    def _ulcer(r):
        nav = np.cumprod(1.0 + r)
        dd = 1.0 - nav / np.maximum.accumulate(nav)
        return float(np.sqrt((dd ** 2).mean())) * 100.0

    d = []
    for _ in range(n_iter):
        starts = rng.integers(0, n, size=n_blocks)
        sel = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
        d.append(_ulcer(rv[sel]) - _ulcer(rb[sel]))
    a = np.asarray(d, dtype=float)
    hi = float(np.percentile(a, 97.5))
    return {
        "n_iter": int(n_iter), "block_weeks": int(block), "n_weeks": int(n),
        "observed_d_ulcer_pp": round(_ulcer(rv) - _ulcer(rb), 4),
        "mean_pp": round(float(a.mean()), 4),
        "ci_lo_pp": round(float(np.percentile(a, 2.5)), 4),
        "ci_hi_pp": round(hi, 4),
        "pct_improving": round(float((a < 0).mean()) * 100, 2),
        "passed": bool(hi < 0),
    }


def e3_cost(delta: float, hold: int, flag_out: pd.Series,
            flag_in: pd.Series, dates: list) -> dict:
    """成本敏感性: Δ调仓为正时, 高费率下改善是否被吃揉.

    基线与变体**在同一费率下各跑一次**, 否则差值里会混入基线自身的费率衰减。
    """
    rows = []
    for bp in E3_FEE_BP:
        fee = bp / 10000.0
        mb = fh.metrics_of(run_baseline(fee_rate=fee))
        res, _au = run_track_b(delta, hold, flag_out, flag_in, dates,
                               fee_rate=fee)
        mv = fh.metrics_of(res)
        rows.append({
            "fee_bp": bp,
            "base_cagr_pct": mb["cagr_pct"], "var_cagr_pct": mv["cagr_pct"],
            "base_ulcer_pct": mb["ulcer_pct"], "var_ulcer_pct": mv["ulcer_pct"],
            "d_ulcer_pp": round(mv["ulcer_pct"] - mb["ulcer_pct"], 4),
            "d_cagr_pp": round(mv["cagr_pct"] - mb["cagr_pct"], 4),
            "d_sharpe": round(mv["sharpe"] - mb["sharpe"], 4),
            "d_turnover": int(mv["rebalance_count"] - mb["rebalance_count"]),
        })
        print(f"    费率 {bp}bp: ΔUlcer={rows[-1]['d_ulcer_pp']:+.4f}pp "
              f"ΔCAGR={rows[-1]['d_cagr_pp']:+.3f}pp", flush=True)
    ok = bool(all(r["d_ulcer_pp"] < 0 for r in rows)
              and all(r["d_cagr_pp"] > CAGR_TOL_PP for r in rows))
    return {"grid": rows, "fee_max_bp": max(E3_FEE_BP), "passed": ok}


def _half_counts(flag: pd.Series, split: str = E3_SPLIT) -> dict:
    """触发在分期切点两侧的次数 —— 判断某半样本的检验功率。"""
    idx = flag[flag.astype(bool)].index
    t = pd.Timestamp(split)
    return {"pre": int((idx < t).sum()), "post": int((idx >= t).sum())}


def e3_branch_diag(delta: float, hold: int, flag_out: pd.Series,
                   flag_in: pd.Series, dates: list, base_res) -> dict:
    """探索性诊断(**不入门禁**): 双向干预里到底哪一支造成分期不一致.

    “多防御”(极端净流出)与“少防御”(极端净流入)是两个独立假说, 合在一起跑时
    总效果可能是一支赚一支亏。分开跑才能说清机制, 但这是看到结果后才做的拆解,
    所以只作诊断依据, 不得用它反过来挽救裁决。
    """
    empty = pd.Series(False, index=flag_out.index)
    variants = {"both": (flag_out, flag_in),
                "only_more_defense": (flag_out, empty),
                "only_less_defense": (empty, flag_in)}
    segs = ("full", "pre_2019", "post_2019")
    keys = ("d_ulcer_pp", "d_maxdd_pp", "d_cagr_pp", "d_sharpe")
    out = {"n_runs": 0}
    for name, (fo, fi) in variants.items():
        res, _au = run_track_b(delta, hold, fo, fi, dates)
        out["n_runs"] += 1
        so = e3_split_oos(base_res, res)
        out[name] = {s: {k: (so.get(s) or {}).get(k) for k in keys}
                     for s in segs}
        print(f"    分支 {name}: 全样本 ΔUlcer="
              f"{out[name]['full']['d_ulcer_pp']} 分期 "
              f"{out[name]['pre_2019']['d_ulcer_pp']} / "
              f"{out[name]['post_2019']['d_ulcer_pp']}", flush=True)
    out["trigger_halves"] = {"more_defense": _half_counts(flag_out),
                             "less_defense": _half_counts(flag_in)}
    return out


def e3_onesided(delta: float, hold: int, flag_out: pd.Series, dates: list,
                base_res, base_ulcer: float) -> dict:
    """对单边变体(只留"净流出->多防御")跑**同一套四项门禁**.

    诚实标注: 这是看到分支诊断后才选定的假说, **已经用掉了自由度**
    (3 个分支里挑了最好的一个)。它的门禁结果**不能**用来改写本次裁决, 只能
    作为"下一轮在新样本上预注册重跑"的候选假说。若按 3 个分支做 Bonferroni
    校正, 安慰剂阈值应从 E3_PLACEBO_MAX_PCT 收紧到其 1/3。
    """
    empty = pd.Series(False, index=flag_out.index)
    var_res, _au = run_track_b(delta, hold, flag_out, empty, dates)
    obs = round(fh.metrics_of(var_res)["ulcer_pct"] - base_ulcer, 4)
    sub = {
        "split_oos": e3_split_oos(base_res, var_res),
        "bootstrap": e3_bootstrap(base_res, var_res),
        "cost": e3_cost(delta, hold, flag_out, empty, dates),
        "placebo": e3_placebo(delta, hold, int(flag_out.sum()), 0,
                              flag_out.index, dates, base_ulcer, obs),
    }
    g = e3_gate(sub)
    pl = sub["placebo"]
    return {
        "branch": "only_more_defense",
        "post_hoc": True,
        "dof_used": 3,
        "bonferroni_placebo_threshold_pct": round(E3_PLACEBO_MAX_PCT / 3.0, 2),
        "placebo_passes_bonferroni": bool(
            pl.get("pct_placebo_better") is not None
            and pl["pct_placebo_better"] <= E3_PLACEBO_MAX_PCT / 3.0),
        "observed_d_ulcer_pp": obs,
        "gate": g,
        "split_oos": sub["split_oos"], "bootstrap": sub["bootstrap"],
        "cost": sub["cost"],
        "placebo": {k: v for k, v in pl.items() if k != "samples_pp"},
        "n_backtests": int(1 + 2 * len(E3_FEE_BP)
                           + int(pl.get("n_iter", 0))),
    }


def e3_gate(e3: dict) -> dict:
    """E3 四项门禁全部先定. 全过才议引入 src/; 部分过只能停在观察层."""
    so = e3.get("split_oos") or {}
    bs = e3.get("bootstrap") or {}
    pl = e3.get("placebo") or {}
    ct = e3.get("cost") or {}
    checks = {
        "split_sign_consistent": bool(so.get("sign_consistent")),
        "bootstrap_ci_all_negative": bool(bs.get("passed")),
        "beats_placebo_band": bool(pl.get("passed")),
        "robust_to_cost": bool(ct.get("passed")),
    }
    n = int(sum(checks.values()))
    return {"checks": checks, "n_pass": n, "n_total": len(checks),
            "passed": bool(n == len(checks)),
            "failed_items": [k for k, v in checks.items() if not v]}


# --------------------------------------------------------------------------
# 裁决
# --------------------------------------------------------------------------
def recompute_verdict(res: dict) -> dict:
    """裁决. 幂等(可对已有 json 重算)。

    E2 权重高于 E1': 用户诉求正是"不要只凭相关性判不可行"。故只要任一轨存在
    稳健改善区域, 即给出(条件) GO; 反之若仅孤立点改善, 那是 3x3 网格的参数噪声。
    """
    ev = res.get("e1", {})
    rob = res.get("e2_robust", {})
    grid = pd.DataFrame(res.get("e2", {}).get("grid", []))

    pk = f"{PREREG[0]}|{PREREG[1]}"
    pre_hits = []
    d = ev.get(pk) or {}
    for t in fh.fb.RISK_TARGETS:
        pt = (d.get("per_target") or {}).get(t)
        if pt and not pt.get("insufficient") and pt.get("sig_05"):
            pre_hits.append(f"{pk}->{t}")
    expl_hits = sum(
        1 for _k, dd in ev.items()
        for _t, pt in (dd.get("per_target") or {}).items()
        if pt and not pt.get("insufficient") and pt.get("sig_05"))

    robust_tracks, isolated_tracks = [], []
    for tk, v in rob.items():
        higher = v["primary_metric"] == "d_sharpe"
        right_dir = (v["best_primary"] > 0) if higher else (v["best_primary"] < 0)
        ok = (v["majority_improve"] and not v["isolated_optimum"]
              and v["cagr_constraint_ok"] and right_dir)
        if ok:
            robust_tracks.append(f"轨{tk}({v['primary_metric']}"
                                 f"{v['best_primary']:+.4f})")
        elif right_dir:
            isolated_tracks.append(f"轨{tk}")

    n_imp = int(sum(v["n_improve"] for v in rob.values()))
    n_cells = int(sum(v["n_cells"] for v in rob.values()))

    # E3 门禁与轨 A no-op 识别(都是纯函数, --render-only 重算逐位一致)
    e3 = res.get("e3") or {}
    gate3 = e3_gate(e3) if e3 else {}
    noop_a = track_a_noop_check(res.get("e2", {}).get("grid", []))
    g3n = gate3.get("n_pass", 0)
    g3f = ", ".join(gate3.get("failed_items", [])) or "无"

    if robust_tracks and pre_hits and not e3:
        case = "G1 E2 稳健改善 + 预注册事件研究显著(E3 未跑)"
        decision, nxt = "条件 GO", (
            "E3 未执行, 尚不能区分真实机制与路径噪声。先跑分期 OOS + 安慰剂噪声带 "
            "+ 块 bootstrap + 成本敏感性。")
    elif robust_tracks and pre_hits and gate3.get("passed"):
        case = "G1a E2 稳健改善 + 预注册显著 + E3 四项全过"
        decision, nxt = "GO", (
            "可议引入 src/: 新增 YAML 开关(默认关) + 单测, 先在周报观察层灰度一个季度, "
            "确认实盘触发频率与回测一致后再打开资金分配。")
    elif robust_tracks and pre_hits and g3n >= 3:
        case = f"G1b E2 稳健改善 + 预注册显著, 但 E3 仅 {g3n}/4 (未过: {g3f})"
        decision, nxt = "条件 GO(仅观察层)", (
            f"改善方向真实但未全部经受住 E3 检验(未过: {g3f})。先作为**周报告警指标**"
            "上线(不改资金分配), 累积实盘触发样本; 未过项转继后再议改 src/。")
    elif robust_tracks and pre_hits:
        case = f"G1c E2 稳健改善 + 预注册显著, 但 E3 仅 {g3n}/4 (未过: {g3f})"
        decision, nxt = "NO-GO(仅保留为观察/告警层)", _g1c_next(e3)
    elif robust_tracks:
        case = "G2 E2 稳健改善但预注册事件研究不显著"
        decision, nxt = "条件 GO", (
            "回测层面有效但机制未独立证实。先做分期 OOS + 噪声带对照, "
            "确认改善超出噪声尺度再谈集成。")
    elif isolated_tracks:
        case = "G3 仅孤立参数点改善"
        decision, nxt = "NO-GO", (
            "3x3 网格里总会有格子碰巧变好, 邻域不一致即参数噪声。不得改 src/。")
    elif n_imp > 0:
        case = "G4 有改善格子但未构成稳健区域"
        decision, nxt = "NO-GO", "改善分散或不满足 CAGR 约束, 不得改 src/。"
    else:
        case = "G5 两轨全网格均无改善"
        decision, nxt = "NO-GO", "份额脉冲在本策略上无正向作用, 方向关闭。"

    return {
        "case": case, "decision": decision, "next": nxt,
        "prereg": pk,
        "prereg_significant": pre_hits,
        "n_prereg_significant": len(pre_hits),
        "n_exploratory_significant_uncorrected": expl_hits,
        "robust_tracks": robust_tracks,
        "isolated_only": isolated_tracks,
        "grid_improve_cells": n_imp, "grid_total_cells": n_cells,
        "absorption": absorption_check(res),
        "e3_gate": gate3,
        "track_a": noop_a,
        "e2_actually_ran": bool(len(grid) > 0),
        "n_backtests": int(len(grid)) + 1 + int(e3.get("n_backtests", 0)),
    }


def _g1c_next(e3: dict) -> str:
    """G1c 下一步文案: 数字从 res 重算, 不硬编码, 保证 --render-only 幂等."""
    so = e3.get("split_oos") or {}
    pl = e3.get("placebo") or {}
    bs = e3.get("bootstrap") or {}
    parts = []
    pre, post = so.get("pre_2019") or {}, so.get("post_2019") or {}
    if pre.get("d_ulcer_pp") is not None and post.get("d_ulcer_pp") is not None:
        parts.append(
            f"全样本改善集中在 {E3_SPLIT[:4]} 前(ΔUlcer {pre['d_ulcer_pp']:+.4f}pp, "
            f"ΔCAGR {pre['d_cagr_pp']:+.3f}pp), 而 {E3_SPLIT[:4]} 后反向"
            f"(ΔUlcer {post['d_ulcer_pp']:+.4f}pp, ΔMaxDD {post['d_maxdd_pp']:+.4f}pp)")
    if bs.get("ci_lo_pp") is not None and not bs.get("passed"):
        parts.append(f"块 bootstrap 95% CI [{bs['ci_lo_pp']:+.4f}, "
                     f"{bs['ci_hi_pp']:+.4f}]pp 跨零, 效应量未超出路径噪声")
    if pl.get("passed"):
        parts.append(f"但安慰剂检验通过(只有 {pl['n_placebo_better']}/{pl['n_iter']} "
                     f"个随机重排优于实测, z={pl['z_vs_placebo']}), 说明**触发时点确实"
                     "携带信息**, 不是随便换个日期都能做到")
    tail = ("结论: 份额脉冲**不接入资金分配**(不改 src/, 不新增 YAML 开关), "
            "而是作为**周报观察/告警项**: 触发时在周报标注'池内异常净流出', "
            "人工复核不自动改仓。待未过项(分期稳定性/噪声带)在新增样本上转继后再议集成。")
    os_ = e3.get("onesided_more_defense") or {}
    og = os_.get("gate") or {}
    if og.get("n_pass") is not None:
        tail += (f"下一轮候选假说(事后选择, 已用掉自由度, 本次不计入裁决): "
                 f"只保留'净流出->多防御'单边干预, 其同套门禁 "
                 f"{og['n_pass']}/{og['n_total']}"
                 + (f" (未过: {', '.join(og.get('failed_items', []))})"
                    if og.get("failed_items") else " 全过")
                 + ", 需在新样本上预注册重跑才能宣称。")
    return ("; ".join(parts) + "。" + tail) if parts else tail


def absorption_check(res: dict) -> dict:
    """信号是否有预测力 vs 预测力是否已被现有防御层吸收(同股指期货线口径)."""
    ev = res.get("e1", {})
    out = {}
    for k, d in ev.items():
        per = d.get("per_target") or {}
        mkt = [abs(float(per[t]["rank_biserial"]))
               for t in ("fwd_vol_4w", "fwd_maxdd_4w")
               if per.get(t) and not per[t].get("insufficient")]
        ps = per.get("fwd_strat_dd_4w")
        if not mkt or not ps or ps.get("insufficient"):
            continue
        st = abs(float(ps["rank_biserial"]))
        mx = max(mkt)
        out[k] = {
            "market_effect_max": round(mx, 4),
            "strategy_effect": round(st, 4),
            "ratio": round(st / mx, 3) if mx > 0 else None,
            "absorbed": bool(mx > 0 and st < 0.5 * mx),
        }
    n_abs = sum(1 for v in out.values() if v["absorbed"])
    return {"per_signal": out, "n_absorbed": n_abs, "n_total": len(out),
            "all_absorbed": bool(out and n_abs == len(out))}


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def _sk(s: str) -> str:
    """信号 key 内含 '|', 直接写入 markdown 表格会被当成列分隔符。"""
    return str(s).replace("|", " / ")


def render(res: dict) -> None:
    v = res["verdict"]
    e2 = res["e2"]
    b = e2["baseline"]
    L = []
    L.append("# ETF 份额脉冲因子: E0 + E1' + E2 双轨集成回测 + E3 噪声/稳定性验证")
    L.append("")
    L.append(f"**裁决: {v['decision']} —— {v['case']}**")
    L.append("")
    L.append(f"下一步: {v['next']}")
    L.append("")
    L.append(f"本次**实跑了 {v['n_backtests']} 次回测**(1 基线 + "
             f"{v['n_backtests'] - 1} 变体)。此前份额因子两次被判否都止步于相关性: "
             "`exp_share_flow` 横截面 rank IC 判 NO-GO、`exp_etf_flow_style` "
             "`proceed_to_e2=False`、`exp_etf_flow_style_e2` 读到该标志后拒绝执行 —— "
             "**E2 从未真正运行过**。连续 IC 隐含'因子每周都线性有效', 对稀疏脉冲有盲区, "
             "故本次换事件研究 + 真回测范式。")
    L.append("")

    L.append("## E0 数据体检与健康子集")
    L.append("")
    L.append("| ETF | 首个有效周 | 上市后空洞 | 日频停滞率(2020+) | 最长冻结段 | 倍数跳变 | 健康 | 剔除原因 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, h in res["e0_health"].items():
        L.append(f"| {k} | {h['first_valid']} | {h['post_listing_gap_pct']}% | "
                 f"{h['daily_stagnation_2020plus_pct']}% | "
                 f"{h['longest_frozen_days']} 天 | {h['n_exact_ratio_jumps']} | "
                 f"{'是' if h['healthy'] else '**否**'} | "
                 f"{h['reject_reason'] or '-'} |")
    L.append("")
    L.append(f"健康子集: {', '.join(res['healthy'])}")
    L.append("")
    L.append("判据用**日频**停滞率(<50%)与冻结段(<120 天), 不用覆盖率 —— 红利低波覆盖 "
             "57.8% 是晚上市(首值 2018-12-21)造成, 上市后空洞率 0%, 用覆盖率剔它是误判。"
             "纳指 55.8% 停滞且有 187 天冻结段延续至 2026-06, 对实盘信号致命。")
    L.append("")

    L.append("## E1' 事件研究(脉冲视角)")
    L.append("")
    L.append(f"预注册主假说(理论驱动, 唯一一条): `{_sk(v['prereg'])}` "
             "—— 池内资金极端净流出 → 后续风险上升。")
    L.append("")
    L.append("| 组合 | 触发 | 年跳 | 覆盖 | 目标 | 触发组中位 | 基准组中位 | 效应量 | p(MWU) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k, d in res["e1"].items():
        cov = d.get("coverage", {})
        for t in fh.fb.RISK_TARGETS:
            pt = (d.get("per_target") or {}).get(t)
            if not pt or pt.get("insufficient"):
                continue
            mark = " (显著)" if pt["sig_05"] else ""
            L.append(f"| {_sk(k)} | {d['n_trigger']} | {cov.get('year_span')} | "
                     f"{'是' if cov.get('coverage_ok') else '否'} | {t} | "
                     f"{pt['med_trig']:.4f} | {pt['med_base']:.4f} | "
                     f"{pt['rank_biserial']:+.3f} | {pt['p_mwu']:.4f}{mark} |")
    L.append("")
    L.append(f"探索性组合中 p<0.05 的有 "
             f"{v['n_exploratory_significant_uncorrected']} 个(未做多重比较校正, 仅参考)。")
    L.append("")

    ab = v.get("absorption") or {}
    if ab.get("per_signal"):
        L.append("## 机制诊断: 无预测力, 还是预测力已被吸收?")
        L.append("")
        L.append("| 信号 | 市场层面效应量 | 策略层面效应量 | 策略/市场 | 已被吸收 |")
        L.append("|---|---|---|---|---|")
        for k, a in ab["per_signal"].items():
            L.append(f"| {_sk(k)} | {a['market_effect_max']:.4f} | "
                     f"{a['strategy_effect']:.4f} | {a['ratio']} | "
                     f"{'是' if a['absorbed'] else '否'} |")
        L.append("")

    L.append("## E2 双轨集成回测(monkeypatch, 未改 src/)")
    L.append("")
    L.append(f"基线 v4.6: CAGR {b['cagr_pct']}% | MaxDD {b['maxdd_pct']}% | "
             f"Ulcer {b['ulcer_pct']}% | Sharpe {b['sharpe']} | Calmar {b['calmar']}")
    L.append("")
    L.append("注入自证: "
             + ("全部对齐 且 无空转 ✓" if (e2["audit_all_aligned"]
                                          and e2["audit_none_never_applied"])
                else "**存在未对齐或空转 ✗**"))
    L.append("")
    L.append("轨 A = 份额流入脉冲 → momentum tiebreaker(patch `compute_all_factors`); "
             "轨 B = 聚合净申赎脉冲 → 双向调防御(patch `calculate_defense_ratio`)。")
    L.append("")
    L.append("轨 B 不能 patch `compute_ashare_vol_boost`: 它在 v4.6 下恒返 0, "
             "'少防御'分支会因 `max(0, 0−delta)=0` 静默失效, 使双向测试退化成单边。")
    L.append("")
    L.append("### 全参数网格(不挑最优, 全部列出)")
    L.append("")
    L.append("| 轨 | 参数 | hold | ΔSharpe | ΔCAGR | ΔUlcer | ΔMaxDD | Δ均回撤 | ΔCalmar | Δ调仓 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in e2["grid"]:
        L.append(f"| {r['track']} | {r['param']} | {r['hold']} | "
                 f"{r['d_sharpe']:+.4f} | {r['d_cagr_pp']:+.3f} | "
                 f"{r['d_ulcer_pp']:+.4f} | {r['d_maxdd_pp']:+.4f} | "
                 f"{r['d_avg_dd_pp']:+.4f} | {r['d_calmar']:+.3f} | "
                 f"{r['d_turnover']:+d} |")
    L.append("")
    na = v.get("track_a") or {}
    if na:
        L.append(f"轨 A 诊断: {na['n_cells_all_zero']}/{na['n_cells']} 个格子的**所有**指标"
                 f"差值恰好为 0; tiebreaker 加分量仅 "
                 f"{na['adj_magnitude_range'][0]}~{na['adj_magnitude_range'][1]}, "
                 f"而 momentum 本身量级在 0.1~0.3。注入确实生效(单轮命中单元格 "
                 f"{na['n_applied_min']}~{na['n_applied_max']}), 但从未翻转横截面排序。")
        L.append("")
        L.append(f"因此轨 A 属**{'未测出结论(no-op)' if na['is_noop'] else '有实质差异'}**"
                 "—— 不得解读为'份额选股被证否'。要真正检验轨 A 必须把加分量放到能改变"
                 "持仓的尺度, 但那就不再是 tiebreaker 而是新的主因子, 属于另一个假说。")
        L.append("")

    L.append("### 稳健区域判定")
    L.append("")
    L.append("主指标按经济含义**分轨先定**: 轨 A 选股 → Sharpe; 轨 B 防御 → Ulcer。"
             "要求多数格子改善 + 最优点邻域一致 + CAGR 恶化不超 0.5pp。")
    L.append("")
    L.append("| 轨 | 主指标 | 改善格子 | 最优值 | 最优参数 | ΔCAGR | ΔSharpe | ΔUlcer | 邻域改善 | 孤立 | 多数 | CAGR约束 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tk, r in res["e2_robust"].items():
        L.append(f"| {tk} | {r['primary_metric']} | "
                 f"{r['n_improve']}/{r['n_cells']} | {r['best_primary']:+.4f} | "
                 f"{r['best_param']}/hold{r['best_hold']} | "
                 f"{r['best_d_cagr_pp']:+.3f}pp | {r['best_d_sharpe']:+.4f} | "
                 f"{r['best_d_ulcer_pp']:+.4f}pp | "
                 f"{r['neighbors_improving']}/{r['neighbors_total']} | "
                 f"{'是' if r['isolated_optimum'] else '否'} | "
                 f"{'是' if r['majority_improve'] else '否'} | "
                 f"{'✓' if r['cagr_constraint_ok'] else '✗'} |")
    L.append("")

    e3 = res.get("e3") or {}
    g3 = v.get("e3_gate") or {}
    if e3:
        L.append("## E3 验证: 改善能否与噪声区分")
        L.append("")
        L.append(f"对轨 B 最优参数(delta={e3.get('best_delta')}, "
                 f"hold={e3.get('best_hold')}) 做四项前置门禁, 共额外实跑 "
                 f"{e3.get('n_backtests')} 次回测。")
        L.append("")
        L.append(f"**E3 门禁: {g3.get('n_pass')}/{g3.get('n_total')} 通过**"
                 + (f" —— 未过: {', '.join(g3.get('failed_items', []))}"
                    if g3.get("failed_items") else " —— 全过 ✓"))
        L.append("")

        so = e3.get("split_oos") or {}
        L.append("### E3-1 分期 OOS(符号一致性)")
        L.append("")
        L.append("| 区间 | 周数 | 起 | 迄 | 基线Ulcer | 变体Ulcer | ΔUlcer | ΔMaxDD | ΔCAGR | ΔSharpe |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for k in ("full", "pre_2019", "post_2019"):
            d = so.get(k) or {}
            if d.get("insufficient") or not d:
                L.append(f"| {k} | 样本不足 | - | - | - | - | - | - | - | - |")
                continue
            ds = d.get("d_sharpe")
            L.append(f"| {k} | {d['n_weeks']} | {d['start']} | {d['end']} | "
                     f"{d['base_ulcer_pct']:.4f}% | {d['var_ulcer_pct']:.4f}% | "
                     f"{d['d_ulcer_pp']:+.4f}pp | {d['d_maxdd_pp']:+.4f}pp | "
                     f"{d['d_cagr_pp']:+.3f}pp | "
                     f"{('%+.4f' % ds) if ds is not None else '-'} |")
        L.append("")
        L.append("判据是**两半样本符号一致**而非各期各自显著: 半样本仅 ~300 周, 要求每期"
                 "都达到全样本级显著性等于要求效应量翻倍; 但若某期方向相反, 则改善来自"
                 "特定时期而非稳定机制。结果: "
                 + ("两期同向改善 ✓" if so.get("sign_consistent")
                    else "**存在方向相反的半样本 ✗**"))
        L.append("")

        pl = e3.get("placebo") or {}
        if pl and not pl.get("insufficient"):
            L.append("### E3-2 安慰剂噪声带(触发日期随机重排)")
            L.append("")
            L.append(f"保持触发次数({pl['n_out']} 次多防御 / {pl['n_in']} 次少防御)与 hold "
                     f"不变, 只把日期随机重排, 跑 {pl['n_iter']} 轮完整回测得零分布。"
                     "这直接回答 bootstrap 回答不了的混淆: 改善是来自**份额信号本身**, "
                     "还是来自'偶尔调一下防御'这个动作本身。")
            L.append("")
            L.append("| 实测 ΔUlcer | 安慰剂均值 | 安慰剂 sd | 安慰剂 p05 | 安慰剂最优 | 优于实测的安慰剂 | z | 通过 |")
            L.append("|---|---|---|---|---|---|---|---|")
            L.append(f"| {pl['observed_d_ulcer_pp']:+.4f}pp | "
                     f"{pl['placebo_mean_pp']:+.4f}pp | {pl['placebo_sd_pp']:.4f} | "
                     f"{pl['placebo_p05_pp']:+.4f}pp | {pl['placebo_min_pp']:+.4f}pp | "
                     f"{pl['n_placebo_better']}/{pl['n_iter']} "
                     f"({pl['pct_placebo_better']}%) | {pl['z_vs_placebo']} | "
                     f"{'✓' if pl['passed'] else '✗'} |")
            L.append("")
            L.append(f"预设阈值: 优于实测的安慰剂比例 ≤ {E3_PLACEBO_MAX_PCT}%。")
            L.append("")

        bs = e3.get("bootstrap") or {}
        if bs and not bs.get("insufficient"):
            L.append("### E3-3 成对循环块 bootstrap(路径噪声)")
            L.append("")
            L.append(f"对**同一组时间下标**重采样基线与变体的周收益(保留配对关系), "
                     f"块长 {bs['block_weeks']} 周(≥ 2×max(hold), 覆盖 4 周前瞻窗口的重叠), "
                     f"{bs['n_iter']} 次重采样。")
            L.append("")
            L.append("| 实测 ΔUlcer | bootstrap 均值 | 95% CI 下 | 95% CI 上 | 改善比例 | 通过(CI全负) |")
            L.append("|---|---|---|---|---|---|")
            L.append(f"| {bs['observed_d_ulcer_pp']:+.4f}pp | {bs['mean_pp']:+.4f}pp | "
                     f"{bs['ci_lo_pp']:+.4f}pp | {bs['ci_hi_pp']:+.4f}pp | "
                     f"{bs['pct_improving']}% | {'✓' if bs['passed'] else '✗'} |")
            L.append("")

        ct = e3.get("cost") or {}
        if ct.get("grid"):
            L.append("### E3-4 成本敏感性")
            L.append("")
            L.append("基线与变体**在同一费率下各跑一次**, 否则差值里会混入基线自身的费率衰减。")
            L.append("")
            L.append("| 单边费率 | 基线CAGR | 变体CAGR | ΔCAGR | ΔUlcer | ΔSharpe | Δ调仓 |")
            L.append("|---|---|---|---|---|---|---|")
            for r in ct["grid"]:
                L.append(f"| {r['fee_bp']}bp | {r['base_cagr_pct']:.4f}% | "
                         f"{r['var_cagr_pct']:.4f}% | {r['d_cagr_pp']:+.3f}pp | "
                         f"{r['d_ulcer_pp']:+.4f}pp | {r['d_sharpe']:+.4f} | "
                         f"{r['d_turnover']:+d} |")
            L.append("")
            L.append(f"预设阈值: 直到 {ct['fee_max_bp']}bp(生产基准的 "
                     f"{ct['fee_max_bp'] / E3_FEE_BP[0]:.0f} 倍)仍需 ΔUlcer<0 且 "
                     f"ΔCAGR>{CAGR_TOL_PP}pp。结果: "
                     + ("✓" if ct.get("passed") else "**✗**"))
            L.append("")

        bdg = e3.get("branch_diag") or {}
        if bdg:
            th = bdg.get("trigger_halves") or {}
            L.append("### E3-5 分支诊断(探索性, **不入门禁**)")
            L.append("")
            L.append("“多防御”(极端净流出)与“少防御”(极端净流入)本是两个独立假说, "
                     "合跑时总效果可能是一支赚一支亏。拆开跑只为说清机制 —— 这是看到结果"
                     "后才做的拆解, 不得用它反过来挽救裁决。")
            L.append("")
            if th:
                L.append(f"触发分布: 多防御 {E3_SPLIT[:4]}前 "
                         f"{th['more_defense']['pre']} 次 / 后 "
                         f"{th['more_defense']['post']} 次; 少防御 前 "
                         f"{th['less_defense']['pre']} 次 / 后 "
                         f"{th['less_defense']['post']} 次。")
                L.append("")
            L.append("| 分支 | 区间 | ΔUlcer | ΔMaxDD | ΔCAGR | ΔSharpe |")
            L.append("|---|---|---|---|---|---|")
            for nm in ("both", "only_more_defense", "only_less_defense"):
                d = bdg.get(nm) or {}
                for seg in ("full", "pre_2019", "post_2019"):
                    r = d.get(seg) or {}
                    if r.get("d_ulcer_pp") is None:
                        continue
                    L.append(f"| {nm} | {seg} | {r['d_ulcer_pp']:+.4f}pp | "
                             f"{r['d_maxdd_pp']:+.4f}pp | "
                             f"{r['d_cagr_pp']:+.3f}pp | {r['d_sharpe']:+.4f} |")
            L.append("")

        os_ = e3.get("onesided_more_defense") or {}
        og = os_.get("gate") or {}
        if og:
            osp = os_.get("placebo") or {}
            oso = os_.get("split_oos") or {}
            osb = os_.get("bootstrap") or {}
            osc = os_.get("cost") or {}
            L.append("### E3-6 单边变体的同套门禁(事后选择, **不计入本次裁决**)")
            L.append("")
            L.append("拆开后只留“净流出→多防御”单边干预, 跑同一套四项门禁。"
                     f"这是看到分支诊断后才选定的假说, **已用掉自由度**"
                     f"({os_.get('dof_used')} 个分支里挑了最好的一个); 若按分支数做 "
                     f"Bonferroni 校正, 安慰剂阈值应收紧到 "
                     f"{os_.get('bonferroni_placebo_threshold_pct')}%。")
            L.append("")
            L.append("| 项 | 结果 | 通过 |")
            L.append("|---|---|---|")
            L.append(f"| 分期 OOS | 2019前 {oso.get('pre_2019', {}).get('d_ulcer_pp')}pp / "
                     f"2019后 {oso.get('post_2019', {}).get('d_ulcer_pp')}pp | "
                     f"{'✓' if oso.get('sign_consistent') else '✗'} |")
            L.append(f"| 块 bootstrap | CI [{osb.get('ci_lo_pp')}, "
                     f"{osb.get('ci_hi_pp')}]pp, 改善比例 "
                     f"{osb.get('pct_improving')}% | "
                     f"{'✓' if osb.get('passed') else '✗'} |")
            L.append(f"| 安慰剂 | 实测 {osp.get('observed_d_ulcer_pp')}pp, 优于实测 "
                     f"{osp.get('n_placebo_better')}/{osp.get('n_iter')} "
                     f"({osp.get('pct_placebo_better')}%), z={osp.get('z_vs_placebo')} | "
                     f"{'✓' if osp.get('passed') else '✗'} |")
            L.append(f"| 成本敏感性 | 至 {osc.get('fee_max_bp')}bp 仍改善 | "
                     f"{'✓' if osc.get('passed') else '✗'} |")
            L.append("")
            L.append(f"单边变体门禁: **{og.get('n_pass')}/{og.get('n_total')}**"
                     + (f" (未过: {', '.join(og.get('failed_items', []))})"
                        if og.get("failed_items") else " 全过")
                     + " —— 仅作为下一轮预注册的候选假说, 本次不据此改写裁决。")
            L.append("")

    L.append("## 结论")
    L.append("")
    L.append(f"- 实跑回测次数: {v['n_backtests']}")
    L.append(f"- 预注册假说显著项: {v['n_prereg_significant']} 个"
             + (f"（{', '.join(_sk(x) for x in v['prereg_significant'])}）"
                if v["prereg_significant"] else ""))
    L.append(f"- 主指标改善格子: {v['grid_improve_cells']}/{v['grid_total_cells']}")
    L.append("- 稳健改善轨: "
             + (", ".join(v["robust_tracks"]) if v["robust_tracks"] else "无"))
    L.append("- 仅孤立点改善: "
             + (", ".join(v["isolated_only"]) if v["isolated_only"] else "无"))
    if g3:
        L.append(f"- E3 门禁: {g3.get('n_pass')}/{g3.get('n_total')}"
                 + (f" (未过: {', '.join(g3.get('failed_items', []))})"
                    if g3.get("failed_items") else " 全过"))
    if na:
        L.append(f"- 轨 A 状态: {na['interpretation']}")
    L.append("")
    L.append(f"**{v['decision']} —— {v['case']}**")
    L.append("")
    L.append(v["next"])
    L.append("")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"  报告: {OUT_MD}")


def plot(res: dict, pool_flow: pd.Series | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        fh.fb.setup_font()
    except Exception:
        pass

    grid = pd.DataFrame(res["e2"]["grid"])
    fig, axes = plt.subplots(2, 3, figsize=(19, 9))

    ax = axes[0][0]
    if pool_flow is not None:
        s = pool_flow * 100
        ax.plot(s.index, s.values, lw=0.8, color="#2c3e50")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title("池内聚合净申赎率(%)  正=净申购")
        ax.grid(alpha=0.3)

    for pos, (tk, col, ttl) in enumerate(
            [("A", "d_sharpe", "轨A ΔSharpe (正=改善)"),
             ("B", "d_ulcer_pp", "轨B ΔUlcer pp (负=改善)")]):
        ax = axes[0][1] if pos == 0 else axes[0][2]
        g = grid[grid.track == tk]
        if not len(g):
            continue
        piv = g.pivot_table(index="param", columns="hold", values=col)
        cmap = "RdYlGn" if col == "d_sharpe" else "RdYlGn_r"
        im = ax.imshow(piv.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        ax.set_xlabel("hold(周)")
        ax.set_ylabel("param")
        ax.set_title(ttl)
        for a_ in range(piv.shape[0]):
            for b_ in range(piv.shape[1]):
                ax.text(b_, a_, f"{piv.values[a_, b_]:+.4f}",
                        ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1][0]
    if len(grid):
        for tk, mk, c in (("A", "o", "#2980b9"), ("B", "s", "#c0392b")):
            g = grid[grid.track == tk]
            if len(g):
                ax.scatter(g.d_cagr_pp, g.d_sharpe, marker=mk, s=34,
                           alpha=0.8, c=c, label=f"轨{tk}")
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("ΔCAGR (pp)")
        ax.set_ylabel("ΔSharpe")
        ax.set_title("全网格权衡: 右上=双赢")
        ax.legend()
        ax.grid(alpha=0.3)

    e3 = res.get("e3") or {}
    pl = e3.get("placebo") or {}
    ax = axes[1][1]
    if pl and not pl.get("insufficient"):
        smp = pl.get("samples_pp") or []
        if smp:
            ax.hist(smp, bins=18, color="#95a5a6", edgecolor="white")
        ax.axvline(pl["observed_d_ulcer_pp"], color="#c0392b", lw=2,
                   label=f"实测 {pl['observed_d_ulcer_pp']:+.4f}pp")
        ax.axvline(pl["placebo_mean_pp"], color="gray", ls="--", lw=1.2,
                   label=f"安慰剂均值 {pl['placebo_mean_pp']:+.4f}pp")
        ax.axvline(pl["placebo_p05_pp"], color="#7f8c8d", ls=":", lw=1.2,
                   label=f"安慰剂 p05 {pl['placebo_p05_pp']:+.4f}pp")
        ax.set_xlabel("ΔUlcer (pp, 负=改善)")
        ax.set_title(f"安慰剂噪声带 n={pl['n_iter']}  "
                     f"优于实测 {pl['pct_placebo_better']}%")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    so = e3.get("split_oos") or {}
    ax = axes[1][2]
    keys = [k for k in ("full", "pre_2019", "post_2019")
            if (so.get(k) or {}).get("d_ulcer_pp") is not None]
    if keys:
        vals = [so[k]["d_ulcer_pp"] for k in keys]
        ax.bar(range(len(keys)), vals,
               color=["#27ae60" if x < 0 else "#c0392b" for x in vals])
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys)
        ax.axhline(0, color="gray", lw=0.8)
        for i, x in enumerate(vals):
            ax.text(i, x, f"{x:+.4f}", ha="center",
                    va="bottom" if x >= 0 else "top", fontsize=8)
        ax.set_ylabel("ΔUlcer (pp)")
        ax.set_title("分期 OOS: 两半样本符号是否一致")
        ax.grid(alpha=0.3)

    v = res["verdict"]
    fig.suptitle(f"ETF 份额脉冲因子 —— {v['decision']}: {v['case']}", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)
    plt.close(fig)
    print(f"  图: {OUT_PNG}")


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-only", action="store_true")
    args = ap.parse_args()

    if args.render_only:
        res = json.loads(OUT_JSON.read_text())
        res["verdict"] = recompute_verdict(res)
        OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=2))
        render(res)
        print(f"裁决: {res['verdict']['decision']} — {res['verdict']['case']}")
        return

    print("[1/5] E0 份额数据体检")
    weekly, _cfg = fh.fb.load_weekly()
    panel, health, healthy = load_share_weekly(weekly.index)
    for k, h in health.items():
        print(f"  {k:12s} 停滞{h['daily_stagnation_2020plus_pct']:5.1f}% "
              f"冻结{h['longest_frozen_days']:4d}天 健康={h['healthy']}"
              f"{'' if h['healthy'] else ' <- ' + str(h['reject_reason'])}")
    if not healthy:
        print("无健康份额序列, 中止")
        sys.exit(1)
    print(f"  健康子集: {healthy}")

    print("[2/5] 构造脉冲信号 + 目标")
    sigs = build_pulse_signals(panel, weekly, healthy)
    base = run_baseline()
    tg = fh.fb.build_targets(weekly, base.nav_series["nav"])

    print("[3/5] E1' 事件研究")
    e1 = run_e1(sigs, tg, healthy)
    print(f"  {len(e1)} 个组合")

    print("[4/5] E2 双轨集成回测(无条件实跑)")
    e2 = e2_run_all(sigs, healthy)
    rob = e2_robust(e2["grid"])

    print("[5/5] E3 分期 OOS / 安慰剂 / 块 bootstrap / 成本敏感性")
    e3 = {}
    if "B" in rob:
        bd = float(rob["B"]["best_param"])
        bh = int(rob["B"]["best_hold"])
        fl_out, _c1 = pulse_and_coverage(sigs["pool_net_flow"], "low")
        fl_in, _c2 = pulse_and_coverage(sigs["pool_net_flow"], "high")
        dates = week_dates()
        base_res = run_baseline()
        var_res, _au = run_track_b(bd, bh, fl_out, fl_in, dates)
        bm, vm = fh.metrics_of(base_res), fh.metrics_of(var_res)
        obs = round(vm["ulcer_pct"] - bm["ulcer_pct"], 4)
        n_out, n_in = int(fl_out.sum()), int(fl_in.sum())
        print(f"  最优参数 delta={bd} hold={bh}  实测 ΔUlcer={obs:+.4f}pp", flush=True)
        e3["best_delta"], e3["best_hold"] = bd, bh
        e3["observed_d_ulcer_pp"] = obs
        e3["split_oos"] = e3_split_oos(base_res, var_res)
        print(f"  E3-1 分期 OOS: 符号一致={e3['split_oos']['sign_consistent']}",
              flush=True)
        e3["bootstrap"] = e3_bootstrap(base_res, var_res)
        print(f"  E3-3 bootstrap CI=[{e3['bootstrap'].get('ci_lo_pp')}, "
              f"{e3['bootstrap'].get('ci_hi_pp')}]", flush=True)
        e3["cost"] = e3_cost(bd, bh, fl_out, fl_in, dates)
        e3["placebo"] = e3_placebo(bd, bh, n_out, n_in, fl_out.index, dates,
                                   bm["ulcer_pct"], obs)
        e3["branch_diag"] = e3_branch_diag(bd, bh, fl_out, fl_in, dates,
                                           base_res)
        e3["onesided_more_defense"] = e3_onesided(bd, bh, fl_out, dates,
                                                  base_res, bm["ulcer_pct"])
        e3["n_backtests"] = int(2 + 2 * len(E3_FEE_BP)
                                + int(e3["placebo"].get("n_iter", 0))
                                + int(e3["branch_diag"].get("n_runs", 0))
                                + int(e3["onesided_more_defense"]
                                      .get("n_backtests", 0)))
        e3["gate"] = e3_gate(e3)
        print(f"  E3 门禁: {e3['gate']['n_pass']}/{e3['gate']['n_total']} "
              f"未过={e3['gate']['failed_items']}", flush=True)
        og = e3["onesided_more_defense"]["gate"]
        print(f"  单边(仅多防御, 事后选择) 门禁: "
              f"{og['n_pass']}/{og['n_total']} 未过={og['failed_items']}",
              flush=True)

    res = {
        "meta": {"lag_steps": LAG_STEPS, "roll_win": ROLL_WIN,
                 "q_pulse": Q_PULSE, "min_events": MIN_EVENTS,
                 "min_year_span": MIN_YEAR_SPAN,
                 "track_a_w": list(TRACK_A_W),
                 "track_b_delta": list(TRACK_B_DELTA),
                 "hold_grid": list(HOLD_GRID),
                 "n_weeks": int(len(weekly))},
        "e0_health": health, "healthy": healthy,
        "e1": e1, "e2": e2, "e2_robust": rob, "e3": e3,
    }
    res["verdict"] = recompute_verdict(res)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    render(res)
    plot(res, sigs["pool_net_flow"])
    print()
    print(f"裁决: {res['verdict']['decision']} — {res['verdict']['case']}")
    print(f"下一步: {res['verdict']['next']}")


if __name__ == "__main__":
    main()
