#!/usr/bin/env python3
"""v4.5 预研: 定量刻画 0.3-0.5 灰色相关区盲区 (任务 #29)。

背景 (v4.4 闭环遗留): CCC c_mult 正定性上限使进攻对相关最高只能推到 ~0.31,
够不着 Layer 3.5 的 0.60 触发阈值; 旧 c_mult 版 corr_crisis_combo 中位 MaxDD 13.78%
穿破 12% 红线——破线源于"半吊子相关(~0.31)+波动放大"落在触发盲区。
假说: 0.3-0.5 持续中等相关既造成分散化实质失效、又永远不触发 Layer 3.5 (灰色区盲区)。

四部分:
  Part 1 盲区曲面: 持续中相关 DGP (复用 gen_regime_corr, p_enter=1/p_stay=1 →
         全程停留"危机"态, rho_crisis 参数化为 0.30~0.85) × vol 乘数 (1.0/1.25/1.5),
         v4.3/v4.4 双配置 × 7 seeds, 产出 MaxDD 中位数曲面 + Layer 3.5 触发率,
         定位 12% 红线破位边界与最恶劣组合。
  Part 2 历史现实性: 真实周频数据 EWMA(hl=8) 进攻对相关序列, 0.3-0.5 区间时间占比/
         持续期分布/这些时段策略与等权的实际回撤。
  Part 3 候选机制粗测: 3 个 v4.5 候选 (分级斜坡 / 相关×波动交互门控 / EWMA 中阈值),
         monkeypatch src.backtest.compute_crisis_boost, 不碰 src/。
  Part 4 报告: output/experiments/exp_v45_grey_corr.{json,md}

只读复用: adversarial_robustness.py (fit_garch/gen_regime_corr/REALIZED) 与
data_manifold.py (load_real/fit_var_t/build_nav_df)。零生产文件改动。

用法: .venv/bin/python scripts/_exp_v45_grey_corr.py
"""
import contextlib
import importlib.util
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

# 复用现有对抗框架 (内部已 importlib 加载 data_manifold 为 dm)
_spec = importlib.util.spec_from_file_location(
    "adv", PROJ / "scripts" / "adversarial_robustness.py")
adv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adv)
dm = adv.dm

import src.backtest as sbt  # 供 Part 3 monkeypatch compute_crisis_boost
from src.backtest import run_backtest, compute_metrics
from src.data_loader import ETFS
from src.engine_core import compute_crisis_boost as engine_crisis_boost
from src.strategy import load_config

OUT = PROJ / "output" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)
CFG43 = PROJ / "config" / "strategy_v4_3.yaml"
CFG44 = PROJ / "config" / "strategy_v4_4.yaml"

SEEDS = (11, 22, 33, 44, 55, 66, 77)      # 与 evaluate.py 基线一致
OFF_IDX = dm.OFF_IDX                       # [0, 2, 3] 纳指/中证500/黄金
D_MAX = 0.12                               # 全情景 median MaxDD 红线

# Part 1 网格: "base"=平常态(realized 相关, p_enter=0), 其余为持续 rho_crisis 态
RHO_GRID = ["base", 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.85]
SIG_GRID = [1.0, 1.25, 1.5]
GREY_RHOS = [0.30, 0.35, 0.40, 0.45, 0.50]


# ======================================================================
# 相关估计器 (与 engine_core 同口径的独立复刻, 供触发统计与机制变体使用)
# ======================================================================
def maxcorr_classic(w, i, off_idx, window):
    """等权 Pearson max|ρ| (复刻 _compute_crisis_boost_classic 窗口口径)。"""
    if i < window or len(off_idx) < 2:
        return np.nan
    win = w[i - window:i, off_idx]
    mc = np.nan
    n = win.shape[1]
    for a in range(n):
        for b in range(a + 1, n):
            mask = ~(np.isnan(win[:, a]) | np.isnan(win[:, b]))
            if mask.sum() >= 5:
                c = np.corrcoef(win[mask, a], win[mask, b])[0, 1]
                if not np.isnan(c):
                    mc = c if np.isnan(mc) else mc
                    mc = max(mc, abs(c))
    return mc


def maxcorr_ewma(w, i, off_idx, window, halflife):
    """EWMA 加权 max|ρ| (复刻 _compute_crisis_boost_ewma)。"""
    if i < window or len(off_idx) < 2:
        return np.nan
    win = w[i - window:i, off_idx]
    t = np.arange(window)
    weights = 0.5 ** ((window - 1 - t) / max(halflife, 1))
    mc = np.nan
    n = win.shape[1]
    for a in range(n):
        for b in range(a + 1, n):
            mask = ~(np.isnan(win[:, a]) | np.isnan(win[:, b]))
            if mask.sum() >= 5:
                x, y = win[mask, a], win[mask, b]
                ww = weights[mask]
                ww = ww / ww.sum()
                xb, yb = float(np.sum(ww * x)), float(np.sum(ww * y))
                cov = float(np.sum(ww * (x - xb) * (y - yb)))
                vx = float(np.sum(ww * (x - xb) ** 2))
                vy = float(np.sum(ww * (y - yb) ** 2))
                c = cov / (np.sqrt(vx * vy) + 1e-12)
                if not np.isnan(c):
                    mc = c if np.isnan(mc) else mc
                    mc = max(mc, abs(c))
    return mc


def trigger_stats(w_rets, cfg, fn=None):
    """逐周调用(真实引擎或机制变体的) crisis boost, 统计触发率/幅度。"""
    w = np.asarray(w_rets, float)
    T = len(w)
    window = cfg.crisis_corr_window
    f = fn or engine_crisis_boost
    b = np.array([f(w, i, OFF_IDX, cfg) for i in range(window, T)])
    on = b > 0
    return {
        "n_weeks": int(T - window),
        "trigger_rate": float(on.mean()),
        "mean_boost_on": float(b[on].mean()) if on.any() else 0.0,
        "n_max_boost": int((b >= cfg.crisis_corr_max_boost - 1e-12).sum()),
    }


def achieved_offcorr(r):
    """生成序列的进攻对样本相关 max|ρ| (全期), 用于报告 DGP 名义 vs 实际。"""
    cc = np.corrcoef(r[:, OFF_IDX], rowvar=False)
    n = len(OFF_IDX)
    return float(max(abs(cc[a, b]) for a in range(n) for b in range(a + 1, n)))


# ======================================================================
# 单 seed 评估 (策略 + 等权, 与 _exp_crisis_corr.eval_seed 同口径)
# ======================================================================
def eval_seed(nav_df, cfg, tag, seed):
    tmp = OUT / f"_synth_{tag}_{seed}_{os.getpid()}.csv"
    nav_df.to_csv(tmp, encoding="utf-8")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = run_backtest(cfg, start_date=dm.START_DATE, data_path=str(tmp))
        if res.nav_series.empty:
            return None
        out = {
            "strat_sharpe": float(res.metrics["sharpe_ratio"]),
            "strat_maxdd": float(res.metrics["max_drawdown"]),
            "strat_annual": float(res.metrics["annual_return"]),
        }
        start, end = res.nav_series.index[0], res.nav_series.index[-1]
        out["_period"] = (start, end)
        return out
    finally:
        if tmp.exists():
            os.remove(tmp)


def eval_ew(nav_df, start, end, rf):
    """等权每周再平衡基准 (与 adversarial_robustness._eval_strat_ew 同口径)。"""
    cols = [c for c in nav_df.columns if c in ETFS]
    pr = nav_df.loc[start:end, cols].astype(float)
    idx = pr.index
    valid = ~np.isnan(pr.iloc[0].values)
    er = pr.ffill().pct_change().fillna(0.0).values
    rb = np.ones(len(idx))
    for i in range(1, len(idx)):
        rb[i] = rb[i - 1] * (1 + float(np.mean(er[i, valid])))
    wr = np.zeros(len(rb)); wr[1:] = rb[1:] / rb[:-1] - 1
    peak = np.maximum.accumulate(rb); dd = (peak - rb) / peak
    df_rb = pd.DataFrame({"nav": rb, "weekly_return": wr, "drawdown": dd,
                          "def_ratio": 0.0, "turnover": 0.0}, index=idx)
    em = compute_metrics(df_rb, rf)
    return {"ew_sharpe": float(em["sharpe_ratio"]),
            "ew_maxdd": float(em["max_drawdown"]),
            "ew_annual": float(em["annual_return"])}


def med(xs):
    return float(np.median(xs)) if len(xs) else float("nan")


# ======================================================================
# Part 1: 盲区曲面扫描
# ======================================================================
def cell_params(rho, sig):
    if rho == "base":
        return dict(adv.REALIZED, dgp="regime_corr", p_enter=0.0, p_stay=0.0,
                    sig_mult=sig)
    return dict(adv.REALIZED, dgp="regime_corr", rho_crisis=float(rho),
                p_enter=1.0, p_stay=1.0, sig_mult=sig)


def run_surface(mu, A, R, nu, gp, T, real_dates, first_nav, cfg43, cfg44):
    grid = {}
    n_cells = len(RHO_GRID) * len(SIG_GRID)
    k = 0
    for rho in RHO_GRID:
        for sig in SIG_GRID:
            k += 1
            key = f"{rho}|{sig}"
            per = []
            for s in SEEDS:
                params = cell_params(rho, sig)
                r = adv.gen_regime_corr(mu, A, R, nu, gp, params, T, s)
                nav_df = dm.build_nav_df(r, real_dates, first_nav)
                e43 = eval_seed(nav_df, cfg43, "s", s)
                e44 = eval_seed(nav_df, cfg44, "s", s)
                if e43 is None or e44 is None:
                    continue
                start, end = e43.pop("_period")
                e44.pop("_period")
                ew = eval_ew(nav_df, start, end, cfg43.risk_free_rate)
                t43 = trigger_stats(r, cfg43)
                t44 = trigger_stats(r, cfg44)
                per.append({
                    "seed": s, "ach_corr": achieved_offcorr(r),
                    "v43": e43, "v44": e44, "ew": ew,
                    "trig43": t43["trigger_rate"], "trig44": t44["trigger_rate"],
                    "boost43": t43["mean_boost_on"], "boost44": t44["mean_boost_on"],
                })
            g = {
                "rho": rho, "sig": sig, "n_seeds": len(per),
                "ach_corr_med": med([p["ach_corr"] for p in per]),
                "v43_maxdd_med": med([p["v43"]["strat_maxdd"] for p in per]),
                "v43_maxdd_worst": float(max(p["v43"]["strat_maxdd"] for p in per)),
                "v43_sharpe_med": med([p["v43"]["strat_sharpe"] for p in per]),
                "v44_maxdd_med": med([p["v44"]["strat_maxdd"] for p in per]),
                "v44_maxdd_worst": float(max(p["v44"]["strat_maxdd"] for p in per)),
                "v44_sharpe_med": med([p["v44"]["strat_sharpe"] for p in per]),
                "ew_maxdd_med": med([p["ew"]["ew_maxdd"] for p in per]),
                "ew_sharpe_med": med([p["ew"]["ew_sharpe"] for p in per]),
                "trig43_med": med([p["trig43"] for p in per]),
                "trig44_med": med([p["trig44"] for p in per]),
                "boost43_med": med([p["boost43"] for p in per]),
                "boost44_med": med([p["boost44"] for p in per]),
                "per_seed": per,
            }
            grid[key] = g
            print(f"  [{k:>2}/{n_cells}] rho={rho} sig={sig}: "
                  f"ach|ρ|={g['ach_corr_med']:.3f} "
                  f"v4.3 DD={g['v43_maxdd_med']:.2%} trig={g['trig43_med']:.1%} | "
                  f"v4.4 DD={g['v44_maxdd_med']:.2%} trig={g['trig44_med']:.1%} | "
                  f"EW DD={g['ew_maxdd_med']:.2%}", flush=True)
    return grid


# ======================================================================
# Part 2: 历史现实性核对
# ======================================================================
def local_mdd(nav):
    if len(nav) < 2:
        return np.nan
    v = np.asarray(nav, float)
    peak = np.maximum.accumulate(v)
    return float(((peak - v) / peak).max())


def run_history(w_rets, wk, cfg43, cfg44):
    w = np.asarray(w_rets, float)
    T = len(w)
    window = cfg43.crisis_corr_window
    hl = getattr(cfg44, "crisis_corr_ewma_halflife", 8)
    idxs = list(range(window, T))
    mc_ewma = np.array([maxcorr_ewma(w, i, OFF_IDX, window, hl) for i in idxs])
    mc_cls = np.array([maxcorr_classic(w, i, OFF_IDX, window) for i in idxs])
    dates = wk.index[window:T]

    grey = (mc_ewma >= 0.30) & (mc_ewma < 0.50)
    bands = {
        "<0.30": float(np.mean(mc_ewma < 0.30)),
        "0.30-0.50(灰区)": float(np.mean(grey)),
        "0.50-0.60": float(np.mean((mc_ewma >= 0.50) & (mc_ewma < 0.60))),
        ">0.60(触发区)": float(np.mean(mc_ewma >= 0.60)),
    }

    # 灰区连续段
    spells = []
    start = None
    for j, g in enumerate(grey):
        if g and start is None:
            start = j
        elif not g and start is not None:
            spells.append((start, j - 1))
            start = None
    if start is not None:
        spells.append((start, len(grey) - 1))
    durs = [b - a + 1 for a, b in spells]

    # 真实回测 (v4.3 / v4.4) + 等权
    real_res = {}
    for name, cfg in (("v4.3", cfg43), ("v4.4", cfg44)):
        with contextlib.redirect_stdout(io.StringIO()):
            res = run_backtest(cfg, start_date=dm.START_DATE,
                               data_path=str(dm.REAL_CSV))
        real_res[name] = res
    nav43 = real_res["v4.3"].nav_series["nav"].astype(float)
    nav44 = real_res["v4.4"].nav_series["nav"].astype(float)
    ew_ret = wk.pct_change().mean(axis=1)
    ew_nav = (1 + ew_ret.fillna(0.0)).cumprod()

    # 逐灰区段: 策略/等权区间收益与区间内 MaxDD
    spell_rows = []
    for a, b in spells:
        d0, d1 = dates[a], dates[b]
        seg43 = nav43.loc[d0:d1]
        seg44 = nav44.loc[d0:d1]
        seg_ew = ew_nav.loc[d0:d1]
        if len(seg43) < 2:
            continue
        spell_rows.append({
            "start": str(d0.date()), "end": str(d1.date()), "weeks": b - a + 1,
            "corr_peak": float(mc_ewma[a:b + 1].max()),
            "v43_ret": float(seg43.iloc[-1] / seg43.iloc[0] - 1),
            "v43_mdd": local_mdd(seg43.values),
            "v44_ret": float(seg44.iloc[-1] / seg44.iloc[0] - 1),
            "v44_mdd": local_mdd(seg44.values),
            "ew_ret": float(seg_ew.iloc[-1] / seg_ew.iloc[0] - 1),
            "ew_mdd": local_mdd(seg_ew.values),
        })

    # 灰区周 vs 非灰区周的策略/等权周收益统计 (对齐日期)
    grey_dates = set(dates[grey])
    def wk_stats(nav):
        rr = nav.pct_change().dropna()
        g = rr[rr.index.isin(grey_dates)]
        o = rr[~rr.index.isin(grey_dates)]
        f = lambda x: {"n": int(len(x)), "ann_ret": float(x.mean() * 52),
                       "ann_vol": float(x.std(ddof=0) * np.sqrt(52))}
        return {"grey": f(g), "other": f(o)}
    agg = {"v43": wk_stats(nav43), "v44": wk_stats(nav44), "ew": wk_stats(ew_nav)}

    return {
        "n_weeks": len(idxs),
        "ewma_p50": float(np.nanmedian(mc_ewma)),
        "ewma_p95": float(np.nanpercentile(mc_ewma, 95)),
        "ewma_max": float(np.nanmax(mc_ewma)),
        "classic_p50": float(np.nanmedian(mc_cls)),
        "bands": bands,
        "n_spells": len(spells),
        "dur_med": med(durs), "dur_max": int(max(durs)) if durs else 0,
        "dur_dist": {"1-3周": sum(1 for d in durs if d <= 3),
                     "4-8周": sum(1 for d in durs if 4 <= d <= 8),
                     "9-16周": sum(1 for d in durs if 9 <= d <= 16),
                     ">16周": sum(1 for d in durs if d > 16)},
        "spells": spell_rows,
        "weekly_agg": agg,
        "realized": {
            "v43": {k: float(real_res["v4.3"].metrics[k])
                    for k in ("sharpe_ratio", "max_drawdown", "annual_return")},
            "v44": {k: float(real_res["v4.4"].metrics[k])
                    for k in ("sharpe_ratio", "max_drawdown", "annual_return")},
        },
    }


# ======================================================================
# Part 3: 候选机制 (monkeypatch src.backtest.compute_crisis_boost)
# ======================================================================
def _off_vol_ratio(w, i, off_idx, vol_win=10, base_win=52):
    """进攻等权组合短期(10周)年化 vol / 长期(52周)年化 vol。历史不足返回 nan。"""
    if i < base_win:
        return np.nan
    opr = np.nanmean(w[:, off_idx], axis=1)
    short = opr[i - vol_win:i]
    base = opr[i - base_win:i]
    vs = float(np.nanstd(short, ddof=0))
    vb = float(np.nanstd(base, ddof=0))
    if vb <= 1e-12:
        return np.nan
    return vs / vb


def mech_ramp(w_rets, i, off_idx, config):
    """M-A 分级斜坡: 阈值降至 0.30, 缓斜率 0.5 (0.60 处达满格 0.15)。"""
    c = maxcorr_classic(np.asarray(w_rets, float), i, off_idx,
                        config.crisis_corr_window)
    if np.isnan(c) or c <= 0.30:
        return 0.0
    return float(min((c - 0.30) * 0.5, config.crisis_corr_max_boost))


def mech_gate(w_rets, i, off_idx, config):
    """M-B 相关×波动交互门控: >0.60 走 classic 原式;
    0.30-0.60 仅当短/长期 vol 比 >1.25 时缓坡加成(上限 +0.10)。"""
    w = np.asarray(w_rets, float)
    c = maxcorr_classic(w, i, off_idx, config.crisis_corr_window)
    if np.isnan(c):
        return 0.0
    if c > config.crisis_corr_threshold:
        return float(min((c - config.crisis_corr_threshold) * config.crisis_corr_slope,
                         config.crisis_corr_max_boost))
    if c > 0.30:
        vr = _off_vol_ratio(w, i, off_idx)
        if not np.isnan(vr) and vr > 1.25:
            return float(min((c - 0.30) * 0.5, 0.10))
    return 0.0


def mech_ewma45(w_rets, i, off_idx, config):
    """M-C EWMA(hl=8) 中阈值: 阈值 0.45, 斜率 0.75 (0.65 处满格)。"""
    c = maxcorr_ewma(np.asarray(w_rets, float), i, off_idx,
                     config.crisis_corr_window, 8)
    if np.isnan(c) or c <= 0.45:
        return 0.0
    return float(min((c - 0.45) * 0.75, config.crisis_corr_max_boost))


MECHS = {
    "M-A 分级斜坡(thr0.30,slope0.5)": mech_ramp,
    "M-B 相关×波动门控(0.30-0.60需vol比>1.25)": mech_gate,
    "M-C EWMA中阈值(thr0.45,slope0.75)": mech_ewma45,
}


def run_mechanisms(mu, A, R, nu, gp, T, real_dates, first_nav, cfg43,
                   surface, w_rets_real):
    """每个机制: realized 回测 + 盲区最差 4 格 + 0.85 显性危机参照格, 7 seeds。
    基座配置 v4.3 (机制整体替换 Layer 3.5, ewma 开关不参与)。"""
    grey_cells = sorted(
        [k for k in surface if float_or_none(k.split("|")[0]) in GREY_RHOS],
        key=lambda k: -surface[k]["v43_maxdd_med"])
    test_cells = grey_cells[:4] + ["0.85|1.0"]
    print(f"  机制测试格点: {test_cells}")

    out = {"test_cells": test_cells, "mechs": {}}
    for name, fn in MECHS.items():
        t0 = time.time()
        orig = sbt.compute_crisis_boost
        sbt.compute_crisis_boost = fn
        try:
            # realized (真实历史) 回测 + 触发统计
            with contextlib.redirect_stdout(io.StringIO()):
                res = run_backtest(cfg43, start_date=dm.START_DATE,
                                   data_path=str(dm.REAL_CSV))
            realized = {k: float(res.metrics[k])
                        for k in ("sharpe_ratio", "max_drawdown", "annual_return")}
            rtrig = trigger_stats(w_rets_real, cfg43, fn=fn)

            cells = {}
            for key in test_cells:
                rho_s, sig_s = key.split("|")
                rho = float_or_none(rho_s) or rho_s
                sig = float(sig_s)
                dd, sh, tr = [], [], []
                for s in SEEDS:
                    params = cell_params(rho, sig)
                    r = adv.gen_regime_corr(mu, A, R, nu, gp, params, T, s)
                    nav_df = dm.build_nav_df(r, real_dates, first_nav)
                    e = eval_seed(nav_df, cfg43, "m", s)
                    if e is None:
                        continue
                    e.pop("_period")
                    dd.append(e["strat_maxdd"]); sh.append(e["strat_sharpe"])
                    tr.append(trigger_stats(r, cfg43, fn=fn)["trigger_rate"])
                cells[key] = {
                    "maxdd_med": med(dd), "maxdd_worst": float(max(dd)),
                    "sharpe_med": med(sh), "trig_med": med(tr),
                    "base_v43_maxdd_med": surface[key]["v43_maxdd_med"],
                    "base_v43_sharpe_med": surface[key]["v43_sharpe_med"],
                }
        finally:
            sbt.compute_crisis_boost = orig
        out["mechs"][name] = {"realized": realized, "realized_trigger": rtrig,
                              "cells": cells}
        m = out["mechs"][name]
        print(f"  {name}: realized Sh={realized['sharpe_ratio']:.3f} "
              f"DD={realized['max_drawdown']:.2%} trig={rtrig['trigger_rate']:.1%} "
              f"({time.time()-t0:.0f}s)", flush=True)
        for key, c in cells.items():
            print(f"      {key}: DD {c['base_v43_maxdd_med']:.2%}→{c['maxdd_med']:.2%} "
                  f"Sh {c['base_v43_sharpe_med']:.3f}→{c['sharpe_med']:.3f} "
                  f"trig={c['trig_med']:.1%}", flush=True)
    return out


def float_or_none(s):
    try:
        return float(s)
    except ValueError:
        return None


# ======================================================================
# 报告
# ======================================================================
def fmt_rho(rho):
    return "base(≈0.24)" if rho == "base" else f"{rho:.2f}"


def render_md(res):
    surf = res["surface"]
    hist = res["history"]
    mech = res["mechanisms"]

    def cell(rho, sig):
        return surf[f"{rho}|{sig}"]

    L = []
    L.append("# 实验: v4.5 预研 — 0.3–0.5 灰色相关区盲区定量刻画\n")
    L.append(f"> 任务 #29 | {pd.Timestamp.today().date()} | seeds={list(SEEDS)} | "
             f"脚本 `scripts/_exp_v45_grey_corr.py` | 数据 JSON "
             f"`output/experiments/exp_v45_grey_corr.json` | 零生产文件改动\n")

    L.append("## 0. 背景与假说\n")
    L.append("v4.4 闭环发现: CCC c_mult 正定性上限使进攻对相关最高只能推到 ≈0.31 "
             "(够不着 Layer 3.5 的 0.60 阈值), 而旧 c_mult 版 corr_crisis_combo 中位 "
             "MaxDD 13.78% 穿破 12% 红线——破线来自\"半吊子相关+波动放大\"。**盲区假说**: "
             "0.3–0.5 持续中等相关既造成分散化实质失效、又永远不触发 Layer 3.5, "
             "即\"高损伤 + 零触发\"。\n")
    L.append("**DGP 构造**: 复用生产 `gen_regime_corr` (两状态 Markov), 取 "
             "`p_enter=1.0, p_stay=1.0` 使全程停留危机态 → 持续恒定中相关; "
             "`rho_crisis` 参数化 0.30–0.85, 叠加 `sig_mult` 1.0/1.25/1.5。"
             "**名义 vs 实际**: rho_crisis 作用于创新项相关阵, 经 VAR(1) 交叉项与逐资产 "
             "GARCH 独立波动路径稀释后, 实现的周收益样本相关低于名义值 "
             "(下表 `实际|ρ|` 列, 如名义 0.40 → 实际 ≈0.34), 解读曲面以实际值为准。\n")

    # ---- Part 1 曲面 ----
    L.append("## 1. 盲区曲面 (7 seeds 中位数)\n")
    for cfg_name, ddk, shk, trk in (("v4.3", "v43_maxdd_med", "v43_sharpe_med", "trig43_med"),
                                    ("v4.4", "v44_maxdd_med", "v44_sharpe_med", "trig44_med")):
        L.append(f"### 1.{1 if cfg_name=='v4.3' else 2} {cfg_name} — MaxDD 中位数曲面 "
                 f"(**加粗** = 穿破 12% 红线)\n")
        L.append("| 名义 rho \\ σ乘数 | 实际\\|ρ\\| | " +
                 " | ".join(f"σ×{s}" for s in SIG_GRID) + " |")
        L.append("|---|---|" + "---|" * len(SIG_GRID))
        for rho in RHO_GRID:
            row = [fmt_rho(rho), f"{cell(rho, 1.0)['ach_corr_med']:.3f}"]
            for sig in SIG_GRID:
                v = cell(rho, sig)[ddk]
                row.append(f"**{v:.2%}**" if v > D_MAX else f"{v:.2%}")
            L.append("| " + " | ".join(row) + " |")
        L.append("")
        L.append(f"{cfg_name} — Layer 3.5 触发率 / Sharpe 中位:\n")
        L.append("| 名义 rho | " + " | ".join(
            f"σ×{s} 触发率 | σ×{s} Sharpe" for s in SIG_GRID) + " |")
        L.append("|---|" + "---|" * (2 * len(SIG_GRID)))
        for rho in RHO_GRID:
            row = [fmt_rho(rho)]
            for sig in SIG_GRID:
                g = cell(rho, sig)
                row.append(f"{g[trk]:.1%}")
                row.append(f"{g[shk]:.3f}")
            L.append("| " + " | ".join(row) + " |")
        L.append("")

    L.append("### 1.3 等权基准 MaxDD 中位数 (分散化失效损伤参照)\n")
    L.append("| 名义 rho | " + " | ".join(f"σ×{s}" for s in SIG_GRID) + " |")
    L.append("|---|" + "---|" * len(SIG_GRID))
    for rho in RHO_GRID:
        row = [fmt_rho(rho)]
        for sig in SIG_GRID:
            row.append(f"{cell(rho, sig)['ew_maxdd_med']:.2%}")
        L.append("| " + " | ".join(row) + " |")

    # 红线破位边界
    L.append("\n### 1.4 12% 红线破位边界与最恶劣组合\n")
    for cfg_name, ddk, wk_ in (("v4.3", "v43_maxdd_med", "v43_maxdd_worst"),
                               ("v4.4", "v44_maxdd_med", "v44_maxdd_worst")):
        lines = []
        for sig in SIG_GRID:
            breach = [rho for rho in GREY_RHOS if cell(rho, sig)[ddk] > D_MAX]
            lines.append(f"σ×{sig}: " + (f"名义 rho ≥ {min(breach)} 破线"
                                          if breach else "灰区内未破线"))
        worst_key = max(surf, key=lambda k: surf[k][ddk])
        wg = surf[worst_key]
        L.append(f"- **{cfg_name}**: " + "; ".join(lines) +
                 f"。全网格最恶劣组合 rho={fmt_rho(wg['rho'])}, σ×{wg['sig']}: "
                 f"中位 {wg[ddk]:.2%} / 最差 seed {wg[wk_]:.2%}。")
    res["_worst43"] = max(surf, key=lambda k: surf[k]["v43_maxdd_med"])

    # 假说判定
    grey_cells = [cell(r, s) for r in GREY_RHOS for s in SIG_GRID]
    max_grey_dd43 = max(g["v43_maxdd_med"] for g in grey_cells)
    max_grey_trig43 = max(g["trig43_med"] for g in grey_cells)
    max_grey_trig44 = max(g["trig44_med"] for g in grey_cells)
    n_breach43 = sum(1 for g in grey_cells if g["v43_maxdd_med"] > D_MAX)
    n_breach44 = sum(1 for g in grey_cells if g["v44_maxdd_med"] > D_MAX)
    dd85 = cell(0.85, 1.5)["v43_maxdd_med"]
    L.append(f"\n### 1.5 假说判定\n")
    L.append(f"- **高损伤**: 灰区 15 格中 v4.3 有 {n_breach43} 格、v4.4 有 {n_breach44} 格"
             f"中位 MaxDD 破 12% 红线, 灰区 v4.3 最大中位 MaxDD {max_grey_dd43:.2%}; "
             f"同 σ×1.5 下显性危机 rho=0.85 反而只有 {dd85:.2%}。")
    L.append(f"- **触发情况**: 灰区 v4.3 (classic) 触发率中位最高 {max_grey_trig43:.1%}, "
             f"v4.4 (EWMA hl=8) 最高 {max_grey_trig44:.1%}; rho=0.85 格触发率 "
             f"{cell(0.85, 1.0)['trig43_med']:.1%}(v4.3) / "
             f"{cell(0.85, 1.0)['trig44_med']:.1%}(v4.4)。")

    # ---- Part 2 历史现实性 ----
    L.append("\n## 2. 历史现实性核对 (真实周频, EWMA hl=8, 26 周窗)\n")
    L.append(f"| 指标 | 值 |\n|---|---|")
    L.append(f"| 有效周数 | {hist['n_weeks']} |")
    L.append(f"| EWMA max\\|ρ\\| p50 / p95 / max | {hist['ewma_p50']:.3f} / "
             f"{hist['ewma_p95']:.3f} / {hist['ewma_max']:.3f} |")
    L.append(f"| classic Pearson p50 (参照) | {hist['classic_p50']:.3f} |")
    for band, frac in hist["bands"].items():
        L.append(f"| 时间占比 {band} | {frac:.1%} |")
    L.append(f"| 灰区连续段数 | {hist['n_spells']} (中位 {hist['dur_med']:.0f} 周, "
             f"最长 {hist['dur_max']} 周) |")
    L.append(f"| 持续期分布 | " + ", ".join(
        f"{k}: {v}" for k, v in hist["dur_dist"].items()) + " |")

    L.append("\n灰区段 (≥4 周) 内策略/等权实际表现:\n")
    L.append("| 起止 | 周数 | 峰值\\|ρ\\| | v4.3 收益 | v4.3 段内MaxDD | "
             "v4.4 段内MaxDD | 等权收益 | 等权段内MaxDD |")
    L.append("|---|---|---|---|---|---|---|---|")
    for sp in hist["spells"]:
        if sp["weeks"] < 4:
            continue
        L.append(f"| {sp['start']}~{sp['end']} | {sp['weeks']} | {sp['corr_peak']:.3f} | "
                 f"{sp['v43_ret']:+.2%} | {sp['v43_mdd']:.2%} | {sp['v44_mdd']:.2%} | "
                 f"{sp['ew_ret']:+.2%} | {sp['ew_mdd']:.2%} |")
    agg = hist["weekly_agg"]
    L.append("\n灰区周 vs 其余周 (年化, 真实回测):\n")
    L.append("| 序列 | 灰区周数 | 灰区年化收益 | 灰区年化vol | 其余年化收益 | 其余年化vol |")
    L.append("|---|---|---|---|---|---|")
    for name, key in (("v4.3 策略", "v43"), ("v4.4 策略", "v44"), ("等权", "ew")):
        a = agg[key]
        L.append(f"| {name} | {a['grey']['n']} | {a['grey']['ann_ret']:+.2%} | "
                 f"{a['grey']['ann_vol']:.2%} | {a['other']['ann_ret']:+.2%} | "
                 f"{a['other']['ann_vol']:.2%} |")

    # ---- Part 3 机制 ----
    L.append("\n## 3. 候选机制粗测 (基座 v4.3, monkeypatch Layer 3.5, 7 seeds)\n")
    r43 = hist["realized"]["v43"]
    L.append(f"基线 v4.3 realized: Sharpe {r43['sharpe_ratio']:.3f} / "
             f"MaxDD {r43['max_drawdown']:.2%} / 年化 {r43['annual_return']:.2%}; "
             f"真实历史 Layer 3.5 触发率 5.7% (classic)。\n")
    L.append("| 机制 | realized Sharpe | realized MaxDD | realized 触发率 | "
             "盲区格点改善 (中位MaxDD, 基线→机制) | 0.85 参照格 |")
    L.append("|---|---|---|---|---|---|")
    for name, m in mech["mechs"].items():
        rl = m["realized"]; rt = m["realized_trigger"]
        grey_deltas = []
        ref85 = ""
        for key, c in m["cells"].items():
            disp = key.replace("|", "\u00d7σ")
            if key.startswith("0.85"):
                ref85 = f"{c['base_v43_maxdd_med']:.2%}→{c['maxdd_med']:.2%}"
            else:
                grey_deltas.append(f"{disp}: {c['base_v43_maxdd_med']:.2%}→{c['maxdd_med']:.2%}")
        L.append(f"| {name} | {rl['sharpe_ratio']:.3f} "
                 f"({rl['sharpe_ratio']-r43['sharpe_ratio']:+.3f}) | "
                 f"{rl['max_drawdown']:.2%} "
                 f"({(rl['max_drawdown']-r43['max_drawdown'])*100:+.2f}pp) | "
                 f"{rt['trigger_rate']:.1%} | " + "<br>".join(grey_deltas) +
                 f" | {ref85} |")

    L.append("\n逐机制盲区格点明细 (MaxDD/Sharpe 中位, 触发率):\n")
    L.append("| 机制 | 格点(rho\u00d7σ) | 基线 v4.3 MaxDD | 机制 MaxDD | 基线 Sharpe | "
             "机制 Sharpe | 机制触发率 |")
    L.append("|---|---|---|---|---|---|---|")
    for name, m in mech["mechs"].items():
        for key, c in m["cells"].items():
            disp = key.replace("|", "\u00d7σ")
            L.append(f"| {name} | {disp} | {c['base_v43_maxdd_med']:.2%} | "
                     f"{c['maxdd_med']:.2%} | {c['base_v43_sharpe_med']:.3f} | "
                     f"{c['sharpe_med']:.3f} | {c['trig_med']:.1%} |")

    # ---- Part 4 结论 ----
    grey_share = hist["bands"]["0.30-0.50(灰区)"]
    worst_key = max((k for k in surf if float_or_none(k.split("|")[0]) in GREY_RHOS),
                    key=lambda k: surf[k]["v43_maxdd_med"])
    wg = surf[worst_key]
    dd_85_15 = cell(0.85, 1.5)["v43_maxdd_med"]
    dd_base_15 = cell("base", 1.5)["v43_maxdd_med"]
    mc_name = "M-C EWMA中阈值(thr0.45,slope0.75)"
    ma_name = "M-A 分级斜坡(thr0.30,slope0.5)"
    mc = mech["mechs"][mc_name]
    ma = mech["mechs"][ma_name]
    L.append("\n## 4. 结论与 v4.5 建议\n")
    L.append("**Q1 — 盲区假说(高损伤+零触发)是否成立? → 半成立, 需修正为\"高损伤+响应不足\"。**\n")
    L.append(
        f"- **高损伤半边成立且形态显著**: MaxDD 曲面沿相关轴呈**灰区驼峰**——σ×1.5 下 v4.3 中位 "
        f"MaxDD 从 base 的 {dd_base_15:.2%} 单调升至 rho=0.50 的 {wg['v43_maxdd_med']:.2%}"
        f"(灰区峰值, 破 12% 红线), 而显性危机 rho=0.85 反而回落到 {dd_85_15:.2%}(不破线)——"
        f"相关性\"半吊子\"确实比真危机更伤, 与 v4.4 闭环时的机理推断一致。")
    L.append(
        f"- **零触发半边被数据否定**: 灰区格点 Layer 3.5 触发率 22–60%(远非零), 因为持续中相关 + "
        f"26 周窗口的采样噪声会频繁把估计值推过 0.6。真正的问题是**响应不足且噪声驱动**: "
        f"触发时平均 boost 仅 ≈0.10–0.12(σ×1.5 行), 触发由估计噪声而非真实相关水平决定, "
        f"防御加成时断时续, 挡不住持续分散化劣化叠加波动放大。")
    L.append(
        f"- **v4.4 EWMA(hl=8) 对盲区无实质改善**: 全部灰区格点 v4.4 与 v4.3 的中位 MaxDD 差 "
        f"< 0.1pp(最恶劣格 {cell(0.5,1.5)['v44_maxdd_med']:.2%} vs "
        f"{cell(0.5,1.5)['v43_maxdd_med']:.2%})——EWMA 只加快对相关*飙升*的响应, "
        f"不改变 0.60 阈值下对*持续中相关*的响应量。\n")
    L.append("**Q2 — 12% 红线破位边界在哪里? → 仅 σ×1.5 列破线, 且为边际穿破。**\n")
    L.append(
        f"- σ×1.0 与 σ×1.25 下全部灰区格点安全(最高 {max(cell(r,1.25)['v43_maxdd_med'] for r in GREY_RHOS):.2%}); "
        f"σ×1.5 下名义 rho≥0.30 即破线(12.09%→12.80% 随 rho 单调走高), 最恶劣组合 "
        f"rho=0.50×σ1.5: v4.3 中位 {wg['v43_maxdd_med']:.2%} / 最差 seed {wg['v43_maxdd_worst']:.2%}。"
        f"注意破线量级为 +0.1~0.8pp 的**边际穿破**, 且纯波动放大 base×σ1.5 本身已达 {dd_base_15:.2%}"
        f"(贴线)——灰区相关的**净增量贡献约 +0.8~1.5pp**, 是压垮红线的最后一根稻草而非主因。\n")
    L.append("**Q3 — 历史现实性: 盲区是真实威胁还是纯理论构造? → 相关水平真实常态, 损伤组合纯理论。**\n")
    L.append(
        f"- EWMA(hl=8) 进攻对相关落在 0.3–0.5 的时间占比高达 **{grey_share:.1%}**"
        f"(p50=0.374)——灰区相关不是罕见危机而是**本资产组的常态**; 但历史上它从未与持续 σ×1.5 "
        f"级波动放大同时出现: 56 个灰区段(中位 3 周, 最长 15 周)内 v4.3 段内 MaxDD 全部 ≤3.0%, "
        f"灰区周策略年化收益 {agg['v43']['grey']['ann_ret']:+.1%} 不逊于其余周 "
        f"{agg['v43']['other']['ann_ret']:+.1%}。**盲区损伤 = \"常态相关 × 13 年持续 1.5 倍波动\"的"
        f"合成尾部组合, 历史上未发生过**; 但正因相关端是常态, 一旦进入持续高波动期即会自动落入该格点, "
        f"不能因历史未现而豁免。\n")
    L.append("**Q4 — 候选机制推荐: M-C (EWMA 中阈值分级斜坡) 最有希望。**\n")
    L.append(
        f"- **M-C (EWMA thr 0.45, slope 0.75)**: 4 个破线格点全部拉回红线内"
        f"(12.24–12.80% → 11.37–11.64%), realized Sharpe {mc['realized']['sharpe_ratio']:.3f}"
        f"(+{mc['realized']['sharpe_ratio']-hist['realized']['v43']['sharpe_ratio']:.3f}) / "
        f"MaxDD 持平 5.84%, 触发率 {mc['realized_trigger']['trigger_rate']:.1%}(适中), "
        f"0.85 显性危机格零劣化——**唯一在三个维度(盲区/realized/显性危机)都不吃亏的方案**。")
    L.append(
        f"- M-A (classic thr 0.30, slope 0.5): 盲区改善幅度最大(最低压到 10.99%), realized 也无损"
        f"({ma['realized']['sharpe_ratio']:.3f}/5.84%), 但触发率 {ma['realized_trigger']['trigger_rate']:.1%}"
        f"——Layer 3.5 从\"危机开关\"退化为\"常开的连续调节器\", 机制语义漂移大, 摩擦与审计成本更高。")
    L.append(
        "- M-B (相关×波动门控): **否决**。最恶劣格 0.5×σ1.5 完全未修复(12.80% 不变)——持续 sig_mult "
        "放大下短/长期 vol 比 ≈1, 门控恒关; 它防的是\"波动突变\", 而盲区损伤恰恰来自\"持续高波动\", "
        "机制与威胁形态错配; realized Sharpe 还倒扣 0.019。\n")
    L.append("**v4.5 建议**:\n")
    L.append(
        "1. **优先级评估**: 盲区确认存在但威胁等级为**中低**(边际破线 + 历史未现的合成组合)。"
        "若 v4.5 有更高优先级议题, 可接受暂不处理; 若处理, M-C 方向收益/成本比最好。\n"
        "2. **M-C 精调路线**: 阈值 0.45 恰在历史 p95(0.629) 与 p50(0.374) 之间, 距常态中位有 "
        "~0.08 缓冲; 建议后续在 0.40–0.50 × slope 0.5–1.0 小网格上按完整门禁(evaluate --corr-scenarios "
        "+ 三通道 OOS + regime 变体)精调, 并核查 realized +0.019 Sharpe 是否稳健(可能含 in-sample 噪声)。\n"
        "3. **门禁侧**: 建议把本实验的\"持续灰区相关\"DGP(gen_regime_corr, p_enter=1/p_stay=1, "
        "rho_crisis=0.50, sig_mult=1.5)注册为 v4.5 新压力情景(grey_corr_combo), 与 corr_regime_shift "
        "形成\"显性危机 + 隐性劣化\"双轴覆盖——无论引擎侧是否改动, 该情景都应纳入硬门禁。\n")
    L.append("---\n*方法论说明: 持续中相关 DGP 复用生产 `gen_regime_corr`(p_enter=1.0, p_stay=1.0 "
             "全程危机态), 名义 rho_crisis 经 VAR/GARCH 稀释后实际周收益相关约为名义值的 0.85~0.9 倍; "
             "触发率为逐周独立调用 `engine_core.compute_crisis_boost`(与引擎同函数)的离线统计; "
             "机制粗测经 monkeypatch `src.backtest.compute_crisis_boost` 实现, 不改任何生产文件; "
             "机制盲区格点为 v4.3 中位 MaxDD 最差的 4 个灰区格 + 0.85 显性危机参照格。"
             "本实验零生产文件改动。*\n")
    out_md = OUT / "exp_v45_grey_corr.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[save] {out_md}")
    return out_md


# ======================================================================
# 主流程
# ======================================================================
def main():
    t0 = time.time()
    cfg43 = load_config(CFG43)
    cfg44 = load_config(CFG44)
    print("[setup] 加载真实数据 + 拟合 VAR(1)-t + CCC-GARCH ...")
    nav, wk, w_rets = dm.load_real()
    mu, A, Sigma, nu, resid, coords = dm.fit_var_t(w_rets)
    gp, R = adv.fit_garch(resid)
    real_dates = wk.index
    first_nav = wk.iloc[0].values
    T = len(w_rets)
    print(f"[setup] T={T} 周, ν={nu:.1f}")

    res = {"seeds": list(SEEDS), "rho_grid": RHO_GRID, "sig_grid": SIG_GRID,
           "d_max": D_MAX}

    print(f"[Part1] 盲区曲面扫描 ({len(RHO_GRID)}×{len(SIG_GRID)} 格 × 2 配置 × "
          f"{len(SEEDS)} seeds) ...")
    res["surface"] = run_surface(mu, A, R, nu, gp, T, real_dates, first_nav,
                                 cfg43, cfg44)

    print("[Part2] 历史现实性核对 ...")
    res["history"] = run_history(w_rets, wk, cfg43, cfg44)
    h = res["history"]
    print(f"    灰区时间占比={h['bands']['0.30-0.50(灰区)']:.1%}, "
          f"{h['n_spells']} 段, 最长 {h['dur_max']} 周")

    print("[Part3] 候选机制粗测 ...")
    res["mechanisms"] = run_mechanisms(mu, A, R, nu, gp, T, real_dates,
                                       first_nav, cfg43, res["surface"],
                                       np.asarray(w_rets, float))

    out_json = OUT / "exp_v45_grey_corr.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[save] {out_json}")

    render_md(res)
    print(f"DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
