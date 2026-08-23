#!/usr/bin/env python3
"""E2: Layer 3.5 相关性口径 —— 全池 max vs 仅选中 TOP2。

研究问题
    生产 Layer 3.5 用全进攻池 (3 只) pairwise max EWMA |corr| 触发防御 boost,
    但策略每周只持有 TOP2。E1 实测: 全池 219 次触发中 122 次 (55.7%) 实际持仓对
    并不高相关 —— 过半 boost 由至少一只未持有资产驱动。改为只看选中 TOP2 是否更优?

    E1 结论是 NO-GO (两口径对前瞻风险均无预测力), 但 E1 的 y 是 4 周前瞻进攻端
    风险代理, 而 Layer 3.5 是经回测 Sharpe/MaxDD 门禁进生产的。E2 直接测最终
    目标函数, 是唯一权威判据。

实验组
    C   基线: 全池 max (现状 v4.6, thr 0.45 / slope 0.75 / split 0.60)
    V-A 持仓对口径, 参数不变          → 触发率降至 ~15% (口径+强度双变化)
    V-B 持仓对口径, 断点分位重标定    → 触发率匹配现状 ~34% (只变口径)

    V-B 重标定: corr_held 分布显著低于 corr_pool (均值 0.234), 直接沿用 thr 会把
    触发率腰斩, 使"口径效应"与"强度效应"混淆。故在 corr_held 分布上取与 corr_pool
    同分位的三个断点 (thr / split / 饱和点), 保留频率结构、只改口径。
    标定的循环性: 断点用基线持仓算 corr_held, 而变体会改变持仓 —— 属一阶近似,
    已在报告中标注。

实现
    inspect 源码手术 (同 _exp_directed_boost_study.py:154-201): 只把 boost 调用
    传入的 off_idx 换成 selected_off 过滤后的集合, 其余逐字符同 src/backtest.py。
    零 src/ 改动。可行性: selected_off 在 backtest.py:390-399 先定稿, boost 在
    L420-427 才调用, 无循环依赖; len<2 时引擎返回 (0,0) 自动安全退化。

门禁 (同项目统一口径)
    ΔSharpe >= +0.01 AND ΔMaxDD <= +0.3pp AND 配对 block bootstrap 200 中位不劣
    + 四个对抗情景 MaxDD 不劣化 (bond_bear 为核心观察点, pe_defense 正是死在这里)

无前视
    口径替换只改变参与 corr 计算的资产集合, 窗口仍为 [i-window, i) 已完成收益;
    selected_off 是第 i 周决策时刻已定的量, 未引入未来信息。

用法: .venv/bin/python scripts/_exp_corr_scope_e2.py
"""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import inspect
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import src.backtest as sbt                                            # noqa: E402
from src.strategy import load_config                                  # noqa: E402
from src.data_loader import classify_etfs                             # noqa: E402
from _exp_hongli_defense_e2 import perf                               # noqa: E402
from _exp_huijin_position_bounds_study import setup_font              # noqa: E402
from _exp_etf_flow_style_e2 import (                                  # noqa: E402
    bootstrap_paired, block_bootstrap_paths, _weekly_input,
    N_BOOT, BLOCK, SEED, GATE_D_SHARPE, GATE_D_MAXDD_PP,
)

_spec = importlib.util.spec_from_file_location(
    "adv", PROJECT / "scripts" / "adversarial_robustness.py")
adv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adv)
dm = adv.dm

OUT_DIR = PROJECT / "output" / "experiments"
OUT_MD = OUT_DIR / "exp_corr_scope_e2.md"
OUT_JSON = OUT_DIR / "exp_corr_scope_e2.json"
OUT_PNG = OUT_DIR / "corr_scope_e2.png"
CFG_PATH = PROJECT / "config" / "strategy_v4_6.yaml"

ADV_SEEDS = (11, 22, 33, 44, 55, 66, 77)
SCENARIOS = {
    "bond_bear":         {"gen": "garch",  "params": dict(adv.REALIZED, mudef_mult=0.5)},
    "grey_corr_combo":   {"gen": "regime", "params": dict(adv.REALIZED, dgp="regime_corr",
                          rho_crisis=0.50, p_enter=1.0, p_stay=1.0, sig_mult=1.5)},
    "corr_regime_shift": {"gen": "regime", "params": dict(adv.REALIZED, dgp="regime_corr",
                          rho_crisis=0.85)},
    "corr_crisis_combo": {"gen": "regime", "params": dict(adv.REALIZED, dgp="regime_corr",
                          rho_crisis=0.85, sig_mult=1.2, muoff_mult=0.8)},
}

# ======================================================================
# 源码手术
# ======================================================================
_ANCHOR = """            crisis_boost, _corr_level = compute_crisis_boost_directed(
                w_rets, i, off_idx, config)"""

_PATCHED = """            crisis_boost, _corr_level = compute_crisis_boost_directed(
                w_rets, i, [_j for _j in off_idx if _j in selected_off], config)"""

# 模块加载时一次性捕获 (后续 exec 会覆盖 sbt.run_backtest, 届时 getsource 失效)
_ORIG_RB_SOURCE = inspect.getsource(sbt.run_backtest)
_ORIG_RB = sbt.run_backtest


def build_held_run_backtest():
    """返回手术后的 run_backtest (口径 = 仅选中 TOP2), 不留污染。

    exec 回 sbt.__dict__ 以便 compute_crisis_boost_directed 等名字动态解析;
    随后把模块属性 run_backtest 恢复为原始对象 —— 手术后的函数对象仍持有
    sbt.__dict__ 作为 __globals__, 可独立工作 (run_backtest 无自递归)。
    """
    assert _ANCHOR in _ORIG_RB_SOURCE, "手术锚点失效: src/backtest.py 源码已漂移"
    src = _ORIG_RB_SOURCE.replace(_ANCHOR, _PATCHED)
    assert _PATCHED in src
    exec(compile(src, "<corr_scope_held>", "exec"), sbt.__dict__)
    fn = sbt.__dict__["run_backtest"]
    sbt.__dict__["run_backtest"] = _ORIG_RB      # 复原模块属性
    return fn


RB_HELD = build_held_run_backtest()


def run_quiet(rb_fn, cfg, data_path=None):
    """静默跑一次回测, 返回 nav_series。"""
    with contextlib.redirect_stdout(io.StringIO()):
        res = (rb_fn(cfg) if data_path is None
               else rb_fn(cfg, start_date=None, data_path=str(data_path)))
    return res["nav_series"] if isinstance(res, dict) else res.nav_series


# ======================================================================
# EWMA |corr| (口径复刻 engine_core._compute_crisis_boost_ewma)
# ======================================================================
def ewma_abs_corr(w_rets, i, ja, jb, window, halflife):
    if i < window:
        return np.nan
    seg = w_rets[i - window:i, [ja, jb]]
    t = np.arange(window)
    wt = 0.5 ** ((window - 1 - t) / max(halflife, 1))
    m = ~(np.isnan(seg[:, 0]) | np.isnan(seg[:, 1]))
    if m.sum() < 5:
        return np.nan
    x, y = seg[m, 0], seg[m, 1]
    q = wt[m]
    q = q / q.sum()
    xb, yb = float(np.sum(q * x)), float(np.sum(q * y))
    cov = float(np.sum(q * (x - xb) * (y - yb)))
    vx, vy = float(np.sum(q * (x - xb) ** 2)), float(np.sum(q * (y - yb) ** 2))
    c = cov / (np.sqrt(vx * vy) + 1e-12)
    return np.nan if np.isnan(c) else abs(c)


def corr_series(cfg, nav_series):
    """给定一次回测结果, 算逐周 (corr_pool, corr_held)。"""
    weekly = _weekly_input(cfg)
    if cfg.start_date:
        weekly = weekly[weekly.index >= pd.to_datetime(cfg.start_date)]
    if cfg.end_date:
        weekly = weekly[weekly.index <= pd.to_datetime(cfg.end_date)]
    w = weekly.values
    wr = np.diff(w, axis=0) / w[:-1]
    names = list(weekly.columns)
    off_idx, _, _ = classify_etfs(names)
    start_idx = list(weekly.index).index(nav_series.index[0]) - 1
    W = nav_series[[f"weight_{names[j]}" for j in off_idx]].values
    win, hl = cfg.crisis_corr_window, cfg.crisis_corr_ewma_halflife
    pool, held = [], []
    for k in range(len(nav_series)):
        i = start_idx + k
        best = np.nan
        for a in range(len(off_idx)):
            for b in range(a + 1, len(off_idx)):
                c = ewma_abs_corr(wr, i, off_idx[a], off_idx[b], win, hl)
                if not np.isnan(c):
                    best = c if np.isnan(best) else max(best, c)
        pool.append(best)
        h = [c for c in range(len(off_idx)) if W[k, c] > 1e-9]
        held.append(ewma_abs_corr(wr, i, off_idx[h[0]], off_idx[h[1]], win, hl)
                    if len(h) == 2 else np.nan)
    return np.array(pool, float), np.array(held, float)


def calibrate_vb(cfg, pool, held):
    """分位匹配三个断点: thr / split / 饱和点, 反解 slope。"""
    p = pool[~np.isnan(pool)]
    h = held[~np.isnan(held)]
    thr0, split0 = cfg.directed_boost_threshold, cfg.directed_boost_corr_split
    sat0 = thr0 + cfg.crisis_corr_max_boost / cfg.directed_boost_slope
    rates = {  # 全池口径下三个断点各自的超越频率
        "thr": float((p > thr0).mean()),
        "split": float((p > split0).mean()),
        "sat": float((p > sat0).mean()),
    }
    q = {k: float(np.percentile(h, 100 * (1 - v))) for k, v in rates.items()}
    # 保序 + 退化保护: 饱和点必须严格大于 thr
    thr_n, split_n, sat_n = q["thr"], q["split"], q["sat"]
    sat_n = max(sat_n, thr_n + 1e-3)
    slope_n = cfg.crisis_corr_max_boost / (sat_n - thr_n)
    return {"pool_rates": rates,
            "thr": thr_n, "split": split_n, "sat": sat_n, "slope": slope_n,
            "orig": {"thr": thr0, "split": split0, "sat": sat0,
                     "slope": cfg.directed_boost_slope}}


def trigger_rate(cfg, corr_arr, thr):
    v = corr_arr[~np.isnan(corr_arr)]
    return float((v > thr).mean()) if len(v) else float("nan")


# ======================================================================
# 主流程
# ======================================================================
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_c = load_config(CFG_PATH)
    print(f"[setup] 基座 {CFG_PATH.name}  pvd={cfg_c.pvd_enabled}  "
          f"directed_boost={cfg_c.directed_boost_enabled}")
    assert cfg_c.directed_boost_enabled, "基座必须启用 directed_boost (v4.6)"

    # ---------- realized: 基线 C ----------
    print("[realized] C 基线 (全池口径) ...")
    nav_c = run_quiet(_ORIG_RB, cfg_c)
    perf_c = perf(nav_c)
    pool, held = corr_series(cfg_c, nav_c)

    # ---------- 手术正确性自证 ----------
    # 手术只改 corr 的资产集合; 若把 selected_off 换成全池应逐位复现基线
    print("[self-check] 手术等价性验证 ...")
    nav_held = run_quiet(RB_HELD, cfg_c)
    same_as_c = bool(np.allclose(nav_c["nav"].values, nav_held["nav"].values,
                                 rtol=0, atol=1e-12))
    n_diff_weeks = int((np.abs(nav_c["def_ratio"].values
                               - nav_held["def_ratio"].values) > 1e-9).sum())
    print(f"  手术后与基线 NAV 逐位相同: {same_as_c} (预期 False)")
    print(f"  def_ratio 不同的周数: {n_diff_weeks} / {len(nav_c)}")
    assert not same_as_c, "手术未生效: 结果与基线完全一致, 锚点可能未命中"

    # ---------- V-B 分位重标定 ----------
    cal = calibrate_vb(cfg_c, pool, held)
    cfg_vb = copy.deepcopy(cfg_c)
    cfg_vb.directed_boost_threshold = cal["thr"]
    cfg_vb.directed_boost_corr_split = cal["split"]
    cfg_vb.directed_boost_slope = cal["slope"]
    print(f"[calibrate] V-B thr {cal['orig']['thr']:.3f}->{cal['thr']:.3f}  "
          f"split {cal['orig']['split']:.3f}->{cal['split']:.3f}  "
          f"slope {cal['orig']['slope']:.3f}->{cal['slope']:.3f}  "
          f"(饱和点 {cal['orig']['sat']:.3f}->{cal['sat']:.3f})")

    print("[realized] V-A (持仓对, 参数不变) ...")
    nav_va = nav_held                      # 与自证复用同一次运行
    perf_va = perf(nav_va)
    print("[realized] V-B (持仓对, 分位重标定) ...")
    nav_vb = run_quiet(RB_HELD, cfg_vb)
    perf_vb = perf(nav_vb)

    # 各组自身口径下的触发率
    _, held_va = corr_series(cfg_c, nav_va)
    _, held_vb = corr_series(cfg_vb, nav_vb)
    trig = {
        "C_pool": trigger_rate(cfg_c, pool, cfg_c.directed_boost_threshold),
        "V_A_held": trigger_rate(cfg_c, held_va, cfg_c.directed_boost_threshold),
        "V_B_held": trigger_rate(cfg_vb, held_vb, cal["thr"]),
    }
    print(f"[trigger] C={trig['C_pool']:.1%}  V-A={trig['V_A_held']:.1%}  "
          f"V-B={trig['V_B_held']:.1%}")

    variants = {
        "V_A": {"label": "持仓对口径 (参数不变)", "perf": perf_va, "cfg": cfg_c},
        "V_B": {"label": "持仓对口径 (分位重标定)", "perf": perf_vb, "cfg": cfg_vb},
    }
    for k, v in variants.items():
        p = v["perf"]
        v["d_sharpe"] = p["sharpe"] - perf_c["sharpe"]
        v["d_maxdd_pp"] = p["maxdd_pct"] - perf_c["maxdd_pct"]
        v["gate_sharpe"] = bool(v["d_sharpe"] >= GATE_D_SHARPE)
        v["gate_maxdd"] = bool(v["d_maxdd_pp"] <= GATE_D_MAXDD_PP)
        print(f"[{k}] Sh={p['sharpe']:.3f} ({v['d_sharpe']:+.3f})  "
              f"DD={p['maxdd_pct']:.2f}% ({v['d_maxdd_pp']:+.2f}pp)  "
              f"CAGR={p['cagr_pct']:.2f}%")

    # ---------- 配对 block bootstrap ----------
    print(f"[bootstrap] 配对 {N_BOOT} 路径 block={BLOCK} seed={SEED} (基线+2变体) ...")
    runners = {
        "V_A": lambda p: {"nav_series": run_quiet(RB_HELD, cfg_c, p)},
        "V_B": lambda p: {"nav_series": run_quiet(RB_HELD, cfg_vb, p)},
    }
    boot = bootstrap_paired(cfg_c, runners)
    for k in variants:
        b = boot.get(k, {})
        variants[k]["bootstrap"] = b
        variants[k]["gate_bootstrap"] = bool(b.get("median_not_worse", False))
        if b.get("usable"):
            print(f"  {k}: ΔSharpe 中位 {b['d_sharpe_median']:+.4f} "
                  f"(5/95: {b['d_sharpe_p05']:+.3f}/{b['d_sharpe_p95']:+.3f}, "
                  f"正比例 {b['d_sharpe_pct_positive']:.0f}%)  "
                  f"ΔMaxDD 中位 {b['d_maxdd_pp_median']:+.3f}pp  "
                  f"→ {'PASS' if b['median_not_worse'] else 'FAIL'}")

    # ---------- 对抗情景 ----------
    # 合成 NAV 无 amount 数据 → PVD 必须关闭 (沿用 _exp_directed_boost_study 先例);
    # 三组统一在 PVD 关闭副本上跑, 保证组间可比。
    print("[adversarial] 拟合 VAR(1)-t + GARCH ...")
    cfg_c_nopvd = copy.deepcopy(cfg_c)
    cfg_c_nopvd.pvd_enabled = False
    cfg_vb_nopvd = copy.deepcopy(cfg_vb)
    cfg_vb_nopvd.pvd_enabled = False
    _, wk, w_rets = dm.load_real()
    mu, A, Sigma, nu, resid, coords = dm.fit_var_t(w_rets)
    gp, R = adv.fit_garch(resid)
    real_dates, first_nav, T = wk.index, wk.iloc[0].values, len(w_rets)
    print(f"[adversarial] T={T} 周, nu={nu:.1f}, {len(SCENARIOS)} 情景 x "
          f"{len(ADV_SEEDS)} seeds x 3 组")

    arms = [("C", _ORIG_RB, cfg_c_nopvd), ("V_A", RB_HELD, cfg_c_nopvd),
            ("V_B", RB_HELD, cfg_vb_nopvd)]
    adv_res = {k: {} for k, _, _ in arms}
    for sc_name, sc in SCENARIOS.items():
        gen = adv.gen_regime_corr if sc["gen"] == "regime" else adv.gen_garch
        paths = []
        for s in ADV_SEEDS:
            rr = gen(mu, A, R, nu, gp, sc["params"], T, s)
            paths.append((s, dm.build_nav_df(rr, real_dates, first_nav)))
        for key, rb, cfgx in arms:
            dds, shs = [], []
            for s, ndf in paths:
                tmp = OUT_DIR / f"_cs_e2_{key}_{sc_name}_{s}_{os.getpid()}.csv"
                ndf.to_csv(tmp, encoding="utf-8")
                try:
                    nv = run_quiet(rb, cfgx, tmp)
                    if nv.empty:
                        continue
                    m = perf(nv)
                    if m["sharpe"] is None:
                        continue
                    dds.append(m["maxdd_pct"])
                    shs.append(m["sharpe"])
                finally:
                    if tmp.exists():
                        tmp.unlink()
            adv_res[key][sc_name] = {
                "n": len(dds),
                "maxdd_med": float(np.median(dds)) if dds else float("nan"),
                "maxdd_worst": float(np.max(dds)) if dds else float("nan"),
                "sharpe_med": float(np.median(shs)) if shs else float("nan"),
            }
        line = "  ".join(
            f"{k}:{adv_res[k][sc_name]['maxdd_med']:.2f}%" for k, _, _ in arms)
        print(f"  [{sc_name}] MaxDD 中位  {line}", flush=True)

    # 对抗门禁: 各情景 MaxDD 中位与最差均不劣于 C 超过 GATE_D_MAXDD_PP
    for k in variants:
        worst_med_delta = max(
            adv_res[k][s]["maxdd_med"] - adv_res["C"][s]["maxdd_med"] for s in SCENARIOS)
        worst_worst_delta = max(
            adv_res[k][s]["maxdd_worst"] - adv_res["C"][s]["maxdd_worst"] for s in SCENARIOS)
        variants[k]["adv_worst_med_delta_pp"] = worst_med_delta
        variants[k]["adv_worst_worst_delta_pp"] = worst_worst_delta
        variants[k]["gate_adversarial"] = bool(worst_med_delta <= GATE_D_MAXDD_PP)

    # ---------- 裁决 ----------
    for k, v in variants.items():
        v["gate_all"] = bool(v["gate_sharpe"] and v["gate_maxdd"]
                             and v["gate_bootstrap"] and v["gate_adversarial"])
    go = [k for k, v in variants.items() if v["gate_all"]]
    verdict = {
        "go_variants": go,
        "decision": "GO" if go else "NO-GO",
        "conclusion": (
            f"{'/'.join(go)} 三门禁全过且对抗不劣, 建议进 E3 (配置开关默认关 + "
            f"TestBaselineUnchanged pin + OOS 全管线)。"
            if go else
            "两个变体均未过门禁。全池口径虽有 55.7% 机制性误触发 (E1 实证), "
            "但改为持仓对口径在回测目标函数上无收益, 归档为 NO-GO, 不改 src/。"),
    }

    res = {
        "config": str(CFG_PATH.name),
        "seed": SEED, "n_boot": N_BOOT, "block": BLOCK,
        "gates": {"d_sharpe": GATE_D_SHARPE, "d_maxdd_pp": GATE_D_MAXDD_PP},
        "calibration_vb": cal,
        "trigger_rates": trig,
        "surgery_selfcheck": {"differs_from_baseline": not same_as_c,
                              "n_weeks_def_ratio_changed": n_diff_weeks,
                              "total_weeks": int(len(nav_c))},
        "baseline": perf_c,
        "variants": {k: {kk: vv for kk, vv in v.items() if kk != "cfg"}
                     for k, v in variants.items()},
        "adversarial": adv_res,
        "adversarial_note": ("合成 NAV 无 amount 数据, 三组统一在 PVD 关闭副本上跑 "
                             "(沿用 _exp_directed_boost_study 先例); realized 臂用完整 v4.6。"),
        "verdict": verdict,
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                        encoding="utf-8")
    render(res)
    plot(res)
    print(f"\n[save] {OUT_JSON}\n[save] {OUT_MD}\n[save] {OUT_PNG}")
    print(f"DONE in {res['runtime_min']:.1f} min  →  {verdict['decision']}")


def render(res):
    v_, cal, trig = res["variants"], res["calibration_vb"], res["trigger_rates"]
    b = res["baseline"]
    L = ["# E2: Layer 3.5 相关性口径 — 全池 max vs 仅选中 TOP2", "",
         f"基线 `{res['config']}` | seed={res['seed']} | bootstrap {res['n_boot']} "
         f"block={res['block']}（重采样输入周频净值, 三组配对共用路径）", "",
         "## 实验组", "",
         "| 组 | 口径 | thr / slope / split | 触发率 |", "|---|---|---|---|",
         f"| C 基线 | 全池 max | {cal['orig']['thr']:.3f} / {cal['orig']['slope']:.3f} / "
         f"{cal['orig']['split']:.3f} | {trig['C_pool']:.1%} |",
         f"| V-A | 仅选中 TOP2 | 同基线 | {trig['V_A_held']:.1%} |",
         f"| V-B | 仅选中 TOP2 | {cal['thr']:.3f} / {cal['slope']:.3f} / "
         f"{cal['split']:.3f}（分位重标定） | {trig['V_B_held']:.1%} |", "",
         f"> V-B 分位匹配: 在 corr_held 分布上取与 corr_pool 同分位的三个断点 —— "
         f"thr（超越率 {cal['pool_rates']['thr']:.1%}）、split（{cal['pool_rates']['split']:.1%}）、"
         f"饱和点（{cal['pool_rates']['sat']:.1%}, {cal['orig']['sat']:.3f}→{cal['sat']:.3f}）, "
         f"再反解 slope。标定用基线持仓算 corr_held 而变体会改变持仓, 属一阶近似。", "",
         f"> 手术自证: 手术后 def_ratio 与基线不同的周数 "
         f"{res['surgery_selfcheck']['n_weeks_def_ratio_changed']}/"
         f"{res['surgery_selfcheck']['total_weeks']}（证明口径替换真实生效）", "",
         "## realized 全样本与门禁", "",
         "| 组 | Sharpe | ΔSharpe | MaxDD | ΔMaxDD | CAGR | 换手 | Sharpe门禁 | MaxDD门禁 | bootstrap门禁 | 对抗门禁 |",
         "|---|---|---|---|---|---|---|---|---|---|---|",
         f"| C 基线 | {b['sharpe']:.3f} | — | {b['maxdd_pct']:.2f}% | — | "
         f"{b['cagr_pct']:.2f}% | {b['turnover_mean_pct']:.2f}% | — | — | — | — |"]
    for k in ("V_A", "V_B"):
        v = v_[k]
        p, bt = v["perf"], v.get("bootstrap", {})
        L.append(
            f"| {k.replace('_', '-')} | {p['sharpe']:.3f} | {v['d_sharpe']:+.3f} | "
            f"{p['maxdd_pct']:.2f}% | {v['d_maxdd_pp']:+.2f}pp | {p['cagr_pct']:.2f}% | "
            f"{p['turnover_mean_pct']:.2f}% | "
            f"{'PASS' if v['gate_sharpe'] else 'FAIL'} | "
            f"{'PASS' if v['gate_maxdd'] else 'FAIL'} | "
            f"{'PASS' if v['gate_bootstrap'] else 'FAIL'} | "
            f"{'PASS' if v['gate_adversarial'] else 'FAIL'} |")
    L += ["", f"门禁: ΔSharpe >= +{res['gates']['d_sharpe']} AND ΔMaxDD <= "
          f"+{res['gates']['d_maxdd_pp']}pp AND 配对 bootstrap 中位不劣 AND 对抗 MaxDD 不劣。"
          "MaxDD 为正数深度, 正 Δ 表示更差。", "",
          "## 配对 bootstrap 分布", "",
          "| 组 | 路径数 | ΔSharpe 中位 | 5% | 95% | 正比例 | ΔMaxDD 中位 |",
          "|---|---|---|---|---|---|---|"]
    for k in ("V_A", "V_B"):
        bt = v_[k].get("bootstrap", {})
        if not bt.get("usable"):
            L.append(f"| {k.replace('_', '-')} | {bt.get('n_paths', 0)} | 不可用 | | | | |")
            continue
        L.append(f"| {k.replace('_', '-')} | {bt['n_paths']} | {bt['d_sharpe_median']:+.4f} | "
                 f"{bt['d_sharpe_p05']:+.3f} | {bt['d_sharpe_p95']:+.3f} | "
                 f"{bt['d_sharpe_pct_positive']:.0f}% | {bt['d_maxdd_pp_median']:+.3f}pp |")

    L += ["", "## 对抗情景 MaxDD（中位 / 最差）", "",
          "| 情景 | C 基线 | V-A | V-B |", "|---|---|---|---|"]
    for sc in res["adversarial"]["C"]:
        row = [f"| {sc} "]
        for k in ("C", "V_A", "V_B"):
            a = res["adversarial"][k][sc]
            row.append(f"| {a['maxdd_med']:.2f}% / {a['maxdd_worst']:.2f}% ")
        L.append("".join(row) + "|")
    L.append("")
    for k in ("V_A", "V_B"):
        L.append(f"- {k.replace('_', '-')} 最坏情景 ΔMaxDD 中位 "
                 f"{v_[k]['adv_worst_med_delta_pp']:+.2f}pp, 最差 "
                 f"{v_[k]['adv_worst_worst_delta_pp']:+.2f}pp")
    # 补充观察: bond_bear 是 defense_asset 硬门禁情景 (pe_defense 死因), 且基线
    # 中位值恰好突破项目 D_MAX=12% 约束 —— 值得单独点出, 但不改变裁决。
    bb = {k: res["adversarial"][k]["bond_bear"] for k in ("C", "V_A", "V_B")}
    L += ["", f"> {res['adversarial_note']}", "",
          "## 补充观察: bond_bear 中位 MaxDD 改善（不改变裁决）", "",
          f"- C 基线 {bb['C']['maxdd_med']:.2f}%, **突破项目 D_MAX=12% 约束**; "
          f"V-A {bb['V_A']['maxdd_med']:.2f}% / V-B {bb['V_B']['maxdd_med']:.2f}% 均回到约束内",
          f"- 但尾部未改善: 最差值 C {bb['C']['maxdd_worst']:.2f}% vs "
          f"V-A {bb['V_A']['maxdd_worst']:.2f}% / V-B {bb['V_B']['maxdd_worst']:.2f}%",
          f"- 样本仅 {bb['C']['n']} seeds, 中位改善属**提示性而非结论性**; bond_bear 是 "
          "defense_asset 硬门禁情景(pe_defense 正是死在这里), 故作为未来方向候选留档, "
          "不构成本次 GO 的理由", "",
          "## 裁决", "", f"**{res['verdict']['decision']}**", "",
          res["verdict"]["conclusion"], "",
          "## 方法说明", "",
          "- 源码手术只替换 boost 调用传入的资产索引集合, 其余逐字符同 src/backtest.py; 零 src/ 改动",
          "- 无前视: corr 窗口仍为 [i-window, i) 已完成收益; selected_off 是第 i 周决策时刻已定量",
          "- bootstrap 重采样输入周频净值并重跑回测, 基线与变体共用同一路径(配对)",
          f"- 运行耗时 {res['runtime_min']:.1f} 分钟", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")


def plot(res):
    """两面板对比图: 配对 bootstrap ΔSharpe (中位 + 5/95) 与对抗情景 MaxDD。

    bootstrap_paired 只返回汇总量 (不含逐路径原始值), 故左图用中位+分位须线
    而非直方图 —— 不为了画图重跑 200 路径。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_font()
    keys = ["V_A", "V_B"]
    labels = ["V-A 参数不变", "V-B 分位重标定"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    med = [res["variants"][k]["bootstrap"].get("d_sharpe_median", np.nan) for k in keys]
    p05 = [res["variants"][k]["bootstrap"].get("d_sharpe_p05", np.nan) for k in keys]
    p95 = [res["variants"][k]["bootstrap"].get("d_sharpe_p95", np.nan) for k in keys]
    x = np.arange(len(keys))
    lo = [m - a for m, a in zip(med, p05)]
    hi = [b - m for m, b in zip(med, p95)]
    ax1.errorbar(x, med, yerr=[lo, hi], fmt="o", capsize=6, color="#c0392b", markersize=8)
    ax1.axhline(0, color="#555", lw=1)
    ax1.axhline(res["gates"]["d_sharpe"], color="#27ae60", ls="--", lw=1,
                label=f"门禁 +{res['gates']['d_sharpe']}")
    for i, m in enumerate(med):
        ax1.annotate(f"{m:+.4f}", (x[i], m), textcoords="offset points",
                     xytext=(12, 0), fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("ΔSharpe (vs C 基线)")
    ax1.set_title(f"配对 block bootstrap {res['n_boot']} 路径: ΔSharpe 中位 (5/95)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    scs = list(res["adversarial"]["C"].keys())
    w = 0.26
    for j, (k, c) in enumerate(zip(["C", "V_A", "V_B"], ["#34495e", "#e67e22", "#2980b9"])):
        vals = [res["adversarial"][k][s]["maxdd_med"] for s in scs]
        ax2.bar(np.arange(len(scs)) + (j - 1) * w, vals, w,
                label={"C": "C 基线", "V_A": "V-A", "V_B": "V-B"}[k], color=c)
    ax2.axhline(12.0, color="#c0392b", ls="--", lw=1.2, label="D_MAX 12%")
    ax2.set_xticks(np.arange(len(scs)))
    ax2.set_xticklabels(scs, rotation=18, ha="right", fontsize=8)
    ax2.set_ylabel("MaxDD 中位 (%)")
    ax2.set_title(f"对抗情景 MaxDD 中位 ({len(ADV_SEEDS)} seeds, PVD 关)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    # --render-only: 从既有 JSON 重渲染报告与图, 免去 17 分钟重跑
    if "--render-only" in sys.argv:
        _r = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        render(_r)
        plot(_r)
        print(f"[save] {OUT_MD}\n[save] {OUT_PNG} (render-only)")
    else:
        main()
