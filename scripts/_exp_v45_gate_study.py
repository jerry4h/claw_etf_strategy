#!/usr/bin/env python3
"""v4.5 预研: 条件门控机制 (M-D) 实验 — 任务 #39。

背景: M-C 纯参数降 threshold(0.45-0.50) 修复 grey 但 bond_bear 超线 12.32%,
因 bond_bear DGP 自然相关中位 ≈0.40 落在灰区。条件门控方案：在触发 boost 前
检查防御资产健康状态，防御下行时关闭 boost → bond_bear 退化为 v4.4 行为。

门控信号 A-D：
  A: 防御端 SMA-13 健康（def_nav / SMA13 > 0.98）
  B: 防御端 26 周累计收益 > 0
  C: 防御端 26 周内最大回撤 < 5%
  D: 防御端近 4 周收益 > -2%（宽松噪声滤波）

参数网格: thr {0.45, 0.475, 0.50} × slope {0.60, 0.75} × gate {A, B, C, D}
= 3 × 2 × 4 = 24 组合

用法: .venv/bin/python scripts/_exp_v45_gate_study.py
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

# Load adversarial framework
_spec = importlib.util.spec_from_file_location(
    "adv", PROJ / "scripts" / "adversarial_robustness.py")
adv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adv)
dm = adv.dm

import src.backtest as sbt
from src.backtest import run_backtest, compute_metrics
from src.data_loader import ETFS
from src.engine_core import compute_crisis_boost as engine_crisis_boost
from src.strategy import load_config

OUT = PROJ / "output" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)
CFG44 = PROJ / "config" / "strategy_v4_4.yaml"

SEEDS = (11, 22, 33, 44, 55, 66, 77)
OFF_IDX = dm.OFF_IDX        # [0, 2, 3]
DEF_IDX = [1, 4]            # 红利低波, 国债
D_MAX = 0.12

# v4.4 baseline (from v45_t2_eval_v44.json)
V44_SCENARIOS = {
    "bond_bear": {"sharpe": 0.8757, "maxdd": 0.10762},
    "grey_corr_combo": {"sharpe": 0.7271, "maxdd": 0.12810},
    "stagflation": {"sharpe": 0.8217, "maxdd": 0.11004},
    "vol_stress": {"sharpe": 0.9814, "maxdd": 0.11282},
    "offense_cooldown": {"sharpe": 0.9905, "maxdd": 0.09862},
    "decorrelation": {"sharpe": 1.2527, "maxdd": 0.07943},
    "corr_regime_shift": {"sharpe": 1.2971, "maxdd": 0.07872},
    "corr_crisis_combo": {"sharpe": 1.0274, "maxdd": 0.09663},
}
V44_REALIZED_SHARPE = 1.4985

# Scenario definitions (same as evaluate.py + corr scenarios)
SCENARIOS = {
    "bond_bear":        {"mudef_mult": 0.5},
    "stagflation":      {"sig_mult": 1.2, "muoff_mult": 0.8},
    "grey_corr_combo":  {"dgp": "regime_corr", "rho_crisis": 0.50,
                         "p_enter": 1.0, "p_stay": 1.0, "sig_mult": 1.5},
    "vol_stress":       {"sig_mult": 1.2},
    "offense_cooldown": {"muoff_mult": 0.8},
    "decorrelation":    {"c_mult": 0.77},
    "corr_regime_shift": {"dgp": "regime_corr", "rho_crisis": 0.85},
    "corr_crisis_combo": {"dgp": "regime_corr", "rho_crisis": 0.85,
                          "sig_mult": 1.2, "muoff_mult": 0.8},
}
FAST_SCENARIOS = ["bond_bear", "grey_corr_combo", "stagflation"]

# Parameter grid
THR_GRID = [0.45, 0.475, 0.50]
SLOPE_GRID = [0.60, 0.75]
GATE_NAMES = ["A", "B", "C", "D"]

# Tolerances
SHARPE_TOL = -0.02
MAXDD_TOL = 0.003     # 0.3pp
REALIZED_SHARPE_MIN = V44_REALIZED_SHARPE - 0.01  # 1.4885


# ======================================================================
# Gate signal implementations
# ======================================================================
def _def_nav_series(w_rets, i):
    """Compute cumulative defense NAV up to week i (equal-weight 红利低波+国债)."""
    if i < 1:
        return np.array([1.0])
    # Equal-weight defense returns
    dr = np.nanmean(w_rets[:i, DEF_IDX], axis=1)
    nav = np.ones(i + 1)
    for t in range(i):
        nav[t + 1] = nav[t] * (1 + dr[t])
    return nav


def gate_A(w_rets, i, _off_idx, _config):
    """Gate A: defense NAV / SMA-13 > 0.98 (seasonal health)."""
    if i < 13:
        return True  # insufficient data, pass through
    nav = _def_nav_series(w_rets, i)
    current = nav[-1]
    sma13 = np.mean(nav[-13:])
    return current / sma13 > 0.98


def gate_B(w_rets, i, _off_idx, _config):
    """Gate B: defense 26-week cumulative return > 0."""
    if i < 26:
        return True
    dr = np.nanmean(w_rets[i-26:i, DEF_IDX], axis=1)
    cum = np.prod(1 + dr) - 1
    return cum > 0.0


def gate_C(w_rets, i, _off_idx, _config):
    """Gate C: defense 26-week max drawdown < 5%."""
    if i < 26:
        return True
    nav = _def_nav_series(w_rets, i)
    window = nav[-27:]  # last 26 weeks + current
    peak = np.maximum.accumulate(window)
    dd = (peak - window) / peak
    return float(dd.max()) < 0.05


def gate_D(w_rets, i, _off_idx, _config):
    """Gate D: defense 4-week return > -2% (lenient noise filter)."""
    if i < 4:
        return True
    dr = np.nanmean(w_rets[i-4:i, DEF_IDX], axis=1)
    cum = np.prod(1 + dr) - 1
    return cum > -0.02


GATES = {"A": gate_A, "B": gate_B, "C": gate_C, "D": gate_D}


# ======================================================================
# EWMA max|ρ| (replicated from engine_core for gated mechanism)
# ======================================================================
def _maxcorr_ewma(w_rets, i, off_idx, window, halflife):
    """EWMA weighted max|ρ| — same as engine_core._compute_crisis_boost_ewma."""
    if i < window or len(off_idx) < 2:
        return np.nan
    win = w_rets[i - window:i, off_idx]
    t = np.arange(window)
    weights = 0.5 ** ((window - 1 - t) / max(halflife, 1))
    mc = 0.0
    n = win.shape[1]
    for a in range(n):
        for b in range(a + 1, n):
            mask = ~(np.isnan(win[:, a]) | np.isnan(win[:, b]))
            if mask.sum() >= 5:
                x, y = win[mask, a], win[mask, b]
                ww = weights[mask]
                ww = ww / ww.sum()
                xb = float(np.sum(ww * x))
                yb = float(np.sum(ww * y))
                cov = float(np.sum(ww * (x - xb) * (y - yb)))
                vx = float(np.sum(ww * (x - xb) ** 2))
                vy = float(np.sum(ww * (y - yb) ** 2))
                c = cov / (np.sqrt(vx * vy) + 1e-12)
                if not np.isnan(c):
                    mc = max(mc, abs(c))
    return mc


def make_gated_boost(threshold, slope, max_boost, window, halflife, gate_fn):
    """Factory: create a gated compute_crisis_boost replacement."""
    def gated_boost(w_rets, i, off_idx, config):
        # Gate check first
        w = np.asarray(w_rets, float)
        if not gate_fn(w, i, off_idx, config):
            return 0.0
        # EWMA correlation computation (M-C style with custom thr/slope)
        c = _maxcorr_ewma(w, i, off_idx, window, halflife)
        if np.isnan(c) or c <= threshold:
            return 0.0
        return float(min((c - threshold) * slope, max_boost))
    return gated_boost


# ======================================================================
# Evaluation helpers (reuse adversarial framework pattern)
# ======================================================================
def eval_scenario(mu, A, R, nu, gp, T, real_dates, first_nav,
                  cfg, sc_name, sc_overrides, boost_fn):
    """Run scenario with monkeypatched boost, return median metrics."""
    gen = adv.gen_regime_corr if sc_overrides.get("dgp") == "regime_corr" else adv.gen_garch
    s_dd, s_sh, s_an = [], [], []
    orig = sbt.compute_crisis_boost
    sbt.compute_crisis_boost = boost_fn
    try:
        for seed in SEEDS:
            params = dict(adv.REALIZED, **sc_overrides)
            r = gen(mu, A, R, nu, gp, params, T, seed)
            nav_df = dm.build_nav_df(r, real_dates, first_nav)
            tmp = OUT / f"_gate_{sc_name}_{seed}_{os.getpid()}.csv"
            nav_df.to_csv(tmp, encoding="utf-8")
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    res = run_backtest(cfg, start_date=dm.START_DATE, data_path=str(tmp))
                if res.nav_series.empty:
                    continue
                s_sh.append(float(res.metrics["sharpe_ratio"]))
                s_dd.append(float(res.metrics["max_drawdown"]))
                s_an.append(float(res.metrics["annual_return"]))
            finally:
                if tmp.exists():
                    os.remove(tmp)
    finally:
        sbt.compute_crisis_boost = orig
    if not s_dd:
        return None
    return {
        "strat_maxdd": float(np.median(s_dd)),
        "strat_sharpe": float(np.median(s_sh)),
        "strat_annual": float(np.median(s_an)),
    }


def eval_realized(cfg, boost_fn):
    """Run realized backtest with monkeypatched boost."""
    orig = sbt.compute_crisis_boost
    sbt.compute_crisis_boost = boost_fn
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = run_backtest(cfg, start_date=dm.START_DATE,
                              data_path=str(dm.REAL_CSV))
        return {
            "sharpe": float(res.metrics["sharpe_ratio"]),
            "maxdd": float(res.metrics["max_drawdown"]),
            "annual": float(res.metrics["annual_return"]),
        }
    finally:
        sbt.compute_crisis_boost = orig


def gate_trigger_rate_baseline(w_rets_real, gate_fn, cfg):
    """Compute gate trigger rate on real historical data (baseline scenario)."""
    w = np.asarray(w_rets_real, float)
    T = len(w)
    window = 26
    n_closed = 0
    n_total = 0
    for i in range(window, T):
        n_total += 1
        if not gate_fn(w, i, OFF_IDX, cfg):
            n_closed += 1
    return {"n_total": n_total, "n_closed": n_closed,
            "close_rate": n_closed / n_total if n_total > 0 else 0.0}


# ======================================================================
# Main experiment
# ======================================================================
def run_experiment():
    print("=" * 70)
    print(" v4.5 Gate Study (M-D): 条件门控机制预研")
    print("=" * 70)
    t_start = time.time()

    # Prepare DGP
    nav, wk, w_rets = dm.load_real()
    mu, A, Sigma, nu, resid, coords = dm.fit_var_t(w_rets)
    gp, R = adv.fit_garch(resid)
    real_dates = wk.index
    first_nav = wk.iloc[0].values
    T = len(w_rets)

    cfg44 = load_config(CFG44)
    window = cfg44.crisis_corr_window    # 26
    halflife = cfg44.crisis_corr_ewma_halflife  # 8
    max_boost = cfg44.crisis_corr_max_boost     # 0.15

    # Phase 0: Gate baseline trigger rates (on real data)
    print("\n[Phase 0] Gate baseline trigger rates (real data)...")
    gate_baselines = {}
    for gname, gfn in GATES.items():
        gb = gate_trigger_rate_baseline(w_rets, gfn, cfg44)
        gate_baselines[gname] = gb
        print(f"  Gate {gname}: close_rate={gb['close_rate']:.1%} "
              f"({gb['n_closed']}/{gb['n_total']} weeks gate CLOSED)")

    # Phase 1: Fast filter (bond_bear + grey + stagflation)
    print(f"\n[Phase 1] Fast filter: {len(THR_GRID)}×{len(SLOPE_GRID)}×{len(GATE_NAMES)} = "
          f"{len(THR_GRID)*len(SLOPE_GRID)*len(GATE_NAMES)} combinations × 3 scenarios...")
    
    grid_results = {}
    n_combos = len(THR_GRID) * len(SLOPE_GRID) * len(GATE_NAMES)
    k = 0
    for thr in THR_GRID:
        for slope in SLOPE_GRID:
            for gname in GATE_NAMES:
                k += 1
                label = f"thr{thr}|slope{slope}|gate{gname}"
                boost_fn = make_gated_boost(thr, slope, max_boost, window, halflife,
                                           GATES[gname])
                fast = {}
                for sc_name in FAST_SCENARIOS:
                    r = eval_scenario(mu, A, R, nu, gp, T, real_dates, first_nav,
                                      cfg44, sc_name, SCENARIOS[sc_name], boost_fn)
                    fast[sc_name] = r

                # Quick check
                grey_ok = fast["grey_corr_combo"]["strat_maxdd"] <= D_MAX
                bond_ok = fast["bond_bear"]["strat_maxdd"] <= D_MAX
                bond_degrad = fast["bond_bear"]["strat_maxdd"] - V44_SCENARIOS["bond_bear"]["maxdd"]
                stag_degrad = fast["stagflation"]["strat_maxdd"] - V44_SCENARIOS["stagflation"]["maxdd"]
                
                passes_fast = (grey_ok and bond_ok and 
                              bond_degrad <= MAXDD_TOL and stag_degrad <= MAXDD_TOL)
                
                grid_results[label] = {
                    "thr": thr, "slope": slope, "gate": gname,
                    "fast": fast,
                    "grey_maxdd": fast["grey_corr_combo"]["strat_maxdd"],
                    "bond_maxdd": fast["bond_bear"]["strat_maxdd"],
                    "stag_maxdd": fast["stagflation"]["strat_maxdd"],
                    "passes_fast": passes_fast,
                }
                
                flag = "✓" if passes_fast else "✗"
                print(f"  [{k:>2}/{n_combos}] {label}: {flag} "
                      f"grey={fast['grey_corr_combo']['strat_maxdd']:.2%} "
                      f"bond={fast['bond_bear']['strat_maxdd']:.2%}(Δ{bond_degrad:+.2%}) "
                      f"stag={fast['stagflation']['strat_maxdd']:.2%}(Δ{stag_degrad:+.2%})")

    # Phase 2: Full evaluation for survivors
    survivors = [k for k, v in grid_results.items() if v["passes_fast"]]
    print(f"\n[Phase 2] Survivors: {len(survivors)}/{n_combos}")
    
    full_results = {}
    for label in survivors:
        r = grid_results[label]
        thr, slope, gname = r["thr"], r["slope"], r["gate"]
        boost_fn = make_gated_boost(thr, slope, max_boost, window, halflife,
                                   GATES[gname])
        
        # Full 8 scenarios
        all_sc = {}
        for sc_name, sc_overrides in SCENARIOS.items():
            if sc_name in r["fast"]:
                all_sc[sc_name] = r["fast"][sc_name]
            else:
                m = eval_scenario(mu, A, R, nu, gp, T, real_dates, first_nav,
                                  cfg44, sc_name, sc_overrides, boost_fn)
                all_sc[sc_name] = m
        
        # Realized backtest
        realized = eval_realized(cfg44, boost_fn)
        
        # Check all constraints
        all_dd_ok = all(sc["strat_maxdd"] <= D_MAX for sc in all_sc.values())
        
        degrad_issues = []
        for sc_name, sc in all_sc.items():
            if sc_name not in V44_SCENARIOS:
                continue
            v44 = V44_SCENARIOS[sc_name]
            sh_delta = sc["strat_sharpe"] - v44["sharpe"]
            dd_delta = sc["strat_maxdd"] - v44["maxdd"]
            if sh_delta < SHARPE_TOL:
                degrad_issues.append(f"{sc_name}: Sharpe {sh_delta:+.4f}")
            if dd_delta > MAXDD_TOL:
                degrad_issues.append(f"{sc_name}: MaxDD {dd_delta:+.4%}")
        
        realized_ok = realized["sharpe"] >= REALIZED_SHARPE_MIN
        gate_baseline_ok = gate_baselines[gname]["close_rate"] < 0.05
        
        verdict = (all_dd_ok and len(degrad_issues) == 0 and 
                  realized_ok and gate_baseline_ok)
        
        full_results[label] = {
            "thr": thr, "slope": slope, "gate": gname,
            "scenarios": all_sc,
            "realized": realized,
            "all_dd_ok": all_dd_ok,
            "degrad_issues": degrad_issues,
            "realized_ok": realized_ok,
            "gate_baseline_close_rate": gate_baselines[gname]["close_rate"],
            "gate_baseline_ok": gate_baseline_ok,
            "verdict": verdict,
        }
        
        flag = "✓ PASS" if verdict else "✗ FAIL"
        print(f"  {label}: {flag} | realized Sh={realized['sharpe']:.4f} | "
              f"worst_dd={max(sc['strat_maxdd'] for sc in all_sc.values()):.2%} | "
              f"gate_close={gate_baselines[gname]['close_rate']:.1%}"
              f"{' | issues: '+', '.join(degrad_issues[:3]) if degrad_issues else ''}")

    # Phase 3: Select winner
    winners = [k for k, v in full_results.items() if v["verdict"]]
    print(f"\n[Phase 3] Winners: {len(winners)}/{len(full_results)} (of {n_combos} total)")
    
    best = None
    if winners:
        # Select by max grey margin; ties broken by proximity to v4.4 defaults
        best = max(winners, key=lambda k: D_MAX - full_results[k]["scenarios"]["grey_corr_combo"]["strat_maxdd"])
        r = full_results[best]
        print(f"\n  ★ Best: {best}")
        print(f"    grey={r['scenarios']['grey_corr_combo']['strat_maxdd']:.4%} "
              f"(margin={D_MAX - r['scenarios']['grey_corr_combo']['strat_maxdd']:.4%})")
        print(f"    bond={r['scenarios']['bond_bear']['strat_maxdd']:.4%}")
        print(f"    realized Sharpe={r['realized']['sharpe']:.4f}")
        print(f"    gate baseline close rate={r['gate_baseline_close_rate']:.1%}")
    else:
        # Find best compromise
        print("\n  No winners. Analyzing best compromises...")
        if full_results:
            # Among full results, find closest to passing
            for label, r in sorted(full_results.items(), 
                                   key=lambda x: len(x[1]["degrad_issues"])):
                print(f"  {label}: issues={r['degrad_issues'][:5]}")
        elif survivors:
            print("  (survivors exist but all failed full eval)")
        else:
            # Among all grid results, find closest
            by_bond = sorted(grid_results.items(), 
                           key=lambda x: abs(x[1]["bond_maxdd"] - V44_SCENARIOS["bond_bear"]["maxdd"]))
            print("  Closest to bond_bear constraint:")
            for label, r in by_bond[:5]:
                print(f"    {label}: bond={r['bond_maxdd']:.4%} grey={r['grey_maxdd']:.4%}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")
    
    return {
        "gate_baselines": gate_baselines,
        "grid_results": {k: {kk: vv for kk, vv in v.items() if kk != "fast"} 
                        for k, v in grid_results.items()},
        "grid_fast_detail": {k: v["fast"] for k, v in grid_results.items()},
        "survivors": survivors,
        "full_results": full_results,
        "winners": winners,
        "best": best,
        "elapsed_s": elapsed,
    }


def render_md(results):
    """Generate markdown report."""
    lines = ["# v4.5 Gate Study (M-D): 条件门控机制预研报告", ""]
    lines.append(f"> 任务 #39 | seeds={list(SEEDS)} | 运行时间 {results['elapsed_s']:.0f}s")
    lines.append("")
    
    # Gate definitions
    lines.append("## 1. 门控信号定义")
    lines.append("")
    lines.append("| 门控 | 信号 | 含义 | 基线关闭率 |")
    lines.append("|---|---|---|---|")
    descs = {
        "A": "defense_NAV / SMA-13 > 0.98",
        "B": "defense 26周累计收益 > 0",
        "C": "defense 26周最大回撤 < 5%",
        "D": "defense 近4周收益 > -2%",
    }
    for g in GATE_NAMES:
        gb = results["gate_baselines"][g]
        ok = "✓" if gb["close_rate"] < 0.05 else "✗"
        lines.append(f"| {g} | {descs[g]} | 防御健康放行, 防御下行关闭 | "
                    f"{gb['close_rate']:.1%} ({gb['n_closed']}/{gb['n_total']}周) {ok} |")
    lines.append("")
    lines.append("约束: 基线关闭率 < 5% 以避免对历史 realized 不可控影响。")
    lines.append("")
    
    # Fast filter grid
    lines.append("## 2. 快筛网格结果 (bond_bear / grey / stagflation)")
    lines.append("")
    lines.append(f"| 组合 | grey MaxDD | bond MaxDD | bond Δv4.4 | stag MaxDD | stag Δv4.4 | 快筛 |")
    lines.append("|---|---|---|---|---|---|---|")
    for label in sorted(results["grid_results"].keys(), 
                       key=lambda k: (results["grid_results"][k]["thr"],
                                     results["grid_results"][k]["slope"],
                                     results["grid_results"][k]["gate"])):
        r = results["grid_results"][label]
        gm = r["grey_maxdd"]
        bm = r["bond_maxdd"]
        sm = r["stag_maxdd"]
        bd = bm - V44_SCENARIOS["bond_bear"]["maxdd"]
        sd = sm - V44_SCENARIOS["stagflation"]["maxdd"]
        flag = "✓" if r["passes_fast"] else "✗"
        lines.append(f"| {label} | {gm:.2%} | {bm:.2%} | {bd:+.2%} | {sm:.2%} | {sd:+.2%} | {flag} |")
    lines.append("")
    lines.append(f"快筛通过: {len(results['survivors'])}/{len(results['grid_results'])} 组合")
    lines.append("")
    
    # Full evaluation results
    if results["full_results"]:
        lines.append("## 3. 完整评估 (8+1 情景)")
        lines.append("")
        for label, r in results["full_results"].items():
            flag = "✓ PASS" if r["verdict"] else "✗ FAIL"
            lines.append(f"### {label} — {flag}")
            lines.append("")
            lines.append(f"realized: Sharpe={r['realized']['sharpe']:.4f} MaxDD={r['realized']['maxdd']:.4%} "
                        f"Annual={r['realized']['annual']:.4%}")
            lines.append("")
            lines.append("| 情景 | MaxDD | Sharpe | Δ MaxDD vs v4.4 | Δ Sharpe vs v4.4 |")
            lines.append("|---|---|---|---|---|")
            for sc_name, sc in r["scenarios"].items():
                v44 = V44_SCENARIOS.get(sc_name, {"sharpe": 0, "maxdd": 0})
                dd_d = sc["strat_maxdd"] - v44["maxdd"]
                sh_d = sc["strat_sharpe"] - v44["sharpe"]
                dd_flag = " **破**" if sc["strat_maxdd"] > D_MAX else ""
                dg_flag = " ⚠" if dd_d > MAXDD_TOL or sh_d < SHARPE_TOL else ""
                lines.append(f"| {sc_name} | {sc['strat_maxdd']:.4%}{dd_flag} | "
                           f"{sc['strat_sharpe']:.4f} | {dd_d:+.4%}{dg_flag} | {sh_d:+.4f}{dg_flag} |")
            lines.append("")
            if r["degrad_issues"]:
                lines.append(f"劣化问题: {', '.join(r['degrad_issues'])}")
                lines.append("")
    
    # Conclusion
    lines.append("## 4. 结论与建议")
    lines.append("")
    if results["best"]:
        r = results["full_results"][results["best"]]
        lines.append(f"**胜出组合: {results['best']}**")
        lines.append("")
        lines.append(f"- grey_corr_combo MaxDD: {r['scenarios']['grey_corr_combo']['strat_maxdd']:.4%} ≤ 12% ✓")
        lines.append(f"- bond_bear MaxDD: {r['scenarios']['bond_bear']['strat_maxdd']:.4%} "
                    f"(v4.4={V44_SCENARIOS['bond_bear']['maxdd']:.4%}, "
                    f"Δ={r['scenarios']['bond_bear']['strat_maxdd']-V44_SCENARIOS['bond_bear']['maxdd']:+.4%}) ≤ +0.3pp ✓")
        lines.append(f"- 全 8 情景 MaxDD ≤ 12% ✓")
        lines.append(f"- 逐情景劣化 ≤ Sharpe -0.02 / MaxDD +0.3pp ✓")
        lines.append(f"- realized Sharpe: {r['realized']['sharpe']:.4f} ≥ 1.4885 ✓")
        lines.append(f"- 门控基线关闭率: {r['gate_baseline_close_rate']:.1%} < 5% ✓")
        lines.append("")
        lines.append("### 推荐参数")
        lines.append(f"- threshold: {r['thr']}")
        lines.append(f"- slope: {r['slope']}")
        lines.append(f"- 满格点: corr = {r['thr']} + 0.15/{r['slope']} = {r['thr']+0.15/r['slope']:.4f}")
        lines.append(f"- 门控: {r['gate']} ({descs[r['gate']]})")
        lines.append(f"- 其余不变: window=26, max_boost=0.15, EWMA halflife=8")
        lines.append("")
        lines.append("### src/ 改动面估计")
        lines.append("- `src/engine_core.py` — `_compute_crisis_boost_ewma()` 函数内, "
                    "在 `if max_pair_corr > threshold` 前加 ~8-12 行门控逻辑")
        lines.append("- `src/strategy.py` — StrategyConfig 新增 1 个布尔字段 "
                    "`crisis_corr_gate_enabled` (默认 False, 向后兼容)")
        lines.append("- 总改动量: ~15-20 行 (含注释)")
    else:
        lines.append("**无胜出组合。**")
        lines.append("")
        lines.append("全部组合均未能同时满足 grey/bond_bear/全情景约束。")
        if results["full_results"]:
            lines.append("")
            lines.append("最接近通过的组合:")
            for label, r in list(results["full_results"].items())[:3]:
                lines.append(f"- {label}: issues={r['degrad_issues'][:3]}")
    
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    results = run_experiment()
    
    # Save JSON
    json_path = OUT / "exp_v45_gate_study.json"
    with open(json_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nJSON saved: {json_path}")
    
    # Save MD report
    md_path = OUT / "exp_v45_gate_study.md"
    md = render_md(results)
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Report saved: {md_path}")
