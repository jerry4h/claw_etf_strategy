#!/usr/bin/env python3
"""E2: 成交量因子策略回测 — PVD/VMR 注入 vs Baseline.

基于 E1 GO 结论(price_volume_divergence IC=0.053, t=2.53, 正交),
测试将量因子整合进策略评分层是否带来实质增量.

实验组:
  Baseline     - 无修改 (v4.3 现状)
  PVD-0.3      - score += 0.3 × PVD (轻量注入)
  PVD-0.5      - score += 0.5 × PVD (中等注入)
  PVD-0.7      - score += 0.7 × PVD (激进注入)
  VMR-Boost    - 当 vol_ma_ratio > 1.5, 额外加 0.02 到 momentum
  VolShrink    - 当 avg vol_change < -0.2, momentum 减 0.01 (缩量防守)
  PVD05+VMR    - PVD-0.5 + VMR-Boost 组合

对抗: 7-seed REALIZED DGP (volume factor使用真实历史数据,价格为合成路径)

用法: .venv/bin/python scripts/_exp_volume_signal_e2.py
"""
import contextlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

# --- Import adversarial framework ---
_spec = importlib.util.spec_from_file_location(
    "adv", PROJ / "scripts" / "adversarial_robustness.py")
adv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adv)
dm = adv.dm

# --- Import backtest infrastructure ---
import src.backtest as sbt
import src.factors as sf
from src.backtest import run_backtest
from src.data_loader import ETFS
from src.strategy import load_config

# --- Import volume factor computation ---
from scripts._exp_volume_signal_study import (
    aggregate_weekly_volume, compute_volume_factors, load_daily_full,
    ETF_MAP, NAV_FILE, ALL_ETFS, HL_REAL_START_512890,
)

OUT = PROJ / "output" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)
CFG_PATH = PROJ / "config" / "strategy_v4_3.yaml"

SEEDS = (11, 22, 33, 44, 55, 66, 77)

EXPERIMENTS = [
    ("Baseline",   "无修改"),
    ("PVD-0.3",    "score += 0.3 × PVD"),
    ("PVD-0.5",    "score += 0.5 × PVD"),
    ("PVD-0.7",    "score += 0.7 × PVD"),
    ("VMR-Boost",  "vol_ma_ratio > 1.5 → mom + 0.02"),
    ("VolShrink",  "avg vol_change < -0.2 → mom − 0.01"),
    ("PVD05+VMR",  "PVD-0.5 + VMR-Boost"),
]


# ======================================================================
# Pre-computed volume factors (global state)
# ======================================================================
_pvd_factor = None     # DataFrame (n_weeks, 5) — price_volume_divergence
_vmr_factor = None     # DataFrame (n_weeks, 5) — volume_ma_ratio
_volchg_factor = None  # DataFrame (n_weeks, 5) — volume_change
_active_experiment = None


def prepare_volume_factors():
    """Compute volume factors from real daily data."""
    global _pvd_factor, _vmr_factor, _volchg_factor
    print("[E2 prep] Computing volume factors from real daily data...")
    nav = pd.read_csv(NAV_FILE, index_col=0, parse_dates=True)
    weekly_vol, weekly_amt = aggregate_weekly_volume(nav.index)
    factors = compute_volume_factors(weekly_vol, weekly_amt, nav)

    # Handle 红利低波 pre-2019
    for fname in factors:
        mask_512890 = factors[fname].index < HL_REAL_START_512890
        factors[fname].loc[mask_512890, "红利低波ETF"] = np.nan

    _pvd_factor = factors["price_volume_divergence"]
    _vmr_factor = factors["volume_ma_ratio"]
    _volchg_factor = factors["volume_change"]
    print(f"  PVD: {_pvd_factor.notna().sum().sum()} non-null values")
    print(f"  VMR: {_vmr_factor.notna().sum().sum()} non-null values")
    print(f"  VolChg: {_volchg_factor.notna().sum().sum()} non-null values")


# ======================================================================
# Monkeypatch machinery
# ======================================================================
_original_caf = sf.compute_all_factors


def _patched_compute_all_factors(*args, **kwargs):
    """Wrapper that injects volume factor into momentum scores."""
    factors = _original_caf(*args, **kwargs)
    if _active_experiment is None or _active_experiment == "Baseline":
        return factors

    mom = factors['momentum'].copy()
    nav_idx = mom.index

    # Align volume factors to whatever nav_index the backtest uses
    # (may be real or synthetic, but same dates)
    pvd_aligned = _pvd_factor.reindex(nav_idx) if _pvd_factor is not None else None
    vmr_aligned = _vmr_factor.reindex(nav_idx) if _vmr_factor is not None else None
    vc_aligned = _volchg_factor.reindex(nav_idx) if _volchg_factor is not None else None

    exp = _active_experiment

    if exp in ("PVD-0.3", "PVD-0.5", "PVD-0.7", "PVD05+VMR"):
        w = {"PVD-0.3": 0.3, "PVD-0.5": 0.5, "PVD-0.7": 0.7, "PVD05+VMR": 0.5}[exp]
        if pvd_aligned is not None:
            # PVD is correlation [-1, 1], positive PVD = price & volume move together
            # We want: high PVD (confirming trend) → boost momentum
            pvd_vals = pvd_aligned.values
            mask = ~np.isnan(pvd_vals) & ~np.isnan(mom.values)
            adjustment = np.where(mask, w * pvd_vals, 0.0)
            mom.iloc[:, :] = mom.values + adjustment

    if exp in ("VMR-Boost", "PVD05+VMR"):
        if vmr_aligned is not None:
            # When volume_ma_ratio > 1.5 (above average), add small boost
            vmr_vals = vmr_aligned.values
            boost_mask = (~np.isnan(vmr_vals)) & (vmr_vals > 1.5) & (~np.isnan(mom.values))
            mom.iloc[:, :] = np.where(boost_mask, mom.values + 0.02, mom.values)

    if exp == "VolShrink":
        if vc_aligned is not None:
            # When average volume_change across ETFs < -0.2 (market-wide shrinkage)
            vc_vals = vc_aligned.values
            row_mean = np.nanmean(vc_vals, axis=1)
            for i in range(len(nav_idx)):
                if row_mean[i] < -0.2:
                    # Penalize all offensive scores → strategy prefers defense
                    for j in range(mom.shape[1]):
                        if not np.isnan(mom.values[i, j]):
                            mom.iloc[i, j] -= 0.01

    factors['momentum'] = mom
    return factors


def set_experiment(name):
    global _active_experiment
    _active_experiment = name
    sbt.compute_all_factors = _patched_compute_all_factors


def clear_experiment():
    global _active_experiment
    _active_experiment = None
    sbt.compute_all_factors = _original_caf


# ======================================================================
# Experiment runners
# ======================================================================
def run_real(cfg, exp_name):
    """Run backtest on real data."""
    set_experiment(exp_name)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = run_backtest(cfg, start_date=dm.START_DATE)
        return {
            "sharpe": float(res.metrics["sharpe_ratio"]),
            "maxdd": float(res.metrics["max_drawdown"]),
            "annual_ret": float(res.metrics["annual_return"]),
            "calmar": float(res.metrics["calmar_ratio"]),
            "total_weeks": int(res.metrics["total_weeks"]),
        }
    finally:
        clear_experiment()


def run_adversarial_seeds(cfg, exp_name, mu, A, R, nu, gp, T, real_dates, first_nav):
    """Run 7-seed adversarial."""
    results = []
    for seed in SEEDS:
        params = dict(adv.REALIZED)
        r = adv.gen_regime_corr(mu, A, R, nu, gp, params, T, seed)
        nav_df = dm.build_nav_df(r, real_dates, first_nav)

        tmp = OUT / f"_e2vol_synth_{seed}_{os.getpid()}.csv"
        nav_df.to_csv(tmp, encoding="utf-8")
        try:
            set_experiment(exp_name)
            with contextlib.redirect_stdout(io.StringIO()):
                res = run_backtest(cfg, start_date=dm.START_DATE, data_path=str(tmp))
            if res.nav_series.empty:
                continue
            results.append({
                "seed": seed,
                "sharpe": float(res.metrics["sharpe_ratio"]),
                "maxdd": float(res.metrics["max_drawdown"]),
                "annual_ret": float(res.metrics["annual_return"]),
            })
        finally:
            clear_experiment()
            if tmp.exists():
                os.remove(tmp)
    return results


def median_val(results, key):
    vals = [r[key] for r in results if key in r]
    return float(np.median(vals)) if vals else float("nan")


# ======================================================================
# Gate decision
# ======================================================================
def gate_decision(results):
    """Apply go/no-go gate #2."""
    baseline = results["Baseline"]
    base_sharpe = baseline["real"]["sharpe"]
    base_maxdd = baseline["real"]["maxdd"]
    base_adv_sharpe = baseline["adv_median_sharpe"]

    # Find best PVD experiment
    pvd_exps = ["PVD-0.3", "PVD-0.5", "PVD-0.7"]
    best_pvd = max(pvd_exps, key=lambda e: results[e]["real"]["sharpe"])
    best_pvd_delta = results[best_pvd]["real"]["sharpe"] - base_sharpe
    best_pvd_maxdd_pp = (results[best_pvd]["real"]["maxdd"] - base_maxdd) * 100

    # Find best overall
    all_treatments = [e[0] for e in EXPERIMENTS if e[0] != "Baseline"]
    best_overall = max(all_treatments, key=lambda e: results[e]["real"]["sharpe"])
    best_delta = results[best_overall]["real"]["sharpe"] - base_sharpe
    best_maxdd_pp = (results[best_overall]["real"]["maxdd"] - base_maxdd) * 100
    best_adv = results[best_overall]["adv_median_sharpe"] - base_adv_sharpe

    # Gate criteria
    go = best_delta >= 0.02 and best_maxdd_pp <= 0.3 and best_adv >= 0
    nogo = best_delta < 0.005 or best_maxdd_pp > 1.0

    if go:
        verdict = "GO"
    elif nogo:
        verdict = "NO-GO"
    else:
        verdict = "CONDITIONAL"

    return {
        "verdict": verdict,
        "best_experiment": best_overall,
        "criteria": {
            "best_sharpe_delta": {"value": best_delta, "threshold": 0.02,
                                   "pass": best_delta >= 0.02},
            "best_maxdd_pp": {"value": best_maxdd_pp, "threshold": 0.3,
                              "pass": best_maxdd_pp <= 0.3},
            "best_adv_deficit": {"value": best_adv, "threshold": 0.0,
                                 "pass": best_adv >= 0},
        },
        "best_pvd": best_pvd,
        "best_pvd_delta": best_pvd_delta,
    }


# ======================================================================
# Report
# ======================================================================
def render_report(results, gate):
    L = ["# E2-Volume: 成交量因子策略回测报告", ""]
    L.append(f"> PVD/VMR/VolShrink 注入 vs Baseline | 门禁 #2 判定: **{gate['verdict']}**")
    L.append("")

    L.append("## 门禁 #2 判定")
    L.append("")
    L.append(f"**结论: {gate['verdict']}**")
    L.append(f"\n最优实验: {gate['best_experiment']}")
    L.append("")
    L.append("| 门禁条件 | 要求 | 实际 | 判定 |")
    L.append("|---|---|---|---|")
    gc = gate["criteria"]
    L.append(f"| Sharpe 改善 (最优) | ≥+0.02 | {gc['best_sharpe_delta']['value']:+.4f} | "
             f"{'✓' if gc['best_sharpe_delta']['pass'] else '✗'} |")
    L.append(f"| MaxDD 恶化 | ≤+0.3pp | {gc['best_maxdd_pp']['value']:+.2f}pp | "
             f"{'✓' if gc['best_maxdd_pp']['pass'] else '✗'} |")
    L.append(f"| 对抗中位 Sharpe | ≥基线 | {gc['best_adv_deficit']['value']:+.4f} | "
             f"{'✓' if gc['best_adv_deficit']['pass'] else '✗'} |")
    L.append("")

    # Real results
    L.append("## 真实历史路径对比")
    L.append("")
    L.append("| 实验 | Sharpe | MaxDD | 年化收益 | Calmar | ΔSharpe |")
    L.append("|---|---|---|---|---|---|")
    base_sh = results["Baseline"]["real"]["sharpe"]
    for name, desc in EXPERIMENTS:
        r = results[name]["real"]
        delta = r["sharpe"] - base_sh
        L.append(f"| {name} | {r['sharpe']:.4f} | {r['maxdd']:.2%} | "
                 f"{r['annual_ret']:.2%} | {r['calmar']:.3f} | {delta:+.4f} |")
    L.append("")

    # Adversarial
    L.append("## 对抗鲁棒性 (7-seed REALIZED DGP)")
    L.append("")
    L.append("| 实验 | 中位 Sharpe | 中位 MaxDD | 中位年化 | Δ中位 Sharpe |")
    L.append("|---|---|---|---|---|")
    base_adv = results["Baseline"]["adv_median_sharpe"]
    for name, desc in EXPERIMENTS:
        ams = results[name]["adv_median_sharpe"]
        amd = results[name]["adv_median_maxdd"]
        amr = results[name]["adv_median_ret"]
        delta = ams - base_adv
        L.append(f"| {name} | {ams:.4f} | {amd:.2%} | {amr:.2%} | {delta:+.4f} |")
    L.append("")

    # Per-seed detail for best
    best = gate["best_experiment"]
    L.append(f"## 最优实验 ({best}) 逐 seed 明细")
    L.append("")
    L.append("| seed | Sharpe | MaxDD | 年化 |")
    L.append("|---|---|---|---|")
    for sr in results[best].get("adv_per_seed", []):
        L.append(f"| {sr['seed']} | {sr['sharpe']:.4f} | {sr['maxdd']:.2%} | {sr['annual_ret']:.2%} |")
    L.append("")

    # Key insight
    L.append("## 关键洞察")
    L.append("")
    L.append("成交量信号在 E1 展现了统计显著的截面预测力 (PVD IC=0.053)，")
    L.append("但从 E1 IC 到 E2 策略 Sharpe 的转化取决于：")
    L.append("(1) 5 只 ETF 窄截面的实际调仓机会, ")
    L.append("(2) 量因子与现有评分的方向一致性, ")
    L.append("(3) 对抗路径下因子失效风险。")
    L.append("")
    return "\n".join(L)


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 70)
    print(" E2-Volume: 成交量因子策略回测")
    print("=" * 70)

    # Prepare
    prepare_volume_factors()
    cfg = load_config(CFG_PATH)

    # DGP params for adversarial (same as standard)
    nav_real, wk, w_rets = dm.load_real()
    mu, A, Sigma, nu, resid, coords = dm.fit_var_t(w_rets)
    gp, R = adv.fit_garch(resid)
    T = len(w_rets)
    real_dates = wk.index
    first_nav = wk.iloc[0].values

    all_results = {}

    # Run experiments
    for exp_name, desc in EXPERIMENTS:
        print(f"\n{'='*50}")
        print(f"  Experiment: {exp_name} — {desc}")
        print(f"{'='*50}")

        # Real data
        print(f"  [real] Running...")
        real_res = run_real(cfg, exp_name)
        print(f"  [real] Sharpe={real_res['sharpe']:.4f}, MaxDD={real_res['maxdd']:.2%}")

        # Adversarial
        print(f"  [adv] Running 7 seeds...")
        adv_results = run_adversarial_seeds(
            cfg, exp_name, mu, A, R, nu, gp, T, real_dates, first_nav)
        adv_med_sharpe = median_val(adv_results, "sharpe")
        adv_med_maxdd = median_val(adv_results, "maxdd")
        adv_med_ret = median_val(adv_results, "annual_ret")
        print(f"  [adv] Median Sharpe={adv_med_sharpe:.4f}, MaxDD={adv_med_maxdd:.2%}")

        all_results[exp_name] = {
            "real": real_res,
            "adv_per_seed": adv_results,
            "adv_median_sharpe": adv_med_sharpe,
            "adv_median_maxdd": adv_med_maxdd,
            "adv_median_ret": adv_med_ret,
        }

    # Gate
    print("\n" + "=" * 70)
    gate = gate_decision(all_results)
    print(f" 门禁 #2 判定: **{gate['verdict']}**")
    print(f"  最优实验: {gate['best_experiment']}")
    gc = gate["criteria"]
    print(f"  ΔSharpe={gc['best_sharpe_delta']['value']:+.4f} "
          f"({'PASS' if gc['best_sharpe_delta']['pass'] else 'FAIL'})")
    print(f"  ΔMaxDD={gc['best_maxdd_pp']['value']:+.2f}pp "
          f"({'PASS' if gc['best_maxdd_pp']['pass'] else 'FAIL'})")
    print(f"  Δadv={gc['best_adv_deficit']['value']:+.4f} "
          f"({'PASS' if gc['best_adv_deficit']['pass'] else 'FAIL'})")
    print("=" * 70)

    # Save
    json_path = OUT / "exp_volume_signal_e2.json"
    with open(json_path, "w") as f:
        json.dump({"results": all_results, "gate": gate}, f,
                  ensure_ascii=False, indent=2, default=str)
    print(f"\n  JSON saved: {json_path}")

    md = render_report(all_results, gate)
    md_path = OUT / "exp_volume_signal_e2.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  Report saved: {md_path}")


if __name__ == "__main__":
    main()
