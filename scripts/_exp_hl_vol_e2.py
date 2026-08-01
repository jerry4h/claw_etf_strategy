#!/usr/bin/env python3
"""E2: 策略回测 A/B 对比 — Mixed Parkinson vol vs CC-vol baseline.

基于 E1 结论(纳指/国债 Parkinson 失效, 中证500/黄金/红利低波可用),
构造"分资产 Mixed vol"方案, 与全 CC-vol 基线对比回测, 判定 go/no-go 门禁 #2.

实验组:
  Baseline   - 全 CC-vol (v4.3 现状)
  Mixed      - 纳指+国债 CC, 中证500/黄金/红利低波 Parkinson
  Full-P     - 全 Parkinson (负面对照)
  Mixed-L1   - 仅评分层用 Mixed vol (cols 2,3)
  Mixed-L2   - N/A (L2 inv-vol 独立用 w_rets 计算, 与 vol factor 无关)
  Mixed-L3   - 仅 M3 防御路径用 Mixed vol (col 2)
  Mixed-Def  - 仅 DefAlloc 用 Mixed vol (col 1)

对抗: 7-seed baseline DGP (REALIZED), 每个实验取中位 Sharpe.

用法: .venv/bin/python scripts/_exp_hl_vol_e2.py
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

# --- Import adversarial framework (also loads data_manifold as dm) ---
_spec = importlib.util.spec_from_file_location(
    "adv", PROJ / "scripts" / "adversarial_robustness.py")
adv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adv)
dm = adv.dm

# --- Import backtest infrastructure ---
import src.backtest as sbt
import src.factors as sf
from src.backtest import run_backtest, compute_metrics
from src.data_loader import ETFS
from src.strategy import load_config

# --- Import E0 functions ---
from scripts._exp_hl_vol_study import (
    ETF_MAP, NAV_FILE, aggregate_weekly_ohlc,
    calc_vol_parkinson, calc_vol_cc_tapered,
)

OUT = PROJ / "output" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)
CFG_PATH = PROJ / "config" / "strategy_v4_3.yaml"

SEEDS = (11, 22, 33, 44, 55, 66, 77)

# ETF indices: 0=纳指, 1=红利低波, 2=中证500, 3=黄金, 4=国债
REPLACE_ETFS_FULL = ['红利低波ETF', '中证500ETF', '黄金ETF']  # Mixed cols [1,2,3]
REPLACE_ETFS_ALL = ETFS  # Full-P: all 5


# ======================================================================
# Data preparation
# ======================================================================
def prepare_vol_data():
    """Compute Parkinson vol and P/CC ratio from real OHLC data."""
    print("[E2 prep] Computing Parkinson vol from real OHLC...")
    weekly_ohlc = aggregate_weekly_ohlc()
    nav = pd.read_csv(NAV_FILE, index_col=0, parse_dates=True)

    high_df = pd.DataFrame(index=nav.index, columns=nav.columns, dtype=float)
    low_df = pd.DataFrame(index=nav.index, columns=nav.columns, dtype=float)
    for code, col_name in ETF_MAP.items():
        wk = weekly_ohlc[code]
        high_df[col_name] = wk["high"]
        low_df[col_name] = wk["low"]

    vol_p = calc_vol_parkinson(high_df, low_df, window=14)

    # Production CC-vol (window=14, taper=7 as per v4.3 config)
    from src.factors import calculate_volatility_tapered
    vol_cc_prod = calculate_volatility_tapered(nav, window=14, taper=7)

    # Compute per-ETF median ratio (Parkinson / CC) for adversarial scaling
    ratios = {}
    for col in ETFS:
        mask = vol_p[col].notna() & vol_cc_prod[col].notna() & (vol_cc_prod[col] > 1e-6)
        if mask.sum() > 50:
            r = (vol_p.loc[mask, col] / vol_cc_prod.loc[mask, col]).median()
            ratios[col] = float(r)
        else:
            ratios[col] = 1.0
    print(f"  Ratios (P/CC): { {k: f'{v:.3f}' for k,v in ratios.items()} }")
    return vol_p, ratios


# ======================================================================
# Monkeypatch machinery
# ======================================================================
_original_caf = sf.compute_all_factors
_vol_p_real = None   # Will be set in main
_vol_ratios = None   # Will be set in main
_active_replace_cols = []
_patch_mode = None   # 'real' or 'ratio'


def _patched_compute_all_factors(weekly_nav, pe_df=None, config=None):
    """Wrapper that replaces specific vol columns after normal computation."""
    factors = _original_caf(weekly_nav, pe_df, config)
    if not _active_replace_cols:
        return factors

    vol = factors['volatility']

    if _patch_mode == 'real':
        # Real data: inject actual Parkinson vol (aligned by index)
        for col in _active_replace_cols:
            if col in vol.columns and col in _vol_p_real.columns:
                aligned = _vol_p_real[col].reindex(vol.index)
                mask = aligned.notna()
                vol.loc[mask, col] = aligned[mask]
    elif _patch_mode == 'ratio':
        # Adversarial data: scale CC-vol by historical P/CC ratio
        for col in _active_replace_cols:
            if col in vol.columns and col in _vol_ratios:
                vol[col] = vol[col] * _vol_ratios[col]

    factors['volatility'] = vol
    return factors


def set_patch(replace_cols, mode):
    """Activate vol replacement patch."""
    global _active_replace_cols, _patch_mode
    _active_replace_cols = replace_cols
    _patch_mode = mode
    sbt.compute_all_factors = _patched_compute_all_factors


def clear_patch():
    """Deactivate patch."""
    global _active_replace_cols, _patch_mode
    _active_replace_cols = []
    _patch_mode = None
    sbt.compute_all_factors = _original_caf


# ======================================================================
# Experiment configurations
# ======================================================================
EXPERIMENTS = [
    ("Baseline",  [],                              "全 CC-vol (v4.3 现状)"),
    ("Mixed",     REPLACE_ETFS_FULL,               "纳指+国债 CC; 中证500/黄金/红利低波 Parkinson"),
    ("Full-P",    list(ETFS),                      "全 Parkinson (负面对照)"),
    ("Mixed-L1",  ['中证500ETF', '黄金ETF'],       "仅评分层进攻 ETF 用 Parkinson (cols 2,3)"),
    ("Mixed-L2",  [],                              "N/A (L2 inv-vol 用 w_rets, 与 vol factor 无关)"),
    ("Mixed-L3",  ['中证500ETF'],                  "仅 M3 防御路径 (col 2, ashare boost)"),
    ("Mixed-Def", ['红利低波ETF'],                  "仅 DefAlloc 动态红利比 (col 1)"),
]


# ======================================================================
# Single experiment runner
# ======================================================================
def run_real(cfg, replace_cols):
    """Run backtest on real data with specified vol replacement."""
    set_patch(replace_cols, 'real')
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
        clear_patch()


def run_adversarial_seeds(cfg, replace_cols, mu, A, R, nu, gp, T,
                          real_dates, first_nav):
    """Run 7-seed adversarial with specified vol replacement. Return per-seed metrics."""
    results = []
    for seed in SEEDS:
        # Generate synthetic path
        params = dict(adv.REALIZED)
        r = adv.gen_regime_corr(mu, A, R, nu, gp, params, T, seed)
        nav_df = dm.build_nav_df(r, real_dates, first_nav)

        # Save to temp file
        tmp = OUT / f"_e2_synth_{seed}_{os.getpid()}.csv"
        nav_df.to_csv(tmp, encoding="utf-8")
        try:
            set_patch(replace_cols, 'ratio')
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
            clear_patch()
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
    """Apply go/no-go gate #2 rules."""
    baseline = results["Baseline"]
    mixed = results["Mixed"]

    sharpe_delta = mixed["real"]["sharpe"] - baseline["real"]["sharpe"]
    maxdd_delta_pp = (mixed["real"]["maxdd"] - baseline["real"]["maxdd"]) * 100

    # Adversarial median Sharpe
    base_adv_sharpe = baseline["adv_median_sharpe"]
    mixed_adv_sharpe = mixed["adv_median_sharpe"]
    adv_deficit = mixed_adv_sharpe - base_adv_sharpe

    # Find best single-layer experiment
    layer_exps = ["Mixed-L1", "Mixed-L3", "Mixed-Def"]
    best_layer = max(layer_exps, key=lambda e: results[e]["real"]["sharpe"])
    best_layer_delta = results[best_layer]["real"]["sharpe"] - baseline["real"]["sharpe"]

    effective_delta = max(sharpe_delta, best_layer_delta)

    # GO: delta >= +0.02 AND maxdd not worse >0.3pp AND adversarial not worse
    go = effective_delta >= 0.02 and maxdd_delta_pp <= 0.3 and adv_deficit >= 0
    # CONDITIONAL: +0.01~+0.02 and no degradation
    conditional = (0.01 <= effective_delta < 0.02 and
                   maxdd_delta_pp <= 0.5 and adv_deficit >= -0.02)
    # NO-GO: delta < 0.01 or maxdd worse >0.5pp or adversarial worse
    no_go = effective_delta < 0.01 or maxdd_delta_pp > 0.5 or adv_deficit < -0.05

    if go:
        verdict = "GO"
    elif conditional and not no_go:
        verdict = "CONDITIONAL GO"
    else:
        verdict = "NO-GO"

    return {
        "verdict": verdict,
        "criteria": {
            "sharpe_delta_mixed": {
                "value": sharpe_delta, "threshold": 0.02, "pass": sharpe_delta >= 0.02
            },
            "sharpe_delta_best_layer": {
                "value": best_layer_delta, "best_layer": best_layer,
                "threshold": 0.02, "pass": best_layer_delta >= 0.02
            },
            "effective_delta": effective_delta,
            "maxdd_degradation_pp": {
                "value": maxdd_delta_pp, "threshold": 0.3, "pass": maxdd_delta_pp <= 0.3
            },
            "adversarial_deficit": {
                "value": adv_deficit, "threshold": 0.0, "pass": adv_deficit >= 0
            },
        },
        "full_p_sharpe": results["Full-P"]["real"]["sharpe"],
        "baseline_sharpe": baseline["real"]["sharpe"],
    }


# ======================================================================
# Report generation
# ======================================================================
def render_report(results, gate):
    L = ["# E2: 策略回测 A/B 对比报告", ""]
    L.append(f"> Mixed Parkinson vol vs CC-vol baseline | 门禁 #2 判定: **{gate['verdict']}**")
    L.append("")

    # Gate summary
    L.append("## 门禁 #2 判定")
    L.append("")
    L.append(f"**结论: {gate['verdict']}**")
    L.append("")
    L.append("| 门禁条件 | 要求 | 实际 | 判定 |")
    L.append("|---|---|---|---|")
    gc = gate["criteria"]
    L.append(f"| Sharpe 改善 (Mixed) | ≥+0.02 | {gc['sharpe_delta_mixed']['value']:+.4f} | "
             f"{'✓' if gc['sharpe_delta_mixed']['pass'] else '✗'} |")
    L.append(f"| Sharpe 改善 (最优层 {gc['sharpe_delta_best_layer']['best_layer']}) | ≥+0.02 | "
             f"{gc['sharpe_delta_best_layer']['value']:+.4f} | "
             f"{'✓' if gc['sharpe_delta_best_layer']['pass'] else '✗'} |")
    L.append(f"| MaxDD 恶化 | ≤+0.3pp | {gc['maxdd_degradation_pp']['value']:+.2f}pp | "
             f"{'✓' if gc['maxdd_degradation_pp']['pass'] else '✗'} |")
    L.append(f"| 对抗中位 Sharpe | ≥基线 | {gc['adversarial_deficit']['value']:+.4f} | "
             f"{'✓' if gc['adversarial_deficit']['pass'] else '✗'} |")
    L.append("")

    # Experiment results table
    L.append("## 实验组对比（真实历史路径）")
    L.append("")
    L.append("| 实验 | Sharpe | MaxDD | 年化收益 | Calmar | Δ Sharpe |")
    L.append("|---|---|---|---|---|---|")
    base_sh = results["Baseline"]["real"]["sharpe"]
    for name, _, desc in EXPERIMENTS:
        r = results[name]["real"]
        delta = r["sharpe"] - base_sh
        L.append(f"| {name} | {r['sharpe']:.4f} | {r['maxdd']:.2%} | "
                 f"{r['annual_ret']:.2%} | {r['calmar']:.3f} | {delta:+.4f} |")
    L.append("")
    L.append("注: Mixed-L2 = Baseline (L2 inv-vol 独立计算, 与 vol factor 无关)")
    L.append("")

    # Adversarial results
    L.append("## 对抗鲁棒性 (7-seed baseline DGP, 中位数)")
    L.append("")
    L.append("| 实验 | 中位 Sharpe | 中位 MaxDD | 中位年化 | Δ中位 Sharpe |")
    L.append("|---|---|---|---|---|")
    base_adv = results["Baseline"]["adv_median_sharpe"]
    for name, _, desc in EXPERIMENTS:
        ams = results[name]["adv_median_sharpe"]
        amd = results[name]["adv_median_maxdd"]
        amr = results[name]["adv_median_ret"]
        delta = ams - base_adv
        L.append(f"| {name} | {ams:.4f} | {amd:.2%} | {amr:.2%} | {delta:+.4f} |")
    L.append("")

    # Internal consistency check
    L.append("## 内部一致性检验")
    L.append("")
    fp_sh = results["Full-P"]["real"]["sharpe"]
    fp_worse = fp_sh < base_sh
    L.append(f"- Full-P (全 Parkinson) Sharpe = {fp_sh:.4f} vs Baseline = {base_sh:.4f}: "
             f"{'✓ Full-P 最差 (预期)' if fp_worse else '⚠ Full-P 非最差 (异常)'}")
    L.append("")

    # Layer decomposition
    L.append("## 层级消融分析")
    L.append("")
    L.append("| 层级 | 替换列 | ΔSharpe | 作用路径 |")
    L.append("|---|---|---|---|")
    for name, cols, desc in EXPERIMENTS:
        if name.startswith("Mixed-"):
            delta = results[name]["real"]["sharpe"] - base_sh
            col_str = str(cols) if cols else "无"
            L.append(f"| {name} | {col_str} | {delta:+.4f} | {desc} |")
    L.append("")
    L.append("Mixed 整体 ΔSharpe 应≈各层之和（非严格加性因交互效应）。")
    L.append("")

    # Description
    L.append("## 方法说明")
    L.append("")
    L.append("- **Monkeypatch**: 替换 `src.factors.compute_all_factors` 返回值中的 volatility 列")
    L.append("- **真实路径**: 对替换列直接注入 E0 计算的 Parkinson vol (window=14)")
    L.append("- **对抗路径**: 对替换列的 CC-vol 乘以历史中位 P/CC 比值（保持波动水平响应）")
    L.append("- **Mixed-L2 无法测试**: inv_vol_weights 从 w_rets 独立计算 vol, 不经过 vol factor")
    L.append("- **层级泄漏**: Mixed-L1 替换 cols [2,3] 同时影响 M3 (col 2); 已在报告中注明")
    L.append("")
    return "\n".join(L)


# ======================================================================
# Main
# ======================================================================
def main():
    global _vol_p_real, _vol_ratios
    t0 = time.time()

    print("=" * 70)
    print(" E2: 策略回测 A/B 对比 — Mixed Parkinson vol")
    print("=" * 70)

    cfg = load_config(CFG_PATH)

    # --- Prepare data ---
    vol_p, ratios = prepare_vol_data()
    _vol_p_real = vol_p
    _vol_ratios = ratios

    # --- Fit DGP for adversarial ---
    print("\n[E2 DGP] Fitting VAR(1)-t + CCC-GARCH for adversarial...")
    nav_real, wk_real, w_rets_real = dm.load_real()
    mu, A, Sigma, nu, resid, coords = dm.fit_var_t(w_rets_real)
    gp, R = adv.fit_garch(resid)
    real_dates = wk_real.index
    first_nav = wk_real.iloc[0].values
    T = len(w_rets_real)
    print(f"  T={T} weeks, ν={nu:.1f}")

    # --- Run experiments ---
    all_results = {}
    for name, replace_cols, desc in EXPERIMENTS:
        print(f"\n[{name}] {desc}")

        # Real path
        t1 = time.time()
        real_metrics = run_real(cfg, replace_cols)
        print(f"  Real: Sharpe={real_metrics['sharpe']:.4f}, "
              f"MaxDD={real_metrics['maxdd']:.2%}, Ann={real_metrics['annual_ret']:.2%} "
              f"({time.time()-t1:.1f}s)")

        # Adversarial (7 seeds)
        t1 = time.time()
        adv_results = run_adversarial_seeds(
            cfg, replace_cols, mu, A, R, nu, gp, T, real_dates, first_nav)
        adv_med_sh = median_val(adv_results, "sharpe")
        adv_med_dd = median_val(adv_results, "maxdd")
        adv_med_ret = median_val(adv_results, "annual_ret")
        print(f"  Adversarial (7-seed median): Sharpe={adv_med_sh:.4f}, "
              f"MaxDD={adv_med_dd:.2%} ({time.time()-t1:.1f}s)")

        all_results[name] = {
            "real": real_metrics,
            "adversarial": adv_results,
            "adv_median_sharpe": adv_med_sh,
            "adv_median_maxdd": adv_med_dd,
            "adv_median_ret": adv_med_ret,
            "description": desc,
            "replace_cols": replace_cols,
        }

    # --- Gate decision ---
    print("\n" + "=" * 70)
    gate = gate_decision(all_results)
    print(f" 门禁 #2 判定: **{gate['verdict']}**")
    print("=" * 70)
    gc = gate["criteria"]
    print(f"  [1] Sharpe Δ (Mixed): {gc['sharpe_delta_mixed']['value']:+.4f} "
          f"{'PASS' if gc['sharpe_delta_mixed']['pass'] else 'FAIL'}")
    print(f"  [2] Sharpe Δ (best layer {gc['sharpe_delta_best_layer']['best_layer']}): "
          f"{gc['sharpe_delta_best_layer']['value']:+.4f} "
          f"{'PASS' if gc['sharpe_delta_best_layer']['pass'] else 'FAIL'}")
    print(f"  [3] MaxDD 恶化: {gc['maxdd_degradation_pp']['value']:+.2f}pp "
          f"{'PASS' if gc['maxdd_degradation_pp']['pass'] else 'FAIL'}")
    print(f"  [4] 对抗中位 Sharpe deficit: {gc['adversarial_deficit']['value']:+.4f} "
          f"{'PASS' if gc['adversarial_deficit']['pass'] else 'FAIL'}")
    print(f"  Full-P Sharpe={gate['full_p_sharpe']:.4f} (应为最差)")

    # --- Save outputs ---
    output_data = {
        "experiments": all_results,
        "gate_decision": gate,
        "config": str(CFG_PATH),
        "seeds": list(SEEDS),
        "replace_etfs_mixed": REPLACE_ETFS_FULL,
        "vol_ratios": ratios,
    }
    json_path = OUT / "exp_hl_vol_e2.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  JSON saved: {json_path}")

    md = render_report(all_results, gate)
    md_path = OUT / "exp_hl_vol_e2.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  Report saved: {md_path}")
    print(f"\n  Total time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
