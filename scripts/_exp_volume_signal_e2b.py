#!/usr/bin/env python3
"""E2b: PVD 因子精细化策略整合 — 四方案 + Block Bootstrap 鲁棒性。

方案 A: 极低权重精细网格 (pvd_w 0.05~0.20)
方案 B: 非线性阈值过滤 (底部排除/顶部加成/衰减)
方案 C: 条件激活 (vol 区间 + score 差距)
方案 D: 防御层集成 (全市场缩量→提高防御)

鲁棒性: Block Bootstrap (13-week blocks, 200 paths) 替代合成 DGP

用法: .venv/bin/python scripts/_exp_volume_signal_e2b.py
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

import src.backtest as sbt
import src.factors as sf
from src.backtest import run_backtest
from src.strategy import load_config

from scripts._exp_volume_signal_study import (
    aggregate_weekly_volume, compute_volume_factors, load_daily_full,
    ETF_MAP, NAV_FILE, ALL_ETFS, HL_REAL_START_512890,
)

OUT = PROJ / "output" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)
CFG_PATH = PROJ / "config" / "strategy_v4_3.yaml"
START_DATE = "2013-05-17"

# ======================================================================
# Pre-computed volume data (global)
# ======================================================================
_pvd_factor = None      # price_volume_divergence (n_weeks, 5)
_vmr_factor = None      # volume_ma_ratio
_volchg_factor = None   # volume_change
_weekly_amt = None      # weekly total amount (n_weeks, 5)
_active_exp = None
_active_params = {}


def prepare_data():
    global _pvd_factor, _vmr_factor, _volchg_factor, _weekly_amt
    print("[prep] Computing volume factors...")
    nav = pd.read_csv(NAV_FILE, index_col=0, parse_dates=True)
    weekly_vol, weekly_amt = aggregate_weekly_volume(nav.index)
    factors = compute_volume_factors(weekly_vol, weekly_amt, nav)
    for fname in factors:
        mask = factors[fname].index < HL_REAL_START_512890
        factors[fname].loc[mask, "红利低波ETF"] = np.nan
    _pvd_factor = factors["price_volume_divergence"]
    _vmr_factor = factors["volume_ma_ratio"]
    _volchg_factor = factors["volume_change"]
    _weekly_amt = weekly_amt
    # Compute 26-week rolling median of total amount for Plan D
    _weekly_amt.attrs["total_med26"] = _weekly_amt.sum(axis=1).rolling(26, min_periods=13).median()
    print(f"  PVD: {_pvd_factor.notna().sum().sum()} values")


# ======================================================================
# Monkeypatch
# ======================================================================
_original_caf = sf.compute_all_factors


def _patched_caf(weekly_nav, pe_df=None, config=None):
    factors = _original_caf(weekly_nav, pe_df, config)
    if _active_exp is None or _active_exp == "Baseline":
        return factors

    mom = factors['momentum'].copy()
    vol = factors['volatility'].copy()
    nav_idx = mom.index
    n_etfs = mom.shape[1]

    pvd = _pvd_factor.reindex(nav_idx).values if _pvd_factor is not None else None
    vmr = _vmr_factor.reindex(nav_idx).values if _vmr_factor is not None else None
    vc = _volchg_factor.reindex(nav_idx).values if _volchg_factor is not None else None
    mom_v = mom.values.copy()
    vol_v = vol.values.copy()

    exp = _active_exp
    p = _active_params

    # ---- Plan A: linear low-weight injection ----
    if exp == "A":
        w = p.get("pvd_w", 0.10)
        if pvd is not None:
            mask = ~np.isnan(pvd) & ~np.isnan(mom_v)
            mom_v = np.where(mask, mom_v + w * pvd, mom_v)

    # ---- Plan B1: bottom filter (worst PVD ETF gets score = -inf) ----
    elif exp == "B1":
        if pvd is not None:
            for i in range(len(nav_idx)):
                row = pvd[i]
                valid = ~np.isnan(row) & ~np.isnan(mom_v[i])
                if valid.sum() >= 3:
                    # Find lowest PVD among valid
                    min_j = -1
                    min_val = np.inf
                    for j in range(n_etfs):
                        if valid[j] and row[j] < min_val:
                            min_val = row[j]
                            min_j = j
                    if min_j >= 0:
                        mom_v[i, min_j] = -np.inf  # excluded from selection

    # ---- Plan B2: top bonus (best PVD ETF gets +bonus) ----
    elif exp == "B2":
        bonus = p.get("bonus", 0.03)
        if pvd is not None:
            for i in range(len(nav_idx)):
                row = pvd[i]
                valid = ~np.isnan(row) & ~np.isnan(mom_v[i])
                if valid.sum() >= 3:
                    max_j = -1
                    max_val = -np.inf
                    for j in range(n_etfs):
                        if valid[j] and row[j] > max_val:
                            max_val = row[j]
                            max_j = j
                    if max_j >= 0:
                        mom_v[i, max_j] += bonus

    # ---- Plan B3: decay for low PVD (< 20th percentile → score × decay) ----
    elif exp == "B3":
        decay = p.get("decay", 0.8)
        if pvd is not None:
            # Pre-compute 20th percentile of PVD across all valid
            all_pvd = pvd[~np.isnan(pvd)]
            if len(all_pvd) > 0:
                p20 = np.percentile(all_pvd, 20)
                for i in range(len(nav_idx)):
                    for j in range(n_etfs):
                        if not np.isnan(pvd[i, j]) and pvd[i, j] < p20:
                            if not np.isnan(mom_v[i, j]):
                                mom_v[i, j] *= decay

    # ---- Plan C: conditional activation ----
    elif exp == "C":
        pvd_w = p.get("pvd_w", 0.10)
        score_gap_thr = p.get("score_gap", 0.05)
        if pvd is not None:
            # nasdaq vol percentiles (pre-computed on the fly)
            nasdaq_idx = 0  # 纳指ETF is col 0
            nasdaq_vol_series = vol_v[:, nasdaq_idx]
            valid_vol = nasdaq_vol_series[~np.isnan(nasdaq_vol_series)]
            if len(valid_vol) > 50:
                vol_25 = np.percentile(valid_vol, 25)
                vol_75 = np.percentile(valid_vol, 75)
            else:
                vol_25, vol_75 = 0.10, 0.25

            for i in range(len(nav_idx)):
                nv = nasdaq_vol_series[i]
                if np.isnan(nv) or nv < vol_25 or nv > vol_75:
                    continue  # outside mid-vol regime → no PVD
                # Check score gap: top 2 scores close?
                valid_scores = [(mom_v[i, j], j) for j in range(n_etfs)
                                if not np.isnan(mom_v[i, j]) and mom_v[i, j] > -np.inf]
                if len(valid_scores) >= 2:
                    valid_scores.sort(key=lambda x: x[0], reverse=True)
                    gap = valid_scores[0][0] - valid_scores[1][0]
                    if gap < score_gap_thr:
                        # Apply PVD adjustment
                        for j in range(n_etfs):
                            if not np.isnan(pvd[i, j]) and not np.isnan(mom_v[i, j]):
                                mom_v[i, j] += pvd_w * pvd[i, j]

    # ---- Plan D: defense boost on volume shrinkage ----
    elif exp == "D":
        boost_pct = p.get("boost_pct", 0.05)  # e.g. 5% added to defense
        if _weekly_amt is not None:
            total_amt = _weekly_amt.sum(axis=1).reindex(nav_idx)
            med26 = _weekly_amt.attrs.get("total_med26")
            if med26 is not None:
                med26_aligned = med26.reindex(nav_idx)
                nasdaq_idx = 0
                for i in range(len(nav_idx)):
                    dt = nav_idx[i]
                    ta = total_amt.get(dt)
                    m26 = med26_aligned.get(dt)
                    if pd.notna(ta) and pd.notna(m26) and m26 > 0:
                        if ta < 0.70 * m26:
                            # Inflate nasdaq vol to trigger more defense
                            # step_low~step_high is 0.15~0.30 typically
                            if not np.isnan(vol_v[i, nasdaq_idx]):
                                vol_v[i, nasdaq_idx] += boost_pct

    mom.iloc[:, :] = mom_v
    vol.iloc[:, :] = vol_v
    factors['momentum'] = mom
    factors['volatility'] = vol
    return factors


def set_exp(name, params=None):
    global _active_exp, _active_params
    _active_exp = name
    _active_params = params or {}
    sbt.compute_all_factors = _patched_caf


def clear_exp():
    global _active_exp, _active_params
    _active_exp = None
    _active_params = {}
    sbt.compute_all_factors = _original_caf


# ======================================================================
# Run helpers
# ======================================================================
def run_single(cfg, exp_name, params=None):
    set_exp(exp_name, params)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = run_backtest(cfg, start_date=START_DATE)
        return {
            "sharpe": float(res.metrics["sharpe_ratio"]),
            "maxdd": float(res.metrics["max_drawdown"]),
            "annual_ret": float(res.metrics["annual_return"]),
            "calmar": float(res.metrics["calmar_ratio"]),
        }
    finally:
        clear_exp()


# ======================================================================
# Block Bootstrap
# ======================================================================
def block_bootstrap(cfg, exp_name, params, n_paths=200, block_size=13, seed=42):
    """Block bootstrap: resample 13-week contiguous blocks from real history.

    Volume factors are also resampled with the same block structure,
    preserving within-block price-volume relationships.
    """
    nav = pd.read_csv(NAV_FILE, index_col=0, parse_dates=True)
    n_weeks = len(nav)
    n_blocks = n_weeks // block_size
    total_len = n_blocks * block_size  # trim to exact multiple

    rng = np.random.default_rng(seed)
    results = []

    for path_i in range(n_paths):
        # Sample blocks with replacement
        block_starts = rng.integers(0, n_weeks - block_size, size=n_blocks)
        # Build resampled nav
        nav_blocks = []
        for bs in block_starts:
            nav_blocks.append(nav.iloc[bs:bs + block_size])

        # Concatenate and rebuild cumulative prices
        # Use returns to chain blocks (avoid level discontinuities)
        first_block = nav_blocks[0]
        resampled_rets = []
        for blk in nav_blocks:
            blk_ret = blk.pct_change().iloc[1:]  # skip first NaN
            resampled_rets.append(blk_ret)

        all_rets = pd.concat(resampled_rets, ignore_index=True)
        # Build price levels from initial nav
        init_prices = nav.iloc[0].values.copy()
        prices = np.zeros((len(all_rets) + 1, nav.shape[1]))
        prices[0] = init_prices
        for t in range(len(all_rets)):
            prices[t + 1] = prices[t] * (1 + all_rets.iloc[t].values)

        # Create nav DataFrame with fake dates (same spacing as original)
        fake_dates = pd.date_range(nav.index[0], periods=len(prices), freq="W-FRI")
        nav_bs = pd.DataFrame(prices, index=fake_dates[:len(prices)], columns=nav.columns)

        # Also resample volume factors with same block structure
        global _pvd_factor, _vmr_factor, _volchg_factor, _weekly_amt
        orig_pvd = _pvd_factor.copy()
        orig_vmr = _vmr_factor.copy()
        orig_vc = _volchg_factor.copy()
        orig_amt = _weekly_amt.copy()

        # Rebuild volume factors for bootstrap path
        pvd_blocks = []
        vmr_blocks = []
        vc_blocks = []
        amt_blocks = []
        for bs in block_starts:
            pvd_blocks.append(orig_pvd.iloc[bs:bs + block_size])
            vmr_blocks.append(orig_vmr.iloc[bs:bs + block_size])
            vc_blocks.append(orig_vc.iloc[bs:bs + block_size])
            amt_blocks.append(orig_amt.iloc[bs:bs + block_size])

        def reindex_blocks(blocks, target_index):
            vals = []
            for blk in blocks:
                vals.append(blk.values[1:] if len(blk) > 1 else blk.values)
            # First block includes row 0
            combined = np.vstack([blocks[0].values[:1]] + vals)
            combined = combined[:len(target_index)]
            return pd.DataFrame(combined, index=target_index[:len(combined)],
                                columns=ALL_ETFS)

        bs_idx = nav_bs.index
        _pvd_factor = reindex_blocks(pvd_blocks, bs_idx)
        _vmr_factor = reindex_blocks(vmr_blocks, bs_idx)
        _volchg_factor = reindex_blocks(vc_blocks, bs_idx)
        _weekly_amt = reindex_blocks(amt_blocks, bs_idx)
        _weekly_amt.attrs["total_med26"] = _weekly_amt.sum(axis=1).rolling(26, min_periods=13).median()

        # Save bootstrap nav to temp file and run
        tmp = OUT / f"_bs_{path_i}_{exp_name}.csv"
        nav_bs.to_csv(tmp, encoding="utf-8")
        try:
            set_exp(exp_name, params)
            with contextlib.redirect_stdout(io.StringIO()):
                res = run_backtest(cfg, start_date=str(fake_dates[0].date()), data_path=str(tmp))
            if not res.nav_series.empty:
                results.append({
                    "sharpe": float(res.metrics["sharpe_ratio"]),
                    "maxdd": float(res.metrics["max_drawdown"]),
                })
        except Exception:
            pass
        finally:
            clear_exp()
            if tmp.exists():
                tmp.unlink()

    # Restore original factors
    _pvd_factor = orig_pvd
    _vmr_factor = orig_vmr
    _volchg_factor = orig_vc
    _weekly_amt = orig_amt

    if not results:
        return {"median_sharpe": np.nan, "median_maxdd": np.nan, "n_paths": 0}
    sharpes = [r["sharpe"] for r in results]
    maxdds = [r["maxdd"] for r in results]
    return {
        "median_sharpe": float(np.median(sharpes)),
        "median_maxdd": float(np.median(maxdds)),
        "p25_sharpe": float(np.percentile(sharpes, 25)),
        "p75_sharpe": float(np.percentile(sharpes, 75)),
        "n_paths": len(results),
    }


# ======================================================================
# Experiment definitions
# ======================================================================
PLAN_A = [(f"A-{w}", "A", {"pvd_w": w}) for w in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]]

PLAN_B = [
    ("B1-filter", "B1", {}),
    ("B2-bonus3", "B2", {"bonus": 0.03}),
    ("B2-bonus5", "B2", {"bonus": 0.05}),
    ("B3-decay80", "B3", {"decay": 0.80}),
    ("B3-decay90", "B3", {"decay": 0.90}),
]

PLAN_C = [
    ("C-w10-g05", "C", {"pvd_w": 0.10, "score_gap": 0.05}),
    ("C-w10-g03", "C", {"pvd_w": 0.10, "score_gap": 0.03}),
    ("C-w15-g05", "C", {"pvd_w": 0.15, "score_gap": 0.05}),
]

PLAN_D = [
    ("D-5pct", "D", {"boost_pct": 0.05}),
    ("D-10pct", "D", {"boost_pct": 0.10}),
]

ALL_EXPERIMENTS = [("Baseline", "Baseline", {})] + PLAN_A + PLAN_B + PLAN_C + PLAN_D


# ======================================================================
# Gate
# ======================================================================
def gate_decision(results, bs_results):
    base = results["Baseline"]
    base_sh = base["sharpe"]
    base_dd = base["maxdd"]
    base_bs_sh = bs_results["Baseline"]["median_sharpe"]

    # Find best overall
    treatments = [name for name, _, _ in ALL_EXPERIMENTS if name != "Baseline"]
    best = max(treatments, key=lambda e: results[e]["sharpe"])
    best_delta = results[best]["sharpe"] - base_sh
    best_maxdd_pp = (results[best]["maxdd"] - base_dd) * 100
    best_bs = bs_results.get(best, {}).get("median_sharpe", np.nan)
    bs_delta = best_bs - base_bs_sh if not np.isnan(best_bs) else np.nan

    go = (best_delta >= 0.01 and best_maxdd_pp <= 0.3 and
          not np.isnan(bs_delta) and bs_delta >= 0)
    conditional = (best_delta >= 0.005 and best_maxdd_pp <= 0.5 and
                   not np.isnan(bs_delta) and bs_delta >= -0.005)
    nogo = best_delta < 0.005 or best_maxdd_pp > 1.0

    if go:
        verdict = "GO"
    elif nogo:
        verdict = "NO-GO"
    elif conditional:
        verdict = "CONDITIONAL"
    else:
        verdict = "NO-GO"

    return {
        "verdict": verdict,
        "best_experiment": best,
        "best_sharpe_delta": best_delta,
        "best_maxdd_pp": best_maxdd_pp,
        "bs_delta": bs_delta,
        "criteria": {
            "sharpe_improve": {"value": best_delta, "pass": best_delta >= 0.01},
            "maxdd_limit": {"value": best_maxdd_pp, "pass": best_maxdd_pp <= 0.3},
            "bootstrap_robust": {"value": bs_delta, "pass": not np.isnan(bs_delta) and bs_delta >= 0},
        },
    }


# ======================================================================
# Report
# ======================================================================
def render_report(results, bs_results, gate):
    L = ["# E2b-Volume: PVD 精细化策略整合报告", ""]
    L.append(f"> 四方案 + Block Bootstrap | 门禁: **{gate['verdict']}**")
    L.append("")

    L.append("## 门禁判定")
    L.append("")
    L.append(f"**{gate['verdict']}** | 最优: {gate['best_experiment']}")
    L.append("")
    L.append("| 条件 | 要求 | 实际 | 判定 |")
    L.append("|---|---|---|---|")
    gc = gate["criteria"]
    L.append(f"| ΔSharpe ≥ +0.01 | ≥0.01 | {gc['sharpe_improve']['value']:+.4f} | "
             f"{'✓' if gc['sharpe_improve']['pass'] else '✗'} |")
    L.append(f"| ΔMaxDD ≤ +0.3pp | ≤0.3pp | {gc['maxdd_limit']['value']:+.2f}pp | "
             f"{'✓' if gc['maxdd_limit']['pass'] else '✗'} |")
    L.append(f"| Bootstrap 中位 ≥ baseline | ≥0 | "
             f"{gc['bootstrap_robust']['value']:+.4f} | "
             f"{'✓' if gc['bootstrap_robust']['pass'] else '✗'} |")
    L.append("")

    # All experiments
    L.append("## 全方案对比 (真实历史)")
    L.append("")
    L.append("| 方案 | 实验 | Sharpe | MaxDD | ΔSharpe | ΔMaxDD(pp) |")
    L.append("|---|---|---|---|---|---|")
    base_sh = results["Baseline"]["sharpe"]
    base_dd = results["Baseline"]["maxdd"]
    for name, _, _ in ALL_EXPERIMENTS:
        r = results[name]
        plan = name.split("-")[0] if "-" in name else name
        ds = r["sharpe"] - base_sh
        dd_pp = (r["maxdd"] - base_dd) * 100
        L.append(f"| {plan} | {name} | {r['sharpe']:.4f} | {r['maxdd']:.2%} | "
                 f"{ds:+.4f} | {dd_pp:+.2f} |")
    L.append("")

    # Bootstrap
    L.append("## Block Bootstrap (200 paths, 13-week blocks)")
    L.append("")
    L.append("| 实验 | 中位 Sharpe | p25 | p75 | 中位 MaxDD | N paths |")
    L.append("|---|---|---|---|---|---|")
    for name in ["Baseline", gate["best_experiment"]]:
        bs = bs_results.get(name, {})
        L.append(f"| {name} | {bs.get('median_sharpe', np.nan):.4f} | "
                 f"{bs.get('p25_sharpe', np.nan):.4f} | {bs.get('p75_sharpe', np.nan):.4f} | "
                 f"{bs.get('median_maxdd', np.nan):.2%} | {bs.get('n_paths', 0)} |")
    L.append("")

    L.append("## 关键洞察")
    L.append("")
    L.append("PVD (量价背离) IC=0.053 在统计层面真实存在，")
    L.append("但在 5 只 ETF 周频轮动中的策略价值取决于非线性整合方式和鲁棒性。")
    L.append("")
    return "\n".join(L)


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 70)
    print(" E2b: PVD 精细化策略整合 + Block Bootstrap")
    print("=" * 70)

    prepare_data()
    cfg = load_config(CFG_PATH)

    # --- Run all experiments on real data ---
    results = {}
    for name, exp_type, params in ALL_EXPERIMENTS:
        r = run_single(cfg, exp_type, params)
        results[name] = r
        ds = r["sharpe"] - results.get("Baseline", r)["sharpe"]
        dd_pp = (r["maxdd"] - results.get("Baseline", r)["maxdd"]) * 100
        print(f"  {name:15s}: Sharpe={r['sharpe']:.4f}, MaxDD={r['maxdd']:.2%}, "
              f"Δ={ds:+.4f}, ΔDD={dd_pp:+.2f}pp")

    # Find best
    treatments = [name for name, _, _ in ALL_EXPERIMENTS if name != "Baseline"]
    best = max(treatments, key=lambda e: results[e]["sharpe"])
    best_params = next((p for n, _, p in ALL_EXPERIMENTS if n == best), {})
    best_type = next((t for n, t, _ in ALL_EXPERIMENTS if n == best), "Baseline")
    print(f"\n  最优: {best} (Sharpe={results[best]['sharpe']:.4f})")

    # --- Block Bootstrap for baseline and best ---
    print(f"\n[Bootstrap] Running 200 paths for Baseline...")
    bs_baseline = block_bootstrap(cfg, "Baseline", {}, n_paths=200)
    print(f"  Baseline: median_sharpe={bs_baseline['median_sharpe']:.4f}, "
          f"n={bs_baseline['n_paths']}")

    print(f"[Bootstrap] Running 200 paths for {best}...")
    bs_best = block_bootstrap(cfg, best_type, best_params, n_paths=200)
    print(f"  {best}: median_sharpe={bs_best['median_sharpe']:.4f}, "
          f"n={bs_best['n_paths']}")

    bs_results = {"Baseline": bs_baseline, best: bs_best}

    # --- Gate ---
    gate = gate_decision(results, bs_results)
    print(f"\n{'='*70}")
    print(f" 门禁判定: **{gate['verdict']}**")
    print(f"  最优: {gate['best_experiment']}, ΔSharpe={gate['best_sharpe_delta']:+.4f}, "
          f"ΔMaxDD={gate['best_maxdd_pp']:+.2f}pp, ΔBS={gate['bs_delta']:+.4f}")
    print(f"{'='*70}")

    # --- Save ---
    payload = {"results": results, "bootstrap": bs_results, "gate": gate}
    json_path = OUT / "exp_volume_signal_e2b.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  JSON: {json_path}")

    md = render_report(results, bs_results, gate)
    md_path = OUT / "exp_volume_signal_e2b.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  Report: {md_path}")


if __name__ == "__main__":
    main()
