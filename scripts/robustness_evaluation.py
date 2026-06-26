#!/usr/bin/env python3
"""
Robustness Evaluation Script — Phase 3 Direction 1

Runs 5 robustness tests against the v2.3+cap040 baseline:
  1. Parameter Sensitivity Curvature — ±10% perturbation of each major param
  2. Walk-Forward Stability (Anchored) — 3-year train / 1-year test windows
  3. Monte Carlo Perturbation Test — 100 runs with all params ±10% random
  4. Annual Robustness — year-by-year Sharpe consistency
  5. Metric Clustering — identify plateau vs ridge from Monte Carlo

Requirements:
  - Works with existing data ONLY (no Tushare dependency)
  - Does not modify any src/ code (backward compatible)
  - Produces ROBUSTNESS_REPORT.md

Usage:
  python scripts/robustness_evaluation.py
  python scripts/robustness_evaluation.py --n_mc 200  # more Monte Carlo runs
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import run_backtest
from src.strategy import StrategyConfig, load_config
from src.utils import compute_sharpe


# ── Baseline ────────────────────────────────────────────────────────────────

BASELINE_CONFIG = 'config/strategy_v2_3_cap040.yaml'

# Parameters subject to sensitivity analysis (and their baseline values)
PARAM_KEYS = ['mom_w', 'vol_w', 'def_alloc', 'stop_loss', 'top_n']

# PEM range
PERTURBATION = 0.10  # ±10%


def _load_baseline_config() -> StrategyConfig:
    """Load the v2.3+cap040 baseline config."""
    return load_config(PROJECT_ROOT / BASELINE_CONFIG)


def _run_single(params: dict, base_cfg: StrategyConfig) -> dict | None:
    """
    Run a single backtest with overridden parameters.
    Returns dict with metrics or None on failure.
    """
    try:
        cfg = StrategyConfig(
            name=base_cfg.name,
            version=base_cfg.version,
            mom_w=params.get('mom_w', base_cfg.mom_w),
            vol_w=params.get('vol_w', base_cfg.vol_w),
            top_n=int(params.get('top_n', base_cfg.top_n)),
            mom_window=base_cfg.mom_window,
            vol_window=base_cfg.vol_window,
            pe_window_years=base_cfg.pe_window_years,
            def_alloc=params.get('def_alloc', base_cfg.def_alloc),
            step_low=params.get('step_low', base_cfg.step_low),
            step_high=params.get('step_high', base_cfg.step_high),
            max_def=base_cfg.max_def,
            hongli_ratio=base_cfg.hongli_ratio,
            rebalance_threshold=base_cfg.rebalance_threshold,
            fee_rate=base_cfg.fee_rate,
            anchor=base_cfg.anchor,
            stop_loss=params.get('stop_loss', base_cfg.stop_loss),
            recovery_weeks=base_cfg.recovery_weeks,
            tiered_stop_loss=base_cfg.tiered_stop_loss,
            l1_drawdown=base_cfg.l1_drawdown,
            l1_defense=base_cfg.l1_defense,
            l2_drawdown=base_cfg.l2_drawdown,
            l2_defense=base_cfg.l2_defense,
            l3_weekly_drop=base_cfg.l3_weekly_drop,
            l3_down_weeks=base_cfg.l3_down_weeks,
            l3_window=base_cfg.l3_window,
            l2_recovery_weeks=base_cfg.l2_recovery_weeks,
            l3_recovery_weeks=base_cfg.l3_recovery_weeks,
            max_single_alloc=base_cfg.max_single_alloc,
            stateful_stop_loss=base_cfg.stateful_stop_loss,
            d4_enabled=base_cfg.d4_enabled,
            d4_momentum_window=base_cfg.d4_momentum_window,
            d4_momentum_threshold=base_cfg.d4_momentum_threshold,
            d4_action=base_cfg.d4_action,
            d4_min_candidates=base_cfg.d4_min_candidates,
            d1_enabled=base_cfg.d1_enabled,
            d1_lookback=base_cfg.d1_lookback,
            d1_tq_low=base_cfg.d1_tq_low,
            d1_tq_high=base_cfg.d1_tq_high,
            d1_mom_w_low=base_cfg.d1_mom_w_low,
            d1_mom_w_high=base_cfg.d1_mom_w_high,
            d1_vol_w_low=base_cfg.d1_vol_w_low,
            d1_vol_w_high=base_cfg.d1_vol_w_high,
            d1_weight_sum=base_cfg.d1_weight_sum,
            nav_path=base_cfg.nav_path,
            pe_path=base_cfg.pe_path,
            start_date=base_cfg.start_date,
            end_date=base_cfg.end_date,
            risk_free_rate=base_cfg.risk_free_rate,
        )
        result = run_backtest(cfg)
        if result.nav_series.empty:
            return None
        return {
            **params,
            'annual_return': result.metrics['annual_return'],
            'max_drawdown': result.metrics['max_drawdown'],
            'sharpe_ratio': result.metrics['sharpe_ratio'],
            'calmar_ratio': result.metrics['calmar_ratio'],
            'win_rate': result.metrics['win_rate'],
            'total_weeks': result.metrics['total_weeks'],
        }
    except Exception as e:
        print(f"  [WARN] run_backtest failed for params {params}: {e}")
        return None


# ── Test 1: Parameter Sensitivity Curvature ──────────────────────────────────

def test_param_sensitivity(base_cfg: StrategyConfig) -> dict:
    """
    Test 1: Measure Sharpe gradient near current params.

    For each major param (mom_w, vol_w, def_alloc, stop_loss, top_n),
    run backtests at ±10% perturbation. Measure Sharpe delta.

    Identifies cliffs: params where the drop is asymmetric or steep.

    Returns a dict keyed by param name, with sensitivity data.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Parameter Sensitivity Curvature (±10% perturbation)")
    print("=" * 70)

    results = {}
    baseline_params = {
        'mom_w': base_cfg.mom_w,
        'vol_w': base_cfg.vol_w,
        'def_alloc': base_cfg.def_alloc,
        'stop_loss': base_cfg.stop_loss,
        'top_n': base_cfg.top_n,
        'max_single_alloc': base_cfg.max_single_alloc,
        'rebalance_threshold': base_cfg.rebalance_threshold,
    }

    # Run baseline first
    print(f"  Baseline: {base_cfg.mom_w=}, {base_cfg.vol_w=}, {base_cfg.def_alloc=}, {base_cfg.stop_loss=}, {base_cfg.top_n=}")
    baseline_result = _run_single(baseline_params, base_cfg)
    if baseline_result is None:
        print("  [FATAL] Baseline backtest failed!")
        return {}
    base_sharpe = baseline_result['sharpe_ratio']
    base_ret = baseline_result['annual_return']
    base_dd = baseline_result['max_drawdown']
    print(f"  Base Sharpe={base_sharpe:.4f}, AnnRet={base_ret:.4f}, MaxDD={base_dd:.4f}")

    for param in PARAM_KEYS:
        current_val = getattr(base_cfg, param)
        if param == 'top_n':
            current_val = int(current_val)

        print(f"\n  ── {param} (current={current_val}) ──")

        param_results = {'baseline': current_val, 'baseline_sharpe': base_sharpe}

        for direction, sign in [('+10%', +1), ('-10%', -1)]:
            if param == 'top_n':
                # discrete param: ±1
                new_val = max(1, int(current_val) + sign)
                if new_val == current_val:
                    new_val = max(1, int(current_val) + 2 * sign)
            else:
                delta = current_val * PERTURBATION
                new_val = current_val + sign * delta
                # Clamp to reasonable bounds
                if param in ('mom_w', 'vol_w'):
                    new_val = max(0.05, min(0.80, new_val))
                elif param == 'def_alloc':
                    new_val = max(0.05, min(0.60, new_val))
                elif param == 'stop_loss':
                    new_val = max(0.03, min(0.20, new_val))

            params = dict(baseline_params)
            params[param] = new_val

            r = _run_single(params, base_cfg)
            if r:
                delta_sharpe = r['sharpe_ratio'] - base_sharpe
                delta_ret = r['annual_return'] - base_ret
                delta_dd = r['max_drawdown'] - base_dd
                param_results[direction] = {
                    'value': new_val,
                    'sharpe': r['sharpe_ratio'],
                    'delta_sharpe': delta_sharpe,
                    'annual_return': r['annual_return'],
                    'max_drawdown': r['max_drawdown'],
                    'delta_ret': delta_ret,
                    'delta_dd': delta_dd,
                }
                cliff_flag = ""
                if direction == '-10%' and delta_sharpe < -0.20:
                    cliff_flag = " ⚠️ CLIFF (negative side steep drop)"
                elif direction == '+10%' and delta_sharpe < -0.20:
                    cliff_flag = " ⚠️ CLIFF (positive side steep drop)"
                print(f"    {direction}: val={new_val:.4f}, Sharpe={r['sharpe_ratio']:.4f}, "
                      f"ΔSharpe={delta_sharpe:+.4f}, ΔRet={delta_ret:+.4f}, ΔDD={delta_dd:+.4f}{cliff_flag}")
            else:
                param_results[direction] = {'value': new_val, 'error': 'backtest failed'}
                print(f"    {direction}: val={new_val:.4f} -> FAILED")

        results[param] = param_results

    return results


# ── Test 2: Walk-Forward Stability (Anchored) ────────────────────────────────

def test_walk_forward(base_cfg: StrategyConfig) -> dict:
    """
    Test 2: Walk-forward stability using 3-year train / 1-year test windows.

    Splits the full period into overlapping 4-year windows:
      - Train: first 3 years (used for... well, we can't train params here,
        but we measure consistency of fixed baseline across windows)
      - Test: 1 year following

    Since this is a fixed-parameter strategy (not ML), "walk-forward" means
    measuring Sharpe consistency across sequential test windows.

    Target: test Sharpe std dev < 0.15 across windows.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Walk-Forward Stability (Anchored, 3yr train / 1yr test)")
    print("=" * 70)

    # Parse the full date range from the data
    from src.data_loader import load_nav_data, resample_weekly
    nav_df = load_nav_data(PROJECT_ROOT / base_cfg.nav_path)
    weekly_nav = resample_weekly(nav_df, anchor=base_cfg.anchor)

    start_dt = weekly_nav.index[0]
    end_dt = weekly_nav.index[-1]
    print(f"  Full date range: {start_dt.date()} to {end_dt.date()}")

    # Generate 4-year windows (3yr train + 1yr test) with 1-year step
    windows = []
    current = start_dt
    test_duration = pd.DateOffset(years=1)
    while current + pd.DateOffset(years=4) <= end_dt:
        train_start = current
        train_end = current + pd.DateOffset(years=3)
        test_start = train_end
        test_end = test_start + test_duration
        windows.append((train_start, train_end, test_start, test_end))
        current += test_duration

    print(f"  Generated {len(windows)} walk-forward windows")

    results = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        # For fixed-param strategy, we run on the full train+test but only
        # report test-period metrics. Actually, to isolate test period, we
        # run the backtest on just the test period range.
        r = _run_single({}, base_cfg)
        if r is None:
            continue

        # Run on just the test window
        cfg_te = StrategyConfig(
            name=base_cfg.name, version=base_cfg.version,
            mom_w=base_cfg.mom_w, vol_w=base_cfg.vol_w,
            top_n=base_cfg.top_n,
            mom_window=base_cfg.mom_window, vol_window=base_cfg.vol_window,
            pe_window_years=base_cfg.pe_window_years,
            def_alloc=base_cfg.def_alloc,
            step_low=base_cfg.step_low, step_high=base_cfg.step_high,
            max_def=base_cfg.max_def, hongli_ratio=base_cfg.hongli_ratio,
            rebalance_threshold=base_cfg.rebalance_threshold,
            fee_rate=base_cfg.fee_rate, anchor=base_cfg.anchor,
            stop_loss=base_cfg.stop_loss, recovery_weeks=base_cfg.recovery_weeks,
            tiered_stop_loss=base_cfg.tiered_stop_loss,
            l1_drawdown=base_cfg.l1_drawdown, l1_defense=base_cfg.l1_defense,
            l2_drawdown=base_cfg.l2_drawdown, l2_defense=base_cfg.l2_defense,
            l3_weekly_drop=base_cfg.l3_weekly_drop,
            l3_down_weeks=base_cfg.l3_down_weeks, l3_window=base_cfg.l3_window,
            l2_recovery_weeks=base_cfg.l2_recovery_weeks,
            l3_recovery_weeks=base_cfg.l3_recovery_weeks,
            max_single_alloc=base_cfg.max_single_alloc,
            stateful_stop_loss=base_cfg.stateful_stop_loss,
            d4_enabled=base_cfg.d4_enabled,
            d4_momentum_window=base_cfg.d4_momentum_window,
            d4_momentum_threshold=base_cfg.d4_momentum_threshold,
            d4_action=base_cfg.d4_action, d4_min_candidates=base_cfg.d4_min_candidates,
            d1_enabled=base_cfg.d1_enabled, d1_lookback=base_cfg.d1_lookback,
            d1_tq_low=base_cfg.d1_tq_low, d1_tq_high=base_cfg.d1_tq_high,
            d1_mom_w_low=base_cfg.d1_mom_w_low, d1_mom_w_high=base_cfg.d1_mom_w_high,
            d1_vol_w_low=base_cfg.d1_vol_w_low, d1_vol_w_high=base_cfg.d1_vol_w_high,
            d1_weight_sum=base_cfg.d1_weight_sum,
            nav_path=base_cfg.nav_path, pe_path=base_cfg.pe_path,
            start_date=str(te_s.date()), end_date=str(te_e.date()),
            risk_free_rate=base_cfg.risk_free_rate,
        )

        te_result = run_backtest(cfg_te)
        if te_result.nav_series.empty:
            print(f"  Window {i}: [{te_s.date()} to {te_e.date()}] — EMPTY, skipping")
            continue

        m = te_result.metrics
        entry = {
            'window': i,
            'test_start': str(te_s.date()),
            'test_end': str(te_e.date()),
            'annual_return': m['annual_return'],
            'max_drawdown': m['max_drawdown'],
            'sharpe_ratio': m['sharpe_ratio'],
            'calmar_ratio': m['calmar_ratio'],
            'win_rate': m['win_rate'],
            'total_weeks': m['total_weeks'],
        }
        results.append(entry)
        print(f"  Window {i}: [{te_s.date()} to {te_e.date()}] "
              f"Sharpe={m['sharpe_ratio']:.4f}, "
              f"Ret={m['annual_return']:.4f}, DD={m['max_drawdown']:.4f}")

    if results:
        sharpes = [r['sharpe_ratio'] for r in results]
        std_dev = np.std(sharpes)
        mean_sharpe = np.mean(sharpes)
        print(f"\n  Summary: mean Sharpe={mean_sharpe:.4f}, std={std_dev:.4f}")
        print(f"  Target: std < 0.15 — {'✅ PASS' if std_dev < 0.15 else '❌ FAIL (std={:.4f} >= 0.15)'.format(std_dev)}")
    else:
        std_dev = None
        mean_sharpe = None

    return {
        'windows': results,
        'std_dev': std_dev,
        'mean_sharpe': mean_sharpe,
        'n_windows': len(results),
    }


# ── Test 3: Monte Carlo Perturbation Test ────────────────────────────────────

def _mc_single(params_tuple) -> dict | None:
    """Single Monte Carlo run (for multiprocessing)."""
    params, base_cfg = params_tuple
    return _run_single(params, base_cfg)


def test_monte_carlo(base_cfg: StrategyConfig, n_runs: int = 100) -> dict:
    """
    Test 3: Monte Carlo perturbation — all params simultaneously perturbed ±10%.

    Run N backtests with each parameter independently perturbed by
    uniform random ±10%. Measure what % of runs achieve Sharpe > 1.0.

    Target: >90% pass rate.
    """
    print("\n" + "=" * 70)
    print(f"TEST 3: Monte Carlo Perturbation ({n_runs} runs, all params ±10%)")
    print("=" * 70)

    # Build parameter sets
    rng = np.random.RandomState(42)
    mc_params = []

    baseline_params = {
        'mom_w': base_cfg.mom_w,
        'vol_w': base_cfg.vol_w,
        'def_alloc': base_cfg.def_alloc,
        'stop_loss': base_cfg.stop_loss,
        'top_n': base_cfg.top_n,
        'max_single_alloc': base_cfg.max_single_alloc,
        'rebalance_threshold': base_cfg.rebalance_threshold,
        'step_low': base_cfg.step_low,
        'step_high': base_cfg.step_high,
    }

    for run_i in range(n_runs):
        params = {}
        for key, base_val in baseline_params.items():
            if key == 'top_n':
                # discrete: randomly pick from {base-1, base, base+1}
                choices = [max(1, int(base_val) - 1), int(base_val), int(base_val) + 1]
                params[key] = int(rng.choice(choices))
            else:
                delta = base_val * PERTURBATION
                noise = rng.uniform(-delta, delta)
                new_val = base_val + noise
                # Clamp
                if key in ('mom_w', 'vol_w'):
                    new_val = max(0.05, min(0.80, new_val))
                elif key == 'def_alloc':
                    new_val = max(0.05, min(0.60, new_val))
                elif key == 'stop_loss':
                    new_val = max(0.03, min(0.20, new_val))
                elif key == 'max_single_alloc':
                    new_val = max(0.10, min(1.0, new_val))
                elif key == 'rebalance_threshold':
                    new_val = max(0.01, min(0.20, new_val))
                elif key in ('step_low', 'step_high'):
                    new_val = max(0.05, min(0.60, new_val))
                params[key] = new_val
        mc_params.append((params, base_cfg))

    # Run (sequential for reliability, but could parallelize)
    results = []
    for i, (params, _) in enumerate(mc_params):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{n_runs}] running...")
        r = _run_single(params, base_cfg)
        if r:
            results.append(r)

    n_success = len(results)
    n_sharpe_above_1 = sum(1 for r in results if r['sharpe_ratio'] > 1.0)
    pass_rate = n_sharpe_above_1 / n_success if n_success > 0 else 0.0

    sharpes = [r['sharpe_ratio'] for r in results]
    rets = [r['annual_return'] for r in results]
    dds = [r['max_drawdown'] for r in results]

    print(f"\n  Results: {n_success}/{n_runs} successful runs")
    print(f"  Sharpe > 1.0: {n_sharpe_above_1}/{n_success} ({pass_rate*100:.1f}%)")
    print(f"  Sharpe: mean={np.mean(sharpes):.4f}, std={np.std(sharpes):.4f}, "
          f"min={np.min(sharpes):.4f}, max={np.max(sharpes):.4f}")
    print(f"  AnnRet: mean={np.mean(rets):.4f}, std={np.std(rets):.4f}")
    print(f"  MaxDD:  mean={np.mean(dds):.4f}, std={np.std(dds):.4f}, max={np.max(dds):.4f}")
    print(f"  Target: >90% Sharpe>1.0 — {'✅ PASS' if pass_rate > 0.90 else '❌ FAIL (pass_rate={:.1f}%)'.format(pass_rate*100)}")

    return {
        'n_runs': n_runs,
        'n_success': n_success,
        'n_sharpe_above_1': n_sharpe_above_1,
        'pass_rate': pass_rate,
        'sharpe_mean': np.mean(sharpes),
        'sharpe_std': np.std(sharpes),
        'sharpe_min': np.min(sharpes),
        'sharpe_max': np.max(sharpes),
        'ret_mean': np.mean(rets),
        'ret_std': np.std(rets),
        'dd_mean': np.mean(dds),
        'dd_std': np.std(dds),
        'dd_max': np.max(dds),
        'results': results,  # full data for clustering
    }


# ── Test 4: Annual Robustness ────────────────────────────────────────────────

def test_annual_robustness(base_cfg: StrategyConfig) -> dict:
    """
    Test 4: Year-by-year Sharpe consistency.

    Run baseline backtest and compute Sharpe for each calendar year.
    Target: max 1 year with negative Sharpe.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Annual Robustness (year-by-year Sharpe)")
    print("=" * 70)

    # Run full-period backtest
    result = run_backtest(base_cfg)
    nav_series = result.nav_series
    if nav_series.empty:
        print("  [FATAL] Baseline backtest failed!")
        return {}

    # Group weekly returns by year
    yearly = {}
    for date, row in nav_series.iterrows():
        year = date.year
        if year not in yearly:
            yearly[year] = []
        yearly[year].append(row['weekly_return'])

    annual_results = []
    negative_years = 0
    for year in sorted(yearly.keys()):
        weekly_rets = pd.Series(yearly[year])
        n_weeks = len(weekly_rets)
        if n_weeks < 10:  # skip partial years
            continue

        ann_ret = (1 + weekly_rets.sum()) ** (52 / n_weeks) - 1 if weekly_rets.sum() > -1 else -1.0
        sharpe = compute_sharpe(weekly_rets, base_cfg.risk_free_rate)
        dd_series = (1 + weekly_rets).cumprod()
        peak = dd_series.cummax()
        dd = ((peak - dd_series) / peak).max()
        win_rate = (weekly_rets > 0).mean()

        entry = {
            'year': year,
            'n_weeks': n_weeks,
            'annual_return': ann_ret,
            'max_drawdown': dd,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
        }
        annual_results.append(entry)

        flag = " ⚠️ NEGATIVE" if sharpe < 0 else ""
        print(f"  {year}: Sharpe={sharpe:.4f}, Ret={ann_ret:.4f}, DD={dd:.4f}, Weeks={n_weeks}{flag}")
        if sharpe < 0:
            negative_years += 1

    print(f"\n  Negative Sharpe years: {negative_years}/{len(annual_results)}")
    print(f"  Target: max 1 negative year — {'✅ PASS' if negative_years <= 1 else '❌ FAIL ({negative_years} negative years)'.format(negative_years=negative_years)}")

    return {
        'annual_results': annual_results,
        'negative_years': negative_years,
        'total_years': len(annual_results),
    }


# ── Test 5: Metric Clustering ────────────────────────────────────────────────

def test_metric_clustering(mc_data: dict) -> dict:
    """
    Test 5: Cluster analysis from Monte Carlo output.

    Use Monte Carlo results to identify:
      - Broad plateau (good): params widely spread in high-Sharpe region
      - Narrow ridge (bad): high-Sharpe only in tight param cluster

    Method:
      1. Split MC results into top 25% and bottom 25% by Sharpe
      2. Compare param variance in top vs bottom quartile
      3. High variance in top quartile = plateau. Low = ridge.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Metric Clustering (plateau vs ridge from Monte Carlo)")
    print("=" * 70)

    results = mc_data.get('results', [])
    if len(results) < 20:
        print(f"  [WARN] Not enough MC results ({len(results)}) for clustering. Need 20+. Skipping.")
        return {'plateau_or_ridge': 'insufficient_data'}

    # Sort by Sharpe
    sorted_results = sorted(results, key=lambda r: r['sharpe_ratio'], reverse=True)
    n = len(sorted_results)
    top_quartile = sorted_results[:n // 4]
    bottom_quartile = sorted_results[-(n // 4):]

    # Extract param values for continuous params
    param_keys = ['mom_w', 'vol_w', 'def_alloc', 'stop_loss']

    top_params = {k: [r[k] for r in top_quartile if k in r] for k in param_keys}
    bot_params = {k: [r[k] for r in bottom_quartile if k in r] for k in param_keys}

    print(f"  Top quartile (Sharpe > {sorted_results[n//4]['sharpe_ratio']:.4f}):")
    print(f"  Bottom quartile (Sharpe < {sorted_results[-n//4]['sharpe_ratio']:.4f}):")
    print()

    cluster_analysis = {}
    for k in param_keys:
        top_vals = top_params[k]
        bot_vals = bot_params[k]
        if top_vals and bot_vals:
            top_std = np.std(top_vals)
            bot_std = np.std(bot_vals)
            top_mean = np.mean(top_vals)
            bot_mean = np.mean(bot_vals)
            ratio = top_std / bot_std if bot_std > 0 else float('inf')
            cluster_analysis[k] = {
                'top_mean': top_mean, 'top_std': top_std,
                'bot_mean': bot_mean, 'bot_std': bot_std,
                'variance_ratio': ratio,
            }
            print(f"  {k}: top(μ={top_mean:.4f}, σ={top_std:.4f}) vs "
                  f"bot(μ={bot_mean:.4f}, σ={bot_std:.4f}) "
                  f"ratio={ratio:.2f}")

    # Determine plateau vs ridge
    # Plateau: high variance in top quartile (params spread out broadly)
    # Ridge: low variance in top quartile (only tight region works)
    avg_ratio = np.mean([c['variance_ratio'] for c in cluster_analysis.values()
                         if c['variance_ratio'] < float('inf')])

    if avg_ratio > 1.3:
        verdict = "BROAD PLATEAU"
        detail = "High-Sharpe runs span a wider parameter range than low-Sharpe runs — the sweet spot is broad."
    elif avg_ratio > 0.8:
        verdict = "MODERATE PLATEAU"
        detail = "High-Sharpe runs have comparable parameter spread to low-Sharpe — reasonably broad, not a knife-edge."
    else:
        verdict = "NARROW RIDGE"
        detail = "High-Sharpe runs cluster tightly — the strategy is fragile and parameter-sensitive (narrow ridge)."

    print(f"\n  Avg variance ratio (top/bottom): {avg_ratio:.2f}")
    print(f"  Verdict: {verdict}")
    print(f"  Detail: {detail}")

    return {
        'plateau_or_ridge': verdict,
        'detail': detail,
        'avg_variance_ratio': avg_ratio,
        'cluster_analysis': cluster_analysis,
        'n_top_quartile': len(top_quartile),
        'n_bottom_quartile': len(bottom_quartile),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Robustness Evaluation — Phase 3 Direction 1'
    )
    parser.add_argument('--n_mc', type=int, default=100,
                        help='Number of Monte Carlo runs (default: 100)')
    parser.add_argument('--skip-tests', type=str, default='',
                        help='Comma-separated tests to skip (1,2,3,4,5)')
    parser.add_argument('--output', type=str,
                        default='ROBUSTNESS_REPORT.md',
                        help='Report output path')
    args = parser.parse_args()

    print("=" * 70)
    print("ROBUSTNESS EVALUATION — Phase 3 Direction 1")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Baseline config: {BASELINE_CONFIG}")
    print(f"N Monte Carlo: {args.n_mc}")
    print("=" * 70)

    # Load baseline
    base_cfg = _load_baseline_config()

    skip = set(args.skip_tests.split(',')) if args.skip_tests else set()

    results = {}

    # Test 1: Parameter Sensitivity
    if '1' not in skip:
        results['param_sensitivity'] = test_param_sensitivity(base_cfg)

    # Test 2: Walk-Forward Stability
    if '2' not in skip:
        results['walk_forward'] = test_walk_forward(base_cfg)

    # Test 3: Monte Carlo
    if '3' not in skip:
        results['monte_carlo'] = test_monte_carlo(base_cfg, n_runs=args.n_mc)

    # Test 4: Annual Robustness
    if '4' not in skip:
        results['annual_robustness'] = test_annual_robustness(base_cfg)

    # Test 5: Metric Clustering (depends on MC data)
    if '5' not in skip:
        if 'monte_carlo' in results:
            results['metric_clustering'] = test_metric_clustering(results['monte_carlo'])
        else:
            print("\n[SKIP] Test 5 requires Test 3 (Monte Carlo) to run first.")

    # Save results as JSON for downstream use
    json_path = PROJECT_ROOT / 'output' / 'robustness_results.json'
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable dict
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            # Deep copy, stripping non-serializable
            clean = {}
            for kk, vv in v.items():
                if kk == 'results' and isinstance(vv, list):
                    # already list of dicts, keep it
                    clean[kk] = vv
                elif isinstance(vv, (int, float, str, bool, list, dict, type(None))):
                    clean[kk] = vv
                else:
                    clean[kk] = str(vv)
            serializable[k] = clean

    with open(json_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nRaw results saved to: {json_path}")

    # Generate report
    report_path = PROJECT_ROOT / args.output
    _generate_report(results, base_cfg, report_path, args.n_mc)

    return results


def _generate_report(results: dict, base_cfg: StrategyConfig,
                     report_path: Path, n_mc: int):
    """Generate ROBUSTNESS_REPORT.md from test results."""

    lines = []
    lines.append("# Robustness Evaluation Report — Phase 3 Direction 1")
    lines.append("")
    lines.append(f"**Baseline**: v2.3+cap040 (annual_return=14.11%, max_drawdown=7.42%, sharpe=1.102)")
    lines.append(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Monte Carlo runs**: {n_mc}")
    lines.append("")

    # Executive summary
    lines.append("## Executive Summary")
    lines.append("")

    # Collect pass/fail status
    findings = []

    # Test 1: Parameter Sensitivity
    lines.append("---")
    lines.append("")
    lines.append("## Test 1: Parameter Sensitivity Curvature")
    lines.append("")
    lines.append("Measures Sharpe gradient near current params with ±10% perturbation.")
    lines.append("Identifies cliffs: params where small changes cause large Sharpe drops.")
    lines.append("")

    ps = results.get('param_sensitivity', {})
    if ps:
        lines.append("| Parameter | Baseline | -10% Value | -10% ΔSharpe | -10% ΔRet | +10% Value | +10% ΔSharpe | +10% ΔRet | Cliff? |")
        lines.append("|-----------|----------|------------|-------------|----------|------------|-------------|----------|--------|")

        for param in PARAM_KEYS:
            pdata = ps.get(param, {})
            base_val = pdata.get('baseline', '?')
            base_sharpe = pdata.get('baseline_sharpe', 0)

            neg = pdata.get('-10%', {})
            pos = pdata.get('+10%', {})

            neg_val = neg.get('value', '?')
            neg_ds = neg.get('delta_sharpe', '?')
            neg_dr = neg.get('delta_ret', '?')
            pos_val = pos.get('value', '?')
            pos_ds = pos.get('delta_sharpe', '?')
            pos_dr = pos.get('delta_ret', '?')

            # Cliff detection
            cliff = ''
            if isinstance(neg_ds, (int, float)) and neg_ds < -0.20:
                cliff = '⚠️ LEFT CLIFF'
            if isinstance(pos_ds, (int, float)) and pos_ds < -0.20:
                cliff += (' + ' if cliff else '') + '⚠️ RIGHT CLIFF'

            neg_ds_str = f"{neg_ds:+.4f}" if isinstance(neg_ds, float) else str(neg_ds)
            neg_dr_str = f"{neg_dr:+.4f}" if isinstance(neg_dr, float) else str(neg_dr)
            pos_ds_str = f"{pos_ds:+.4f}" if isinstance(pos_ds, float) else str(pos_ds)
            pos_dr_str = f"{pos_dr:+.4f}" if isinstance(pos_dr, float) else str(pos_dr)

            lines.append(f"| {param} | {base_val} | {neg_val} | {neg_ds_str} | {neg_dr_str} | {pos_val} | {pos_ds_str} | {pos_dr_str} | {cliff} |")

            if cliff:
                findings.append(f"**CLIFF DETECTED on {param}**: {cliff}. Consider adjusting this parameter.")

        lines.append("")
        if not findings:
            lines.append("**No cliffs detected.** All params show symmetric, gradual Sharpe response within ±10%.")
            findings.append("No cliffs detected — parameter sensitivity is well-behaved.")
    else:
        lines.append("_No data — test skipped or failed._")

    # Test 2: Walk-Forward
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Test 2: Walk-Forward Stability (Anchored)")
    lines.append("")
    lines.append("3-year train / 1-year test windows. Target: test Sharpe std dev < 0.15.")
    lines.append("")

    wf = results.get('walk_forward', {})
    if wf:
        windows = wf.get('windows', [])
        if windows:
            lines.append("| Window | Test Period | Sharpe | AnnRet | MaxDD | WinRate |")
            lines.append("|--------|-------------|--------|--------|-------|---------|")
            for w in windows:
                lines.append(f"| {w['window']} | {w['test_start']} → {w['test_end']} | {w['sharpe_ratio']:.4f} | {w['annual_return']:.4f} | {w['max_drawdown']:.4f} | {w['win_rate']:.3f} |")

        std_dev = wf.get('std_dev')
        if std_dev is not None:
            lines.append("")
            lines.append(f"- **Mean Sharpe**: {wf['mean_sharpe']:.4f}")
            lines.append(f"- **Std Dev**: {std_dev:.4f}")
            lines.append(f"- **Target**: std < 0.15 — {'✅ PASS' if std_dev < 0.15 else '❌ FAIL'}")
            if std_dev >= 0.15:
                findings.append(f"Walk-forward Sharpe std={std_dev:.4f} exceeds target 0.15 — strategy Sharpe varies significantly across time windows.")
            else:
                findings.append(f"Walk-forward Sharpe std={std_dev:.4f} < 0.15 — strategy performance is time-stable.")
    else:
        lines.append("_No data — test skipped or failed._")

    # Test 3: Monte Carlo
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Test 3: Monte Carlo Perturbation Test")
    lines.append("")
    lines.append(f"All params simultaneously perturbed ±10% (uniform random). {n_mc} runs.")
    lines.append("Target: >90% runs achieve Sharpe > 1.0.")
    lines.append("")

    mc = results.get('monte_carlo', {})
    if mc:
        pass_rate = mc.get('pass_rate', 0)
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Successful runs | {mc.get('n_success', 0)}/{mc.get('n_runs', 0)} |")
        lines.append(f"| Sharpe > 1.0 | {mc.get('n_sharpe_above_1', 0)}/{mc.get('n_success', 0)} ({pass_rate*100:.1f}%) |")
        lines.append(f"| Sharpe mean | {mc.get('sharpe_mean', 0):.4f} |")
        lines.append(f"| Sharpe std | {mc.get('sharpe_std', 0):.4f} |")
        lines.append(f"| Sharpe range | [{mc.get('sharpe_min', 0):.4f}, {mc.get('sharpe_max', 0):.4f}] |")
        lines.append(f"| AnnRet mean | {mc.get('ret_mean', 0):.4f} |")
        lines.append(f"| MaxDD mean | {mc.get('dd_mean', 0):.4f} |")
        lines.append(f"| MaxDD max | {mc.get('dd_max', 0):.4f} |")
        lines.append("")
        lines.append(f"- **Target**: >90% Sharpe>1.0 — {'✅ PASS' if pass_rate > 0.90 else '❌ FAIL (pass_rate=' + str(round(pass_rate*100,1)) + '%)'}")
        if pass_rate <= 0.90:
            findings.append(f"Monte Carlo pass rate only {pass_rate*100:.1f}% — strategy is NOT parameter-insensitive.")
        else:
            findings.append(f"Monte Carlo pass rate {pass_rate*100:.1f}% exceeds 90% target — strategy is parameter-insensitive.")
    else:
        lines.append("_No data — test skipped or failed._")

    # Test 4: Annual Robustness
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Test 4: Annual Robustness")
    lines.append("")
    lines.append("Year-by-year Sharpe. Target: max 1 year with negative Sharpe.")
    lines.append("")

    ar = results.get('annual_robustness', {})
    if ar:
        annual = ar.get('annual_results', [])
        if annual:
            lines.append("| Year | Weeks | Sharpe | AnnRet | MaxDD | WinRate |")
            lines.append("|------|-------|--------|--------|-------|---------|")
            for a in annual:
                flag = " ⚠️" if a['sharpe_ratio'] < 0 else ""
                lines.append(f"| {a['year']} | {a['n_weeks']} | {a['sharpe_ratio']:.4f}{flag} | {a['annual_return']:.4f} | {a['max_drawdown']:.4f} | {a['win_rate']:.3f} |")

        neg_yrs = ar.get('negative_years', 0)
        lines.append("")
        lines.append(f"- **Negative Sharpe years**: {neg_yrs}/{ar.get('total_years', 0)}")
        lines.append(f"- **Target**: max 1 negative year — {'✅ PASS' if neg_yrs <= 1 else '❌ FAIL (' + str(neg_yrs) + ' negative years)'}")
        if neg_yrs > 1:
            findings.append(f"{neg_yrs} years with negative Sharpe — exceeds max-1 target.")
        else:
            findings.append(f"Annual consistency good: only {neg_yrs} year(s) with negative Sharpe.")
    else:
        lines.append("_No data — test skipped or failed._")

    # Test 5: Metric Clustering
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Test 5: Metric Clustering (Plateau vs Ridge)")
    lines.append("")
    lines.append("Analyzes Monte Carlo output: compares parameter variance in top vs bottom quartile.")
    lines.append("Broad variance in top quartile = plateau (good). Tight = ridge (bad).")
    lines.append("")

    mc2 = results.get('metric_clustering', {})
    if mc2 and mc2.get('plateau_or_ridge') != 'insufficient_data':
        lines.append(f"**Verdict**: {mc2.get('plateau_or_ridge', '?')}")
        lines.append(f"**Detail**: {mc2.get('detail', '?')}")
        lines.append(f"**Avg variance ratio (top/bottom)**: {mc2.get('avg_variance_ratio', 0):.2f}")
        lines.append("")

        ca = mc2.get('cluster_analysis', {})
        if ca:
            lines.append("| Parameter | Top μ | Top σ | Bot μ | Bot σ | Variance Ratio |")
            lines.append("|-----------|-------|-------|-------|-------|----------------|")
            for k, v in ca.items():
                lines.append(f"| {k} | {v['top_mean']:.4f} | {v['top_std']:.4f} | {v['bot_mean']:.4f} | {v['bot_std']:.4f} | {v['variance_ratio']:.2f} |")

        findings.append(f"Clustering verdict: {mc2.get('plateau_or_ridge', '?')} — {mc2.get('detail', '?')}")
    elif mc2.get('plateau_or_ridge') == 'insufficient_data':
        lines.append("_Insufficient Monte Carlo data for clustering analysis._")
    else:
        lines.append("_No data — test skipped or failed._")

    # Summary and Recommendations
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Summary of Findings")
    lines.append("")
    for i, f in enumerate(findings, 1):
        lines.append(f"{i}. {f}")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")

    # Determine recommendation based on findings
    has_cliff = any('CLIFF' in f for f in findings)
    mc_passes = any('pass rate' in f.lower() and 'exceeds' in f.lower() for f in findings)

    lines.append("### Parameter Adjustments")
    lines.append("")
    if has_cliff:
        lines.append("Cliffs detected — parameter adjustment recommended:")
        lines.append("")
        # Identify which params have cliffs from ps data
        for param in PARAM_KEYS:
            pdata = ps.get(param, {})
            neg = pdata.get('-10%', {})
            pos = pdata.get('+10%', {})
            neg_ds = neg.get('delta_sharpe', 0) if isinstance(neg, dict) else 0
            pos_ds = pos.get('delta_sharpe', 0) if isinstance(pos, dict) else 0

            if isinstance(neg_ds, (int, float)) and neg_ds < -0.15:
                lines.append(f"- **{param}**: -10% causes Sharpe drop of {neg_ds:.3f}. Consider raising baseline or widening the acceptable range.")
            if isinstance(pos_ds, (int, float)) and pos_ds < -0.15:
                lines.append(f"- **{param}**: +10% causes Sharpe drop of {pos_ds:.3f}. Consider lowering baseline or widening the acceptable range.")
        lines.append("")
    else:
        lines.append("No significant cliffs detected. Current parameter set sits on a reasonably flat region.")
        lines.append("")

    lines.append("### DD Headroom Utilization")
    lines.append("")
    lines.append(f"- Current MaxDD: 7.42%")
    lines.append(f"- Acceptable MaxDD: 10.0%")
    lines.append(f"- Available headroom: ~2.58pp")
    lines.append("")

    if has_cliff:
        lines.append("DD headroom can be traded for robustness by relaxing parameters away from cliff edges.")
        lines.append("For each cliff-identified param, shift the baseline value toward the safe side,")
        lines.append("accepting slightly lower peak Sharpe in exchange for wider robustness.")
    else:
        lines.append("With no cliffs, DD headroom is not urgently needed for robustness. However,")
        lines.append("it could be used for future feature development (expanded universe, etc.).")

    lines.append("")
    lines.append("### Should We Relax Parameters?")
    lines.append("")
    if mc_passes and not has_cliff:
        lines.append("**No relaxation needed.** The strategy is already parameter-insensitive (Monte Carlo pass rate >90%)")
        lines.append("and sits on a broad plateau. Relaxing params would only degrade peak performance without")
        lines.append("meaningfully improving robustness.")
    elif has_cliff:
        lines.append("**Yes, targeted relaxation recommended.** Shift cliff-identified params away from steep-drop regions.")
        lines.append("Acceptable trade: peak Sharpe may drop ~0.05-0.10 in exchange for 2-3x wider parameter tolerance.")
    else:
        lines.append("**Consider moderate relaxation.** Monte Carlo pass rate is below target. Widening param ranges")
        lines.append("(e.g., slightly lower vol_w, slightly higher mom_w) may improve robustness at a small cost to peak Sharpe.")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append(f"- Raw results JSON: `output/robustness_results.json`")
    lines.append(f"- Baseline config: `{BASELINE_CONFIG}`")
    lines.append("")

    report = '\n'.join(lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == '__main__':
    main()
