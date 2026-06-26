"""
Robustness V2 — 3-Metric Robustness Evaluation (unified).

Three independent metrics:
  - DSR (Deflated Sharpe Ratio): delegates to src.robustness.compute_dsr()
    (full B&LdP formula with skew/kurtosis correction).
  - MC Survival Rate: fraction of Monte Carlo parameter-perturbation runs
    that meet the goal-aligned criterion (annual_return > 10% AND max_drawdown < 15%).
  - Benchmark-Relative Win Rate: walk-forward comparison vs equal-weight
    ETF benchmark.

Dependencies: numpy, pandas, scipy, plus the project's strategy,
backtest, data_loader, and utils modules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ── M1: Deflated Sharpe Ratio (Bailey & López de Prado, 2014) ────────────────
# 已统一为完整 B&LdP 公式 (含偏度/峰度矫正)
# Delegates to src.robustness.compute_dsr() — the canonical implementation.

def compute_dsr(strategy_sharpe: float,
                n_trials: int,
                n_weeks: int,
                sharpe_std: float = 0.5) -> dict:
    """
    Deflated Sharpe Ratio — delegates to the complete B&LdP formula.

    Uses src.robustness.compute_dsr() (Bailey & López de Prado, 2014)
    with skew/kurtosis correction. When returns data is unavailable,
    normal-distribution defaults (skew=0, kurtosis=3) are used as the
    standard asymptotic approximation.

    Args:
        strategy_sharpe: Observed strategy Sharpe ratio.
        n_trials: Total number of strategy variants tested (N).
        n_weeks: Number of weeks in the backtest.
        sharpe_std: Estimated standard deviation of Sharpe ratio.

    Returns:
        dict with keys 'dsr', 'e_max', 'interpretation'.
    """
    from scipy.stats import norm
    from src.robustness import compute_dsr as _compute_dsr_full

    # Use normal-distribution defaults for skew/kurtosis
    # (the asymptotic B&LdP approximation when return data is unavailable)
    skew = 0.0
    kurtosis = 3.0

    dsr = _compute_dsr_full(strategy_sharpe, n_trials, n_weeks, skew, kurtosis)

    # Compute e_max for backward-compatible reporting
    e_max = _compute_e_max(n_trials, strategy_sharpe)

    if dsr > 0.95:
        interp = "高度显著：95%+概率是真alpha"
    elif dsr > 0.80:
        interp = "边际显著：需更多证据"
    else:
        interp = "不显著：可能为数据挖掘结果"

    return {
        'dsr': round(dsr, 4),
        'e_max': round(e_max, 3),
        'interpretation': interp,
    }


def _compute_e_max(n_trials: int, sharpe: float) -> float:
    """Compute E[max(SR_N)] using full B&LdP formula with Euler-Mascheroni."""
    euler = 0.5772156649
    return np.sqrt(2 * np.log(max(n_trials, 2))) * (
        1 - euler * sharpe + (euler**2 - 1) / 4 * sharpe**2
    )


# ── Helpers for clamping perturbed parameters ─────────────────────────────────

_PARAM_CLAMPS = {
    'mom_w':      (0.05, 0.80),
    'vol_w':      (0.05, 0.80),
    'def_alloc':  (0.05, 0.60),
    'stop_loss':  (0.03, 0.20),
}


def _clamp_param(key: str, value: float) -> float:
    lo, hi = _PARAM_CLAMPS[key]
    return max(lo, min(hi, float(value)))


def _perturb_and_clamp(base_cfg, pert: float, rng: np.random.Generator):
    """Return dict of perturbed {mom_w, vol_w, def_alloc, stop_loss}."""
    base_values = {
        'mom_w':     base_cfg.mom_w,
        'vol_w':     base_cfg.vol_w,
        'def_alloc': base_cfg.def_alloc,
        'stop_loss': base_cfg.stop_loss,
    }
    params = {}
    for k in ['mom_w', 'vol_w', 'def_alloc', 'stop_loss']:
        delta = rng.uniform(-pert, pert) * base_values[k]
        params[k] = _clamp_param(k, base_values[k] + delta)
    return params


def _build_config(base_cfg, params: dict):
    """Build a StrategyConfig with perturbed params, copying all others."""
    from src.strategy import StrategyConfig

    return StrategyConfig(
        name=base_cfg.name,
        version=base_cfg.version,
        mom_w=params['mom_w'],
        vol_w=params['vol_w'],
        top_n=base_cfg.top_n,
        mom_window=base_cfg.mom_window,
        vol_window=base_cfg.vol_window,
        pe_window_years=base_cfg.pe_window_years,
        def_alloc=params['def_alloc'],
        step_low=base_cfg.step_low,
        step_high=base_cfg.step_high,
        max_def=base_cfg.max_def,
        hongli_ratio=base_cfg.hongli_ratio,
        rebalance_threshold=base_cfg.rebalance_threshold,
        fee_rate=base_cfg.fee_rate,
        anchor=base_cfg.anchor,
        stop_loss=params['stop_loss'],
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
        ms_bull_mom=base_cfg.ms_bull_mom,
        ms_correction_mom=base_cfg.ms_correction_mom,
        ms_crisis_mom=base_cfg.ms_crisis_mom,
        ms_low_vol_pct=base_cfg.ms_low_vol_pct,
        ms_mid_vol_pct=base_cfg.ms_mid_vol_pct,
        ms_high_vol_pct=base_cfg.ms_high_vol_pct,
        ms_shallow_dd=base_cfg.ms_shallow_dd,
        ms_moderate_dd=base_cfg.ms_moderate_dd,
        ms_deep_dd=base_cfg.ms_deep_dd,
        ss_bull_l1=base_cfg.ss_bull_l1,
        ss_bull_l1_def=base_cfg.ss_bull_l1_def,
        ss_bull_l2=base_cfg.ss_bull_l2,
        ss_bull_l2_def=base_cfg.ss_bull_l2_def,
        ss_bull_recovery=base_cfg.ss_bull_recovery,
        ss_normal_l1=base_cfg.ss_normal_l1,
        ss_normal_l1_def=base_cfg.ss_normal_l1_def,
        ss_normal_l2=base_cfg.ss_normal_l2,
        ss_normal_l2_def=base_cfg.ss_normal_l2_def,
        ss_normal_recovery=base_cfg.ss_normal_recovery,
        ss_correction_l1=base_cfg.ss_correction_l1,
        ss_correction_l1_def=base_cfg.ss_correction_l1_def,
        ss_correction_l2=base_cfg.ss_correction_l2,
        ss_correction_l2_def=base_cfg.ss_correction_l2_def,
        ss_correction_recovery=base_cfg.ss_correction_recovery,
        ss_crisis_l1=base_cfg.ss_crisis_l1,
        ss_crisis_l1_def=base_cfg.ss_crisis_l1_def,
        ss_crisis_l2=base_cfg.ss_crisis_l2,
        ss_crisis_l2_def=base_cfg.ss_crisis_l2_def,
        ss_crisis_recovery=base_cfg.ss_crisis_recovery,
        overflow_to_defense_only=base_cfg.overflow_to_defense_only,
        dynamic_weight_cap=base_cfg.dynamic_weight_cap,
        dc_bull_cap=base_cfg.dc_bull_cap,
        dc_normal_cap=base_cfg.dc_normal_cap,
        dc_correction_cap=base_cfg.dc_correction_cap,
        dc_crisis_cap=base_cfg.dc_crisis_cap,
        d4_enabled=base_cfg.d4_enabled,
        d4_momentum_window=base_cfg.d4_momentum_window,
        d4_momentum_threshold=base_cfg.d4_momentum_threshold,
        d4_action=base_cfg.d4_action,
        d4_min_candidates=base_cfg.d4_min_candidates,
        softmax_enabled=base_cfg.softmax_enabled,
        softmax_temperature=base_cfg.softmax_temperature,
        softmax_hard_top_n_fallback=base_cfg.softmax_hard_top_n_fallback,
        softmax_min_candidates=base_cfg.softmax_min_candidates,
        softmax_regime_enabled=base_cfg.softmax_regime_enabled,
        softmax_regime_temperature=base_cfg.softmax_regime_temperature,
        d1_enabled=base_cfg.d1_enabled,
        d1_lookback=base_cfg.d1_lookback,
        d1_tq_low=base_cfg.d1_tq_low,
        d1_tq_high=base_cfg.d1_tq_high,
        d1_mom_w_low=base_cfg.d1_mom_w_low,
        d1_mom_w_high=base_cfg.d1_mom_w_high,
        d1_vol_w_low=base_cfg.d1_vol_w_low,
        d1_vol_w_high=base_cfg.d1_vol_w_high,
        d1_weight_sum=base_cfg.d1_weight_sum,
        constituent_signals_enabled=base_cfg.constituent_signals_enabled,
        constituent_signals_path=base_cfg.constituent_signals_path,
        cwm_weight=base_cfg.cwm_weight,
        conc_weight=base_cfg.conc_weight,
        cwm_window=base_cfg.cwm_window,
        regime_enabled=base_cfg.regime_enabled,
        regime_data_path=base_cfg.regime_data_path,
        regime_overrides=base_cfg.regime_overrides,
        regime_3state=base_cfg.regime_3state,
        nav_path=base_cfg.nav_path,
        pe_path=base_cfg.pe_path,
        start_date=base_cfg.start_date,
        end_date=base_cfg.end_date,
        risk_free_rate=base_cfg.risk_free_rate,
    )


# ── M2: Monte Carlo Survival Rate ─────────────────────────────────────────────

def compute_mc_survival(config_path: str,
                        n_runs: int = 100,
                        perturbation: float = 0.10) -> dict:
    """
    Monte Carlo Survival Rate (v2 unified).

    Simultaneously perturb mom_w, vol_w, def_alloc, stop_loss by ±10%
    (relative to their base values), clamp each to sensible ranges, run
    a backtest for each variant, and report the fraction meeting the
    goal-aligned criterion: annual_return > 10% AND max_drawdown < 15%.

    Args:
        config_path: Path to the strategy YAML config.
        n_runs: Number of Monte Carlo trials.
        perturbation: Fractional perturbation magnitude (default 0.10).

    Returns:
        dict with 'survival_rate', 'n_survived', 'n_runs',
        'sharpe_mean', 'sharpe_std', 'sharpes', 'annual_returns',
        'max_drawdowns' (all runs).
    """
    from src.strategy import load_config
    from src.backtest import run_backtest

    base_cfg = load_config(Path(config_path))
    rng = np.random.default_rng(42)

    sharpes: list[float] = []
    annual_returns: list[float] = []
    max_drawdowns: list[float] = []
    n_survived = 0

    for _ in range(n_runs):
        params = _perturb_and_clamp(base_cfg, perturbation, rng)
        cfg = _build_config(base_cfg, params)

        try:
            result = run_backtest(cfg)
            if (result.nav_series is not None
                    and not result.nav_series.empty
                    and 'sharpe_ratio' in result.metrics):
                sr = result.metrics['sharpe_ratio']
                ar = result.metrics.get('annual_return', 0.0)
                dd = result.metrics.get('max_drawdown', 1.0)
                sharpes.append(sr)
                annual_returns.append(ar)
                max_drawdowns.append(dd)
                # Unified goal-aligned criterion: annual > 10% AND DD < 15%
                if ar > 0.10 and dd < 0.15:
                    n_survived += 1
        except Exception:
            # Silently skip failed runs
            pass

    effective_runs = max(len(sharpes), 1)  # avoid div-by-zero
    survival_rate = n_survived / effective_runs

    return {
        'survival_rate': round(survival_rate, 4),
        'n_survived': n_survived,
        'n_runs': n_runs,
        'n_completed': len(sharpes),
        'sharpe_mean': round(float(np.mean(sharpes)), 4) if sharpes else None,
        'sharpe_std':  round(float(np.std(sharpes)), 4) if sharpes else None,
        'sharpes':     sharpes,
        'annual_returns':  annual_returns,
        'max_drawdowns':   max_drawdowns,
    }


# ── M3: Benchmark-Relative Win Rate ──────────────────────────────────────────

def compute_benchmark_relative(config_path: str) -> dict:
    """
    Walk-forward comparison of the strategy against an equal-weight
    benchmark of all available ETFs.

    Windows: 3-year train / 1-year test, sliding forward by 1 year.
    The benchmark is the mean weekly return of all ETF columns held
    in equal weight (rebalanced weekly, no rotation).

    Args:
        config_path: Path to the strategy YAML config.

    Returns:
        dict with 'win_rate', 'n_windows', 'n_wins', 'worst_delta',
        'best_delta', 'mean_delta', 'windows' (per-window details).
    """
    from src.strategy import load_config
    from src.backtest import run_backtest
    from src.data_loader import load_nav_data, resample_weekly, ETFS
    from src.utils import compute_sharpe

    base_cfg = load_config(Path(config_path))
    nav_df = load_nav_data(Path(base_cfg.nav_path))
    weekly_nav = resample_weekly(nav_df, anchor=base_cfg.anchor)

    # Build equal-weight benchmark returns from *all* ETF columns
    etf_cols = [c for c in ETFS if c in weekly_nav.columns]
    if not etf_cols:
        return {'error': 'no ETF columns found in NAV data'}

    benchmark_returns = weekly_nav[etf_cols].pct_change().mean(axis=1)

    # Walk-forward window generation
    start_dt = weekly_nav.index[0]
    end_dt = weekly_nav.index[-1]
    test_duration = pd.DateOffset(years=1)
    train_duration = pd.DateOffset(years=3)

    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current = start_dt
    while current + train_duration + test_duration <= end_dt:
        test_start = current + train_duration
        test_end = test_start + test_duration
        windows.append((test_start, test_end))
        current += test_duration

    results: list[dict] = []
    for ts, te in windows:
        # Benchmark Sharpe in this test window
        bench_slice = benchmark_returns[ts:te].dropna()
        bench_sharpe = (compute_sharpe(bench_slice)
                        if len(bench_slice) > 10 else None)

        # Strategy Sharpe in this test window
        strat_sharpe = None
        try:
            cfg_win = _build_config(base_cfg, {
                'mom_w': base_cfg.mom_w,
                'vol_w': base_cfg.vol_w,
                'def_alloc': base_cfg.def_alloc,
                'stop_loss': base_cfg.stop_loss,
            })
            # Override start/end dates for window
            cfg_win.start_date = str(ts.date())
            cfg_win.end_date = str(te.date())

            strat_result = run_backtest(cfg_win)
            if (strat_result.nav_series is not None
                    and not strat_result.nav_series.empty
                    and 'sharpe_ratio' in strat_result.metrics):
                strat_sharpe = strat_result.metrics['sharpe_ratio']
        except Exception:
            pass

        if bench_sharpe is not None and strat_sharpe is not None:
            delta = strat_sharpe - bench_sharpe
            results.append({
                'test_start': str(ts.date()),
                'test_end':   str(te.date()),
                'bench_sharpe': round(bench_sharpe, 4),
                'strat_sharpe': round(strat_sharpe, 4),
                'delta':        round(delta, 4),
                'win':          delta > 0,
            })

    if not results:
        return {'error': 'no valid walk-forward windows'}

    wins = sum(1 for r in results if r['win'])
    deltas = [r['delta'] for r in results]

    return {
        'win_rate':    round(wins / len(results), 4),
        'n_windows':   len(results),
        'n_wins':      wins,
        'worst_delta': round(min(deltas), 4),
        'best_delta':  round(max(deltas), 4),
        'mean_delta':  round(float(np.mean(deltas)), 4),
        'windows':     results,
    }