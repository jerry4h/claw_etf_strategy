"""Shared strategy computation functions — single source of truth for
backtest engine (backtest.py) and live rebalance script (rebalance_live.py).

Contains: Layer 3.5 crisis correlation boost, dynamic hongli ratio,
inv-vol weighting, dynamic score margin, and trend confirmation.

All functions are pure (no side effects) and operate on numpy arrays.
"""

import math

import numpy as np
import pandas as pd

from src.strategy import StrategyConfig


def compute_crisis_boost(
    w_rets: np.ndarray,
    i: int,
    off_idx: list[int],
    config: StrategyConfig,
) -> float:
    """
    Layer 3.5: Crisis correlation convergence defense boost.

    When max pairwise |correlation| among offensive ETFs exceeds
    crisis_corr_threshold over a rolling window, boost defense ratio
    linearly up to crisis_corr_max_boost.

    Uses returns from positions [i-window, i), i.e., the `window`
    completed returns BEFORE the current week.

    Args:
        w_rets: Weekly returns array, shape (n_weeks, n_etfs).
                w_rets[k] = return from week k to k+1.
        i: Current week index.
        off_idx: Offensive ETF column indices.
        config: Strategy configuration.

    Returns:
        Defense ratio boost (0 to crisis_corr_max_boost).
    """
    window = config.crisis_corr_window
    threshold = config.crisis_corr_threshold
    slope = config.crisis_corr_slope
    max_boost = config.crisis_corr_max_boost

    if i < window or not off_idx or len(off_idx) < 2:
        return 0.0

    # Returns [i-window, i): the `window` completed returns before week i.
    off_ret_win = w_rets[i - window:i, off_idx]
    max_pair_corr = 0.0
    n_off = off_ret_win.shape[1]

    for a in range(n_off):
        for b in range(a + 1, n_off):
            mask = ~(np.isnan(off_ret_win[:, a]) | np.isnan(off_ret_win[:, b]))
            if mask.sum() >= 5:
                c = np.corrcoef(off_ret_win[mask, a], off_ret_win[mask, b])[0, 1]
                if not np.isnan(c):
                    max_pair_corr = max(max_pair_corr, abs(c))

    if max_pair_corr > threshold:
        return min((max_pair_corr - threshold) * slope, max_boost)
    return 0.0




def compute_ashare_vol_boost(
    vol_values: np.ndarray,
    i: int,
    ashare_idx: int,
    config: "StrategyConfig",
) -> float:
    """M3: A-share broad-market (中证500 ETF) volatility crisis boost.

    Additive defense boost mirroring compute_crisis_boost. Converts the
    中证500 ETF annualized vol to a trailing percentile (ex-ante, over
    ashare_vol_pct_window weeks) and boosts defense when the percentile
    exceeds ashare_vol_crisis_threshold. vol_values[i, ashare_idx] is causal
    (rolling std over past weeks, no look-ahead).

    Goal: catch A-share-native crashes (e.g. 2015) that the Nasdaq-only
    trigger misses.

    Returns:
        Defense ratio boost (0 to ashare_vol_max_boost).
    """
    if not getattr(config, "ashare_vol_boost_enabled", False):
        return 0.0
    if ashare_idx < 0 or i < 20:
        return 0.0
    threshold = config.ashare_vol_crisis_threshold
    slope = config.ashare_vol_slope
    max_boost = config.ashare_vol_max_boost
    window = config.ashare_vol_pct_window

    current_vol = vol_values[i, ashare_idx]
    if pd.isna(current_vol):
        return 0.0
    lo = max(20, i - window)
    hist = [vol_values[j, ashare_idx] for j in range(lo, i)
            if not pd.isna(vol_values[j, ashare_idx])]
    if len(hist) < 10:
        return 0.0
    pct = sum(1 for v in hist if v < current_vol) / len(hist)
    if pct > threshold:
        return min((pct - threshold) * slope, max_boost)
    return 0.0


def compute_dynamic_hongli(hl_vol: float, config: StrategyConfig) -> float:
    """
    Dynamic hongli_ratio based on hongli ETF's own volatility.

    Formula: clip(intercept - vol_coeff * vol, 0, intercept)

    Args:
        hl_vol: Hongli ETF annualized volatility.
        config: Strategy configuration.

    Returns:
        Effective hongli ratio (0 to hongli_intercept).
    """
    if np.isnan(hl_vol):
        return config.hongli_ratio
    return float(np.clip(
        config.hongli_intercept - config.hongli_vol_coeff * hl_vol,
        0,
        config.hongli_intercept,
    ))


def compute_inv_vol_weights(
    w_rets: np.ndarray,
    indices: list[int],
    i: int,
    window: int,
    vol_ddof: int = 0,
) -> list[float]:
    """
    Inverse-volatility weights for selected ETFs.

    Args:
        w_rets: Weekly returns array, shape (n_weeks, n_etfs).
                Can also be a pandas DataFrame (uses .iloc).
        indices: Column indices of selected ETFs.
        i: Current week index.
        window: Lookback window for volatility calculation.

    Returns:
        List of weights (same order as `indices`), summing to 1.0.
    """
    if i < window or not indices:
        n = max(len(indices), 1)
        return [1.0 / n] * len(indices)

    inv_vols = []
    for j in indices:
        # Extract returns: works for both numpy and pandas
        if hasattr(w_rets, 'iloc'):
            rets = w_rets.iloc[i - window:i, j].dropna().values
        else:
            rets = w_rets[i - window:i, j]
            rets = rets[~np.isnan(rets)]

        if len(rets) < 3:
            inv_vols.append(1.0 / 0.20)  # default vol = 20%
        else:
            vol = float(np.std(rets, ddof=vol_ddof) * math.sqrt(52))
            inv_vols.append(1.0 / max(vol, 0.05))

    total_inv = sum(inv_vols)
    if total_inv > 0:
        return [v / total_inv for v in inv_vols]
    n = max(len(indices), 1)
    return [1.0 / n] * len(indices)


def compute_score_margin(
    gap: float,
    gap_history: list[float],
    config: StrategyConfig,
) -> tuple[float, list[float]]:
    """
    Compute effective score margin (static + dynamic).

    effective_margin = score_margin + dynamic_sensitivity * std(gap_history)

    Args:
        gap: Current score gap between #top_n and #(top_n+1).
        gap_history: Rolling history of recent gaps (mutated in place).
        config: Strategy configuration.

    Returns:
        (effective_margin, updated_gap_history)
    """
    eff_margin = config.score_margin

    if config.dynamic_margin_sensitivity > 0:
        gap_history.append(gap)
        if len(gap_history) > config.dynamic_margin_window:
            gap_history.pop(0)
        if len(gap_history) >= 2:
            gap_std = float(np.std(gap_history))
            eff_margin = config.score_margin + config.dynamic_margin_sensitivity * gap_std

    return eff_margin, gap_history


def compute_snr_margin(
    gap: float,
    current_vol: float,
    snr_state: dict,
    config: "StrategyConfig",
) -> tuple[float, float, dict]:
    """v4.0 SNR 自适应 effective margin + threshold。

    用 EWMA 平滑 score gap 计算信噪比(SNR)——EWMA 无硬截断窗口,
    不受"一期数据进出 rolling 窗口"导致的因子跳变污染。
    SNR 低时 margin 放大(减少噪声换仓),波动高于 baseline 时 threshold 放大(减少摩擦)。

    Returns: (effective_margin, effective_threshold, updated_snr_state)
    """
    hl = max(config.snr_ewma_halflife, 1)
    alpha = 1.0 - np.exp(-np.log(2.0) / hl)

    # EWMA 更新 gap 均值和方差
    eg = snr_state.get('ewma_gap', gap)
    ev = snr_state.get('ewma_var', 0.0)
    eg = alpha * gap + (1.0 - alpha) * eg
    ev = alpha * (gap - eg) ** 2 + (1.0 - alpha) * ev

    # SNR = |均值| / 标准差
    snr = abs(eg) / (np.sqrt(ev) + 1e-8)

    # margin 自适应: SNR < 1.5(经验"可信下限") 时放大 margin
    margin_mult = max(1.0, 1.5 / (snr + 1e-8))
    eff_margin = config.score_margin * margin_mult

    # threshold 自适应: 当前波动 > baseline 时按比例放大
    baseline = config.snr_vol_baseline if config.snr_vol_baseline > 0 else 0.18
    vol_mult = max(1.0, current_vol / baseline) if current_vol > 0 else 1.0
    eff_threshold = config.rebalance_threshold * vol_mult

    new_state = {'ewma_gap': float(eg), 'ewma_var': float(ev)}
    return eff_margin, eff_threshold, new_state


def apply_trend_confirmation(
    candidate_sel: list[int],
    last_selected: list[int] | None,
    pending_selected: frozenset | None,
    pending_count: int,
    config: StrategyConfig,
) -> tuple[list[int], frozenset | None, int]:
    """
    Apply trend confirmation filter.

    Only switch to new selection if the candidate set has been consistent
    for `trend_confirm_weeks` consecutive weeks.

    Args:
        candidate_sel: Candidate offensive ETF indices.
        last_selected: Previously selected offensive ETF indices.
        pending_selected: Current pending candidate frozenset.
        pending_count: Weeks the pending set has been consistent.
        config: Strategy configuration.

    Returns:
        (selected_off, updated_pending, updated_count)
    """
    if config.trend_confirm_weeks <= 0 or last_selected is None:
        return candidate_sel, pending_selected, pending_count

    candidate_set = frozenset(candidate_sel)
    last_set = frozenset(last_selected)

    if candidate_set != pending_selected:
        pending_selected = candidate_set
        pending_count = 1
    else:
        pending_count += 1

    if candidate_set != last_set and pending_count < config.trend_confirm_weeks:
        return list(last_selected), pending_selected, pending_count

    return candidate_sel, pending_selected, pending_count
