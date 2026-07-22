"""D1: Dynamic Momentum/Volatility Weights -- DISABLED in v3.0 final.

Dynamically adjusts scoring weights (mom_w, vol_w) based on the
offensive layer's overall trend quality:
  trend_quality = mean(12w_return) / max(mean(12w_vol), 0.01)
  tq_norm = clamp((tq - tq_low) / (tq_high - tq_low), 0, 1)
  mom_w = mom_w_low + tq_norm * (mom_w_high - mom_w_low)

Disabled because fixed weights (mom_w=1.0, vol_w=1.10) proved more
robust across all market regimes in hyperparameter scanning.
"""

import numpy as np


# Import deferred to avoid circular dependency at module load time
def _get_offensive_idx():
    from src.strategy import OFFENSIVE_IDX
    return OFFENSIVE_IDX


def compute_dynamic_weights(
    w_rets: np.ndarray,
    i: int,
    config,
    off_idx: list[int] | None = None,
) -> tuple[float, float]:
    """Dynamic momentum/volatility weights based on trend quality.

    Pure stateless: only uses the current bar lookback window.

    Returns:
        (dynamic_mom_w, dynamic_vol_w)
    """
    lookback = config.d1_lookback
    _off_idx = off_idx if off_idx is not None else _get_offensive_idx()

    if i < lookback:
        return (config.mom_w, config.vol_w)

    off_rets = []
    off_vols = []
    for j in _off_idx:
        etf_rets = w_rets[i - lookback:i, j]
        valid = etf_rets[~np.isnan(etf_rets)]
        if len(valid) < lookback * 0.5:
            continue
        cum_ret = np.prod(1 + valid) - 1
        vol = np.std(valid, ddof=0) * np.sqrt(52)
        off_rets.append(cum_ret)
        off_vols.append(vol)

    if not off_rets or not off_vols:
        return (config.mom_w, config.vol_w)

    mean_ret = np.mean(off_rets)
    mean_vol = np.mean(off_vols)

    trend_quality = mean_ret / max(mean_vol, 0.01)

    tq_norm = (trend_quality - config.d1_tq_low) / max(config.d1_tq_high - config.d1_tq_low, 0.01)
    tq_norm = max(0.0, min(1.0, tq_norm))

    dynamic_mom_w = config.d1_mom_w_low + tq_norm * (config.d1_mom_w_high - config.d1_mom_w_low)
    dynamic_vol_w = config.d1_weight_sum - dynamic_mom_w

    dynamic_vol_w = max(config.d1_vol_w_low, min(config.d1_vol_w_high, dynamic_vol_w))

    return (dynamic_mom_w, dynamic_vol_w)
