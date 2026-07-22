"""D6: Inverse-Volatility Weighted Allocation -- ENABLED in v3.0 final.

This module is a reference implementation. The actual inv-vol logic is
inline in backtest.py for performance (avoids function call overhead in
the hot loop). Kept here for documentation and standalone testing.

Config: inv_vol_allocation.enabled=true, window=10
"""

import numpy as np


def apply_inv_vol_allocation(
    selected_off: list[int],
    w_rets: np.ndarray,
    i: int,
    inv_vol_window: int,
    def_ratio: float,
    alloc: np.ndarray,
) -> np.ndarray:
    """Apply inverse-volatility weighting to offensive allocation.

    Lower volatility ETFs receive higher weight. Pure continuous, no gating.

    Args:
        selected_off: Indices of selected offensive ETFs
        w_rets: Weekly returns matrix (n_weeks-1, n_etfs)
        i: Current week index
        inv_vol_window: Lookback window for volatility calculation
        def_ratio: Current defense ratio
        alloc: Current allocation array (modified in place)

    Returns:
        Modified allocation array
    """
    if not selected_off:
        return alloc

    if i >= inv_vol_window:
        inv_vols = []
        for j in selected_off:
            rets = w_rets[i - inv_vol_window:i, j]
            rets = rets[~np.isnan(rets)]
            if len(rets) < 3:
                inv_vols.append(0.0)
            else:
                vol = np.std(rets, ddof=0) * np.sqrt(52)
                inv_vols.append(1.0 / vol if vol > 0 else 0.0)
        total_inv = sum(inv_vols)
        if total_inv > 0:
            for k, j in enumerate(selected_off):
                alloc[j] = (1 - def_ratio) * (inv_vols[k] / total_inv)
        else:
            w = (1 - def_ratio) / len(selected_off)
            for j in selected_off:
                alloc[j] = w
    else:
        w = (1 - def_ratio) / len(selected_off)
        for j in selected_off:
            alloc[j] = w

    return alloc
