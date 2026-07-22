"""D4: Individual Momentum Filter -- DISABLED in v3.0 final.

Filters individually weak ETFs from the selected offensive set based on
an independent momentum window (separate from the scoring Momentum window).
Weak ETFs are either replaced by the next-best candidate or shifted to defense.

Disabled because it introduced parameter cliffs and did not improve
risk-adjusted returns in out-of-sample testing.
"""

import numpy as np


# Import deferred to avoid circular dependency at module load time
def _get_offensive_idx():
    from src.strategy import OFFENSIVE_IDX
    return OFFENSIVE_IDX


def apply_individual_momentum_filter(
    selected_idx: list[int],
    w_rets: np.ndarray,
    i: int,
    config,
    scores_vec: np.ndarray,
    off_idx: list[int] | None = None,
) -> tuple[list[int], float]:
    """D4: Individual ETF momentum filter.

    Pure stateless: only uses the current bar lookback window.

    Returns:
        (filtered_idx, extra_defense)
    """
    if not config.d4_enabled:
        return (selected_idx, 0.0)

    _off_idx = off_idx if off_idx is not None else _get_offensive_idx()

    window = config.d4_momentum_window
    threshold = config.d4_momentum_threshold

    if i < window:
        return (selected_idx, 0.0)

    valid_off = [j for j in _off_idx if not np.isnan(scores_vec[j])]
    if len(valid_off) < config.d4_min_candidates:
        return (selected_idx, 0.0)

    extra_defense = 0.0
    filtered = list(selected_idx)

    for idx in list(selected_idx):
        cumulative = np.prod(1 + w_rets[i - window:i, idx]) - 1

        if np.isnan(cumulative):
            continue

        if cumulative < threshold:
            if config.d4_action == 'replace':
                remaining = [j for j in _off_idx
                             if j not in filtered and not np.isnan(scores_vec[j])]
                if remaining:
                    remaining.sort(key=lambda j: scores_vec[j], reverse=True)
                    replacement = remaining[0]
                    filtered[filtered.index(idx)] = replacement
                else:
                    filtered.remove(idx)
                    extra_defense += 1.0 / (len(selected_idx) if len(selected_idx) > 0 else 1)
            elif config.d4_action == 'defense':
                filtered.remove(idx)
                extra_defense += 1.0 / (len(selected_idx) if len(selected_idx) > 0 else 1)

    # Ensure at least 1 offensive ETF retained
    if not filtered:
        return (selected_idx, 0.0)

    return (filtered, extra_defense)
