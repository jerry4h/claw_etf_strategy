"""D5: Softmax-Weighted Allocation -- DISABLED in v3.0 final.

Replaces hard top-N selection with continuous softmax weights across
all offensive ETFs. Lower temperature = more concentrated allocation.

Disabled because equal-weight top-N with inv-vol tilt (D6) outperformed
in risk-adjusted terms.
"""

import numpy as np


def compute_softmax_allocation(
    scores: dict[str, float],
    temperature: float = 1.0,
) -> dict[str, float]:
    """Softmax-weighted allocation.

    weight_i = exp(score_i / temperature) / sum(exp(score_j / temperature))

    Returns:
        {etf_name: weight} where weights sum to 1.0
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    valid = {k: v for k, v in scores.items() if np.isfinite(v)}

    if not valid:
        return {k: 1.0 / len(scores) for k in scores}

    if len(valid) == 1:
        etf = next(iter(valid))
        return {etf: 1.0}

    values = list(valid.values())
    keys = list(valid.keys())
    max_val = max(values)
    exps = np.exp((np.array(values) - max_val) / temperature)
    exps_sum = np.sum(exps)

    if exps_sum == 0 or not np.isfinite(exps_sum):
        return {k: 1.0 / len(keys) for k in keys}

    weights = exps / exps_sum
    return dict(zip(keys, weights))
