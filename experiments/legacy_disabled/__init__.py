"""Legacy feature modules -- disabled in v3.0 final config but preserved for
backward compatibility, testing, and potential future re-enablement.

All public symbols are re-exported here for convenient access:
    from experiments.legacy_disabled import MarketState, check_stop_loss_tiered, ...
"""

from experiments.legacy_disabled.tiered_stop import check_stop_loss_tiered
from experiments.legacy_disabled.ptiered_stop import check_stop_loss_ptiered
from experiments.legacy_disabled.market_state import (
    MarketState,
    detect_market_state,
    check_stop_loss_stateful,
)
from experiments.legacy_disabled.d4_filter import apply_individual_momentum_filter
from experiments.legacy_disabled.softmax import compute_softmax_allocation
from experiments.legacy_disabled.inv_vol import apply_inv_vol_allocation
from experiments.legacy_disabled.dynamic_weights import compute_dynamic_weights
from experiments.legacy_disabled.constituent_signals import (
    load_constituent_signals,
    build_constituent_lookup,
    apply_constituent_bonus,
)

__all__ = [
    "check_stop_loss_tiered",
    "check_stop_loss_ptiered",
    "MarketState",
    "detect_market_state",
    "check_stop_loss_stateful",
    "apply_individual_momentum_filter",
    "compute_softmax_allocation",
    "apply_inv_vol_allocation",
    "compute_dynamic_weights",
    "load_constituent_signals",
    "build_constituent_lookup",
    "apply_constituent_bonus",
]
