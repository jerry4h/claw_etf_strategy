"""Legacy feature modules -- disabled in v3.0 final config but preserved for
backward compatibility, testing, and potential future re-enablement.

All public symbols are re-exported here for convenient access:
    from src.legacy import MarketState, check_stop_loss_tiered, ...
"""

from src.legacy.tiered_stop import check_stop_loss_tiered
from src.legacy.ptiered_stop import check_stop_loss_ptiered
from src.legacy.market_state import (
    MarketState,
    detect_market_state,
    check_stop_loss_stateful,
)
from src.legacy.d4_filter import apply_individual_momentum_filter
from src.legacy.softmax import compute_softmax_allocation
from src.legacy.inv_vol import apply_inv_vol_allocation
from src.legacy.dynamic_weights import compute_dynamic_weights
from src.legacy.constituent_signals import (
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
