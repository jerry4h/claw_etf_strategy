"""P1 Fix #1: Market State Detection & Stateful Stop Loss -- DISABLED in v3.0 final.

Implements a three-signal voting system to classify market regime
(BULL / NORMAL / CRISIS) and applies state-dependent stop-loss parameters
with asymptotic ramp and decay recovery.

Disabled because the simple single-layer stop proved more robust
in hyperparameter scanning and Monte Carlo simulation.
"""

from enum import Enum


class MarketState(Enum):
    BULL = "bull"
    NORMAL = "normal"
    CORRECTION = "correction"
    CRISIS = "crisis"


def detect_market_state(
    nasdaq_12w_ret: float,
    nasdaq_vol_pct: float,
    current_drawdown: float,
    config,
) -> MarketState:
    """Three-signal voting market state detection (v2.7 three-state).

    v2.7 merges CORRECTION into NORMAL for a three-state system.

    Args:
        nasdaq_12w_ret: Nasdaq rolling 12-week return
        nasdaq_vol_pct: Nasdaq volatility 2-year rolling percentile (0-1)
        current_drawdown: Current drawdown
        config: Strategy configuration

    Returns:
        MarketState (BULL, NORMAL, or CRISIS)
    """
    # Signal 1: momentum
    if nasdaq_12w_ret > config.ms_bull_mom:
        mom_signal = MarketState.BULL
    elif nasdaq_12w_ret < config.ms_crisis_mom:
        mom_signal = MarketState.CRISIS
    else:
        mom_signal = MarketState.NORMAL

    # Signal 2: volatility percentile
    if nasdaq_vol_pct < config.ms_low_vol_pct:
        vol_signal = MarketState.BULL
    elif nasdaq_vol_pct > config.ms_high_vol_pct:
        vol_signal = MarketState.CRISIS
    else:
        vol_signal = MarketState.NORMAL

    # Signal 3: drawdown
    if current_drawdown >= config.ms_deep_dd:
        dd_signal = MarketState.CRISIS
    elif current_drawdown < config.ms_shallow_dd:
        dd_signal = MarketState.BULL
    else:
        dd_signal = MarketState.NORMAL

    # Vote: majority wins, ties go to more conservative state
    state_order = [MarketState.CRISIS, MarketState.NORMAL, MarketState.BULL]
    votes = {s: 0 for s in state_order}
    for signal in [mom_signal, vol_signal, dd_signal]:
        votes[signal] += 1

    max_votes = max(votes.values())
    for state in state_order:  # CRISIS first = more conservative
        if votes[state] == max_votes:
            return state
    return MarketState.NORMAL


def check_stop_loss_stateful(
    current_nav: float,
    peak_nav: float,
    state: MarketState,
    config,
    previous_def: float = 0.0,
    recovery_counter: int = 0,
    in_recovery: bool = False,
) -> tuple[float, bool, int]:
    """State-aware stop loss (v2.7).

    Asymptotic ramp + decay recovery instead of step function + lock.

    Returns:
        (defense_ratio, in_recovery, recovery_counter)
    """
    if peak_nav <= 0:
        return (config.def_alloc, False, 0)

    drawdown = (peak_nav - current_nav) / peak_nav

    # State-specific L1/L2 thresholds
    state_params = {
        MarketState.BULL:   (config.ss_bull_l1, config.ss_bull_l1_def,
                              config.ss_bull_l2, config.ss_bull_l2_def,
                              config.ss_bull_recovery),
        MarketState.NORMAL: (config.ss_normal_l1, config.ss_normal_l1_def,
                              config.ss_normal_l2, config.ss_normal_l2_def,
                              config.ss_normal_recovery),
        MarketState.CRISIS: (config.ss_crisis_l1, config.ss_crisis_l1_def,
                              config.ss_crisis_l2, config.ss_crisis_l2_def,
                              config.ss_crisis_recovery),
    }
    l1_dd, l1_def, l2_dd, l2_def, recovery_wks = state_params.get(
        state,
        (config.ss_normal_l1, config.ss_normal_l1_def,
         config.ss_normal_l2, config.ss_normal_l2_def,
         config.ss_normal_recovery),
    )

    # === Asymptotic ramp ===
    if drawdown >= l2_dd:
        effective = max(previous_def, l2_def)
        return (effective, True, 0)
    elif drawdown >= l1_dd:
        t = (drawdown - l1_dd) / (l2_dd - l1_dd)
        effective = l1_def + t * (l2_def - l1_def)
    else:
        effective = 0.0

    # === Decay recovery ===
    if in_recovery:
        decay_per_week = l2_def / max(recovery_wks, 1)
        decay_def = max(0.0, l2_def - decay_per_week * recovery_counter)
        recovery_counter += 1
        if decay_def <= 0.0:
            return (max(effective, decay_def), False, 0)
        return (max(effective, decay_def), True, recovery_counter)

    return (effective, False, 0)
