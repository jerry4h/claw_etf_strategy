"""D1: 三层分级止损 (Tiered Stop Loss) -- DISABLED in v3.0 final.

This module implements a three-tier stop-loss mechanism:
  L1 (warning): drawdown >= 4% -> forced defense >= 50%
  L2 (forced):  drawdown >= 6% -> forced defense = 95%
  L3 (fuse):    single-week drop >= 3% or sustained decline -> defense = 95%

Disabled because the simple single-layer stop_loss (8% drawdown, 4-week recovery)
in the v3.0 final config proved more robust after hyperparameter scanning.
"""


def check_stop_loss_tiered(
    current_nav: float,
    peak_nav: float,
    weekly_return: float,
    recent_weekly_returns: list[float],
    config,
) -> tuple[int, float]:
    """Three-tier stop loss.

    Returns:
        (level, forced_defense_ratio)
        level=0: no trigger
        level=1: L1 warning
        level=2: L2 forced stop
        level=3: L3 circuit breaker
    """
    if peak_nav <= 0:
        return (0, config.l2_defense)

    drawdown = (peak_nav - current_nav) / peak_nav

    # L3: circuit breaker -- single-week crash or sustained decline
    if weekly_return <= config.l3_weekly_drop:
        return (3, config.l2_defense)
    if len(recent_weekly_returns) >= config.l3_window:
        window_rets = recent_weekly_returns[-config.l3_window:]
        down_weeks = sum(1 for r in window_rets if r < 0)
        if down_weeks >= config.l3_down_weeks:
            return (3, config.l2_defense)

    # L2: forced stop
    if drawdown >= config.l2_drawdown:
        return (2, config.l2_defense)

    # L1: warning
    if drawdown >= config.l1_drawdown:
        return (1, config.l1_defense)

    return (0, 0.0)
