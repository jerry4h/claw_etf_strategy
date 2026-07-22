"""Phase A-2: Position-based Tiered Stop Loss -- DISABLED in v3.0 final.

Progressive position reduction based on drawdown severity:
  L0: DD < 5%  -> no trigger, use vol-based defense
  L1: DD 5-8%  -> gentle reduction: position 100% -> 80%
  L2: DD 8-12% -> reduction: position 80% -> 50%
  L3: DD > 12% -> emergency: position forced to 20%

Disabled because the simple single-layer stop proved more robust.
"""


def check_stop_loss_ptiered(
    current_nav: float,
    peak_nav: float,
    config,
) -> tuple[int, float]:
    """Phase A-2 position-based tiered stop loss.

    Returns:
        (level, defense_ratio)
        level=0: no trigger (defense_ratio=0.0)
        level=1: L1 gentle reduction
        level=2: L2 reduction
        level=3: L3 emergency
    """
    if peak_nav <= 0:
        return (0, 0.0)

    drawdown = (peak_nav - current_nav) / peak_nav

    # L3: DD >12% -> emergency, position forced to 20%
    if drawdown >= config.p_l3_dd_threshold:
        return (3, 1.0 - config.p_l3_position)   # def_ratio = 0.80

    # L2: DD 8-12% -> position 80% -> 50%
    if drawdown >= config.p_l2_dd_low:
        t = (drawdown - config.p_l2_dd_low) / (config.p_l2_dd_high - config.p_l2_dd_low)
        t = min(max(t, 0.0), 1.0)
        position = config.p_l1_position + t * (config.p_l2_position - config.p_l1_position)
        return (2, 1.0 - position)

    # L1: DD 5-8% -> position 100% -> 80%
    if drawdown >= config.p_l1_dd_low:
        t = (drawdown - config.p_l1_dd_low) / (config.p_l1_dd_high - config.p_l1_dd_low)
        t = min(max(t, 0.0), 1.0)
        position = 1.0 + t * (config.p_l1_position - 1.0)   # 1.0 -> 0.80
        return (1, 1.0 - position)

    return (0, 0.0)
