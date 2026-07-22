"""Strategy logic -- scoring, selection, defense ratio, allocation, rebalance/stop-loss checks.

v3.0 Final (C1 optimal parameters):
  - ENABLED:  inv_vol_allocation (D6), max_single_alloc cap (D2B), overflow_to_defense_only
  - DISABLED: tiered stop loss (D1), ptiered stop loss (Phase A-2), stateful stop loss,
              individual momentum filter (D4), softmax allocation (D5),
              dynamic weights (D1), constituent signals (T32), regime classifier (T35)

Legacy/disabled feature implementations are in src/legacy/ and re-exported
from this module for backward compatibility with backtest.py and tests.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


@dataclass
class StrategyConfig:
    """Strategy parameters loaded from YAML."""

    # 策略标识
    name: str = "虾池ETF轮动 v3.0"
    version: str = "3.0"

    # 评分权重
    mom_w: float = 0.35       # 4周动量权重
    vol_w: float = 0.30       # 20周波动率权重

    # 选股
    top_n: int = 2            # 选几只进攻 ETF
    score_margin: float = 0.0     # TOP_N 分数差距门槛（防噪声换仓）

    # 因子窗口
    mom_window: int = 4
    vol_window: int = 20
    pe_window_years: int = 5

    # 防御参数
    def_alloc: float = 0.25   # 基准防御比例
    step_low: float = 0.20    # vol 三段式下限
    step_high: float = 0.35   # vol 三段式上限
    max_def: float = 0.95     # 极限防御比例
    hongli_ratio: float = 0.50  # 防御层中红利低波占比

    # 调仓
    rebalance_threshold: float = 0.07   # 调仓阈值
    fee_rate: float = 0.00005           # 单边费率
    anchor: str = 'W-MON'

    # 风控（原始）
    stop_loss: float = 0.08        # 止损阈值
    recovery_weeks: int = 4        # 止损恢复观察周数

    # Legacy features preserved for backward compatibility
    # === 三层分级止损（D1）=== DISABLED in v3.0 final
    tiered_stop_loss: bool = False
    l1_drawdown: float = 0.04
    l1_defense: float = 0.50
    l2_drawdown: float = 0.06
    l2_defense: float = 0.95
    l3_weekly_drop: float = -0.03
    l3_down_weeks: int = 3
    l3_window: int = 4
    l2_recovery_weeks: int = 4
    l3_recovery_weeks: int = 2

    # === Phase A-2: 分层止损 (position-based) === DISABLED in v3.0 final
    ptiered_stop_loss: bool = False
    p_recovery_weeks: int = 8
    p_l1_dd_low: float = 0.05
    p_l1_dd_high: float = 0.08
    p_l1_position: float = 0.80
    p_l2_dd_low: float = 0.08
    p_l2_dd_high: float = 0.12
    p_l2_position: float = 0.50
    p_l3_dd_threshold: float = 0.12
    p_l3_position: float = 0.20

    # === 权重上限（D2B）=== ENABLED in v3.0 final (max_single_alloc=0.40)
    max_single_alloc: float = 1.0

    # === P1 Fix #1: 市场状态感知止损 === DISABLED in v3.0 final
    stateful_stop_loss: bool = False
    ms_bull_mom: float = 0.10
    ms_correction_mom: float = -0.05
    ms_crisis_mom: float = -0.12
    ms_low_vol_pct: float = 0.33
    ms_mid_vol_pct: float = 0.50
    ms_high_vol_pct: float = 0.67
    ms_shallow_dd: float = 0.03
    ms_moderate_dd: float = 0.08
    ms_deep_dd: float = 0.15
    ss_bull_l1: float = 0.08
    ss_bull_l1_def: float = 0.50
    ss_bull_l2: float = 0.12
    ss_bull_l2_def: float = 0.80
    ss_bull_recovery: int = 1
    ss_normal_l1: float = 0.06
    ss_normal_l1_def: float = 0.50
    ss_normal_l2: float = 0.10
    ss_normal_l2_def: float = 0.85
    ss_normal_recovery: int = 2
    ss_correction_l1: float = 0.04
    ss_correction_l1_def: float = 0.60
    ss_correction_l2: float = 0.07
    ss_correction_l2_def: float = 0.95
    ss_correction_recovery: int = 2
    ss_crisis_l1: float = 0.03
    ss_crisis_l1_def: float = 0.70
    ss_crisis_l2: float = 0.05
    ss_crisis_l2_def: float = 0.95
    ss_crisis_recovery: int = 2

    # === P1 Fix #2: 权重上限修复 === ENABLED (overflow_to_defense_only=true)
    overflow_to_defense_only: bool = True
    # Dynamic cap DISABLED in v3.0 final
    dynamic_weight_cap: bool = False
    dc_bull_cap: float = 0.50
    dc_normal_cap: float = 0.35
    dc_correction_cap: float = 0.30
    dc_crisis_cap: float = 0.25

    # === Direction D4: 单ETF动量过滤器 === DISABLED in v3.0 final
    d4_enabled: bool = False
    d4_momentum_window: int = 8
    d4_momentum_threshold: float = 0.0
    d4_action: str = 'replace'
    d4_min_candidates: int = 3

    # === Direction D5: Softmax-Weighted Allocation === DISABLED in v3.0 final
    softmax_enabled: bool = False
    softmax_temperature: float = 1.0
    softmax_hard_top_n_fallback: int = 2
    softmax_min_candidates: int = 2

    # === Phase 5b: Regime-Conditional Softmax === DISABLED in v3.0 final
    softmax_regime_enabled: dict = field(default_factory=dict)
    softmax_regime_temperature: dict = field(default_factory=dict)

    # === Direction D6: Inv-Vol Weighted Allocation === ENABLED in v3.0 final
    inv_vol_enabled: bool = False
    inv_vol_window: int = 8

    # === Direction D1: 动态动量/波动率权重 === DISABLED in v3.0 final
    d1_enabled: bool = False
    d1_lookback: int = 12
    d1_tq_low: float = 0.0
    d1_tq_high: float = 2.0
    d1_mom_w_low: float = 0.25
    d1_mom_w_high: float = 0.45
    d1_vol_w_low: float = 0.20
    d1_vol_w_high: float = 0.40
    d1_weight_sum: float = 0.65

    # === T32: Constituent-Stock Signals (CWM + CONC) === DISABLED in v3.0 final
    constituent_signals_enabled: bool = False
    constituent_signals_path: str = 'data/tushare/constituent_signals.csv'
    cwm_weight: float = 0.10
    conc_weight: float = 0.03
    cwm_window: int = 12

    # === T35: Market Regime Classifier === DISABLED in v3.0 final
    regime_enabled: bool = False
    regime_data_path: str = 'data/tushare/regime_signals.csv'
    regime_overrides: dict = field(default_factory=dict)
    regime_3state: bool = False

    # 数据路径
    nav_path: str = 'data/all_etfs_nav_2013_2026_h20269_scaled.csv'
    pe_path: str = 'data/300etf_pe_percentile_weekly.csv'
    start_date: str | None = None
    end_date: str | None = None

    # 报告
    risk_free_rate: float = 0.025


# ---------------------------------------------------------------------------
# Config loading helpers (regime-related parsing)
# ---------------------------------------------------------------------------

def _parse_regime_overrides(raw: dict) -> dict:
    """Convert string-keyed regime overrides from YAML to Regime-enum-keyed dict."""
    if not raw:
        return {}
    from src.regime import Regime
    STRING_TO_REGIME = {
        'RISK_ON': Regime.RISK_ON,
        'CAUTIOUS': Regime.CAUTIOUS,
        'DEFENSIVE': Regime.DEFENSIVE,
        'BUBBLE_WARN': Regime.BUBBLE_WARN,
        'CRISIS': Regime.CRISIS,
    }
    result = {}
    for key, val in raw.items():
        regime = STRING_TO_REGIME.get(key)
        if regime is not None and isinstance(val, dict):
            result[regime] = dict(val)
    return result


def _parse_regime_softmax_enabled(raw: dict) -> dict:
    """Parse per-regime softmax_enabled overrides from YAML regime section."""
    if not raw:
        return {}
    from src.regime import Regime
    STRING_TO_REGIME = {
        'RISK_ON': Regime.RISK_ON,
        'CAUTIOUS': Regime.CAUTIOUS,
        'DEFENSIVE': Regime.DEFENSIVE,
        'BUBBLE_WARN': Regime.BUBBLE_WARN,
        'CRISIS': Regime.CRISIS,
    }
    result = {}
    for key, val in raw.items():
        regime = STRING_TO_REGIME.get(key)
        if regime is not None and isinstance(val, dict):
            sm_enabled = val.get('softmax_enabled')
            if sm_enabled is not None:
                result[regime] = bool(sm_enabled)
    return result


def _parse_regime_softmax_temperature(raw: dict) -> dict:
    """Parse per-regime softmax_temperature overrides from YAML regime section."""
    if not raw:
        return {}
    from src.regime import Regime
    STRING_TO_REGIME = {
        'RISK_ON': Regime.RISK_ON,
        'CAUTIOUS': Regime.CAUTIOUS,
        'DEFENSIVE': Regime.DEFENSIVE,
        'BUBBLE_WARN': Regime.BUBBLE_WARN,
        'CRISIS': Regime.CRISIS,
    }
    result = {}
    for key, val in raw.items():
        regime = STRING_TO_REGIME.get(key)
        if regime is not None and isinstance(val, dict):
            sm_temp = val.get('softmax_temperature')
            if sm_temp is not None:
                result[regime] = float(sm_temp)
    return result


def load_config(config_path: str | Path) -> StrategyConfig:
    """
    Load strategy configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        StrategyConfig instance
    """
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)

    strategy = raw.get('strategy', {})
    scoring = raw.get('scoring', {})
    selection = raw.get('selection', {})
    factors_cfg = raw.get('factors', {})
    defense = raw.get('defense', {})
    rebalance = raw.get('rebalance', {})
    risk = raw.get('risk_control', {})
    stop_loss_cfg = raw.get('stop_loss', {})
    phase_a2_cfg = raw.get('phase_a2_stop_loss', {})
    allocation_cfg = raw.get('allocation', {})
    market_state_cfg = raw.get('market_state', {})
    d4_cfg = raw.get('d4_individual_filter', {})
    softmax_cfg = raw.get('softmax_allocation', {})
    inv_vol_cfg = raw.get('inv_vol_allocation', {})
    d1_cfg = raw.get('dynamic_weighting', {})
    constituent_cfg = raw.get('constituent_signals', {})
    regime_cfg = raw.get('regime_classifier', {})
    data_cfg = raw.get('data', {})
    reporting = raw.get('reporting', {})

    return StrategyConfig(
        name=strategy.get('name', '虾池ETF轮动'),
        version=strategy.get('version', '2.3'),
        mom_w=scoring.get('mom_w', 0.35),
        vol_w=scoring.get('vol_w', 0.30),
        top_n=selection.get('top_n', 2),
        score_margin=selection.get('score_margin', 0.0),
        mom_window=factors_cfg.get('mom_window', 4),
        vol_window=factors_cfg.get('vol_window', 20),
        pe_window_years=factors_cfg.get('pe_window_years', 5),
        def_alloc=defense.get('def_alloc', 0.25),
        step_low=defense.get('step_low', 0.20),
        step_high=defense.get('step_high', 0.35),
        max_def=defense.get('max_def', 0.95),
        hongli_ratio=defense.get('hongli_ratio', 0.50),
        rebalance_threshold=rebalance.get('threshold', 0.07),
        fee_rate=rebalance.get('fee_rate', 0.00005),
        anchor=rebalance.get('anchor', 'W-MON'),
        stop_loss=risk.get('stop_loss', 0.08),
        recovery_weeks=risk.get('recovery_weeks', 4),
        # D1: 三层分级止损 (DISABLED)
        tiered_stop_loss=stop_loss_cfg.get('tiered', False),
        l1_drawdown=stop_loss_cfg.get('l1_drawdown', 0.04),
        l1_defense=stop_loss_cfg.get('l1_defense', 0.50),
        l2_drawdown=stop_loss_cfg.get('l2_drawdown', 0.06),
        l2_defense=stop_loss_cfg.get('l2_defense', 0.95),
        l3_weekly_drop=stop_loss_cfg.get('l3_weekly_drop', -0.03),
        l3_down_weeks=stop_loss_cfg.get('l3_down_weeks', 3),
        l3_window=stop_loss_cfg.get('l3_window', 4),
        l2_recovery_weeks=stop_loss_cfg.get('l2_recovery_weeks', 4),
        # Phase A-2: 分层止损 (DISABLED)
        ptiered_stop_loss=phase_a2_cfg.get('enabled', False),
        p_recovery_weeks=phase_a2_cfg.get('recovery_weeks', 8),
        p_l1_dd_low=phase_a2_cfg.get('l1_dd_low', 0.05),
        p_l1_dd_high=phase_a2_cfg.get('l1_dd_high', 0.08),
        p_l1_position=phase_a2_cfg.get('l1_position', 0.80),
        p_l2_dd_low=phase_a2_cfg.get('l2_dd_low', 0.08),
        p_l2_dd_high=phase_a2_cfg.get('l2_dd_high', 0.12),
        p_l2_position=phase_a2_cfg.get('l2_position', 0.50),
        p_l3_dd_threshold=phase_a2_cfg.get('l3_dd_threshold', 0.12),
        p_l3_position=phase_a2_cfg.get('l3_position', 0.20),
        l3_recovery_weeks=stop_loss_cfg.get('l3_recovery_weeks', 2),
        # D2B: 权重上限
        max_single_alloc=allocation_cfg.get('max_single_alloc', 1.0),
        # P1 Fix #1: 市场状态感知止损 (DISABLED)
        stateful_stop_loss=market_state_cfg.get('stateful_stop_loss', False),
        ms_bull_mom=market_state_cfg.get('ms_bull_mom', 0.10),
        ms_correction_mom=market_state_cfg.get('ms_correction_mom', -0.05),
        ms_crisis_mom=market_state_cfg.get('ms_crisis_mom', -0.12),
        ms_low_vol_pct=market_state_cfg.get('ms_low_vol_pct', 0.33),
        ms_mid_vol_pct=market_state_cfg.get('ms_mid_vol_pct', 0.50),
        ms_high_vol_pct=market_state_cfg.get('ms_high_vol_pct', 0.67),
        ms_shallow_dd=market_state_cfg.get('ms_shallow_dd', 0.03),
        ms_moderate_dd=market_state_cfg.get('ms_moderate_dd', 0.08),
        ms_deep_dd=market_state_cfg.get('ms_deep_dd', 0.15),
        ss_bull_l1=market_state_cfg.get('ss_bull_l1', 0.08),
        ss_bull_l1_def=market_state_cfg.get('ss_bull_l1_def', 0.50),
        ss_bull_l2=market_state_cfg.get('ss_bull_l2', 0.12),
        ss_bull_l2_def=market_state_cfg.get('ss_bull_l2_def', 0.80),
        ss_bull_recovery=market_state_cfg.get('ss_bull_recovery', 1),
        ss_normal_l1=market_state_cfg.get('ss_normal_l1', 0.06),
        ss_normal_l1_def=market_state_cfg.get('ss_normal_l1_def', 0.50),
        ss_normal_l2=market_state_cfg.get('ss_normal_l2', 0.10),
        ss_normal_l2_def=market_state_cfg.get('ss_normal_l2_def', 0.85),
        ss_normal_recovery=market_state_cfg.get('ss_normal_recovery', 2),
        ss_correction_l1=market_state_cfg.get('ss_correction_l1', 0.04),
        ss_correction_l1_def=market_state_cfg.get('ss_correction_l1_def', 0.60),
        ss_correction_l2=market_state_cfg.get('ss_correction_l2', 0.07),
        ss_correction_l2_def=market_state_cfg.get('ss_correction_l2_def', 0.95),
        ss_correction_recovery=market_state_cfg.get('ss_correction_recovery', 2),
        ss_crisis_l1=market_state_cfg.get('ss_crisis_l1', 0.03),
        ss_crisis_l1_def=market_state_cfg.get('ss_crisis_l1_def', 0.70),
        ss_crisis_l2=market_state_cfg.get('ss_crisis_l2', 0.05),
        ss_crisis_l2_def=market_state_cfg.get('ss_crisis_l2_def', 0.95),
        ss_crisis_recovery=market_state_cfg.get('ss_crisis_recovery', 2),
        # P1 Fix #2: 权重上限修复
        overflow_to_defense_only=allocation_cfg.get('overflow_to_defense_only', True),
        dynamic_weight_cap=allocation_cfg.get('dynamic_weight_cap', False),
        dc_bull_cap=allocation_cfg.get('dc_bull_cap', 0.50),
        dc_normal_cap=allocation_cfg.get('dc_normal_cap', 0.35),
        dc_correction_cap=allocation_cfg.get('dc_correction_cap', 0.30),
        dc_crisis_cap=allocation_cfg.get('dc_crisis_cap', 0.25),
        # D4: 单ETF动量过滤器 (DISABLED)
        d4_enabled=d4_cfg.get('enabled', False),
        d4_momentum_window=min(d4_cfg.get('momentum_window', 8), 8),
        d4_momentum_threshold=max(d4_cfg.get('momentum_threshold', 0.0), -0.07),
        d4_action=d4_cfg.get('action', 'replace'),
        d4_min_candidates=d4_cfg.get('min_candidates', 3),
        # D5: Softmax-Weighted Allocation (DISABLED)
        softmax_enabled=softmax_cfg.get('enabled', False),
        softmax_temperature=softmax_cfg.get('temperature', 1.0),
        softmax_hard_top_n_fallback=softmax_cfg.get('hard_top_n_fallback', 2),
        softmax_min_candidates=softmax_cfg.get('min_candidates', 2),
        # D6: Inv-Vol Weighted Allocation (ENABLED)
        inv_vol_enabled=inv_vol_cfg.get('enabled', False),
        inv_vol_window=inv_vol_cfg.get('window', 8),
        # D1: 动态权重 (DISABLED)
        d1_enabled=d1_cfg.get('enabled', False),
        d1_lookback=d1_cfg.get('lookback', 12),
        d1_tq_low=d1_cfg.get('tq_low', 0.0),
        d1_tq_high=d1_cfg.get('tq_high', 2.0),
        d1_mom_w_low=d1_cfg.get('mom_w_low', 0.25),
        d1_mom_w_high=d1_cfg.get('mom_w_high', 0.45),
        d1_vol_w_low=d1_cfg.get('vol_w_low', 0.20),
        d1_vol_w_high=d1_cfg.get('vol_w_high', 0.40),
        d1_weight_sum=d1_cfg.get('weight_sum', 0.65),
        # T32: Constituent-Stock Signals (DISABLED)
        constituent_signals_enabled=constituent_cfg.get('enabled', False),
        constituent_signals_path=constituent_cfg.get('signals_path', 'data/tushare/constituent_signals.csv'),
        cwm_weight=constituent_cfg.get('cwm_weight', 0.10),
        conc_weight=constituent_cfg.get('conc_weight', 0.03),
        cwm_window=constituent_cfg.get('cwm_window', 12),
        # T35: Regime classifier (DISABLED)
        regime_enabled=regime_cfg.get('enabled', False),
        regime_data_path=regime_cfg.get('data_path', 'data/tushare/regime_signals.csv'),
        regime_overrides=_parse_regime_overrides(regime_cfg.get('regimes', {})),
        regime_3state=regime_cfg.get('three_state', False),
        # Phase 5b: regime-conditional softmax (DISABLED)
        softmax_regime_enabled=_parse_regime_softmax_enabled(
            regime_cfg.get('regimes', {})
        ),
        softmax_regime_temperature=_parse_regime_softmax_temperature(
            regime_cfg.get('regimes', {})
        ),
        nav_path=data_cfg.get('nav_path', ''),
        pe_path=data_cfg.get('pe_path', ''),
        start_date=data_cfg.get('start_date'),
        end_date=data_cfg.get('end_date'),
        risk_free_rate=reporting.get('risk_free_rate', 0.025),
    )


# ---------------------------------------------------------------------------
# ETF definitions (aligned with data_loader)
# ---------------------------------------------------------------------------
ETFS = ['纳指ETF', '红利低波ETF', '中证500ETF', '黄金ETF', '国债ETF']
OFFENSIVE_IDX = [0, 2, 3]  # 纳指, 中证500, 黄金
DEFENSIVE_IDX = [1, 4]     # 红利低波, 国债


# ---------------------------------------------------------------------------
# Core strategy functions (ACTIVE in v3.0 final)
# ---------------------------------------------------------------------------

def score_offensive(
    momentum: pd.DataFrame,
    volatility: pd.DataFrame,
    date: pd.Timestamp,
    config: StrategyConfig,
    off_idx: list[int] | None = None
) -> dict[str, float]:
    """
    Compute offensive ETF composite scores (v2.3 formula, val_w removed).

    score = mom_w * momentum - vol_w * volatility

    Only computed for OFFENSIVE ETFs. Active in v3.0 final.

    Args:
        momentum: Momentum DataFrame (already shifted)
        volatility: Volatility DataFrame (already shifted)
        date: Current rebalance date
        config: Strategy configuration
        off_idx: Offensive ETF index list (defaults to global OFFENSIVE_IDX)

    Returns:
        {"纳指ETF": 0.12, "中证500ETF": 0.05, "黄金ETF": 0.08}
    """
    _off_idx = off_idx if off_idx is not None else OFFENSIVE_IDX

    if date not in momentum.index:
        return {}

    scores = {}
    for j in _off_idx:
        etf = ETFS[j]
        if etf in momentum.columns and etf in volatility.columns:
            mom_val = momentum.loc[date, etf]
            vol_val = volatility.loc[date, etf]
            if pd.notna(mom_val) and pd.notna(vol_val):
                scores[etf] = config.mom_w * mom_val - config.vol_w * vol_val
            else:
                scores[etf] = float('-inf')

    return scores


def select_top(
    scores: dict[str, float],
    top_n: int
) -> list[str]:
    """
    Select the top_n highest-scoring offensive ETFs. Active in v3.0 final.

    Args:
        scores: ETF -> score dictionary
        top_n: Number to select

    Returns:
        ["纳指ETF", "黄金ETF"]
    """
    sorted_etfs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    valid = [(etf, s) for etf, s in sorted_etfs if s > float('-inf')]
    return [etf for etf, _ in valid[:top_n]]


def calculate_defense_ratio(
    nasdaq_vol: float,
    config: StrategyConfig
) -> float:
    """
    Three-segment defense ratio based on Nasdaq volatility. Active in v3.0 final.

    - nasdaq_vol < step_low:    def_alloc (baseline, e.g. 25%)
    - nasdaq_vol > step_high:   max_def (maximum, e.g. 95%)
    - between:                  linear interpolation

    Args:
        nasdaq_vol: Nasdaq annualized volatility
        config: Strategy configuration

    Returns:
        Defense ratio (0~1)
    """
    if pd.isna(nasdaq_vol):
        return config.def_alloc

    if nasdaq_vol < config.step_low:
        return config.def_alloc
    elif nasdaq_vol > config.step_high:
        return config.max_def
    else:
        ratio = (nasdaq_vol - config.step_low) / (config.step_high - config.step_low)
        return config.def_alloc + (config.max_def - config.def_alloc) * ratio


def allocate(
    selected: list[str],
    defense_ratio: float,
    config: StrategyConfig,
    etf_names: list[str] | None = None,
    off_idx: list[int] | None = None,
    def_idx: list[int] | None = None
) -> np.ndarray:
    """
    Compute full position allocation (numpy array). Active in v3.0 final.

    Defense layer: first defensive ETF gets hongli_ratio, rest split (1-hongli_ratio)
    Offensive layer: selected ETFs split (1-defense_ratio) equally

    Args:
        selected: Selected offensive ETF names
        defense_ratio: Defense ratio
        config: Strategy configuration
        etf_names: ETF name list (defaults to global ETFS)
        off_idx: Offensive ETF index list
        def_idx: Defensive ETF index list

    Returns:
        np.ndarray shape=(n_etfs,), per-ETF positions
    """
    _etf_names = etf_names if etf_names is not None else ETFS
    _def_idx = def_idx if def_idx is not None else DEFENSIVE_IDX
    n_etfs = len(_etf_names)

    alloc = np.zeros(n_etfs)

    # Defense layer
    if _def_idx:
        alloc[_def_idx[0]] = defense_ratio * config.hongli_ratio
        n_rest = len(_def_idx) - 1
        if n_rest > 0:
            rest_weight = defense_ratio * (1 - config.hongli_ratio) / n_rest
            for j in _def_idx[1:]:
                alloc[j] = rest_weight

    # Offensive layer
    if selected:
        off_weight = (1 - defense_ratio) / len(selected)
        for etf in selected:
            idx = _etf_names.index(etf)
            alloc[idx] = off_weight

    return alloc


def check_rebalance(
    current_alloc: np.ndarray,
    new_alloc: np.ndarray,
    threshold: float
) -> bool:
    """
    Check if any ETF position change exceeds threshold. Active in v3.0 final.

    Args:
        current_alloc: Current allocation array
        new_alloc: New allocation array
        threshold: Rebalance threshold (max single-ETF change)

    Returns:
        True if rebalancing is needed
    """
    max_change = np.max(np.abs(new_alloc - current_alloc))
    return max_change >= threshold


def check_stop_loss(
    current_nav: float,
    peak_nav: float,
    threshold: float = 0.08
) -> bool:
    """
    Check if stop-loss is triggered. Active in v3.0 final.

    Args:
        current_nav: Current NAV
        peak_nav: Peak NAV
        threshold: Stop-loss threshold (e.g. 0.08)

    Returns:
        True if stop-loss triggered
    """
    if peak_nav == 0:
        return False
    return (peak_nav - current_nav) / peak_nav >= threshold


# ---------------------------------------------------------------------------
# D2B: Max allocation cap (ACTIVE in v3.0 final with max_single_alloc=0.40)
# ---------------------------------------------------------------------------

def apply_max_alloc_cap(
    alloc: np.ndarray,
    max_single: float,
    offensive_idx: list[int],
    overflow_to_defense_only: bool = True,
    dynamic_cap: bool = False,
    market_state=None,
    config: StrategyConfig | None = None,
    def_idx: list[int] | None = None
) -> np.ndarray:
    """
    Apply single-asset weight cap to offensive ETFs.

    v2.5: overflow_to_defense_only=True sends excess to defense layer only.
    dynamic_cap=True uses market-state-dependent caps (DISABLED in v3.0 final).

    Active in v3.0 final (max_single_alloc=0.40).

    Args:
        alloc: Original allocation array (shape=(n_etfs,))
        max_single: Max single-asset weight (e.g. 0.40)
        offensive_idx: Offensive ETF index list
        overflow_to_defense_only: True=v2.5 fix, False=v2.4 behavior
        dynamic_cap: Whether to enable dynamic caps (disabled in v3.0)
        market_state: Current market state (needed for dynamic_cap)
        config: Strategy config (needed for dynamic_cap)
        def_idx: Defensive ETF index list (defaults to global DEFENSIVE_IDX)

    Returns:
        Corrected allocation array
    """
    # Import MarketState only when needed (for dynamic_cap path)
    _def_idx = def_idx if def_idx is not None else DEFENSIVE_IDX

    alloc = alloc.copy()
    if max_single >= 1.0:
        return alloc

    # Dynamic cap
    effective_max = max_single
    if dynamic_cap and market_state is not None and config is not None:
        from src.legacy.market_state import MarketState
        dynamic_caps = {
            MarketState.BULL:       config.dc_bull_cap,
            MarketState.NORMAL:     config.dc_normal_cap,
            MarketState.CORRECTION: config.dc_correction_cap,
            MarketState.CRISIS:     config.dc_crisis_cap,
        }
        effective_max = dynamic_caps.get(market_state, max_single)

    for idx in offensive_idx:
        if alloc[idx] > effective_max:
            excess = alloc[idx] - effective_max
            alloc[idx] = effective_max

            if overflow_to_defense_only:
                # v2.5: excess goes to defense layer
                n_def = len(_def_idx)
                if n_def > 0:
                    def_total = sum(alloc[j] for j in _def_idx)
                    if def_total > 0:
                        for j in _def_idx:
                            alloc[j] += excess * (alloc[j] / def_total)
                    else:
                        for j in _def_idx:
                            alloc[j] += excess / n_def
            else:
                # v2.4: original behavior (distribute to other offensive ETFs first)
                other_off = [j for j in offensive_idx
                            if j != idx and alloc[j] < effective_max]
                if other_off:
                    total_capacity = sum(effective_max - alloc[j] for j in other_off)
                    if total_capacity >= excess:
                        for j in other_off:
                            remaining = effective_max - alloc[j]
                            share = excess * (remaining / total_capacity)
                            alloc[j] += share
                    else:
                        for j in other_off:
                            alloc[j] = effective_max
                        residual = excess - total_capacity
                        n_def = len(_def_idx)
                        if n_def > 0:
                            def_total = sum(alloc[j] for j in _def_idx)
                            if def_total > 0:
                                for j in _def_idx:
                                    alloc[j] += residual * (alloc[j] / def_total)
                            else:
                                for j in _def_idx:
                                    alloc[j] += residual / n_def
                else:
                    # No other offensive ETFs to absorb -> send to defense
                    n_def = len(_def_idx)
                    if n_def > 0:
                        def_total = sum(alloc[j] for j in _def_idx)
                        if def_total > 0:
                            for j in _def_idx:
                                alloc[j] += excess * (alloc[j] / def_total)
                        else:
                            for j in _def_idx:
                                alloc[j] += excess / n_def

    return alloc


# ---------------------------------------------------------------------------
# Legacy feature re-exports (implementations in src/legacy/)
# These functions are DISABLED in v3.0 final but re-exported for
# backward compatibility with backtest.py and tests.
# ---------------------------------------------------------------------------

from src.legacy.tiered_stop import check_stop_loss_tiered  # noqa: E402  # DISABLED in v3.0
from src.legacy.ptiered_stop import check_stop_loss_ptiered  # noqa: E402  # DISABLED in v3.0
from src.legacy.market_state import (  # noqa: E402  # DISABLED in v3.0
    MarketState,
    detect_market_state,
    check_stop_loss_stateful,
)
from src.legacy.d4_filter import apply_individual_momentum_filter  # noqa: E402  # DISABLED in v3.0
from src.legacy.softmax import compute_softmax_allocation  # noqa: E402  # DISABLED in v3.0
from src.legacy.dynamic_weights import compute_dynamic_weights  # noqa: E402  # DISABLED in v3.0
