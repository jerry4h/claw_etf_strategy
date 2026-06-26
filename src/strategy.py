"""策略逻辑 — 评分、选股、防御比例、仓位分配、调仓/止损检查。"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


@dataclass
class StrategyConfig:
    """从 YAML 加载的策略参数"""

    # 策略标识
    name: str = "虾池ETF轮动 v2.3"
    version: str = "2.3"

    # 评分权重
    mom_w: float = 0.35       # 4周动量权重
    vol_w: float = 0.30       # 20周波动率权重

    # 选股
    top_n: int = 2            # 选几只进攻 ETF

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

    # === 三层分级止损（D1）=== 默认关闭
    tiered_stop_loss: bool = False
    l1_drawdown: float = 0.04      # L1 预警阈值
    l1_defense: float = 0.50       # L1 强制防御比例
    l2_drawdown: float = 0.06      # L2 强制止损阈值
    l2_defense: float = 0.95       # L2 强制防御比例
    l3_weekly_drop: float = -0.03  # L3 熔断单周跌幅
    l3_down_weeks: int = 3         # L3 熔断下跌周数
    l3_window: int = 4             # L3 观察窗口
    l2_recovery_weeks: int = 4     # L2 恢复观察期
    l3_recovery_weeks: int = 2     # L3 恢复观察期

    # === Phase A-2: 分层止损 (position-based) === 默认关闭
    ptiered_stop_loss: bool = False
    p_recovery_weeks: int = 8
    p_l1_dd_low: float = 0.05        # L1 回撤下界 (5%)
    p_l1_dd_high: float = 0.08       # L1 回撤上界 (8%)
    p_l1_position: float = 0.80      # L1 仓位保留比例 (80%)
    p_l2_dd_low: float = 0.08        # L2 回撤下界 (8%)
    p_l2_dd_high: float = 0.12       # L2 回撤上界 (12%)
    p_l2_position: float = 0.50      # L2 仓位保留比例 (50%)
    p_l3_dd_threshold: float = 0.12  # L3 触发阈值 (12%)
    p_l3_position: float = 0.20      # L3 仓位保留比例 (20%)

    # === 权重上限（D2B）=== 默认关闭（1.0 = 无上限）
    max_single_alloc: float = 1.0  # 单资产权重上限

    # === P1 Fix #1: 市场状态感知止损 === 默认关闭
    stateful_stop_loss: bool = False
    # 动量阈值
    ms_bull_mom: float = 0.10          # 12w收益 > 10% → BULL
    ms_correction_mom: float = -0.05   # 12w收益 < -5% → CORRECTION
    ms_crisis_mom: float = -0.12       # 12w收益 < -12% → CRISIS
    # 波动率百分位阈值
    ms_low_vol_pct: float = 0.33
    ms_mid_vol_pct: float = 0.50
    ms_high_vol_pct: float = 0.67
    # 回撤阈值
    ms_shallow_dd: float = 0.03
    ms_moderate_dd: float = 0.08
    ms_deep_dd: float = 0.15
    # 状态感知止损参数 — BULL
    ss_bull_l1: float = 0.08
    ss_bull_l1_def: float = 0.50
    ss_bull_l2: float = 0.12
    ss_bull_l2_def: float = 0.80
    ss_bull_recovery: int = 1
    # 状态感知止损参数 — NORMAL
    ss_normal_l1: float = 0.06
    ss_normal_l1_def: float = 0.50
    ss_normal_l2: float = 0.10
    ss_normal_l2_def: float = 0.85
    ss_normal_recovery: int = 2
    # 状态感知止损参数 — CORRECTION
    ss_correction_l1: float = 0.04
    ss_correction_l1_def: float = 0.60
    ss_correction_l2: float = 0.07
    ss_correction_l2_def: float = 0.95
    ss_correction_recovery: int = 2
    # 状态感知止损参数 — CRISIS
    ss_crisis_l1: float = 0.03
    ss_crisis_l1_def: float = 0.70
    ss_crisis_l2: float = 0.05
    ss_crisis_l2_def: float = 0.95
    ss_crisis_recovery: int = 2

    # === P1 Fix #2: 权重上限修复 ===
    overflow_to_defense_only: bool = True   # True=溢出仅入防御层
    # 动态上限（默认关闭，仅 stateful_stop_loss 启用时使用）
    dynamic_weight_cap: bool = False
    dc_bull_cap: float = 0.50
    dc_normal_cap: float = 0.35
    dc_correction_cap: float = 0.30
    dc_crisis_cap: float = 0.25

    # === Direction D4: 单ETF动量过滤器 === 默认关闭
    d4_enabled: bool = False
    d4_momentum_window: int = 8       # 独立于 scoring mom_window(4w) 的动量窗口
    d4_momentum_threshold: float = 0.0  # 动量阈值，低于此值标记为弱
    d4_action: str = 'replace'         # 'replace' 或 'defense'
    d4_min_candidates: int = 3         # 最少候选数才启用过滤

    # === Direction D5: Softmax-Weighted Allocation === 默认关闭
    softmax_enabled: bool = False
    softmax_temperature: float = 1.0
    softmax_hard_top_n_fallback: int = 2   # disabled 时使用此 top_n
    softmax_min_candidates: int = 2

    # === Phase 5b: Regime-Conditional Softmax (Direction A+D hybrid) ===
    softmax_regime_enabled: dict = field(default_factory=dict)  # Regime → bool
    softmax_regime_temperature: dict = field(default_factory=dict)  # Regime → float

    # === Direction D6: Inv-Vol8 Weighted Allocation (v3.0 Layer 2) === 默认关闭
    inv_vol_enabled: bool = False
    inv_vol_window: int = 8           # 波动率倒数加权回看窗口（周）

    # === Direction D1: 动态动量/波动率权重 === 默认关闭
    d1_enabled: bool = False
    d1_lookback: int = 12              # 趋势质量计算回看窗口 (12周)
    d1_tq_low: float = 0.0            # 趋势质量归一化下限
    d1_tq_high: float = 2.0           # 趋势质量归一化上限
    d1_mom_w_low: float = 0.25        # 动量权重下限
    d1_mom_w_high: float = 0.45       # 动量权重上限
    d1_vol_w_low: float = 0.20        # 波动率权重下限
    d1_vol_w_high: float = 0.40       # 波动率权重上限
    d1_weight_sum: float = 0.65       # mom_w + vol_w 目标和

    # === T32: Constituent-Stock Signals (CWM + CONC) === 默认关闭
    constituent_signals_enabled: bool = False
    constituent_signals_path: str = 'data/tushare/constituent_signals.csv'
    cwm_weight: float = 0.10           # α for CWM bonus modifier
    conc_weight: float = 0.03           # β for CONC bonus modifier
    cwm_window: int = 12                # CWM momentum lookback (weeks)

    # === T35: Market Regime Classifier (Direction A) === 默认关闭
    regime_enabled: bool = False
    regime_data_path: str = 'data/tushare/regime_signals.csv'
    regime_overrides: dict = field(default_factory=dict)
    # === T40 Fix 2: 3-State Simplified Regime === 默认关闭
    regime_3state: bool = False

    # 数据路径
    nav_path: str = 'data/all_etfs_nav_2013_2026_h20269_scaled.csv'
    pe_path: str = 'data/300etf_pe_percentile_weekly.csv'
    start_date: str | None = None
    end_date: str | None = None

    # 报告
    risk_free_rate: float = 0.025


def _parse_regime_overrides(raw: dict) -> dict:
    """Convert string-keyed regime overrides from YAML to Regime-enum-keyed dict."""
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
    从 YAML 文件加载策略配置。

    Args:
        config_path: YAML 配置文件路径

    Returns:
        StrategyConfig 实例
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
        # D1: 三层分级止损
        tiered_stop_loss=stop_loss_cfg.get('tiered', False),
        l1_drawdown=stop_loss_cfg.get('l1_drawdown', 0.04),
        l1_defense=stop_loss_cfg.get('l1_defense', 0.50),
        l2_drawdown=stop_loss_cfg.get('l2_drawdown', 0.06),
        l2_defense=stop_loss_cfg.get('l2_defense', 0.95),
        l3_weekly_drop=stop_loss_cfg.get('l3_weekly_drop', -0.03),
        l3_down_weeks=stop_loss_cfg.get('l3_down_weeks', 3),
        l3_window=stop_loss_cfg.get('l3_window', 4),
        l2_recovery_weeks=stop_loss_cfg.get('l2_recovery_weeks', 4),
        # Phase A-2: 分层止损 (position-based)
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
        # P1 Fix #1: 市场状态感知止损
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
        # D4: 单ETF动量过滤器 (Phase A-1: hard-clamp to fix parameter cliffs)
        d4_enabled=d4_cfg.get('enabled', False),
        d4_momentum_window=min(d4_cfg.get('momentum_window', 8), 8),
        d4_momentum_threshold=max(d4_cfg.get('momentum_threshold', 0.0), -0.07),
        d4_action=d4_cfg.get('action', 'replace'),
        d4_min_candidates=d4_cfg.get('min_candidates', 3),
        # D5: Softmax-Weighted Allocation
        softmax_enabled=softmax_cfg.get('enabled', False),
        softmax_temperature=softmax_cfg.get('temperature', 1.0),
        softmax_hard_top_n_fallback=softmax_cfg.get('hard_top_n_fallback', 2),
        softmax_min_candidates=softmax_cfg.get('min_candidates', 2),
        # D6: Inv-Vol8 Weighted Allocation
        inv_vol_enabled=inv_vol_cfg.get('enabled', False),
        inv_vol_window=inv_vol_cfg.get('window', 8),
        # D1: 动态权重
        d1_enabled=d1_cfg.get('enabled', False),
        d1_lookback=d1_cfg.get('lookback', 12),
        d1_tq_low=d1_cfg.get('tq_low', 0.0),
        d1_tq_high=d1_cfg.get('tq_high', 2.0),
        d1_mom_w_low=d1_cfg.get('mom_w_low', 0.25),
        d1_mom_w_high=d1_cfg.get('mom_w_high', 0.45),
        d1_vol_w_low=d1_cfg.get('vol_w_low', 0.20),
        d1_vol_w_high=d1_cfg.get('vol_w_high', 0.40),
        d1_weight_sum=d1_cfg.get('weight_sum', 0.65),
        # T32: Constituent-Stock Signals
        constituent_signals_enabled=constituent_cfg.get('enabled', False),
        constituent_signals_path=constituent_cfg.get('signals_path', 'data/tushare/constituent_signals.csv'),
        cwm_weight=constituent_cfg.get('cwm_weight', 0.10),
        conc_weight=constituent_cfg.get('conc_weight', 0.03),
        cwm_window=constituent_cfg.get('cwm_window', 12),
        # T35: Regime classifier
        regime_enabled=regime_cfg.get('enabled', False),
        regime_data_path=regime_cfg.get('data_path', 'data/tushare/regime_signals.csv'),
        regime_overrides=_parse_regime_overrides(regime_cfg.get('regimes', {})),
        # T40 Fix 2: 3-state regime
        regime_3state=regime_cfg.get('three_state', False),
        # Phase 5b: regime-conditional softmax
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


# === ETF 定义（与 data_loader 保持一致） ===
ETFS = ['纳指ETF', '红利低波ETF', '沪深300ETF', '黄金ETF', '国债ETF']
OFFENSIVE_IDX = [0, 2, 3]  # 纳指, 沪深300, 黄金
DEFENSIVE_IDX = [1, 4]     # 红利低波, 国债


def score_offensive(
    momentum: pd.DataFrame,
    volatility: pd.DataFrame,
    date: pd.Timestamp,
    config: StrategyConfig,
    off_idx: list[int] | None = None
) -> dict[str, float]:
    """
    计算进攻层 ETF 综合得分（v2.3 公式，已移除 val_w）。

    score = mom_w × momentum − vol_w × volatility

    仅对 OFFENSIVE ETF 计算。

    Args:
        momentum: 动量 DataFrame (已 shift)
        volatility: 波动率 DataFrame (已 shift)
        date: 当前调仓日期
        config: 策略配置
        off_idx: 进攻层 ETF 索引列表（默认使用全局 OFFENSIVE_IDX）

    Returns:
        {"纳指ETF": 0.12, "沪深300ETF": 0.05, "黄金ETF": 0.08}
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
    选择得分最高的 top_n 只进攻 ETF。

    Args:
        scores: ETF → 得分 的字典
        top_n: 选取数量

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
    vol 三段式防御比例计算。

    以纳指波动率代表进攻层整体风险：
    - nasdaq_vol < step_low:    def_alloc（基准防御，如 25%）
    - nasdaq_vol > step_high:   max_def（极限防御，如 95%）
    - 中间:                     线性插值

    Args:
        nasdaq_vol: 纳指年化波动率
        config: 策略配置

    Returns:
        防御比例 (0~1)
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
    计算完整仓位分配（numpy 数组）。

    防御层: 第一个防御ETF得 hongli_ratio，其余平分 (1-hongli_ratio)
    进攻层: selected ETFs 平分 (1-defence_ratio)

    Args:
        selected: 选中的进攻 ETF 名称列表
        defense_ratio: 防御比例
        config: 策略配置
        etf_names: ETF 名称列表（默认使用全局 ETFS）
        off_idx: 进攻层 ETF 索引列表
        def_idx: 防御层 ETF 索引列表

    Returns:
        np.ndarray shape=(n_etfs,), 各 ETF 仓位
    """
    _etf_names = etf_names if etf_names is not None else ETFS
    _def_idx = def_idx if def_idx is not None else DEFENSIVE_IDX
    n_etfs = len(_etf_names)

    alloc = np.zeros(n_etfs)

    # 防御层: 第一个防御ETF得 hongli_ratio，其余平分 (1-hongli_ratio)
    if _def_idx:
        alloc[_def_idx[0]] = defense_ratio * config.hongli_ratio
        n_rest = len(_def_idx) - 1
        if n_rest > 0:
            rest_weight = defense_ratio * (1 - config.hongli_ratio) / n_rest
            for j in _def_idx[1:]:
                alloc[j] = rest_weight

    # 进攻层
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
    检查是否有 ETF 仓位变化超过阈值，决定是否调仓。

    Args:
        current_alloc: 当前仓位数组
        new_alloc: 新仓位数组
        threshold: 调仓阈值（单只最大变化）

    Returns:
        True 表示需要调仓
    """
    max_change = np.max(np.abs(new_alloc - current_alloc))
    return max_change >= threshold


def check_stop_loss(
    current_nav: float,
    peak_nav: float,
    threshold: float = 0.08
) -> bool:
    """
    检查是否触发止损。

    Args:
        current_nav: 当前净值
        peak_nav: 峰值净值
        threshold: 止损阈值（如 0.08）

    Returns:
        True 表示触发止损
    """
    if peak_nav == 0:
        return False
    return (peak_nav - current_nav) / peak_nav >= threshold


# === D1: 三层分级止损 ===

def check_stop_loss_tiered(
    current_nav: float,
    peak_nav: float,
    weekly_return: float,
    recent_weekly_returns: list[float],
    config: StrategyConfig
) -> tuple[int, float]:
    """
    三层分级止损。

    层级:
      L1 (预警): 回撤 >= l1_drawdown(4%) → 防御比例 >= 50%
      L2 (强制): 回撤 >= l2_drawdown(6%) → 防御比例 = 95%
      L3 (熔断): 单周跌幅 >= l3_weekly_drop(3%) 或 l3_window周中至少l3_down_weeks周下跌 → 防御比例 = 95%

    Args:
        current_nav: 当前净值
        peak_nav: 峰值净值
        weekly_return: 当前周策略收益率
        recent_weekly_returns: 最近几周策略周收益率列表
        config: 策略配置

    Returns:
        (level, forced_defense_ratio)
        level=0: 无触发
        level=1: L1 预警
        level=2: L2 强制止损
        level=3: L3 熔断
    """
    if peak_nav <= 0:
        return (0, config.l2_defense)

    drawdown = (peak_nav - current_nav) / peak_nav

    # L3: 熔断 — 单周暴跌 或 连续下跌
    if weekly_return <= config.l3_weekly_drop:
        return (3, config.l2_defense)
    if len(recent_weekly_returns) >= config.l3_window:
        window_rets = recent_weekly_returns[-config.l3_window:]
        down_weeks = sum(1 for r in window_rets if r < 0)
        if down_weeks >= config.l3_down_weeks:
            return (3, config.l2_defense)

    # L2: 强制止损
    if drawdown >= config.l2_drawdown:
        return (2, config.l2_defense)

    # L1: 预警
    if drawdown >= config.l1_drawdown:
        return (1, config.l1_defense)

    return (0, 0.0)


# === Phase A-2: 分层止损 (position-based) ===

def check_stop_loss_ptiered(
    current_nav: float,
    peak_nav: float,
    config: StrategyConfig
) -> tuple[int, float]:
    """
    Phase A-2 分层止损 — 按回撤幅度逐步收紧风控。

    层级:
      L0: DD < 5%         → 无触发，使用 vol 三段式防御
      L1: DD 5-8%         → 温和减仓: 仓位从100%线性降至80% (def 0%→20%)
      L2: DD 8-12%        → 减仓: 仓位从80%线性降至50% (def 20%→50%)
      L3: DD >12%         → 紧急风控: 仓位强制20% (def 80%)

    与 D4 兼容: 分层止损设定防御下限，D4 extra_defense 可叠加增加防御。
    与 stateful_stop_loss 互斥: 两者不能同时启用。

    Args:
        current_nav: 当前净值
        peak_nav: 峰值净值
        config: 策略配置

    Returns:
        (level, defense_ratio)
        level=0: 无触发 (defense_ratio=0.0, 使用 vol 三段式防御)
        level=1: L1 温和减仓
        level=2: L2 减仓
        level=3: L3 紧急风控
    """
    if peak_nav <= 0:
        return (0, 0.0)

    drawdown = (peak_nav - current_nav) / peak_nav

    # L3: DD >12% → 紧急风控，仓位强制降至 20%
    if drawdown >= config.p_l3_dd_threshold:
        return (3, 1.0 - config.p_l3_position)   # def_ratio = 0.80

    # L2: DD 8-12% → 仓位从80%线性降至50%
    if drawdown >= config.p_l2_dd_low:
        t = (drawdown - config.p_l2_dd_low) / (config.p_l2_dd_high - config.p_l2_dd_low)
        t = min(max(t, 0.0), 1.0)
        position = config.p_l1_position + t * (config.p_l2_position - config.p_l1_position)
        return (2, 1.0 - position)

    # L1: DD 5-8% → 温和减仓: 仓位从100%线性降至80%
    if drawdown >= config.p_l1_dd_low:
        t = (drawdown - config.p_l1_dd_low) / (config.p_l1_dd_high - config.p_l1_dd_low)
        t = min(max(t, 0.0), 1.0)
        position = 1.0 + t * (config.p_l1_position - 1.0)   # 1.0 → 0.80
        return (1, 1.0 - position)

    return (0, 0.0)


# === P1 Fix #1: 市场状态感知止损 ===

class MarketState(Enum):
    BULL = "bull"
    NORMAL = "normal"
    CORRECTION = "correction"
    CRISIS = "crisis"


def detect_market_state(
    nasdaq_12w_ret: float,
    nasdaq_vol_pct: float,
    current_drawdown: float,
    config: StrategyConfig
) -> MarketState:
    """
    基于三信号投票判定市场状态 (v2.7 三态: BULL/NORMAL/CRISIS)。

    v2.7 将 CORRECTION 并入 NORMAL，形成三态系统。
    
    Args:
        nasdaq_12w_ret: 纳指滚动12周收益
        nasdaq_vol_pct: 纳指波动率2年滚动百分位 (0-1)
        current_drawdown: 当前回撤
        config: 策略配置
    
    Returns:
        MarketState (BULL, NORMAL, or CRISIS)
    """
    # 信号1: 动量 (v2.7: 三态，CORRECTION 并入 NORMAL)
    if nasdaq_12w_ret > config.ms_bull_mom:
        mom_signal = MarketState.BULL
    elif nasdaq_12w_ret < config.ms_crisis_mom:
        mom_signal = MarketState.CRISIS
    else:
        mom_signal = MarketState.NORMAL  # 含原 CORRECTION 区间
    
    # 信号2: 波动率百分位 (v2.7: 三态，CORRECTION 并入 NORMAL)
    if nasdaq_vol_pct < config.ms_low_vol_pct:
        vol_signal = MarketState.BULL
    elif nasdaq_vol_pct > config.ms_high_vol_pct:
        vol_signal = MarketState.CRISIS
    else:
        vol_signal = MarketState.NORMAL  # 含原 CORRECTION 区间
    
    # 信号3: 回撤 (v2.7: 三态，CORRECTION 并入 NORMAL)
    if current_drawdown >= config.ms_deep_dd:
        dd_signal = MarketState.CRISIS
    elif current_drawdown < config.ms_shallow_dd:
        dd_signal = MarketState.BULL
    else:
        dd_signal = MarketState.NORMAL  # 原 CORRECTION + NORMAL 合并
    
    # 投票: 每个信号 → 状态名，取多数
    # v2.7 三态顺序: CRISIS > NORMAL > BULL (保守优先)
    state_order = [MarketState.CRISIS, MarketState.NORMAL, MarketState.BULL]
    votes = {s: 0 for s in state_order}
    for signal in [mom_signal, vol_signal, dd_signal]:
        votes[signal] += 1
    
    # 取最高票，平局时取更保守的（更高防御的）
    max_votes = max(votes.values())
    for state in state_order:  # CRISIS first = more conservative
        if votes[state] == max_votes:
            return state
    return MarketState.NORMAL


def check_stop_loss_stateful(
    current_nav: float,
    peak_nav: float,
    state: MarketState,
    config: StrategyConfig,
    previous_def: float = 0.0,
    recovery_counter: int = 0,
    in_recovery: bool = False
) -> tuple[float, bool, int]:
    """
    市场状态感知止损 (v2.7)。

    替代 v2.6 的阶跃函数 + 锁定恢复，改为渐近坡道 + 衰减恢复。

    渐近坡道 (gradual ramp):
      - DD >= l2_dd: effective = l2_def (封顶), 用 max(previous_def, l2_def) 平滑
      - l1_dd <= DD < l2_dd: l1_def 到 l2_def 之间线性插值
      - DD < l1_dd: effective = 0.0

    衰减恢复 (decay recovery, 替代锁定):
      - decay = l2_def / recovery_wks 每周
      - effective_def = max(0.0, l2_def - decay * recovery_counter)
      - recovery_counter 每周递增，直到 effective_def <= 0 退出恢复

    Returns:
        (defense_ratio, in_recovery, recovery_counter)
    """
    if peak_nav <= 0:
        return (config.def_alloc, False, 0)

    drawdown = (peak_nav - current_nav) / peak_nav

    # 各状态的 L1/L2 阈值 (v2.7 三态: BULL/NORMAL/CRISIS)
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
    # 防御性回退：未知状态用 NORMAL 参数
    l1_dd, l1_def, l2_dd, l2_def, recovery_wks = state_params.get(
        state,
        (config.ss_normal_l1, config.ss_normal_l1_def,
         config.ss_normal_l2, config.ss_normal_l2_def,
         config.ss_normal_recovery),
    )

    # === 渐近坡道 (gradual ramp) ===
    if drawdown >= l2_dd:
        # DD >= l2_dd: l2_def 封顶，用 max(previous_def, l2_def) 平滑防骤降
        effective = max(previous_def, l2_def)
        # 触发 L2 → 进入恢复模式（counter 从 0 开始）
        return (effective, True, 0)
    elif drawdown >= l1_dd:
        # l1_dd <= DD < l2_dd: l1_def 到 l2_def 之间线性插值
        t = (drawdown - l1_dd) / (l2_dd - l1_dd)
        effective = l1_def + t * (l2_def - l1_def)
    else:
        # DD < l1_dd: 无触发
        effective = 0.0

    # === 衰减恢复 (decay recovery) ===
    if in_recovery:
        decay_per_week = l2_def / max(recovery_wks, 1)
        decay_def = max(0.0, l2_def - decay_per_week * recovery_counter)
        recovery_counter += 1
        if decay_def <= 0.0:
            # 恢复完成，释放给 vol 三段式
            return (max(effective, decay_def), False, 0)
        # 恢复期间：取衰减值与坡道值中较大者
        return (max(effective, decay_def), True, recovery_counter)

    return (effective, False, 0)


# === D2B: 单资产权重上限 ===

def apply_max_alloc_cap(
    alloc: np.ndarray,
    max_single: float,
    offensive_idx: list[int],
    overflow_to_defense_only: bool = True,
    dynamic_cap: bool = False,
    market_state: MarketState | None = None,
    config: StrategyConfig | None = None,
    def_idx: list[int] | None = None
) -> np.ndarray:
    """
    对进攻层 ETF 应用单资产权重上限。
    
    v2.5: overflow_to_defense_only=True 时，超出部分仅分配给防御层。
    dynamic_cap=True 时，max_single 由市场状态决定。
    
    Args:
        alloc: 原始仓位数组 (shape=(n_etfs,))
        max_single: 单资产最大权重（如 0.30），dynamic_cap 时为基础值
        offensive_idx: 进攻层 ETF 索引列表
        overflow_to_defense_only: True=v2.5修复, False=v2.4行为
        dynamic_cap: 是否启用动态上限
        market_state: 当前市场状态（dynamic_cap 时需要）
        config: 策略配置（dynamic_cap 时需要）
        def_idx: 防御层 ETF 索引列表（默认使用全局 DEFENSIVE_IDX）
    
    Returns:
        修正后的仓位数组
    """
    alloc = alloc.copy()
    if max_single >= 1.0:
        return alloc

    # 使用传入的 def_idx 或回退到全局 DEFENSIVE_IDX
    _def_idx = def_idx if def_idx is not None else DEFENSIVE_IDX
    
    # 动态上限
    effective_max = max_single
    if dynamic_cap and market_state is not None and config is not None:
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
                # v2.5: 超出部分全部转入防御层
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
                # v2.4: 原有行为（先分给其他进攻ETF）
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
                    # 无其他进攻ETF可吸收 → 直接归防御层
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


# === Direction D4: 单ETF动量过滤器 ===

def apply_individual_momentum_filter(
    selected_idx: list[int],
    w_rets: np.ndarray,
    i: int,
    config: StrategyConfig,
    scores_vec: np.ndarray,
    off_idx: list[int] | None = None
) -> tuple[list[int], float]:
    """
    D4: 单ETF动量过滤器。对已选中的进攻层 ETF，逐个检查独立动量窗口。
    动量低于阈值的 ETF 被标记为弱，根据 action 决定替换或转入防御。
    
    纯无状态：仅使用当前 bar 的回看窗口，无跨周持久状态。
    
    Args:
        selected_idx: 已选中的进攻 ETF 索引列表
        w_rets: 周收益率矩阵 (n_weeks-1, n_etfs)
        i: 当前周索引
        config: 策略配置
        scores_vec: 所有 ETF 的得分数组 (shape=(n_etfs,))
        off_idx: 进攻层 ETF 索引列表（默认使用全局 OFFENSIVE_IDX）
    
    Returns:
        (filtered_idx, extra_defense): 过滤后的选中索引列表，额外防御比例
    """
    if not config.d4_enabled:
        return (selected_idx, 0.0)

    _off_idx = off_idx if off_idx is not None else OFFENSIVE_IDX
    
    window = config.d4_momentum_window
    threshold = config.d4_momentum_threshold
    
    # 检查是否有足够历史数据
    if i < window:
        return (selected_idx, 0.0)
    
    # 检查候选数是否足够
    valid_off = [j for j in _off_idx if not np.isnan(scores_vec[j])]
    if len(valid_off) < config.d4_min_candidates:
        return (selected_idx, 0.0)
    
    extra_defense = 0.0
    filtered = list(selected_idx)
    
    # 对每个选中的 ETF 逐个检查动量
    for idx in list(selected_idx):
        # 计算 d4_momentum_window 动量
        # 使用 w_rets 累乘计算总收益
        cumulative = np.prod(1 + w_rets[i - window:i, idx]) - 1
        
        if np.isnan(cumulative):
            continue
        
        if cumulative < threshold:
            # 标记为弱
            if config.d4_action == 'replace':
                # 从剩余 offense 中找最高分未选中的 ETF
                remaining = [j for j in _off_idx
                           if j not in filtered and not np.isnan(scores_vec[j])]
                if remaining:
                    # 按得分排序，选最高分者替换
                    remaining.sort(key=lambda j: scores_vec[j], reverse=True)
                    replacement = remaining[0]
                    filtered[filtered.index(idx)] = replacement
                else:
                    # 无可用替换 → 移除，份额转入防御
                    filtered.remove(idx)
                    extra_defense += 1.0 / (len(selected_idx) if len(selected_idx) > 0 else 1)
            elif config.d4_action == 'defense':
                # 直接移除，份额转入防御
                filtered.remove(idx)
                extra_defense += 1.0 / (len(selected_idx) if len(selected_idx) > 0 else 1)
    
    # 确保至少保留 1 个进攻 ETF
    if not filtered:
        # 回退到原始选择
        return (selected_idx, 0.0)
    
    return (filtered, extra_defense)


# === Direction D1: 动态动量/波动率权重 ===

def compute_dynamic_weights(
    w_rets: np.ndarray,
    i: int,
    config: StrategyConfig,
    off_idx: list[int] | None = None
) -> tuple[float, float]:
    """
    D1: 动态动量/波动率权重。基于进攻层整体趋势质量动态调整权重。
    
    纯无状态：仅使用当前 bar 的回看窗口，无跨周持久状态。
    
    trend_quality = mean(offensive_12w_ret) / max(mean(offensive_12w_vol), 0.01)
    tq_norm = clamp((tq - tq_low) / (tq_high - tq_low), 0, 1)
    mom_w = mom_w_low + tq_norm * (mom_w_high - mom_w_low)
    vol_w = weight_sum - mom_w
    
    Args:
        w_rets: 周收益率矩阵 (n_weeks-1, n_etfs)
        i: 当前周索引
        config: 策略配置
        off_idx: 进攻层 ETF 索引列表（默认使用全局 OFFENSIVE_IDX）
    
    Returns:
        (dynamic_mom_w, dynamic_vol_w)
    """
    lookback = config.d1_lookback

    _off_idx = off_idx if off_idx is not None else OFFENSIVE_IDX

    if i < lookback:
        return (config.mom_w, config.vol_w)
    
    # 计算所有进攻 ETF 的 12 周收益和波动率
    off_rets = []
    off_vols = []
    for j in _off_idx:
        etf_rets = w_rets[i - lookback:i, j]
        valid = etf_rets[~np.isnan(etf_rets)]
        if len(valid) < lookback * 0.5:  # 至少 50% 有效数据
            continue
        # 12 周累积收益
        cum_ret = np.prod(1 + valid) - 1
        # 12 周年化波动率
        vol = np.std(valid, ddof=0) * np.sqrt(52)
        off_rets.append(cum_ret)
        off_vols.append(vol)
    
    if not off_rets or not off_vols:
        return (config.mom_w, config.vol_w)
    
    mean_ret = np.mean(off_rets)
    mean_vol = np.mean(off_vols)
    
    # 趋势质量
    trend_quality = mean_ret / max(mean_vol, 0.01)
    
    # 归一化
    tq_norm = (trend_quality - config.d1_tq_low) / max(config.d1_tq_high - config.d1_tq_low, 0.01)
    tq_norm = max(0.0, min(1.0, tq_norm))  # clamp to [0, 1]
    
    # 映射到权重范围
    dynamic_mom_w = config.d1_mom_w_low + tq_norm * (config.d1_mom_w_high - config.d1_mom_w_low)
    dynamic_vol_w = config.d1_weight_sum - dynamic_mom_w
    
    # 确保 vol_w 在合理范围内
    dynamic_vol_w = max(config.d1_vol_w_low, min(config.d1_vol_w_high, dynamic_vol_w))
    
    return (dynamic_mom_w, dynamic_vol_w)


# === Direction D5: Softmax-Weighted Allocation ===

def compute_softmax_allocation(
    scores: dict[str, float],
    temperature: float = 1.0
) -> dict[str, float]:
    """
    D5: Softmax-weighted allocation. Replaces hard top_n selection with
    continuous softmax weights across all offensive ETFs.

    weight_i = exp(score_i / temperature) / sum(exp(score_j / temperature) for all j)

    The sum of all softmax weights = 1.0, so total offensive budget unchanged.

    Args:
        scores: {etf_name: score} for all offensive ETFs
        temperature: Lower = more concentrated, higher = more spread

    Returns:
        {etf_name: weight} where weights sum to 1.0

    Edge cases:
        - NaN scores: filtered out
        - All -inf scores: uniform weights
        - Single ETF: weight = 1.0
        - temperature <= 0: raises ValueError
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    # Filter: only finite (not NaN, not -inf) scores
    valid = {k: v for k, v in scores.items() if np.isfinite(v)}

    if not valid:
        # All scores invalid — return uniform across all offensive ETFs
        return {k: 1.0 / len(scores) for k in scores}

    if len(valid) == 1:
        # Single valid ETF — weight = 1.0
        etf = next(iter(valid))
        return {etf: 1.0}

    # Softmax: exp(score / temperature) / sum
    # Use max-subtraction for numerical stability
    values = list(valid.values())
    keys = list(valid.keys())
    max_val = max(values)
    exps = np.exp((np.array(values) - max_val) / temperature)
    exps_sum = np.sum(exps)

    if exps_sum == 0 or not np.isfinite(exps_sum):
        # Degenerate: uniform
        return {k: 1.0 / len(keys) for k in keys}

    weights = exps / exps_sum
    return dict(zip(keys, weights))


# === I3: 梯度调仓 (已移除 — v2.5) ===
