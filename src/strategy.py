"""策略逻辑 — 评分、选股、防御比例、仓位分配、调仓/止损检查。"""

from dataclasses import dataclass, field
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

    # 风控
    stop_loss: float = 0.08        # 止损阈值
    recovery_weeks: int = 4        # 止损恢复观察周数

    # 数据路径
    nav_path: str = 'data/all_etfs_nav_2013_2026_h20269_scaled.csv'
    pe_path: str = 'data/300etf_pe_percentile_weekly.csv'
    start_date: str | None = None
    end_date: str | None = None

    # 报告
    risk_free_rate: float = 0.025


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
    config: StrategyConfig
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

    Returns:
        {"纳指ETF": 0.12, "沪深300ETF": 0.05, "黄金ETF": 0.08}
    """
    if date not in momentum.index:
        return {}

    scores = {}
    for j in OFFENSIVE_IDX:
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
    config: StrategyConfig
) -> np.ndarray:
    """
    计算完整仓位分配（numpy 数组）。

    防御层: 红利低波(defense_ratio × hongli_ratio) + 国债(defense_ratio × (1−hongli_ratio))
    进攻层: selected ETFs 平分 (1−defense_ratio)

    Args:
        selected: 选中的进攻 ETF 名称列表
        defense_ratio: 防御比例
        config: 策略配置

    Returns:
        np.ndarray shape=(5,), 各 ETF 仓位
    """
    alloc = np.zeros(5)

    # 防御层
    alloc[DEFENSIVE_IDX[0]] = defense_ratio * config.hongli_ratio       # 红利低波
    alloc[DEFENSIVE_IDX[1]] = defense_ratio * (1 - config.hongli_ratio)  # 国债

    # 进攻层
    if selected:
        off_weight = (1 - defense_ratio) / len(selected)
        for etf in selected:
            idx = ETFS.index(etf)
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
