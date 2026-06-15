"""共享工具函数 — 年化、回撤、夏普、卡尔马等指标计算。零依赖。"""

import numpy as np
import pandas as pd


def annualize_return(total_return: float, n_weeks: int) -> float:
    """
    年化收益率: (1 + r) ^ (52 / n) − 1

    Args:
        total_return: 总收益率（如 4.4 表示 440%）
        n_weeks: 回测周期总周数

    Returns:
        年化收益率（小数，如 0.1406 表示 14.06%）
    """
    return (1 + total_return) ** (52 / n_weeks) - 1


def compute_max_drawdown(nav: pd.Series) -> float:
    """
    最大回撤。

    Args:
        nav: 净值序列

    Returns:
        最大回撤（正数，如 0.0821 表示 8.21%）
    """
    peak = nav.cummax()
    dd = (peak - nav) / peak
    return dd.max()


def compute_sharpe(returns: pd.Series, risk_free: float = 0.025) -> float:
    """
    标准夏普比率（扣无风险利率）。

    Args:
        returns: 周收益率序列
        risk_free: 年化无风险利率（默认 2.5%）

    Returns:
        标准夏普比率（年化）
    """
    if returns.std() == 0:
        return 0.0
    rfr_weekly = risk_free / 52
    excess = returns - rfr_weekly
    return excess.mean() / excess.std() * np.sqrt(52)


def compute_simple_sharpe(returns: pd.Series) -> float:
    """
    简化夏普比率（不扣无风险利率，仅供参考）。

    Args:
        returns: 周收益率序列

    Returns:
        简化夏普比率（年化）
    """
    if returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(52)


def compute_calmar(annual_return: float, max_drawdown: float) -> float:
    """
    卡尔马比率 = 年化收益率 / 最大回撤。

    Args:
        annual_return: 年化收益率（小数）
        max_drawdown: 最大回撤（正数，小数）

    Returns:
        卡尔马比率
    """
    if max_drawdown == 0:
        return float('inf')
    return annual_return / max_drawdown


def compute_annual_volatility(weekly_returns: pd.Series) -> float:
    """
    年化波动率 = std(周收益) × √52
    """
    return weekly_returns.std() * np.sqrt(52)
