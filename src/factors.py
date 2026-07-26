"""因子计算 — 动量、波动率、PE 分位数。纯函数，无副作用。"""

import numpy as np
import pandas as pd


def calculate_momentum(
    weekly_nav: pd.DataFrame,
    window: int = 4
) -> pd.DataFrame:
    """
    4 周动量（v2.3 公式）。

    计算：prod(1 + wrets[i−window:i]) − 1
    直接使用 numpy 对齐 reproduce_original.py 引擎。

    Args:
        weekly_nav: 周频净值 DataFrame, index=日期, columns=ETF
        window: 动量计算窗口（周数），默认 4

    Returns:
        DataFrame, index=日期, columns=ETF, values=动量(float)
    """
    prices = weekly_nav.values  # shape (n_weeks, n_etfs)
    n_weeks, n_etfs = prices.shape

    # 周收益率: w_rets[i] = (price[i+1] - price[i]) / price[i]
    # shape (n_weeks-1, n_etfs), 与 reproduce_original.py 对齐
    w_rets = np.diff(prices, axis=0) / prices[:-1]

    momentum = np.full((n_weeks, n_etfs), np.nan)
    for i in range(window, n_weeks):
        momentum[i] = np.prod(1 + w_rets[i - window:i], axis=0) - 1

    return pd.DataFrame(momentum, index=weekly_nav.index, columns=weekly_nav.columns)


def calculate_volatility(
    weekly_nav: pd.DataFrame,
    window: int = 20,
    ddof: int = 0
) -> pd.DataFrame:
    """
    20 周年化波动率。

    计算：std(wrets[i−window:i], ddof=0) × √52
    使用 ddof=0 对齐 reproduce_original.py 引擎。

    Args:
        weekly_nav: 周频净值 DataFrame
        window: 波动率计算窗口（周数），默认 20

    Returns:
        DataFrame, index=日期, columns=ETF, values=年化波动率(float)
    """
    prices = weekly_nav.values
    n_weeks, n_etfs = prices.shape

    # 周收益率（对齐 reproduce）
    w_rets = np.diff(prices, axis=0) / prices[:-1]

    volatility = np.full((n_weeks, n_etfs), np.nan)
    for i in range(window, n_weeks):
        volatility[i] = np.std(w_rets[i - window:i], axis=0, ddof=ddof) * np.sqrt(52)

    return pd.DataFrame(volatility, index=weekly_nav.index, columns=weekly_nav.columns)


def calculate_pe_percentile(
    pe_df: pd.DataFrame,
    window_years: int = 5
) -> pd.DataFrame:
    """
    沪深300 PE-TTM 5年滚动分位数（向量化实现）。

    对每个日期，计算当前 PE 值在过去 window_years 年窗口中的百分位。
    使用 numpy broadcast + searchsorted 替代逐行循环，速度提升 ~50x。

    ⚠️ 必须 shift(1) 确保无前视偏差：本周调仓只能用上周及之前的分位数。

    Args:
        pe_df: PE 分位数数据，index=日期, 单列 pe_percentile(float, 0~100)
        window_years: 滚动窗口年数，默认 5

    Returns:
        DataFrame, index=日期, 单列 pe_percentile(float, 0~1, 归一化)
    """
    col = pe_df.columns[0]
    series = pe_df[col]
    values = series.values.astype(float)
    dates = series.index
    n = len(series)
    window_days = window_years * 365

    # Pre-compute each position's lookback start index (binary search on dates)
    start_indices = np.searchsorted(
        dates, dates - pd.Timedelta(days=window_days)
    )

    min_points = max(window_years * 40, 20)

    # Vectorized rolling percentile using broadcast comparison:
    # For each position i, count how many values in [start_i, i] are < values[i]
    idx = np.arange(n)
    window_size = max((idx - start_indices).max() + 1, 1)

    # Build window matrix: window[i, k] = values[i - window_size + 1 + k]
    # Pad with NaN on the left for early positions
    padded = np.full(n + window_size, np.nan)
    padded[window_size:] = values

    k_offsets = np.arange(window_size)
    window_matrix = padded[idx[:, None] + k_offsets]  # shape (n, window_size)

    # Mask: positions within [start_i, i] (date range)
    col_pos = np.arange(window_size)
    in_window = col_pos >= (window_size - (idx[:, None] - start_indices[:, None]))

    # Count values < current value within date window.
    # NaN < x evaluates to False (numpy), matching old behavior:
    # numerator excludes NaN, denominator (len(past)) includes NaN.
    current_vals = values[:, None]
    less_than = (window_matrix < current_vals) & in_window
    counts = less_than.sum(axis=1).astype(float)
    total_in_window = in_window.sum(axis=1).astype(float)

    # Compute percentile; NaN where insufficient data
    result_values = np.where(
        total_in_window >= min_points,
        counts / np.maximum(total_in_window, 1),
        np.nan,
    )

    result = pd.DataFrame(
        np.clip(result_values, 0, 1),
        columns=['pe_percentile'],
        index=pe_df.index,
    )
    return result


def compute_all_factors(
    weekly_nav: pd.DataFrame,
    pe_df: pd.DataFrame | None = None,
    config: dict | None = None
) -> dict[str, pd.DataFrame]:
    """
    一次计算所有因子，自动 shift(1) 防前视偏差。

    Args:
        weekly_nav: 周频净值 DataFrame
        pe_df: PE 分位数数据（可选）
        config: 策略配置字典（从 YAML 加载），含 mom_window, vol_window 等

    Returns:
        {
            "momentum":     DataFrame (未 shift — 滚动窗口仅含历史收益, 无前视偏差),
            "volatility":   DataFrame (未 shift — 同上),
            "pe_percentile": DataFrame (已 shift(1), 可选 — PE为市场指标需额外延迟一周)
        }
    """
    if config is None:
        config = {}

    mom_window = config.get('factors', {}).get('mom_window', 6)
    vol_window = config.get('factors', {}).get('vol_window', 11)
    vol_ddof = config.get('factors', {}).get('vol_ddof', 0)
    pe_window_years = config.get('factors', {}).get('pe_window_years', 5)

    momentum = calculate_momentum(weekly_nav, window=mom_window)
    volatility = calculate_volatility(weekly_nav, window=vol_window, ddof=vol_ddof)

    result = {
        'momentum': momentum,
        'volatility': volatility,
    }

    if pe_df is not None and not pe_df.empty:
        pe_pct = calculate_pe_percentile(pe_df, window_years=pe_window_years)
        pe_pct = pe_pct.shift(1)
        result['pe_percentile'] = pe_pct

    return result
