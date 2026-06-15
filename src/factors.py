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
    window: int = 20
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
        volatility[i] = np.std(w_rets[i - window:i], axis=0, ddof=0) * np.sqrt(52)

    return pd.DataFrame(volatility, index=weekly_nav.index, columns=weekly_nav.columns)


def calculate_pe_percentile(
    pe_df: pd.DataFrame,
    window_years: int = 5
) -> pd.DataFrame:
    """
    沪深300 PE-TTM 5年滚动分位数。

    ⚠️ 必须 shift(1) 确保无前视偏差：本周调仓只能用上周及之前的分位数。

    Args:
        pe_df: PE 分位数数据，index=日期, 单列 pe_percentile(float, 0~100)
        window_years: 滚动窗口年数，默认 5

    Returns:
        DataFrame, index=日期, 单列 pe_percentile(float, 0~1, 归一化)
    """
    col = pe_df.columns[0]
    window_days = window_years * 365

    def _rolling_percentile(series, window):
        result = pd.Series(np.nan, index=series.index)
        for i in range(len(series)):
            start = series.index[i] - pd.Timedelta(days=window)
            past = series[series.index <= series.index[i]]
            past = past[past.index >= start]
            if len(past) >= max(window_years * 40, 20):  # 至少需要足够数据点
                result.iloc[i] = (past < series.iloc[i]).mean()
        return result

    raw_pct = _rolling_percentile(pe_df[col], window_days)

    # 归一化为 0-1
    result = pd.DataFrame(raw_pct, columns=['pe_percentile'], index=pe_df.index)
    result = result.clip(0, 1)

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
            "momentum":     DataFrame (已 shift),
            "volatility":   DataFrame (已 shift),
            "pe_percentile": DataFrame (已 shift, 可选)
        }
    """
    if config is None:
        config = {}

    mom_window = config.get('factors', {}).get('mom_window', 4)
    vol_window = config.get('factors', {}).get('vol_window', 20)
    pe_window_years = config.get('factors', {}).get('pe_window_years', 5)

    momentum = calculate_momentum(weekly_nav, window=mom_window)
    volatility = calculate_volatility(weekly_nav, window=vol_window)

    # 注意：周频数据下，momentum[i] 使用 ret[i-window:i] 即价格[i-window]..价格[i]
    # 第 i 周的价格是已知的（周一收盘价），无前视偏差，因此不需要 shift(1)
    # reproduce_original.py 验证了这一点

    result = {
        'momentum': momentum,
        'volatility': volatility,
    }

    if pe_df is not None and not pe_df.empty:
        pe_pct = calculate_pe_percentile(pe_df, window_years=pe_window_years)
        pe_pct = pe_pct.shift(1)
        result['pe_percentile'] = pe_pct

    return result
