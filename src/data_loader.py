"""数据加载与预处理 — 加载 ETF 净值数据，清洗，生成周频序列。"""

from pathlib import Path

import numpy as np
import pandas as pd

# === ETF 定义 ===
ETFS: list[str] = ['纳指ETF', '红利低波ETF', '中证500ETF', '黄金ETF', '国债ETF']
OFFENSIVE: list[str] = ['纳指ETF', '中证500ETF', '黄金ETF']  # 进攻层
DEFENSIVE: list[str] = ['红利低波ETF', '国债ETF']           # 防御层

# ETF 索引（用于 numpy 数组操作）
ETFS_IDX = {name: i for i, name in enumerate(ETFS)}
OFFENSIVE_IDX = [ETFS_IDX[n] for n in OFFENSIVE]  # [0, 2, 3]
DEFENSIVE_IDX = [ETFS_IDX[n] for n in DEFENSIVE]   # [1, 4]


def classify_etfs(etf_names: list[str]) -> tuple[list[int], list[int], int]:
    """
    Dynamically classify ETFs as offensive or defensive based on name patterns.
    Returns (OFFENSIVE_IDX, DEFENSIVE_IDX, NASDAQ_IDX).

    Classification rules:
      - DEFENSIVE: 红利低波, 红利, 可转债, 国债, 债券, 信用债
      - Everything else: OFFENSIVE
      - NASDAQ_IDX: matches 纳指, 标普500, NASDAQ, S&P 500 (for vol defense trigger)

    Args:
        etf_names: List of ETF name strings

    Returns:
        (offensive_idx: list[int], defensive_idx: list[int], nasdaq_idx: int)
    """
    defensive_keywords = ['红利低波', '红利', '可转债', '国债', '债券', '信用债']
    nasdaq_keywords = ['纳指', '标普500', 'NASDAQ', 'S&P', 'S&P500', 'SP500']

    offensive_idx = []
    defensive_idx = []
    nasdaq_idx = None

    for i, name in enumerate(etf_names):
        is_defensive = any(kw in name for kw in defensive_keywords)
        if is_defensive:
            defensive_idx.append(i)
        else:
            offensive_idx.append(i)

        # Find NASDAQ proxy for vol defense trigger
        if nasdaq_idx is None:
            for kw in nasdaq_keywords:
                if kw.lower() in name.lower():
                    nasdaq_idx = i
                    break

    # Fallback: if no NASDAQ found, use first offensive ETF
    if nasdaq_idx is None:
        nasdaq_idx = offensive_idx[0] if offensive_idx else 0

    return offensive_idx, defensive_idx, nasdaq_idx


def load_nav_data(data_path: str | Path, etf_list: list[str] | None = None) -> pd.DataFrame:
    """
    加载 ETF 日净值数据,执行清洗。

    处理步骤（按 goal.md §3 的正确顺序）：
    1. 读 CSV → datetime index
    2. 删除全市场休市行（所有 ETF 为 NaN）
    3. 单只 ETF 缺失值 ffill
    4. 截断至所有 ETF 都有数据之后

    Args:
        data_path: CSV 文件路径
        etf_list: ETF 名称列表。若为 None，使用默认5-ETF列表。
                  若CSV已包含ETF名称作为列名，自动检测并使用。

    Returns:
        DataFrame, index=日期(datetime), columns=ETF名称, values=净值(float)
    """
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Auto-detect: if columns are already meaningful ETF names, use them
    # Only rename if columns look like numeric indices (old format)
    if etf_list is None:
        # Check if columns are already named (Tushare format)
        first_col = str(df.columns[0])
        if first_col.isdigit() or (first_col.isascii() and len(first_col) <= 3):
            # Numeric column headers → old format, use default 5-ETF list
            etf_list = ETFS
        else:
            # Named columns → use as-is (Tushare format)
            etf_list = list(df.columns)

    if len(df.columns) != len(etf_list):
        # If mismatch, try to use columns as-is (variable-length universe)
        if list(df.columns) != etf_list:
            df.columns = etf_list[:len(df.columns)] if len(etf_list) >= len(df.columns) else etf_list

    # 步骤 1: 确保数值类型
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 步骤 2: 删除全市场休市行（所有 ETF 为 NaN — 周末 + A 股节假日）
    all_nan = df[df.columns].isna().all(axis=1)
    df = df[~all_nan].copy()

    # 步骤 3: 单只 ETF 缺失值 ffill（如 QDII 暂停申赎）
    for col in df.columns:
        df[col] = df[col].ffill()

    # 步骤 4: 截断至所有 ETF 都有数据之后
    all_valid = df[df.columns].notna().all(axis=1)
    if all_valid.any():
        first_valid = all_valid.idxmax()
        df = df[df.index >= first_valid].copy()

    df = df.dropna(how='all')
    return df


def calculate_daily_returns(nav_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算日收益率。

    Args:
        nav_df: 日净值 DataFrame

    Returns:
        DataFrame, index=日期, columns=ETF名称, values=日收益率(float)
    """
    return nav_df.pct_change()


def resample_weekly(nav_df: pd.DataFrame, anchor: str = 'W-MON') -> pd.DataFrame:
    """
    将日净值降采样为周净值（每周一收盘价）。

    如果数据已经是周频（索引间隔 ~7天），则直接返回。

    Args:
        nav_df: 日净值 DataFrame（或已是周频）
        anchor: pandas 周锚点 ('W-MON', 'W-TUE', ...)

    Returns:
        DataFrame, 周频净值
    """
    # 检测是否已经是周频数据（使用中位数间隔，比首条间隔更鲁棒）
    if len(nav_df) >= 3:
        gaps = np.diff(nav_df.index.astype('int64')) / 1e9 / 86400  # 转换为天数
        median_gap = np.median(gaps)
        if 6 <= median_gap <= 8:
            return nav_df.copy()

    weekly = nav_df.resample(anchor).last()
    return weekly.dropna(how='all')


def load_pe_percentile(pe_path: str | Path) -> pd.DataFrame:
    """
    加载沪深300 PE-TTM 分位数（0-100）。

    Args:
        pe_path: PE 分位数 CSV 文件路径

    Returns:
        DataFrame, index=日期, columns=['pe_percentile'], values=分位数(float, 0~100)
    """
    df = pd.read_csv(pe_path, index_col=0, parse_dates=True)
    df = df.dropna()
    return df
