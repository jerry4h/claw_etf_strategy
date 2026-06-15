"""数据加载与预处理 — 加载 ETF 净值数据，清洗，生成周频序列。"""

from pathlib import Path

import numpy as np
import pandas as pd

# === ETF 定义 ===
ETFS: list[str] = ['纳指ETF', '红利低波ETF', '沪深300ETF', '黄金ETF', '国债ETF']
OFFENSIVE: list[str] = ['纳指ETF', '沪深300ETF', '黄金ETF']  # 进攻层
DEFENSIVE: list[str] = ['红利低波ETF', '国债ETF']           # 防御层

# ETF 索引（用于 numpy 数组操作）
ETFS_IDX = {name: i for i, name in enumerate(ETFS)}
OFFENSIVE_IDX = [ETFS_IDX[n] for n in OFFENSIVE]  # [0, 2, 3]
DEFENSIVE_IDX = [ETFS_IDX[n] for n in DEFENSIVE]   # [1, 4]


def load_nav_data(data_path: str | Path) -> pd.DataFrame:
    """
    加载 ETF 日净值数据，执行清洗。

    处理步骤（按 goal.md §3 的正确顺序）：
    1. 读 CSV → datetime index
    2. 删除全市场休市行（所有 ETF 为 NaN）
    3. 单只 ETF 缺失值 ffill
    4. 截断至所有 ETF 都有数据之后

    Args:
        data_path: CSV 文件路径

    Returns:
        DataFrame, index=日期(datetime), columns=ETF名称, values=净值(float)
    """
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # 如果列名不是 ETF 名称（可能是整数），重命名
    if df.columns[0] not in ETFS:
        df.columns = ETFS[:len(df.columns)]

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
