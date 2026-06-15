import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent / "meta_data" / "all_etfs_nav_2013_2026_merged.csv"

# 红利ETF 和 红利低波ETF 已合并为 红利低波ETF
ETFS = ["纳指ETF", "红利低波ETF", "沪深300ETF", "黄金ETF", "国债ETF"]
DEFENSIVE = ["红利低波ETF", "国债ETF"]  # 防御层: 红利低波 + 国债
OFFENSIVE = ["纳指ETF", "沪深300ETF", "黄金ETF"]  # 进攻层

def load_nav_data():
    df = pd.read_csv(DATA_PATH)
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.set_index("日期").sort_index()

    # 步骤1：先把空字符串转为NaN（保持数值列的纯净性）
    for col in ETFS:
        if col in df.columns:
            df[col] = df[col].replace("", np.nan)

    # 步骤2：删除"全市场休市"的日期（所有ETF都是NaN）
    # 这些包括周末和节假日，不应该出现在日频序列中
    all_nan = df[ETFS].isna().all(axis=1)
    df = df[~all_nan].copy()

    # 步骤3：对单只ETF的个别缺失值做ffill（如停牌、数据缺失）
    for col in ETFS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].ffill()

    # 步骤4：删除所有ETF都有数据之前的行（如黄金ETF上市较晚，前期全NaN）
    # 找到所有ETF都有数据的第一个日期
    all_have_data = df[ETFS].notna().all(axis=1)
    first_valid = all_have_data.idxmax() if all_have_data.any() else None
    if first_valid is not None:
        df = df[df.index >= first_valid].copy()

    df = df.dropna(how="all")
    return df

def calculate_returns(nav_df):
    return_df = nav_df[ETFS].pct_change()
    return return_df

def get_weekly_data(nav_df):
    weekly = nav_df.resample("W-MON").last()
    return weekly

if __name__ == "__main__":
    nav = load_nav_data()
    print(f"Data loaded: {nav.shape}")
    print(f"Date range: {nav.index.min()} to {nav.index.max()}")
    print(nav.tail())
