import pandas as pd
import numpy as np
from data_loader import ETFS

def calculate_momentum(ret_df, window=20):
    mom = ret_df.rolling(window=window).sum() * 100
    return mom

def calculate_volatility(ret_df, window=20):
    vol = ret_df.rolling(window=window).std() * np.sqrt(252) * 100
    return vol

def calculate_valuation_percentile(nav_df, window=60):
    pct = {}
    for etf in ETFS:
        if etf in nav_df.columns:
            prices = nav_df[etf].dropna()
            if len(prices) > window:
                rolling_min = prices.rolling(window=window).min()
                rolling_max = prices.rolling(window=window).max()
                denom = rolling_max - rolling_min
                # 避免除零：当60日价格不变时，denom=0，此时估值百分位置为50（中性）
                pct[etf] = ((prices - rolling_min) / denom.replace(0, np.nan) * 100).fillna(50)
            else:
                pct[etf] = pd.Series(50, index=nav_df.index)
    return pd.DataFrame(pct)

def calculate_all_factors(nav_df, ret_df, mom_window=20, vol_window=20, val_window=60):
    momentum = calculate_momentum(ret_df, mom_window).shift(1)
    volatility = calculate_volatility(ret_df, vol_window).shift(1)
    valuation = calculate_valuation_percentile(nav_df, val_window).shift(1)

    factors = {
        "momentum": momentum,
        "volatility": volatility,
        "valuation": valuation
    }
    return factors
