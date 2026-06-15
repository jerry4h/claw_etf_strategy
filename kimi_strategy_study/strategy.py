import pandas as pd
import numpy as np
from data_loader import ETFS

DEFENSIVE = ["红利低波ETF", "国债ETF"]
OFFENSIVE = ["纳指ETF", "沪深300ETF", "黄金ETF"]

def calculate_composite_score(momentum, volatility, valuation, etf_list,
                               mom_w=0.4, vol_w=0.4, val_w=0.2):
    scores = pd.DataFrame(index=momentum.index)

    for etf in etf_list:
        if etf in momentum.columns and etf in volatility.columns and etf in valuation.columns:
            mom_norm = momentum[etf] / 100
            vol_norm = volatility[etf] / 100
            val_norm = valuation[etf] / 100

            # 估值因子：价格百分位越低（越接近60日低点），得分越高（低估值加分）
            # 以50%中位数为0点，范围 [-val_w/2, +val_w/2]
            scores[etf] = (mom_w * mom_norm - vol_w * vol_norm + val_w * (0.5 - val_norm))
        else:
            scores[etf] = 0

    return scores

def select_top_offensive(scores, top_n=2):
    if scores.empty:
        return []

    offensive_scores = scores[OFFENSIVE].dropna(axis=1, how="all")
    if offensive_scores.empty or offensive_scores.iloc[-1].max() == 0:
        return []

    top = offensive_scores.iloc[-1].nlargest(top_n).index.tolist()
    return top

def get_allocation(selected_offensive, defensive_allocation=0.55):
    allocation = {}

    def_weight = defensive_allocation / len(DEFENSIVE)
    for etf in DEFENSIVE:
        allocation[etf] = def_weight

    if selected_offensive:
        off_weight = (1 - defensive_allocation) / len(selected_offensive)
        for etf in selected_offensive:
            allocation[etf] = off_weight
    else:
        extra = (1 - defensive_allocation) / len(DEFENSIVE)
        for etf in DEFENSIVE:
            allocation[etf] += extra

    return allocation

def check_stop_loss(current_nav, peak_nav, threshold=0.08):
    if peak_nav == 0 or current_nav == 0:
        return False
    drawdown = (peak_nav - current_nav) / peak_nav
    return drawdown > threshold
