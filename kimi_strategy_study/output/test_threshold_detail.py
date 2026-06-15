import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from backtest import run_backtest

ETFS = ["纳指ETF", "红利低波ETF", "沪深300ETF", "黄金ETF", "国债ETF"]
OFFENSIVE = ["纳指ETF", "沪深300ETF", "黄金ETF"]

params = dict(mom_w=0.15, vol_w=0.2, val_w=0.0, top_n=2, defensive_allocation=0.40)

for threshold in [0.0, 0.02, 0.05, 0.10]:
    r = run_backtest(min_trade_threshold=threshold, stop_loss_offensive_n=2, **params)

    # 统计详细调仓数据
    trade_count = 0  # 有多少周发生了调仓
    total_trades = 0  # 总调仓标的人次
    trade_sizes = []  # 每次调仓的幅度
    offensive_switches = 0  # 进攻层标的切换次数（某个标从0变>0或从>0变0）

    for i in range(1, len(r)):
        prev = r.iloc[i-1]
        curr = r.iloc[i]

        week_trades = 0
        week_trade_size = 0
        for etf in ETFS:
            diff = abs(curr[f"weight_{etf}"] - prev[f"weight_{etf}"])
            if diff > 0.0001:  # 有实际变化
                week_trades += 1
                week_trade_size += diff

        if week_trades > 0:
            trade_count += 1
            total_trades += week_trades
            trade_sizes.append(week_trade_size / 2)  # /2因为买入+卖出

        # 进攻层切换（乒乓效应）
        for etf in OFFENSIVE:
            pw = prev[f"weight_{etf}"]
            cw = curr[f"weight_{etf}"]
            if (pw == 0 and cw > 0) or (pw > 0 and cw == 0):
                offensive_switches += 1

    avg_trade_size = np.mean(trade_sizes) if trade_sizes else 0

    # 手续费
    turnovers = []
    for i in range(1, len(r)):
        prev = r.iloc[i-1]
        curr = r.iloc[i]
        change = sum(abs(curr[f"weight_{e}"] - prev[f"weight_{e}"]) for e in ETFS)
        turnovers.append(change / 2)
    total_fee = sum(turnovers) * 2 * 0.00005

    final_val = r.iloc[-1]["portfolio_value"]
    annual = (final_val) ** (52 / len(r)) - 1
    max_dd = ((r["peak_value"] - r["portfolio_value"]) / r["peak_value"]).max()

    print(f"\n阈值={threshold:.0%}:")
    print(f"  发生调仓的周数: {trade_count}/{len(r)} ({trade_count/len(r)*100:.1f}%)")
    print(f"  总调仓人次: {total_trades}")
    print(f"  平均每次调仓幅度: {avg_trade_size*100:.2f}%")
    print(f"  进攻层标的切换次数: {offensive_switches}")
    print(f"  总手续费损耗: {total_fee*100:.3f}%")
    print(f"  年化: {annual*100:.2f}%  回撤: {-max_dd*100:.2f}%")
