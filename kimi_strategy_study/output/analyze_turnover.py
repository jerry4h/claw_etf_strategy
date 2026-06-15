import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from backtest import run_backtest
import pandas as pd
import numpy as np

# 当前最优参数
params = dict(mom_w=0.15, vol_w=0.2, val_w=0.0, top_n=2, defensive_allocation=0.40)

result = run_backtest(**params)

# 计算进攻层的换手
offensive_etfs = ["纳指ETF", "沪深300ETF", "黄金ETF"]

# 提取进攻层权重
for etf in offensive_etfs:
    result[f"weight_{etf}"] = result[f"weight_{etf}"].fillna(0)

# 计算进攻层总权重
result["off_total"] = result[[f"weight_{e}" for e in offensive_etfs]].sum(axis=1)

# 计算进攻层内部换手（相邻周权重变化的绝对值之和 / 2）
turnovers = []
for i in range(1, len(result)):
    prev = result.iloc[i-1]
    curr = result.iloc[i]
    change = sum(abs(curr[f"weight_{e}"] - prev[f"weight_{e}"]) for e in offensive_etfs)
    # 进攻层内部换手 = 变化量 / 2（买入+卖出各算一次）
    turnovers.append(change / 2)

avg_turnover = np.mean(turnovers)
annual_turnover = avg_turnover * 52

print("=" * 60)
print("当前策略（3选2）进攻层换手分析")
print("=" * 60)
print(f"平均每周进攻层内部换手: {avg_turnover*100:.2f}%")
print(f"年化进攻层内部换手: {annual_turnover*100:.0f}%")
print(f"总调仓次数: {len(result)}周")

# 统计进攻层切换事件（某标的从0变>0或从>0变0）
switches = 0
for etf in offensive_etfs:
    w = result[f"weight_{etf}"].values
    for i in range(1, len(w)):
        if (w[i-1] == 0 and w[i] > 0) or (w[i-1] > 0 and w[i] == 0):
            switches += 1

print(f"进攻标的出现/消失次数（乒乓切换）: {switches}次")
print(f"平均每年切换: {switches / (len(result)/52):.1f}次")

# 统计进攻层各标的持仓周数
print("\n进攻层各标的持仓周数:")
for etf in offensive_etfs:
    weeks = (result[f"weight_{etf}"] > 0).sum()
    print(f"  {etf}: {weeks}周 ({weeks/len(result)*100:.1f}%)")

# 展示典型乒乓案例
print("\n前10周进攻层权重变化（展示乒乓效应）:")
for i in range(min(10, len(result))):
    r = result.iloc[i]
    weights = {e: r[f"weight_{e}"] for e in offensive_etfs}
    active = [e for e, w in weights.items() if w > 0]
    print(f"  {r['date'].strftime('%Y-%m-%d')}: {active} | " + " ".join([f"{e}={weights[e]*100:.1f}%" for e in offensive_etfs]))
