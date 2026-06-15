import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from backtest import run_backtest

FEE_RATE = 0.00005
ETFS = ["纳指ETF", "红利低波ETF", "沪深300ETF", "黄金ETF", "国债ETF"]


def calc_metrics(result_df):
    if result_df.empty:
        return None
    final_val = result_df.iloc[-1]["portfolio_value"]
    total_return = final_val - 1
    annual_return = (1 + total_return) ** (52 / len(result_df)) - 1
    max_dd = ((result_df["peak_value"] - result_df["portfolio_value"]) / result_df["peak_value"]).max()
    defensive_days = result_df["in_defensive"].sum()

    turnovers = []
    for i in range(1, len(result_df)):
        prev = result_df.iloc[i-1]
        curr = result_df.iloc[i]
        change = sum(abs(curr[f"weight_{e}"] - prev[f"weight_{e}"]) for e in ETFS)
        turnovers.append(change / 2)
    avg_turnover = np.mean(turnovers)
    annual_turnover = avg_turnover * 52
    total_fee = sum(turnovers) * 2 * FEE_RATE

    return {
        "annual_return": annual_return,
        "max_drawdown": -max_dd,
        "total_return": total_return,
        "avg_weekly_turnover": avg_turnover,
        "annual_turnover": annual_turnover,
        "total_fee_impact": total_fee,
        "defensive_weeks": int(defensive_days),
    }


# 测试参数
params_list = [
    dict(mom_w=0.15, vol_w=0.2, val_w=0.0, top_n=2, defensive_allocation=0.40),
    dict(mom_w=0.30, vol_w=0.4, val_w=0.0, top_n=2, defensive_allocation=0.30),
]

threshold_values = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]

print("=" * 95)
print("调仓阈值测试（基于原始 backtest.py）")
print("=" * 95)
print("阈值含义: 只有当某标的目标权重与当前权重的差异 >= 阈值时，才执行该标的的调仓")
print()

results = []

for p in params_list:
    label = f"mom={p['mom_w']},vol={p['vol_w']},def={p['defensive_allocation']}"

    for threshold in threshold_values:
        r = run_backtest(min_trade_threshold=threshold, stop_loss_offensive_n=2, **p)
        m = calc_metrics(r)
        m["方案"] = f"阈值={threshold:.0%}" if threshold > 0 else "无阈值(当前)"
        m["参数"] = label
        m["threshold"] = threshold
        results.append(m)
        calmar = m['annual_return'] / (-m['max_drawdown'])
        print(f"{label} | 阈值={threshold:.0%}: 年化={m['annual_return']*100:.2f}% 回撤={m['max_drawdown']*100:.2f}% Calmar={calmar:.3f} 换手={m['annual_turnover']*100:.0f}% 手续费={m['total_fee_impact']*100:.3f}%")

# 汇总表
print("\n" + "=" * 95)
print("汇总对比")
print("=" * 95)
df = pd.DataFrame(results)
df['Calmar'] = df['annual_return'] / (-df['max_drawdown'])
df['annual_pct'] = (df['annual_return']*100).round(2).astype(str)+'%'
df['dd_pct'] = (df['max_drawdown']*100).round(2).astype(str)+'%'
df['turnover_pct'] = (df['annual_turnover']*100).round(0).astype(str)+'%'
df['fee_pct'] = (df['total_fee_impact']*100).round(3).astype(str)+'%'
print(df[['方案', '参数', 'annual_pct', 'dd_pct', 'Calmar', 'turnover_pct', 'fee_pct']].to_string(index=False))
