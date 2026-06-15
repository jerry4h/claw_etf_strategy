import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from backtest import run_backtest
import backtest
import pandas as pd

def calc_metrics(result_df):
    """从DataFrame计算回测指标"""
    if result_df.empty:
        return None
    final_val = result_df.iloc[-1]["portfolio_value"]
    total_return = final_val - 1
    annual_return = (1 + total_return) ** (52 / len(result_df)) - 1
    max_dd = ((result_df["peak_value"] - result_df["portfolio_value"]) / result_df["peak_value"]).max()
    defensive_days = result_df["in_defensive"].sum()
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": -max_dd,
        "defensive_weeks": int(defensive_days),
        "total_weeks": len(result_df)
    }

# 当前最优参数（top_n=2, Calmar最优）
params_base = dict(mom_w=0.15, vol_w=0.2, val_w=0.0, top_n=2, defensive_allocation=0.40)

# 高收益备选参数
params_alt = dict(mom_w=0.30, vol_w=0.4, val_w=0.0, top_n=2, defensive_allocation=0.30)

results = []

# 方案1: 当前（有国债）
print("=" * 60)
print("方案1: 当前配置（有国债）")
print("=" * 60)
r = calc_metrics(run_backtest(**params_base))
print(f"  年化: {r['annual_return']*100:.2f}%  回撤: {r['max_drawdown']*100:.2f}%")
results.append({"方案": "当前(有国债)", "params": "mom=0.15,vol=0.2,def=0.40", **r})

r2 = calc_metrics(run_backtest(**params_alt))
print(f"  备选: 年化: {r2['annual_return']*100:.2f}%  回撤: {r2['max_drawdown']*100:.2f}%")
results.append({"方案": "备选(有国债)", "params": "mom=0.30,vol=0.4,def=0.30", **r2})

original_defensive = backtest.DEFENSIVE_POOL.copy()

try:
    # 方案2: 去掉国债，def_alloc不变=0.40
    backtest.DEFENSIVE_POOL = ["红利低波ETF"]
    print("\n" + "=" * 60)
    print("方案2: 去掉国债，def_alloc=0.40（红利低波独占40%）")
    print("=" * 60)
    r = calc_metrics(run_backtest(**params_base))
    print(f"  年化: {r['annual_return']*100:.2f}%  回撤: {r['max_drawdown']*100:.2f}%")
    results.append({"方案": "去国债_def40", "params": "mom=0.15,vol=0.2,def=0.40", **r})

    r2 = calc_metrics(run_backtest(**params_alt))
    print(f"  备选: 年化: {r2['annual_return']*100:.2f}%  回撤: {r2['max_drawdown']*100:.2f}%")
    results.append({"方案": "备选去国债_def40", "params": "mom=0.30,vol=0.4,def=0.30", **r2})

    # 方案3: 去掉国债，def_alloc=0.30
    print("\n" + "=" * 60)
    print("方案3: 去掉国债，def_alloc=0.30（红利低波独占30%）")
    print("=" * 60)
    p3 = dict(params_base, defensive_allocation=0.30)
    r = calc_metrics(run_backtest(**p3))
    print(f"  年化: {r['annual_return']*100:.2f}%  回撤: {r['max_drawdown']*100:.2f}%")
    results.append({"方案": "去国债_def30", "params": "mom=0.15,vol=0.2,def=0.30", **r})

    p3a = dict(params_alt, defensive_allocation=0.30)
    r2 = calc_metrics(run_backtest(**p3a))
    print(f"  备选: 年化: {r2['annual_return']*100:.2f}%  回撤: {r2['max_drawdown']*100:.2f}%")
    results.append({"方案": "备选去国债_def30", "params": "mom=0.30,vol=0.4,def=0.30", **r2})

    # 方案4: 去掉国债，def_alloc=0.20
    print("\n" + "=" * 60)
    print("方案4: 去掉国债，def_alloc=0.20（红利低波独占20%）")
    print("=" * 60)
    p4 = dict(params_base, defensive_allocation=0.20)
    r = calc_metrics(run_backtest(**p4))
    print(f"  年化: {r['annual_return']*100:.2f}%  回撤: {r['max_drawdown']*100:.2f}%")
    results.append({"方案": "去国债_def20", "params": "mom=0.15,vol=0.2,def=0.20", **r})

    p4a = dict(params_alt, defensive_allocation=0.20)
    r2 = calc_metrics(run_backtest(**p4a))
    print(f"  备选: 年化: {r2['annual_return']*100:.2f}%  回撤: {r2['max_drawdown']*100:.2f}%")
    results.append({"方案": "备选去国债_def20", "params": "mom=0.30,vol=0.4,def=0.20", **r2})

finally:
    backtest.DEFENSIVE_POOL = original_defensive

# 汇总
print("\n" + "=" * 60)
print("汇总对比")
print("=" * 60)
df = pd.DataFrame(results)
df['Calmar'] = df['annual_return'] / (-df['max_drawdown'])
df['annual_pct'] = (df['annual_return'] * 100).round(2).astype(str) + '%'
df['dd_pct'] = (df['max_drawdown'] * 100).round(2).astype(str) + '%'
df['total_pct'] = (df['total_return'] * 100).round(2).astype(str) + '%'
print(df[['方案', 'params', 'annual_pct', 'dd_pct', 'Calmar', 'total_pct']].to_string(index=False))
