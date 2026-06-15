from backtest import run_backtest
import pandas as pd

result = run_backtest(
    mom_w=0.40, vol_w=0.60, val_w=0.0,
    top_n=2, defensive_allocation=0.30,
    stop_loss_threshold=0.08
)

print('回测结果形状:', result.shape)
print('回测日期范围:', result['date'].min(), 'to', result['date'].max())

total_ret = result.iloc[-1]['portfolio_value'] - 1
print('总收益:', f"{total_ret:.2%}")

max_dd = ((result['peak_value'] - result['portfolio_value']) / result['peak_value']).max()
print('最大回撤:', f"{max_dd:.2%}")

annual_ret = (1 + total_ret) ** (52 / len(result)) - 1
print('年化收益:', f"{annual_ret:.2%}")

print('周数:', len(result))
print()
print('前5行:')
print(result[['date', 'portfolio_value', 'weekly_return', 'in_defensive']].head())
