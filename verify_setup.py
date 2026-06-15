import sys
sys.path.insert(0, '/home/ubuntu/claw_eft_strategy/kimi_strategy_study')

print('=== Verifying data loading ===')
from data_loader import load_nav_data, ETFS
df = load_nav_data()
print('Data shape:', df.shape)
print('Date range:', df.index.min(), 'to', df.index.max())
print('Columns:', list(df.columns))
print('First 3 rows:')
print(df.head(3))
print()

print('=== Verifying factor calculation ===')
from data_loader import calculate_returns
from factors import calculate_all_factors
ret_df = calculate_returns(df)
factors = calculate_all_factors(df, ret_df, mom_window=4, vol_window=20, val_window=60)
print('Momentum shape:', factors['momentum'].shape)
print('Volatility shape:', factors['volatility'].shape)
print('Valuation shape:', factors['valuation'].shape)
print()

print('=== Verifying backtest ===')
from backtest import run_backtest
result = run_backtest(
    mom_w=0.35, vol_w=0.30, val_w=0.0,
    top_n=2, defensive_allocation=0.25,
    stop_loss_threshold=0.08,
    recovery_weeks=4
)
print('Backtest result length:', len(result))
if not result.empty:
    final_val = result.iloc[-1]['portfolio_value']
    total_ret = final_val - 1
    n_weeks = len(result)
    annual_ret = (1 + total_ret) ** (52 / n_weeks) - 1
    dd = ((result['peak_value'] - result['portfolio_value']) / result['peak_value']).max()
    sharpe = result['weekly_return'].mean() / result['weekly_return'].std() * (52 ** 0.5)
    print(f'Final NAV: {final_val:.4f}')
    print(f'Total return: {total_ret:.2%}')
    print(f'Annual return: {annual_ret:.2%}')
    print(f'Max drawdown: {dd:.2%}')
    print(f'Sharpe (simple): {sharpe:.3f}')
print()

print('=== Verification complete ===')
