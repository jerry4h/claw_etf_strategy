import sys
sys.path.insert(0, '.')
from src.strategy import load_config
from src.backtest import run_backtest

cfg = load_config('config/strategy_v2_3.yaml')
cfg.rebalance_threshold = 0.0  # disable threshold
result = run_backtest(cfg)
m = result.metrics
print(f'无阈值模式: 年化={m["annual_return"]*100:.2f}% 回撤={m["max_drawdown"]*100:.2f}% 夏普={m["sharpe_ratio"]:.3f}')
