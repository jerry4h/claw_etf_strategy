#!/usr/bin/env python3
"""Quick temperature sweep for D5 evaluation."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.strategy import load_config
from src.backtest import run_backtest

print(f"{'Temp':>6}  {'Return':>8}  {'MaxDD':>7}  {'Sharpe':>7}")
print("-" * 35)

for temp in [0.2, 0.1, 0.05, 0.02, 0.01]:
    config = load_config(PROJECT_ROOT / 'config' / 'strategy_v2_3_cap040_D4_tuned_D5.yaml')
    config.softmax_temperature = temp
    result = run_backtest(config)
    m = result.metrics
    print(f"{temp:>5.2f}  {m['annual_return']*100:>7.2f}%  {m['max_drawdown']*100:>6.2f}%  {m['sharpe_ratio']:>7.3f}")