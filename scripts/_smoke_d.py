#!/usr/bin/env python3
"""Smoke test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategy import StrategyConfig, load_config, apply_individual_momentum_filter, compute_dynamic_weights
print('strategy.py imports: OK')

c = load_config('config/strategy_v2_3_cap040.yaml')
print(f'Config: {c.name} v{c.version}')
print(f'D4 enabled={c.d4_enabled}, window={c.d4_momentum_window}, threshold={c.d4_momentum_threshold}')
print(f'D1 enabled={c.d1_enabled}, lookback={c.d1_lookback}')

from src.backtest import run_backtest
print('backtest.py imports: OK')
print('All imports passed!')
