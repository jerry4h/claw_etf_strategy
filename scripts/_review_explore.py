#!/usr/bin/env python3
"""Review script: reproduce T23 results + parameter exploration."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategy import load_config
from src.backtest import run_backtest

import numpy as np

def test_config(name, check=None):
    """Run backtest and return key metrics."""
    cfg = load_config(f'config/{name}.yaml')
    if check:
        for k, v in check.items():
            assert getattr(cfg, k, None) == v, f"{name}: {k} expected {v}, got {getattr(cfg, k)}"
    result = run_backtest(cfg)
    m = result.metrics
    return {
        'name': name,
        'annual_return': m['annual_return'],
        'max_drawdown': m['max_drawdown'],
        'sharpe_ratio': m['sharpe_ratio'],
        'calmar_ratio': m['calmar_ratio'],
        'win_rate': m['win_rate'],
        'defensive_weeks': m['defensive_weeks'],
        'rebalance_count': m['rebalance_count'],
        'annual_volatility': m['annual_volatility'],
    }

# Reproduce T23 results
print("="*80)
print("1. REPRODUCING T23 HANDOFF RESULTS")
print("="*80)

results = {}
for name in ['strategy_v2_3_cap040', 'strategy_v2_3_cap040_D4_only',
             'strategy_v2_3_cap040_D1_only', 'strategy_v2_3_cap040_D4_D1']:
    r = test_config(name)
    results[name] = r
    print(f"{r['name']}: ret={r['annual_return']:.4f} dd={r['max_drawdown']:.4f} "
          f"sharpe={r['sharpe_ratio']:.3f} vol={r['annual_volatility']:.4f} "
          f"def_wk={r['defensive_weeks']}")

print()
print("="*80)
print("2. D4 PARAMETER EXPLORATION")
print("="*80)

# D4: what happens with different thresholds?
from src.strategy import StrategyConfig
import yaml

with open('config/strategy_v2_3_cap040.yaml') as f:
    base = yaml.safe_load(f)

for threshold in [-0.05, -0.10, -0.15, -0.03, 0.02, 0.05]:
    for window in [4, 6, 8, 12]:
        for action in ['replace', 'defense']:
            base['d4_individual_filter'] = {
                'enabled': True,
                'momentum_window': window,
                'momentum_threshold': threshold,
                'action': action,
                'min_candidates': 3,
            }
            base['dynamic_weighting'] = base.get('dynamic_weighting', {})
            base['dynamic_weighting']['enabled'] = False
            
            tmp_path = '/tmp/_review_test.yaml'
            with open(tmp_path, 'w') as f2:
                yaml.dump(base, f2)
            
            cfg = load_config(tmp_path)
            result = run_backtest(cfg)
            m = result.metrics
            
            print(f"D4 th={threshold:+5.2f} w={window} act={action:7s}: "
                  f"ret={m['annual_return']:.4f} dd={m['max_drawdown']:.4f} "
                  f"sharpe={m['sharpe_ratio']:.3f} calmar={m['calmar_ratio']:.3f} "
                  f"def_wk={m['defensive_weeks']}")
