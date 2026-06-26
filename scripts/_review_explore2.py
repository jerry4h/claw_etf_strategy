#!/usr/bin/env python3
"""Review script part 2: D1 tuning + D4+D1 combos + best-candidate search."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategy import load_config
from src.backtest import run_backtest
import yaml

with open('config/strategy_v2_3_cap040.yaml') as f:
    base = yaml.safe_load(f)

print("="*80)
print("3. D1 PARAMETER EXPLORATION")
print("="*80)

for tq_low, tq_high in [(0.0, 2.0), (0.0, 1.0), (-0.5, 1.5), (-1.0, 2.0), (0.0, 0.5)]:
    for mom_low, mom_high in [(0.25, 0.45), (0.30, 0.40), (0.20, 0.50)]:
        vol_low, vol_high = 0.20, 0.40
        base['d4_individual_filter'] = base.get('d4_individual_filter', {})
        base['d4_individual_filter']['enabled'] = False
        base['dynamic_weighting'] = {
            'enabled': True, 'lookback': 12,
            'tq_low': tq_low, 'tq_high': tq_high,
            'mom_w_low': mom_low, 'mom_w_high': mom_high,
            'vol_w_low': vol_low, 'vol_w_high': vol_high,
            'weight_sum': 0.65,
        }
        with open('/tmp/_rt.yaml', 'w') as f2:
            yaml.dump(base, f2)
        cfg = load_config('/tmp/_rt.yaml')
        result = run_backtest(cfg)
        m = result.metrics
        print(f"D1 tq=[{tq_low:+.1f},{tq_high:+.1f}] mw=[{mom_low:.2f},{mom_high:.2f}]: "
              f"ret={m['annual_return']:.4f} dd={m['max_drawdown']:.4f} "
              f"sharpe={m['sharpe_ratio']:.3f} calmar={m['calmar_ratio']:.3f}")

print()
print("="*80)
print("4. D4+D1 COMBINATIONS")
print("="*80)

best_params = [
    (-0.10, 8, 'replace', 0.0, 2.0, 0.25, 0.45, 0.20, 0.40),
    (-0.10, 6, 'replace', 0.0, 1.0, 0.25, 0.45, 0.20, 0.40),
    (-0.05, 8, 'defense', 0.0, 2.0, 0.25, 0.45, 0.20, 0.40),
    (-0.10, 8, 'replace', 0.0, 1.0, 0.30, 0.40, 0.20, 0.40),
]
for params in best_params:
    d4_th, d4_w, d4_act, tq_low, tq_high, mom_low, mom_high, vol_low, vol_high = params
    base['d4_individual_filter'] = {
        'enabled': True, 'momentum_window': d4_w,
        'momentum_threshold': d4_th, 'action': d4_act, 'min_candidates': 3,
    }
    base['dynamic_weighting'] = {
        'enabled': True, 'lookback': 12,
        'tq_low': tq_low, 'tq_high': tq_high,
        'mom_w_low': mom_low, 'mom_w_high': mom_high,
        'vol_w_low': vol_low, 'vol_w_high': vol_high, 'weight_sum': 0.65,
    }
    with open('/tmp/_rt.yaml', 'w') as f2:
        yaml.dump(base, f2)
    cfg = load_config('/tmp/_rt.yaml')
    result = run_backtest(cfg)
    m = result.metrics
    label = f"D4(th={d4_th:+.2f},w={d4_w},{d4_act})+D1(tq[{tq_low}, {tq_high}],mw[{mom_low},{mom_high}])"
    print(f"{label}: ret={m['annual_return']:.4f} dd={m['max_drawdown']:.4f} "
          f"sharpe={m['sharpe_ratio']:.3f} calmar={m['calmar_ratio']:.3f} def_wk={m['defensive_weeks']}")
