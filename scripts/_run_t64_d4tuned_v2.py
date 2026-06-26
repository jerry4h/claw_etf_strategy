#!/usr/bin/env python3
"""Phase 6 Full-Grid for D4 tuned — sequential execution."""
import sys, os, json, numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.robustness import (
    build_full_grid, _grid_point_worker, evaluate_robustness,
    GridPointResult,
)

config = "config/strategy_v2_3_cap040_D4_tuned.yaml"
output_dir = "output/robustness_phase6"

print("=" * 70)
print("Phase 6 Full-Parameter Grid — D4 tuned (sequential workers)")
print(f"Config: {config}")
print(f"Output: {output_dir}")
print("=" * 70)

# First get the standard metrics
print("\n[Step 1] Running standard evaluation...")
result = evaluate_robustness(
    config_path=config,
    n_mc=400, n_wf_windows=9, n_trials=52,
    n_jobs=-1, perturbation=0.15,
    oat=False, pbo=False, sps=False,
    full_grid=False,
)

print("\n[Step 2] Building full grid configs...")
grid_configs = build_full_grid(config)
print(f"  Total grid points: {len(grid_configs)}")

# Run grid points sequentially
print("\n[Step 3] Running grid points sequentially...")
raw_results = []
n_total = len(grid_configs)
for i, gc in enumerate(grid_configs):
    gp_dict = {
        'param_name': gc.param_name,
        'level': gc.level,
        'actual_value': gc.actual_value,
        'param_overrides': gc.param_overrides,
    }
    if i % 10 == 0:
        print(f"  Grid point {i+1}/{n_total}...")
    r = _grid_point_worker((gp_dict, config, 50))
    if r is not None:
        raw_results.append(r)

print(f"  Completed: {len(raw_results)}/{n_total} grid points")

# Organize results
grid_results = {}
for param_name, gp_dict in raw_results:
    if param_name not in grid_results:
        grid_results[param_name] = []
    gpr = GridPointResult(
        param_name=gp_dict['param_name'],
        level=gp_dict['level'],
        actual_value=gp_dict['actual_value'],
        sharpe=gp_dict['sharpe'],
        annual_return=gp_dict['annual_return'],
        max_drawdown=gp_dict['max_drawdown'],
        relative_sharpe=gp_dict['relative_sharpe'],
        mc_survival_rate=gp_dict['mc_survival_rate'],
        mc_details=gp_dict.get('mc_details', []),
    )
    grid_results[param_name].append(gpr)

for pname in grid_results:
    grid_results[pname].sort(key=lambda x: x.level)

result.full_grid = grid_results

n_params = len(grid_results)
n_points = sum(len(v) for v in grid_results.values())
print(f"\nD4 tuned metrics:")
print(f"  Sharpe: {result.strategy_metrics['sharpe_ratio']:.4f}")
print(f"  Annual Return: {result.strategy_metrics['annual_return']*100:.2f}%")
print(f"  Max Drawdown: {result.strategy_metrics['max_drawdown']*100:.2f}%")
print(f"  DSR: {result.dsr:.4f}")
print(f"  MC Survival: {result.mc_survival_rate*100:.1f}%")
print(f"  WF Win Rate: {result.benchmark_relative_win_rate*100:.1f}%")
print(f"  Full Grid: {n_params} params, {n_points} grid points")

# Save
out = Path(output_dir)
out.mkdir(parents=True, exist_ok=True)

grid_serializable = {}
for pname, glist in grid_results.items():
    grid_serializable[pname] = []
    for gp in glist:
        grid_serializable[pname].append({
            'param_name': gp.param_name,
            'level': gp.level,
            'actual_value': gp.actual_value,
            'sharpe': gp.sharpe if not np.isnan(gp.sharpe) else None,
            'annual_return': gp.annual_return if not np.isnan(gp.annual_return) else None,
            'max_drawdown': gp.max_drawdown if not np.isnan(gp.max_drawdown) else None,
            'relative_sharpe': gp.relative_sharpe if not np.isnan(gp.relative_sharpe) else None,
            'mc_survival_rate': gp.mc_survival_rate,
        })

entry = {
    'label': 'D4 tuned',
    'config': config,
    'dsr': result.dsr,
    'mc_survival_rate': result.mc_survival_rate,
    'benchmark_relative_win_rate': result.benchmark_relative_win_rate,
    'full_grid': grid_serializable,
    'strategy_metrics': {k: float(v) if isinstance(v, (np.floating,)) else v for k, v in result.strategy_metrics.items()},
}

with open(out / 'd4tuned_intermediate.json', 'w') as f:
    json.dump(entry, f, indent=2, default=str, ensure_ascii=False)

print(f"\nIntermediate result saved to {out / 'd4tuned_intermediate.json'}")
print("D4 tuned Phase 6 grid complete!")