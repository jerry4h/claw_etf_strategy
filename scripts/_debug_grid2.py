#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.robustness import build_full_grid, run_full_grid

config = "config/strategy_v2_3_cap040.yaml"

print("=== build_full_grid ===")
grid_configs = build_full_grid(config)
print(f"Number of grid configs: {len(grid_configs)}")
if grid_configs:
    print(f"First: {grid_configs[0]}")
    print(f"Last: {grid_configs[-1]}")
    # Check param names
    param_names = set(gc.param_name for gc in grid_configs)
    print(f"Params: {sorted(param_names)}")

print("\n=== run_full_grid ===")
result = run_full_grid(config, n_local_mc=50, n_jobs=-1)
print(f"Result keys: {result.keys()}")
for k, v in result.items():
    print(f"  {k}: {len(v)} points")
    if v:
        print(f"    First: sharpe={v[0].sharpe}, ret={v[0].annual_return}, dd={v[0].max_drawdown}, mc={v[0].mc_survival_rate}")