#!/usr/bin/env python3
"""Debug single grid point."""
import sys, os, multiprocessing, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.robustness import build_full_grid, _grid_point_worker

config = "config/strategy_v2_3_cap040.yaml"
grid_configs = build_full_grid(config)

# Try first grid point
gc = grid_configs[0]
gp_dict = {
    'param_name': gc.param_name,
    'level': gc.level,
    'actual_value': gc.actual_value,
    'param_overrides': gc.param_overrides,
}

print(f"Testing first grid point: {gp_dict}")
result = _grid_point_worker((gp_dict, config, 50))
print(f"Result: {result}")