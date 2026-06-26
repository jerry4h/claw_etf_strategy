#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.strategy import load_config

config = "config/strategy_v2_3_cap040.yaml"
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_abs = os.path.join(project_root, config)
cfg = load_config(config_abs)

print(f"d4_enabled: {cfg.d4_enabled}")
print(f"mom_w: {cfg.mom_w}")
print(f"vol_w: {cfg.vol_w}")
print(f"def_alloc: {cfg.def_alloc}")
print(f"stop_loss: {cfg.stop_loss}")
print(f"top_n: {cfg.top_n}")
print(f"d4_momentum_window: {cfg.d4_momentum_window}")
print(f"d4_momentum_threshold: {cfg.d4_momentum_threshold}")