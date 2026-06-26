#!/usr/bin/env python3
"""Quick config load verification."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.strategy import load_config

for path in [
    'config/strategy_v2_3_cap040_D4_tuned.yaml',
    'config/strategy_v2_3_cap040_D4_tuned_constituent.yaml',
    'config/strategy_v2_3_cap040.yaml',
]:
    c = load_config(path)
    print(f'{path}: name={c.name}, constituent_enabled={c.constituent_signals_enabled}, cwm_weight={c.cwm_weight}, conc_weight={c.conc_weight}')
print('OK: all configs load!')
