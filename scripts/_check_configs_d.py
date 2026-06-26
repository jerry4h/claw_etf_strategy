#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.strategy import load_config

for name in ['strategy_v2_3_cap040_D4_only', 'strategy_v2_3_cap040_D1_only', 'strategy_v2_3_cap040_D4_D1']:
    c = load_config(f'config/{name}.yaml')
    print(f'{name}: D4_enabled={c.d4_enabled}, D1_enabled={c.d1_enabled}')
