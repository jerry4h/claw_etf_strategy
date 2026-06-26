#!/usr/bin/env python3
import json
with open('/home/ubuntu/claw_etf_strategy/output/robustness_phase6/baseline_intermediate.json') as f:
    d = json.load(f)
fg = d.get('full_grid')
print(f'full_grid: {fg}')
print(f'type: {type(fg)}')
if fg:
    print(f'keys: {list(fg.keys())}')
    for k, v in fg.items():
        print(f'  {k}: {len(v)} points')
else:
    print('full_grid is None or empty')