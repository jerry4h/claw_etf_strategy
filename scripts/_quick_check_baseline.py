#!/usr/bin/env python3
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.robustness import GRID_CONTINUOUS_PARAMS, GRID_DISCRETE_PARAMS, PERTURBATION_LEVELS

with open("output/robustness_phase6/baseline_intermediate.json") as f:
    d = json.load(f)

print("=== Baseline Grid Summary ===")
print(f"DSR: {d['dsr']:.4f}")
print(f"MC Survival: {d['mc_survival_rate']*100:.1f}%")
print(f"WF Win Rate: {d['benchmark_relative_win_rate']*100:.1f}%")
print()

grid = d['full_grid']
print(f"Params: {len(grid)}")
for pname, glist in grid.items():
    vals = []
    for g in glist:
        s = f"{g['sharpe']:.3f}" if g['sharpe'] is not None else 'N/A'
        r = f"{g['annual_return']*100:.2f}%" if g['annual_return'] is not None else 'N/A'
        dd = f"{g['max_drawdown']*100:.2f}%" if g['max_drawdown'] is not None else 'N/A'
        mc = f"{g['mc_survival_rate']*100:.0f}%"
        vals.append(f"lvl={g['level']}:S={s},R={r},DD={dd},MC={mc}")
    
    avg_sharpe = np.mean([g['sharpe'] for g in glist if g['sharpe'] is not None])
    avg_mc = np.mean([g['mc_survival_rate'] for g in glist])
    min_mc = min([g['mc_survival_rate'] for g in glist])
    max_mc = max([g['mc_survival_rate'] for g in glist])
    print(f"  {pname}: {len(glist)} pts, avg_S={avg_sharpe:.3f}, avg_MC={avg_mc*100:.1f}% [{min_mc*100:.0f}%-{max_mc*100:.0f}%]")