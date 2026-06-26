#!/usr/bin/env python3
"""Phase 6 Full-Grid for D4 tuned"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.robustness import evaluate_robustness

config = "config/strategy_v2_3_cap040_D4_tuned.yaml"
output_dir = "output/robustness_phase6"

print("=" * 70)
print("Phase 6 Full-Parameter Grid — D4 tuned")
print(f"Config: {config}")
print(f"Output: {output_dir}")
print("=" * 70)

result = evaluate_robustness(
    config_path=config,
    n_mc=400,
    n_wf_windows=9,
    n_trials=52,
    n_jobs=-1,
    perturbation=0.15,
    oat=False,
    pbo=False,
    sps=False,
    full_grid=True,
    n_local_mc=50,
)

# Print summary
n_params = len(result.full_grid) if result.full_grid else 0
n_points = sum(len(v) for v in (result.full_grid or {}).values())
print(f"\nD4 tuned metrics:")
print(f"  Sharpe: {result.strategy_metrics['sharpe_ratio']:.4f}")
print(f"  Annual Return: {result.strategy_metrics['annual_return']*100:.2f}%")
print(f"  Max Drawdown: {result.strategy_metrics['max_drawdown']*100:.2f}%")
print(f"  DSR: {result.dsr:.4f}")
print(f"  MC Survival: {result.mc_survival_rate*100:.1f}%")
print(f"  WF Win Rate: {result.benchmark_relative_win_rate*100:.1f}%")
print(f"  Full Grid: {n_params} params, {n_points} grid points")

# Save intermediate result
import json
from pathlib import Path

out = Path(output_dir)
out.mkdir(parents=True, exist_ok=True)

grid_serializable = {}
if result.full_grid:
    for pname, glist in result.full_grid.items():
        grid_serializable[pname] = []
        for gp in glist:
            grid_serializable[pname].append({
                'param_name': gp.param_name,
                'level': gp.level,
                'actual_value': gp.actual_value,
                'sharpe': gp.sharpe if not (hasattr(gp, 'sharpe') and gp.sharpe != gp.sharpe) else None,
                'annual_return': gp.annual_return,
                'max_drawdown': gp.max_drawdown,
                'relative_sharpe': gp.relative_sharpe,
                'mc_survival_rate': gp.mc_survival_rate,
            })

entry = {
    'label': 'D4 tuned',
    'config': config,
    'dsr': result.dsr,
    'mc_survival_rate': result.mc_survival_rate,
    'benchmark_relative_win_rate': result.benchmark_relative_win_rate,
    'full_grid': grid_serializable,
    'strategy_metrics': result.strategy_metrics,
}

with open(out / 'd4tuned_intermediate.json', 'w') as f:
    json.dump(entry, f, indent=2, default=str, ensure_ascii=False)

print(f"\nIntermediate result saved to {out / 'd4tuned_intermediate.json'}")
print("D4 tuned Phase 6 grid complete!")