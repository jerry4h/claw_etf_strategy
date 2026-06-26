#!/usr/bin/env python3 -u
"""Phase 6 Full-Grid using concurrent.futures.ProcessPoolExecutor (avoids pickle issues)."""
import sys, os, json, numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.robustness import (
    build_full_grid, _grid_point_worker, evaluate_robustness,
    GridPointResult,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def run_grid(config_path: str, label: str, output_dir: str):
    print("=" * 70)
    print(f"Phase 6 Full-Parameter Grid — {label}")
    print(f"Config: {config_path}")
    print("=" * 70)

    # Step 1: Standard evaluation
    print("\n[Step 1] Running standard evaluation...")
    result = evaluate_robustness(
        config_path=config_path,
        n_mc=400, n_wf_windows=9, n_trials=52,
        n_jobs=-1, perturbation=0.15,
        oat=False, pbo=False, sps=False,
        full_grid=False,
    )

    # Step 2: Build grid
    print("\n[Step 2] Building full grid configs...")
    grid_configs = build_full_grid(config_path)
    print(f"  Total grid points: {len(grid_configs)}")

    # Step 3: Run with ProcessPoolExecutor
    print("\n[Step 3] Running grid points in parallel...")
    worker_args = []
    for gc in grid_configs:
        gp_dict = {
            'param_name': gc.param_name,
            'level': gc.level,
            'actual_value': gc.actual_value,
            'param_overrides': gc.param_overrides,
        }
        worker_args.append((gp_dict, config_path, 50))

    raw_results = []
    n_cpu = multiprocessing.cpu_count()
    print(f"  Using {n_cpu} workers")
    sys.stdout.flush()

    with ProcessPoolExecutor(max_workers=n_cpu) as executor:
        futures = {executor.submit(_grid_point_worker, args): i 
                   for i, args in enumerate(worker_args)}
        done = 0
        total = len(futures)
        for f in as_completed(futures):
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  Grid point {done}/{total}...")
                sys.stdout.flush()
            try:
                r = f.result()
                if r is not None:
                    raw_results.append(r)
            except Exception as e:
                idx = futures[f]
                print(f"  [WARN] Grid point {idx} failed: {e}")

    print(f"  Completed: {len(raw_results)}/{total} grid points")
    sys.stdout.flush()

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
    print(f"\n{label} metrics:")
    print(f"  Sharpe: {result.strategy_metrics['sharpe_ratio']:.4f}")
    print(f"  Annual Return: {result.strategy_metrics['annual_return']*100:.2f}%")
    print(f"  Max Drawdown: {result.strategy_metrics['max_drawdown']*100:.2f}%")
    print(f"  DSR: {result.dsr:.4f}")
    print(f"  MC Survival: {result.mc_survival_rate*100:.1f}%")
    print(f"  WF Win Rate: {result.benchmark_relative_win_rate*100:.1f}%")
    print(f"  Full Grid: {n_params} params, {n_points} grid points")
    sys.stdout.flush()

    # Save intermediate
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

    tag = 'baseline' if '基线' in label or 'baseline' in label.lower() else 'd4tuned'
    entry = {
        'label': label,
        'config': config_path,
        'dsr': result.dsr,
        'mc_survival_rate': result.mc_survival_rate,
        'benchmark_relative_win_rate': result.benchmark_relative_win_rate,
        'full_grid': grid_serializable,
        'strategy_metrics': {k: float(v) if isinstance(v, (np.floating,)) else v 
                             for k, v in result.strategy_metrics.items()},
    }

    fname = f'{tag}_intermediate.json'
    with open(out / fname, 'w') as f:
        json.dump(entry, f, indent=2, default=str, ensure_ascii=False)

    print(f"\nSaved to {out / fname}")
    print(f"{label} Phase 6 grid complete!")


if __name__ == '__main__':
    # Determine which config from CLI arg
    mode = sys.argv[1] if len(sys.argv) > 1 else 'both'
    
    if mode in ('baseline', 'both'):
        run_grid(
            config_path="config/strategy_v2_3_cap040.yaml",
            label="v2.3 baseline",
            output_dir="output/robustness_phase6"
        )
    
    if mode in ('d4tuned', 'both'):
        run_grid(
            config_path="config/strategy_v2_3_cap040_D4_tuned.yaml",
            label="D4 tuned",
            output_dir="output/robustness_phase6"
        )