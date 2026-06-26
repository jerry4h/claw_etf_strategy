#!/usr/bin/env python3
"""Phase 6 Full-Grid using concurrent.futures.ProcessPoolExecutor.
Writes progress to log files for monitoring.
"""
import sys, os, json, numpy as np
from pathlib import Path

# Make stdout unbuffered
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)  # line-buffered

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.robustness import (
    build_full_grid, _grid_point_worker, evaluate_robustness,
    GridPointResult,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import datetime

LOG_DIR = Path("output/robustness_phase6")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)

def run_grid(config_path: str, label: str, tag: str):
    log("=" * 70)
    log(f"Phase 6 Full-Parameter Grid — {label}")
    log(f"Config: {config_path}")
    log("=" * 70)

    # Step 1: Standard evaluation
    log("[Step 1] Running standard evaluation...")
    result = evaluate_robustness(
        config_path=config_path,
        n_mc=400, n_wf_windows=9, n_trials=52,
        n_jobs=-1, perturbation=0.15,
        oat=False, pbo=False, sps=False,
        full_grid=False,
    )

    # Step 2: Build grid
    log("[Step 2] Building full grid configs...")
    grid_configs = build_full_grid(config_path)
    log(f"  Total grid points: {len(grid_configs)}")

    # Step 3: Run with ProcessPoolExecutor
    log("[Step 3] Running grid points in parallel...")
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
    log(f"  Using {n_cpu} workers")

    with ProcessPoolExecutor(max_workers=n_cpu) as executor:
        futures = {executor.submit(_grid_point_worker, args): i 
                   for i, args in enumerate(worker_args)}
        done = 0
        total = len(futures)
        for f in as_completed(futures):
            done += 1
            if done % 5 == 0 or done == total:
                log(f"  Grid point {done}/{total}...")
            try:
                r = f.result()
                if r is not None:
                    raw_results.append(r)
            except Exception as e:
                idx = futures[f]
                log(f"  [WARN] Grid point {idx} failed: {e}")

    log(f"  Completed: {len(raw_results)}/{total} grid points")

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
    log(f"\n{label} metrics:")
    log(f"  Sharpe: {result.strategy_metrics['sharpe_ratio']:.4f}")
    log(f"  Annual Return: {result.strategy_metrics['annual_return']*100:.2f}%")
    log(f"  Max Drawdown: {result.strategy_metrics['max_drawdown']*100:.2f}%")
    log(f"  DSR: {result.dsr:.4f}")
    log(f"  MC Survival: {result.mc_survival_rate*100:.1f}%")
    log(f"  WF Win Rate: {result.benchmark_relative_win_rate*100:.1f}%")
    log(f"  Full Grid: {n_params} params, {n_points} grid points")

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
    with open(LOG_DIR / fname, 'w') as f:
        json.dump(entry, f, indent=2, default=str, ensure_ascii=False)

    log(f"Saved to {LOG_DIR / fname}")
    log(f"{label} Phase 6 grid complete!")
    return entry


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'both'

    if mode in ('baseline', 'both'):
        run_grid(
            config_path="config/strategy_v2_3_cap040.yaml",
            label="v2.3 baseline",
            tag="baseline"
        )

    if mode in ('d4tuned', 'both'):
        run_grid(
            config_path="config/strategy_v2_3_cap040_D4_tuned.yaml",
            label="D4 tuned",
            tag="d4tuned"
        )

    log("ALL DONE")