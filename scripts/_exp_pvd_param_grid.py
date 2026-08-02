#!/usr/bin/env python3
"""PVD v4.5 mom_w/vol_w 联合参数网格校验 (Step 3)

3×3 网格: mom_w ∈ {0.9, 1.0, 1.1} × vol_w ∈ {1.0, 1.1, 1.2}, pvd_w 固定 0.15
逐配置 run_backtest 收集 Sharpe/MaxDD
对 Pareto 前沿 ≤3 候选跑 block bootstrap 100 路径
"""
import dataclasses
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.strategy import load_config
from src.backtest import run_backtest

# --- 配置 ---
BASE_CFG_PATH = PROJ / "config" / "strategy_v4_5_pvd.yaml"
MOM_WS = [0.9, 1.0, 1.1]
VOL_WS = [1.0, 1.1, 1.2]
BOOTSTRAP_PATHS = 100
BOOTSTRAP_BLOCK = 13
BOOTSTRAP_SEED_BASE = 8800


def grid_backtest():
    """运行 3×3 网格回测"""
    base_cfg = load_config(BASE_CFG_PATH)
    results = []
    print("=" * 70)
    print(" PVD v4.5 mom_w/vol_w 3×3 联合参数网格")
    print("=" * 70)
    print(f"  pvd_w 固定 = {base_cfg.pvd_w}")
    print(f"  mom_w ∈ {MOM_WS}")
    print(f"  vol_w ∈ {VOL_WS}")
    print()
    print(f"  {'mom_w':>6s}  {'vol_w':>6s}  {'Sharpe':>8s}  {'MaxDD':>8s}  {'Annual':>8s}")
    print(f"  {'-'*46}")

    for mw, vw in product(MOM_WS, VOL_WS):
        cfg = dataclasses.replace(base_cfg, mom_w=mw, vol_w=vw)
        r = run_backtest(cfg)
        m = r.metrics
        sharpe = m["sharpe_ratio"]
        maxdd = m["max_drawdown"]
        annual = m["annual_return"]
        results.append({"mom_w": mw, "vol_w": vw, "sharpe": sharpe, "maxdd": maxdd, "annual": annual})
        print(f"  {mw:>6.1f}  {vw:>6.1f}  {sharpe:>8.4f}  {maxdd*100:>7.2f}%  {annual*100:>7.2f}%")

    print()
    return results


def pareto_front(results, max_pareto=3):
    """提取 Pareto 前沿: maximize Sharpe, minimize MaxDD"""
    front = []
    for r in results:
        dominated = False
        for other in results:
            if other is r:
                continue
            # other dominates r if: other.sharpe >= r.sharpe AND other.maxdd <= r.maxdd (strict one)
            if other["sharpe"] >= r["sharpe"] and other["maxdd"] <= r["maxdd"]:
                if other["sharpe"] > r["sharpe"] or other["maxdd"] < r["maxdd"]:
                    dominated = True
                    break
        if not dominated:
            front.append(r)
    # Sort by Sharpe descending
    front.sort(key=lambda x: x["sharpe"], reverse=True)
    return front[:max_pareto]


def run_bootstrap_for_candidates(candidates):
    """对 Pareto 前沿候选跑 block bootstrap"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("rj", PROJ / "scripts" / "robustness_joint.py")
    rj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rj)

    base_cfg = load_config(BASE_CFG_PATH)
    real_rets, real_dates, first_nav = rj.prepare_real_data()

    print(f"\n{'='*70}")
    print(f" Block Bootstrap 验证 (n={BOOTSTRAP_PATHS}, block={BOOTSTRAP_BLOCK})")
    print(f"{'='*70}")
    print(f"  {'mom_w':>6s}  {'vol_w':>6s}  {'win_rate':>9s}  {'alpha_p10':>10s}  {'pass':>6s}")
    print(f"  {'-'*50}")

    bootstrap_results = []
    for cand in candidates:
        cfg = dataclasses.replace(base_cfg, mom_w=cand["mom_w"], vol_w=cand["vol_w"])
        base_m = rj.eval_on_real(cfg, real_rets, real_dates, first_nav, "base")
        rows, _ = rj.run_test2(cfg, real_rets, real_dates, first_nav,
                               n_paths=BOOTSTRAP_PATHS, block_len=BOOTSTRAP_BLOCK,
                               seed_base=BOOTSTRAP_SEED_BASE)
        if rows and base_m:
            dist = rj.judge_test2(rows, base_m)
            wr = dist["win_rate_over_ew"]
            ap10 = dist["alpha_sharpe_p10"]
            passed = dist["pass_relative_alpha"]
        else:
            wr, ap10, passed = 0.0, 0.0, False

        bootstrap_results.append({**cand, "win_rate": wr, "alpha_p10": ap10, "bootstrap_pass": passed})
        print(f"  {cand['mom_w']:>6.1f}  {cand['vol_w']:>6.1f}  {wr*100:>8.1f}%  {ap10:>10.4f}  {'PASS' if passed else 'FAIL':>6s}")

    return bootstrap_results


def main():
    t0 = time.time()
    results = grid_backtest()

    # Pareto front
    front = pareto_front(results)
    print(f"  Pareto 前沿 ({len(front)} 候选):")
    for r in front:
        print(f"    mom_w={r['mom_w']}, vol_w={r['vol_w']} → Sharpe={r['sharpe']:.4f}, MaxDD={r['maxdd']*100:.2f}%")

    # Bootstrap
    bs_results = run_bootstrap_for_candidates(front)

    # Conclusion
    print(f"\n{'='*70}")
    print(" 结论")
    print(f"{'='*70}")
    # Current default
    current = next((r for r in results if r["mom_w"] == 1.0 and r["vol_w"] == 1.1), None)
    best = max((r for r in bs_results if r["bootstrap_pass"]), key=lambda x: x["sharpe"], default=None)
    if best is None:
        best = max(bs_results, key=lambda x: x["sharpe"])

    print(f"  当前配置: mom_w=1.0, vol_w=1.1 → Sharpe={current['sharpe']:.4f}, MaxDD={current['maxdd']*100:.2f}%")
    print(f"  最优候选: mom_w={best['mom_w']}, vol_w={best['vol_w']} → Sharpe={best['sharpe']:.4f}, MaxDD={best['maxdd']*100:.2f}%")

    if best["mom_w"] == 1.0 and best["vol_w"] == 1.1:
        print("  ✅ 当前参数即为最优，无需调整")
    else:
        delta_sharpe = best["sharpe"] - current["sharpe"]
        print(f"  ⚠️ 建议微调: ΔSharpe = {delta_sharpe:+.4f}")
        print(f"     → 需要更新 config/strategy_v4_5_pvd.yaml: mom_w={best['mom_w']}, vol_w={best['vol_w']}")

    print(f"\n  总耗时: {(time.time()-t0)/60:.1f} min")
    return results, bs_results, best


if __name__ == "__main__":
    main()
