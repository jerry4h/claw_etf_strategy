#!/usr/bin/env python3
"""任务6-第二部分: rounded_fine 圆整配置的数据轴 bootstrap 双重检测 (Test 2 相对 alpha 判据)。

设计:
  复用 scripts/robustness_joint.py 的 run_test2 依赖链 (oos.block_bootstrap /
  oos.eval_strat_ew_on_returns) 与 judge_test2, 对
  config/experiments/v4_3_rounded_fine.yaml 跑 moving block bootstrap
  (block=13 周) × 200 条路径。

  seed 序列与生产基线完全一致: seed = seed_base(8000) + i, i∈[0,200)
  (见 robustness_joint.run_test2)。block_bootstrap 是 seed 确定性的
  (np.random.default_rng(seed)), 且 data/all_etfs_nav_latest.csv 自基线运行
  (2026-07-29) 后未变 → 200 条路径与基线 JSON
  output/robustness/robustness_joint_all_20260729_114702.json 的 test2_rows
  逐 seed 可配对。

流程:
  Step 0  复现校验: 用 v4.3 生产配置重跑基线的 3 个 seed, 与基线 JSON 逐位对比
          (|ΔSharpe| < 1e-6 判定可配对; 否则回退为重跑基线 200 条做同批对照)。
  Step 1  rounded_fine × 200 路径 (策略 + 等权)。
  Step 2  judge_test2 分位数/胜率/alpha 判据 + 逐路径配对差 (rounded − baseline)。

只读复用, 不修改任何现有文件。基线 JSON 只读引用。
用法: .venv/bin/python scripts/_exp_rounded_robust_boot.py
输出: output/experiments/exp_rounded_robust_boot.json
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.strategy import load_config


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, PROJ / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rj = _load("rj", "scripts/robustness_joint.py")   # 内部已加载 oos/dm

OUT = PROJ / "output" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)
BASELINE_JSON = PROJ / "output" / "robustness" / "robustness_joint_all_20260729_114702.json"
ROUNDED_CFG = PROJ / "config" / "experiments" / "v4_3_rounded_fine.yaml"
BASE_CFG = PROJ / "config" / "strategy_v4_3.yaml"

N_PATHS = 200
BLOCK = 13
SEED_BASE = 8000          # 与 robustness_joint.py --seed-base 默认值一致
REPRO_CHECK_SEEDS = (8000, 8047, 8199)   # 头/中/尾抽 3 个 seed 校验复现
REPRO_TOL = 1e-6


def q(a, p):
    return float(np.quantile(np.asarray(a, float), p))


def main():
    t0 = time.time()
    base_cfg = load_config(BASE_CFG)
    rounded_cfg = load_config(ROUNDED_CFG)
    print(f"[cfg] baseline: def_alloc={base_cfg.def_alloc}, step_low={base_cfg.step_low}, "
          f"step_high={base_cfg.step_high}, max_def={base_cfg.max_def}")
    print(f"[cfg] rounded : def_alloc={rounded_cfg.def_alloc}, step_low={rounded_cfg.step_low}, "
          f"step_high={rounded_cfg.step_high}, max_def={rounded_cfg.max_def}")

    real_returns, real_dates, first_nav = rj.prepare_real_data()
    print(f"[data] 真实周收益 T-1={len(real_returns)}, K={real_returns.shape[1]}")

    with open(BASELINE_JSON, encoding="utf-8") as f:
        bj = json.load(f)
    base_rows = {int(r["seed"]): r for r in bj["test2_rows"]}
    # 基线 JSON 的 test2_verdict 为旧版字段(缺 alpha/胜率), 用同一 judge_test2 从
    # 其 test2_rows 重算(纯内存, 不重跑回测; 数值与生产报告同源)。
    base_verdict = rj.judge_test2(bj["test2_rows"], bj["base_metrics"])
    print(f"[baseline] 引用 {BASELINE_JSON.name}: {len(base_rows)} 条 test2 路径, "
          f"seeds {min(base_rows)}..{max(base_rows)}")
    print(f"[baseline] verdict 重算: 胜率={base_verdict['win_rate_over_ew']:.1%}, "
          f"alpha_p10={base_verdict['alpha_sharpe_p10']:+.3f}")

    result = {
        "task": "rounded_fine bootstrap Test2 双重检测 (block=13, n=200, seeds 8000..8199)",
        "rounded_config": str(ROUNDED_CFG.relative_to(PROJ)),
        "baseline_config": bj["base_config"],
        "baseline_source": str(BASELINE_JSON.relative_to(PROJ)),
        "n_paths": N_PATHS, "block_len": BLOCK, "seed_base": SEED_BASE,
    }

    # ---------- Step 0: seed 复现校验 (v4.3 生产配置 vs 基线 JSON) ----------
    print("[Step 0] seed 复现校验 (v4.3 × 3 seeds vs 基线 JSON) ...")
    repro = []
    for s in REPRO_CHECK_SEEDS:
        m = rj.eval_on_bootstrap(base_cfg, real_returns, real_dates, first_nav,
                                 BLOCK, s, f"repro_{s}")
        ref = base_rows[s]
        d_sh = abs(m["sharpe"] - ref["sharpe"])
        d_ew = abs(m["ew_sharpe"] - ref["ew_sharpe"])
        repro.append({"seed": s, "sharpe_new": m["sharpe"], "sharpe_ref": ref["sharpe"],
                      "abs_diff_sharpe": d_sh, "abs_diff_ew_sharpe": d_ew})
        print(f"    seed={s}: new={m['sharpe']:.10f} ref={ref['sharpe']:.10f} "
              f"|d|={d_sh:.2e} (ew |d|={d_ew:.2e})")
    reproducible = all(r["abs_diff_sharpe"] < REPRO_TOL and r["abs_diff_ew_sharpe"] < REPRO_TOL
                       for r in repro)
    result["repro_check"] = {"seeds": list(REPRO_CHECK_SEEDS), "tol": REPRO_TOL,
                             "rows": repro, "reproducible": reproducible}
    print(f"    -> 复现{'成功: 基线 200 路径直接引用, 可逐路径配对' if reproducible else '失败: 将重跑基线 200 条做同批对照'}")

    # ---------- (回退) 基线不可复现时重跑 ----------
    if not reproducible:
        print("[Step 0b] 重跑基线 v4.3 × 200 路径 ...")
        rows_b, failed_b = rj.run_test2(base_cfg, real_returns, real_dates, first_nav,
                                        N_PATHS, BLOCK, SEED_BASE)
        base_rows = {int(r["seed"]): r for r in rows_b}
        base_verdict = rj.judge_test2(rows_b, bj["base_metrics"])
        result["baseline_rerun"] = {"rows": rows_b, "failed": failed_b,
                                    "verdict": base_verdict}

    # ---------- Step 1: rounded_fine × 200 路径 ----------
    print(f"[Step 1] rounded_fine × {N_PATHS} 路径 (block={BLOCK}, seeds {SEED_BASE}..{SEED_BASE+N_PATHS-1}) ...")
    rows_r, failed_r = rj.run_test2(rounded_cfg, real_returns, real_dates, first_nav,
                                    N_PATHS, BLOCK, SEED_BASE)
    verdict_r = rj.judge_test2(rows_r, bj["base_metrics"])
    result["rounded_rows"] = rows_r
    result["rounded_failed_seeds"] = failed_r
    result["rounded_verdict"] = verdict_r
    result["baseline_verdict_ref"] = base_verdict

    # ---------- Step 2: 逐路径配对差 (rounded − baseline) ----------
    print("[Step 2] 逐路径配对差 (rounded − baseline) ...")
    paired = []
    for r in rows_r:
        s = int(r["seed"])
        b = base_rows.get(s)
        if b is None:
            continue
        paired.append({
            "seed": s,
            "d_sharpe": r["sharpe"] - b["sharpe"],
            "d_maxdd": r["maxdd"] - b["maxdd"],
            "d_annual": r["annual"] - b["annual"],
            "d_alpha_sharpe": (r["sharpe"] - r["ew_sharpe"]) - (b["sharpe"] - b["ew_sharpe"]),
        })
    dsh = [p["d_sharpe"] for p in paired]
    ddd = [p["d_maxdd"] for p in paired]
    dan = [p["d_annual"] for p in paired]
    identical = sum(1 for x in dsh if abs(x) < 1e-12)
    pair_stats = {
        "n_paired": len(paired),
        "n_identical_sharpe": identical,
        "d_sharpe_mean": float(np.mean(dsh)), "d_sharpe_std": float(np.std(dsh)),
        "d_sharpe_p10": q(dsh, 0.10), "d_sharpe_p50": q(dsh, 0.50), "d_sharpe_p90": q(dsh, 0.90),
        "d_sharpe_min": float(np.min(dsh)), "d_sharpe_max": float(np.max(dsh)),
        "d_sharpe_win_rate": float(np.mean([x > 0 for x in dsh])),
        "d_maxdd_mean_pp": float(np.mean(ddd)) * 100,
        "d_maxdd_p10_pp": q(ddd, 0.10) * 100, "d_maxdd_p90_pp": q(ddd, 0.90) * 100,
        "d_annual_mean_pp": float(np.mean(dan)) * 100,
    }
    result["paired_diff"] = pair_stats
    result["paired_rows"] = paired

    # ---------- 生产门禁判定 ----------
    gate = {
        "win_rate_min": 0.90, "alpha_p10_min": 0.0,
        "rounded_win_rate": verdict_r["win_rate_over_ew"],
        "rounded_alpha_p10": verdict_r["alpha_sharpe_p10"],
        "rounded_pass_relative_alpha": verdict_r["pass_relative_alpha"],
        "baseline_win_rate": base_verdict["win_rate_over_ew"],
        "baseline_alpha_p10": base_verdict["alpha_sharpe_p10"],
        "baseline_pass_relative_alpha": base_verdict["pass_relative_alpha"],
    }
    result["gate"] = gate

    out_json = OUT / "exp_rounded_robust_boot.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=float)
    print(f"[save] {out_json}")

    # ---------- 摘要 ----------
    v, b = verdict_r, base_verdict
    print("\n===== 摘要 =====")
    print(f"  Sharpe P10/P50/P90  rounded {v['sharpe_p10']:.3f}/{v['sharpe_p50']:.3f}/{v['sharpe_p90']:.3f}"
          f"  vs baseline {b['sharpe_p10']:.3f}/{b['sharpe_p50']:.3f}/{b['sharpe_p90']:.3f}")
    print(f"  MaxDD  P10/P50/P90  rounded {v['maxdd_p10']:.2%}/{v['maxdd_p50']:.2%}/{v['maxdd_p90']:.2%}"
          f"  vs baseline {b['maxdd_p10']:.2%}/{b['maxdd_p50']:.2%}/{b['maxdd_p90']:.2%}")
    print(f"  胜率(策略>等权)     rounded {v['win_rate_over_ew']:.1%} vs baseline {b['win_rate_over_ew']:.1%}")
    print(f"  alpha P10/P50/P90   rounded {v['alpha_sharpe_p10']:+.3f}/{v['alpha_sharpe_p50']:+.3f}/{v['alpha_sharpe_p90']:+.3f}"
          f"  vs baseline {b['alpha_sharpe_p10']:+.3f}/{b['alpha_sharpe_p50']:+.3f}/{b['alpha_sharpe_p90']:+.3f}")
    print(f"  相对 alpha 判据 (胜率>=90% & alpha_p10>0): rounded "
          f"{'PASS' if gate['rounded_pass_relative_alpha'] else 'FAIL'}, baseline "
          f"{'PASS' if gate['baseline_pass_relative_alpha'] else 'FAIL'}")
    ps = pair_stats
    print(f"  配对差 dSharpe (rounded-baseline): mean={ps['d_sharpe_mean']:+.4f} "
          f"P10={ps['d_sharpe_p10']:+.4f} P90={ps['d_sharpe_p90']:+.4f} "
          f"(完全相同路径 {ps['n_identical_sharpe']}/{ps['n_paired']})")
    print(f"DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
