#!/usr/bin/env python3
"""联合鲁棒性检验 — 参数轴 × 数据轴 × 联合。

背景：v4.3 目前已过 CCC-GARCH 对抗 + 3 通道 OOS，但两轴都是"一轴 fix、另一轴动"。
本脚本回答用户提出的科学问题：
    参数邻域 + 数据邻域 + 联合曲面 是否都光滑（no cliff, no thin ridge）？

Test 1: 参数轴局部敏感度（fix 真实历史数据）
  8 个活参 × ±5/10/15%（连续）或 ±1/2/3 步（离散）单参扫描
  判据：Δ Sharpe ≤ 20% base、Δ MaxDD ≤ +3pp、方向单调无断崖

Test 2: 数据轴 block bootstrap 分布（fix v4.3 参数）
  200 次 moving block bootstrap on 原始周收益, block=13 周（≈1 季度）
  判据：Sharpe P10 ≥ 1.0, MaxDD P90 ≤ 10%, 年化 P10 > EW 基线

Test 3: 联合鲁棒性（参数 ε ∈ ±10% × bootstrap seed）
  LHS 采样 N 组 (Δparams, seed) 对同时扰动
  判据：PASS 率 ≥ 70%; 联合损失 ≤ (参数边缘 + 数据边缘) × 1.3（无强非线性交互）

用法：
  python scripts/robustness_joint.py --test all                 # 全套
  python scripts/robustness_joint.py --test t1                  # 只跑参数轴
  python scripts/robustness_joint.py --test t2 --n 200          # 数据轴 200 次
  python scripts/robustness_joint.py --test t3 --n 200 --eps 0.10  # 联合
"""
import argparse
import contextlib
import dataclasses
import importlib.util
import io
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.strategy import load_config
from src.data_loader import load_nav_data, resample_weekly, ETFS


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, PROJ / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


dm = _load("dm", "scripts/data_manifold.py")
oos = _load("oos", "scripts/oos_validation.py")

OUT = PROJ / "output" / "robustness"
OUT.mkdir(parents=True, exist_ok=True)


# ===== 8 个活参 (7 SPACE_TAPER + mom_window). 每项: (cfg 字段, 边界 lo, hi, is_int, 扰动类型) =====
# 扰动类型: "rel" = 相对倍率 ±5/10/15%; "abs_int" = 离散 ±1/2/3 步
ACTIVE_PARAMS = [
    ("max_def",          0.50, 1.00,  False, "rel"),
    ("def_alloc",        0.10, 0.50,  False, "rel"),
    ("top_n",            2,    3,     True,  "abs_int"),
    ("step_low",         0.05, 0.25,  False, "rel"),
    ("step_high",        0.20, 0.70,  False, "rel"),
    ("vol_taper_window", 8,    20,    True,  "abs_int"),
    ("vol_taper_len",    2,    8,     True,  "abs_int"),
    ("mom_window",       4,    10,    True,  "abs_int"),
]

REL_DELTAS = [-0.15, -0.10, -0.05, 0.05, 0.10, 0.15]
ABS_DELTAS = [-3, -2, -1, 1, 2, 3]


def perturb_single(base_cfg, name, lo, hi, is_int, dtype, delta):
    """对 base_cfg 的字段 name 施加 delta 扰动, 返回 (new_cfg, effective_new_val, applied_bool)。"""
    cur = getattr(base_cfg, name)
    if dtype == "rel":
        new = float(cur) * (1.0 + delta)
    else:
        new = int(cur) + int(delta)
    if is_int:
        new = int(round(new))
    new = max(lo, min(hi, new))
    if is_int and int(new) == int(cur):
        return None, cur, False  # 边界或整数无变化
    if (not is_int) and abs(new - cur) < 1e-9:
        return None, cur, False
    kw = {name: new}
    # 联动约束: vol_taper_window 变 → inv_vol_window 跟随、taper_len 保 window-2 上限
    if name == "vol_taper_window":
        kw["inv_vol_window"] = int(new)
        if int(base_cfg.vol_taper_len) >= int(new) - 1:
            kw["vol_taper_len"] = max(1, int(new) - 2)
    # def_alloc 上穿 max_def → 抬 max_def
    if name == "def_alloc" and new > base_cfg.max_def:
        kw["max_def"] = min(1.0, new + 0.15)
    # step_low 上穿 step_high → 抬 step_high
    if name == "step_low" and new > base_cfg.step_high:
        kw["step_high"] = min(0.70, new + 0.05)
    # vol_taper_len 上穿 window-1 → 收
    if name == "vol_taper_len" and int(new) >= int(base_cfg.vol_taper_window) - 1:
        kw["vol_taper_len"] = max(1, int(base_cfg.vol_taper_window) - 2)
    return dataclasses.replace(base_cfg, **kw), kw[name], True


def perturb_joint(base_cfg, deltas_map):
    """一次性对多个字段扰动 (Test 3 用). deltas_map: {name: delta_val}"""
    cfg = base_cfg
    applied = {}
    for name, lo, hi, is_int, dtype in ACTIVE_PARAMS:
        d = deltas_map.get(name)
        if d is None or (is_int and int(d) == 0):
            continue
        new_cfg, new_val, ok = perturb_single(cfg, name, lo, hi, is_int, dtype, d)
        if ok:
            cfg = new_cfg
            applied[name] = new_val
    return cfg, applied


def prepare_real_data():
    """加载真实周收益, 返回 (real_returns[T-1,K], real_dates[T], first_nav[K])"""
    nav = load_nav_data(dm.REAL_CSV)
    wk = resample_weekly(nav)
    w_rets = wk.pct_change().dropna()
    return w_rets.values, wk.index, wk.iloc[0].values


def eval_on_real(cfg, real_returns, real_dates, first_nav, tag):
    """在真实历史数据上跑一次 backtest, 返回 metrics dict (含 EW 对比)."""
    m = oos.eval_strat_ew_on_returns(real_returns, real_dates, first_nav, cfg, tag)
    if m is None:
        return None
    return {
        "sharpe": float(m["strat_sharpe"]),
        "maxdd": float(m["strat_maxdd"]),
        "annual": float(m["strat_annual"]),
        "ew_sharpe": float(m["ew_sharpe"]),
        "ew_annual": float(m["ew_annual"]),
    }


def eval_on_bootstrap(cfg, real_returns, real_dates, first_nav, block_len, seed, tag):
    """block bootstrap 一条路径后跑 backtest"""
    boot = oos.block_bootstrap(real_returns, block_len, seed)
    m = oos.eval_strat_ew_on_returns(boot, real_dates, first_nav, cfg, tag)
    if m is None:
        return None
    return {
        "sharpe": float(m["strat_sharpe"]),
        "maxdd": float(m["strat_maxdd"]),
        "annual": float(m["strat_annual"]),
        "ew_sharpe": float(m["ew_sharpe"]),
        "ew_annual": float(m["ew_annual"]),
    }


# =============== Test 1: 参数轴 ===============
def run_test1(base_cfg, base_metrics, real_returns, real_dates, first_nav):
    print("\n===== Test 1: 参数轴局部敏感度 (fix 真实数据) =====")
    rows = []
    t0 = time.time()
    for name, lo, hi, is_int, dtype in ACTIVE_PARAMS:
        deltas = REL_DELTAS if dtype == "rel" else ABS_DELTAS
        for d in deltas:
            new_cfg, new_val, ok = perturb_single(base_cfg, name, lo, hi, is_int, dtype, d)
            if not ok:
                continue
            m = eval_on_real(new_cfg, real_returns, real_dates, first_nav, f"t1_{name}_{d}")
            if m is None:
                continue
            m.update({
                "param": name, "delta_raw": d, "delta_type": dtype,
                "base_val": getattr(base_cfg, name), "new_val": new_val,
                "d_sharpe_rel": (m["sharpe"] - base_metrics["sharpe"]) / max(1e-6, abs(base_metrics["sharpe"])),
                "d_maxdd_pp":   (m["maxdd"]  - base_metrics["maxdd"])  * 100,
                "d_annual_pp":  (m["annual"] - base_metrics["annual"]) * 100,
            })
            rows.append(m)
            print(f"  {name:20s} Δ={d:+.3g}  ->  Sh={m['sharpe']:.3f}({m['d_sharpe_rel']*100:+.1f}%) "
                  f"DD={m['maxdd']*100:5.2f}%({m['d_maxdd_pp']:+.2f}pp) "
                  f"ann={m['annual']*100:5.2f}%({m['d_annual_pp']:+.2f}pp)", flush=True)
    print(f"  Test 1 完成: {len(rows)} 次回测, 耗时 {time.time()-t0:.1f}s")
    return rows


def judge_test1(base_metrics, rows, sharpe_drop_pct=0.20, maxdd_rise_pp=3.0):
    """按参数分组判 (a) 最大 Sharpe 掉幅 ≤ 20% (b) MaxDD 上升 ≤ 3pp (c) 单调性"""
    by_param = {}
    for r in rows:
        by_param.setdefault(r["param"], []).append(r)
    verdict = {}
    for p, rs in by_param.items():
        rs_sorted = sorted(rs, key=lambda x: x["delta_raw"])
        max_sh_drop = min(r["d_sharpe_rel"] for r in rs)  # 最负
        max_dd_rise = max(r["d_maxdd_pp"] for r in rs)
        # 断崖: 相邻点 Sharpe 差 > 15%
        cliff = False
        for i in range(len(rs_sorted) - 1):
            if abs(rs_sorted[i+1]["d_sharpe_rel"] - rs_sorted[i]["d_sharpe_rel"]) > 0.15:
                cliff = True
        pass_sh = max_sh_drop >= -sharpe_drop_pct
        pass_dd = max_dd_rise <= maxdd_rise_pp
        pass_all = pass_sh and pass_dd and not cliff
        verdict[p] = {
            "max_sharpe_drop_pct": max_sh_drop * 100,
            "max_maxdd_rise_pp": max_dd_rise,
            "cliff_detected": cliff,
            "pass_sharpe": pass_sh, "pass_maxdd": pass_dd, "pass_cliff": not cliff,
            "pass_all": pass_all,
        }
    return verdict


# =============== Test 2: 数据轴 ===============
def run_test2(base_cfg, real_returns, real_dates, first_nav, n_paths, block_len, seed_base):
    print(f"\n===== Test 2: 数据轴 block bootstrap (fix v4.3), n={n_paths}, block={block_len} =====")
    rows = []
    t0 = time.time()
    for i in range(n_paths):
        m = eval_on_bootstrap(base_cfg, real_returns, real_dates, first_nav, block_len, seed_base + i, f"t2_{i}")
        if m is None:
            continue
        rows.append({"seed": seed_base + i, **m})
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (n_paths - i - 1)
            print(f"  [{i+1:3d}/{n_paths}] ETA {eta/60:.1f}min", flush=True)
    print(f"  Test 2 完成: {len(rows)} 条 bootstrap 路径, 耗时 {(time.time()-t0)/60:.1f}min")
    return rows


def judge_test2(rows, base_metrics, sharpe_p10_min=1.0, maxdd_p90_max=0.10):
    sh = np.array([r["sharpe"] for r in rows])
    dd = np.array([r["maxdd"] for r in rows])
    an = np.array([r["annual"] for r in rows])
    ewa = np.array([r["ew_annual"] for r in rows])
    def q(a, p): return float(np.quantile(a, p))
    dist = {
        "sharpe_p10": q(sh, 0.10), "sharpe_p50": q(sh, 0.50), "sharpe_p90": q(sh, 0.90),
        "maxdd_p10":  q(dd, 0.10), "maxdd_p50":  q(dd, 0.50), "maxdd_p90":  q(dd, 0.90),
        "annual_p10": q(an, 0.10), "annual_p50": q(an, 0.50), "annual_p90": q(an, 0.90),
        "sharpe_pass_rate": float((sh >= 1.0).mean()),
        "maxdd_pass_rate":  float((dd <= 0.10).mean()),
        "beat_ew_annual_rate": float((an > ewa).mean()),
    }
    dist["pass_all"] = (
        dist["sharpe_p10"] >= sharpe_p10_min and
        dist["maxdd_p90"]  <= maxdd_p90_max and
        dist["annual_p10"] >  0
    )
    return dist


# =============== Test 3: 联合 ===============
def lhs_signed(n, dim, seed=2026):
    """LHS in [-1, +1]^dim."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, dim))
    for j in range(dim):
        cuts = np.linspace(-1, 1, n + 1)
        pts = cuts[:-1] + rng.random(n) * (cuts[1:] - cuts[:-1])
        rng.shuffle(pts)
        out[:, j] = pts
    return out


def run_test3(base_cfg, real_returns, real_dates, first_nav, n, eps, block_len, seed_base):
    print(f"\n===== Test 3: 联合 (参数 ε=±{eps*100:.0f}% × bootstrap seed), n={n} =====")
    U = lhs_signed(n, len(ACTIVE_PARAMS), seed=2026)
    rows = []
    t0 = time.time()
    for i in range(n):
        deltas = {}
        for j, (name, lo, hi, is_int, dtype) in enumerate(ACTIVE_PARAMS):
            u = float(U[i, j])
            if dtype == "rel":
                deltas[name] = u * eps
            else:
                # 映射 [-1,+1] → 整数 ±3 步 (含 0 视为无扰动)
                deltas[name] = int(round(u * 3))
        cfg_p, applied = perturb_joint(base_cfg, deltas)
        m = eval_on_bootstrap(cfg_p, real_returns, real_dates, first_nav, block_len, seed_base + i, f"t3_{i}")
        if m is None:
            continue
        rows.append({"i": i, "seed": seed_base + i, "deltas": deltas, "applied": applied, **m})
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (n - i - 1)
            print(f"  [{i+1:3d}/{n}] Sh={m['sharpe']:.2f} DD={m['maxdd']*100:.1f}% ETA {eta/60:.1f}min", flush=True)
    print(f"  Test 3 完成: {len(rows)} 组, 耗时 {(time.time()-t0)/60:.1f}min")
    return rows


def judge_test3(rows, base_metrics, sharpe_min=1.0, maxdd_max=0.10, pass_rate_min=0.70):
    sh = np.array([r["sharpe"] for r in rows])
    dd = np.array([r["maxdd"] for r in rows])
    an = np.array([r["annual"] for r in rows])
    pass_mask = (sh >= sharpe_min) & (dd <= maxdd_max)
    return {
        "n": len(rows),
        "sharpe_p10": float(np.quantile(sh, 0.10)),
        "sharpe_p50": float(np.quantile(sh, 0.50)),
        "maxdd_p90":  float(np.quantile(dd, 0.90)),
        "annual_p10": float(np.quantile(an, 0.10)),
        "pass_rate":  float(pass_mask.mean()),
        "pass_all":   float(pass_mask.mean()) >= pass_rate_min,
    }


def compare_marginal_vs_joint(t1_rows, t2_rows, t3_rows, base_metrics):
    """粗略比较 (a) 参数边缘最大 Sharpe 掉幅 (b) 数据边缘 P10 Sharpe 掉幅 (c) 联合 P10 Sharpe 掉幅
    如果联合损失 > (t1+t2) × 1.3 说明有强交互(薄峰)."""
    base_sh = base_metrics["sharpe"]
    t1_max_drop = -min(r["d_sharpe_rel"] for r in t1_rows) * base_sh  # abs Sharpe drop
    t2_p10 = float(np.quantile([r["sharpe"] for r in t2_rows], 0.10))
    t2_drop = max(0.0, base_sh - t2_p10)
    t3_p10 = float(np.quantile([r["sharpe"] for r in t3_rows], 0.10))
    t3_drop = max(0.0, base_sh - t3_p10)
    lin_predict = t1_max_drop + t2_drop
    ratio = t3_drop / lin_predict if lin_predict > 1e-6 else float("nan")
    return {
        "base_sharpe": base_sh,
        "t1_max_param_drop":    t1_max_drop,
        "t2_data_p10_drop":     t2_drop,
        "t3_joint_p10_drop":    t3_drop,
        "linear_predict_drop":  lin_predict,
        "joint_over_linear_ratio": ratio,
        "no_thin_ridge": (not math.isnan(ratio)) and ratio <= 1.30,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/strategy_v4_3.yaml")
    ap.add_argument("--test", choices=["all", "t1", "t2", "t3"], default="all")
    ap.add_argument("--n", type=int, default=200, help="Test 2/3 采样数")
    ap.add_argument("--block", type=int, default=13, help="bootstrap block 长度 (周)")
    ap.add_argument("--eps", type=float, default=0.10, help="Test 3 参数扰动幅度 (相对)")
    ap.add_argument("--seed-base", type=int, default=8000)
    args = ap.parse_args()

    base_cfg = load_config(PROJ / args.config)
    print(f"Base config: {args.config}")
    print(f"  taper_window={base_cfg.vol_taper_window}, taper_len={base_cfg.vol_taper_len}, "
          f"top_n={base_cfg.top_n}, def_alloc={base_cfg.def_alloc:.4f}, max_def={base_cfg.max_def:.4f}")

    real_returns, real_dates, first_nav = prepare_real_data()
    print(f"  真实周收益: T-1={len(real_returns)}, K={real_returns.shape[1]}")

    # baseline (真实数据)
    base_m = eval_on_real(base_cfg, real_returns, real_dates, first_nav, "base")
    print(f"  Base Sharpe={base_m['sharpe']:.4f}, MaxDD={base_m['maxdd']*100:.2f}%, annual={base_m['annual']*100:.2f}%")

    out = {"base_config": args.config, "base_metrics": base_m,
           "params": {n: getattr(base_cfg, n) for n, *_ in ACTIVE_PARAMS}}

    if args.test in ("all", "t1"):
        t1 = run_test1(base_cfg, base_m, real_returns, real_dates, first_nav)
        t1_verdict = judge_test1(base_m, t1)
        out["test1_rows"] = t1
        out["test1_verdict"] = t1_verdict
        print("\n--- Test 1 逐参判定 ---")
        for p, v in t1_verdict.items():
            print(f"  {p:20s} maxΔSh={v['max_sharpe_drop_pct']:+6.1f}%  maxΔDD={v['max_maxdd_rise_pp']:+.2f}pp  "
                  f"cliff={v['cliff_detected']}  → {'PASS' if v['pass_all'] else 'FAIL'}")

    if args.test in ("all", "t2"):
        t2 = run_test2(base_cfg, real_returns, real_dates, first_nav, args.n, args.block, args.seed_base)
        t2_verdict = judge_test2(t2, base_m)
        out["test2_rows"] = t2
        out["test2_verdict"] = t2_verdict
        print("\n--- Test 2 数据轴分布 ---")
        v = t2_verdict
        print(f"  Sharpe P10/P50/P90 = {v['sharpe_p10']:.3f} / {v['sharpe_p50']:.3f} / {v['sharpe_p90']:.3f}")
        print(f"  MaxDD  P10/P50/P90 = {v['maxdd_p10']*100:.2f}% / {v['maxdd_p50']*100:.2f}% / {v['maxdd_p90']*100:.2f}%")
        print(f"  Annual P10/P50/P90 = {v['annual_p10']*100:.2f}% / {v['annual_p50']*100:.2f}% / {v['annual_p90']*100:.2f}%")
        print(f"  → 判据: Sh_p10 ≥ 1.0 [{v['sharpe_p10']>=1.0}], DD_p90 ≤ 10% [{v['maxdd_p90']<=0.10}], ann_p10>0 [{v['annual_p10']>0}] → {'PASS' if v['pass_all'] else 'FAIL'}")

    if args.test in ("all", "t3"):
        t3 = run_test3(base_cfg, real_returns, real_dates, first_nav, args.n, args.eps, args.block, args.seed_base + 10000)
        t3_verdict = judge_test3(t3, base_m)
        out["test3_rows"] = t3
        out["test3_verdict"] = t3_verdict
        v = t3_verdict
        print("\n--- Test 3 联合 ---")
        print(f"  n={v['n']}  Sharpe P10/P50 = {v['sharpe_p10']:.3f} / {v['sharpe_p50']:.3f}")
        print(f"  MaxDD P90 = {v['maxdd_p90']*100:.2f}%   Annual P10 = {v['annual_p10']*100:.2f}%")
        print(f"  PASS 率 = {v['pass_rate']*100:.1f}%  (阈值 70%) → {'PASS' if v['pass_all'] else 'FAIL'}")

    if args.test == "all":
        cmp = compare_marginal_vs_joint(out["test1_rows"], out["test2_rows"], out["test3_rows"], base_m)
        out["marginal_vs_joint"] = cmp
        print("\n--- 边缘 vs 联合损失(Sharpe) ---")
        print(f"  base Sharpe                  = {cmp['base_sharpe']:.3f}")
        print(f"  Test 1 最大参数掉幅            = {cmp['t1_max_param_drop']:.3f}")
        print(f"  Test 2 数据 P10 掉幅          = {cmp['t2_data_p10_drop']:.3f}")
        print(f"  线性预测(边缘和)             = {cmp['linear_predict_drop']:.3f}")
        print(f"  Test 3 联合 P10 掉幅          = {cmp['t3_joint_p10_drop']:.3f}")
        print(f"  联合/线性 比率                = {cmp['joint_over_linear_ratio']:.2f}   "
              f"→ 无薄峰 [{cmp['no_thin_ridge']}]")

    ts = time.strftime("%Y%m%d_%H%M%S")
    fp = OUT / f"robustness_joint_{args.test}_{ts}.json"
    with open(fp, "w") as f:
        json.dump(out, f, ensure_ascii=False, default=str, indent=2)
    print(f"\n结果落盘: {fp}")


if __name__ == "__main__":
    main()
