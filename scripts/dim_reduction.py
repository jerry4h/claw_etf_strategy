#!/usr/bin/env python3
"""v4.0 节点2: 维度约简 — Morris 敏感度筛选各机制主控超参。

方法: Morris Elementary Effects (Morris 1991)
  p=4 grid levels, delta = p/(2(p-1)) = 2/3
  r 条轨迹, 每条 k+1 点 in [0,1]^k, 每步只在一个维度 +delta
  每个参数收集 r 个 EE_i = (y_after - y_before) / delta
    μ*_i = mean(|EE_i|)  ── 该参数对输出的总重要性(含线性+交互)
    σ_i  = std(EE_i)     ── 非线性 / 交互强度

多输出敏感度(把 evaluate_full 拆成机制维度):
  - realized_annual   : realized 年化收益(优化目标)
  - realized_maxdd    : realized 最大回撤(硬约束)
  - adv_worst_maxdd   : 全情景对抗最大回撤(硬约束; DD≤D_max 依据)
  - vd_worst_sharpe   : vol_defense 最差 Sharpe (硬门禁, v4_1 当前 FAIL)
  - cp_worst_sharpe   : composite 最差 Sharpe   (硬门禁, v4_1 当前 FAIL)
  - ds_worst_sharpe   : defense_asset 最差 Sharpe
  - di_worst_sharpe   : dispersion 最差 Sharpe

用途: 筛选出各机制的主控超参 → 节点3 优化器只在少数维度搜, 把 ~10^16 -> ~10^5。

Notes:
  * 筛选用 3 seed 加速; 相对排序对 seed 数不敏感, 但绝对 μ*/σ 量级不作门禁使用。
  * step_high 通过 step_delta_high 参数化 (step_high = step_low + step_delta_high) 保证 > step_low;
    max_def 通过 max_def_extra 参数化 (max_def = min(1.0, def_alloc + max_def_extra)) 保证 > def_alloc。
"""
import argparse
import dataclasses
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.strategy import load_config

# --- 12 个候选主控超参 (覆盖 realized 与所有对抗机制) ---
# name: (low, high, is_discrete)
PARAM_SPACE = [
    ("vol_w",                 (0.50, 1.50), False),
    ("mom_w",                 (0.50, 1.50), False),
    ("top_n",                 (2,    3   ), True ),
    ("mom_window",            (4,    12  ), True ),
    ("vol_window",            (6,    14  ), True ),
    ("def_alloc",             (0.10, 0.50), False),
    ("step_low",              (0.05, 0.25), False),
    ("step_delta_high",       (0.05, 0.35), False),   # -> step_high = step_low + this
    ("max_def_extra",         (0.15, 0.75), False),   # -> max_def   = min(1, def_alloc + this)
    ("crisis_corr_threshold", (0.40, 0.80), False),
    ("crisis_corr_max_boost", (0.00, 0.30), False),
    ("stop_loss",             (0.05, 0.15), False),
]
OUT = PROJ / "output" / "adversarial"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, PROJ / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def morris_trajectories(k, r, p=4, seed=42):
    """r 条 Morris 轨迹, 每条 k+1 点 in [0,1]^k, 逐维 +delta。"""
    delta = p / (2 * (p - 1))                       # p=4 → 2/3
    grid = np.linspace(0, 1, p)                     # {0, 1/3, 2/3, 1}
    base_pool = grid[grid + delta <= 1.0 + 1e-9]    # {0, 1/3}
    rng = np.random.default_rng(seed)
    traj = np.zeros((r, k + 1, k))
    perms = np.zeros((r, k), dtype=int)
    for t in range(r):
        x = rng.choice(base_pool, size=k)
        perm = rng.permutation(k)
        perms[t] = perm
        traj[t, 0] = x
        cur = x.copy()
        for i, j in enumerate(perm):
            cur[j] += delta
            traj[t, i + 1] = cur
    return traj, perms, delta


def unit_to_cfg(base_cfg, unit_vec):
    """[0,1]^k → StrategyConfig(dataclasses.replace) + 派生约束。"""
    kwargs = {}
    for (name, (lo, hi), discrete), u in zip(PARAM_SPACE, unit_vec):
        v = lo + u * (hi - lo)
        if discrete:
            v = int(round(v))
        kwargs[name] = v
    # 派生: step_high, max_def
    kwargs["step_high"] = kwargs["step_low"] + kwargs.pop("step_delta_high")
    kwargs["max_def"]   = min(1.0, kwargs["def_alloc"] + kwargs.pop("max_def_extra"))
    return dataclasses.replace(base_cfg, **kwargs)


def outputs_for_cfg(cfg, ev_mod, seeds):
    try:
        ev = ev_mod.evaluate_full(cfg, d_max=0.12, seeds=seeds)
    except Exception as e:
        return {"realized_annual": 0.0, "realized_maxdd": 1.0,
                "adv_worst_maxdd": 1.0,
                "vd_worst_sharpe": -2.0, "cp_worst_sharpe": -2.0,
                "ds_worst_sharpe": -2.0, "di_worst_sharpe": -2.0,
                "_error": str(e)[:100]}
    r = ev["realized"]; mg = ev["constraints"]["mechanism_gates"]
    return {
        "realized_annual": float(r["annual_return"]),
        "realized_maxdd":  float(r["max_drawdown"]),
        "adv_worst_maxdd": float(ev["adversarial"]["worst_maxdd"]),
        "vd_worst_sharpe": float(mg["vol_defense"]["worst_sharpe"]),
        "cp_worst_sharpe": float(mg["composite"]["worst_sharpe"]),
        "ds_worst_sharpe": float(mg["defense_asset"]["worst_sharpe"]),
        "di_worst_sharpe": float(mg["dispersion"]["worst_sharpe"]),
    }


def analyze_morris(trajs, perms, outputs, delta):
    r, kp1, k = trajs.shape
    out_keys = [ok for ok in outputs[0].keys() if not ok.startswith("_")]
    stats = {ok: {n: {"ees": []} for (n, _, _) in PARAM_SPACE} for ok in out_keys}
    n_skipped = 0
    idx = 0
    for t in range(r):
        for step in range(k):
            j = perms[t, step]
            name = PARAM_SPACE[j][0]
            y_b = outputs[idx + step]; y_a = outputs[idx + step + 1]
            # M3 修: 任一端崩溃时哨兵值会产生虚假巨大梯度污染 μ*/σ, 跳过。
            if "_error" in y_b or "_error" in y_a:
                n_skipped += 1
                continue
            for ok in out_keys:
                stats[ok][name]["ees"].append((y_a[ok] - y_b[ok]) / delta)
        idx += kp1
    if n_skipped:
        print(f"  [warn] analyze_morris: 跳过 {n_skipped} 个受崩溃 eval 污染的 EE (共 {r*k} 期望)")
    for ok in out_keys:
        for name, d in stats[ok].items():
            ees = np.asarray(d["ees"])
            stats[ok][name] = {"mu_star": float(np.mean(np.abs(ees))),
                                "sigma":   float(np.std(ees, ddof=0)),
                                "mu":      float(np.mean(ees)),
                                "n":       int(len(ees))}
    return stats, out_keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/strategy_v4_3.yaml")
    p.add_argument("--r", type=int, default=4, help="Morris 轨迹数")
    p.add_argument("--seeds", default="11,22,33")
    p.add_argument("--out", default=str(OUT / "morris_sensitivity.json"))
    args = p.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(","))
    base_cfg = load_config(PROJ / args.config)
    ev_mod = _load("ev_dim", "scripts/evaluate.py")

    k = len(PARAM_SPACE)
    trajs, perms, delta = morris_trajectories(k, args.r, p=4, seed=42)
    total = args.r * (k + 1)
    print(f"Morris: k={k} params, r={args.r} trajs, {total} evaluations, seeds/eval={seeds}")
    outputs = []; t0 = time.time()
    for i in range(args.r):
        for step in range(k + 1):
            unit = trajs[i, step]
            cfg = unit_to_cfg(base_cfg, unit)
            o = outputs_for_cfg(cfg, ev_mod, seeds)
            outputs.append(o)
            done = len(outputs); elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"  [{done:3d}/{total}] annual={o.get('realized_annual',0):+.3f} "
                  f"DDadv={o.get('adv_worst_maxdd',0):.3f} vdSh={o.get('vd_worst_sharpe',0):+.2f} "
                  f"cpSh={o.get('cp_worst_sharpe',0):+.2f}  ETA={eta/60:.1f}min", flush=True)

    stats, out_keys = analyze_morris(trajs, perms, outputs, delta)

    ranked = {}
    print("\n===== Morris μ* 排序 (Top-6/输出) =====")
    for ok in out_keys:
        rows = sorted(stats[ok].items(), key=lambda x: -x[1]["mu_star"])
        ranked[ok] = [(n, r["mu_star"], r["sigma"]) for n, r in rows]
        print(f"\n[{ok}]")
        for n, ms, sg in ranked[ok][:6]:
            print(f"  {n:<28s} μ*={ms:>8.4f}  σ={sg:>8.4f}")

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": args.config, "seeds": list(seeds), "r": args.r, "k": k, "delta": delta,
        "param_space": [(n, lo, hi, d) for (n, (lo, hi), d) in PARAM_SPACE],
        "stats": stats, "ranked": ranked,
        "elapsed_min": (time.time() - t0) / 60,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n结果已存: {out_path}")


if __name__ == "__main__":
    main()
