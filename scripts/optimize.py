#!/usr/bin/env python3
"""v4.0 节点3: 约束优化器 — LHS 采样 + 双阶段筛选(6 主控)。

搜索空间(节点2 Morris 筛出的 6 主控):
  max_def_extra    [0.15, 0.75]  → max_def = min(1, def_alloc + max_def_extra)
  def_alloc        [0.10, 0.50]
  top_n            [2, 3] int
  step_low         [0.05, 0.25]
  step_delta_high  [0.05, 0.35]  → step_high = step_low + step_delta_high
  vol_window       [6, 14] int
其他保持 v4_1 默认(Morris μ*≈0 已证无关)。

目标 & 约束 (D_max 用户定 12%):
  max  realized 年化收益
  s.t. realized_maxdd ≤ D_max
       realized_annual > realized_ew_annual
       adv_worst_maxdd ≤ D_max
       vol_defense/defense_asset/dispersion/composite  strat_Sharpe > ew_Sharpe

双阶段:
  Stage A(粗筛): LHS N 点, 3-seed 评估, 宽约束(DD slack 1pp + Sharpe margin slack 0.05)
                 → 通过者按 realized_annual 排序, 取 top-K
  Stage B(精验): K 候选, 7-seed 严格约束
                 → 最优严格PASS 者写 config; 无者展示 Top-3 最接近候选表

输出:
  output/adversarial/optimize_stageA.json  ── 所有 N 点结果 + 参数
  output/adversarial/optimize_stageB.json  ── K 候选精验结果
  config/strategy_v4_next.yaml         ── 最优严格PASS config(若存在)
"""
import argparse
import dataclasses
import importlib.util
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.strategy import load_config

SPACE = [
    ("max_def_extra",   (0.15, 0.75), False),
    ("def_alloc",       (0.10, 0.50), False),
    ("top_n",           (2,    3   ), True ),
    ("step_low",        (0.05, 0.25), False),
    ("step_delta_high", (0.05, 0.35), False),
    ("vol_window",      (6,    14  ), True ),
]
HARD_MECH = ("vol_defense", "defense_asset", "dispersion", "composite")
OUT = PROJ / "output" / "adversarial"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, PROJ / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def lhs(n, d, seed=42):
    """Latin Hypercube Sampling: n 点, 每维 n 层, 层内均匀采样, 逐维独立乱序。"""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, d))
    for j in range(d):
        cuts = np.linspace(0, 1, n + 1)
        pts = cuts[:-1] + rng.random(n) * (cuts[1:] - cuts[:-1])
        rng.shuffle(pts)
        out[:, j] = pts
    return out


def unit_to_cfg(base_cfg, u):
    kw = {}
    for (name, (lo, hi), disc), v in zip(SPACE, u):
        val = lo + v * (hi - lo)
        if disc: val = int(round(val))
        kw[name] = val
    kw["step_high"] = kw["step_low"] + kw.pop("step_delta_high")
    kw["max_def"]   = min(1.0, kw["def_alloc"] + kw.pop("max_def_extra"))
    # P1-4 修: inv_vol_window 与 vol_window 语义联动 (二者都是"波动率窗口"),
    # 让优化器改 vol_window 时 inv_vol_window 同步跟随, 避免静默分裂。
    kw["inv_vol_window"] = kw["vol_window"]
    return dataclasses.replace(base_cfg, **kw)


def cfg_params_flat(cfg):
    """将 cfg 的 6+ 派生字段扁平化, 便于 JSON 存/打印。"""
    return {
        "top_n": int(cfg.top_n), "vol_window": int(cfg.vol_window),
        "inv_vol_window": int(cfg.inv_vol_window),
        "def_alloc": float(cfg.def_alloc), "step_low": float(cfg.step_low),
        "step_high": float(cfg.step_high), "max_def": float(cfg.max_def),
    }


def eval_summary(cfg, ev_mod, seeds, d_max):
    """把 evaluate_full 输出压缩到关键字段 + 各机制 Sharpe margin (strat-ew)。"""
    try:
        ev = ev_mod.evaluate_full(cfg, d_max=d_max, seeds=seeds)
    except Exception as e:
        return {"error": str(e)[:120]}
    r = ev["realized"]; adv = ev["adversarial"]; mg = ev["constraints"]["mechanism_gates"]
    scen = adv["scenarios"]
    # 各机制 Sharpe margin: 该机制所属情景里最小的 (strat-ew) sharpe
    mech_margin = {}
    for m in mg:
        names = [n for n, v in scen.items() if v.get("mechanism") == m]
        if names:
            mech_margin[m] = float(min(scen[n]["strategy"] - scen[n]["ew_rebal"] for n in names))
        else:
            mech_margin[m] = float("nan")
    return {
        "verdict": ev["verdict"],
        "realized_annual": float(r["annual_return"]),
        "realized_maxdd":  float(r["max_drawdown"]),
        "realized_ew_annual": float(r["ew_annual_return"]),
        "adv_worst_maxdd": float(adv["worst_maxdd"]),
        "mech_pass":       {m: float(mg[m]["pass_rate_sharpe"]) for m in mg},
        "mech_margin":     mech_margin,
        "mech_worst_dd":   {m: float(mg[m]["worst_maxdd"]) for m in mg},
        "failed":          ev["failed_constraints"],
    }


def stage_a_ok(s, d_max, dd_slack=0.01, margin_slack=-0.05):
    """宽松初筛: 允许 3-seed 采样噪声, 但方向必须正确。"""
    if "error" in s: return False
    if s["realized_maxdd"]    > d_max:                     return False
    if s["realized_annual"]  <= s["realized_ew_annual"]:   return False
    if s["adv_worst_maxdd"]   > d_max + dd_slack:          return False
    for m in HARD_MECH:
        mg = s["mech_margin"][m]
        # 显式 NaN 守卫: nan 参与比较返回 False, 会静默通过约束,
        # 一旦未来 SCENARIO_MECHANISM 移除某 mech 会立刻踩雷。
        if math.isnan(mg) or mg < margin_slack:            return False
    return True


def stage_b_ok(s, d_max):
    """严格约束(用户口径): DD≤D_max & 硬机制 Sharpe strict > EW。"""
    if "error" in s: return False
    if s["realized_maxdd"]    > d_max:                     return False
    if s["realized_annual"]  <= s["realized_ew_annual"]:   return False
    if s["adv_worst_maxdd"]   > d_max:                     return False
    for m in HARD_MECH:
        mg = s["mech_margin"][m]
        if math.isnan(mg) or mg <= 0:                      return False
    return True


def cfg_to_yaml(cfg, out_path, base_yaml_path, note=""):
    """把 6 主控派生字段写回 yaml (其他字段保留 base 内容)。"""
    import yaml
    y = yaml.safe_load(open(base_yaml_path, "r", encoding="utf-8"))
    y["strategy"] = {"name": "虾池ETF轮动 (v4.0 框架优化候选)", "version": "候选"}
    if note:
        y["strategy"]["note"] = note
    y.setdefault("selection", {})["top_n"] = int(cfg.top_n)
    y.setdefault("factors", {})["vol_window"] = int(cfg.vol_window)
    # P1-4 修: inv_vol_window 与 vol_window 联动写回, 避免用户看到 yaml 里 vol_window=8 但 inv_vol_window=10 的静默分裂
    y.setdefault("inv_vol_allocation", {})["window"] = int(cfg.inv_vol_window)
    defense = y.setdefault("defense", {})
    defense["def_alloc"] = round(float(cfg.def_alloc), 4)
    defense["step_low"]  = round(float(cfg.step_low),  4)
    defense["step_high"] = round(float(cfg.step_high), 4)
    defense["max_def"]   = round(float(cfg.max_def),   4)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(y, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def save_partial(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="config/strategy_v4_2.yaml")
    p.add_argument("--dmax",     type=float, default=0.12)
    p.add_argument("--n",        type=int, default=200)
    p.add_argument("--k",        type=int, default=15)
    p.add_argument("--seeds-a",  default="11,22,33")
    p.add_argument("--seeds-b",  default="11,22,33,44,55,66,77")
    p.add_argument("--out-yaml", default="config/strategy_v4_next.yaml")
    args = p.parse_args()

    seeds_a = tuple(int(x) for x in args.seeds_a.split(","))
    seeds_b = tuple(int(x) for x in args.seeds_b.split(","))
    base_cfg = load_config(PROJ / args.config)
    ev_mod = _load("ev_opt", "scripts/evaluate.py")

    # ==================== Stage A ====================
    print(f"===== Stage A: LHS N={args.n} 6D, seeds={seeds_a}, D_max={args.dmax} =====")
    U = lhs(args.n, len(SPACE), seed=42)
    stage_a = []; t0 = time.time()
    for i, u in enumerate(U):
        cfg = unit_to_cfg(base_cfg, u)
        s = eval_summary(cfg, ev_mod, seeds_a, args.dmax)
        s["params"] = cfg_params_flat(cfg)
        stage_a.append(s)
        ok = stage_a_ok(s, args.dmax)
        el = time.time() - t0; eta = el / (i + 1) * (args.n - i - 1)
        marg = s.get("mech_margin", {})
        print(f" [{i+1:3d}/{args.n}] {'OK ' if ok else '-- '}"
              f"ann={s.get('realized_annual',0):+.3f} DDadv={s.get('adv_worst_maxdd',0):.3f} "
              f"vd_m={marg.get('vol_defense', float('nan')):+.2f} "
              f"cp_m={marg.get('composite',   float('nan')):+.2f}   ETA={eta/60:.1f}min",
              flush=True)
        if (i + 1) % 25 == 0:
            save_partial(OUT / "optimize_stageA.json",
                         {"config": args.config, "dmax": args.dmax, "n": args.n,
                          "seeds": list(seeds_a), "results": stage_a, "partial": True})

    save_partial(OUT / "optimize_stageA.json",
                 {"config": args.config, "dmax": args.dmax, "n": args.n,
                  "seeds": list(seeds_a), "results": stage_a, "partial": False})

    # ==================== Stage B ====================
    cand_all = [(i, s) for i, s in enumerate(stage_a) if stage_a_ok(s, args.dmax)]
    cand_all.sort(key=lambda x: -x[1].get("realized_annual", 0))
    total_ok = len(cand_all)
    cand = cand_all[:args.k]
    print(f"\n===== Stage B: 精验 {len(cand)} 候选 (共 {total_ok}/{args.n} 过初筛, 取 realized_annual Top-{args.k}), seeds={seeds_b} =====")
    if not cand:
        print(" Stage A 无候选过初筛。请扩大搜索空间/放宽 D_max/放宽 Sharpe margin slack。")
        return

    stage_b = []; t0 = time.time()
    for j, (idx, sa) in enumerate(cand):
        cfg = unit_to_cfg(base_cfg, U[idx])
        s = eval_summary(cfg, ev_mod, seeds_b, args.dmax)
        s["params"] = cfg_params_flat(cfg); s["stage_a_idx"] = int(idx)
        stage_b.append(s)
        ok = stage_b_ok(s, args.dmax)
        el = time.time() - t0; eta = el / (j + 1) * (len(cand) - j - 1)
        print(f" [{j+1:2d}/{len(cand)}] {'PASS' if ok else 'fail'} "
              f"ann={s.get('realized_annual',0):+.4f} DDreal={s.get('realized_maxdd',0):.3f} "
              f"DDadv={s.get('adv_worst_maxdd',0):.3f} "
              f"vd_m={s.get('mech_margin',{}).get('vol_defense', 0):+.3f} "
              f"cp_m={s.get('mech_margin',{}).get('composite',   0):+.3f}   ETA={eta/60:.1f}min",
              flush=True)

    save_partial(OUT / "optimize_stageB.json",
                 {"config": args.config, "dmax": args.dmax,
                  "seeds": list(seeds_b), "results": stage_b})

    # ==================== 选优 & 输出 ====================
    passing = [s for s in stage_b if stage_b_ok(s, args.dmax)]
    print(f"\n===== 结果 =====")
    print(f" Stage A: {total_ok}/{args.n} 过初筛; 送 Stage B: {len(cand)} (取 realized_annual Top-{args.k})")
    print(f" Stage B: {len(passing)}/{len(stage_b)} 严格 PASS")

    if passing:
        best = max(passing, key=lambda s: s["realized_annual"])
        print(f"\n>>> 最优严格 PASS:")
        print(f"    realized_annual  = {best['realized_annual']:.4%}  (v4_1 基线 17.05%)")
        print(f"    realized_maxdd   = {best['realized_maxdd']:.4%}")
        print(f"    adv_worst_maxdd  = {best['adv_worst_maxdd']:.4%}")
        print(f"    机制 margin      = {best['mech_margin']}")
        print(f"    params           = {best['params']}")
        best_cfg = unit_to_cfg(base_cfg, U[best["stage_a_idx"]])
        yaml_path = PROJ / args.out_yaml
        cfg_to_yaml(best_cfg, yaml_path, PROJ / args.config,
                    note=f"LHS N={args.n} + 7-seed 严格精验; realized_annual={best['realized_annual']:.4f}")
        print(f"\n最优 config 已写: {yaml_path}")
    else:
        # M1 修: 过滤掉 eval 崩溃的条目(它们没有 mech_margin 键)
        stage_b_clean = [s for s in stage_b if "error" not in s]
        print(f" 无严格 PASS 候选。展示 Stage B 里 未过约束最少 者 Top-3 (共 {len(stage_b_clean)} 有效, "
              f"{len(stage_b)-len(stage_b_clean)} 崩溃跳过):")
        if not stage_b_clean:
            print("   Stage B 全部崩溃, 无可展示候选。")
            return
        def rank(s):
            fails = sum(1 for m in HARD_MECH if s["mech_margin"][m] <= 0)
            fails += 0 if s["adv_worst_maxdd"] <= args.dmax else 1
            return (fails, -s.get("realized_annual", 0))
        stage_b_clean.sort(key=rank)
        for s in stage_b_clean[:3]:
            print(f"  ann={s['realized_annual']:.3%} DDreal={s['realized_maxdd']:.3%} "
                  f"DDadv={s['adv_worst_maxdd']:.3%} "
                  f"vd_m={s['mech_margin']['vol_defense']:+.3f} "
                  f"cp_m={s['mech_margin']['composite']:+.3f} "
                  f"ds_m={s['mech_margin']['defense_asset']:+.3f} "
                  f"di_m={s['mech_margin']['dispersion']:+.3f}")
            print(f"    params={s['params']}")


if __name__ == "__main__":
    main()
