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

# v4.2 血统: rolling vol, 6 主控 (默认)
SPACE_ROLLING = [
    ("max_def_extra",   (0.15, 0.75), False),
    ("def_alloc",       (0.10, 0.50), False),
    ("top_n",           (2,    3   ), True ),
    ("step_low",        (0.05, 0.25), False),
    ("step_delta_high", (0.05, 0.35), False),
    ("vol_window",      (6,    14  ), True ),
]
# v4.3: tapered vol (消除窗口跳变), 去掉硬窗口 vol_window, 改搜 taper 窗口+降权长度, 7 主控
SPACE_TAPER = [
    ("max_def_extra",    (0.15, 0.75), False),
    ("def_alloc",        (0.10, 0.50), False),
    ("top_n",            (2,    3   ), True ),
    ("step_low",         (0.05, 0.25), False),
    ("step_delta_high",  (0.05, 0.35), False),
    ("vol_taper_window", (8,    20  ), True ),   # tapered vol 窗口
    ("vol_taper_len",    (2,    8   ), True ),   # 最老 N 周线性降权 (unit_to_cfg 里约束 < window-1)
]
SPACE = SPACE_ROLLING       # main() 按 --space 切换
TAPER_MODE = False          # main() 按 --space 切换
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
    if TAPER_MODE:
        # v4.3: tapered vol. 约束 taper_len 至少留 2 个满权重周 (1 <= len <= window-2)。
        kw["vol_taper_enabled"] = True
        kw["vol_taper_len"] = max(1, min(kw["vol_taper_len"], kw["vol_taper_window"] - 2))
        # inv_vol_window 跟随有效波动率窗口 (taper window)
        kw["inv_vol_window"] = kw["vol_taper_window"]
    else:
        # P1-4 修: rolling 模式下 inv_vol_window 与 vol_window 联动, 避免静默分裂。
        kw["inv_vol_window"] = kw["vol_window"]
    return dataclasses.replace(base_cfg, **kw)


def cfg_params_flat(cfg):
    """将 cfg 的主控派生字段扁平化, 便于 JSON 存/打印 (按 rolling/taper 模式)。"""
    d = {
        "top_n": int(cfg.top_n), "inv_vol_window": int(cfg.inv_vol_window),
        "def_alloc": float(cfg.def_alloc), "step_low": float(cfg.step_low),
        "step_high": float(cfg.step_high), "max_def": float(cfg.max_def),
    }
    if TAPER_MODE:
        d["vol_taper_enabled"] = bool(cfg.vol_taper_enabled)
        d["vol_taper_window"] = int(cfg.vol_taper_window)
        d["vol_taper_len"] = int(cfg.vol_taper_len)
    else:
        d["vol_window"] = int(cfg.vol_window)
    return d


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
        "realized_sharpe": float(r["sharpe"]),
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
    """把主控派生字段写回 yaml (其他字段保留 base 内容); 按 rolling/taper 模式区分。"""
    import yaml
    y = yaml.safe_load(open(base_yaml_path, "r", encoding="utf-8"))
    label = "v4.3 tapered-vol 优化" if TAPER_MODE else "v4.0 框架优化候选"
    y["strategy"] = {"name": f"虾池ETF轮动 ({label})", "version": "候选"}
    if note:
        y["strategy"]["note"] = note
    y.setdefault("selection", {})["top_n"] = int(cfg.top_n)
    factors = y.setdefault("factors", {})
    if TAPER_MODE:
        factors["vol_taper_enabled"] = True
        factors["vol_taper_window"] = int(cfg.vol_taper_window)
        factors["vol_taper_len"] = int(cfg.vol_taper_len)
        # P1-3 修: taper 模式下 vol_window 不生效, 从生成物中删除避免误导性并存
        factors.pop("vol_window", None)
    else:
        factors["vol_window"] = int(cfg.vol_window)
    # inv_vol_window 联动写回, 避免与有效波动率窗口静默分裂
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
    global SPACE, TAPER_MODE
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="config/strategy_v4_3.yaml")
    p.add_argument("--space",    choices=["rolling", "taper"], default="rolling",
                   help="rolling=v4.2 6 主控(含 vol_window); taper=v4.3 7 主控(vol_taper_window/len 取代 vol_window)")
    p.add_argument("--dmax",     type=float, default=0.12)
    p.add_argument("--n",        type=int, default=200)
    p.add_argument("--k",        type=int, default=15)
    p.add_argument("--objective", choices=["annual", "sharpe"], default="annual",
                   help="优化目标: annual=max realized 年化(v4.2 血统); sharpe=max realized Sharpe")
    p.add_argument("--seeds-a",  default="11,22,33")
    p.add_argument("--seeds-b",  default="11,22,33,44,55,66,77")
    p.add_argument("--oos-seeds", default="",
                   help="Stage C 泛化门的独立 seed 集(如 100,101,...,116); 空=跳过。候选须在此独立seed上仍PASS才入围, 防过拟合训练seed。")
    p.add_argument("--out-yaml", default="config/strategy_v4_next.yaml")
    args = p.parse_args()

    # 按 --space 切换搜索空间 (影响 unit_to_cfg / cfg_params_flat / cfg_to_yaml 的模块级读取)
    if args.space == "taper":
        SPACE = SPACE_TAPER
        TAPER_MODE = True
    else:
        SPACE = SPACE_ROLLING
        TAPER_MODE = False

    seeds_a = tuple(int(x) for x in args.seeds_a.split(","))
    seeds_b = tuple(int(x) for x in args.seeds_b.split(","))
    base_cfg = load_config(PROJ / args.config)
    ev_mod = _load("ev_opt", "scripts/evaluate.py")

    # ==================== Stage A ====================
    print(f"===== Stage A: LHS N={args.n} {len(SPACE)}D ({args.space}), seeds={seeds_a}, D_max={args.dmax} =====")
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
                         {"config": args.config, "space": args.space, "dmax": args.dmax, "n": args.n,
                          "seeds": list(seeds_a), "results": stage_a, "partial": True})

    save_partial(OUT / "optimize_stageA.json",
                 {"config": args.config, "dmax": args.dmax, "n": args.n,
                  "seeds": list(seeds_a), "results": stage_a, "partial": False})

    # ==================== Stage B ====================
    obj_key = "realized_sharpe" if args.objective == "sharpe" else "realized_annual"
    cand_all = [(i, s) for i, s in enumerate(stage_a) if stage_a_ok(s, args.dmax)]
    cand_all.sort(key=lambda x: -x[1].get(obj_key, 0))
    total_ok = len(cand_all)
    cand = cand_all[:args.k]
    print(f"\n===== Stage B: 精验 {len(cand)} 候选 (共 {total_ok}/{args.n} 过初筛, 取 {obj_key} Top-{args.k}), seeds={seeds_b} =====")
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
              f"Sh={s.get('realized_sharpe',0):.3f} ann={s.get('realized_annual',0):+.4f} "
              f"DDreal={s.get('realized_maxdd',0):.3f} DDadv={s.get('adv_worst_maxdd',0):.3f} "
              f"vd_m={s.get('mech_margin',{}).get('vol_defense', 0):+.3f} "
              f"cp_m={s.get('mech_margin',{}).get('composite',   0):+.3f}   ETA={eta/60:.1f}min",
              flush=True)

    save_partial(OUT / "optimize_stageB.json",
                 {"config": args.config, "space": args.space, "objective": args.objective,
                  "dmax": args.dmax, "seeds": list(seeds_b), "results": stage_b})

    passing_b = [s for s in stage_b if stage_b_ok(s, args.dmax)]

    # ==================== Stage C: OOS 泛化门 (防过拟合训练 seed) ====================
    oos_seeds = tuple(int(x) for x in args.oos_seeds.split(",") if x.strip()) if args.oos_seeds else ()
    eligible = passing_b
    if oos_seeds and passing_b:
        print(f"\n===== Stage C: OOS 泛化门, 独立 seeds={oos_seeds} (Stage-B-PASS 须在此仍 PASS 才入围) =====")
        survivors = []
        for j, s in enumerate(passing_b):
            cfg = unit_to_cfg(base_cfg, U[s["stage_a_idx"]])
            oos = eval_summary(cfg, ev_mod, oos_seeds, args.dmax)
            oos_ok = stage_b_ok(oos, args.dmax)
            s["oos"] = {"adv_worst_maxdd": oos.get("adv_worst_maxdd"),
                        "mech_margin": oos.get("mech_margin"), "pass": bool(oos_ok)}
            if oos_ok:
                survivors.append(s)
            print(f" [{j+1:2d}/{len(passing_b)}] {'OOS-PASS' if oos_ok else 'OOS-fail'} "
                  f"Sh={s.get('realized_sharpe',0):.3f} "
                  f"OOS_DDadv={oos.get('adv_worst_maxdd',0):.3f} "
                  f"OOS_vd_m={oos.get('mech_margin',{}).get('vol_defense',0):+.3f} "
                  f"OOS_cp_m={oos.get('mech_margin',{}).get('composite',0):+.3f}", flush=True)
        eligible = survivors

    # ==================== 选优 & 输出 ====================
    print(f"\n===== 结果 =====")
    print(f" Stage A: {total_ok}/{args.n} 过初筛; 送 Stage B: {len(cand)} (取 {obj_key} Top-{args.k})")
    print(f" Stage B: {len(passing_b)}/{len(stage_b)} 训练seed严格 PASS")
    if oos_seeds:
        print(f" Stage C: {len(eligible)}/{len(passing_b)} 通过 OOS 泛化门 (独立 seed 仍 PASS)")

    if eligible:
        best = max(eligible, key=lambda s: s[obj_key])
        gate = "B+C(OOS泛化)" if oos_seeds else "B(训练seed)"
        print(f"\n>>> 最优 (目标={args.objective}, 通过门={gate}):")
        print(f"    realized_sharpe  = {best['realized_sharpe']:.4f}  (v4.2 基线 1.635)")
        print(f"    realized_annual  = {best['realized_annual']:.4%}")
        print(f"    realized_maxdd   = {best['realized_maxdd']:.4%}")
        print(f"    adv_worst_maxdd  = {best['adv_worst_maxdd']:.4%}")
        print(f"    机制 margin(训练) = {best['mech_margin']}")
        if "oos" in best:
            print(f"    OOS 泛化(独立seed) = DDadv {best['oos']['adv_worst_maxdd']:.4f}, margin {best['oos']['mech_margin']}")
        print(f"    params           = {best['params']}")
        best_cfg = unit_to_cfg(base_cfg, U[best["stage_a_idx"]])
        yaml_path = PROJ / args.out_yaml
        note = (f"LHS N={args.n} {args.space} 目标={args.objective}; realized_sharpe={best['realized_sharpe']:.4f} "
                f"年化={best['realized_annual']:.4f}; 通过门={gate}")
        cfg_to_yaml(best_cfg, yaml_path, PROJ / args.config, note=note)
        print(f"\n最优 config 已写: {yaml_path}")
    else:
        stage_b_clean = [s for s in stage_b if "error" not in s]
        reason = "无候选通过 OOS 泛化门" if oos_seeds else "无严格 PASS 候选"
        print(f" {reason}。展示 Stage B 里 未过约束最少 者 Top-3 (共 {len(stage_b_clean)} 有效):")
        if not stage_b_clean:
            print("   Stage B 全部崩溃, 无可展示候选。")
            return
        def rank(s):
            fails = sum(1 for m in HARD_MECH if s["mech_margin"][m] <= 0)
            fails += 0 if s["adv_worst_maxdd"] <= args.dmax else 1
            return (fails, -s.get(obj_key, 0))
        stage_b_clean.sort(key=rank)
        for s in stage_b_clean[:3]:
            print(f"  Sh={s['realized_sharpe']:.3f} ann={s['realized_annual']:.3%} "
                  f"DDreal={s['realized_maxdd']:.3%} DDadv={s['adv_worst_maxdd']:.3%} "
                  f"vd_m={s['mech_margin']['vol_defense']:+.3f} cp_m={s['mech_margin']['composite']:+.3f} "
                  f"ds_m={s['mech_margin']['defense_asset']:+.3f} di_m={s['mech_margin']['dispersion']:+.3f}")
            print(f"    params={s['params']}")


if __name__ == "__main__":
    main()
