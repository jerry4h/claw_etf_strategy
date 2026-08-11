#!/usr/bin/env python3
"""v4.0 统一评估入口 — realized + adversarial 双维度 + 多目标约束判定。

多目标定位(约束优化,非单目标最大化 Sharpe)：
    max  realized 年化收益
    s.t. 全情景(realized + 压力情景) 最大回撤 ≤ D_max (默认 12%)
     AND realized 收益 > 等权每周再平衡
     AND 各"硬门禁"机制在压力情景下 收益 > 等权

分机制门禁(不同鲁棒方向由不同机制控制; 见 adversarial_robustness.SCENARIO_MECHANISM)：
    vol_defense / defense_asset / dispersion / composite  → 硬门禁(必须全过)
    selection                                             → 软门禁(仅记录)
        —— selection 抗"进攻资产收益退化"是策略"天花板", 非现有超参可解,
           强行当硬约束会把优化器逼进死角; 记录但不阻断, 交由 universe 层解决。

用法:
  python scripts/evaluate.py                                   # 评估 v4_1, D_max=0.12
  python scripts/evaluate.py --config config/xxx.yaml --dmax 0.12
  python scripts/evaluate.py --json                            # JSON 输出
  python scripts/evaluate.py --save-baseline                   # 存基线快照
  python scripts/evaluate.py --vs-baseline                     # 与基线快照对比
"""
import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.strategy import load_config


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, PROJ / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_adv = _load("adv_eval", "scripts/adversarial_robustness.py")
_bench = _load("bench_eval", "scripts/benchmark_compare.py")

BASELINE_PATH = PROJ / "output" / "adversarial" / "baseline_metrics.json"

# 机制门禁策略: hard=必须过, soft=仅记录不阻断
MECHANISM_GATE = {
    "vol_defense":   "hard",
    "defense_asset": "hard",
    "dispersion":    "hard",
    "composite":     "hard",
    "corr_crisis":   "hard",
    "selection":     "soft",
}


def evaluate_full(cfg, d_max=0.12, seeds=(11, 22, 33, 44, 55, 66, 77), include_corr_scenarios=False):
    """双维度评估 + 约束判定。返回结构化 dict。

    d_max: 全情景最大回撤上限(回撤约束)。
    include_corr_scenarios: True 时额外评估 v4.4 相关性危机情景(corr_crisis 硬门禁)。
    """
    # --- 维度 1: realized 历史 (真实数据, 同口径三方基准) ---
    bench = _bench.compute_benchmarks(cfg)
    r_strat, r_ew = bench["strategy"], bench["ew_rebalanced"]
    realized = {
        "annual_return": r_strat["annual_return"],
        "max_drawdown": r_strat["max_drawdown"],
        "sharpe": r_strat["sharpe_ratio"],
        "ew_annual_return": r_ew["annual_return"],
        "ew_max_drawdown": r_ew["max_drawdown"],
        "ew_sharpe": r_ew["sharpe_ratio"],
        "window": bench["window"],
    }

    # --- 维度 2: adversarial 压力情景 (CCC-GARCH 合成) ---
    adv = _adv.robustness_score(cfg, seeds=seeds, include_corr_scenarios=include_corr_scenarios)

    # --- 约束判定 ---
    realized_dd_ok = bool(realized["max_drawdown"] <= d_max)
    realized_beats_ew = bool(realized["annual_return"] > realized["ew_annual_return"])
    adv_dd_ok = bool(adv["worst_maxdd"] <= d_max)

    # 分机制门禁(口径=Sharpe): 硬门禁机制要求 策略Sharpe≥等权 在其全部情景成立(pass_rate==1.0)。
    # 全情景回撤 DD≤D_max 由 adv_dd_ok 全局约束覆盖; 收益口径仅记录(防御型策略结构上难赢原始收益)。
    mech_gates = {}
    hard_pass = True
    for mech, d in adv["by_mechanism"].items():
        gate = MECHANISM_GATE.get(mech, "hard")
        passed = bool(d["pass_rate"] >= 1.0)
        mech_gates[mech] = {
            "gate": gate, "passed": passed,
            "pass_rate_sharpe": d["pass_rate"],
            "pass_rate_return": d["pass_rate_return"],
            "worst_maxdd": d["worst_maxdd"], "worst_sharpe": d["worst_sharpe"],
        }
        if gate == "hard" and not passed:
            hard_pass = False

    constraints = {
        "d_max": d_max,
        "realized_dd_ok": realized_dd_ok,
        "realized_beats_ew": realized_beats_ew,
        "adv_dd_ok": adv_dd_ok,
        "adv_return_pass_rate": adv["pass_rate_return"],
        "mechanism_gates": mech_gates,
    }
    verdict = bool(realized_dd_ok and realized_beats_ew and adv_dd_ok and hard_pass)
    fails = [k for k in ("realized_dd_ok", "realized_beats_ew", "adv_dd_ok") if not constraints[k]]
    fails += [f"hard:{m}" for m, g in mech_gates.items() if g["gate"] == "hard" and not g["passed"]]

    # --- PVD bootstrap 升格：pvd_enabled 时 block bootstrap 胜率纳入 verdict ---
    bootstrap_info = None
    if getattr(cfg, 'pvd_enabled', False):
        _rj = _load("rj", "scripts/robustness_joint.py")
        real_rets, real_dates, first_nav = _rj.prepare_real_data()
        base_m = _rj.eval_on_real(cfg, real_rets, real_dates, first_nav, "base")
        rows, _ = _rj.run_test2(cfg, real_rets, real_dates, first_nav,
                                n_paths=200, block_len=13, seed_base=7700)
        if rows and base_m:
            dist = _rj.judge_test2(rows, base_m)
            bootstrap_info = {
                "win_rate": dist["win_rate_over_ew"],
                "alpha_p10": dist["alpha_sharpe_p10"],
                "pass": dist["pass_relative_alpha"],
            }
            if not bootstrap_info["pass"]:
                verdict = False
                fails.append("pvd_bootstrap")

    return {
        "objective": realized["annual_return"],   # 优化目标: 最大化 realized 年化收益
        "verdict": "PASS" if verdict else "FAIL",
        "failed_constraints": fails,
        "realized": realized,
        "adversarial": adv,
        "constraints": constraints,
        "bootstrap": bootstrap_info,
    }


def _print_report(ev, cfg_name):
    r = ev["realized"]; c = ev["constraints"]
    print("=" * 68)
    print(f" 统一评估: {cfg_name}   D_max={c['d_max']:.0%}   判定: {ev['verdict']}")
    print("=" * 68)
    w = r["window"]
    print(f" [realized {w['start']}~{w['end']}, {w['weeks']}w]")
    print(f"   策略  年化={r['annual_return']:>7.2%}  MaxDD={r['max_drawdown']:>6.2%}  Sharpe={r['sharpe']:.3f}")
    print(f"   等权  年化={r['ew_annual_return']:>7.2%}  MaxDD={r['ew_max_drawdown']:>6.2%}  Sharpe={r['ew_sharpe']:.3f}")
    print(f"   收益>等权: {'Y' if c['realized_beats_ew'] else 'N'}   回撤≤D_max: {'Y' if c['realized_dd_ok'] else 'N'}")
    print(f" [adversarial 压力情景]  全情景最大回撤={ev['adversarial']['worst_maxdd']:.2%}  "
          f"回撤≤D_max: {'Y' if c['adv_dd_ok'] else 'N'}")
    print(f"   收益口径通过率={c['adv_return_pass_rate']:.0%}")
    print(" 分机制门禁(口径=Sharpe≥等权):")
    for mech, g in c["mechanism_gates"].items():
        flag = "PASS" if g["passed"] else "FAIL"
        print(f"   [{g['gate']:<4s}] {mech:<14s} Sharpe胜率={g['pass_rate_sharpe']:.0%}  "
              f"(收益胜率={g['pass_rate_return']:.0%})  worstDD={g['worst_maxdd']:.2%}  -> {flag}")
    print("-" * 68)
    if ev["failed_constraints"]:
        print(f" 未过约束: {ev['failed_constraints']}")
    print(f" 优化目标(realized 年化收益) = {ev['objective']:.4%}")


def main():
    p = argparse.ArgumentParser(description="v4.0 统一评估(realized+adversarial+约束)")
    p.add_argument("--config", default="config/strategy_v4_6.yaml")
    p.add_argument("--dmax", type=float, default=0.12, help="全情景最大回撤上限")
    p.add_argument("--seeds", default="11,22,33,44,55,66,77")
    p.add_argument("--json", action="store_true")
    p.add_argument("--corr-scenarios", action="store_true",
                   help="额外评估相关性危机情景集: corr_regime_shift / corr_crisis_combo + grey_corr_combo(灰区监控)")
    p.add_argument("--save-baseline", action="store_true")
    p.add_argument("--vs-baseline", action="store_true")
    args = p.parse_args()

    seeds = tuple(int(x) for x in args.seeds.split(","))
    cfg = load_config(PROJ / args.config)
    ev = evaluate_full(cfg, d_max=args.dmax, seeds=seeds, include_corr_scenarios=args.corr_scenarios)

    if args.json:
        print(json.dumps(ev, ensure_ascii=False, indent=2, default=str))
    else:
        _print_report(ev, args.config)

    if args.vs_baseline and BASELINE_PATH.exists():
        base = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        print("-" * 68)
        print(f" vs 基线({base.get('_config','?')}): "
              f"年化 {base['objective']:.2%} -> {ev['objective']:.2%}  "
              f"全情景DD {base['adversarial']['worst_maxdd']:.2%} -> {ev['adversarial']['worst_maxdd']:.2%}")

    if args.save_baseline:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        snap = dict(ev); snap["_config"] = args.config; snap["_dmax"] = args.dmax
        BASELINE_PATH.write_text(json.dumps(snap, ensure_ascii=False, indent=2, default=str),
                                 encoding="utf-8")
        print(f"\n基线快照已存: {BASELINE_PATH}")


if __name__ == "__main__":
    main()
