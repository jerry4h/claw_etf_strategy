#!/usr/bin/env python3
"""
中证500ETF 候选参数完整评估 — MC生存率 + WF + DSR + SPS
直接调用 robustness 模块，对 3 个候选配置依次评估。
"""
import sys, os, json, tempfile
from pathlib import Path
import yaml

os.environ["MPLBACKEND"] = "Agg"

ROOT = Path("/home/ubuntu/claw_etf_strategy")
CFG_TEMPLATE = ROOT / "config" / "strategy_v3_0_final.yaml"
_V = ROOT / ".venv" / "lib" / "python3.12" / "site-packages"
if _V.exists():
    sys.path.insert(0, str(_V))
sys.path.insert(0, str(ROOT))

from src.backtest import run_backtest
from src.strategy import load_config, StrategyConfig
from src.robustness import (
    compute_dsr,
    run_mc_survival_test,
    compute_starting_point_sensitivity,
    compute_benchmark_relative_win_rate,
)

# ── 3 个候选配置 ──
candidates = [
    {
        "name": "C1: mom4_volw11",
        "desc": "mom=4, vol_window=11, vol_w=1.12, thresh=0.025",
        "params": {"mom_window": 4, "vol_window": 11, "vol_w": 1.12, "rebalance_threshold": 0.025},
    },
    {
        "name": "C2: mom6_volw12",
        "desc": "mom=6, vol_window=12, vol_w=1.15, thresh=0.03",
        "params": {"mom_window": 6, "vol_window": 12, "vol_w": 1.15, "rebalance_threshold": 0.03},
    },
    {
        "name": "C3: mom5_volw11_DDmin",
        "desc": "mom=5, vol_window=11, vol_w=1.10, thresh=0.05 (最优DD)",
        "params": {"mom_window": 5, "vol_window": 11, "vol_w": 1.10, "rebalance_threshold": 0.05},
    },
]

# Common non-sweep params
common_params = {"inv_vol_window": 10, "step_low": 0.15}

# ── 创建临时 config 文件 ──
def make_config(name: str, params: dict) -> Path:
    p = ROOT / "tmp" / f"cfg_{name.replace(':','_').replace(' ','_')}.yaml"
    with open(CFG_TEMPLATE) as f:
        cfg = yaml.safe_load(f)
    # Apply params
    for k, v in params.items():
        if k == "vol_w":
            cfg["scoring"]["vol_w"] = v
        elif k == "mom_window":
            cfg["factors"]["mom_window"] = v
        elif k == "vol_window":
            cfg["factors"]["vol_window"] = v
        elif k == "rebalance_threshold":
            cfg["rebalance"]["threshold"] = v
        elif k == "inv_vol_window":
            cfg["inv_vol_allocation"]["window"] = v
        elif k == "step_low":
            cfg["defense"]["step_low"] = v
    for k, v in common_params.items():
        if k == "inv_vol_window":
            cfg["inv_vol_allocation"]["window"] = v
        elif k == "step_low":
            cfg["defense"]["step_low"] = v
    with open(p, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return p


# ── 评估每个候选 ──
results = []
for cand in candidates:
    name = cand["name"]
    desc = cand["desc"]
    params = {**cand["params"]}

    print(f"\n{'=' * 70}")
    print(f"  评估: {name} — {desc}")
    print(f"{'=' * 70}")

    # 1. 创建临时 config 并运行基础回测
    cfg_path = make_config(name, params)
    cfg = load_config(cfg_path)
    r = run_backtest(cfg)
    m = r.metrics
    nav = r.nav_series["nav"]
    wrets = nav.pct_change().dropna()

    print(f"  基础回测:")
    print(f"    Sharpe={m['sharpe_ratio']:.4f}  AnnRet={m['annual_return']*100:.2f}%")
    print(f"    DD={m['max_drawdown']*100:.2f}%  Vol={m['annual_volatility']*100:.2f}%")
    print(f"    WinRate={m['win_rate']*100:.1f}%  Final={nav.iloc[-1]:.2f}x")

    # 2. DSR — Deflated Sharpe Ratio
    n_trials = sum(len([
        0.80, 0.90, 1.00, 1.05, 1.10, 1.15, 1.20,
        8, 10, 12, 14, 16, 20,
        0.10, 0.12, 0.15, 0.18, 0.20,
        4, 5, 6, 8, 10,
        10, 13, 16, 20, 25,
        0.03, 0.05, 0.06, 0.07, 0.10,
    ]))  # approximate number of backtests run during search
    n_obs = len(wrets)
    skew = wrets.skew()
    kurt = wrets.kurtosis()
    dsr = compute_dsr(m["sharpe_ratio"], n_trials, n_obs, skew, kurt)
    print(f"  DSR:         {dsr:.4f}" + (" 🟢" if dsr > 0.95 else " 🟡" if dsr > 0.50 else " 🔴"))

    # 3. MC 生存率 (100 次, 快)
    try:
        mc_rate, mc_details = run_mc_survival_test(str(cfg_path), n_runs=100, perturbation=0.15, n_jobs=4)
        print(f"  MC 生存率:   {mc_rate*100:.1f}%" + (" 🟢" if mc_rate > 0.95 else " 🟡" if mc_rate > 0.80 else " 🔴"))
    except Exception as e:
        print(f"  MC 生存率:   ERROR — {e}")
        mc_rate = None

    # 4. WF — Walk-Forward (9 窗口)
    try:
        wf_rate, wf_details = compute_benchmark_relative_win_rate(str(cfg_path), n_windows=9)
        print(f"  WF:          {wf_rate*100:.1f}%" + (" 🟢" if wf_rate > 0.80 else " 🟡" if wf_rate > 0.55 else " 🔴"))
    except Exception as e:
        print(f"  WF:          ERROR — {e}")
        wf_rate = None

    # 5. SPS — 起点敏感度
    try:
        sps = compute_starting_point_sensitivity(str(cfg_path))
        sps_label = f"  起点敏感度: 最差={sps.get('worst_annual_return',0)*100:.2f}%  均值={sps.get('mean_annual_return',0)*100:.2f}%"
        print(sps_label)
    except Exception as e:
        print(f"  起点敏感度:  ERROR — {e}")
        sps = {}

    results.append({
        "name": name,
        "desc": desc,
        "sharpe": m["sharpe_ratio"],
        "ann_ret": m["annual_return"],
        "dd": m["max_drawdown"],
        "final": float(nav.iloc[-1]),
        "dsr": dsr,
        "mc": mc_rate,
        "wf": wf_rate,
        "sps_worst": sps.get("worst_annual_return", None),
        "sps_mean": sps.get("mean_annual_return", None),
    })

    # 清理临时 config
    os.unlink(cfg_path)

# ── 汇总对比 ──
print(f"\n\n{'=' * 70}")
print(f"  汇总对比")
print(f"{'=' * 70}")
print(f"  {'候选':>20s} {'Sharpe':>8s} {'年化':>7s} {'DD':>7s} {'终值':>7s} {'DSR':>6s} {'MC':>5s} {'WF':>5s}")
print(f"  {'-'*70}")
for r in results:
    mc_str = f"{r['mc']*100:.0f}%" if r["mc"] is not None else "ERR"
    wf_str = f"{r['wf']*100:.0f}%" if r["wf"] is not None else "ERR"
    dsr_str = f"{r['dsr']:.2f}" if r["dsr"] is not None else "ERR"
    print(f"  {r['name']:>20s} {r['sharpe']:>8.3f} {r['ann_ret']*100:>6.2f}% {r['dd']*100:>6.2f}% {r['final']:>6.2f}x {dsr_str:>6s} {mc_str:>5s} {wf_str:>5s}")

# ── 推荐 ──
print(f"\n{'=' * 70}")
best = max(results, key=lambda r: r["sharpe"] * (0.5 + 0.5 * (r["wf"] or 0)))
print(f"  推荐: {best['name']} ({best['desc']})")
print(f"  Sharpe={best['sharpe']:.3f}  DD={best['dd']*100:.2f}%  WF={best['wf']*100:.0f}%  DSR={best['dsr']:.2f}")
print(f"{'=' * 70}")