#!/usr/bin/env python3
"""C2 研究: 换手抑制 / 滞回带 (holding-bias) — 成本侧净 Sharpe 评估。

假说: top-2 排名在 score gap 小时周间翻转产生纯成本损耗; holding-bias
(现持仓 +ε 才允许替换) 可能改善净 Sharpe。与现有 score_margin=0.02 动态
margin 机制和 PVD gap<0.05 条件有交互——先隔离评估 (M0 关闭 margin 对照)。

机制参考 (聚宽调研 jq_community_survey.md #2/#3/#9):
  - 信号分歧时保持原仓位 (RSRS+均线双信号帖) — holding-bias 的理论依据
  - 分数差值双向过滤 ("枪打出头鸟", 核资轮-添油加醋帖) — H2 极端离散不换仓
  - 最小持有期/止盈冷却 (区间涨幅帖) — 本轮不做 (状态机复杂度高, 留后续)

变体 (基座 = v4.5-pvd 生产 config, 源码手术 monkeypatch, 零 src/ 改动):
  BASE : 生产现状
  M0   : 隔离对照 — 关闭 score margin 机制 (量化现有换手抑制的贡献)
  H1a  : holding bias ε=0.01 (现持仓评分 +ε 后再选 top-2)
  H1b  : holding bias ε=0.02
  H2a  : 极端离散冻结 — 进攻评分极差 >0.35 时保持上周选择 (不追极端)
  H2b  : 极端离散冻结 — 极差 >0.50

判定口径: 净 Sharpe (引擎已同口径扣 fee_rate); E2 gate:
  ΔSharpe ≥ +0.01 AND ΔMaxDD ≤ +0.3pp AND block bootstrap 中位不劣。

用法: .venv/bin/python scripts/_exp_hysteresis_study.py
"""
import contextlib
import inspect
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

import src.backtest as sbt
from src.backtest import run_backtest
from src.strategy import load_config

OUT = PROJ / "output" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)
CFG45 = PROJ / "config" / "strategy_v4_5_pvd.yaml"
REAL_CSV = PROJ / "data" / "all_etfs_nav_latest.csv"

BOOT_N = 120
BOOT_BLOCK = 13
BOOT_SEED = 7711

_ORIG_SRC = inspect.getsource(sbt.run_backtest)

_MARGIN_ANCHOR = "                if config.score_margin > 0 or config.dynamic_margin_sensitivity > 0 or config.snr_adaptive_enabled:"

_SORT_ANCHOR = """        off_scores = [(scores_vec[j], j) for j in off_idx if not np.isnan(scores_vec[j])]
        off_scores.sort(key=lambda x: x[0], reverse=True)"""

_SEL_ANCHOR = """        if eff_softmax_enabled:
            selected_off = [j for _, j in off_scores]
        else:
            selected_off = [j for _, j in off_scores[:eff_top_n]]"""


def build_run_backtest(mode):
    """mode ∈ BASE/M0/H1/H2 → 手术后的 run_backtest。

    H1 的 ε 与 H2 的 band 通过模块全局 _HB_EPS / _SPREAD_BAND 注入 (动态解析)。
    """
    src = _ORIG_SRC
    if mode == "BASE":
        pass
    elif mode == "M0":
        assert _MARGIN_ANCHOR in src, "margin 锚点失效"
        src = src.replace(_MARGIN_ANCHOR, "                if False:  # C2-M0: 隔离对照", 1)
    elif mode == "H1":
        assert _SORT_ANCHOR in src, "sort 锚点失效"
        inject = _SORT_ANCHOR + """
        if _HB_EPS > 0 and last_selected is not None:
            off_scores = [(s + (_HB_EPS if j in last_selected else 0.0), j)
                          for s, j in off_scores]
            off_scores.sort(key=lambda x: x[0], reverse=True)"""
        src = src.replace(_SORT_ANCHOR, inject, 1)
    elif mode == "H2":
        assert _SEL_ANCHOR in src, "selection 锚点失效"
        inject = _SEL_ANCHOR + """
        if _SPREAD_BAND > 0 and last_selected is not None and len(off_scores) >= eff_top_n:
            _spread = off_scores[0][0] - off_scores[-1][0]
            if _spread > _SPREAD_BAND:
                _valid_last = [j for j in last_selected
                               if j in off_idx and not np.isnan(scores_vec[j])]
                if len(_valid_last) == eff_top_n:
                    selected_off = _valid_last"""
        src = src.replace(_SEL_ANCHOR, inject, 1)
    else:
        raise ValueError(mode)
    exec(compile(src, f"<hyst_{mode}>", "exec"), sbt.__dict__)
    return sbt.__dict__["run_backtest"]


VARIANTS = [
    # (key, mode, globals)
    ("BASE", "BASE", {}),
    ("M0_无margin对照", "M0", {}),
    ("H1a_eps0.01", "H1", {"_HB_EPS": 0.01}),
    ("H1b_eps0.02", "H1", {"_HB_EPS": 0.02}),
    ("H2a_band0.35", "H2", {"_SPREAD_BAND": 0.35}),
    ("H2b_band0.50", "H2", {"_SPREAD_BAND": 0.50}),
]


def run_once(rb_fn, cfg, data_path):
    with contextlib.redirect_stdout(io.StringIO()):
        res = rb_fn(cfg, start_date=None, data_path=str(data_path))
    m = res.metrics
    return {"sharpe": float(m["sharpe_ratio"]), "maxdd": float(m["max_drawdown"]),
            "annual": float(m["annual_return"]),
            "turnover": float(res.nav_series["turnover"].mean())}


def apply_globals(g):
    sbt._HB_EPS = g.get("_HB_EPS", 0.0)
    sbt._SPREAD_BAND = g.get("_SPREAD_BAND", 0.0)


def block_bootstrap_paths(weekly_df, n_paths, block, seed):
    rng = np.random.default_rng(seed)
    T = len(weekly_df)
    n_blocks = int(np.ceil(T / block))
    rets = weekly_df.pct_change().values
    for _ in range(n_paths):
        starts = rng.integers(0, T - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
        boot = np.zeros((T, weekly_df.shape[1]))
        boot[0] = weekly_df.values[0]
        for t in range(1, T):
            r = rets[idx[t]]
            boot[t] = boot[t - 1] * (1 + np.where(np.isnan(r), 0.0, r))
        yield pd.DataFrame(boot, index=weekly_df.index, columns=weekly_df.columns)


def main():
    t0 = time.time()
    cfg = load_config(CFG45)

    # 诊断: 进攻评分极差分布 (H2 band 标定依据)
    from src.data_loader import load_nav_data, resample_weekly
    from src.factors import calculate_momentum, calculate_volatility_tapered
    weekly = resample_weekly(load_nav_data(REAL_CSV), anchor=cfg.anchor)
    mom = calculate_momentum(weekly, window=cfg.mom_window)
    vol = calculate_volatility_tapered(weekly, window=cfg.vol_taper_window,
                                       taper=cfg.vol_taper_len)
    off_cols = ["纳指ETF", "中证500ETF", "黄金ETF"]
    score = (cfg.mom_w * mom[off_cols] - cfg.vol_w * vol[off_cols]).dropna()
    spread = score.max(axis=1) - score.min(axis=1)
    diag = {"spread_p50": float(spread.quantile(0.5)),
            "spread_p90": float(spread.quantile(0.9)),
            "spread_p95": float(spread.quantile(0.95)),
            "spread_max": float(spread.max())}
    print(f"[diag] 进攻评分极差: p50={diag['spread_p50']:.3f} "
          f"p90={diag['spread_p90']:.3f} p95={diag['spread_p95']:.3f}")

    print("[realized] 各变体真实历史回测 ...")
    res = {"spread_diag": diag, "variants": {}}
    for key, mode, g in VARIANTS:
        apply_globals(g)
        rb = build_run_backtest(mode)
        r = run_once(rb, cfg, REAL_CSV)
        res["variants"][key] = {"realized": r, "params": g}
        print(f"  {key:<16s}: Sh={r['sharpe']:.4f} DD={r['maxdd']:.2%} "
              f"ann={r['annual']:.2%} turnover={r['turnover']:.4f}", flush=True)

    base_sh = res["variants"]["BASE"]["realized"]["sharpe"]
    base_dd = res["variants"]["BASE"]["realized"]["maxdd"]
    base_to = res["variants"]["BASE"]["realized"]["turnover"]

    print(f"[bootstrap] block={BOOT_BLOCK} n={BOOT_N} ...")
    boot_rows = list(block_bootstrap_paths(weekly, BOOT_N, BOOT_BLOCK, BOOT_SEED))
    for key, mode, g in VARIANTS:
        apply_globals(g)
        rb = build_run_backtest(mode)
        shs = []
        for bi, bdf in enumerate(boot_rows):
            tmp = OUT / f"_c2_boot_{key}_{bi}_{os.getpid()}.csv"
            bdf.to_csv(tmp, encoding="utf-8")
            try:
                shs.append(run_once(rb, cfg, tmp)["sharpe"])
            finally:
                if tmp.exists():
                    os.remove(tmp)
        res["variants"][key]["boot_sharpe_med"] = float(np.median(shs))
        print(f"  {key:<16s}: bootstrap median Sh={np.median(shs):.4f}", flush=True)

    base_med = res["variants"]["BASE"]["boot_sharpe_med"]
    res["gates"] = {}
    print("\n[gates]")
    for key, _, _ in VARIANTS:
        if key == "BASE":
            continue
        v = res["variants"][key]
        d_sh = v["realized"]["sharpe"] - base_sh
        d_dd = (v["realized"]["maxdd"] - base_dd) * 100
        d_to = v["realized"]["turnover"] - base_to
        boot_ok = v["boot_sharpe_med"] >= base_med - 1e-4
        g1, g2 = d_sh >= 0.01, d_dd <= 0.3
        verdict = "PASS" if (g1 and g2 and boot_ok) else "NO-GO"
        res["gates"][key] = {"d_sharpe": d_sh, "d_maxdd_pp": d_dd,
                             "d_turnover": d_to, "boot_not_worse": bool(boot_ok),
                             "verdict": verdict}
        note = "(隔离对照)" if key.startswith("M0") else ""
        print(f"  {key:<16s}: ΔSharpe={d_sh:+.4f} ΔMaxDD={d_dd:+.2f}pp "
              f"Δturnover={d_to:+.4f} boot={'✓' if boot_ok else '✗'} "
              f"→ {verdict} {note}")

    out_json = OUT / "exp_hysteresis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[save] {out_json}")
    render_md(res)
    print(f"DONE in {(time.time()-t0)/60:.1f} min")


def render_md(res):
    L = []
    L.append("# 实验: C2 换手抑制 / 滞回带 (holding-bias) 研究\n")
    L.append(f"> {pd.Timestamp.today().date()} | 基座 v4.5-pvd | bootstrap block=13 "
             f"n={BOOT_N} | 脚本 `scripts/_exp_hysteresis_study.py` | 零 src/ 改动\n")
    d = res["spread_diag"]
    L.append("## 1. 诊断与设计\n")
    L.append(f"进攻评分极差分布 (标定 H2 band): p50={d['spread_p50']:.3f}, "
             f"p90={d['spread_p90']:.3f}, p95={d['spread_p95']:.3f}, max={d['spread_max']:.3f}\n")
    L.append("- M0: 关闭 score margin (隔离对照, 量化现有换手抑制贡献)\n"
             "- H1: 现持仓评分 +ε 后选 top-2 (holding bias); H2: 评分极差超 band 时冻结上周选择 "
             "(聚宽'枪打出头鸟'/信号分歧保持原仓位 机制)\n")
    L.append("## 2. Realized\n")
    L.append("| 变体 | Sharpe | MaxDD | 年化 | 平均换手 |")
    L.append("|---|---|---|---|---|")
    for key, v in res["variants"].items():
        r = v["realized"]
        L.append(f"| {key} | {r['sharpe']:.4f} | {r['maxdd']:.2%} | "
                 f"{r['annual']:.2%} | {r['turnover']:.4f} |")
    L.append("\n## 3. E2 gate (净 Sharpe 口径; ΔSharpe≥+0.01 AND ΔMaxDD≤+0.3pp AND bootstrap 中位不劣)\n")
    L.append("| 变体 | ΔSharpe | ΔMaxDD(pp) | Δ换手 | bootstrap | 判定 |")
    L.append("|---|---|---|---|---|---|")
    for key, g in res["gates"].items():
        L.append(f"| {key} | {g['d_sharpe']:+.4f} | {g['d_maxdd_pp']:+.2f} | "
                 f"{g['d_turnover']:+.4f} | {'✓' if g['boot_not_worse'] else '✗'} | "
                 f"**{g['verdict']}** |")
    n_pass = sum(1 for g in res["gates"].items()
                 if not g[0].startswith("M0") and g[1]["verdict"] == "PASS")
    n_var = sum(1 for k in res["gates"] if not k.startswith("M0"))
    L.append(f"\n**结论**: {n_pass}/{n_var} 个滞回带变体通过 E2 gate。"
             + ("" if n_pass else
                " 现有 score margin + rebalance threshold + PVD gap 条件的换手抑制组合"
                " 已接近成本侧最优, 额外滞回带未能带来净 Sharpe 增量, C2 方向 NO-GO 归档"
                " (机制清单留档供未来资产池/费率变化时复评)。"))
    out_md = OUT / "exp_hysteresis.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[save] {out_md}")


if __name__ == "__main__":
    main()
