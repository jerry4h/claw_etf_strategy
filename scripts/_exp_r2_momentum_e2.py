#!/usr/bin/env python3
"""R²动量→E2: 回归动量替换变体 (v4.6 候选特性门禁)。

依据 E1 (exp_r2_momentum_e1.md): R2M-A/B 截面 IC 双过门禁 (+0.0385/+0.0375,
t≈1.7), 优于生产 mom6 自身 (+0.0139 未过门); corr(mom6, R2M)≈0.60 → 替换变体。

E2 口径 (计划锁定):
  - **替换**而非叠加 (窄截面线性叠加 MaxDD +14pp 教训)
  - 单位匹配: 社区原版"年化斜率×R²"在本策略 score=1.1×mom−1.1×vol 中量纲
    失衡 (年化 ±50% vs vol 15%), 改用 6 周等效斜率 × R²: (exp(slope×6)−1)×R²,
    与 mom6 (6周收益) 同量纲, R² 保留趋势质量加权
  - 变体: R2U-OLS (无权重) / R2U-WLS (线性递增权, 社区原版加权)

实现: monkeypatch sbt.compute_all_factors (替换 momentum 输出), 基座 v4.5-pvd,
零 src/ 改动。注意: momentum 替换会连带影响 PVD 的 top-2 gap 激活条件
(结构性耦合, 属替换语义的一部分)。

门禁: ΔSharpe ≥ +0.01 AND ΔMaxDD ≤ +0.3pp AND block bootstrap 200 中位不劣;
额外记录换手率 (R² 价值预期在排名稳定性)。

用法: .venv/bin/python scripts/_exp_r2_momentum_e2.py
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
from src.data_loader import load_nav_data, resample_weekly
from src.strategy import load_config

OUT = PROJ / "output" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)
CFG45 = PROJ / "config" / "strategy_v4_5_pvd.yaml"
REAL_CSV = PROJ / "data" / "all_etfs_nav_latest.csv"

WINDOW = 6
WEEK_FACTOR = 6.0     # 6 周等效斜率 (单位匹配 mom6)
BOOT_N = 200
BOOT_BLOCK = 13
BOOT_SEED = 7744


def r2_momentum(weekly: pd.DataFrame, window: int, weighted: bool) -> pd.DataFrame:
    """滚动回归动量: (exp(slope×6)−1) × R², 与 mom6 同量纲。无前视。"""
    logp = np.log(weekly.values.astype(float))
    n, k = logp.shape
    out = np.full((n, k), np.nan)
    x = np.arange(window, dtype=float)
    w = (x + 1.0) if weighted else np.ones(window)
    w = w / w.sum()
    xw_mean = float(np.sum(w * x))
    varx = float(np.sum(w * (x - xw_mean) ** 2))
    for t in range(window - 1, n):
        for j in range(k):
            y = logp[t - window + 1:t + 1, j]
            if np.isnan(y).any():
                continue
            yw_mean = float(np.sum(w * y))
            slope = float(np.sum(w * (x - xw_mean) * (y - yw_mean))) / varx
            yhat = slope * (x - xw_mean) + yw_mean
            ss_res = float(np.sum(w * (y - yhat) ** 2))
            ss_tot = float(np.sum(w * (y - yw_mean) ** 2))
            r2 = max(1.0 - ss_res / ss_tot, 0.0) if ss_tot > 0 else 0.0
            out[t, j] = (np.exp(slope * WEEK_FACTOR) - 1.0) * r2
    return pd.DataFrame(out, index=weekly.index, columns=weekly.columns)


_ORIG_CAF = sbt.compute_all_factors
_MODE = {"estimator": None}   # None=mom6 原样; 'ols'/'wls' 替换 momentum


def patched_compute_all_factors(weekly_nav, pe_df=None, config=None, weekly_vol=None):
    factors = _ORIG_CAF(weekly_nav, pe_df, config, weekly_vol=weekly_vol)
    est = _MODE["estimator"]
    if est in ("ols", "wls"):
        factors["momentum"] = r2_momentum(weekly_nav, WINDOW, weighted=(est == "wls"))
    return factors


def run_once(cfg, data_path):
    with contextlib.redirect_stdout(io.StringIO()):
        res = run_backtest(cfg, start_date=None, data_path=str(data_path))
    m = res.metrics
    return {"sharpe": float(m["sharpe_ratio"]), "maxdd": float(m["max_drawdown"]),
            "annual": float(m["annual_return"]),
            "turnover": float(res.nav_series["turnover"].mean())}


def block_bootstrap_paths(weekly_df, n_paths, block, seed):
    rng = np.random.default_rng(seed)
    T = len(weekly_df)
    n_blocks = int(np.ceil(T / block))
    rets_v = weekly_df.pct_change().values
    for _ in range(n_paths):
        starts = rng.integers(0, T - block, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
        boot = np.zeros((T, weekly_df.shape[1]))
        boot[0] = weekly_df.values[0]
        for t in range(1, T):
            r = rets_v[idx[t]]
            boot[t] = boot[t - 1] * (1 + np.where(np.isnan(r), 0.0, r))
        yield pd.DataFrame(boot, index=weekly_df.index, columns=weekly_df.columns)


VARIANTS = [("BASE", None), ("R2U-OLS", "ols"), ("R2U-WLS", "wls")]


def main():
    t0 = time.time()
    cfg = load_config(CFG45)
    sbt.compute_all_factors = patched_compute_all_factors

    print("[realized] 各变体真实历史回测 ...")
    res = {"window": WINDOW, "week_factor": WEEK_FACTOR, "variants": {}}
    try:
        for key, est in VARIANTS:
            _MODE["estimator"] = est
            r = run_once(cfg, REAL_CSV)
            res["variants"][key] = {"estimator": est, "realized": r}
            print(f"  {key:<9s}: Sh={r['sharpe']:.4f} DD={r['maxdd']:.2%} "
                  f"ann={r['annual']:.2%} turnover={r['turnover']:.4f}", flush=True)

        base_sh = res["variants"]["BASE"]["realized"]["sharpe"]
        base_dd = res["variants"]["BASE"]["realized"]["maxdd"]

        print(f"[bootstrap] block={BOOT_BLOCK} n={BOOT_N} ...")
        weekly = resample_weekly(load_nav_data(REAL_CSV), anchor=cfg.anchor)
        boot_rows = list(block_bootstrap_paths(weekly, BOOT_N, BOOT_BLOCK, BOOT_SEED))
        for key, est in VARIANTS:
            _MODE["estimator"] = est
            shs = []
            for bi, bdf in enumerate(boot_rows):
                tmp = OUT / f"_r2_e2_{key}_{bi}_{os.getpid()}.csv"
                bdf.to_csv(tmp, encoding="utf-8")
                try:
                    shs.append(run_once(cfg, tmp)["sharpe"])
                finally:
                    if tmp.exists():
                        os.remove(tmp)
            res["variants"][key]["boot_sharpe_med"] = float(np.median(shs))
            res["variants"][key]["boot_sharpe_p10"] = float(np.percentile(shs, 10))
            print(f"  {key:<9s}: bootstrap median Sh={np.median(shs):.4f} "
                  f"P10={np.percentile(shs,10):.4f}", flush=True)
    finally:
        sbt.compute_all_factors = _ORIG_CAF

    base_med = res["variants"]["BASE"]["boot_sharpe_med"]
    base_to = res["variants"]["BASE"]["realized"]["turnover"]
    res["gates"] = {}
    print("\n[gates] ΔSharpe≥+0.01 AND ΔMaxDD≤+0.3pp AND bootstrap 中位不劣")
    for key, est in VARIANTS:
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
        print(f"  {key:<9s}: ΔSharpe={d_sh:+.4f} ({'✓' if g1 else '✗'}) "
              f"ΔMaxDD={d_dd:+.2f}pp ({'✓' if g2 else '✗'}) Δturnover={d_to:+.4f} "
              f"boot={'✓' if boot_ok else '✗'} → {verdict}")

    out_json = OUT / "exp_r2_momentum_e2.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[save] {out_json}")
    render_md(res)
    print(f"DONE in {(time.time()-t0)/60:.1f} min")


def render_md(res):
    L = []
    L.append("# 实验: R²动量→E2 替换变体 (v4.6 候选门禁)\n")
    L.append(f"> {pd.Timestamp.today().date()} | 基座 v4.5-pvd | window={res['window']} 周, "
             f"单位匹配 (exp(slope×{res['week_factor']:.0f})−1)×R² | "
             f"bootstrap block=13 n={BOOT_N} | 零 src/ 改动\n")
    L.append("替换语义: momentum 因子整体替换 (非叠加); PVD 的 top-2 gap 激活条件随 "
             "momentum 口径联动 (结构性耦合, 属替换语义的一部分)。\n")
    L.append("## Realized\n")
    L.append("| 变体 | Sharpe | MaxDD | 年化 | 换手 |")
    L.append("|---|---|---|---|---|")
    for key, v in res["variants"].items():
        r = v["realized"]
        L.append(f"| {key} | {r['sharpe']:.4f} | {r['maxdd']:.2%} | "
                 f"{r['annual']:.2%} | {r['turnover']:.4f} |")
    L.append("\n## Block bootstrap (中位 Sharpe)\n")
    L.append("| 变体 | 中位 | P10 |")
    L.append("|---|---|---|")
    for key, v in res["variants"].items():
        L.append(f"| {key} | {v['boot_sharpe_med']:.4f} | {v['boot_sharpe_p10']:.4f} |")
    L.append("\n## E2 门禁判定\n")
    L.append("| 变体 | ΔSharpe | ΔMaxDD(pp) | Δ换手 | bootstrap | 判定 |")
    L.append("|---|---|---|---|---|---|")
    for key, g in res["gates"].items():
        L.append(f"| {key} | {g['d_sharpe']:+.4f} | {g['d_maxdd_pp']:+.2f} | "
                 f"{g['d_turnover']:+.4f} | {'✓' if g['boot_not_worse'] else '✗'} | "
                 f"**{g['verdict']}** |")
    n_pass = sum(1 for g in res["gates"].values() if g["verdict"] == "PASS")
    notes = []
    for key, g in res["gates"].items():
        if g["verdict"] == "PASS" and g["d_turnover"] > 0.01:
            notes.append(f"{key} 换手上升 {g['d_turnover']:+.4f}, 成本敏感性需关注")
    L.append(f"\n**结论**: {n_pass}/{len(res['gates'])} 变体通过 E2 门禁 → "
             + ("R² 动量替换进入 v4.6 E3 集成。" if n_pass
                else "R² 动量替换 NO-GO, 不进 v4.6。E1 的截面 IC 优势未能转化为 "
                "realized 净 Sharpe 改善 (先验提示的'噪声抑制而非新信息'未兑现为组合收益), "
                "mom6 维持生产口径。"))
    for nt in notes:
        L.append(f"- 成本提示: {nt}")
    out_md = OUT / "exp_r2_momentum_e2.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[save] {out_md}")


if __name__ == "__main__":
    main()
