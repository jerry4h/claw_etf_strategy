#!/usr/bin/env python3
"""C1→E2: PE 估值分位防御调制 (v4.6 候选特性门禁)。

依据 C1 E1 结论 (exp_pe_percentile_e1.md): h13 等权组合 IC=-0.109 (t=-2.82)
过门禁——高估值 → 中期收益弱。E2 只测计划锁定的单一形态 (防过拟合面扩张):

    pe_pct(t-1) > 0.9 时  def_ratio = min(def_ratio + δ, max_def),  δ ∈ {0.05, 0.10}

口径: PE 分位复用生产 calculate_pe_percentile(5年窗) + shift(1) 防前视;
对齐修复 (C1 发现): PE CSV 周一日期 vs NAV 周五标签, ffill asof 对齐。
注入位置: Layer3.5/M3 boost 之后、止损之前 (估值是慢变量, 调制基础防御倾向)。

实现: 源码手术 monkeypatch (B1/B2 同款模式), 基座 v4.5-pvd, 零 src/ 改动。
门禁: ΔSharpe ≥ +0.01 AND ΔMaxDD ≤ +0.3pp AND block bootstrap 200 中位不劣。

用法: .venv/bin/python scripts/_exp_pe_defense_e2.py
"""
import contextlib
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
from src.data_loader import load_nav_data, resample_weekly, load_pe_percentile
from src.factors import calculate_pe_percentile
from src.strategy import load_config

OUT = PROJ / "output" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)
CFG45 = PROJ / "config" / "strategy_v4_5_pvd.yaml"
REAL_CSV = PROJ / "data" / "all_etfs_nav_latest.csv"
PE_PATH = PROJ / "data" / "300etf_pe_percentile_weekly.csv"

PE_THR = 0.90
DELTAS = {"D05": 0.05, "D10": 0.10}
BOOT_N = 200
BOOT_BLOCK = 13
BOOT_SEED = 7733

_ORIG_SRC = None  # main 中捕获

_BOOST_ANCHOR = """        # --- M3: 中证500 vol 危机加成 ---
        ashare_boost = compute_ashare_vol_boost(vol_values, i, CSI500_IDX, config)
        if ashare_boost > 0:
            def_ratio = min(def_ratio + ashare_boost, 1.0)"""


def build_pe_series(cfg):
    """生产口径 PE 分位 + shift(1) + ffill asof 对齐到周频标签 (周五)。"""
    nav = load_nav_data(REAL_CSV)
    weekly_idx = resample_weekly(nav, anchor=cfg.anchor).index
    pe_raw = load_pe_percentile(PE_PATH)
    pe = calculate_pe_percentile(pe_raw, window_years=5).shift(1)["pe_percentile"]
    return pe.reindex(weekly_idx, method="ffill")


def build_run_backtest(delta: float):
    """delta>0 → 手术注入 PE 调制; delta=0 → 原样 (BASE)。"""
    src = _ORIG_SRC
    if delta > 0:
        inject = _BOOST_ANCHOR + f"""

        # --- PE 估值防御调制 (E2 实验注入, 慢变量调制基础防御) ---
        _pe_v = _PE_LOOKUP.get(date, np.nan)
        if not np.isnan(_pe_v) and _pe_v > {PE_THR}:
            def_ratio = min(def_ratio + {delta}, config.max_def)"""
        assert _BOOST_ANCHOR in src, "手术锚点失效 (src/backtest.py 已漂移)"
        src = src.replace(_BOOST_ANCHOR, inject, 1)
    exec(compile(src, f"<pe_e2_{delta}>", "exec"), sbt.__dict__)
    return sbt.__dict__["run_backtest"]


def run_once(rb_fn, cfg, data_path):
    with contextlib.redirect_stdout(io.StringIO()):
        res = rb_fn(cfg, start_date=None, data_path=str(data_path))
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


def main():
    global _ORIG_SRC
    t0 = time.time()
    _ORIG_SRC = __import__("inspect").getsource(sbt.run_backtest)
    cfg = load_config(CFG45)

    pe_series = build_pe_series(cfg)
    sbt._PE_LOOKUP = {d: v for d, v in pe_series.items() if not np.isnan(v)}
    n_hi = int((pe_series > PE_THR).sum())
    print(f"[data] PE 分位对齐 {pe_series.notna().sum()} 周, "
          f"pe>{PE_THR} 共 {n_hi} 周 ({n_hi/pe_series.notna().sum():.1%})")

    print("[realized] 各变体真实历史回测 ...")
    res = {"pe_threshold": PE_THR, "variants": {}}
    for key, delta in [("BASE", 0.0)] + [(k, v) for k, v in DELTAS.items()]:
        rb = build_run_backtest(delta)
        r = run_once(rb, cfg, REAL_CSV)
        res["variants"][key] = {"delta": delta, "realized": r}
        print(f"  {key}: Sh={r['sharpe']:.4f} DD={r['maxdd']:.2%} "
              f"ann={r['annual']:.2%} turnover={r['turnover']:.4f}", flush=True)

    base_sh = res["variants"]["BASE"]["realized"]["sharpe"]
    base_dd = res["variants"]["BASE"]["realized"]["maxdd"]

    print(f"[bootstrap] block={BOOT_BLOCK} n={BOOT_N} ...")
    weekly = resample_weekly(load_nav_data(REAL_CSV), anchor=cfg.anchor)
    boot_rows = list(block_bootstrap_paths(weekly, BOOT_N, BOOT_BLOCK, BOOT_SEED))
    for key, delta in [("BASE", 0.0)] + [(k, v) for k, v in DELTAS.items()]:
        rb = build_run_backtest(delta)
        shs = []
        for bi, bdf in enumerate(boot_rows):
            tmp = OUT / f"_pe_e2_{key}_{bi}_{os.getpid()}.csv"
            bdf.to_csv(tmp, encoding="utf-8")
            try:
                shs.append(run_once(rb, cfg, tmp)["sharpe"])
            finally:
                if tmp.exists():
                    os.remove(tmp)
        res["variants"][key]["boot_sharpe_med"] = float(np.median(shs))
        res["variants"][key]["boot_sharpe_p10"] = float(np.percentile(shs, 10))
        print(f"  {key}: bootstrap median Sh={np.median(shs):.4f} "
              f"P10={np.percentile(shs,10):.4f}", flush=True)

    base_med = res["variants"]["BASE"]["boot_sharpe_med"]
    res["gates"] = {}
    print("\n[gates] ΔSharpe≥+0.01 AND ΔMaxDD≤+0.3pp AND bootstrap 中位不劣")
    for key in DELTAS:
        v = res["variants"][key]
        d_sh = v["realized"]["sharpe"] - base_sh
        d_dd = (v["realized"]["maxdd"] - base_dd) * 100
        boot_ok = v["boot_sharpe_med"] >= base_med - 1e-4
        g1, g2 = d_sh >= 0.01, d_dd <= 0.3
        verdict = "PASS" if (g1 and g2 and boot_ok) else "NO-GO"
        res["gates"][key] = {"d_sharpe": d_sh, "d_maxdd_pp": d_dd,
                             "boot_not_worse": bool(boot_ok), "verdict": verdict}
        print(f"  {key}: ΔSharpe={d_sh:+.4f} ({'✓' if g1 else '✗'}) "
              f"ΔMaxDD={d_dd:+.2f}pp ({'✓' if g2 else '✗'}) "
              f"boot={'✓' if boot_ok else '✗'} → {verdict}")

    out_json = OUT / "exp_pe_defense_e2.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[save] {out_json}")
    render_md(res, n_hi, pe_series)
    print(f"DONE in {(time.time()-t0)/60:.1f} min")


def render_md(res, n_hi, pe_series):
    L = []
    L.append("# 实验: C1→E2 PE 估值分位防御调制 (v4.6 候选门禁)\n")
    L.append(f"> {pd.Timestamp.today().date()} | 基座 v4.5-pvd | 形态锁定: "
             f"pe_pct(t-1)>{res['pe_threshold']} → def_ratio +δ (封顶 max_def) | "
             f"bootstrap block=13 n={BOOT_N} | 零 src/ 改动\n")
    L.append(f"PE 数据: 对齐 {pe_series.notna().sum()} 周, pe>{res['pe_threshold']} "
             f"共 {n_hi} 周 ({n_hi/max(1,pe_series.notna().sum()):.1%}); "
             "口径 calculate_pe_percentile(5y) + shift(1) + ffill asof (周一PE→周五标签)。\n")
    L.append("## Realized\n")
    L.append("| 变体 | δ | Sharpe | MaxDD | 年化 | 换手 |")
    L.append("|---|---|---|---|---|---|")
    for key, v in res["variants"].items():
        r = v["realized"]
        L.append(f"| {key} | {v['delta']} | {r['sharpe']:.4f} | {r['maxdd']:.2%} | "
                 f"{r['annual']:.2%} | {r['turnover']:.4f} |")
    L.append("\n## Block bootstrap (中位 Sharpe)\n")
    L.append("| 变体 | 中位 | P10 |")
    L.append("|---|---|---|")
    for key, v in res["variants"].items():
        L.append(f"| {key} | {v['boot_sharpe_med']:.4f} | {v['boot_sharpe_p10']:.4f} |")
    L.append("\n## E2 门禁判定\n")
    L.append("| 变体 | ΔSharpe | ΔMaxDD(pp) | bootstrap | 判定 |")
    L.append("|---|---|---|---|---|")
    for key, g in res["gates"].items():
        L.append(f"| {key} | {g['d_sharpe']:+.4f} | {g['d_maxdd_pp']:+.2f} | "
                 f"{'✓' if g['boot_not_worse'] else '✗'} | **{g['verdict']}** |")
    n_pass = sum(1 for g in res["gates"].values() if g["verdict"] == "PASS")
    L.append(f"\n**结论**: {n_pass}/{len(res['gates'])} 变体通过 E2 门禁 → "
             + ("PE 防御调制进入 v4.6 E3 集成。" if n_pass
                else "PE 防御调制 NO-GO, 不进 v4.6 (E1 的 h13 预测力未能转化为 "
                "realized 改善——慢变量 × 周频决策 × 高估周多防御的传导链弱; "
                "与'全球宏观共同因子'归因警告一致)。PE 数据保留管线内不接入决策。"))
    out_md = OUT / "exp_pe_defense_e2.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[save] {out_md}")


if __name__ == "__main__":
    main()
