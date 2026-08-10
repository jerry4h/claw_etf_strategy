#!/usr/bin/env python3
"""B2 探针: PVD quantile 门控单点验证 (PVD 深化三方向中仅此一项)。

依据 docs/v4_5_pvd_factor_closure.md §6: quantile 门控复用现有 expanding
门限基建 (engine_core.compute_pvd_vol_gates 同款模式), 最便宜。窗口自适应与
crisis_corr 联动边际预期低且各增一个过拟合面, 本轮不做。

设计: 用 PVD 自身 expanding 历史分位数门控注入强度。定义 PVD 截面离散度
disp_i = std(pvd[i, off_idx])——离散度高 = PVD 能区分进攻资产 (tiebreaker
有意义); 离散度≈0 = PVD 无排序信息。disp 的 expanding 分位 q_i (无前视) 作门控。

变体 (基座 = v4.5-pvd 生产 config, monkeypatch PVD 注入块, 零 src/ 改动):
  BASE : 生产现状 (vol∈[p25,p75] AND top2 mom gap<0.05 → score += pvd_w×pvd)
  B2-V1: 叠加门 — 现状双门 AND q_i ≥ 0.5 (PVD 离散度处自身历史上半区才注入)
  B2-V2: 替换门 — vol 门 AND q_i ≥ 0.5 (去掉 gap 硬门, 改由 PVD 自身强度门控)
  B2-V3: 权重缩放 — 现状双门, 注入权重 pvd_w × q_i (强信号加权大, 弱信号加权小)

E2 gate: ΔSharpe ≥ +0.01 AND ΔMaxDD ≤ +0.3pp AND block bootstrap 中位不劣。

用法: .venv/bin/python scripts/_exp_pvd_quantile_gate.py
"""
import contextlib
import io
import inspect
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

Q_GATE = 0.50          # quantile 门控分位
BOOT_N = 120           # block bootstrap 路径数
BOOT_BLOCK = 13        # 块长(周), 与 robustness_joint 一致
BOOT_SEED = 7700

_ORIG_SRC = inspect.getsource(sbt.run_backtest)

# 预计算锚点: 在 vol gate 预计算后追加 PVD 离散度 + expanding 分位
_PRECOMPUTE_ANCHOR = """        _pvd_gate_lo, _pvd_gate_hi = compute_pvd_vol_gates(
            vol_values[:, NASDAQ_IDX], config.pvd_vol_pct_range)"""

_PRECOMPUTE_INJECT = """        _pvd_gate_lo, _pvd_gate_hi = compute_pvd_vol_gates(
            vol_values[:, NASDAQ_IDX], config.pvd_vol_pct_range)
        # B2: PVD 截面离散度 + expanding 分位 (无前视)
        _n_wk = _pvd_values.shape[0]
        _pvd_disp = np.full(_n_wk, np.nan)
        for _i in range(_n_wk):
            _row = _pvd_values[_i, off_idx]
            _row = _row[~np.isnan(_row)]
            if len(_row) >= 2:
                _pvd_disp[_i] = float(np.std(_row, ddof=0))
        _pvd_disp_q = np.full(_n_wk, 0.5)
        for _i in range(1, _n_wk):
            if np.isnan(_pvd_disp[_i]):
                _pvd_disp_q[_i] = _pvd_disp_q[_i - 1]
                continue
            _hist = _pvd_disp[:_i]
            _hist = _hist[~np.isnan(_hist)]
            if len(_hist) >= 10:
                _pvd_disp_q[_i] = float(np.mean(_hist <= _pvd_disp[_i]))
            else:
                _pvd_disp_q[_i] = 0.5"""

_INJECT_ANCHOR = """        if _pvd_active:
            _nv = vol_values[i, NASDAQ_IDX]
            if not np.isnan(_nv) and _pvd_gate_lo[i] <= _nv <= _pvd_gate_hi[i]:
                # 检查 top-2 momentum gap (与 E2b 一致：使用原始动量值，遍历所有 ETF)
                _valid_mom = [(mom_values[i, j], j) for j in range(n_etfs)
                              if not np.isnan(mom_values[i, j]) and mom_values[i, j] > -np.inf]
                if len(_valid_mom) >= 2:
                    _valid_mom.sort(key=lambda x: x[0], reverse=True)
                    _gap = _valid_mom[0][0] - _valid_mom[1][0]
                    if _gap < config.pvd_score_gap_threshold:
                        for j in off_idx:
                            if not np.isnan(_pvd_values[i, j]) and scores_vec[j] > -np.inf:
                                scores_vec[j] += config.pvd_w * _pvd_values[i, j]"""


def _inject_block(mode):
    """mode ∈ {'V1','V2','V3'} → 替换后的 PVD 注入块。"""
    if mode == "V1":  # 叠加门: 双门 AND q≥Q_GATE
        return f"""        if _pvd_active:
            _nv = vol_values[i, NASDAQ_IDX]
            if not np.isnan(_nv) and _pvd_gate_lo[i] <= _nv <= _pvd_gate_hi[i]:
                _valid_mom = [(mom_values[i, j], j) for j in range(n_etfs)
                              if not np.isnan(mom_values[i, j]) and mom_values[i, j] > -np.inf]
                if len(_valid_mom) >= 2:
                    _valid_mom.sort(key=lambda x: x[0], reverse=True)
                    _gap = _valid_mom[0][0] - _valid_mom[1][0]
                    if _gap < config.pvd_score_gap_threshold and _pvd_disp_q[i] >= {Q_GATE}:
                        for j in off_idx:
                            if not np.isnan(_pvd_values[i, j]) and scores_vec[j] > -np.inf:
                                scores_vec[j] += config.pvd_w * _pvd_values[i, j]"""
    if mode == "V2":  # 替换门: vol 门 AND q≥Q_GATE (去 gap 硬门)
        return f"""        if _pvd_active:
            _nv = vol_values[i, NASDAQ_IDX]
            if not np.isnan(_nv) and _pvd_gate_lo[i] <= _nv <= _pvd_gate_hi[i]:
                if _pvd_disp_q[i] >= {Q_GATE}:
                    for j in off_idx:
                        if not np.isnan(_pvd_values[i, j]) and scores_vec[j] > -np.inf:
                            scores_vec[j] += config.pvd_w * _pvd_values[i, j]"""
    if mode == "V3":  # 权重缩放: 双门, 权重 pvd_w × q
        return f"""        if _pvd_active:
            _nv = vol_values[i, NASDAQ_IDX]
            if not np.isnan(_nv) and _pvd_gate_lo[i] <= _nv <= _pvd_gate_hi[i]:
                _valid_mom = [(mom_values[i, j], j) for j in range(n_etfs)
                              if not np.isnan(mom_values[i, j]) and mom_values[i, j] > -np.inf]
                if len(_valid_mom) >= 2:
                    _valid_mom.sort(key=lambda x: x[0], reverse=True)
                    _gap = _valid_mom[0][0] - _valid_mom[1][0]
                    if _gap < config.pvd_score_gap_threshold:
                        for j in off_idx:
                            if not np.isnan(_pvd_values[i, j]) and scores_vec[j] > -np.inf:
                                scores_vec[j] += config.pvd_w * _pvd_disp_q[i] * _pvd_values[i, j]"""
    raise ValueError(mode)


def build_run_backtest(mode):
    """mode='BASE' → 原样; 否则手术 (预计算 + 注入块)。"""
    src = _ORIG_SRC
    if mode == "BASE":
        pass
    else:
        assert _PRECOMPUTE_ANCHOR in src and _INJECT_ANCHOR in src, "手术锚点失效 (src 已漂移)"
        src = src.replace(_PRECOMPUTE_ANCHOR, _PRECOMPUTE_INJECT, 1)
        src = src.replace(_INJECT_ANCHOR, _inject_block(mode), 1)
    exec(compile(src, f"<pvd_q_{mode}>", "exec"), sbt.__dict__)
    return sbt.__dict__["run_backtest"]


def run_once(rb_fn, cfg, data_path):
    with contextlib.redirect_stdout(io.StringIO()):
        res = rb_fn(cfg, start_date=None, data_path=str(data_path))
    m = res.metrics
    return {"sharpe": float(m["sharpe_ratio"]), "maxdd": float(m["max_drawdown"]),
            "annual": float(m["annual_return"]),
            "turnover": float(res.nav_series["turnover"].mean())}


def block_bootstrap_paths(weekly_df, n_paths, block, seed):
    """对周频 NAV 做 moving-block bootstrap, yield 重采样 DataFrame。"""
    rng = np.random.default_rng(seed)
    T = len(weekly_df)
    n_blocks = int(np.ceil(T / block))
    for _ in range(n_paths):
        starts = rng.integers(0, T - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
        # 用价格比重建连续净值 (避免块拼接处的跳变)
        rets = weekly_df.pct_change().values
        boot = np.zeros((T, weekly_df.shape[1]))
        boot[0] = weekly_df.values[0]
        for t in range(1, T):
            boot[t] = boot[t - 1] * (1 + rets[idx[t]] if not np.any(np.isnan(rets[idx[t]])) else 1.0)
        yield pd.DataFrame(boot, index=weekly_df.index, columns=weekly_df.columns)


def main():
    t0 = time.time()
    cfg = load_config(CFG45)
    assert cfg.pvd_enabled, "B2 基座须 pvd_enabled"

    print("[realized] 各变体真实历史回测 ...")
    res = {"q_gate": Q_GATE, "variants": {}}
    for mode in ("BASE", "V1", "V2", "V3"):
        rb = build_run_backtest(mode)
        r = run_once(rb, cfg, REAL_CSV)
        res["variants"][mode] = {"realized": r}
        print(f"  {mode}: Sh={r['sharpe']:.4f} DD={r['maxdd']:.2%} "
              f"ann={r['annual']:.2%} turnover={r['turnover']:.4f}", flush=True)

    base_sh = res["variants"]["BASE"]["realized"]["sharpe"]
    base_dd = res["variants"]["BASE"]["realized"]["maxdd"]

    print(f"[bootstrap] block={BOOT_BLOCK} n={BOOT_N} ...")
    from src.data_loader import load_nav_data, resample_weekly
    weekly = resample_weekly(load_nav_data(REAL_CSV), anchor=cfg.anchor)
    boot_rows = list(block_bootstrap_paths(weekly, BOOT_N, BOOT_BLOCK, BOOT_SEED))
    for mode in ("BASE", "V1", "V2", "V3"):
        rb = build_run_backtest(mode)
        shs = []
        for bi, bdf in enumerate(boot_rows):
            tmp = OUT / f"_b2_boot_{mode}_{bi}_{os.getpid()}.csv"
            bdf.to_csv(tmp, encoding="utf-8")
            try:
                r = run_once(rb, cfg, tmp)
                shs.append(r["sharpe"])
            finally:
                if tmp.exists():
                    os.remove(tmp)
        res["variants"][mode]["boot_sharpe_med"] = float(np.median(shs))
        res["variants"][mode]["boot_sharpe_p10"] = float(np.percentile(shs, 10))
        print(f"  {mode}: bootstrap median Sh={np.median(shs):.4f} "
              f"P10={np.percentile(shs,10):.4f}", flush=True)

    base_med = res["variants"]["BASE"]["boot_sharpe_med"]
    res["gates"] = {}
    print("\n[gates]")
    for mode in ("V1", "V2", "V3"):
        v = res["variants"][mode]
        d_sh = v["realized"]["sharpe"] - base_sh
        d_dd = (v["realized"]["maxdd"] - base_dd) * 100
        boot_ok = v["boot_sharpe_med"] >= base_med - 1e-4
        g1 = d_sh >= 0.01
        g2 = d_dd <= 0.3
        verdict = "PASS" if (g1 and g2 and boot_ok) else "NO-GO"
        res["gates"][mode] = {"d_sharpe": d_sh, "d_maxdd_pp": d_dd,
                              "boot_not_worse": bool(boot_ok), "verdict": verdict}
        print(f"  {mode}: ΔSharpe={d_sh:+.4f} ({'✓' if g1 else '✗'}) "
              f"ΔMaxDD={d_dd:+.2f}pp ({'✓' if g2 else '✗'}) "
              f"bootstrap中位不劣={'✓' if boot_ok else '✗'} → {verdict}")

    out_json = OUT / "exp_pvd_quantile_gate.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[save] {out_json}")
    render_md(res)
    print(f"DONE in {(time.time()-t0)/60:.1f} min")


def render_md(res):
    L = []
    L.append("# 实验: B2 PVD quantile 门控单点验证\n")
    L.append(f"> {pd.Timestamp.today().date()} | 基座 v4.5-pvd 生产 config | q_gate={res['q_gate']} | "
             f"bootstrap block=13 n={BOOT_N} | 脚本 `scripts/_exp_pvd_quantile_gate.py` | 零 src/ 改动\n")
    L.append("## 1. 设计\n")
    L.append("PVD 截面离散度 disp_i = std(pvd[i, off_idx]); expanding 分位 q_i (无前视)。"
             "离散度高 = PVD 能区分进攻资产, tiebreaker 有意义; 离散度≈0 = 无排序信息。\n")
    L.append("- BASE: 生产现状 (vol∈[p25,p75] AND top2 gap<0.05)\n"
             "- V1 叠加门: 现状双门 AND q≥0.5 | V2 替换门: vol 门 AND q≥0.5 (去 gap) | "
             "V3 权重缩放: 双门, 权重 pvd_w×q\n")
    L.append("## 2. Realized\n")
    L.append("| 变体 | Sharpe | MaxDD | 年化 | 换手 |")
    L.append("|---|---|---|---|---|")
    for mode, v in res["variants"].items():
        r = v["realized"]
        L.append(f"| {mode} | {r['sharpe']:.4f} | {r['maxdd']:.2%} | "
                 f"{r['annual']:.2%} | {r['turnover']:.4f} |")
    L.append("\n## 3. Block bootstrap (中位 Sharpe)\n")
    L.append("| 变体 | bootstrap 中位 | P10 |")
    L.append("|---|---|---|")
    for mode, v in res["variants"].items():
        L.append(f"| {mode} | {v['boot_sharpe_med']:.4f} | {v['boot_sharpe_p10']:.4f} |")
    L.append("\n## 4. E2 gate 判定 (ΔSharpe≥+0.01 AND ΔMaxDD≤+0.3pp AND bootstrap 中位不劣)\n")
    L.append("| 变体 | ΔSharpe | ΔMaxDD(pp) | bootstrap | 判定 |")
    L.append("|---|---|---|---|---|")
    for mode, g in res["gates"].items():
        L.append(f"| {mode} | {g['d_sharpe']:+.4f} | {g['d_maxdd_pp']:+.2f} | "
                 f"{'✓' if g['boot_not_worse'] else '✗'} | **{g['verdict']}** |")
    n_pass = sum(1 for g in res["gates"].values() if g["verdict"] == "PASS")
    L.append(f"\n**结论**: {n_pass}/3 变体通过 E2 gate。"
             + ("建议将最优变体纳入 v4.5-pvd 后续迭代 (enabled 开关, 默认关闭)。" if n_pass
                else " quantile 门控未能在不牺牲 realized Sharpe 的前提下改善风险——先验成立 "
                "(Sharpe 已 1.60, 边际预期 < +0.03), B2 方向 NO-GO 归档, PVD 维持现状。"))
    out_md = OUT / "exp_pvd_quantile_gate.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[save] {out_md}")


if __name__ == "__main__":
    main()
