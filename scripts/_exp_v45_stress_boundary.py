#!/usr/bin/env python3
"""v4.5-pvd 对抗鲁棒性边界专项调查 (2026-08) — 实验档案脚本

目标: 量化 v4_5_pvd "能抵御多大的对抗, 失控边界在哪里"。
  Q2: 递增强度压力测试 — 波动轴(sig_mult 1.5→3.0) / 相关轴(rho_crisis 0.85→0.95)
      / 漂移轴(muoff_mult 0.6→0.0) / 组合冲击, v4_5_pvd vs v4_3 对照。
  Q3a: 伪平静阴跌情景(vol 处于 PVD 激活带内 + 进攻资产负漂移) + PVD 激活率统计。
  Q3b: PVD 信号失效实验(真实数据上 amount 行置换 ×10 / PVD 信号反转)。

失控判据(单元格 FAIL): 中位 MaxDD > 12% 或 中位 Sharpe < 0.5 或 中位 Sharpe <= 等权。

方法: 复用 adversarial_robustness.py 的 CCC-GARCH / regime_corr DGP 与
_eval_strat_ew 评估器(策略 vs 等权同数据对照), 每格 5 seeds 取中位数。
不改任何生产代码; 通过 monkeypatch 注入 amount 置换 / PVD 反转。

已知方法论边界(报告中必须声明): 合成 DGP 只生成价格路径, PVD 在合成数据上
以"真实 amount × 合成价格"计算 → 等价于噪声化 PVD 信号, 而非有信息 PVD。
故 Q2 对照度量的是"PVD 作为噪声注入"在极端环境下的边际影响。

用法: .venv/bin/python scripts/_exp_v45_stress_boundary.py
输出: output/experiments/exp_v45_pvd_stress_boundary.json (报告 md 由调查人撰写)
"""
import importlib.util
import io
import json
import contextlib
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

_spec = importlib.util.spec_from_file_location("ar", PROJ / "scripts" / "adversarial_robustness.py")
ar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ar)
dm = ar.dm

import src.data_loader as dl
import src.factors as fx
from src.data_loader import classify_etfs
from src.strategy import load_config
from src.backtest import run_backtest
from src.factors import compute_all_factors

CFG_V45 = load_config(PROJ / "config" / "strategy_v4_5_pvd.yaml")
CFG_V43 = load_config(PROJ / "config" / "strategy_v4_3.yaml")
OUT_JSON = PROJ / "output" / "experiments" / "exp_v45_pvd_stress_boundary.json"
SEEDS = (11, 22, 33, 44, 55)
CACHE_DIR = PROJ / "data" / "experiments" / "tushare_cache"

# 失控判据
FAIL_MAXDD = 0.12
FAIL_SHARPE = 0.5

# ----------------------------------------------------------------------
# monkeypatch 1: 周频 amount 加载缓存(v4_5 每次合成回测都会重读 51 个日频 CSV,
# 索引恒为真实周频日期 → 结果恒同, 缓存后单次回测 2.1s → 0.7s)
# ----------------------------------------------------------------------
_ORIG_LOADER = dl.load_weekly_volume_from_cache
_VOL_CACHE = {}


def _cached_loader(cache_dir, weekly_index, etf_names=None):
    key = (str(cache_dir), str(weekly_index[0]), str(weekly_index[-1]),
           len(weekly_index), tuple(etf_names or ()))
    if key not in _VOL_CACHE:
        _VOL_CACHE[key] = _ORIG_LOADER(cache_dir, weekly_index, etf_names)
    return _VOL_CACHE[key].copy()


dl.load_weekly_volume_from_cache = _cached_loader


# ----------------------------------------------------------------------
# Q2: 递增强度压力扫描
# ----------------------------------------------------------------------
# 每格: 名称, DGP 参数覆写(dgp 缺省 = gen_garch; regime_corr 时 p_enter/p_stay 用
# v4.4 默认 0.03/0.92, 即间歇性危机态而非 grey_corr_combo 的全程危机)
Q2_CELLS = [
    ("baseline",   {}),
    # 波动轴: 现有情景最大 sig_mult=1.5 (grey_corr_combo), 向上递增
    ("sig_1.50",   {"sig_mult": 1.50}),
    ("sig_1.75",   {"sig_mult": 1.75}),
    ("sig_2.00",   {"sig_mult": 2.00}),
    ("sig_2.50",   {"sig_mult": 2.50}),
    ("sig_3.00",   {"sig_mult": 3.00}),
    # 相关轴: 现有情景 rho_crisis=0.85, 向上递增(间歇危机态 Markov 默认转移)
    ("rho_0.85",   {"dgp": "regime_corr", "rho_crisis": 0.85}),
    ("rho_0.90",   {"dgp": "regime_corr", "rho_crisis": 0.90}),
    ("rho_0.95",   {"dgp": "regime_corr", "rho_crisis": 0.95}),
    # 漂移轴: 现有情景最低 muoff_mult=0.6 (adv-oos), 向下递减到 0(进攻零收益)
    ("muoff_0.6",  {"muoff_mult": 0.6}),
    ("muoff_0.4",  {"muoff_mult": 0.4}),
    ("muoff_0.2",  {"muoff_mult": 0.2}),
    ("muoff_0.0",  {"muoff_mult": 0.0}),
    # 组合冲击: σ×2.0 + 危机相关 0.85 + 进攻漂移减半
    ("combo",      {"dgp": "regime_corr", "rho_crisis": 0.85,
                    "sig_mult": 2.0, "muoff_mult": 0.5}),
]


def judge(m):
    """失控判据: 返回 (fail:bool, reasons:list)。"""
    reasons = []
    if np.isnan(m["strat_sharpe"]):
        return True, ["no_result"]
    if m["strat_maxdd"] > FAIL_MAXDD:
        reasons.append(f"MaxDD {m['strat_maxdd']:.1%}>12%")
    if m["strat_sharpe"] < FAIL_SHARPE:
        reasons.append(f"Sharpe {m['strat_sharpe']:.2f}<0.5")
    if m["strat_sharpe"] <= m["ew_sharpe"]:
        reasons.append(f"跑输等权({m['strat_sharpe']:.2f}<={m['ew_sharpe']:.2f})")
    return bool(reasons), reasons


def run_q2(mu, A, R, nu, gp, T, real_dates, first_nav):
    print("=" * 78)
    print(" Q2: 递增强度压力扫描 (5 seeds/格, 中位数; 判据 DD>12% | Sh<0.5 | 输等权)")
    print("=" * 78)
    out = {}
    hdr = f" {'cell':<11s}{'cfg':<6s}{'Sharpe':>8s}{'等权Sh':>8s}{'MaxDD':>8s}{'等权DD':>8s}{'Annual':>8s}{'裁决':>6s}"
    print(hdr)
    print("-" * 78)
    for name, overrides in Q2_CELLS:
        params = dict(ar.REALIZED, **overrides)
        out[name] = {"params": overrides}
        for tag, cfg in (("v4_5", CFG_V45), ("v4_3", CFG_V43)):
            t0 = time.time()
            m = ar._eval_strat_ew(mu, A, R, nu, gp, params, T, real_dates, first_nav, cfg, SEEDS)
            fail, reasons = judge(m)
            out[name][tag] = {**m, "fail": fail, "fail_reasons": reasons,
                              "elapsed_s": round(time.time() - t0, 1)}
            print(f" {name:<11s}{tag:<6s}{m['strat_sharpe']:>8.3f}{m['ew_sharpe']:>8.3f}"
                  f"{m['strat_maxdd']:>8.2%}{m['ew_maxdd']:>8.2%}{m['strat_annual']:>8.2%}"
                  f"{'FAIL' if fail else 'ok':>6s}  {';'.join(reasons)}")
        sys.stdout.flush()
    return out


# ----------------------------------------------------------------------
# Q3a: 伪平静阴跌 — vol 不越过 Layer3 高档阈值/停留 PVD 激活带, 进攻资产负漂移
# ----------------------------------------------------------------------
Q3A_CELLS = [
    # sig<=1.0 保证合成 vol 分布不整体右移(伪平静), muoff<0 = 缓慢阴跌
    ("calm_bleed_-0.3", {"sig_mult": 0.9, "muoff_mult": -0.3}),
    ("calm_bleed_-0.6", {"sig_mult": 0.9, "muoff_mult": -0.6}),
    ("deep_calm_bleed", {"sig_mult": 0.7, "muoff_mult": -0.6}),
]


def run_q3a(mu, A, R, nu, gp, T, real_dates, first_nav):
    print("\n" + "=" * 78)
    print(" Q3a: 伪平静阴跌情景 (sig<=1.0 + 进攻负漂移; PVD 激活带内的缓慢崩盘)")
    print("=" * 78)
    out = {}
    for name, overrides in Q3A_CELLS:
        params = dict(ar.REALIZED, **overrides)
        out[name] = {"params": overrides}
        for tag, cfg in (("v4_5", CFG_V45), ("v4_3", CFG_V43)):
            m = ar._eval_strat_ew(mu, A, R, nu, gp, params, T, real_dates, first_nav, cfg, SEEDS)
            fail, reasons = judge(m)
            out[name][tag] = {**m, "fail": fail, "fail_reasons": reasons}
            print(f" {name:<16s}{tag:<6s} Sh={m['strat_sharpe']:>7.3f} (等权 {m['ew_sharpe']:>7.3f})"
                  f"  DD={m['strat_maxdd']:>7.2%}  Ann={m['strat_annual']:>7.2%}"
                  f"  {'FAIL' if fail else 'ok'} {';'.join(reasons)}")
        d = out[name]
        d["delta_sharpe_v45_vs_v43"] = d["v4_5"]["strat_sharpe"] - d["v4_3"]["strat_sharpe"]
        d["delta_maxdd_v45_vs_v43"] = d["v4_5"]["strat_maxdd"] - d["v4_3"]["strat_maxdd"]
        sys.stdout.flush()
    return out


# ----------------------------------------------------------------------
# PVD 激活率统计(复刻 backtest.py 334-347 门控逻辑, 只读不改生产代码)
# ----------------------------------------------------------------------
def pvd_activation_stats(cfg, weekly_nav, weekly_vol, ret_mask=False):
    """返回 {n_weeks, n_vol_band, n_active, act_rate, ...}; 与回测循环同口径。"""
    config_dict = {"factors": {
        "mom_window": cfg.mom_window, "vol_window": cfg.vol_window,
        "vol_ddof": cfg.vol_ddof, "pe_window_years": cfg.pe_window_years,
        "ewma_factors_enabled": cfg.ewma_factors_enabled,
        "ewma_mom_halflife": cfg.ewma_mom_halflife,
        "ewma_vol_halflife": cfg.ewma_vol_halflife,
        "vol_taper_enabled": cfg.vol_taper_enabled,
        "vol_taper_window": cfg.vol_taper_window, "vol_taper_len": cfg.vol_taper_len,
        "pvd_enabled": True, "pvd_window": cfg.pvd_window,
        "pvd_min_periods": cfg.pvd_min_periods,
    }}
    factors = compute_all_factors(weekly_nav, None, config_dict, weekly_vol=weekly_vol)
    mom = factors["momentum"].values
    vol = factors["volatility"].values
    pvd = factors["pvd"].values
    etf_names = list(weekly_nav.columns)
    off_idx, _, NASDAQ_IDX = classify_etfs(etf_names)
    nv_all = vol[:, NASDAQ_IDX]
    valid_nv = nv_all[~np.isnan(nv_all)]
    p25 = np.percentile(valid_nv, cfg.pvd_vol_pct_range[0] * 100)
    p75 = np.percentile(valid_nv, cfg.pvd_vol_pct_range[1] * 100)
    start_idx = max(cfg.vol_taper_window, cfg.mom_window) if cfg.vol_taper_enabled \
        else max(cfg.vol_window, cfg.mom_window)
    n_weeks = n_band = n_active = 0
    active_mask = []
    n = len(weekly_nav)
    for i in range(start_idx, n - 1):
        n_weeks += 1
        act = False
        nvi = nv_all[i]
        if not np.isnan(nvi) and p25 <= nvi <= p75:
            n_band += 1
            vm = [(mom[i, j], j) for j in range(len(etf_names))
                  if not np.isnan(mom[i, j]) and mom[i, j] > -np.inf]
            if len(vm) >= 2:
                vm.sort(key=lambda x: x[0], reverse=True)
                if vm[0][0] - vm[1][0] < cfg.pvd_score_gap_threshold:
                    if any(not np.isnan(pvd[i, j]) for j in off_idx):
                        n_active += 1
                        act = True
        active_mask.append(act)
    res = {"n_weeks": n_weeks, "n_vol_band": n_band, "n_active": n_active,
           "vol_band_rate": n_band / n_weeks, "act_rate": n_active / n_weeks,
           "vol_p25": float(p25), "vol_p75": float(p75)}
    if ret_mask:
        res["mask"] = active_mask
    return res


def run_activation(mu, A, R, nu, gp, T, real_dates, first_nav, wk):
    print("\n" + "-" * 78)
    print(" PVD 激活率: 真实数据 vs 伪平静阴跌 vs 高波动 (门控行为诊断)")
    print("-" * 78)
    out = {}
    weekly_vol_real = _cached_loader(CACHE_DIR, wk.index, list(wk.columns))
    st = pvd_activation_stats(CFG_V45, wk, weekly_vol_real)
    out["real"] = st
    print(f" real          : 激活 {st['n_active']}/{st['n_weeks']} 周 ({st['act_rate']:.1%})"
          f"  vol带内 {st['vol_band_rate']:.1%}  [p25={st['vol_p25']:.3f}, p75={st['vol_p75']:.3f}]")
    for name, overrides in [("calm_bleed_-0.6", {"sig_mult": 0.9, "muoff_mult": -0.6}),
                            ("sig_2.00", {"sig_mult": 2.0})]:
        rates, bands = [], []
        for s in SEEDS[:3]:
            params = dict(ar.REALIZED, **overrides)
            gen = ar.gen_regime_corr if params.get("dgp") == "regime_corr" else ar.gen_garch
            r = gen(mu, A, R, nu, gp, params, T, s)
            nav_df = dm.build_nav_df(r, wk.index, first_nav)
            wv = _cached_loader(CACHE_DIR, nav_df.index, list(nav_df.columns))
            st = pvd_activation_stats(CFG_V45, nav_df, wv)
            rates.append(st["act_rate"]); bands.append(st["vol_band_rate"])
        out[name] = {"act_rate_med": float(np.median(rates)),
                     "vol_band_rate_med": float(np.median(bands))}
        print(f" {name:<14s}: 激活率中位 {np.median(rates):.1%}  vol带内 {np.median(bands):.1%}")
    return out


# ----------------------------------------------------------------------
# Q3b: PVD 信号失效 — 真实数据上 amount 置换 / PVD 反转
# ----------------------------------------------------------------------
def _real_eval(cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        r = run_backtest(cfg)
    return {"sharpe": float(r.metrics["sharpe_ratio"]),
            "maxdd": float(r.metrics["max_drawdown"]),
            "annual": float(r.metrics["annual_return"])}


def run_q3b():
    print("\n" + "=" * 78)
    print(" Q3b: PVD 信号失效实验 (真实数据; amount 行置换 ×10 / PVD 信号反转)")
    print("=" * 78)
    out = {}
    out["v4_5_baseline"] = _real_eval(CFG_V45)
    out["v4_3_baseline"] = _real_eval(CFG_V43)
    b = out["v4_5_baseline"]
    print(f" v4_5 baseline : Sh={b['sharpe']:.4f} DD={b['maxdd']:.2%} Ann={b['annual']:.2%}")
    b3 = out["v4_3_baseline"]
    print(f" v4_3 baseline : Sh={b3['sharpe']:.4f} DD={b3['maxdd']:.2%} Ann={b3['annual']:.2%}")

    # (1) amount 行置换: 打断真实量价对齐(同一置换作用于全部列, 保留截面结构)
    perm_rows = []
    for seed in range(10):
        def _perm_loader(cache_dir, weekly_index, etf_names=None, _s=seed):
            df = _cached_loader(cache_dir, weekly_index, etf_names)
            rng = np.random.default_rng(9000 + _s)
            perm = rng.permutation(len(df))
            return pd.DataFrame(df.values[perm], index=df.index, columns=df.columns)
        dl.load_weekly_volume_from_cache = _perm_loader
        try:
            m = _real_eval(CFG_V45)
        finally:
            dl.load_weekly_volume_from_cache = _cached_loader
        perm_rows.append(m)
        print(f"   perm seed {seed}: Sh={m['sharpe']:.4f} DD={m['maxdd']:.2%}")
    sh = [x["sharpe"] for x in perm_rows]
    dd = [x["maxdd"] for x in perm_rows]
    out["amount_permuted"] = {
        "n": len(perm_rows), "rows": perm_rows,
        "sharpe_med": float(np.median(sh)), "sharpe_min": float(np.min(sh)),
        "sharpe_max": float(np.max(sh)),
        "maxdd_med": float(np.median(dd)), "maxdd_max": float(np.max(dd)),
        "delta_sharpe_med_vs_v45": float(np.median(sh)) - b["sharpe"],
        "delta_sharpe_worst_vs_v45": float(np.min(sh)) - b["sharpe"],
        "delta_maxdd_worst_vs_v45": float(np.max(dd)) - b["maxdd"],
    }
    a = out["amount_permuted"]
    print(f" 置换汇总: Sh中位 {a['sharpe_med']:.4f} / 最差 {a['sharpe_min']:.4f}"
          f"  DD最差 {a['maxdd_max']:.2%}  ΔSh最差 {a['delta_sharpe_worst_vs_v45']:+.4f}")

    # (2) PVD 信号反转: 历史量价关系完全反向(最坏情形上界)
    _orig_pvd = fx.compute_pvd_factor
    fx.compute_pvd_factor = lambda *args, **kw: -_orig_pvd(*args, **kw)
    try:
        m = _real_eval(CFG_V45)
    finally:
        fx.compute_pvd_factor = _orig_pvd
    out["pvd_inverted"] = {**m,
                           "delta_sharpe_vs_v45": m["sharpe"] - b["sharpe"],
                           "delta_maxdd_vs_v45": m["maxdd"] - b["maxdd"],
                           "delta_sharpe_vs_v43": m["sharpe"] - b3["sharpe"]}
    print(f" PVD 反转     : Sh={m['sharpe']:.4f} DD={m['maxdd']:.2%}"
          f"  ΔSh vs v4_5 {m['sharpe']-b['sharpe']:+.4f}  vs v4_3 {m['sharpe']-b3['sharpe']:+.4f}")
    return out


# ----------------------------------------------------------------------
def main():
    t0 = time.time()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    print("拟合 VAR(1)+t 与 CCC-GARCH (一次) ...")
    nav, wk, w_rets = dm.load_real()
    mu, A, Sigma, nu, resid, coords = dm.fit_var_t(w_rets)
    gp, R = ar.fit_garch(resid)
    real_dates = wk.index
    first_nav = wk.iloc[0].values
    T = len(w_rets)
    print(f"  T={T} 周, nu={nu:.1f}")

    results = {
        "meta": {
            "date": "2026-08-02",
            "seeds": list(SEEDS),
            "fail_criteria": "median MaxDD>12% OR median Sharpe<0.5 OR median Sharpe<=等权",
            "note": ("合成 DGP 仅生成价格路径; v4_5 的 PVD 在合成数据上 = 真实 amount × "
                     "合成价格的滚动相关 ≈ 噪声化 PVD。Q2 对照度量'PVD 作为噪声'的边际影响; "
                     "'PVD 作为反向信号'的最坏损害见 Q3b pvd_inverted。"),
        },
        "q2_sweep": run_q2(mu, A, R, nu, gp, T, real_dates, first_nav),
        "q3a_calm_bleed": run_q3a(mu, A, R, nu, gp, T, real_dates, first_nav),
        "pvd_activation": run_activation(mu, A, R, nu, gp, T, real_dates, first_nav, wk),
        "q3b_pvd_failure": run_q3b(),
    }
    results["meta"]["elapsed_min"] = round((time.time() - t0) / 60, 1)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=str)
    print(f"\nDONE in {results['meta']['elapsed_min']} min -> {OUT_JSON}")


if __name__ == "__main__":
    main()
