#!/usr/bin/env python3
"""PVD 软门控预研 (任务 #65) — monkeypatch 实验, 不改生产代码。

背景: 生产 PVD 双重硬门控 (vol ∈ [p25,p75] AND gap<0.05 → 全量注入, 否则零)
存在边界悬崖。本实验对比三方案:
  A (基线): 现状硬门控          w = 1{p25≤vol≤p75} × 1{gap<0.05}
  B (全软化): 梯形 vol 隶属度(两侧带宽 (p75-p25)×τ) × gap 线性斜坡
  C (非对称): vol 下边界软化 + 上边界 p75 硬切断(危机隔离) × gap 线性斜坡
注入统一: score += pvd_w × w × PVD (pvd_w=0.15, expanding 无前视门限)。

机制: monkeypatch src.backtest.compute_all_factors — 在因子层把 pvd 行乘 w 序列;
同时 bypass 主循环硬门控 (gates=±1e18 + gap_threshold=1e9), 使循环内注入
退化为无条件 pvd_w×(w×pvd)。方案 A 的 w∈{0,1} 复刻生产口径 → bit-exact 校验。

测试: T1 边界扰动敏感性(20 seeds ±0.1% 乘性噪声) / T2 realized 对比 /
      T3 block bootstrap (n=100, seed_base=8000, 三方案配对) / T4 τ 带宽敏感性。
"""
import argparse
import contextlib
import dataclasses
import importlib.util
import io
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

import src.backtest as sbt
from src.backtest import run_backtest
from src.strategy import load_config
from src.engine_core import compute_pvd_vol_gates as _real_gates
from src.data_loader import classify_etfs
from src.factors import compute_all_factors as _orig_caf

CFG_PATH = PROJ / 'config' / 'strategy_v4_5_pvd.yaml'
NAV_CSV = PROJ / 'data' / 'all_etfs_nav_latest.csv'
OUT_DIR = PROJ / 'output' / 'experiments'
JSON_PATH = OUT_DIR / 'exp_pvd_soft_gate.json'
MD_PATH = OUT_DIR / 'exp_pvd_soft_gate.md'
TMP_DIR = PROJ / 'output' / '_tmp_softgate'

# ---------------- 软门控权重构造 ----------------
_MODE = {"scheme": None, "tau": 0.25, "last_w": None}


def _gap_series(mom_values):
    """每周 top-2 momentum gap (与 backtest 循环口径一致: 非 NaN 且 >-inf, 全 ETF)。"""
    n = mom_values.shape[0]
    gap = np.full(n, np.nan)
    for i in range(n):
        v = [x for x in mom_values[i] if not np.isnan(x) and x > -np.inf]
        if len(v) >= 2:
            v.sort(reverse=True)
            gap[i] = v[0] - v[1]
    return gap


def weight_series(nv, mom_values, scheme, tau, gap_thr, pct_range):
    """w[i] ∈ [0,1]: PVD 注入门控权重 (expanding 无前视门限)。"""
    lo, hi = _real_gates(nv, pct_range)
    gap = _gap_series(mom_values)
    n = len(nv)
    w = np.zeros(n)
    for i in range(n):
        if np.isnan(nv[i]) or np.isnan(gap[i]):
            continue  # 生产口径: vol NaN 或 valid_mom<2 → 不注入
        v = nv[i]
        if scheme == 'A':
            w[i] = 1.0 if (lo[i] <= v <= hi[i] and gap[i] < gap_thr) else 0.0
            continue
        f_gap = max(0.0, 1.0 - gap[i] / gap_thr)
        band = (hi[i] - lo[i]) * tau
        if band <= 0:
            f_vol = 1.0 if lo[i] <= v <= hi[i] else 0.0
        elif scheme == 'B':
            if lo[i] <= v <= hi[i]:
                f_vol = 1.0
            elif v < lo[i]:
                f_vol = max(0.0, 1.0 - (lo[i] - v) / band)
            else:
                f_vol = max(0.0, 1.0 - (v - hi[i]) / band)
        else:  # C: 上边界硬切断 (危机隔离)
            if v > hi[i]:
                f_vol = 0.0
            elif v >= lo[i]:
                f_vol = 1.0
            else:
                f_vol = max(0.0, 1.0 - (lo[i] - v) / band)
        w[i] = f_vol * f_gap
    return w


def _patched_caf(*args, **kwargs):
    """monkeypatch: 因子层行乘 w 序列 (对任何输入 NAV 自洽, bootstrap 路径亦然)。"""
    factors = _orig_caf(*args, **kwargs)
    if _MODE["scheme"] is None or 'pvd' not in factors:
        return factors
    cfg = _MODE["cfg"]
    nv = factors['volatility']['纳指ETF'].values
    w = weight_series(nv, factors['momentum'].values, _MODE["scheme"], _MODE["tau"],
                      cfg.pvd_score_gap_threshold, cfg.pvd_vol_pct_range)
    _MODE["last_w"] = w.copy()
    factors['pvd'] = factors['pvd'].mul(pd.Series(w, index=factors['pvd'].index), axis=0)
    return factors


def _bypass_gates(nv, pct_range=(0.25, 0.75), min_samples=50):
    n = len(np.asarray(nv))
    return np.full(n, -1e18), np.full(n, 1e18)


def install_patch(base_cfg):
    sbt.compute_all_factors = _patched_caf
    sbt.compute_pvd_vol_gates = _bypass_gates
    # bypass 循环内 gap 硬阈值 (软权重已在因子层生效); 原 cfg 参数存 _MODE 供 w 构造
    _MODE["cfg"] = base_cfg
    return dataclasses.replace(base_cfg, pvd_score_gap_threshold=1e9)


def set_scheme(scheme, tau=0.25):
    _MODE["scheme"] = scheme
    _MODE["tau"] = tau


# ---------------- 回测封装 ----------------
def bt(cfg, data_path=None):
    with contextlib.redirect_stdout(io.StringIO()):
        r = run_backtest(cfg, data_path=str(data_path) if data_path else None)
    return r


def off_sets(result, off_names):
    """每周进攻持仓集合 {date: frozenset}"""
    out = {}
    for rec in result.weekly_records:
        s = frozenset(e for e in off_names if rec.get(f'weight_{e}', 0.0) > 1e-9)
        out[rec['date']] = s
    return out


# ---------------- T2: realized ----------------
def run_t2(bypass_cfg):
    print("\n===== T2: realized 三方案对比 =====")
    rows = {}
    for scheme in ('A', 'B', 'C'):
        set_scheme(scheme)
        r = bt(bypass_cfg)
        m = r.metrics
        w = _MODE["last_w"]
        rows[scheme] = {
            "sharpe": float(m['sharpe_ratio']), "maxdd": float(m['max_drawdown']),
            "annual": float(m['annual_return']),
            "w_nonzero_weeks": int((w > 1e-12).sum()),
            "w_partial_weeks": int(((w > 1e-12) & (w < 1 - 1e-12)).sum()),
            "w_mean": float(w.mean()),
        }
        print(f"  {scheme}: Sharpe={m['sharpe_ratio']:.4f}  MaxDD={m['max_drawdown']*100:.2f}%  "
              f"Annual={m['annual_return']*100:.2f}%  注入周={rows[scheme]['w_nonzero_weeks']}"
              f" (部分注入 {rows[scheme]['w_partial_weeks']})")
    return rows


# ---------------- T1: 边界扰动敏感性 ----------------
def run_t1(bypass_cfg, n_seeds):
    print(f"\n===== T1: 边界扰动敏感性 (n_seeds={n_seeds}, ε~N(0,0.001)) =====")
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    nav_raw = pd.read_csv(NAV_CSV)
    etf_cols = [c for c in nav_raw.columns if c != '日期']
    off_names = [e for e in ('纳指ETF', '中证500ETF', '黄金ETF') if e in etf_cols]

    # 无噪声基准 (三方案)
    base = {}
    for scheme in ('A', 'B', 'C'):
        set_scheme(scheme)
        r = bt(bypass_cfg, NAV_CSV)
        base[scheme] = {"sel": off_sets(r, off_names), "w": _MODE["last_w"].copy(),
                        "sharpe": float(r.metrics['sharpe_ratio'])}

    per_seed = {s: [] for s in 'ABC'}
    t0 = time.time()
    for k in range(n_seeds):
        rng = np.random.default_rng(20260800 + k)
        pert = nav_raw.copy()
        noise = rng.normal(0.0, 0.001, size=(len(pert), len(etf_cols)))
        pert[etf_cols] = pert[etf_cols].values * (1.0 + noise)
        tmp = TMP_DIR / f'nav_pert_{k}.csv'
        pert.to_csv(tmp, index=False)
        for scheme in ('A', 'B', 'C'):
            set_scheme(scheme)
            r = bt(bypass_cfg, tmp)
            sel = off_sets(r, off_names)
            b = base[scheme]
            common = [d for d in b["sel"] if d in sel]
            flips = sum(1 for d in common if sel[d] != b["sel"][d])
            dw = np.abs(_MODE["last_w"] - b["w"])
            per_seed[scheme].append({
                "seed": k, "sharpe": float(r.metrics['sharpe_ratio']),
                "flip_rate": flips / max(len(common), 1),
                "dw_mean": float(dw.mean()), "dw_max": float(dw.max()),
                "dw_full_flips": int((dw > 0.999).sum()),   # 全量翻转周数 (0↔1)
                "dw_nonzero": int((dw > 1e-9).sum()),
            })
        tmp.unlink()
        if (k + 1) % 5 == 0:
            print(f"  [{k+1}/{n_seeds}] 耗时 {time.time()-t0:.0f}s", flush=True)

    summary = {}
    for scheme in 'ABC':
        arr = per_seed[scheme]
        sh = np.array([x["sharpe"] for x in arr])
        fl = np.array([x["flip_rate"] for x in arr])
        summary[scheme] = {
            "flip_rate_mean": float(fl.mean()), "flip_rate_max": float(fl.max()),
            "sharpe_mean": float(sh.mean()), "sharpe_std": float(sh.std(ddof=1)),
            "dw_mean_avg": float(np.mean([x["dw_mean"] for x in arr])),
            "dw_full_flips_avg": float(np.mean([x["dw_full_flips"] for x in arr])),
            "dw_nonzero_avg": float(np.mean([x["dw_nonzero"] for x in arr])),
            "base_sharpe": base[scheme]["sharpe"],
        }
        print(f"  {scheme}: 翻转率 mean={fl.mean()*100:.2f}% max={fl.max()*100:.2f}%  "
              f"Sharpe std={sh.std(ddof=1):.4f}  |Δw| mean={summary[scheme]['dw_mean_avg']:.4f}  "
              f"全量翻转周均值={summary[scheme]['dw_full_flips_avg']:.1f}")
    return {"summary": summary, "per_seed": per_seed}


# ---------------- T3: block bootstrap ----------------
def run_t3(bypass_cfg, n_paths, seed_base=8000, block_len=13):
    print(f"\n===== T3: block bootstrap (n={n_paths}, block={block_len}, seed_base={seed_base}) =====")
    def _load(name, rel):
        spec = importlib.util.spec_from_file_location(name, PROJ / rel)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    oos = _load("oos_sg", "scripts/oos_validation.py")
    rj = _load("rj_sg", "scripts/robustness_joint.py")
    rets, dates, first_nav = rj.prepare_real_data()
    out = {}
    for scheme in ('A', 'B', 'C'):
        set_scheme(scheme)
        t0 = time.time()
        rows = []
        for i in range(n_paths):
            seed = seed_base + i
            boot = oos.block_bootstrap(rets, block_len, seed)
            m = oos.eval_strat_ew_on_returns(boot, dates, first_nav, bypass_cfg, f"sg_{scheme}_{i}")
            if m is None:
                continue
            rows.append({"seed": seed, "sharpe": float(m["strat_sharpe"]),
                         "maxdd": float(m["strat_maxdd"]), "annual": float(m["strat_annual"]),
                         "ew_sharpe": float(m["ew_sharpe"])})
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  {scheme} [{i+1}/{n_paths}] ETA {(el/(i+1)*(n_paths-i-1))/60:.1f}min", flush=True)
        sh = np.array([r["sharpe"] for r in rows])
        ews = np.array([r["ew_sharpe"] for r in rows])
        alpha = sh - ews
        out[scheme] = {
            "n": len(rows),
            "win_rate_over_ew": float((alpha > 0).mean()),
            "alpha_p10": float(np.quantile(alpha, 0.10)),
            "alpha_p50": float(np.quantile(alpha, 0.50)),
            "sharpe_p10": float(np.quantile(sh, 0.10)),
            "sharpe_p50": float(np.quantile(sh, 0.50)),
            "rows": rows,
        }
        print(f"  {scheme}: 胜率={out[scheme]['win_rate_over_ew']*100:.1f}%  "
              f"alpha P10={out[scheme]['alpha_p10']:+.3f}  Sharpe P50={out[scheme]['sharpe_p50']:.3f}")
    return out


# ---------------- T4: τ 敏感性 ----------------
def run_t4(bypass_cfg, scheme):
    print(f"\n===== T4: τ 带宽敏感性 (方案 {scheme}) =====")
    out = {}
    for tau in (0.15, 0.25, 0.40):
        set_scheme(scheme, tau=tau)
        r = bt(bypass_cfg)
        out[f"tau_{tau}"] = {"sharpe": float(r.metrics['sharpe_ratio']),
                             "maxdd": float(r.metrics['max_drawdown'])}
        print(f"  τ={tau}: Sharpe={r.metrics['sharpe_ratio']:.4f}  MaxDD={r.metrics['max_drawdown']*100:.2f}%")
    shs = [v["sharpe"] for v in out.values()]
    out["max_sharpe_spread"] = float(max(shs) - min(shs))
    print(f"  Sharpe 极差 = {out['max_sharpe_spread']:.4f} (<0.01 → 不敏感)")
    return out


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-seeds', type=int, default=20)
    ap.add_argument('--n-boot', type=int, default=100)
    ap.add_argument('--stage', default='all', choices=['all', 't1', 't2', 't3', 't4'])
    a = ap.parse_args()

    base_cfg = load_config(CFG_PATH)
    bypass_cfg = install_patch(base_cfg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = json.loads(JSON_PATH.read_text()) if JSON_PATH.exists() else {}

    # sanity: 方案 A 必须复现生产 realized (bit-exact 口径校验)
    set_scheme('A')
    r = bt(bypass_cfg)
    sbt_prod = None
    sbt.compute_all_factors = _orig_caf
    sbt.compute_pvd_vol_gates = _real_gates
    with contextlib.redirect_stdout(io.StringIO()):
        sbt_prod = run_backtest(base_cfg)
    sbt.compute_all_factors = _patched_caf
    sbt.compute_pvd_vol_gates = _bypass_gates
    dif = abs(r.metrics['sharpe_ratio'] - sbt_prod.metrics['sharpe_ratio'])
    print(f"Sanity: 方案A={r.metrics['sharpe_ratio']:.6f} vs 生产={sbt_prod.metrics['sharpe_ratio']:.6f} (Δ={dif:.2e})")
    assert dif < 1e-9, "方案 A 未能 bit-exact 复现生产路径!"
    results["sanity"] = {"scheme_A": float(r.metrics['sharpe_ratio']),
                         "production": float(sbt_prod.metrics['sharpe_ratio'])}

    if a.stage in ('all', 't2'):
        results["t2"] = run_t2(bypass_cfg)
    if a.stage in ('all', 't1'):
        results["t1"] = run_t1(bypass_cfg, a.n_seeds)
    if a.stage in ('all', 't3'):
        results["t3"] = run_t3(bypass_cfg, a.n_boot)
    if a.stage in ('all', 't4'):
        # 胜出候选: 默认 C (非对称, 危机隔离保留); 若 C 在 t2 劣于 B 超 0.01 则用 B
        winner = 'C'
        if "t2" in results and results["t2"]["B"]["sharpe"] - results["t2"]["C"]["sharpe"] > 0.01:
            winner = 'B'
        results["t4"] = run_t4(bypass_cfg, winner)
        results["t4"]["scheme"] = winner

    JSON_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n结果落盘: {JSON_PATH}")


if __name__ == '__main__':
    main()
