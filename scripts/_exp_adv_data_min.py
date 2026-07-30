#!/usr/bin/env python
"""T13 对抗性数据最小扰动实验 — 最小地修改历史周收益, 使 v4.4 策略不达预期。

四个攻击面 (各自求 F1/F2/F3 的最小扰动):
  A1 慢跌灰区攻击     — 重仓资产区间整体下移 δ/周 (纯平移, 窗口内 std 不变 → vol 防御钝感)
  A2 单周崩盘攻击     — 高进攻仓位周对持仓资产注入单周 −X% (vol 滞后), 不够则扩 2-3 周
  A3 调仓鞭打攻击     — 临界周 ±ε 制造来回换仓 + 新持仓走弱 (预期低效, 如实报告)
  A4 相关性灰区同步阴跌 — 三进攻资产同步 −δ/周 共同分量 n 周 (corr 不过 0.6, vol 不过 step_high)

失败判据:
  F1: MaxDD > 12% | F2: Sharpe < 同扰动数据等权再平衡 Sharpe | F3: Sharpe < 1.0

硬约束: 不修改 data/、src/、scripts/ 既有文件、config/ 生产配置。
扰动只在内存 + output/experiments/adv_data/ 下的临时 CSV 上实施, 用后即删。

用法: .venv/bin/python scripts/_exp_adv_data_min.py
输出: output/experiments/exp_adv_data_min.md / .json
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.strategy import load_config  # noqa: E402
from src.backtest import run_backtest, compute_metrics  # noqa: E402
from src.data_loader import load_nav_data, resample_weekly, classify_etfs  # noqa: E402
from src.factors import compute_all_factors  # noqa: E402

CFG_PATH = ROOT / "config" / "strategy_v4_4.yaml"
TMP_DIR = ROOT / "output" / "experiments" / "adv_data"
OUT_MD = ROOT / "output" / "experiments" / "exp_adv_data_min.md"
OUT_JSON = ROOT / "output" / "experiments" / "exp_adv_data_min.json"

F1_MAXDD = 0.12
F3_SHARPE = 1.0

_run_counter = {"n": 0}


# ---------------------------------------------------------------- 基础设施
def load_base():
    cfg = load_config(CFG_PATH)
    nav = load_nav_data(ROOT / cfg.nav_path)
    weekly = resample_weekly(nav, anchor=cfg.anchor)
    weekly = weekly[weekly.index >= pd.to_datetime(cfg.start_date)]
    return cfg, weekly


def hist_bounds(weekly: pd.DataFrame) -> dict:
    """各资产历史周收益 min/max — 扰动后单周收益必须留在此范围内(貌似合理约束)。"""
    rets = weekly.pct_change().dropna()
    return {c: (float(rets[c].min()), float(rets[c].max())) for c in weekly.columns}


def perturb_nav(weekly: pd.DataFrame, edits: dict, bounds: dict):
    """edits: {(date_ts, etf_name): delta_ret}。改周收益后自尾部重建 NAV。

    返回 (new_weekly_df, applied)  applied: [(date, etf, r_old, r_new, dr)]
    单周改后收益 clamp 到该资产历史 [min, max] 内; 返回实际生效的 Δr。
    """
    new = weekly.copy()
    dates = list(new.index)
    pos = {d: k for k, d in enumerate(dates)}
    applied = []
    # 按资产聚合, 每资产一次性重建净值 (edits 按收益叠加)
    by_etf: dict[str, dict] = {}
    for (d, etf), dr in edits.items():
        by_etf.setdefault(etf, {})[d] = by_etf.setdefault(etf, {}).get(d, 0.0) + dr
    for etf, dmap in by_etf.items():
        px = new[etf].values.astype(float).copy()
        r = np.zeros_like(px)
        r[1:] = px[1:] / px[:-1] - 1.0
        lo, hi = bounds[etf]
        for d, dr in sorted(dmap.items()):
            t = pos[d]
            if t == 0:
                continue
            r_old = r[t]
            r_new = float(np.clip(r_old + dr, lo, hi))
            applied.append((d, etf, r_old, r_new, r_new - r_old))
            r[t] = r_new
        # 自第一处修改起重建 (保持其余周收益不变)
        t0 = min(pos[d] for d in dmap)
        for t in range(t0, len(px)):
            px[t] = px[t - 1] * (1 + r[t])
        new[etf] = px
    return new, applied


def ew_metrics_on(nav_df: pd.DataFrame, res, rf: float) -> dict:
    """同扰动数据上的等权每周再平衡基准 (口径同 scripts/adversarial_robustness._eval_strat_ew)。"""
    start, end = res.nav_series.index[0], res.nav_series.index[-1]
    pr = nav_df.loc[start:end].astype(float)
    er = pr.ffill().pct_change().fillna(0.0).values
    rb = np.ones(len(pr))
    for i in range(1, len(pr)):
        rb[i] = rb[i - 1] * (1 + float(np.mean(er[i])))
    wr = np.zeros(len(rb))
    wr[1:] = rb[1:] / rb[:-1] - 1
    peak = np.maximum.accumulate(rb)
    dd = (peak - rb) / peak
    df = pd.DataFrame({"nav": rb, "weekly_return": wr, "drawdown": dd,
                       "def_ratio": 0.0, "turnover": 0.0}, index=pr.index)
    return compute_metrics(df, rf)


def evaluate(cfg, weekly_pert: pd.DataFrame, tag: str = "x") -> dict:
    """在扰动后的周 NAV 上跑一次完整回测 + 等权基准, 返回判据所需指标。"""
    _run_counter["n"] += 1
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = TMP_DIR / f"_tmp_{tag}_{os.getpid()}.csv"
    weekly_pert.to_csv(tmp, encoding="utf-8")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = run_backtest(cfg, data_path=str(tmp))
        m = res.metrics
        ewm = ew_metrics_on(weekly_pert, res, cfg.risk_free_rate)
        ns = res.nav_series
        return {
            "sharpe": m["sharpe_ratio"], "maxdd": m["max_drawdown"],
            "annual": m["annual_return"],
            "ew_sharpe": ewm["sharpe_ratio"], "ew_maxdd": ewm["max_drawdown"],
            "stop_loss_weeks": int(ns["in_stop_loss"].sum()),
            "rebalance_count": m["rebalance_count"],
            "F1": bool(m["max_drawdown"] > F1_MAXDD),
            "F2": bool(m["sharpe_ratio"] < ewm["sharpe_ratio"]),
            "F3": bool(m["sharpe_ratio"] < F3_SHARPE),
            "_res": res,
        }
    finally:
        if tmp.exists():
            os.remove(tmp)


def sum_abs_dr(applied) -> float:
    return float(sum(abs(a[-1]) for a in applied))


def pack(ev: dict, applied, extra: dict | None = None) -> dict:
    out = {k: v for k, v in ev.items() if k != "_res"}
    out["n_weeks_modified"] = len({(a[0], a[1]) for a in applied})
    out["sum_abs_dr"] = round(sum_abs_dr(applied), 4)
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------- 目标定位
def find_heavy_runs(res, etf: str, min_w: float = 0.4):
    """从 weekly_records 找该资产权重 >= min_w 的连续区间 [(dates,...), ...]。"""
    ns = res.nav_series
    mask = ns[f"weight_{etf}"] >= min_w
    runs, cur = [], []
    for d, ok in mask.items():
        if ok:
            cur.append(d)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return sorted(runs, key=len, reverse=True)


def find_calm_run(res, def_cap: float = 0.45, min_len: int = 15):
    """找防御比例贴地(≈base 0.35)的最长平静区间, 供 A4 注入。"""
    ns = res.nav_series
    mask = ns["def_ratio"] <= def_cap
    runs, cur = [], []
    for d, ok in mask.items():
        if ok:
            cur.append(d)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    runs = [r for r in runs if len(r) >= min_len]
    return sorted(runs, key=len, reverse=True)


# ---------------------------------------------------------------- 搜索工具
def bisect_delta(make_edits, crit: str, cfg, weekly, bounds, lo, hi, tag, iters=6):
    """在 δ∈[lo,hi] 二分求触发 crit 的最小 δ。hi 必须已触发。返回 (δ*, pack)。"""
    ev_hi = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pert, applied = perturb_nav(weekly, make_edits(mid), bounds)
        ev = evaluate(cfg, pert, tag)
        if ev[crit]:
            hi, ev_hi = mid, pack(ev, applied, {"delta": round(mid, 5)})
        else:
            lo = mid
    return hi, ev_hi


# ---------------------------------------------------------------- A1 慢跌灰区
def attack_A1(cfg, weekly, bounds, base_res):
    print("\n=== A1 慢跌灰区攻击 ===")
    # 选策略重仓时间最长的进攻资产及其最长重仓区间 (max_single_alloc=0.4 → 重仓阈值取 0.3)
    off_names = [c for c in weekly.columns if classify_etfs([c])[0]]
    best = None
    for etf in off_names:
        runs = find_heavy_runs(base_res, etf, 0.3)
        if runs and (best is None or len(runs[0]) > len(best[1])):
            best = (etf, runs[0])
    etf, run = best
    print(f"  目标: {etf} 重仓区间 {run[0].date()} ~ {run[-1].date()} ({len(run)} 周)")

    def make_edits(delta, n):
        return {(d, etf): -delta for d in run[:n]}

    grid_n = [4, 6, 8, 10, 13, 17, 26]
    grid_n = [n for n in grid_n if n <= len(run)]
    grid_d = [0.005, 0.01, 0.015, 0.02, 0.03]
    results = {"target": etf, "window_start": str(run[0].date()),
               "window_len_available": len(run), "grid": [], "min": {}}
    hits = {"F1": [], "F2": [], "F3": []}
    for n in grid_n:
        for d in grid_d:
            pert, applied = perturb_nav(weekly, make_edits(d, n), bounds)
            ev = evaluate(cfg, pert, "a1")
            row = pack(ev, applied, {"n": n, "delta": d})
            results["grid"].append(row)
            for f in hits:
                if ev[f]:
                    hits[f].append(row)
            print(f"  n={n:2d} δ={d:.3f} Sh={ev['sharpe']:.3f} DD={ev['maxdd']:.3f} "
                  f"ewSh={ev['ew_sharpe']:.3f} F1={ev['F1']} F2={ev['F2']} F3={ev['F3']}")
    # 各判据: 取 Σ|Δr| 最小的网格命中点, 再对其 n 二分精化 δ
    for f, rows in hits.items():
        if not rows:
            results["min"][f] = None
            continue
        cand = min(rows, key=lambda r: r["sum_abs_dr"])
        n = cand["n"]
        _, refined = bisect_delta(lambda d: make_edits(d, n), f, cfg, weekly, bounds,
                                  lo=0.0, hi=cand["delta"], tag="a1b")
        best_row = refined if refined else cand
        best_row["n"] = n
        results["min"][f] = best_row
        print(f"  [A1 min {f}] n={n} δ={best_row['delta']:.4f} Σ|Δr|={best_row['sum_abs_dr']:.3f} "
              f"Sh={best_row['sharpe']:.3f} DD={best_row['maxdd']:.3f}")

    # 自适应贪心变体: 只在"策略当前实际持有目标资产"的周上逐周加扰动
    # (回答"策略轮动逃跑后攻击能否追着打": 每加一周 −δ 重跑, 找下一个仍持有的周)
    results["greedy"] = {}
    for d in (0.01, 0.02, 0.03):
        max_n = 45 if d == 0.01 else 90  # 大 δ 放宽上限, 探 F3/F2 能否被追击达成
        traj = greedy_follow(cfg, weekly, bounds, etf, run[0], d, max_n=max_n)
        results["greedy"][f"delta_{d}"] = traj
        ft = traj["first_trigger"]
        print(f"  [A1 greedy δ={d:.2f}] 最终 n={traj['n_final']} Sh={traj['final']['sharpe']:.3f} "
              f"DD={traj['final']['maxdd']:.3f} 首触发: F1={ft['F1']} F2={ft['F2']} F3={ft['F3']}")
    return results


def greedy_follow(cfg, weekly, bounds, etf, start_date, delta, max_n=45, min_w=0.15):
    """贪心追击: 每步在当前(已扰动)回测中找最早一个 ≥min_w 持有目标资产且未扰动的周,
    注入 −δ 后重跑; 记录 F1/F2/F3 首次触发时的 (n, Σ|Δr|)。"""
    edits: dict = {}
    first = {"F1": None, "F2": None, "F3": None}
    last_row = None
    for step in range(max_n + 1):
        pert, applied = perturb_nav(weekly, edits, bounds) if edits else (weekly.copy(), [])
        ev = evaluate(cfg, pert, "a1g")
        last_row = pack(ev, applied)
        for f in first:
            if first[f] is None and ev[f]:
                first[f] = {"n": len(edits), "sum_abs_dr": last_row["sum_abs_dr"],
                            "sharpe": ev["sharpe"], "maxdd": ev["maxdd"],
                            "ew_sharpe": ev["ew_sharpe"],
                            "stop_loss_weeks": ev["stop_loss_weeks"]}
        if all(first.values()) or step == max_n:
            break
        ns = ev["_res"].nav_series
        cand = [dt for dt in ns.index
                if dt >= start_date and ns.loc[dt, f"weight_{etf}"] >= min_w
                and (dt, etf) not in edits]
        if not cand:
            break  # 策略彻底逃离该资产, 攻击无法继续
        edits[(cand[0], etf)] = -delta
    return {"delta": delta, "n_final": len(edits), "final": last_row, "first_trigger": first}


# ---------------------------------------------------------------- A2 单周崩盘
def attack_A2(cfg, weekly, bounds, base_res):
    print("\n=== A2 单周崩盘攻击 ===")
    ns = base_res.nav_series
    off_names = [c for c in weekly.columns if classify_etfs([c])[0]]
    off_w = sum(ns[f"weight_{c}"] for c in off_names)
    d0 = off_w.idxmax()
    held = [c for c in off_names if ns.loc[d0, f"weight_{c}"] > 0.05]
    w_held = {c: float(ns.loc[d0, f"weight_{c}"]) for c in held}
    print(f"  目标周: {d0.date()} 进攻仓位 {float(off_w.loc[d0]):.2%} 持仓 {w_held}")
    dates = list(ns.index)
    i0 = dates.index(d0)

    results = {"crash_start": str(d0.date()), "holdings": w_held, "frontier": [], "min": {}}

    def make_edits(x, n):
        e = {}
        for k in range(n):
            d = dates[min(i0 + k, len(dates) - 1)]
            for c in held:
                e[(d, c)] = e.get((d, c), 0.0) - x
        return e

    found = {"F1": None, "F2": None, "F3": None}
    for n in (1, 2, 3):
        # x 上限: 保证扰动后仍在历史范围内的上界近似 (逐资产 clamp 会自动兜底)
        x_hi = 0.20
        pert, applied = perturb_nav(weekly, make_edits(x_hi, n), bounds)
        ev_hi = evaluate(cfg, pert, "a2")
        print(f"  n={n} X=0.20(封顶探测) Sh={ev_hi['sharpe']:.3f} DD={ev_hi['maxdd']:.3f} "
              f"stopwk={ev_hi['stop_loss_weeks']} F1={ev_hi['F1']} F2={ev_hi['F2']} F3={ev_hi['F3']}")
        row_hi = pack(ev_hi, applied, {"n": n, "X": x_hi, "note": "clamp到历史min为界"})
        results["frontier"].append(row_hi)
        for f in ("F1", "F2", "F3"):
            if found[f] is not None or not ev_hi[f]:
                continue
            _, refined = bisect_delta(lambda x: make_edits(x, n), f, cfg, weekly, bounds,
                                      lo=0.0, hi=x_hi, tag="a2b", iters=7)
            if refined:
                refined["n"] = n
                refined["X"] = refined.pop("delta")
                found[f] = refined
                print(f"  [A2 min {f}] n={n} X={refined['X']:.4f} Σ|Δr|={refined['sum_abs_dr']:.3f} "
                      f"Sh={refined['sharpe']:.3f} DD={refined['maxdd']:.3f} stopwk={refined['stop_loss_weeks']}")
    results["min"] = found

    # 止损价值检验: 对 F1 最小扰动场景, 关闭止损重跑对比
    probe = found["F1"] or found["F3"] or found["F2"]
    if probe:
        import dataclasses
        cfg_nosl = dataclasses.replace(cfg, stop_loss=1.0)  # 永不触发
        pert, _ = perturb_nav(weekly, make_edits(probe["X"], probe["n"]), bounds)
        ev_ns = evaluate(cfg_nosl, pert, "a2n")
        results["stop_loss_check"] = {
            "with_stop": {"maxdd": probe["maxdd"], "sharpe": probe["sharpe"],
                          "stop_loss_weeks": probe["stop_loss_weeks"]},
            "without_stop": {"maxdd": ev_ns["maxdd"], "sharpe": ev_ns["sharpe"]},
        }
        print(f"  [止损检验] 有止损 DD={probe['maxdd']:.3f} / 无止损 DD={ev_ns['maxdd']:.3f}")
    return results


# ---------------------------------------------------------------- A3 调仓鞭打
def attack_A3(cfg, weekly, bounds, base_res):
    print("\n=== A3 调仓鞭打攻击 ===")
    # 探针: 离线重算评分, 找 gap 接近 eff_margin 的临界周
    cfg_dict = {"factors": {
        "mom_window": cfg.mom_window, "vol_window": cfg.vol_window,
        "vol_ddof": cfg.vol_ddof, "pe_window_years": cfg.pe_window_years,
        "ewma_factors_enabled": cfg.ewma_factors_enabled,
        "ewma_mom_halflife": cfg.ewma_mom_halflife, "ewma_vol_halflife": cfg.ewma_vol_halflife,
        "vol_taper_enabled": cfg.vol_taper_enabled, "vol_taper_window": cfg.vol_taper_window,
        "vol_taper_len": cfg.vol_taper_len}}
    fac = compute_all_factors(weekly, None, cfg_dict)
    mom, vol = fac["momentum"], fac["volatility"]
    off_idx, _, _ = classify_etfs(list(weekly.columns))
    names = list(weekly.columns)
    scores = cfg.mom_w * mom - cfg.vol_w * vol

    gap_hist, criticals = [], []
    start_idx = max(cfg.vol_taper_window, cfg.mom_window)
    for i in range(start_idx, len(weekly) - 1):
        row = [(scores.iloc[i, j], j) for j in off_idx if not np.isnan(scores.iloc[i, j])]
        if len(row) <= cfg.top_n:
            continue
        row.sort(reverse=True)
        gap = row[cfg.top_n - 1][0] - row[cfg.top_n][0]
        gap_hist.append(gap)
        hist3 = gap_hist[-cfg.dynamic_margin_window:]
        eff_margin = cfg.score_margin + (cfg.dynamic_margin_sensitivity * float(np.std(hist3))
                                         if len(hist3) >= 2 else 0.0)
        if abs(gap - eff_margin) < 0.01:
            criticals.append({"date": weekly.index[i], "gap": gap, "eff_margin": eff_margin,
                              "rank2": names[row[cfg.top_n - 1][1]], "rank3": names[row[cfg.top_n][1]],
                              "i": i})
    print(f"  临界周 (|gap−eff_margin|<0.01): {len(criticals)} 个")

    results = {"n_critical_weeks": len(criticals), "grid": [], "min": {}}
    if not criticals:
        return results
    # 均匀抽取 K 个临界周: 每个临界周 t 上, 给 rank3(挑战者) 的 t-1 周收益 +ε (推它上位),
    # 并在 t+1 周给挑战者 −ε (新持仓走弱) → 一次鞭打 2 周修改
    dates = list(weekly.index)

    def make_edits(K, eps):
        picks = criticals[:: max(1, len(criticals) // K)][:K]
        e = {}
        for c in picks:
            i = c["i"]
            e[(dates[i], c["rank3"])] = e.get((dates[i], c["rank3"]), 0.0) + eps
            if i + 1 < len(dates):
                e[(dates[i + 1], c["rank3"])] = e.get((dates[i + 1], c["rank3"]), 0.0) - eps
        return e

    hits = {"F1": [], "F2": [], "F3": []}
    for K in (5, 10, 20, min(40, len(criticals))):
        for eps in (0.005, 0.01, 0.02):
            pert, applied = perturb_nav(weekly, make_edits(K, eps), bounds)
            ev = evaluate(cfg, pert, "a3")
            row = pack(ev, applied, {"K": K, "eps": eps})
            results["grid"].append(row)
            for f in hits:
                if ev[f]:
                    hits[f].append(row)
            print(f"  K={K:2d} ε={eps:.3f} Sh={ev['sharpe']:.3f} DD={ev['maxdd']:.3f} "
                  f"rebal={ev['rebalance_count']} (基线 {base_res.metrics['rebalance_count']}) "
                  f"F2={ev['F2']} F3={ev['F3']}")
    for f, rows in hits.items():
        results["min"][f] = min(rows, key=lambda r: r["sum_abs_dr"]) if rows else None
    return results


# ---------------------------------------------------------------- A4 相关性灰区同步阴跌
def attack_A4(cfg, weekly, bounds, base_res):
    print("\n=== A4 相关性灰区同步阴跌 ===")
    runs = find_calm_run(base_res)
    run = runs[0]
    off_names = [c for c in weekly.columns if classify_etfs([c])[0]]
    print(f"  平静区间: {run[0].date()} ~ {run[-1].date()} ({len(run)} 周), 注入资产: {off_names}")
    # 共同阴跌分量: 每周对三只进攻资产同时 −δ, 但按 0.7/1.0/1.3 轮换缩放
    # (幅度错开 → 注入分量的截面相关适中, 避免 EWMA 相关直冲 0.6)
    scale_cycle = [0.7, 1.0, 1.3]

    def make_edits(delta, n):
        e = {}
        for k, d in enumerate(run[:n]):
            for j, c in enumerate(off_names):
                s = scale_cycle[(k + j) % 3]
                e[(d, c)] = e.get((d, c), 0.0) - delta * s
        return e

    grid_n = [4, 6, 8, 10, 13, 17, 22, 26]
    grid_n = [n for n in grid_n if n <= len(run)]
    grid_d = [0.005, 0.0075, 0.01, 0.015, 0.02]
    results = {"window_start": str(run[0].date()), "window_len_available": len(run),
               "grid": [], "min": {}}
    hits = {"F1": [], "F2": [], "F3": []}
    base_ns = base_res.nav_series
    for n in grid_n:
        for d in grid_d:
            pert, applied = perturb_nav(weekly, make_edits(d, n), bounds)
            ev = evaluate(cfg, pert, "a4")
            # 防御是否被激活: 攻击窗口内 def_ratio 峰值 vs 基线同窗峰值
            win = [x for x in run[:n]]
            ns = ev["_res"].nav_series
            def_peak = float(ns.loc[ns.index.isin(win), "def_ratio"].max())
            base_peak = float(base_ns.loc[base_ns.index.isin(win), "def_ratio"].max())
            row = pack(ev, applied, {"n": n, "delta": d,
                                     "def_peak_in_window": round(def_peak, 3),
                                     "def_peak_baseline": round(base_peak, 3)})
            results["grid"].append(row)
            for f in hits:
                if ev[f]:
                    hits[f].append(row)
            print(f"  n={n:2d} δ={d:.4f} Sh={ev['sharpe']:.3f} DD={ev['maxdd']:.3f} "
                  f"defpeak={def_peak:.2f}(基线{base_peak:.2f}) F1={ev['F1']} F2={ev['F2']} F3={ev['F3']}")
    for f, rows in hits.items():
        if not rows:
            results["min"][f] = None
            continue
        cand = min(rows, key=lambda r: r["sum_abs_dr"])
        n = cand["n"]
        _, refined = bisect_delta(lambda d: make_edits(d, n), f, cfg, weekly, bounds,
                                  lo=0.0, hi=cand["delta"], tag="a4b")
        best_row = refined if refined else cand
        best_row["n"] = n
        results["min"][f] = best_row
        print(f"  [A4 min {f}] n={n} δ={best_row['delta']:.4f} Σ|Δr|={best_row['sum_abs_dr']:.3f} "
              f"Sh={best_row['sharpe']:.3f} DD={best_row['maxdd']:.3f}")
    return results


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    cfg, weekly = load_base()
    bounds = hist_bounds(weekly)
    print("历史周收益边界 (貌似合理约束):")
    for c, (lo, hi) in bounds.items():
        print(f"  {c}: [{lo:+.4f}, {hi:+.4f}]")

    with contextlib.redirect_stdout(io.StringIO()):
        base_res = run_backtest(cfg)
    bm = base_res.metrics
    base_ev_ew = ew_metrics_on(weekly, base_res, cfg.risk_free_rate)
    print(f"\n基线: Sharpe={bm['sharpe_ratio']:.3f} 年化={bm['annual_return']:.2%} "
          f"MaxDD={bm['max_drawdown']:.2%} | 等权 Sharpe={base_ev_ew['sharpe_ratio']:.3f}")

    out = {
        "meta": {
            "config": "config/strategy_v4_4.yaml",
            "criteria": {"F1": "MaxDD>12%", "F2": "Sharpe<同扰动数据等权Sharpe", "F3": "Sharpe<1.0"},
            "bounds": {k: [round(v[0], 4), round(v[1], 4)] for k, v in bounds.items()},
        },
        "baseline": {"sharpe": bm["sharpe_ratio"], "annual": bm["annual_return"],
                     "maxdd": bm["max_drawdown"], "ew_sharpe": base_ev_ew["sharpe_ratio"]},
    }
    out["A1_slow_decline"] = attack_A1(cfg, weekly, bounds, base_res)
    out["A2_crash"] = attack_A2(cfg, weekly, bounds, base_res)
    out["A3_whipsaw"] = attack_A3(cfg, weekly, bounds, base_res)
    out["A4_corr_gray_decline"] = attack_A4(cfg, weekly, bounds, base_res)
    out["meta"]["total_backtests"] = _run_counter["n"]
    out["meta"]["elapsed_sec"] = round(time.time() - t0, 1)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nJSON → {OUT_JSON}")
    print(f"总回测次数: {_run_counter['n']}, 用时 {out['meta']['elapsed_sec']}s")

    # 清理临时目录残留
    if TMP_DIR.exists():
        for p in TMP_DIR.glob("_tmp_*.csv"):
            p.unlink()
        try:
            TMP_DIR.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
