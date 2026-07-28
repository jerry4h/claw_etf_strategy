#!/usr/bin/env python3
"""v4.0 对抗性鲁棒性评估 — 策略失效边界定位。

思路(类比 CV adversarial examples)：从 realized 数据结构出发,在 DGP 参数空间里
找"最小扰动使策略 Sharpe 翻到 0(或指定阈值)"的临界值和方向。
临界距离 = 安全半径(robustness margin),方向 = 结构性脆弱性所在。

方法：
  1. CCC-GARCH DGP(已验证波动聚集有效) → 合成数据 → 真实 run_backtest → Sharpe
  2. 逐轴二分法：沿每轴从 realized(mult=1.0) 向两端搜索 Sharpe=threshold 的临界 mult
  3. 最脆弱 2 轴 2D 密集扫描 → 失效边界等值线图

5 轴定义(每轴 mult=1.0 = realized 数据结构)：
  ρ(rho_mult):  趋势持续性 — diag(A) 缩放
  σ(sig_mult):  波动水平 — GARCH 无条件方差缩放
  c(c_mult):    截面相关 — 条件相关 R 缩放
  μdef(mudef_mult): 防御漂移 — mu[DEF] 缩放
  μoff(muoff_mult): 进攻漂移 — mu[OFF] 缩放

用法:
  python scripts/adversarial_robustness.py                # 全部(逐轴+2D)
  python scripts/adversarial_robustness.py --threshold 0.5  # 自定义失效阈值
  python scripts/adversarial_robustness.py --json         # JSON 输出
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

_spec = importlib.util.spec_from_file_location("dm", PROJ / "scripts" / "data_manifold.py")
dm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dm)

from src.backtest import run_backtest
from arch import arch_model

OFF_IDX = dm.OFF_IDX
DEF_IDX = dm.DEF_IDX
ETF_NAMES = dm.ETF_NAMES
OUT = PROJ / "output" / "adversarial"

AXES = ["rho_mult", "c_mult", "sig_mult", "mudef_mult", "muoff_mult"]
AX_LABEL = {
    "rho_mult": "趋势持续性(×realized)",
    "c_mult": "截面相关(×realized)",
    "sig_mult": "波动水平(×realized)",
    "mudef_mult": "防御漂移(×realized)",
    "muoff_mult": "进攻漂移(×realized)",
}
REALIZED = {a: 1.0 for a in AXES}

# 固定压力情景集(纳入常规评估的对抗稳定性指标; 每个含明确金融含义)
STRESS_SCENARIOS = {
    "baseline":          {},                                      # realized 参照
    "vol_stress":        {"sig_mult": 1.2},                       # 波动放大20%(2018/2022级)
    "offense_cooldown":  {"muoff_mult": 0.8},                     # 进攻收益降20%(牛市降温)
    "bond_bear":         {"mudef_mult": 0.5},                     # 防御漂移减半(债牛结束)
    "decorrelation":     {"c_mult": 0.77},                        # 相关降低(分散噪声化)
    "stagflation":       {"sig_mult": 1.2, "muoff_mult": 0.8},    # 滞胀式组合冲击
}
# 情景 → 主控机制映射(不同鲁棒方向由不同机制控制; 分机制门禁而非单一标量)
#   vol_defense : Layer3 波动择时(nasdaq-vol 三档防御) — 抗波动放大
#   selection   : Layer1 打分/动量选择 — 抗进攻资产收益退化
#   defense_asset: DefAlloc 防御标的(红利低波 vs 国债) — 抗债牛结束
#   dispersion  : inv-vol 加权/轮动 — 抗相关结构变化
#   composite   : 复合冲击(多机制同时承压)
SCENARIO_MECHANISM = {
    "vol_stress":        "vol_defense",
    "offense_cooldown":  "selection",
    "bond_bear":         "defense_asset",
    "decorrelation":     "dispersion",
    "stagflation":       "composite",
}
# 搜索范围(合理边界)
BOUNDS = {
    "rho_mult": (0.0, 3.0),
    "c_mult": (0.0, 2.5),
    "sig_mult": (0.4, 2.5),
    "mudef_mult": (0.0, 2.5),
    "muoff_mult": (0.0, 2.5),
}


def fit_garch(resid):
    """逐资产 GARCH(1,1) on VAR 残差 → per-asset params + CCC 相关阵。"""
    T, k = resid.shape
    gp = []
    std = np.zeros_like(resid)
    for j in range(k):
        r = resid[:, j] * 100.0
        am = arch_model(r, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
        res = am.fit(disp="off")
        o = float(res.params["omega"]); a = float(res.params["alpha[1]"]); b = float(res.params["beta[1]"])
        if a + b > 0.995:
            b = 0.995 - a
        cv = np.asarray(res.conditional_volatility)
        std[:, j] = r / cv
        gp.append({"omega": o / 1e4, "alpha": a, "beta": b})
    R = np.corrcoef(std, rowvar=False)
    return gp, R


def _scale_corr(R, c_mult):
    lam_min = float(np.linalg.eigvalsh(R).min())
    cap = 1.0 / (1.0 - lam_min) if lam_min < 1 else 2.0
    c_mult = min(c_mult, max(0.0, cap - 1e-3))
    M = (1 - c_mult) * np.eye(len(R)) + c_mult * R
    M = (M + M.T) / 2
    ev = np.linalg.eigvalsh(M)
    if ev.min() < 1e-9:
        M += np.eye(len(R)) * (1e-9 - ev.min())
    d = np.sqrt(np.diag(M))
    return M / np.outer(d, d)


def gen_garch(mu, A, R, nu, gp, params, T, seed):
    """CCC-GARCH 生成,支持 5 轴旋钮(含 muoff_mult)。"""
    rng = np.random.default_rng(seed)
    A2, *_ = dm._stationary_A(A, params.get("rho_mult", 1.0))
    Rc = _scale_corr(R, params.get("c_mult", 1.0))
    try:
        L = np.linalg.cholesky(Rc)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(Rc + np.eye(len(Rc)) * 1e-9)
    mu2 = mu.copy()
    mu2[DEF_IDX] = mu[DEF_IDX] * params.get("mudef_mult", 1.0)
    mu2[OFF_IDX] = mu[OFF_IDX] * params.get("muoff_mult", 1.0)
    k = len(mu)
    sm2 = params.get("sig_mult", 1.0) ** 2
    omega = np.array([g["omega"] for g in gp]) * sm2
    alpha = np.array([g["alpha"] for g in gp])
    beta = np.array([g["beta"] for g in gp])
    uncond = omega / np.maximum(1e-6, 1 - alpha - beta)
    h = uncond.copy()
    eps2 = uncond.copy()
    unit = np.sqrt(max(nu - 2, 0.5) / nu)
    r = np.zeros((T, k)); r_prev = np.zeros(k)
    for t in range(T):
        h = omega + alpha * eps2 + beta * h
        zc = rng.standard_normal(k) @ L.T
        g = rng.chisquare(nu)
        z = zc * np.sqrt(nu / g) * unit
        eps = np.sqrt(h) * z
        r[t] = mu2 + A2 @ r_prev + eps
        eps2 = eps ** 2
        r_prev = r[t]
    return r


def eval_sharpe(mu, A, R, nu, gp, params, T, real_dates, first_nav, cfg, seeds=(11, 22, 33, 44, 55)):
    """多 seed 生成+回测,返回中位数 Sharpe。"""
    import os as _os
    sharpes = []
    for s in seeds:
        r = gen_garch(mu, A, R, nu, gp, params, T, s)
        nav_df = dm.build_nav_df(r, real_dates, first_nav)
        # P1-3 修: 加 pid 后缀防未来并行调用同一 seed 时互相覆盖临时文件
        tmp = OUT / f"_synth_{s}_{_os.getpid()}.csv"
        nav_df.to_csv(tmp, encoding="utf-8")
        try:
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                res = run_backtest(cfg, start_date=dm.START_DATE, data_path=str(tmp))
            if not res.nav_series.empty:
                sharpes.append(float(res.metrics["sharpe_ratio"]))
        finally:
            if tmp.exists():
                import os; os.remove(tmp)
    return float(np.median(sharpes)) if sharpes else np.nan


def bisect_axis(axis, direction, mu, A, R, nu, gp, T, real_dates, first_nav, cfg,
                threshold=0.0, tol=0.03, max_iter=12):
    """沿单轴从 realized(1.0) 向 direction(+1/-1) 二分搜索 Sharpe=threshold 的临界 mult。
    返回 (critical_mult, sharpe_at_critical) 或 None(轴范围内未翻转)。"""
    lo, hi = 1.0, BOUNDS[axis][1] if direction > 0 else BOUNDS[axis][0]
    if direction < 0:
        lo, hi = hi, lo  # ensure lo < hi numerically by swapping meaning
        lo, hi = min(lo, hi), max(lo, hi)

    # 先检查边界是否翻转
    params_lo = dict(REALIZED, **{axis: 1.0})
    params_hi = dict(REALIZED, **{axis: hi if direction > 0 else lo})
    sh_start = eval_sharpe(mu, A, R, nu, gp, params_lo, T, real_dates, first_nav, cfg)
    sh_end = eval_sharpe(mu, A, R, nu, gp, params_hi, T, real_dates, first_nav, cfg)

    if sh_start <= threshold:
        return 1.0, sh_start  # 已在阈值以下
    if sh_end > threshold:
        return None  # 整个范围内未翻转

    # 二分
    a, b = 1.0, (hi if direction > 0 else lo)
    for _ in range(max_iter):
        mid = (a + b) / 2
        params_mid = dict(REALIZED, **{axis: mid})
        sh_mid = eval_sharpe(mu, A, R, nu, gp, params_mid, T, real_dates, first_nav, cfg)
        if sh_mid > threshold:
            a = mid
        else:
            b = mid
        if abs(b - a) < tol:
            break
    critical = (a + b) / 2
    return critical, eval_sharpe(mu, A, R, nu, gp, dict(REALIZED, **{axis: critical}), T, real_dates, first_nav, cfg)


def _eval_strat_ew(mu, A, R, nu, gp, params, T, real_dates, first_nav, cfg, seeds=(11, 22, 33)):
    """在同一批合成数据上评估 策略 vs 等权每周再平衡, 返回中位数指标 dict。

    返回: {strat_sharpe, ew_sharpe, strat_maxdd, ew_maxdd, strat_annual, ew_annual}
    - sharpe: 夏普比率 (原有口径, 向后兼容)
    - maxdd:  最大回撤 (多目标框架的回撤约束 DD≤D_max 用此)
    - annual: 年化收益 (收益>等权约束用此)
    """
    from src.backtest import run_backtest, compute_metrics
    from src.data_loader import ETFS
    import io, contextlib, os
    s_sh, e_sh = [], []
    s_dd, e_dd = [], []
    s_an, e_an = [], []
    for seed in seeds:
        r = gen_garch(mu, A, R, nu, gp, params, T, seed)
        nav_df = build_nav_df_local(r, real_dates, first_nav)
        # P1-3 修: 加 pid 后缀防未来并行调用同一 seed 时互相覆盖临时文件
        tmp = OUT / f"_score_{seed}_{os.getpid()}.csv"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        nav_df.to_csv(tmp, encoding="utf-8")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                res = run_backtest(cfg, start_date=dm.START_DATE, data_path=str(tmp))
            if res.nav_series.empty:
                continue
            s_sh.append(res.metrics["sharpe_ratio"])
            s_dd.append(res.metrics["max_drawdown"])
            s_an.append(res.metrics["annual_return"])
            start, end = res.nav_series.index[0], res.nav_series.index[-1]
            cols = [c for c in nav_df.columns if c in ETFS]
            pr = nav_df.loc[start:end, cols].astype(float)
            idx = pr.index
            valid = ~np.isnan(pr.iloc[0].values)
            er = pr.ffill().pct_change().fillna(0.0).values
            rb = np.ones(len(idx))
            for i in range(1, len(idx)):
                rb[i] = rb[i-1] * (1 + float(np.mean(er[i, valid])))
            wr = np.zeros(len(rb)); wr[1:] = rb[1:] / rb[:-1] - 1
            peak = np.maximum.accumulate(rb); dd = (peak - rb) / peak
            import pandas as pd
            df_rb = pd.DataFrame({"nav": rb, "weekly_return": wr, "drawdown": dd,
                                  "def_ratio": 0.0, "turnover": 0.0}, index=idx)
            em = compute_metrics(df_rb, cfg.risk_free_rate)
            e_sh.append(em["sharpe_ratio"])
            e_dd.append(em["max_drawdown"])
            e_an.append(em["annual_return"])
        finally:
            if tmp.exists():
                os.remove(tmp)
    med = lambda xs: float(np.median(xs)) if xs else float("nan")
    return {
        "strat_sharpe": med(s_sh), "ew_sharpe": med(e_sh),
        "strat_maxdd": med(s_dd), "ew_maxdd": med(e_dd),
        "strat_annual": med(s_an), "ew_annual": med(e_an),
    }


def build_nav_df_local(r, real_dates, first_nav):
    return dm.build_nav_df(r, real_dates, first_nav)


def robustness_score(cfg, seeds=(11, 22, 33)):
    """对抗稳定性指标——固定压力情景集下策略 vs 等权。可复现、快、可比较。

    返回 dict (向后兼容原字段 + 多目标/分机制扩展):
    - pass_rate      : 压力情景中 策略 Sharpe > 等权 的比例 (原口径, 向后兼容)
    - pass_rate_return: 压力情景中 策略 年化收益 > 等权 的比例 (多目标框架的收益约束口径)
    - worst_sharpe/scenario : 最脆弱情景及其策略 Sharpe
    - worst_maxdd    : 全情景(含 baseline)策略最大回撤的最大值 (回撤约束 DD≤D_max 用此)
    - baseline_sharpe/baseline_retention : baseline Sharpe 与压力情景平均保留率
    - by_mechanism   : 按主控机制分组的分维门禁(不同鲁棒方向由不同机制控制)
    - scenarios{...} : 逐情景明细(strategy/ew_rebal/beats_ew + maxdd/annual)
    """
    nav, wk, w_rets = dm.load_real()
    mu, A, Sigma, nu, resid, coords = dm.fit_var_t(w_rets)
    gp, R = fit_garch(resid)
    real_dates = wk.index
    first_nav = wk.iloc[0].values
    T = len(w_rets)

    results = {}
    for name, overrides in STRESS_SCENARIOS.items():
        params = dict(REALIZED, **overrides)
        m = _eval_strat_ew(mu, A, R, nu, gp, params, T, real_dates, first_nav, cfg, seeds)
        results[name] = {
            "strategy": m["strat_sharpe"], "ew_rebal": m["ew_sharpe"],
            "beats_ew": bool(m["strat_sharpe"] > m["ew_sharpe"]),
            "beats_ew_return": bool(m["strat_annual"] > m["ew_annual"]),
            "strat_maxdd": m["strat_maxdd"], "ew_maxdd": m["ew_maxdd"],
            "strat_annual": m["strat_annual"], "ew_annual": m["ew_annual"],
            "mechanism": SCENARIO_MECHANISM.get(name, "baseline"),
        }

    baseline_sh = results["baseline"]["strategy"]
    stress = {k: v for k, v in results.items() if k != "baseline"}
    n_pass = sum(1 for v in stress.values() if v["beats_ew"])
    n_pass_ret = sum(1 for v in stress.values() if v["beats_ew_return"])
    worst_name = min(stress, key=lambda k: stress[k]["strategy"])
    avg_stress = float(np.mean([v["strategy"] for v in stress.values()]))
    worst_maxdd = float(np.nanmax([v["strat_maxdd"] for v in results.values()]))

    # --- 分机制门禁 ---
    by_mechanism = {}
    for name, v in stress.items():
        mech = v["mechanism"]
        by_mechanism.setdefault(mech, {"scenarios": []})["scenarios"].append(name)
    for mech, d in by_mechanism.items():
        sc_list = d["scenarios"]
        d["n"] = len(sc_list)
        d["pass_rate"] = sum(1 for s in sc_list if stress[s]["beats_ew"]) / len(sc_list)
        d["pass_rate_return"] = sum(1 for s in sc_list if stress[s]["beats_ew_return"]) / len(sc_list)
        d["worst_sharpe"] = float(min(stress[s]["strategy"] for s in sc_list))
        d["worst_maxdd"] = float(np.nanmax([stress[s]["strat_maxdd"] for s in sc_list]))

    return {
        "pass_rate": n_pass / len(stress),
        "pass_rate_return": n_pass_ret / len(stress),
        "n_pass": n_pass,
        "n_pass_return": n_pass_ret,
        "n_stress": len(stress),
        "worst_sharpe": stress[worst_name]["strategy"],
        "worst_scenario": worst_name,
        "worst_maxdd": worst_maxdd,
        "baseline_sharpe": baseline_sh,
        "baseline_retention": avg_stress / baseline_sh if baseline_sh else float("nan"),
        "by_mechanism": by_mechanism,
        "scenarios": results,
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description="v4.0 对抗性鲁棒性评估")
    p.add_argument("--threshold", type=float, default=0.0, help="失效阈值(Sharpe, default=0)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--score", action="store_true", help="快速对抗稳定性指标(固定压力情景集)")
    args = p.parse_args()
    threshold = args.threshold

    if args.score:
        from src.strategy import load_config
        cfg = load_config(PROJ / "config" / "strategy_v4_3.yaml")
        sc = robustness_score(cfg)
        if args.json:
            print(json.dumps(sc, ensure_ascii=False, indent=2, default=str))
        else:
            print("=" * 60)
            print(" 对抗稳定性指标 (固定压力情景集)")
            print("=" * 60)
            print(f" baseline Sharpe: {sc['baseline_sharpe']:.3f}")
            print(f" {'情景':<18s} {'策略Sh':>7s} {'等权Sh':>7s} {'策略DD':>7s} {'胜Sh':>5s} {'胜Ret':>5s}")
            print("-" * 56)
            for name, v in sc["scenarios"].items():
                mark = "Y" if v["beats_ew"] else "-"
                markr = "Y" if v.get("beats_ew_return") else "-"
                tag = " (baseline)" if name == "baseline" else ""
                print(f" {name:<18s} {v['strategy']:>7.3f} {v['ew_rebal']:>7.3f} "
                      f"{v.get('strat_maxdd', float('nan')):>7.2%} {mark:>5s} {markr:>5s}{tag}")
            print("-" * 56)
            print(f" pass_rate(Sharpe): {sc['n_pass']}/{sc['n_stress']} ({sc['pass_rate']*100:.0f}%)  "
                  f"pass_rate(收益): {sc['n_pass_return']}/{sc['n_stress']} ({sc['pass_rate_return']*100:.0f}%)")
            print(f" 最脆弱: {sc['worst_scenario']}(Sh={sc['worst_sharpe']:.3f})  "
                  f"全情景最大回撤: {sc['worst_maxdd']:.2%}  保留率: {sc['baseline_retention']*100:.0f}%")
            print("-" * 56)
            print(" 分机制门禁 (机制: 胜率Sh/胜率Ret worstSh worstDD):")
            for mech, d in sc["by_mechanism"].items():
                print(f"   {mech:<14s} {d['pass_rate']*100:>3.0f}%/{d['pass_rate_return']*100:>3.0f}%  "
                      f"Sh={d['worst_sharpe']:>6.3f}  DD={d['worst_maxdd']:>6.2%}  "
                      f"({','.join(d['scenarios'])})")
        return


    OUT.mkdir(parents=True, exist_ok=True)
    nav, wk, w_rets = dm.load_real()
    mu, A, Sigma, nu, resid, coords = dm.fit_var_t(w_rets)
    gp, R = fit_garch(resid)
    real_dates = wk.index
    first_nav = wk.iloc[0].values
    T = len(w_rets)
    cfg = dm._cfg()

    # realized Sharpe (DGP = realized 结构)
    sh_realized = eval_sharpe(mu, A, R, nu, gp, REALIZED, T, real_dates, first_nav, cfg)
    print(f"realized (all mult=1.0) median Sharpe = {sh_realized:.3f}")
    print(f"失效阈值 = {threshold}")
    print()

    # --- Phase 1: 逐轴二分法 ---
    print("=" * 70)
    print(" Phase 1: 逐轴对抗搜索 (从 realized 出发, 找 Sharpe={threshold} 临界)")
    print("=" * 70)
    results = {}
    for axis in AXES:
        r_pos = bisect_axis(axis, +1, mu, A, R, nu, gp, T, real_dates, first_nav, cfg, threshold)
        r_neg = bisect_axis(axis, -1, mu, A, R, nu, gp, T, real_dates, first_nav, cfg, threshold)
        margin_pos = (r_pos[0] - 1.0) if r_pos else None
        margin_neg = (1.0 - r_neg[0]) if r_neg else None
        # 取较近的那侧作为"安全半径"
        if margin_pos is not None and margin_neg is not None:
            margin = min(abs(margin_pos), abs(margin_neg))
            direction = "+" if abs(margin_pos) <= abs(margin_neg) else "-"
        elif margin_pos is not None:
            margin = abs(margin_pos); direction = "+"
        elif margin_neg is not None:
            margin = abs(margin_neg); direction = "-"
        else:
            margin = float("inf"); direction = "N/A"
        results[axis] = {
            "margin": margin, "direction": direction,
            "critical_pos": r_pos[0] if r_pos else None,
            "critical_neg": r_neg[0] if r_neg else None,
        }
        sym = "∞(范围内不翻转)" if margin == float("inf") else f"{margin:.3f}"
        cp = f"{r_pos[0]:.2f}" if r_pos else "N/A"
        cn = f"{r_neg[0]:.2f}" if r_neg else "N/A"
        print(f"  {AX_LABEL[axis]:<22s}  安全半径={sym:>8s}  临界: +→{cp:>5s}  -→{cn:>5s}")

    # 排序: 最脆弱轴
    ranked = sorted(results.items(), key=lambda x: x[1]["margin"])
    print(f"\n  最脆弱轴: {AX_LABEL[ranked[0][0]]} (margin={ranked[0][1]['margin']:.3f})")
    print(f"  次脆弱轴: {AX_LABEL[ranked[1][0]]} (margin={ranked[1][1]['margin']:.3f})")

    # --- Phase 2: 最脆弱 2 轴 2D 密集扫描 ---
    ax1, ax2 = ranked[0][0], ranked[1][0]
    print(f"\n{'='*70}")
    print(f" Phase 2: 2D 等值线扫描 ({AX_LABEL[ax1]} × {AX_LABEL[ax2]})")
    print(f"{'='*70}")
    n_grid = 10
    grid1 = np.linspace(BOUNDS[ax1][0] + 0.1, BOUNDS[ax1][1], n_grid)
    grid2 = np.linspace(BOUNDS[ax2][0] + 0.1, BOUNDS[ax2][1], n_grid)
    sharpe_map = np.zeros((n_grid, n_grid))
    for i, v1 in enumerate(grid1):
        for j, v2 in enumerate(grid2):
            params = dict(REALIZED, **{ax1: v1, ax2: v2})
            sharpe_map[i, j] = eval_sharpe(mu, A, R, nu, gp, params, T, real_dates, first_nav, cfg,
                                           seeds=(11, 22))
        print(f"  row {i+1}/{n_grid} done")

    # 保存结果
    output = {
        "realized_sharpe": sh_realized,
        "threshold": threshold,
        "per_axis": results,
        "ranked_axes": [(a, r) for a, r in ranked],
        "grid_2d": {
            "ax1": ax1, "ax2": ax2,
            "grid1": grid1.tolist(), "grid2": grid2.tolist(),
            "sharpe_map": sharpe_map.tolist(),
        },
    }
    out_path = OUT / "adversarial_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已存: {out_path}")

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    else:
        # 简要 2D 结果
        print(f"\n  2D Sharpe 范围: [{sharpe_map.min():.3f}, {sharpe_map.max():.3f}]")
        below = (sharpe_map < threshold).sum()
        print(f"  Sharpe < {threshold} 的格点: {below}/{n_grid**2} ({below/n_grid**2*100:.1f}%)")
        # realized 点在 grid 中的位置
        print(f"  realized 点 ({ax1}=1.0, {ax2}=1.0) Sharpe ≈ {sh_realized:.3f}")


if __name__ == "__main__":
    main()
