"""
交付物 B: Layer 3.5 相关性口径 E1 探针 —— 全池 max corr vs 实际持仓对 corr。

研究问题
    生产代码 off_idx 取全进攻池 (backtest.py:194 classify_etfs), Layer 3.5 用
    全池 pairwise max EWMA |corr| 触发防御 boost。但策略每周只持有 TOP2。
    若高相关的那一对中有一只未被持有 (被当作"同一风险的代表"排除), 实际持仓
    分散度并未下降 —— 此时 boost 是否属于误触发?

    本周 (信号日 2026-08-21) 正是此情形: 驱动对 = 纳指~中证500 (0.646), 但
    中证500 未持有; 实际持仓对 纳指~黄金 = 0.437 < 阈值 0.45。

对计划的一处数学更正
    计划原设三臂 (corr_pool / corr_held / corr_max=两者取大)。因持仓对必为全池
    对的子集, corr_held <= corr_pool 恒成立, 故 corr_max ≡ corr_pool, 第三臂
    无信息量。改为两臂对比 + 增量回归检验。

方法论偏离说明
    这是时序风险指标而非横截面选股因子, 故不用横截面 rank_IC, 改用时序 Spearman
    + Newey-West HAC t 值 (滞后 4, 修正重叠窗口引致的自相关)。

只读脚本: 不修改 src/, 不改配置, 不提交实验数据。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, rankdata, spearmanr

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.backtest import run_backtest                                      # noqa: E402
from src.data_loader import classify_etfs, load_nav_data, resample_weekly  # noqa: E402
from src.engine_core import compute_crisis_boost_directed                  # noqa: E402
from src.strategy import load_config                                       # noqa: E402

CFG = "config/strategy_v4_6.yaml"
OUT_DIR = PROJ / "output" / "experiments"
FWD = 4          # 前瞻窗口 (周)
NW_LAG = 4       # Newey-West 滞后


def build_weekly(cfg):
    nav_path = Path(cfg.nav_path)
    if not nav_path.is_absolute():
        nav_path = PROJ / nav_path
    weekly = resample_weekly(load_nav_data(nav_path), anchor=cfg.anchor)
    if cfg.start_date:
        weekly = weekly[weekly.index >= pd.to_datetime(cfg.start_date)]
    if cfg.end_date:
        weekly = weekly[weekly.index <= pd.to_datetime(cfg.end_date)]
    return weekly


def ewma_abs_corr(w_rets, i, ja, jb, window, halflife):
    """单配对的 EWMA 加权 |corr|, 口径完全复刻 engine_core._compute_crisis_boost_ewma。

    窗口 [i-window, i) = 决策周 i 之前已完成的 window 个收益 (无前视)。
    """
    if i < window:
        return np.nan
    seg = w_rets[i - window:i, [ja, jb]]
    t = np.arange(window)
    weights = 0.5 ** ((window - 1 - t) / halflife)
    mask = ~(np.isnan(seg[:, 0]) | np.isnan(seg[:, 1]))
    if mask.sum() < 5:
        return np.nan
    x, y = seg[mask, 0], seg[mask, 1]
    w = weights[mask]
    w = w / w.sum()
    xb, yb = float(np.sum(w * x)), float(np.sum(w * y))
    cov = float(np.sum(w * (x - xb) * (y - yb)))
    vx, vy = float(np.sum(w * (x - xb) ** 2)), float(np.sum(w * (y - yb) ** 2))
    c = cov / (np.sqrt(vx * vy) + 1e-12)
    return np.nan if np.isnan(c) else abs(c)


def max_drawdown(rets):
    cum = np.cumprod(1.0 + np.asarray(rets))
    peak = np.maximum.accumulate(cum)
    return float(np.min(cum / peak - 1.0))


def hac_t(y, X_cols, lag=NW_LAG):
    """OLS on ranks + Newey-West HAC t 值。返回 {name: (beta, t, p)}。"""
    yr = rankdata(y)
    Xr = np.column_stack([rankdata(v) for v in X_cols.values()])
    # 标准化便于系数可比
    yr = (yr - yr.mean()) / yr.std()
    Xr = (Xr - Xr.mean(axis=0)) / Xr.std(axis=0)
    X = sm.add_constant(Xr)
    fit = sm.OLS(yr, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    out = {}
    for k, name in enumerate(X_cols.keys(), start=1):
        out[name] = {"beta": float(fit.params[k]), "t": float(fit.tvalues[k]),
                     "p": float(fit.pvalues[k])}
    out["_r2"] = float(fit.rsquared)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(CFG)
    window, halflife = cfg.crisis_corr_window, cfg.crisis_corr_ewma_halflife
    thr, split = cfg.directed_boost_threshold, cfg.directed_boost_corr_split

    weekly = build_weekly(cfg)
    w_prices, w_index = weekly.values, weekly.index
    n_weeks = len(w_index)
    w_rets = np.diff(w_prices, axis=0) / w_prices[:-1]
    etf_names = list(weekly.columns)
    off_idx, _, _ = classify_etfs(etf_names)
    off_names = [etf_names[j] for j in off_idx]
    print(f"  进攻池: {off_names}  (window={window}, halflife={halflife}, thr={thr}, split={split})")

    nav = run_backtest(cfg).nav_series
    start_idx = list(w_index).index(nav.index[0]) - 1
    n_bt = len(nav)
    Wmat = nav[[f"weight_{n}" for n in off_names]].values   # 每周进攻端权重

    rows = []
    for k in range(n_bt):
        i = start_idx + k
        held_local = [c for c in range(len(off_idx)) if Wmat[k, c] > 1e-9]
        if len(held_local) != 2:
            continue                                  # 实测覆盖率 100%, 保留守卫
        # --- 全池 max (现状口径) 与驱动对 ---
        best_c, best_pair = -1.0, None
        for a in range(len(off_idx)):
            for b in range(a + 1, len(off_idx)):
                c = ewma_abs_corr(w_rets, i, off_idx[a], off_idx[b], window, halflife)
                if not np.isnan(c) and c > best_c:
                    best_c, best_pair = c, (a, b)
        if best_pair is None:
            continue
        corr_pool = best_c
        # --- 实际持仓对 ---
        ha, hb = held_local
        corr_held = ewma_abs_corr(w_rets, i, off_idx[ha], off_idx[hb], window, halflife)
        driver_held = set(best_pair) <= set(held_local)

        # --- 前瞻 y: 进攻端 buy-and-hold 组合 (权重在进攻内归一) ---
        if i + FWD > len(w_rets):
            continue
        wts = Wmat[k, held_local] / Wmat[k, held_local].sum()
        fwd = w_rets[i:i + FWD][:, [off_idx[ha], off_idx[hb]]]
        if np.isnan(fwd).any():
            continue
        pr = fwd @ wts
        rows.append({
            "signal_date": str(w_index[i].date()),
            "held": f"{off_names[ha]}+{off_names[hb]}",
            "corr_pool": corr_pool, "corr_held": corr_held,
            "driver_pair_held": driver_held,
            "driver_pair": f"{off_names[best_pair[0]]}~{off_names[best_pair[1]]}",
            "fwd_vol": float(np.std(pr, ddof=1) * np.sqrt(52)),
            "fwd_dd": max_drawdown(pr),
            "fwd_ret": float(np.prod(1 + pr) - 1),
        })
    df = pd.DataFrame(rows).dropna(subset=["corr_pool", "corr_held"])
    print(f"  有效样本 {len(df)} 周 ({df['signal_date'].iloc[0]} ~ {df['signal_date'].iloc[-1]})")

    # ---------- 口径自证: 我的 pool 值必须等于引擎返回的 corr_level ----------
    devs = []
    for k in range(n_bt):
        i = start_idx + k
        _, engine_corr = compute_crisis_boost_directed(w_rets, i, off_idx, cfg)
        mine = max((ewma_abs_corr(w_rets, i, off_idx[a], off_idx[b], window, halflife)
                    for a in range(3) for b in range(a + 1, 3)), default=np.nan)
        if not np.isnan(mine) and engine_corr > 0:
            devs.append(abs(mine - engine_corr))
    max_dev = float(max(devs)) if devs else float("nan")
    assert max_dev < 1e-12, f"pool corr 与引擎不一致, max_dev={max_dev}"
    print(f"  [self-check] pool corr 与 engine_core 逐周一致 (max_dev={max_dev:.2e}, n={len(devs)})")

    result = {"meta": {
        "weeks": len(df), "period": f"{df['signal_date'].iloc[0]} ~ {df['signal_date'].iloc[-1]}",
        "fwd_window": FWD, "nw_lag": NW_LAG, "thr": thr, "split": split,
        "corr_max_arm_dropped": "corr_held <= corr_pool 恒成立, corr_max 与 corr_pool 数学等价",
        "pool_corr_selfcheck_max_dev": max_dev}}

    # ---------- B2: 口径分歧规模 ----------
    pool_fire, held_fire = df["corr_pool"] > thr, df["corr_held"] > thr
    result["b2_scope_divergence"] = {
        "pool_fire_weeks": int(pool_fire.sum()),
        "pool_fire_pct": float(pool_fire.mean() * 100),
        "held_fire_weeks": int(held_fire.sum()),
        "held_fire_pct": float(held_fire.mean() * 100),
        "pool_fire_but_held_not": int((pool_fire & ~held_fire).sum()),
        "pool_fire_but_held_not_pct_of_fires": float((pool_fire & ~held_fire).sum() / max(pool_fire.sum(), 1) * 100),
        "held_fire_but_pool_not": int((held_fire & ~pool_fire).sum()),
        "driver_pair_not_held_among_fires": int((~df.loc[pool_fire, "driver_pair_held"]).sum()),
        "driver_pair_not_held_pct_of_fires": float((~df.loc[pool_fire, "driver_pair_held"]).mean() * 100),
        "mean_gap_pool_minus_held": float((df["corr_pool"] - df["corr_held"]).mean()),
        "driver_pair_freq": df.loc[pool_fire, "driver_pair"].value_counts().to_dict(),
    }
    # 显性危机区 (>split) 的同类统计
    pool_split, held_split = df["corr_pool"] > split, df["corr_held"] > split
    result["b2_split_zone"] = {
        "pool_above_split": int(pool_split.sum()),
        "held_above_split": int(held_split.sum()),
        "pool_above_but_held_not": int((pool_split & ~held_split).sum()),
    }

    # ---------- B3: 预测力 ----------
    b3 = {}
    for yname in ("fwd_vol", "fwd_dd", "fwd_ret"):
        y = df[yname].values
        entry = {}
        for xname in ("corr_pool", "corr_held"):
            rho, _ = spearmanr(df[xname].values, y)
            single = hac_t(y, {xname: df[xname].values})
            entry[xname] = {"spearman_ic": float(rho), "hac_t": single[xname]["t"],
                            "hac_p": single[xname]["p"],
                            "gate_pass": bool(abs(rho) >= 0.03 and abs(single[xname]["t"]) >= 1.5)}
        joint = hac_t(y, {"corr_held": df["corr_held"].values, "corr_pool": df["corr_pool"].values})
        entry["joint_regression"] = {
            "corr_held": joint["corr_held"], "corr_pool": joint["corr_pool"], "r2": joint["_r2"]}
        b3[yname] = entry
    result["b3_predictive"] = b3

    # ---------- B3b: 分组对照 (决定性证据) ----------
    g1 = df[pool_fire & held_fire]                       # 两口径都告警
    g2 = df[pool_fire & ~held_fire]                      # 仅全池告警 (误触发候选)
    g3 = df[~pool_fire]                                  # 都不告警
    groups = {"G1_both_fire": g1, "G2_pool_only": g2, "G3_neither": g3}
    gstats = {}
    for name, g in groups.items():
        gstats[name] = {"weeks": len(g)}
        if len(g) == 0:
            continue
        gstats[name].update({
            "mean_fwd_vol": float(g["fwd_vol"].mean()), "median_fwd_vol": float(g["fwd_vol"].median()),
            "mean_fwd_dd": float(g["fwd_dd"].mean()), "median_fwd_dd": float(g["fwd_dd"].median()),
            "mean_fwd_ret": float(g["fwd_ret"].mean()),
        })
    # G2 vs G3 / G2 vs G1: 未来风险是否真的抬升?
    for a, b in (("G2_pool_only", "G3_neither"), ("G2_pool_only", "G1_both_fire")):
        ga, gb = groups[a], groups[b]
        if len(ga) > 5 and len(gb) > 5:
            for yname in ("fwd_vol", "fwd_dd"):
                u, p = mannwhitneyu(ga[yname], gb[yname], alternative="two-sided")
                gstats[f"{a}_vs_{b}_{yname}"] = {"mannwhitney_p": float(p),
                                                 "delta_mean": float(ga[yname].mean() - gb[yname].mean())}
    gstats["_caveat"] = ("前瞻窗口重叠 (相邻周共用 3 周收益) 会低估 p 值, "
                         "故以效应量 (均值/中位差) 为主要判据, p 值仅供参考。")
    result["b3b_group_contrast"] = gstats

    # ---------- 裁决 ----------
    pool_better = sum(1 for y in b3 if abs(b3[y]["corr_pool"]["spearman_ic"]) >
                      abs(b3[y]["corr_held"]["spearman_ic"]))
    pool_incremental = any(abs(b3[y]["joint_regression"]["corr_pool"]["t"]) >= 1.5 for y in b3)
    held_incremental = any(abs(b3[y]["joint_regression"]["corr_held"]["t"]) >= 1.5 for y in b3)
    result["verdict"] = {
        "pool_stronger_on_n_of_3_y": pool_better,
        "pool_has_incremental_power": pool_incremental,
        "held_has_incremental_power": held_incremental,
    }

    (OUT_DIR / "exp_corr_scope_e1.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # ---------- 报告 ----------
    m, b2 = result["meta"], result["b2_scope_divergence"]
    L = ["# Layer 3.5 相关性口径 E1: 全池 vs 实际持仓对\n",
         f"样本 {m['weeks']} 周 ({m['period']}), 前瞻 {FWD} 周, Newey-West 滞后 {NW_LAG}\n",
         f"> 计划更正: corr_held <= corr_pool 恒成立 (持仓对是全池对的子集), 故原计划第三臂",
         f"> corr_max=两者取大 与 corr_pool 数学等价, 已删除, 改为两臂 + 增量回归检验。\n",
         f"> 口径自证: 本脚本 pool corr 与 engine_core 逐周一致 (max_dev={m['pool_corr_selfcheck_max_dev']:.1e})\n",
         "## B2 口径分歧规模\n",
         f"- 全池口径触发 (corr>{thr}): {b2['pool_fire_weeks']} 周 ({b2['pool_fire_pct']:.1f}%)",
         f"- 持仓对口径触发: {b2['held_fire_weeks']} 周 ({b2['held_fire_pct']:.1f}%)",
         f"- **全池触发但持仓对不触发: {b2['pool_fire_but_held_not']} 周 "
         f"(占全池触发的 {b2['pool_fire_but_held_not_pct_of_fires']:.1f}%)**",
         f"- 持仓对触发但全池不触发: {b2['held_fire_but_pool_not']} 周 (理论应为 0)",
         f"- 触发周中驱动对未被完整持有: {b2['driver_pair_not_held_among_fires']} 周 "
         f"({b2['driver_pair_not_held_pct_of_fires']:.1f}%)",
         f"- corr_pool - corr_held 平均差: {b2['mean_gap_pool_minus_held']:.4f}",
         f"- 显性危机区 (>{split}): 全池 {result['b2_split_zone']['pool_above_split']} 周, "
         f"持仓对 {result['b2_split_zone']['held_above_split']} 周, "
         f"仅全池 {result['b2_split_zone']['pool_above_but_held_not']} 周",
         "\n驱动对出现频次 (触发周):\n"]
    for k, v in b2["driver_pair_freq"].items():
        L.append(f"- {k}: {v} 周")

    L.append("\n## B3 预测力 (Spearman IC + HAC t)\n")
    L.append("| y (未来4周) | 口径 | IC | HAC t | p | 门禁(IC>=.03 & |t|>=1.5) |")
    L.append("|---|---|---|---|---|---|")
    for yname in ("fwd_vol", "fwd_dd", "fwd_ret"):
        for xname in ("corr_pool", "corr_held"):
            e = b3[yname][xname]
            L.append(f"| {yname} | {xname} | {e['spearman_ic']:+.4f} | {e['hac_t']:+.2f} | "
                     f"{e['hac_p']:.3f} | {'PASS' if e['gate_pass'] else 'FAIL'} |")

    L.append("\n### 增量回归 (rank(y) ~ rank(corr_held) + rank(corr_pool), HAC)\n")
    L.append("| y | corr_held beta (t) | corr_pool beta (t) | R2 |")
    L.append("|---|---|---|---|")
    for yname in ("fwd_vol", "fwd_dd", "fwd_ret"):
        j = b3[yname]["joint_regression"]
        L.append(f"| {yname} | {j['corr_held']['beta']:+.3f} ({j['corr_held']['t']:+.2f}) | "
                 f"{j['corr_pool']['beta']:+.3f} ({j['corr_pool']['t']:+.2f}) | {j['r2']:.4f} |")

    L.append("\n## B3b 分组对照: 仅全池告警的周, 未来风险真的更高吗?\n")
    L.append("| 分组 | 周数 | 平均未来4周波动 | 中位 | 平均最大回撤 | 中位 | 平均收益 |")
    L.append("|---|---|---|---|---|---|---|")
    for name, label in (("G1_both_fire", "G1 两口径都告警"), ("G2_pool_only", "G2 仅全池告警"),
                        ("G3_neither", "G3 都不告警")):
        s = gstats[name]
        if s["weeks"] == 0:
            L.append(f"| {label} | 0 | - | - | - | - | - |")
            continue
        L.append(f"| {label} | {s['weeks']} | {s['mean_fwd_vol']*100:.1f}% | {s['median_fwd_vol']*100:.1f}% | "
                 f"{s['mean_fwd_dd']*100:.2f}% | {s['median_fwd_dd']*100:.2f}% | {s['mean_fwd_ret']*100:+.2f}% |")
    L.append("")
    for key in sorted(k for k in gstats if "_vs_" in k):
        s = gstats[key]
        L.append(f"- {key}: Δmean={s['delta_mean']*100:+.2f}pp, Mann-Whitney p={s['mannwhitney_p']:.4f}")
    L.append(f"\n> {gstats['_caveat']}")

    L.append("\n## 裁决输入\n")
    v = result["verdict"]
    L.append(f"- 全池 IC 更强的 y 个数: {v['pool_stronger_on_n_of_3_y']}/3")
    L.append(f"- 全池在控制持仓对后仍有增量解释力: {v['pool_has_incremental_power']}")
    L.append(f"- 持仓对在控制全池后仍有增量解释力: {v['held_has_incremental_power']}")

    (OUT_DIR / "exp_corr_scope_e1.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n" + "\n".join(L[5:]))
    print(f"\n  已写出: {OUT_DIR/'exp_corr_scope_e1.md'} / .json")


if __name__ == "__main__":
    main()
