"""
交付物 A: v4.6 (directed_boost) vs v4.5-pvd Layer 3.5 历史分歧统计。

研究问题
    本周 (信号日 2026-08-21) 两版防御比例出现 6.1pp 分歧 (76.2% vs 70.1%)。
    这是罕见事件还是常态? 分歧幅度的历史分布如何?

两种视角
    A1 机制隔离: 仅 Layer 3 基础映射 + Layer 3.5 boost, 排除 M3/PE/止损,
                 纯粹刻画 directed_boost 触发器本身带来的差异。
    A2 端到端  : 两版完整回测的 def_ratio 逐周对比。因两版 NAV 路径会分叉
                 (止损状态可能不同), 该视角包含路径效应, 是实盘真实差异。

样本范围 (关键)
    回测循环止于 n_weeks-2 (末周无已实现收益), 故末次决策 (即"下周仓位"信号)
    不在回测样本内。本脚本把 i 扩到 range(start_idx, n_weeks) 以纳入该周 ——
    def_ratio 只依赖 [i-window, i) 的历史收益, 无需未来数据, 不引入前视。

只读脚本: 不修改 src/, 不改配置, 不提交实验数据。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.backtest import run_backtest                                    # noqa: E402
from src.data_loader import classify_etfs, load_nav_data, resample_weekly  # noqa: E402
from src.factors import compute_all_factors                               # noqa: E402
from src.engine_core import compute_crisis_boost, compute_crisis_boost_directed  # noqa: E402
from src.strategy import calculate_defense_ratio, load_config            # noqa: E402

CFG46 = "config/strategy_v4_6.yaml"
CFG45 = "config/strategy_v4_5_pvd.yaml"
OUT_DIR = PROJ / "output" / "experiments"
TOL = 0.001  # 0.1pp: 分歧判定门限


def build_weekly(cfg):
    """完全复刻 run_backtest 的周频 NAV 构建 (backtest.py:109-123)。"""
    nav_path = Path(cfg.nav_path)
    if not nav_path.is_absolute():
        nav_path = PROJ / nav_path
    weekly = resample_weekly(load_nav_data(nav_path), anchor=cfg.anchor)
    if cfg.start_date:
        weekly = weekly[weekly.index >= pd.to_datetime(cfg.start_date)]
    if cfg.end_date:
        weekly = weekly[weekly.index <= pd.to_datetime(cfg.end_date)]
    return weekly


def build_volatility(weekly, cfg):
    """复刻 run_backtest 的因子计算 (backtest.py:143-159), 取 volatility。

    需要它是因为回测 nav_series 只记录到 n_weeks-2, 而末次决策周 (i=n_weeks-1)
    的 nasdaq_vol 无对应记录行。脚本会断言两者在重叠区间逐位相等。
    """
    config_dict = {'factors': {
        'mom_window': cfg.mom_window, 'vol_window': cfg.vol_window, 'vol_ddof': cfg.vol_ddof,
        'pe_window_years': cfg.pe_window_years,
        'ewma_factors_enabled': cfg.ewma_factors_enabled,
        'ewma_mom_halflife': cfg.ewma_mom_halflife, 'ewma_vol_halflife': cfg.ewma_vol_halflife,
        'vol_taper_enabled': cfg.vol_taper_enabled, 'vol_taper_window': cfg.vol_taper_window,
        'vol_taper_len': cfg.vol_taper_len,
        'pvd_enabled': cfg.pvd_enabled, 'pvd_window': cfg.pvd_window,
        'pvd_min_periods': cfg.pvd_min_periods,
    }}
    return compute_all_factors(weekly, None, config_dict)['volatility']


def derive_start_idx(w_index, first_record_date):
    """由对齐关系反推 start_idx: 回测首行 date == w_index[start_idx + 1]。

    比复刻 backtest.py:233-238 的分支逻辑更稳健 (配置变化不会失配)。
    """
    pos = list(w_index).index(pd.Timestamp(first_record_date))
    return pos - 1


def apply_directed(base, boost, corr, split):
    """复刻 backtest.py:420-427 的 v4.6 分级应用。"""
    if boost <= 0:
        return base
    if corr > split:
        return min(base + boost, 1.0)
    return min(base + boost * (1.0 - base), 1.0)


def pct_rank(arr, value):
    """value 在 arr 中的百分位 (0~100)。"""
    arr = np.asarray(arr)
    if len(arr) == 0:
        return float("nan")
    return float((arr <= value).sum() / len(arr) * 100)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg46, cfg45 = load_config(CFG46), load_config(CFG45)

    # 前置校验: A1 的 base 计算假设 def_alloc 未被 regime 覆盖
    if getattr(cfg46, "regime_enabled", False):
        print("  [warn] regime_enabled=True — A1 的 base 可能被 regime 覆盖, A1 结果仅供参考")
    print(f"  v4.6 directed: thr={cfg46.directed_boost_threshold} slope={cfg46.directed_boost_slope} "
          f"split={cfg46.directed_boost_corr_split} max={cfg46.crisis_corr_max_boost}")
    print(f"  v4.5 classic : thr={cfg45.crisis_corr_threshold} slope={cfg45.crisis_corr_slope} "
          f"max={cfg45.crisis_corr_max_boost}")
    print(f"  饱和点: v4.5={cfg45.crisis_corr_threshold + cfg45.crisis_corr_max_boost / cfg45.crisis_corr_slope:.3f}"
          f"  v4.6={cfg46.directed_boost_threshold + cfg46.crisis_corr_max_boost / cfg46.directed_boost_slope:.3f}")

    weekly = build_weekly(cfg46)
    w_prices, w_index = weekly.values, weekly.index
    n_weeks = len(w_index)
    w_rets = np.diff(w_prices, axis=0) / w_prices[:-1]   # 同 backtest.py:185
    etf_names = list(weekly.columns)
    off_idx, _, NASDAQ_IDX = classify_etfs(etf_names)
    vol_df = build_volatility(weekly, cfg46)
    vol_nasdaq = vol_df[etf_names[NASDAQ_IDX]].values

    print("\n  跑两版回测 (端到端视角)...")
    nav46 = run_backtest(cfg46).nav_series
    nav45 = run_backtest(cfg45).nav_series
    assert len(nav46) == len(nav45), "两版回测行数不一致, 无法逐周对比"
    start_idx = derive_start_idx(w_index, nav46.index[0])
    n_bt = len(nav46)
    print(f"  start_idx={start_idx}  回测 {n_bt} 周  {nav46.index[0].date()} ~ {nav46.index[-1].date()}")

    # 口径自证: 重算的 nasdaq_vol 必须与回测记录逐位相等 (重叠区间)
    recomputed = np.array([vol_nasdaq[start_idx + k] for k in range(n_bt)])
    max_dev = float(np.nanmax(np.abs(recomputed - nav46["nasdaq_vol"].values)))
    assert max_dev < 1e-12, f"nasdaq_vol 口径不一致, max_dev={max_dev}"
    print(f"  [self-check] 重算 nasdaq_vol 与回测记录一致 (max_dev={max_dev:.2e})")

    # ---------- 逐周计算 (含末次决策周 i=n_weeks-1, 即“下周仓位”信号) ----------
    rows = []
    for i in range(start_idx, n_weeks):
        k = i - start_idx                        # 回测行号 (k < n_bt 时有效)
        in_bt = k < n_bt
        base = calculate_defense_ratio(vol_nasdaq[i], cfg46)
        b45 = compute_crisis_boost(w_rets, i, off_idx, cfg45)
        b46, corr = compute_crisis_boost_directed(w_rets, i, off_idx, cfg46)
        d45 = min(base + b45, 1.0) if b45 > 0 else base
        d46 = apply_directed(base, b46, corr, cfg46.directed_boost_corr_split)
        rows.append({
            "signal_date": str(w_index[i].date()),
            "record_date": str(nav46.index[k].date()) if in_bt else "(未实现)",
            "in_backtest": in_bt,
            "corr": corr, "base": base, "b45": b45, "b46": b46,
            "def45_iso": d45, "def46_iso": d46, "delta_iso": d46 - d45,
            "def45_e2e": nav45["def_ratio"].iloc[k] if in_bt else np.nan,
            "def46_e2e": nav46["def_ratio"].iloc[k] if in_bt else np.nan,
            "delta_e2e": (nav46["def_ratio"].iloc[k] - nav45["def_ratio"].iloc[k]) if in_bt else np.nan,
        })
    df = pd.DataFrame(rows)
    print(f"  样本 {len(df)} 周 (含末次决策周 {df['signal_date'].iloc[-1]}, 该周不在回测内)")

    # A1 vs A2 一致性诊断: 若完全相等, 说明 M3/PE/止损 从未改动 def_ratio
    bt = df[df["in_backtest"]]
    dev46 = float(np.max(np.abs(bt["def46_iso"].values - bt["def46_e2e"].values)))
    dev45 = float(np.max(np.abs(bt["def45_iso"].values - bt["def45_e2e"].values)))
    print(f"  [self-check] 机制隔离 vs 端到端 def 最大偏差: v4.6={dev46:.2e}  v4.5={dev45:.2e}")
    iso_equals_e2e = max(dev46, dev45) < 1e-12

    result = {"config": {
        "v46": {"thr": cfg46.directed_boost_threshold, "slope": cfg46.directed_boost_slope,
                "split": cfg46.directed_boost_corr_split, "max_boost": cfg46.crisis_corr_max_boost},
        "v45": {"thr": cfg45.crisis_corr_threshold, "slope": cfg45.crisis_corr_slope,
                "max_boost": cfg45.crisis_corr_max_boost},
        "weeks": len(df), "weeks_in_backtest": int(len(bt)),
        "period": f"{df['signal_date'].iloc[0]} ~ {df['signal_date'].iloc[-1]}",
        "iso_equals_e2e": iso_equals_e2e,
        "iso_vs_e2e_max_dev": max(dev46, dev45)}}

    for tag in ("iso", "e2e"):
        d = df[f"delta_{tag}"].values
        d = d[~np.isnan(d)]
        div = np.abs(d) > TOL
        nz = d[div]
        stats = {
            "diverge_weeks": int(div.sum()), "total_weeks": len(d),
            "diverge_pct": float(div.sum() / len(d) * 100),
            "v46_more_defensive": int((d > TOL).sum()), "v46_less_defensive": int((d < -TOL).sum()),
        }
        if len(nz) > 0:
            stats.update({
                "mean_pp": float(np.mean(nz) * 100), "median_pp": float(np.median(np.abs(nz)) * 100),
                "p75_pp": float(np.percentile(np.abs(nz), 75) * 100),
                "p90_pp": float(np.percentile(np.abs(nz), 90) * 100),
                "max_pp": float(np.max(np.abs(nz)) * 100),
            })
        cur_series = df[f"delta_{tag}"].dropna()
        cur = float(cur_series.iloc[-1])
        stats["current_week_pp"] = cur * 100
        stats["current_week_signal_date"] = df.loc[cur_series.index[-1], "signal_date"]
        stats["current_pct_rank_among_diverge"] = pct_rank(np.abs(nz), abs(cur)) if len(nz) else float("nan")
        result[tag] = stats

    # ---------- corr 分桶 ----------
    edges = [0, cfg46.directed_boost_threshold, cfg46.directed_boost_corr_split, 0.65, 0.68, 1.01]
    labels = ["<0.45 双未触发", "0.45-0.60 灰区(仅v4.6)", "0.60-0.65", "0.65-0.68", ">=0.68 双饱和"]
    df["bucket"] = pd.cut(df["corr"], bins=edges, labels=labels, right=False)
    buckets = []
    for lab in labels:
        sub = df[df["bucket"] == lab]
        if len(sub) == 0:
            buckets.append({"bucket": lab, "weeks": 0})
            continue
        buckets.append({
            "bucket": lab, "weeks": len(sub), "pct_of_all": float(len(sub) / len(df) * 100),
            "mean_delta_iso_pp": float(sub["delta_iso"].mean() * 100),
            "max_delta_iso_pp": float(sub["delta_iso"].abs().max() * 100),
            "mean_b45_pp": float(sub["b45"].mean() * 100), "mean_b46_pp": float(sub["b46"].mean() * 100),
        })
    result["corr_buckets"] = buckets
    last = df.iloc[-1]
    result["current_week"] = {
        "signal_date": last["signal_date"], "corr": float(last["corr"]),
        "bucket": str(last["bucket"]), "in_backtest": bool(last["in_backtest"]),
        "b45_pp": float(last["b45"] * 100), "b46_pp": float(last["b46"] * 100),
        "base": float(last["base"]),
        "def45_iso": float(last["def45_iso"]), "def46_iso": float(last["def46_iso"]),
        "delta_iso_pp": float(last["delta_iso"] * 100),
    }
    # 最近 8 周明细 (供人工核对)
    result["recent_8w"] = [
        {"signal_date": r["signal_date"], "corr": round(float(r["corr"]), 4),
         "b45_pp": round(float(r["b45"] * 100), 1), "b46_pp": round(float(r["b46"] * 100), 1),
         "def45_iso_pct": round(float(r["def45_iso"] * 100), 1),
         "def46_iso_pct": round(float(r["def46_iso"] * 100), 1),
         "delta_pp": round(float(r["delta_iso"] * 100), 2)}
        for _, r in df.tail(8).iterrows()
    ]

    # ---------- 输出 ----------
    (OUT_DIR / "exp_v46_v45_divergence.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# v4.6 vs v4.5-pvd Layer 3.5 分歧统计\n",
         f"样本: {result['config']['weeks']} 周 ({result['config']['period']}), 其中 "
         f"{result['config']['weeks_in_backtest']} 周在回测内, 末周为尚未实现的“下周仓位”决策\n",
         "## 触发器参数\n",
         f"- v4.5 classic : thr={cfg45.crisis_corr_threshold} slope={cfg45.crisis_corr_slope} "
         f"→ 饱和点 {cfg45.crisis_corr_threshold + cfg45.crisis_corr_max_boost / cfg45.crisis_corr_slope:.3f}",
         f"- v4.6 directed: thr={cfg46.directed_boost_threshold} slope={cfg46.directed_boost_slope} "
         f"split={cfg46.directed_boost_corr_split} → 饱和点 "
         f"{cfg46.directed_boost_threshold + cfg46.crisis_corr_max_boost / cfg46.directed_boost_slope:.3f}\n",
         "## 分歧统计\n",
         "| 视角 | 分歧周数 | 占比 | 均值 | 中位(abs) | p75 | p90 | max | v4.6更防御 | v4.6更进攻 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for tag, name in (("iso", "A1 机制隔离"), ("e2e", "A2 端到端")):
        s = result[tag]
        L.append(f"| {name} | {s['diverge_weeks']} | {s['diverge_pct']:.1f}% | "
                 f"{s.get('mean_pp', 0):+.2f}pp | {s.get('median_pp', 0):.2f}pp | "
                 f"{s.get('p75_pp', 0):.2f}pp | {s.get('p90_pp', 0):.2f}pp | {s.get('max_pp', 0):.2f}pp | "
                 f"{s['v46_more_defensive']} | {s['v46_less_defensive']} |")
    if iso_equals_e2e:
        L.append("\n> A1 与 A2 逐位相等 (最大偏差 < 1e-12): 本配置下 M3/PE/止损 从未改动 "
                 "def_ratio, 且 def_ratio 只依赖 ETF 价格序列而非策略 NAV 路径, 故两视角重合。")
    else:
        L.append(f"\n> A1 与 A2 最大偏差 {max(dev46, dev45)*100:.2f}pp (M3/PE/止损 在部分周改动了 def_ratio)。")

    L.append("\n## 本周定位 (末次决策 = 下周仓位信号)\n")
    cw = result["current_week"]
    L.append(f"- 信号日 {cw['signal_date']}, corr={cw['corr']:.4f}, 落桶: {cw['bucket']}"
             f"{'' if cw['in_backtest'] else '  (不在回测样本内)'}")
    L.append(f"- Layer 3 基础映射 = {cw['base']*100:.1f}%")
    L.append(f"- boost: v4.5={cw['b45_pp']:.1f}pp  v4.6={cw['b46_pp']:.1f}pp")
    L.append(f"- 防御: v4.5={cw['def45_iso']*100:.1f}%  v4.6={cw['def46_iso']*100:.1f}%  "
             f"Δ={cw['delta_iso_pp']:+.2f}pp  "
             f"(在 {result['iso']['diverge_weeks']} 个分歧周中处于 "
             f"{pct_rank(np.abs(df['delta_iso'][np.abs(df['delta_iso']) > TOL]), abs(cw['delta_iso_pp'] / 100)):.0f}% 分位)")

    L.append("\n## 最近 8 周明细\n")
    L.append("| 信号日 | corr | b45 | b46 | v4.5 def | v4.6 def | Δ |")
    L.append("|---|---|---|---|---|---|---|")
    for r in result["recent_8w"]:
        L.append(f"| {r['signal_date']} | {r['corr']:.3f} | {r['b45_pp']:.1f}pp | {r['b46_pp']:.1f}pp | "
                 f"{r['def45_iso_pct']:.1f}% | {r['def46_iso_pct']:.1f}% | {r['delta_pp']:+.2f}pp |")
    L.append("\n## corr 分桶 (机制隔离视角)\n")
    L.append("| corr 区间 | 周数 | 占比 | 平均 Δdef | 最大 Δdef | 平均 b45 | 平均 b46 |")
    L.append("|---|---|---|---|---|---|---|")
    for b in buckets:
        if b["weeks"] == 0:
            L.append(f"| {b['bucket']} | 0 | - | - | - | - | - |")
            continue
        L.append(f"| {b['bucket']} | {b['weeks']} | {b['pct_of_all']:.1f}% | "
                 f"{b['mean_delta_iso_pp']:+.2f}pp | {b['max_delta_iso_pp']:.2f}pp | "
                 f"{b['mean_b45_pp']:.1f}pp | {b['mean_b46_pp']:.1f}pp |")
    (OUT_DIR / "exp_v46_v45_divergence.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("\n" + "\n".join(L[4:]))
    print(f"\n  已写出: {OUT_DIR/'exp_v46_v45_divergence.md'} / .json")


if __name__ == "__main__":
    main()
