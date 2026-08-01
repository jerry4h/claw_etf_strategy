#!/usr/bin/env python3
"""E0-RV + E1-RV: Realized Volatility (日频收盘价) 信息增量评估。

核心思路: 用日频 close 计算 realized vol（每周 ~5 个日收益率观测），
避免 H/L 的 QDII 溢价问题，测试是否比周频 CC-tapered vol 有增量。

用法: .venv/bin/python scripts/_exp_rv_vol_study.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from scripts._exp_hl_vol_study import (
    ETF_MAP, NAV_FILE, HL_REAL_START_512890,
    calc_vol_cc_tapered, load_daily_ohlc,
)

OUT = PROJ / "output" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)

ALL_ETFS = ["纳指ETF", "红利低波ETF", "中证500ETF", "黄金ETF", "国债ETF"]
GATE_ETFS = ["纳指ETF", "中证500ETF", "黄金ETF"]  # Full coverage, excl 红利低波(ref only)

CRISIS_EVENTS = {
    "2020-03 全球股灾": pd.Timestamp("2020-02-21"),
    "2022-04 上海封城": pd.Timestamp("2022-04-01"),
    "2024-02 春节后暴跌": pd.Timestamp("2024-02-02"),
}


# ======================================================================
# E0-RV: Realized Volatility computation
# ======================================================================
def calc_vol_realized(nav_index: pd.DatetimeIndex, window: int = 14) -> pd.DataFrame:
    """Compute Realized Volatility from daily close prices.

    For each week in nav_index:
      1. Gather daily log-returns within that ISO week
      2. RV_week = sqrt(sum(r_daily²))  (weekly realized vol, not yet annualized)
      3. Rolling mean of RV_week over `window` weeks, then annualize × sqrt(52)

    Returns DataFrame aligned to nav_index with same columns as NAV file.
    """
    nav = pd.read_csv(NAV_FILE, index_col=0, parse_dates=True)

    # Build ISO week map from nav_index
    nav_week_map = {}  # (isoyear, isoweek) -> nav_date
    for dt in nav_index:
        iy = dt.isocalendar().year
        iw = dt.isocalendar().week
        nav_week_map[(iy, iw)] = dt

    rv_weekly = pd.DataFrame(index=nav_index, columns=nav.columns, dtype=float)

    for code, col_name in ETF_MAP.items():
        daily = load_daily_ohlc(code)
        # Daily log returns using close/pre_close (handles splits/dividends correctly)
        daily = daily.sort_values("trade_date").reset_index(drop=True)
        daily["log_ret"] = np.log(daily["close"] / daily["pre_close"])
        daily["isoyear"] = daily["trade_date"].dt.isocalendar().year.values
        daily["isoweek"] = daily["trade_date"].dt.isocalendar().week.values

        # Aggregate per week: RV_week = sqrt(sum(r²))
        for (year, week), grp in daily.groupby(["isoyear", "isoweek"]):
            nav_date = nav_week_map.get((year, week))
            if nav_date is None:
                continue
            rets = grp["log_ret"].dropna().values
            if len(rets) >= 1:
                rv_week = np.sqrt(np.sum(rets**2))
                rv_weekly.loc[nav_date, col_name] = rv_week

    # For 512890 pre-2019: no daily data → leave NaN (will be handled in analysis)
    # Rolling smooth + annualize
    vol_rv = rv_weekly.astype(float).rolling(window=window, min_periods=window).mean() * np.sqrt(52)
    return vol_rv


def check_daily_vs_nav_consistency():
    """Verify daily close end-of-week matches weekly NAV (within tolerance)."""
    nav = pd.read_csv(NAV_FILE, index_col=0, parse_dates=True)
    nav_week_map = {}
    for dt in nav.index:
        iy = dt.isocalendar().year
        iw = dt.isocalendar().week
        nav_week_map[(iy, iw)] = dt

    print("\n" + "=" * 70)
    print(" E0-RV: 日频 close 周末值 vs weekly NAV 一致性校验")
    print("=" * 70)

    results = {}
    for code, col_name in ETF_MAP.items():
        daily = load_daily_ohlc(code)
        daily = daily.sort_values("trade_date").reset_index(drop=True)
        daily["isoyear"] = daily["trade_date"].dt.isocalendar().year.values
        daily["isoweek"] = daily["trade_date"].dt.isocalendar().week.values

        diffs = []
        for (year, week), grp in daily.groupby(["isoyear", "isoweek"]):
            nav_date = nav_week_map.get((year, week))
            if nav_date is None:
                continue
            nav_val = nav.loc[nav_date, col_name]
            daily_close = grp.sort_values("trade_date").iloc[-1]["close"]
            if pd.notna(nav_val) and daily_close > 0:
                # Relative diff (daily close is raw, NAV is forward-adjusted)
                # They WON'T match exactly because NAV is adjusted and daily close is raw
                diffs.append(abs(daily_close - nav_val) / nav_val)

        if diffs:
            med_diff = np.median(diffs)
            max_diff = np.max(diffs)
            # For raw vs adjusted, we expect large diffs (due to forward adjustment)
            # Instead check correlation of weekly returns
            results[col_name] = {"med_rel_diff": med_diff, "max_rel_diff": max_diff,
                                 "n_weeks": len(diffs)}
            print(f"  {col_name}: {len(diffs)} weeks, "
                  f"med_rel_diff={med_diff:.4f}, max={max_diff:.4f}")
            if med_diff > 0.5:
                print(f"    (预期: raw close vs 前复权 NAV 差异大, 用收益率对比)")
    print("  注: daily close 为原始价格, NAV 为前复权——绝对值差异预期, RV 用日收益率不受影响")
    return results


def check_return_consistency():
    """Check that weekly returns from daily log(close/pre_close) match NAV weekly returns."""
    nav = pd.read_csv(NAV_FILE, index_col=0, parse_dates=True)
    nav_week_map = {}
    for dt in nav.index:
        iy = dt.isocalendar().year
        iw = dt.isocalendar().week
        nav_week_map[(iy, iw)] = dt

    print("\n" + "=" * 70)
    print(" E0-RV: 周收益率一致性校验 (日频 log(C/preC) 周累加 vs NAV 周频)")
    print("=" * 70)

    for code, col_name in ETF_MAP.items():
        daily = load_daily_ohlc(code)
        daily = daily.sort_values("trade_date").reset_index(drop=True)
        # Use close/pre_close to handle splits/dividends
        daily["log_ret"] = np.log(daily["close"] / daily["pre_close"])
        daily["isoyear"] = daily["trade_date"].dt.isocalendar().year.values
        daily["isoweek"] = daily["trade_date"].dt.isocalendar().week.values

        # Weekly return from daily: sum of log returns within the week
        weekly_logret = {}
        for (year, week), grp in daily.groupby(["isoyear", "isoweek"]):
            nav_date = nav_week_map.get((year, week))
            if nav_date is None:
                continue
            weekly_logret[nav_date] = grp["log_ret"].dropna().sum()

        wret_daily = pd.Series(weekly_logret).sort_index()

        # NAV weekly log returns
        nav_logret = np.log(nav[col_name] / nav[col_name].shift(1)).dropna()

        # Align
        common = wret_daily.index.intersection(nav_logret.index)
        if len(common) < 50:
            print(f"  {col_name}: insufficient overlap ({len(common)} weeks)")
            continue
        corr = wret_daily.loc[common].corr(nav_logret.loc[common])
        diff = (wret_daily.loc[common] - nav_logret.loc[common]).abs()
        print(f"  {col_name}: corr(weekly_logret_daily, weekly_logret_nav) = {corr:.6f}, "
              f"mae={diff.mean():.6f}, max={diff.max():.6f} ({len(common)} weeks)")


# ======================================================================
# E1-RV: Information increment analysis
# ======================================================================
def get_valid_mask(col_name: str, index: pd.DatetimeIndex) -> pd.Series:
    """Valid mask: 红利低波 only from 2019+, others full."""
    if col_name == "红利低波ETF":
        return pd.Series(index >= HL_REAL_START_512890, index=index)
    return pd.Series(True, index=index)


def analysis_correlation(vol_rv, vol_cc):
    """corr(RV, CC) per ETF."""
    results = {}
    for col in ALL_ETFS:
        mask = get_valid_mask(col, vol_rv.index)
        valid = mask & vol_rv[col].notna() & vol_cc[col].notna()
        if valid.sum() < 30:
            results[col] = {"corr_rv_cc": np.nan, "n": 0}
            continue
        rv = vol_rv.loc[valid, col]
        cc = vol_cc.loc[valid, col]
        results[col] = {"corr_rv_cc": float(rv.corr(cc)), "n": int(valid.sum())}
    return results


def analysis_noise(vol_rv, vol_cc):
    """Noise comparison: std of week-over-week relative changes."""
    results = {}
    for col in ALL_ETFS:
        mask = get_valid_mask(col, vol_rv.index)
        valid = mask & vol_rv[col].notna() & vol_cc[col].notna()
        rv = vol_rv.loc[valid, col]
        cc = vol_cc.loc[valid, col]
        drv = rv.pct_change().dropna()
        dcc = cc.pct_change().dropna()
        results[col] = {
            "noise_std_RV": float(drv.std()),
            "noise_std_CC": float(dcc.std()),
            "RV_smoother_than_CC": bool(drv.std() < dcc.std()),
            "noise_ratio_RV_over_CC": float(drv.std() / dcc.std()) if dcc.std() > 0 else np.nan,
        }
    return results


def analysis_lead_lag(vol_rv, vol_cc):
    """Cross-correlation at lags -4..+4."""
    results = {}
    lags = list(range(-4, 5))
    for col in ALL_ETFS:
        mask = get_valid_mask(col, vol_rv.index)
        valid = mask & vol_rv[col].notna() & vol_cc[col].notna()
        rv = vol_rv.loc[valid, col].values
        cc = vol_cc.loc[valid, col].values
        n = len(rv)
        xcorr = {}
        for lag in lags:
            if lag >= 0:
                x = rv[:n - lag] if lag > 0 else rv
                y = cc[lag:] if lag > 0 else cc
            else:
                x = rv[-lag:]
                y = cc[:n + lag]
            if len(x) > 10:
                xcorr[lag] = float(np.corrcoef(x, y)[0, 1])
            else:
                xcorr[lag] = np.nan
        best_lag = max(xcorr, key=lambda k: xcorr.get(k, -999))
        results[col] = {
            "xcorr": xcorr,
            "best_lag": best_lag,
            "best_corr": xcorr[best_lag],
            "RV_leads": best_lag > 0,
        }
    return results


def analysis_info_complement(vol_rv, vol_cc):
    """Statistics of vol_rv - vol_cc."""
    results = {}
    for col in ALL_ETFS:
        mask = get_valid_mask(col, vol_rv.index)
        valid = mask & vol_rv[col].notna() & vol_cc[col].notna()
        diff = vol_rv.loc[valid, col] - vol_cc.loc[valid, col]
        if len(diff) < 10:
            results[col] = {}
            continue
        ac1 = float(diff.autocorr(lag=1)) if len(diff) > 2 else np.nan
        results[col] = {
            "mean": float(diff.mean()),
            "std": float(diff.std()),
            "skew": float(diff.skew()),
            "ac1": ac1,
            "pct_positive": float((diff > 0).mean()),
            "n": int(len(diff)),
        }
    return results


def analysis_extreme_events(vol_rv, vol_cc):
    """Breach timing of 75th percentile during crises."""
    results = {}
    for col in GATE_ETFS:
        mask = get_valid_mask(col, vol_rv.index)
        valid = mask & vol_rv[col].notna() & vol_cc[col].notna()
        rv_full = vol_rv.loc[valid, col]
        cc_full = vol_cc.loc[valid, col]
        rv_75 = rv_full.quantile(0.75)
        cc_75 = cc_full.quantile(0.75)

        col_events = {}
        for event_name, event_start in CRISIS_EVENTS.items():
            window_start = event_start - pd.Timedelta(weeks=2)
            window_end = event_start + pd.Timedelta(weeks=8)
            rv_win = rv_full[(rv_full.index >= window_start) & (rv_full.index <= window_end)]
            cc_win = cc_full[(cc_full.index >= window_start) & (cc_full.index <= window_end)]
            rv_breach = rv_win[rv_win > rv_75]
            cc_breach = cc_win[cc_win > cc_75]
            rv_first = rv_breach.index[0] if len(rv_breach) > 0 else None
            cc_first = cc_breach.index[0] if len(cc_breach) > 0 else None
            if rv_first and cc_first:
                lead_weeks = (cc_first - rv_first).days / 7
            else:
                lead_weeks = None
            col_events[event_name] = {
                "RV_first_breach": str(rv_first.date()) if rv_first else None,
                "CC_first_breach": str(cc_first.date()) if cc_first else None,
                "RV_leads_weeks": float(lead_weeks) if lead_weeks is not None else None,
                "RV_leads": lead_weeks > 0 if lead_weeks is not None else None,
            }
        results[col] = col_events
    return results


def analysis_qdii_comparison(corr_results, noise_results):
    """Compare 纳指 (QDII) vs domestic ETFs — RV should NOT show QDII anomaly."""
    qdii = "纳指ETF"
    domestic = ["中证500ETF", "黄金ETF", "国债ETF"]
    qdii_corr = corr_results[qdii]["corr_rv_cc"]
    dom_corrs = [corr_results[c]["corr_rv_cc"] for c in domestic]
    dom_mean_corr = np.mean(dom_corrs)
    qdii_noise = noise_results[qdii]["noise_ratio_RV_over_CC"]
    dom_noises = [noise_results[c]["noise_ratio_RV_over_CC"] for c in domestic]
    dom_mean_noise = np.mean(dom_noises)
    return {
        "qdii_corr_rv_cc": qdii_corr,
        "domestic_mean_corr_rv_cc": dom_mean_corr,
        "corr_diff": qdii_corr - dom_mean_corr,
        "qdii_noise_ratio": qdii_noise,
        "domestic_mean_noise_ratio": dom_mean_noise,
        "noise_diff": qdii_noise - dom_mean_noise,
        "qdii_anomalous": abs(qdii_corr - dom_mean_corr) > 0.10 or
                          abs(qdii_noise - dom_mean_noise) > 0.15,
    }


def gate_decision(corr_results, noise_results, event_results):
    """Apply go/no-go gate rules (same as Parkinson E1)."""
    gate_corrs = [corr_results[c]["corr_rv_cc"] for c in GATE_ETFS]
    corr_in_range = all(0.60 <= c <= 0.95 for c in gate_corrs)
    corr_too_high = any(c > 0.98 for c in gate_corrs)
    corr_too_low = any(c < 0.50 for c in gate_corrs)

    # Lead events
    lead_count = 0
    total_events = 0
    for col in GATE_ETFS:
        for ev_name, ev_data in event_results[col].items():
            total_events += 1
            if ev_data.get("RV_leads") is True and (ev_data.get("RV_leads_weeks") or 0) >= 1:
                lead_count += 1
    lead_ratio = lead_count / total_events if total_events > 0 else 0
    leads_pass = lead_ratio >= 2 / 3

    # Noise
    noise_pass = all(noise_results[c]["RV_smoother_than_CC"] for c in GATE_ETFS)

    go = corr_in_range and leads_pass and noise_pass
    no_go_reasons = []
    if corr_too_high:
        no_go_reasons.append("corr > 0.98 (完全冗余)")
    if corr_too_low:
        no_go_reasons.append("corr < 0.50 (信号质量差)")
    if not noise_pass and any(noise_results[c]["noise_ratio_RV_over_CC"] > 1.2 for c in GATE_ETFS):
        no_go_reasons.append("噪声更大")

    if go:
        verdict = "GO"
    elif no_go_reasons:
        verdict = "NO-GO"
    elif corr_too_high:
        verdict = "NO-GO (冗余)"
    else:
        verdict = "NO-GO"

    return {
        "verdict": verdict,
        "criteria": {
            "corr_in_range": {"pass": corr_in_range, "values": {c: corr_results[c]["corr_rv_cc"] for c in GATE_ETFS}},
            "RV_leads_events": {"pass": leads_pass, "lead_ratio": lead_ratio,
                                "lead_count": lead_count, "total": total_events},
            "noise_smoother": {"pass": noise_pass,
                               "ratios": {c: noise_results[c]["noise_ratio_RV_over_CC"] for c in GATE_ETFS}},
        },
        "no_go_reasons": no_go_reasons,
        "corr_too_high": corr_too_high,
        "corr_too_low": corr_too_low,
    }


# ======================================================================
# Report
# ======================================================================
def render_report(corr_res, noise_res, lead_res, info_res, event_res, qdii_res, gate_res):
    L = ["# E1-RV: Realized Volatility 信息增量评估报告", ""]
    L.append(f"> 日频收盘价 Realized Vol vs CC-tapered vol | 门禁判定: **{gate_res['verdict']}**")
    L.append("")

    # Gate
    L.append("## 门禁判定")
    L.append("")
    L.append(f"**结论: {gate_res['verdict']}**")
    if gate_res["no_go_reasons"]:
        L.append(f"\nNO-GO 原因: {'; '.join(gate_res['no_go_reasons'])}")
    L.append("")
    L.append("| 门禁条件 | 要求 | 实际 | 判定 |")
    L.append("|---|---|---|---|")
    gc = gate_res["criteria"]
    corr_vals = ", ".join(f"{c}={v:.3f}" for c, v in gc["corr_in_range"]["values"].items())
    L.append(f"| corr(RV,CC) ∈ [0.60, 0.95] | 全部门控 ETF | {corr_vals} | "
             f"{'✓' if gc['corr_in_range']['pass'] else '✗'} |")
    L.append(f"| RV 领先事件 ≥ 2/3 | ≥66.7% | {gc['RV_leads_events']['lead_ratio']:.1%} "
             f"({gc['RV_leads_events']['lead_count']}/{gc['RV_leads_events']['total']}) | "
             f"{'✓' if gc['RV_leads_events']['pass'] else '✗'} |")
    noise_vals = ", ".join(f"{c}={v:.3f}" for c, v in gc["noise_smoother"]["ratios"].items())
    L.append(f"| noise(RV) < noise(CC) | RV/CC ratio < 1 | {noise_vals} | "
             f"{'✓' if gc['noise_smoother']['pass'] else '✗'} |")
    L.append("")

    # Analysis 1
    L.append("## 1. 时序相关性")
    L.append("")
    L.append("| ETF | corr(RV, CC) | N (周) |")
    L.append("|---|---|---|")
    for col in ALL_ETFS:
        r = corr_res[col]
        L.append(f"| {col} | {r['corr_rv_cc']:.4f} | {r['n']} |")
    L.append("")

    # Analysis 2
    L.append("## 2. 噪声比")
    L.append("")
    L.append("| ETF | std(ΔRV/RV) | std(ΔCC/CC) | RV/CC ratio | RV更平滑 |")
    L.append("|---|---|---|---|---|")
    for col in ALL_ETFS:
        r = noise_res[col]
        L.append(f"| {col} | {r['noise_std_RV']:.4f} | {r['noise_std_CC']:.4f} | "
                 f"{r['noise_ratio_RV_over_CC']:.3f} | "
                 f"{'✓' if r['RV_smoother_than_CC'] else '✗'} |")
    L.append("")

    # Analysis 3
    L.append("## 3. 领先/滞后关系")
    L.append("")
    L.append("| ETF | 最优 lag | 最优 corr | RV 领先? | xcorr[-2..+2] |")
    L.append("|---|---|---|---|---|")
    for col in ALL_ETFS:
        r = lead_res[col]
        xc_str = " / ".join(f"{r['xcorr'].get(l, np.nan):.3f}" for l in [-2, -1, 0, 1, 2])
        L.append(f"| {col} | {r['best_lag']} | {r['best_corr']:.4f} | "
                 f"{'✓' if r['RV_leads'] else '✗'} | {xc_str} |")
    L.append("")
    L.append("正 lag = RV 领先 CC")
    L.append("")

    # Analysis 4
    L.append("## 4. 信息补集 (RV − CC)")
    L.append("")
    L.append("| ETF | 均值 | 标准差 | 偏度 | AC(1) | 正值占比 |")
    L.append("|---|---|---|---|---|---|")
    for col in ALL_ETFS:
        r = info_res[col]
        if r:
            L.append(f"| {col} | {r['mean']:.4f} | {r['std']:.4f} | "
                     f"{r['skew']:.3f} | {r['ac1']:.3f} | {r['pct_positive']:.1%} |")
    L.append("")

    # Analysis 5
    L.append("## 5. 极端事件响应")
    L.append("")
    for col in GATE_ETFS:
        L.append(f"### {col}")
        L.append("")
        L.append("| 事件 | RV 首次突破75% | CC 首次突破75% | RV 领先(周) |")
        L.append("|---|---|---|---|")
        for ev_name, ev_data in event_res[col].items():
            rv_d = ev_data["RV_first_breach"] or "未触发"
            cc_d = ev_data["CC_first_breach"] or "未触发"
            lead = f"{ev_data['RV_leads_weeks']:.1f}" if ev_data["RV_leads_weeks"] is not None else "N/A"
            L.append(f"| {ev_name} | {rv_d} | {cc_d} | {lead} |")
        L.append("")

    # Analysis 6
    L.append("## 6. QDII 对比 (RV 应无溢价异常)")
    L.append("")
    L.append(f"| 指标 | 纳指ETF (QDII) | 境内均值 | 差值 |")
    L.append("|---|---|---|---|")
    L.append(f"| corr(RV, CC) | {qdii_res['qdii_corr_rv_cc']:.4f} | "
             f"{qdii_res['domestic_mean_corr_rv_cc']:.4f} | {qdii_res['corr_diff']:+.4f} |")
    L.append(f"| noise ratio (RV/CC) | {qdii_res['qdii_noise_ratio']:.4f} | "
             f"{qdii_res['domestic_mean_noise_ratio']:.4f} | {qdii_res['noise_diff']:+.4f} |")
    L.append("")
    L.append(f"**异常判定**: {'⚠ QDII 仍有异常' if qdii_res['qdii_anomalous'] else '✓ 无显著异常 (RV 不受溢价影响)'}")
    L.append("")

    # Key insight
    L.append("## 关键洞察")
    L.append("")
    L.append("RV 使用日收盘价（不含盘中溢价极值），理论上对 QDII ETF 应表现正常。")
    L.append("但 RV 与 CC-vol 本质上估计的是同一目标（收盘价波动率），")
    L.append("区别仅在于采样频率（日 vs 周）和平滑方式。")
    L.append("")
    return "\n".join(L)


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 70)
    print(" E0-RV + E1-RV: Realized Volatility 信息增量评估")
    print("=" * 70)

    nav = pd.read_csv(NAV_FILE, index_col=0, parse_dates=True)

    # --- E0: Compute RV ---
    print("\n[E0] 计算 Realized Volatility...")
    check_return_consistency()
    vol_rv = calc_vol_realized(nav.index, window=14)
    vol_cc = calc_vol_cc_tapered(nav, window=14, taper=5)
    print(f"  vol_rv shape: {vol_rv.shape}, non-null: {vol_rv.notna().sum().to_dict()}")
    print(f"  vol_cc shape: {vol_cc.shape}")

    # --- E1: Analyses ---
    print("\n[1] 时序相关性...")
    corr_res = analysis_correlation(vol_rv, vol_cc)
    for col in ALL_ETFS:
        print(f"  {col}: corr(RV,CC)={corr_res[col]['corr_rv_cc']:.4f}")

    print("\n[2] 噪声比...")
    noise_res = analysis_noise(vol_rv, vol_cc)
    for col in ALL_ETFS:
        r = noise_res[col]
        print(f"  {col}: RV/CC={r['noise_ratio_RV_over_CC']:.3f} "
              f"{'✓ RV更平滑' if r['RV_smoother_than_CC'] else '✗ CC更平滑'}")

    print("\n[3] 领先/滞后...")
    lead_res = analysis_lead_lag(vol_rv, vol_cc)
    for col in ALL_ETFS:
        r = lead_res[col]
        print(f"  {col}: best_lag={r['best_lag']}, corr={r['best_corr']:.4f}")

    print("\n[4] 信息补集 (RV-CC)...")
    info_res = analysis_info_complement(vol_rv, vol_cc)
    for col in ALL_ETFS:
        r = info_res[col]
        if r:
            print(f"  {col}: mean={r['mean']:.4f}, std={r['std']:.4f}, "
                  f"skew={r['skew']:.3f}, positive={r['pct_positive']:.1%}")

    print("\n[5] 极端事件响应...")
    event_res = analysis_extreme_events(vol_rv, vol_cc)
    for col in GATE_ETFS:
        for ev_name, ev_data in event_res[col].items():
            lead = ev_data.get("RV_leads_weeks")
            lead_str = f"{lead:+.1f}w" if lead is not None else "N/A"
            print(f"  {col} | {ev_name}: RV leads {lead_str}")

    print("\n[6] QDII 对比...")
    qdii_res = analysis_qdii_comparison(corr_res, noise_res)
    print(f"  corr diff={qdii_res['corr_diff']:+.4f}, "
          f"noise diff={qdii_res['noise_diff']:+.4f}, "
          f"anomalous={qdii_res['qdii_anomalous']}")

    # --- Gate ---
    print("\n" + "=" * 70)
    gate_res = gate_decision(corr_res, noise_res, event_res)
    print(f" 门禁判定: **{gate_res['verdict']}**")
    print("=" * 70)
    gc = gate_res["criteria"]
    print(f"  [1] corr ∈ [0.60, 0.95]: {'PASS' if gc['corr_in_range']['pass'] else 'FAIL'}")
    print(f"  [2] RV leads ≥ 2/3 events: {'PASS' if gc['RV_leads_events']['pass'] else 'FAIL'} "
          f"({gc['RV_leads_events']['lead_ratio']:.1%})")
    print(f"  [3] noise(RV) < noise(CC): {'PASS' if gc['noise_smoother']['pass'] else 'FAIL'}")
    if gate_res["no_go_reasons"]:
        print(f"  NO-GO: {'; '.join(gate_res['no_go_reasons'])}")

    # --- Save ---
    all_results = {
        "correlation": corr_res,
        "noise": noise_res,
        "lead_lag": lead_res,
        "info_complement": info_res,
        "extreme_events": event_res,
        "qdii_comparison": qdii_res,
        "gate_decision": gate_res,
    }

    json_path = OUT / "exp_rv_vol_e1.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  JSON saved: {json_path}")

    md = render_report(corr_res, noise_res, lead_res, info_res, event_res, qdii_res, gate_res)
    md_path = OUT / "exp_rv_vol_e1.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  Report saved: {md_path}")


if __name__ == "__main__":
    main()
