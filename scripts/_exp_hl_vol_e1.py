#!/usr/bin/env python3
"""E1: 离线信息增量评估 — Parkinson vs CC-tapered vol 深度对比。

基于 E0 产出的 vol_parkinson / vol_cc / vol_gk，评估 Parkinson 估计器
是否为策略提供增量信息，并判定 go/no-go 门禁。

用法: .venv/bin/python scripts/_exp_hl_vol_e1.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

# Import E0 functions
from scripts._exp_hl_vol_study import (
    ETF_MAP, NAV_FILE, HL_REAL_START_512890,
    aggregate_weekly_ohlc, calc_vol_parkinson, calc_vol_cc_tapered,
    calc_vol_garman_klass,
)

OUT = PROJ / "output" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)

# ETFs for gate decision (full coverage, exclude 红利低波 which is reference only)
GATE_ETFS = ["纳指ETF", "中证500ETF", "黄金ETF"]
ALL_ETFS = ["纳指ETF", "红利低波ETF", "中证500ETF", "黄金ETF", "国债ETF"]

# Crisis events for lead/lag analysis
CRISIS_EVENTS = {
    "2020-03 全球股灾": pd.Timestamp("2020-02-21"),  # start of crash
    "2022-04 上海封城": pd.Timestamp("2022-04-01"),
    "2024-02 春节后暴跌": pd.Timestamp("2024-02-02"),
}


def get_valid_mask(col_name: str, index: pd.DatetimeIndex) -> pd.Series:
    """Get mask for valid (real OHLC) data period for an ETF."""
    if col_name == "红利低波ETF":
        return pd.Series(index >= HL_REAL_START_512890, index=index)
    return pd.Series(True, index=index)


# ======================================================================
# Analysis 1: Time-series correlation
# ======================================================================
def analysis_correlation(vol_p, vol_cc, vol_gk):
    """Compute corr(Parkinson, CC) and corr(GK, CC) per ETF."""
    results = {}
    for col in ALL_ETFS:
        mask = get_valid_mask(col, vol_p.index)
        valid = mask & vol_p[col].notna() & vol_cc[col].notna()
        if valid.sum() < 30:
            results[col] = {"corr_p_cc": np.nan, "corr_gk_cc": np.nan, "n": 0}
            continue
        p = vol_p.loc[valid, col]
        cc = vol_cc.loc[valid, col]
        gk_valid = mask & vol_gk[col].notna() & vol_cc[col].notna()
        gk = vol_gk.loc[gk_valid, col]
        cc_gk = vol_cc.loc[gk_valid, col]
        results[col] = {
            "corr_p_cc": float(p.corr(cc)),
            "corr_gk_cc": float(gk.corr(cc_gk)),
            "n": int(valid.sum()),
        }
    return results


# ======================================================================
# Analysis 2: Noise ratio (week-over-week relative change std)
# ======================================================================
def analysis_noise(vol_p, vol_cc, vol_gk):
    """Compare std of relative week-over-week changes."""
    results = {}
    for col in ALL_ETFS:
        mask = get_valid_mask(col, vol_p.index)
        valid = mask & vol_p[col].notna() & vol_cc[col].notna()
        p = vol_p.loc[valid, col]
        cc = vol_cc.loc[valid, col]
        gk_valid = mask & vol_gk[col].notna() & vol_cc[col].notna()
        gk = vol_gk.loc[gk_valid, col]

        # Relative change: Δvol / vol_t (avoid division by zero)
        dp = p.pct_change().dropna()
        dcc = cc.pct_change().dropna()
        dgk = gk.pct_change().dropna()

        results[col] = {
            "noise_std_P": float(dp.std()),
            "noise_std_CC": float(dcc.std()),
            "noise_std_GK": float(dgk.std()),
            "P_smoother_than_CC": bool(dp.std() < dcc.std()),
            "noise_ratio_P_over_CC": float(dp.std() / dcc.std()) if dcc.std() > 0 else np.nan,
        }
    return results


# ======================================================================
# Analysis 3: Lead/lag cross-correlation
# ======================================================================
def analysis_lead_lag(vol_p, vol_cc):
    """Cross-correlation at lags -4..+4 (positive lag = P leads CC)."""
    results = {}
    lags = list(range(-4, 5))
    for col in ALL_ETFS:
        mask = get_valid_mask(col, vol_p.index)
        valid = mask & vol_p[col].notna() & vol_cc[col].notna()
        p = vol_p.loc[valid, col].values
        cc = vol_cc.loc[valid, col].values
        n = len(p)

        xcorr = {}
        for lag in lags:
            if lag >= 0:
                x = p[:n - lag] if lag > 0 else p
                y = cc[lag:] if lag > 0 else cc
            else:
                x = p[-lag:]
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
            "P_leads": best_lag > 0,
        }
    return results


# ======================================================================
# Analysis 4: Information complement (P - CC)
# ======================================================================
def analysis_info_complement(vol_p, vol_cc):
    """Statistics of vol_parkinson - vol_cc."""
    results = {}
    for col in ALL_ETFS:
        mask = get_valid_mask(col, vol_p.index)
        valid = mask & vol_p[col].notna() & vol_cc[col].notna()
        diff = vol_p.loc[valid, col] - vol_cc.loc[valid, col]

        if len(diff) < 10:
            results[col] = {}
            continue

        # Autocorrelation at lag 1
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


# ======================================================================
# Analysis 5: Extreme event response timing
# ======================================================================
def analysis_extreme_events(vol_p, vol_cc):
    """For each crisis, find when P and CC first breach their 75th percentile."""
    results = {}

    for col in GATE_ETFS:
        mask = get_valid_mask(col, vol_p.index)
        valid = mask & vol_p[col].notna() & vol_cc[col].notna()
        p_full = vol_p.loc[valid, col]
        cc_full = vol_cc.loc[valid, col]

        # 75th percentile thresholds (computed on full history)
        p_75 = p_full.quantile(0.75)
        cc_75 = cc_full.quantile(0.75)

        col_events = {}
        for event_name, event_start in CRISIS_EVENTS.items():
            # Look at 8-week window around event start
            window_start = event_start - pd.Timedelta(weeks=2)
            window_end = event_start + pd.Timedelta(weeks=8)

            p_window = p_full[(p_full.index >= window_start) & (p_full.index <= window_end)]
            cc_window = cc_full[(cc_full.index >= window_start) & (cc_full.index <= window_end)]

            # First breach of 75th percentile
            p_breach = p_window[p_window > p_75]
            cc_breach = cc_window[cc_window > cc_75]

            p_first = p_breach.index[0] if len(p_breach) > 0 else None
            cc_first = cc_breach.index[0] if len(cc_breach) > 0 else None

            if p_first and cc_first:
                lead_weeks = (cc_first - p_first).days / 7
            else:
                lead_weeks = None

            col_events[event_name] = {
                "P_first_breach": str(p_first.date()) if p_first else None,
                "CC_first_breach": str(cc_first.date()) if cc_first else None,
                "P_leads_weeks": float(lead_weeks) if lead_weeks is not None else None,
                "P_leads": lead_weeks > 0 if lead_weeks is not None else None,
            }

        results[col] = col_events
    return results


# ======================================================================
# Analysis 6: QDII premium distortion check (513100 vs others)
# ======================================================================
def analysis_qdii_premium(corr_results, noise_results):
    """Check if 513100 (QDII) shows anomalous corr or noise vs domestic ETFs."""
    qdii = "纳指ETF"
    domestic = ["中证500ETF", "黄金ETF", "国债ETF"]

    qdii_corr = corr_results[qdii]["corr_p_cc"]
    dom_corrs = [corr_results[c]["corr_p_cc"] for c in domestic]
    dom_mean_corr = np.mean(dom_corrs)

    qdii_noise = noise_results[qdii]["noise_ratio_P_over_CC"]
    dom_noises = [noise_results[c]["noise_ratio_P_over_CC"] for c in domestic]
    dom_mean_noise = np.mean(dom_noises)

    return {
        "qdii_corr_p_cc": qdii_corr,
        "domestic_mean_corr_p_cc": dom_mean_corr,
        "corr_diff": qdii_corr - dom_mean_corr,
        "qdii_noise_ratio": qdii_noise,
        "domestic_mean_noise_ratio": dom_mean_noise,
        "noise_diff": qdii_noise - dom_mean_noise,
        "qdii_anomalous": abs(qdii_corr - dom_mean_corr) > 0.10 or
                          abs(qdii_noise - dom_mean_noise) > 0.15,
    }


# ======================================================================
# Gate decision
# ======================================================================
def gate_decision(corr_results, noise_results, event_results):
    """Apply go/no-go gate rules."""
    # Rule 1: corr(P, CC) ∈ [0.60, 0.95] for gate ETFs
    gate_corrs = [corr_results[c]["corr_p_cc"] for c in GATE_ETFS]
    corr_in_range = all(0.60 <= c <= 0.95 for c in gate_corrs)
    corr_too_high = any(c > 0.98 for c in gate_corrs)
    corr_too_low = any(c < 0.50 for c in gate_corrs)

    # Rule 2: P leads CC in ≥2/3 crisis events (across gate ETFs)
    lead_count = 0
    total_events = 0
    for col in GATE_ETFS:
        for ev_name, ev_data in event_results[col].items():
            total_events += 1
            if ev_data.get("P_leads") is True and (ev_data.get("P_leads_weeks") or 0) >= 1:
                lead_count += 1
    lead_ratio = lead_count / total_events if total_events > 0 else 0
    leads_pass = lead_ratio >= 2 / 3

    # Rule 3: noise std(ΔP/P) < std(ΔCC/CC) for gate ETFs
    noise_pass = all(noise_results[c]["P_smoother_than_CC"] for c in GATE_ETFS)

    # Overall
    go = corr_in_range and leads_pass and noise_pass
    no_go = corr_too_high or corr_too_low or (not noise_pass and
            any(noise_results[c]["noise_ratio_P_over_CC"] > 1.2 for c in GATE_ETFS))

    # PIVOT: check GK as alternative
    gk_corrs = [corr_results[c]["corr_gk_cc"] for c in GATE_ETFS]
    gk_corr_ok = all(0.60 <= c <= 0.95 for c in gk_corrs if not np.isnan(c))

    if not go and not no_go:
        verdict = "PIVOT (建议切换 GK)" if gk_corr_ok else "NO-GO"
    elif go:
        verdict = "GO"
    else:
        verdict = "NO-GO"

    return {
        "verdict": verdict,
        "criteria": {
            "corr_in_range": {"pass": corr_in_range, "values": {c: corr_results[c]["corr_p_cc"] for c in GATE_ETFS}},
            "P_leads_events": {"pass": leads_pass, "lead_ratio": lead_ratio,
                              "lead_count": lead_count, "total": total_events},
            "noise_smoother": {"pass": noise_pass,
                              "ratios": {c: noise_results[c]["noise_ratio_P_over_CC"] for c in GATE_ETFS}},
        },
        "corr_too_high": corr_too_high,
        "corr_too_low": corr_too_low,
        "gk_corrs": {c: corr_results[c]["corr_gk_cc"] for c in GATE_ETFS},
    }


# ======================================================================
# Report generation
# ======================================================================
def render_report(corr_res, noise_res, lead_res, info_res, event_res, qdii_res, gate_res):
    """Generate markdown report."""
    lines = ["# E1: 离线信息增量评估报告", ""]
    lines.append(f"> Parkinson vs CC-tapered vol 深度对比 | 门禁判定: **{gate_res['verdict']}**")
    lines.append("")

    # Gate summary
    lines.append("## 门禁判定")
    lines.append("")
    lines.append(f"**结论: {gate_res['verdict']}**")
    lines.append("")
    lines.append("| 门禁条件 | 要求 | 实际 | 判定 |")
    lines.append("|---|---|---|---|")
    gc = gate_res["criteria"]
    corr_vals = ", ".join(f"{c}={v:.3f}" for c, v in gc["corr_in_range"]["values"].items())
    lines.append(f"| corr(P,CC) ∈ [0.60, 0.95] | 全部门控 ETF | {corr_vals} | "
                 f"{'✓' if gc['corr_in_range']['pass'] else '✗'} |")
    lines.append(f"| P 领先事件 ≥ 2/3 | ≥66.7% | {gc['P_leads_events']['lead_ratio']:.1%} "
                 f"({gc['P_leads_events']['lead_count']}/{gc['P_leads_events']['total']}) | "
                 f"{'✓' if gc['P_leads_events']['pass'] else '✗'} |")
    noise_vals = ", ".join(f"{c}={v:.3f}" for c, v in gc["noise_smoother"]["ratios"].items())
    lines.append(f"| noise(P) < noise(CC) | P/CC ratio < 1 | {noise_vals} | "
                 f"{'✓' if gc['noise_smoother']['pass'] else '✗'} |")
    lines.append("")

    # Analysis 1
    lines.append("## 1. 时序相关性")
    lines.append("")
    lines.append("| ETF | corr(P, CC) | corr(GK, CC) | N (周) |")
    lines.append("|---|---|---|---|")
    for col in ALL_ETFS:
        r = corr_res[col]
        lines.append(f"| {col} | {r['corr_p_cc']:.4f} | {r['corr_gk_cc']:.4f} | {r['n']} |")
    lines.append("")

    # Analysis 2
    lines.append("## 2. 噪声比")
    lines.append("")
    lines.append("| ETF | std(ΔP/P) | std(ΔCC/CC) | std(ΔGK/GK) | P/CC ratio | P更平滑 |")
    lines.append("|---|---|---|---|---|---|")
    for col in ALL_ETFS:
        r = noise_res[col]
        lines.append(f"| {col} | {r['noise_std_P']:.4f} | {r['noise_std_CC']:.4f} | "
                     f"{r['noise_std_GK']:.4f} | {r['noise_ratio_P_over_CC']:.3f} | "
                     f"{'✓' if r['P_smoother_than_CC'] else '✗'} |")
    lines.append("")

    # Analysis 3
    lines.append("## 3. 领先/滞后关系")
    lines.append("")
    lines.append("| ETF | 最优 lag | 最优 corr | P 领先? | xcorr[-2..+2] |")
    lines.append("|---|---|---|---|---|")
    for col in ALL_ETFS:
        r = lead_res[col]
        xc_str = " / ".join(f"{r['xcorr'].get(l, np.nan):.3f}" for l in [-2, -1, 0, 1, 2])
        lines.append(f"| {col} | {r['best_lag']} | {r['best_corr']:.4f} | "
                     f"{'✓' if r['P_leads'] else '✗'} | {xc_str} |")
    lines.append("")
    lines.append("正 lag = P 领先 CC（P 的当前值预测 CC 未来值）")
    lines.append("")

    # Analysis 4
    lines.append("## 4. 信息补集 (P − CC)")
    lines.append("")
    lines.append("| ETF | 均值 | 标准差 | 偏度 | AC(1) | 正值占比 |")
    lines.append("|---|---|---|---|---|---|")
    for col in ALL_ETFS:
        r = info_res[col]
        if r:
            lines.append(f"| {col} | {r['mean']:.4f} | {r['std']:.4f} | "
                         f"{r['skew']:.3f} | {r['ac1']:.3f} | {r['pct_positive']:.1%} |")
    lines.append("")

    # Analysis 5
    lines.append("## 5. 极端事件响应")
    lines.append("")
    for col in GATE_ETFS:
        lines.append(f"### {col}")
        lines.append("")
        lines.append("| 事件 | P 首次突破75% | CC 首次突破75% | P 领先(周) |")
        lines.append("|---|---|---|---|")
        for ev_name, ev_data in event_res[col].items():
            p_d = ev_data["P_first_breach"] or "未触发"
            cc_d = ev_data["CC_first_breach"] or "未触发"
            lead = f"{ev_data['P_leads_weeks']:.1f}" if ev_data["P_leads_weeks"] is not None else "N/A"
            lines.append(f"| {ev_name} | {p_d} | {cc_d} | {lead} |")
        lines.append("")

    # Analysis 6
    lines.append("## 6. QDII 溢价影响排查")
    lines.append("")
    lines.append(f"| 指标 | 纳指ETF (QDII) | 境内均值 | 差值 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| corr(P, CC) | {qdii_res['qdii_corr_p_cc']:.4f} | "
                 f"{qdii_res['domestic_mean_corr_p_cc']:.4f} | {qdii_res['corr_diff']:+.4f} |")
    lines.append(f"| noise ratio (P/CC) | {qdii_res['qdii_noise_ratio']:.4f} | "
                 f"{qdii_res['domestic_mean_noise_ratio']:.4f} | {qdii_res['noise_diff']:+.4f} |")
    lines.append("")
    lines.append(f"**异常判定**: {'⚠ QDII 溢价扭曲显著' if qdii_res['qdii_anomalous'] else '✓ 无显著异常'}")
    lines.append("")

    return "\n".join(lines)


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 70)
    print(" E1: 离线信息增量评估")
    print("=" * 70)

    # --- Recompute vol from E0 ---
    print("\n[准备] 复用 E0 数据管线...")
    weekly_ohlc = aggregate_weekly_ohlc()
    nav = pd.read_csv(NAV_FILE, index_col=0, parse_dates=True)

    high_df = pd.DataFrame(index=nav.index, columns=nav.columns, dtype=float)
    low_df = pd.DataFrame(index=nav.index, columns=nav.columns, dtype=float)
    ohlc_full = {}
    for code, col_name in ETF_MAP.items():
        wk = weekly_ohlc[code]
        high_df[col_name] = wk["high"]
        low_df[col_name] = wk["low"]
        ohlc_full[col_name] = wk[["open", "high", "low", "close"]]

    vol_p = calc_vol_parkinson(high_df, low_df, window=14)
    vol_cc = calc_vol_cc_tapered(nav, window=14, taper=5)
    vol_gk = pd.DataFrame(index=nav.index, columns=nav.columns, dtype=float)
    for col_name in nav.columns:
        vol_gk[col_name] = calc_vol_garman_klass(ohlc_full[col_name], window=14)
    print("  ✓ vol_P, vol_CC, vol_GK ready")

    # --- Run analyses ---
    print("\n[1] 时序相关性...")
    corr_res = analysis_correlation(vol_p, vol_cc, vol_gk)
    for col in ALL_ETFS:
        print(f"  {col}: corr(P,CC)={corr_res[col]['corr_p_cc']:.4f}, "
              f"corr(GK,CC)={corr_res[col]['corr_gk_cc']:.4f}")

    print("\n[2] 噪声比...")
    noise_res = analysis_noise(vol_p, vol_cc, vol_gk)
    for col in ALL_ETFS:
        r = noise_res[col]
        print(f"  {col}: P/CC={r['noise_ratio_P_over_CC']:.3f} "
              f"{'✓ P更平滑' if r['P_smoother_than_CC'] else '✗ CC更平滑'}")

    print("\n[3] 领先/滞后...")
    lead_res = analysis_lead_lag(vol_p, vol_cc)
    for col in ALL_ETFS:
        r = lead_res[col]
        print(f"  {col}: best_lag={r['best_lag']}, corr={r['best_corr']:.4f}")

    print("\n[4] 信息补集 (P-CC)...")
    info_res = analysis_info_complement(vol_p, vol_cc)
    for col in ALL_ETFS:
        r = info_res[col]
        if r:
            print(f"  {col}: mean={r['mean']:.4f}, std={r['std']:.4f}, "
                  f"skew={r['skew']:.3f}, positive={r['pct_positive']:.1%}")

    print("\n[5] 极端事件响应...")
    event_res = analysis_extreme_events(vol_p, vol_cc)
    for col in GATE_ETFS:
        for ev_name, ev_data in event_res[col].items():
            lead = ev_data.get("P_leads_weeks")
            lead_str = f"{lead:+.1f}w" if lead is not None else "N/A"
            print(f"  {col} | {ev_name}: P leads {lead_str}")

    print("\n[6] QDII 溢价排查...")
    qdii_res = analysis_qdii_premium(corr_res, noise_res)
    print(f"  corr diff={qdii_res['corr_diff']:+.4f}, "
          f"noise diff={qdii_res['noise_diff']:+.4f}, "
          f"anomalous={qdii_res['qdii_anomalous']}")

    # --- Gate decision ---
    print("\n" + "=" * 70)
    gate_res = gate_decision(corr_res, noise_res, event_res)
    print(f" 门禁判定: **{gate_res['verdict']}**")
    print("=" * 70)
    gc = gate_res["criteria"]
    print(f"  [1] corr ∈ [0.60, 0.95]: {'PASS' if gc['corr_in_range']['pass'] else 'FAIL'}")
    print(f"  [2] P leads ≥ 2/3 events: {'PASS' if gc['P_leads_events']['pass'] else 'FAIL'} "
          f"({gc['P_leads_events']['lead_ratio']:.1%})")
    print(f"  [3] noise(P) < noise(CC): {'PASS' if gc['noise_smoother']['pass'] else 'FAIL'}")

    # --- Save outputs ---
    all_results = {
        "correlation": corr_res,
        "noise": noise_res,
        "lead_lag": lead_res,
        "info_complement": info_res,
        "extreme_events": event_res,
        "qdii_premium": qdii_res,
        "gate_decision": gate_res,
    }

    json_path = OUT / "exp_hl_vol_e1.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  JSON saved: {json_path}")

    md = render_report(corr_res, noise_res, lead_res, info_res, event_res, qdii_res, gate_res)
    md_path = OUT / "exp_hl_vol_e1.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  Report saved: {md_path}")


if __name__ == "__main__":
    main()
