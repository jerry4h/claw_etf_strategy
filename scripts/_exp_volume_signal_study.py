#!/usr/bin/env python3
"""E0 + E1: 成交量/成交额信号增量价值探索。

核心思路: 成交量是完全不同于价格波动的信息维度（流动性/情绪/套利活跃度）。
计算 5 个候选量因子的 rank IC、正交性、自相关、分组回测，判定是否有增量。

用法: .venv/bin/python scripts/_exp_volume_signal_study.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from scripts._exp_hl_vol_study import ETF_MAP, NAV_FILE, HL_REAL_START_512890, load_daily_ohlc

CACHE = PROJ / "data" / "experiments" / "tushare_cache"


def load_daily_full(code: str) -> pd.DataFrame:
    """Load full daily data including vol/amount."""
    path = CACHE / f"fund_daily_{code}.csv"
    df = pd.read_csv(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df

OUT = PROJ / "output" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)

ALL_ETFS = ["纳指ETF", "红利低波ETF", "中证500ETF", "黄金ETF", "国债ETF"]
FACTOR_NAMES = [
    "volume_change",
    "volume_ma_ratio",
    "price_volume_divergence",
    "turnover_intensity",
    "volume_volatility",
]


# ======================================================================
# E0: Data aggregation — daily vol/amount to weekly
# ======================================================================
def aggregate_weekly_volume(nav_index: pd.DatetimeIndex):
    """Aggregate daily vol and amount to weekly, aligned to NAV dates.

    Returns:
        weekly_vol: DataFrame (n_weeks, 5) — weekly total volume per ETF
        weekly_amt: DataFrame (n_weeks, 5) — weekly total amount per ETF
    """
    nav_week_map = {}  # (isoyear, isoweek) -> nav_date
    for dt in nav_index:
        iy = dt.isocalendar().year
        iw = dt.isocalendar().week
        nav_week_map[(iy, iw)] = dt

    weekly_vol = pd.DataFrame(index=nav_index, columns=ALL_ETFS, dtype=float)
    weekly_amt = pd.DataFrame(index=nav_index, columns=ALL_ETFS, dtype=float)

    for code, col_name in ETF_MAP.items():
        daily = load_daily_full(code)
        daily["isoyear"] = daily["trade_date"].dt.isocalendar().year.values
        daily["isoweek"] = daily["trade_date"].dt.isocalendar().week.values

        for (year, week), grp in daily.groupby(["isoyear", "isoweek"]):
            nav_date = nav_week_map.get((year, week))
            if nav_date is None:
                continue
            weekly_vol.loc[nav_date, col_name] = grp["vol"].sum()
            weekly_amt.loc[nav_date, col_name] = grp["amount"].sum()

    return weekly_vol.astype(float), weekly_amt.astype(float)


# ======================================================================
# E0: Factor construction
# ======================================================================
def compute_volume_factors(weekly_vol, weekly_amt, nav):
    """Compute 5 volume-based factors.

    Returns dict of factor_name -> DataFrame (n_weeks, 5).
    """
    factors = {}

    # 1. volume_change: log(vol_t / vol_t-1)
    vol_change = np.log(weekly_vol / weekly_vol.shift(1))
    factors["volume_change"] = vol_change

    # 2. volume_ma_ratio: vol_t / rolling_mean(vol, 14 weeks)
    vol_ma = weekly_vol.rolling(window=14, min_periods=10).mean()
    vol_ma_ratio = weekly_vol / vol_ma
    factors["volume_ma_ratio"] = vol_ma_ratio

    # 3. price_volume_divergence: rolling corr(ret, vol_change) over 8 weeks
    #    Negative corr = divergence (price up + volume down = bearish signal)
    weekly_ret = np.log(nav / nav.shift(1))
    pv_div = weekly_ret.rolling(window=8, min_periods=6).corr(vol_change)
    factors["price_volume_divergence"] = pv_div

    # 4. turnover_intensity: amount / close (standardized liquidity proxy)
    #    Higher = more active trading relative to price level
    turnover = weekly_amt / nav  # amount per unit of NAV
    # Cross-sectional z-score for each week
    turnover_z = turnover.apply(lambda row: (row - row.mean()) / row.std()
                                if row.std() > 0 else row * 0, axis=1)
    factors["turnover_intensity"] = turnover_z

    # 5. volume_volatility: std of daily vol within each week (normalized by mean)
    #    = coefficient of variation of daily volume within week
    factors["volume_volatility"] = _compute_vol_volatility(nav.index)

    return factors


def _compute_vol_volatility(nav_index):
    """CV of daily volume within each week."""
    nav_week_map = {}
    for dt in nav_index:
        iy = dt.isocalendar().year
        iw = dt.isocalendar().week
        nav_week_map[(iy, iw)] = dt

    result = pd.DataFrame(index=nav_index, columns=ALL_ETFS, dtype=float)

    for code, col_name in ETF_MAP.items():
        daily = load_daily_full(code)
        daily["isoyear"] = daily["trade_date"].dt.isocalendar().year.values
        daily["isoweek"] = daily["trade_date"].dt.isocalendar().week.values

        for (year, week), grp in daily.groupby(["isoyear", "isoweek"]):
            nav_date = nav_week_map.get((year, week))
            if nav_date is None:
                continue
            vols = grp["vol"].values
            if len(vols) >= 2 and vols.mean() > 0:
                cv = vols.std() / vols.mean()
                result.loc[nav_date, col_name] = cv

    return result.astype(float)


# ======================================================================
# E1: Signal quality analysis
# ======================================================================
def compute_forward_returns(nav, horizon=1):
    """Compute forward log returns (shift by -horizon for alignment with current factors)."""
    fwd = np.log(nav.shift(-horizon) / nav)
    return fwd


def analysis_rank_ic(factors, fwd_ret):
    """Rank IC: weekly cross-sectional Spearman rank correlation between factor and forward return.

    With only 5 assets, we compute panel (time-series) rank IC instead:
    For each ETF, compute time-series correlation between factor rank and fwd return rank.
    Then also compute pooled cross-sectional IC each week.
    """
    results = {}
    for fname, fdf in factors.items():
        # Cross-sectional IC per week (across 5 ETFs)
        ic_series = []
        valid_idx = fdf.dropna(how="all").index.intersection(fwd_ret.dropna(how="all").index)
        for dt in valid_idx:
            f_row = fdf.loc[dt].dropna()
            r_row = fwd_ret.loc[dt].dropna()
            common = f_row.index.intersection(r_row.index)
            if len(common) >= 4:  # need at least 4 for meaningful rank corr
                rho, _ = sp_stats.spearmanr(f_row[common], r_row[common])
                if not np.isnan(rho):
                    ic_series.append(rho)

        ic_arr = np.array(ic_series)
        if len(ic_arr) < 30:
            results[fname] = {"mean_IC": np.nan, "std_IC": np.nan, "t_stat": np.nan,
                              "n_weeks": len(ic_arr), "ir": np.nan}
            continue
        mean_ic = ic_arr.mean()
        std_ic = ic_arr.std()
        t_stat = mean_ic / (std_ic / np.sqrt(len(ic_arr))) if std_ic > 0 else 0
        ir = mean_ic / std_ic if std_ic > 0 else 0
        results[fname] = {
            "mean_IC": float(mean_ic),
            "std_IC": float(std_ic),
            "t_stat": float(t_stat),
            "n_weeks": len(ic_arr),
            "ir": float(ir),
            "pct_positive": float((ic_arr > 0).mean()),
        }
    return results


def analysis_orthogonality(factors, nav, config_window=14):
    """Correlation of each volume factor with momentum and volatility."""
    from src.factors import calculate_momentum, calculate_volatility_tapered
    momentum = calculate_momentum(nav, window=6)
    volatility = calculate_volatility_tapered(nav, window=config_window, taper=7)

    results = {}
    for fname, fdf in factors.items():
        # Pool all ETF-weeks together for correlation
        f_vals = []
        mom_vals = []
        vol_vals = []
        valid = fdf.dropna(how="all").index
        valid = valid.intersection(momentum.dropna(how="all").index)
        valid = valid.intersection(volatility.dropna(how="all").index)

        for col in ALL_ETFS:
            mask = fdf[col].notna() & momentum[col].notna() & volatility[col].notna()
            f_vals.extend(fdf.loc[mask, col].values)
            mom_vals.extend(momentum.loc[mask, col].values)
            vol_vals.extend(volatility.loc[mask, col].values)

        f_arr = np.array(f_vals)
        mom_arr = np.array(mom_vals)
        vol_arr = np.array(vol_vals)

        if len(f_arr) > 30:
            corr_mom = float(np.corrcoef(f_arr, mom_arr)[0, 1])
            corr_vol = float(np.corrcoef(f_arr, vol_arr)[0, 1])
        else:
            corr_mom = np.nan
            corr_vol = np.nan

        results[fname] = {
            "corr_with_momentum": corr_mom,
            "corr_with_volatility": corr_vol,
            "orthogonal_to_both": abs(corr_mom) < 0.30 and abs(corr_vol) < 0.30,
            "n_obs": len(f_arr),
        }
    return results


def analysis_autocorrelation(factors):
    """Factor autocorrelation AC(1) — pooled across ETFs."""
    results = {}
    for fname, fdf in factors.items():
        ac1_list = []
        for col in ALL_ETFS:
            s = fdf[col].dropna()
            if len(s) > 20:
                ac1_list.append(float(s.autocorr(lag=1)))
        results[fname] = {
            "mean_AC1": float(np.nanmean(ac1_list)),
            "per_etf_AC1": {ALL_ETFS[i]: ac1_list[i] for i in range(len(ac1_list))},
        }
    return results


def analysis_quintile_returns(factors, fwd_ret):
    """Quintile (actually tercile for 5 assets) sort: top vs bottom group returns.

    With 5 assets, we split into top-2 vs bottom-2 each week.
    """
    results = {}
    for fname, fdf in factors.items():
        top_rets = []
        bot_rets = []
        valid_idx = fdf.dropna(how="all").index.intersection(fwd_ret.dropna(how="all").index)
        for dt in valid_idx:
            f_row = fdf.loc[dt].dropna()
            r_row = fwd_ret.loc[dt].dropna()
            common = f_row.index.intersection(r_row.index)
            if len(common) >= 4:
                ranked = f_row[common].rank()
                top_mask = ranked >= ranked.quantile(0.6)
                bot_mask = ranked <= ranked.quantile(0.4)
                top_r = r_row[common][top_mask].mean()
                bot_r = r_row[common][bot_mask].mean()
                if not np.isnan(top_r) and not np.isnan(bot_r):
                    top_rets.append(top_r)
                    bot_rets.append(bot_r)

        if len(top_rets) < 30:
            results[fname] = {"top_mean": np.nan, "bot_mean": np.nan, "spread": np.nan}
            continue
        top_arr = np.array(top_rets)
        bot_arr = np.array(bot_rets)
        spread = top_arr - bot_arr
        results[fname] = {
            "top_mean_annual": float(top_arr.mean() * 52),
            "bot_mean_annual": float(bot_arr.mean() * 52),
            "spread_annual": float(spread.mean() * 52),
            "spread_t_stat": float(spread.mean() / (spread.std() / np.sqrt(len(spread))))
            if spread.std() > 0 else 0,
            "n_weeks": len(spread),
            "spread_positive_pct": float((spread > 0).mean()),
        }
    return results


def analysis_qdii_comparison(factors, fwd_ret):
    """Compare IC for 纳指 (QDII) vs domestic ETFs to check if QDII volume is distorted."""
    results = {}
    domestic = ["中证500ETF", "黄金ETF", "国债ETF"]
    for fname, fdf in factors.items():
        # Per-ETF time-series IC (rank corr between factor and next-week return)
        etf_ics = {}
        for col in ALL_ETFS:
            f_s = fdf[col].dropna()
            r_s = fwd_ret[col].dropna()
            common = f_s.index.intersection(r_s.index)
            if len(common) > 50:
                rho, _ = sp_stats.spearmanr(f_s[common], r_s[common])
                etf_ics[col] = float(rho)
            else:
                etf_ics[col] = np.nan
        qdii_ic = etf_ics.get("纳指ETF", np.nan)
        dom_ics = [etf_ics[c] for c in domestic if not np.isnan(etf_ics.get(c, np.nan))]
        dom_mean = float(np.mean(dom_ics)) if dom_ics else np.nan
        results[fname] = {
            "per_etf_IC": etf_ics,
            "qdii_ic": qdii_ic,
            "domestic_mean_ic": dom_mean,
            "qdii_anomalous": abs(qdii_ic - dom_mean) > 0.10 if not np.isnan(qdii_ic) and not np.isnan(dom_mean) else None,
        }
    return results


def analysis_economic_intuition(factors, fwd_ret, weekly_vol):
    """Economic sense check: do high-volume weeks predict better or worse returns?"""
    results = {}
    # Simple test: when vol_ma_ratio > 1.5 (above average), what's mean fwd return?
    vol_ma_ratio = factors["volume_ma_ratio"]
    high_vol_rets = []
    low_vol_rets = []
    valid = vol_ma_ratio.dropna(how="all").index.intersection(fwd_ret.dropna(how="all").index)
    for dt in valid:
        for col in ALL_ETFS:
            vm = vol_ma_ratio.loc[dt, col] if pd.notna(vol_ma_ratio.loc[dt, col]) else None
            fr = fwd_ret.loc[dt, col] if pd.notna(fwd_ret.loc[dt, col]) else None
            if vm is not None and fr is not None:
                if vm > 1.5:
                    high_vol_rets.append(fr)
                elif vm < 0.7:
                    low_vol_rets.append(fr)

    high_arr = np.array(high_vol_rets) if high_vol_rets else np.array([np.nan])
    low_arr = np.array(low_vol_rets) if low_vol_rets else np.array([np.nan])
    results["high_volume_weeks"] = {
        "mean_fwd_ret_annual": float(np.nanmean(high_arr) * 52),
        "n_obs": len(high_vol_rets),
    }
    results["low_volume_weeks"] = {
        "mean_fwd_ret_annual": float(np.nanmean(low_arr) * 52),
        "n_obs": len(low_vol_rets),
    }
    results["interpretation"] = (
        "放量买入" if np.nanmean(high_arr) > np.nanmean(low_arr)
        else "缩量买入"
    ) + " 表现更优" if not np.isnan(np.nanmean(high_arr)) else "数据不足"
    return results


# ======================================================================
# Gate decision
# ======================================================================
def gate_decision(ic_results, ortho_results):
    """Apply go/no-go gate.

    GO: |IC| >= 0.05 AND |t| >= 2.0 AND orthogonal (corr < 0.30 with both mom/vol)
    CONDITIONAL: |IC| in [0.03, 0.05]
    NO-GO: all |IC| < 0.03 or high correlation with existing factors
    """
    go_factors = []
    conditional_factors = []
    nogo_reasons = []

    for fname in FACTOR_NAMES:
        ic = ic_results[fname]
        ort = ortho_results[fname]
        abs_ic = abs(ic["mean_IC"]) if not np.isnan(ic["mean_IC"]) else 0
        abs_t = abs(ic["t_stat"]) if not np.isnan(ic["t_stat"]) else 0
        is_orthogonal = ort["orthogonal_to_both"]

        if abs_ic >= 0.05 and abs_t >= 2.0 and is_orthogonal:
            go_factors.append(fname)
        elif abs_ic >= 0.03 and abs_t >= 1.5:
            conditional_factors.append(fname)

    # Check if any factor is highly correlated with existing
    high_corr_factors = []
    for fname in FACTOR_NAMES:
        ort = ortho_results[fname]
        if abs(ort["corr_with_momentum"]) > 0.50 or abs(ort["corr_with_volatility"]) > 0.50:
            high_corr_factors.append(fname)

    if go_factors:
        verdict = "GO"
    elif conditional_factors:
        verdict = "CONDITIONAL"
    else:
        verdict = "NO-GO"

    if not go_factors and not conditional_factors:
        nogo_reasons.append("全部因子 |IC| < 0.03 或 |t| < 1.5")
    if high_corr_factors:
        nogo_reasons.append(f"高相关因子: {', '.join(high_corr_factors)}")

    return {
        "verdict": verdict,
        "go_factors": go_factors,
        "conditional_factors": conditional_factors,
        "high_corr_factors": high_corr_factors,
        "nogo_reasons": nogo_reasons,
        "details": {
            fname: {
                "abs_IC": abs(ic_results[fname]["mean_IC"]) if not np.isnan(ic_results[fname]["mean_IC"]) else 0,
                "abs_t": abs(ic_results[fname]["t_stat"]) if not np.isnan(ic_results[fname]["t_stat"]) else 0,
                "orthogonal": ortho_results[fname]["orthogonal_to_both"],
            }
            for fname in FACTOR_NAMES
        },
    }


# ======================================================================
# Report
# ======================================================================
def render_report(ic_res, ortho_res, ac_res, quint_res, qdii_res, econ_res, gate_res, coverage):
    L = ["# E1-Volume: 成交量信号增量价值评估报告", ""]
    L.append(f"> 5 个量因子 × 5 只 ETF 周频 | 门禁判定: **{gate_res['verdict']}**")
    L.append("")

    # Gate summary
    L.append("## 门禁判定")
    L.append("")
    L.append(f"**结论: {gate_res['verdict']}**")
    if gate_res["go_factors"]:
        L.append(f"\nGO 因子: {', '.join(gate_res['go_factors'])}")
    if gate_res["conditional_factors"]:
        L.append(f"\nCONDITIONAL 因子: {', '.join(gate_res['conditional_factors'])}")
    if gate_res["nogo_reasons"]:
        L.append(f"\nNO-GO 原因: {'; '.join(gate_res['nogo_reasons'])}")
    L.append("")
    L.append("| 因子 | |IC| | |t-stat| | 正交? | 判定 |")
    L.append("|---|---|---|---|---|")
    for fname in FACTOR_NAMES:
        d = gate_res["details"][fname]
        status = "GO" if fname in gate_res["go_factors"] else (
            "COND" if fname in gate_res["conditional_factors"] else "—")
        L.append(f"| {fname} | {d['abs_IC']:.4f} | {d['abs_t']:.2f} | "
                 f"{'✓' if d['orthogonal'] else '✗'} | {status} |")
    L.append("")

    # Data coverage
    L.append("## 0. 数据覆盖")
    L.append("")
    L.append("| ETF | 日频天数 | 起止日期 | 周频有效 |")
    L.append("|---|---|---|---|")
    for etf, info in coverage.items():
        L.append(f"| {etf} | {info['n_days']} | {info['start']} ~ {info['end']} | {info['n_weeks']} 周 |")
    L.append("")

    # 1. Rank IC
    L.append("## 1. Rank IC (截面秩相关)")
    L.append("")
    L.append("| 因子 | mean IC | std IC | t-stat | IR | IC>0 占比 | N周 |")
    L.append("|---|---|---|---|---|---|---|")
    for fname in FACTOR_NAMES:
        r = ic_res[fname]
        L.append(f"| {fname} | {r['mean_IC']:.4f} | {r['std_IC']:.4f} | "
                 f"{r['t_stat']:.2f} | {r['ir']:.3f} | "
                 f"{r.get('pct_positive', 0):.1%} | {r['n_weeks']} |")
    L.append("")
    L.append("门禁标准: |IC| ≥ 0.05 且 |t| ≥ 2.0 → GO; |IC| ∈ [0.03,0.05] → CONDITIONAL")
    L.append("")

    # 2. Orthogonality
    L.append("## 2. 与现有因子正交性")
    L.append("")
    L.append("| 因子 | corr(momentum) | corr(volatility) | 正交(<0.30) |")
    L.append("|---|---|---|---|")
    for fname in FACTOR_NAMES:
        r = ortho_res[fname]
        L.append(f"| {fname} | {r['corr_with_momentum']:+.4f} | "
                 f"{r['corr_with_volatility']:+.4f} | "
                 f"{'✓' if r['orthogonal_to_both'] else '✗'} |")
    L.append("")

    # 3. Autocorrelation
    L.append("## 3. 因子自相关 AC(1)")
    L.append("")
    L.append("| 因子 | 均值 AC(1) | 解读 |")
    L.append("|---|---|---|")
    for fname in FACTOR_NAMES:
        r = ac_res[fname]
        ac = r["mean_AC1"]
        interp = "高持续性" if ac > 0.5 else ("中等" if ac > 0.2 else "低/噪声")
        L.append(f"| {fname} | {ac:.3f} | {interp} |")
    L.append("")

    # 4. Quintile returns
    L.append("## 4. 分组回测 (top-2 vs bottom-2)")
    L.append("")
    L.append("| 因子 | Top年化 | Bottom年化 | 多空价差年化 | 价差t | 正向率 |")
    L.append("|---|---|---|---|---|---|")
    for fname in FACTOR_NAMES:
        r = quint_res[fname]
        L.append(f"| {fname} | {r['top_mean_annual']:.2%} | {r['bot_mean_annual']:.2%} | "
                 f"{r['spread_annual']:.2%} | {r['spread_t_stat']:.2f} | "
                 f"{r.get('spread_positive_pct', 0):.1%} |")
    L.append("")

    # 5. QDII comparison
    L.append("## 5. QDII (纳指) vs 境内 ETF")
    L.append("")
    L.append("| 因子 | 纳指 IC | 境内均值 IC | 差值 | 异常? |")
    L.append("|---|---|---|---|---|")
    for fname in FACTOR_NAMES:
        r = qdii_res[fname]
        qi = r["qdii_ic"]
        di = r["domestic_mean_ic"]
        diff = qi - di if not np.isnan(qi) and not np.isnan(di) else np.nan
        anom = "⚠" if r["qdii_anomalous"] else "✓"
        L.append(f"| {fname} | {qi:.4f} | {di:.4f} | {diff:+.4f} | {anom} |")
    L.append("")

    # 6. Economic intuition
    L.append("## 6. 经济直觉验证")
    L.append("")
    L.append(f"- 放量周 (vol_ma_ratio > 1.5) 后一周均值收益年化: "
             f"{econ_res['high_volume_weeks']['mean_fwd_ret_annual']:.2%} "
             f"(N={econ_res['high_volume_weeks']['n_obs']})")
    L.append(f"- 缩量周 (vol_ma_ratio < 0.7) 后一周均值收益年化: "
             f"{econ_res['low_volume_weeks']['mean_fwd_ret_annual']:.2%} "
             f"(N={econ_res['low_volume_weeks']['n_obs']})")
    L.append(f"- 解读: {econ_res['interpretation']}")
    L.append("")

    # Key insight
    L.append("## 关键洞察")
    L.append("")
    L.append("ETF 成交量反映的是流动性/套利行为而非方向性信息。")
    L.append("在仅 5 只 ETF 的窄截面中，量因子的截面区分能力天然受限。")
    L.append("即便单因子 IC 显著，在现有策略（动量+波动率+PE）基础上的边际增量仍需回测验证。")
    L.append("")
    return "\n".join(L)


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 70)
    print(" E0 + E1: 成交量信号增量价值探索")
    print("=" * 70)

    nav = pd.read_csv(NAV_FILE, index_col=0, parse_dates=True)

    # --- E0: Aggregate data ---
    print("\n[E0] 聚合日频成交量/成交额到周频...")
    weekly_vol, weekly_amt = aggregate_weekly_volume(nav.index)
    print(f"  weekly_vol shape: {weekly_vol.shape}, non-null per ETF:")
    for col in ALL_ETFS:
        nn = weekly_vol[col].notna().sum()
        print(f"    {col}: {nn} weeks")

    # Coverage report
    coverage = {}
    for code, col_name in ETF_MAP.items():
        daily = load_daily_full(code)
        coverage[col_name] = {
            "n_days": len(daily),
            "start": str(daily["trade_date"].iloc[0].date()),
            "end": str(daily["trade_date"].iloc[-1].date()),
            "n_weeks": int(weekly_vol[col_name].notna().sum()),
        }

    # --- E0: Compute factors ---
    print("\n[E0] 计算 5 个量因子...")
    factors = compute_volume_factors(weekly_vol, weekly_amt, nav)
    for fname in FACTOR_NAMES:
        fdf = factors[fname]
        nn = fdf.notna().sum().sum()
        print(f"  {fname}: {nn} non-null values")

    # Handle 红利低波 pre-2019: set to NaN (already NaN from data absence)
    for fname in FACTOR_NAMES:
        mask_512890 = factors[fname].index < HL_REAL_START_512890
        factors[fname].loc[mask_512890, "红利低波ETF"] = np.nan

    # --- E1: Analyses ---
    fwd_ret = compute_forward_returns(nav, horizon=1)

    print("\n[E1-1] Rank IC...")
    ic_res = analysis_rank_ic(factors, fwd_ret)
    for fname in FACTOR_NAMES:
        r = ic_res[fname]
        print(f"  {fname}: IC={r['mean_IC']:.4f}, t={r['t_stat']:.2f}")

    print("\n[E1-2] 正交性...")
    ortho_res = analysis_orthogonality(factors, nav)
    for fname in FACTOR_NAMES:
        r = ortho_res[fname]
        print(f"  {fname}: corr_mom={r['corr_with_momentum']:+.3f}, "
              f"corr_vol={r['corr_with_volatility']:+.3f}, "
              f"orthogonal={'✓' if r['orthogonal_to_both'] else '✗'}")

    print("\n[E1-3] 自相关...")
    ac_res = analysis_autocorrelation(factors)
    for fname in FACTOR_NAMES:
        print(f"  {fname}: AC(1)={ac_res[fname]['mean_AC1']:.3f}")

    print("\n[E1-4] 分组回测...")
    quint_res = analysis_quintile_returns(factors, fwd_ret)
    for fname in FACTOR_NAMES:
        r = quint_res[fname]
        print(f"  {fname}: spread={r['spread_annual']:.2%}, t={r['spread_t_stat']:.2f}")

    print("\n[E1-5] QDII 对比...")
    qdii_res = analysis_qdii_comparison(factors, fwd_ret)
    for fname in FACTOR_NAMES:
        r = qdii_res[fname]
        print(f"  {fname}: qdii_IC={r['qdii_ic']:.4f}, domestic={r['domestic_mean_ic']:.4f}, "
              f"anomalous={r['qdii_anomalous']}")

    print("\n[E1-6] 经济直觉...")
    econ_res = analysis_economic_intuition(factors, fwd_ret, weekly_vol)
    print(f"  放量周后: {econ_res['high_volume_weeks']['mean_fwd_ret_annual']:.2%}")
    print(f"  缩量周后: {econ_res['low_volume_weeks']['mean_fwd_ret_annual']:.2%}")
    print(f"  解读: {econ_res['interpretation']}")

    # --- Gate ---
    print("\n" + "=" * 70)
    gate_res = gate_decision(ic_res, ortho_res)
    print(f" 门禁判定: **{gate_res['verdict']}**")
    print("=" * 70)
    for fname in FACTOR_NAMES:
        d = gate_res["details"][fname]
        status = "GO" if fname in gate_res["go_factors"] else (
            "COND" if fname in gate_res["conditional_factors"] else "—")
        print(f"  {fname}: |IC|={d['abs_IC']:.4f}, |t|={d['abs_t']:.2f}, "
              f"orth={'✓' if d['orthogonal'] else '✗'} → {status}")
    if gate_res["nogo_reasons"]:
        print(f"  原因: {'; '.join(gate_res['nogo_reasons'])}")

    # --- Save ---
    all_results = {
        "coverage": coverage,
        "rank_ic": ic_res,
        "orthogonality": ortho_res,
        "autocorrelation": ac_res,
        "quintile_returns": quint_res,
        "qdii_comparison": qdii_res,
        "economic_intuition": econ_res,
        "gate_decision": gate_res,
    }

    json_path = OUT / "exp_volume_signal_e1.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  JSON saved: {json_path}")

    md = render_report(ic_res, ortho_res, ac_res, quint_res, qdii_res, econ_res, gate_res, coverage)
    md_path = OUT / "exp_volume_signal_e1.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  Report saved: {md_path}")


if __name__ == "__main__":
    main()
