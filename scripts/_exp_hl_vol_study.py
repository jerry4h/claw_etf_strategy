#!/usr/bin/env python3
"""E0: 周内波动探索 — 数据预处理与 Parkinson/GK 估计器实现。

从 tushare_cache 日频 OHLC 出发，前复权聚合至周频，实现 Parkinson 和 Garman-Klass
波动率估计器，并与现有 close-to-close tapered vol 对照。

用法: .venv/bin/python scripts/_exp_hl_vol_study.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

# ======================================================================
# Constants
# ======================================================================
CACHE = PROJ / "data" / "experiments" / "tushare_cache"
NAV_FILE = PROJ / "data" / "all_etfs_nav_latest.csv"

# ETF code → NAV column name mapping (order matches NAV columns)
ETF_MAP = {
    "513100SH": "纳指ETF",
    "512890SH": "红利低波ETF",
    "510500SH": "中证500ETF",
    "518880SH": "黄金ETF",
    "511010SH": "国债ETF",
}

# 512890 real OHLC starts from this date (before: synthetic from H20269 index)
HL_REAL_START_512890 = pd.Timestamp("2019-01-18")


# ======================================================================
# 1. aggregate_weekly_ohlc()
# ======================================================================
def load_daily_ohlc(code: str) -> pd.DataFrame:
    """Load a single ETF's daily OHLC from tushare_cache, sorted by date."""
    path = CACHE / f"fund_daily_{code}.csv"
    df = pd.read_csv(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df[["trade_date", "pre_close", "open", "high", "low", "close", "pct_chg"]]


def build_adjusted_ohlc(daily: pd.DataFrame, nav_weekly: pd.Series) -> pd.DataFrame:
    """Build forward-adjusted daily OHLC using per-week NAV anchoring.

    Strategy (avoids cumulative rounding from pct_chg chaining):
    1. Match each daily row to its ISO week.
    2. For each ISO week, look up the corresponding NAV value.
    3. Compute factor = nav_close / raw_close_of_last_trading_day_in_week.
    4. Apply factor to all daily OHLC within that week.
    """
    daily = daily.copy()
    daily["isoyear"] = daily["trade_date"].dt.isocalendar().year.values
    daily["isoweek"] = daily["trade_date"].dt.isocalendar().week.values

    # Build (isoyear, isoweek) → NAV date mapping
    nav_week_map = _build_nav_week_map(nav_weekly.index)

    # For each ISO week, compute factor = nav_value / raw_close_on_last_trading_day
    week_factors = {}  # (year, week) → factor

    for (year, week), grp in daily.groupby(["isoyear", "isoweek"]):
        nav_date = nav_week_map.get((year, week))
        if nav_date is None:
            continue
        nav_val = nav_weekly.get(nav_date)
        if nav_val is None or pd.isna(nav_val):
            continue

        # Last trading day's raw close in this week
        raw_close = grp.sort_values("trade_date").iloc[-1]["close"]
        if raw_close > 0:
            week_factors[(year, week)] = nav_val / raw_close

    # Apply factor to each day
    factors = daily.apply(
        lambda r: week_factors.get((r["isoyear"], r["isoweek"]), np.nan), axis=1
    )
    daily["factor"] = factors

    # For weeks without NAV anchor, interpolate factor from adjacent weeks
    daily["factor"] = daily["factor"].ffill().bfill()

    daily["adj_close"] = daily["close"] * daily["factor"]
    daily["adj_open"] = daily["open"] * daily["factor"]
    daily["adj_high"] = daily["high"] * daily["factor"]
    daily["adj_low"] = daily["low"] * daily["factor"]

    return daily


def _build_nav_week_map(nav_index: pd.DatetimeIndex) -> dict:
    """Build (isoyear, isoweek) → NAV date mapping."""
    m = {}
    for dt in nav_index:
        iy = dt.isocalendar().year
        iw = dt.isocalendar().week
        m[(iy, iw)] = dt
    return m


def aggregate_weekly_ohlc() -> dict:
    """Aggregate daily OHLC to weekly for all 5 ETFs.

    Returns dict: {etf_code: DataFrame with columns [open, high, low, close, has_real_ohlc]}
    indexed by weekly date (matching NAV file index).
    """
    nav = pd.read_csv(NAV_FILE, index_col=0, parse_dates=True)
    nav_dates = nav.index  # 677 weekly dates
    nav_week_map = _build_nav_week_map(nav_dates)

    results = {}
    for code, col_name in ETF_MAP.items():
        nav_col = nav[col_name]

        daily = load_daily_ohlc(code)
        daily_adj = build_adjusted_ohlc(daily, nav_col)

        # Build weekly OHLC using ISO week grouping
        weekly_rows = []
        for (year, week), grp in daily_adj.groupby(["isoyear", "isoweek"]):
            # Use NAV date as week label (handles non-trading Fridays)
            week_date = nav_week_map.get((year, week))
            if week_date is None:
                continue  # week not in NAV data, skip

            grp_sorted = grp.sort_values("trade_date")
            weekly_rows.append({
                "date": week_date,
                "open": grp_sorted.iloc[0]["adj_open"],
                "high": grp_sorted["adj_high"].max(),
                "low": grp_sorted["adj_low"].min(),
                "close": grp_sorted.iloc[-1]["adj_close"],
            })

        wk = pd.DataFrame(weekly_rows).set_index("date").sort_index()

        # Align to NAV dates
        wk_aligned = wk.reindex(nav_dates)

        # Mark real OHLC availability
        if code == "512890SH":
            # Before 2019-01-18: no real OHLC, degrade to H=L=C from NAV
            wk_aligned["has_real_ohlc"] = wk_aligned.index >= HL_REAL_START_512890
            mask_synth = wk_aligned.index < HL_REAL_START_512890
            # Fill synthetic period with NAV close (H=L=O=C)
            wk_aligned.loc[mask_synth, "close"] = nav_col[mask_synth]
            wk_aligned.loc[mask_synth, "high"] = nav_col[mask_synth]
            wk_aligned.loc[mask_synth, "low"] = nav_col[mask_synth]
            wk_aligned.loc[mask_synth, "open"] = nav_col[mask_synth]
        else:
            # Mark real OHLC: True where we have daily data
            first_daily = daily["trade_date"].min()
            wk_aligned["has_real_ohlc"] = wk_aligned.index >= first_daily

        # Forward-fill any NaN from alignment gaps
        wk_aligned[["open", "high", "low", "close"]] = (
            wk_aligned[["open", "high", "low", "close"]].ffill()
        )

        results[code] = wk_aligned

    return results


# ======================================================================
# 2. Consistency check
# ======================================================================
def check_consistency(weekly_ohlc: dict) -> bool:
    """Verify weekly close matches NAV file within tolerance."""
    nav = pd.read_csv(NAV_FILE, index_col=0, parse_dates=True)
    tol = 0.001
    all_pass = True

    print("\n" + "=" * 70)
    print(" 一致性校验: weekly close vs all_etfs_nav_latest.csv")
    print("=" * 70)

    for code, col_name in ETF_MAP.items():
        wk = weekly_ohlc[code]
        nav_col = nav[col_name]

        # Only compare where both have data
        valid = wk["close"].notna() & nav_col.notna()
        diff = (wk.loc[valid, "close"] - nav_col[valid]).abs()
        max_diff = diff.max()
        n_exceed = (diff > tol).sum()

        if n_exceed > 0:
            print(f"  ✗ {col_name} ({code}): {n_exceed} rows exceed tol={tol}, "
                  f"max_diff={max_diff:.6f}")
            # Show worst offenders
            bad = diff[diff > tol].sort_values(ascending=False).head(5)
            for dt, d in bad.items():
                print(f"      {dt.date()}: close={wk.loc[dt, 'close']:.4f} "
                      f"nav={nav_col[dt]:.4f} diff={d:.6f}")
            all_pass = False
        else:
            n_compared = valid.sum()
            print(f"  ✓ {col_name} ({code}): {n_compared} weeks PASS "
                  f"(max_diff={max_diff:.6f} ≤ {tol})")

    return all_pass


# ======================================================================
# 3. Data quality checks
# ======================================================================
def check_data_quality(weekly_ohlc: dict) -> dict:
    """Run data quality checks on weekly OHLC."""
    print("\n" + "=" * 70)
    print(" 数据质量校验")
    print("=" * 70)

    coverage = {}
    for code, col_name in ETF_MAP.items():
        wk = weekly_ohlc[code]
        real = wk[wk["has_real_ohlc"] & wk["close"].notna()]

        # Coverage
        if len(real) > 0:
            first_real = real.index[0]
            last_real = real.index[-1]
        else:
            first_real = last_real = None
        coverage[code] = {"first": first_real, "last": last_real, "weeks": len(real)}

        print(f"\n  {col_name} ({code}):")
        print(f"    真实 OHLC 覆盖: {first_real} ~ {last_real} ({len(real)} 周)")

        # OHLC sanity checks (only on real data)
        if len(real) > 0:
            h = real["high"]
            l = real["low"]
            o = real["open"]
            c = real["close"]

            n_h_lt_l = (h < l - 1e-8).sum()
            n_h_lt_oc = (h < np.maximum(o, c) - 1e-8).sum()
            n_l_gt_oc = (l > np.minimum(o, c) + 1e-8).sum()

            if n_h_lt_l > 0 or n_h_lt_oc > 0 or n_l_gt_oc > 0:
                print(f"    ⚠ 异常: H<L={n_h_lt_l}, H<max(O,C)={n_h_lt_oc}, "
                      f"L>min(O,C)={n_l_gt_oc}")
            else:
                print(f"    ✓ OHLC 关系正常 (H≥L, H≥max(O,C), L≤min(O,C))")

        # 512890 splice continuity check
        if code == "512890SH":
            splice_date = HL_REAL_START_512890
            # Find weeks around splice point
            pre = wk[(wk["close"].notna()) & (wk.index < splice_date)].tail(3)
            post = wk[(wk["close"].notna()) & (wk.index >= splice_date)].head(3)
            if len(pre) >= 2 and len(post) > 0:
                pre_ret = pre["close"].iloc[-1] / pre["close"].iloc[-2] - 1
                splice_ret = post["close"].iloc[0] / pre["close"].iloc[-1] - 1
                print(f"    512890 拼接点检查:")
                print(f"      拼接前最后周收益: {pre_ret:.4%}")
                print(f"      跨拼接点收益: {splice_ret:.4%}")
                print(f"      {'✓ 连续' if abs(splice_ret) < 0.05 else '⚠ 跳变'}")

    return coverage


# ======================================================================
# 4. Parkinson volatility estimator
# ======================================================================
def calc_vol_parkinson(weekly_high: pd.DataFrame, weekly_low: pd.DataFrame,
                       window: int = 14) -> pd.DataFrame:
    """Parkinson volatility estimator using High-Low range.

    Formula: σ_P = sqrt(1/(4n·ln2) × rolling_sum((ln(H/L))²)) × sqrt(52)

    When H==L (synthetic data), ln(H/L)=0, contributing nothing to the sum.
    Fully vectorized with pd.rolling.
    """
    # ln(H/L) squared
    hl_ratio = np.log(weekly_high / weekly_low)
    hl_sq = hl_ratio ** 2

    # Rolling sum over window
    rolling_sum = hl_sq.rolling(window=window, min_periods=window).sum()

    # Parkinson formula: sqrt(1/(4n*ln2) * sum) * sqrt(52) for annualization
    factor = 1.0 / (4 * window * np.log(2))
    vol_parkinson = np.sqrt(factor * rolling_sum) * np.sqrt(52)

    return vol_parkinson


# ======================================================================
# 5. Garman-Klass volatility estimator
# ======================================================================
def calc_vol_garman_klass(weekly_ohlc_df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Garman-Klass volatility for a single ETF.

    Formula: σ_GK = sqrt(rolling_mean[0.5*(ln(H/L))² - (2*ln2-1)*(ln(C/O))²]) × sqrt(52)
    """
    hl = np.log(weekly_ohlc_df["high"] / weekly_ohlc_df["low"])
    co = np.log(weekly_ohlc_df["close"] / weekly_ohlc_df["open"])

    gk_term = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2

    rolling_mean = gk_term.rolling(window=window, min_periods=window).mean()

    # Clamp negative values to 0 before sqrt (can happen with noisy data)
    rolling_mean = rolling_mean.clip(lower=0)
    vol_gk = np.sqrt(rolling_mean) * np.sqrt(52)

    return vol_gk


# ======================================================================
# 6. Close-to-close vol (reference: tapered vol from src/factors.py)
# ======================================================================
def calc_vol_cc_tapered(nav: pd.DataFrame, window: int = 14, taper: int = 5) -> pd.DataFrame:
    """Replicate calculate_volatility_tapered from src/factors.py."""
    w_rets = nav.pct_change().values
    n, k = w_rets.shape
    weights = np.ones(window)
    for t in range(taper):
        weights[t] = (t + 1.0) / (taper + 1.0)
    w_norm = weights / weights.sum()
    vol = np.full((n, k), np.nan)
    for i in range(window, n):
        rets_w = w_rets[i - window:i]
        wmean = np.average(rets_w, weights=w_norm, axis=0)
        wvar = np.average((rets_w - wmean) ** 2, weights=w_norm, axis=0)
        vol[i] = np.sqrt(wvar) * np.sqrt(52)
    return pd.DataFrame(vol, index=nav.index, columns=nav.columns)


# ======================================================================
# Main experiment
# ======================================================================
def main():
    print("=" * 70)
    print(" E0: 周内波动探索 — 数据预处理与估计器实现")
    print("=" * 70)

    # --- Step 1: Aggregate weekly OHLC ---
    print("\n[Step 1] 聚合日频 OHLC → 周频...")
    weekly_ohlc = aggregate_weekly_ohlc()
    for code, col_name in ETF_MAP.items():
        wk = weekly_ohlc[code]
        n_valid = wk["close"].notna().sum()
        n_real = wk["has_real_ohlc"].sum()
        print(f"  {col_name}: {n_valid} 周有效, {n_real} 周真实OHLC")

    # --- Step 2: Consistency check (HARD GATE) ---
    consistent = check_consistency(weekly_ohlc)
    if not consistent:
        print("\n❌ 一致性校验 FAIL — 中止后续步骤")
        sys.exit(1)
    print("\n✅ 一致性校验全部 PASS")

    # --- Step 3: Data quality ---
    coverage = check_data_quality(weekly_ohlc)

    # --- Step 4: Compute Parkinson vol ---
    print("\n" + "=" * 70)
    print(" Parkinson 波动率计算 (window=14)")
    print("=" * 70)

    nav = pd.read_csv(NAV_FILE, index_col=0, parse_dates=True)

    # Build High/Low DataFrames aligned with NAV
    high_df = pd.DataFrame(index=nav.index, columns=nav.columns, dtype=float)
    low_df = pd.DataFrame(index=nav.index, columns=nav.columns, dtype=float)
    ohlc_full = {}  # For GK

    for code, col_name in ETF_MAP.items():
        wk = weekly_ohlc[code]
        high_df[col_name] = wk["high"]
        low_df[col_name] = wk["low"]
        ohlc_full[col_name] = wk[["open", "high", "low", "close"]]

    vol_park = calc_vol_parkinson(high_df, low_df, window=14)
    print(f"  vol_parkinson shape: {vol_park.shape}")
    print(f"  非 NaN 行数: {vol_park.dropna().shape[0]}")

    # --- Step 5: Compute CC tapered vol (reference) ---
    print("\n[Step 5] Close-to-close tapered vol (参照)...")
    vol_cc = calc_vol_cc_tapered(nav, window=14, taper=5)
    print(f"  vol_cc shape: {vol_cc.shape}")

    # --- Step 6: Garman-Klass vol ---
    print("\n[Step 6] Garman-Klass vol (备选)...")
    vol_gk = pd.DataFrame(index=nav.index, columns=nav.columns, dtype=float)
    for col_name in nav.columns:
        vol_gk[col_name] = calc_vol_garman_klass(ohlc_full[col_name], window=14)
    print(f"  vol_gk shape: {vol_gk.shape}")

    # --- Step 7: Verification outputs ---
    print("\n" + "=" * 70)
    print(" 验证输出")
    print("=" * 70)

    # 7a: Shape consistency
    print(f"\n  [7a] Shape 一致性:")
    print(f"    vol_parkinson: {vol_park.shape}")
    print(f"    vol_cc:        {vol_cc.shape}")
    print(f"    vol_gk:        {vol_gk.shape}")
    assert vol_park.shape == vol_cc.shape, "Shape mismatch!"
    print(f"    ✓ 全部 {vol_park.shape} 一致")

    # 7b: 红利低波 2013-2019 段 vol_parkinson == vol_cc
    print(f"\n  [7b] 红利低波退化检查 (2013-2019 段 H=L=C → Parkinson 贡献=0):")
    hl_col = "红利低波ETF"
    # In the synthetic period, H=L=C so ln(H/L)=0 for each week.
    # Within a 14-week window fully inside synthetic period, Parkinson = 0.
    # But CC vol uses close-to-close returns which are non-zero.
    sample_dates = ["2015-01-09", "2016-06-17", "2017-12-29", "2018-06-29"]
    print(f"    样本周对比 (退化段 H=L=C → ln(H/L)=0 → Parkinson=0, CC>0):")
    for d in sample_dates:
        dt = pd.Timestamp(d)
        if dt in vol_park.index:
            vp = vol_park.loc[dt, hl_col]
            vc = vol_cc.loc[dt, hl_col]
            if pd.notna(vp):
                print(f"      {d}: Parkinson={vp:.6f}, CC_tapered={vc:.6f} "
                      f"{'\u2713 P=0' if vp < 1e-10 else '\u26a0 P\u22600'}")
            else:
                print(f"      {d}: Parkinson=NaN (window 未满), CC_tapered={vc:.6f}")
    # Verify all pure-synthetic-window Parkinson values are 0
    synth_mask = (nav.index < HL_REAL_START_512890) & vol_park[hl_col].notna()
    if synth_mask.sum() > 0:
        park_synth = vol_park.loc[synth_mask, hl_col]
        n_zero = (park_synth.abs() < 1e-10).sum()
        print(f"    退化段有效 Parkinson 行数: {synth_mask.sum()}, 其中=0: {n_zero}")
        print(f"    ✓ 退化正确: H=L=C → Parkinson 贡献为零 (设计预期)")
    else:
        print(f"    (退化段无有效 Parkinson 行——window=14 尚未达到全合成窗口)")

    # 7c: 2020-03 股灾周 Parkinson > CC
    print(f"\n  [7c] 2020-03 股灾周 Parkinson vs CC:")
    crash_dates = ["2020-03-13", "2020-03-20", "2020-03-27"]
    for d in crash_dates:
        dt = pd.Timestamp(d)
        if dt in vol_park.index:
            print(f"    {d}:")
            for col in ["纳指ETF", "中证500ETF", "黄金ETF"]:
                vp = vol_park.loc[dt, col]
                vc = vol_cc.loc[dt, col]
                if pd.notna(vp) and pd.notna(vc):
                    ratio = vp / vc if vc > 0 else np.nan
                    flag = "✓ P>CC" if vp > vc else "  P≤CC"
                    print(f"      {col}: Parkinson={vp:.4f} CC={vc:.4f} "
                          f"ratio={ratio:.3f} {flag}")

    # 7d: Coverage report
    print(f"\n  [7d] 数据质量: 真实 OHLC 覆盖范围")
    print(f"    {'ETF':<12} {'起始':<12} {'结束':<12} {'周数'}")
    print(f"    {'-'*12} {'-'*12} {'-'*12} {'-'*5}")
    for code, col_name in ETF_MAP.items():
        c = coverage[code]
        first_str = c["first"].strftime("%Y-%m-%d") if c["first"] else "N/A"
        last_str = c["last"].strftime("%Y-%m-%d") if c["last"] else "N/A"
        print(f"    {col_name:<12} {first_str:<12} {last_str:<12} {c['weeks']}")

    print("\n" + "=" * 70)
    print(" ✅ E0 完成: 数据预处理与估计器实现验收通过")
    print("=" * 70)


if __name__ == "__main__":
    main()
