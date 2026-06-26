#!/usr/bin/env python3
"""
Build weekly NAV CSV from Tushare daily NAV data.
Output format compatible with existing backtest engine.
"""
import json
import csv
from pathlib import Path
from datetime import datetime

NAV_JSON = "data/tushare/etf_daily_nav.json"
OUTPUT_DIR = Path("data/tushare")


def build_weekly_nav():
    """Build weekly NAV table from daily Tushare data."""

    # Load daily NAV
    with open(NAV_JSON, 'r', encoding='utf-8') as f:
        daily_nav = json.load(f)

    print(f"Loaded NAV for {len(daily_nav)} ETFs")

    # Build a dict: date -> {ts_code: close}
    print("Building date-indexed price matrix...")
    date_prices = {}
    for ts_code, info in daily_nav.items():
        for row in info['rows']:
            d = row['trade_date']
            close = row.get('close')
            if close is None:
                continue
            if d not in date_prices:
                date_prices[d] = {}
            date_prices[d][ts_code] = close

    print(f"  {len(date_prices)} unique trading dates")

    # Sort dates
    all_dates = sorted(date_prices.keys())
    print(f"  Date range: {all_dates[0]} — {all_dates[-1]}")

    # Convert string dates to datetime for proper resampling
    dt_dates = [datetime.strptime(d, '%Y%m%d') for d in all_dates]

    # Build daily series
    from collections import defaultdict
    dailies = defaultdict(dict)
    for d_str, prices in date_prices.items():
        dt = datetime.strptime(d_str, '%Y%m%d')
        dailies[dt] = prices

    # Group by ISO week (Monday anchor)
    # We'll use ISO year-week for grouping
    print("Resampling to weekly (Monday anchor)...")
    weekly = {}  # (iso_year, iso_week) -> {ts_code: last_close}
    week_order = []

    for dt in sorted(dailies.keys()):
        iso_year, iso_week, iso_day = dt.isocalendar()
        key = (iso_year, iso_week)
        # Take the last trading day's close for each week
        if key not in weekly:
            weekly[key] = {}
            week_order.append(key)
        weekly[key].update(dailies[dt])

    print(f"  {len(weekly)} weekly periods")

    # Get the list of ETFs (sorted by ts_code)
    etf_codes = sorted(daily_nav.keys())
    etf_names = [daily_nav[c]['name'] for c in etf_codes]

    # Write CSV
    csv_path = OUTPUT_DIR / "tushare_etf_weekly_nav.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header: date as ISO week string + ETF names
        header = ['week'] + etf_names
        writer.writerow(header)

        for iso_year, iso_week in week_order:
            # Find the Monday date for this ISO week
            # ISO week 1 is the week with the first Thursday
            from datetime import date
            # Get the Monday of this ISO week
            monday = date.fromisocalendar(iso_year, iso_week, 1)
            week_str = monday.strftime('%Y-%m-%d')

            prices = weekly[(iso_year, iso_week)]
            row = [week_str]
            for code in etf_codes:
                row.append(prices.get(code, ''))
            writer.writerow(row)

    print(f"\n✅ Saved weekly NAV CSV: {csv_path}")
    print(f"   {len(etf_codes)} ETFs × {len(week_order)} weeks")

    # Also write a metadata summary
    summary_path = OUTPUT_DIR / "tushare_weekly_nav_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Tushare Weekly NAV Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"ETFs: {len(etf_codes)}\n")
        f.write(f"Weekly periods: {len(week_order)}\n")
        f.write(f"Date range: {week_order[0]} → {week_order[-1]}\n")
        f.write("\nETF List:\n")
        for code, name in zip(etf_codes, etf_names):
            info = daily_nav[code]
            f.write(f"  {code}  {name:20s}  {info['first_date']} — {info['last_date']}  ({info['count']} rows)\n")
    print(f"✅ Saved summary: {summary_path}")

    # Print overview
    print(f"\nETF coverage:")
    for code, name in zip(etf_codes, etf_names):
        info = daily_nav[code]
        weeks_with_data = sum(1 for (y, w) in week_order
                             if code in weekly.get((y, w), {}))
        coverage = weeks_with_data / len(week_order) * 100
        print(f"  {name:20s}  {info['first_date']} — {info['last_date']}  {info['count']:>5} daily  {weeks_with_data:>4}/{len(week_order)} weeks  ({coverage:.0f}%)")

    return csv_path


if __name__ == '__main__':
    path = build_weekly_nav()
    print(f"\nDone! Weekly NAV at: {path}")
