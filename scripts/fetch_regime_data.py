#!/usr/bin/env python3
"""
T35: Regime Classifier Data Pipeline

Fetches Tushare data for market regime classification:
  - index_daily: CSI300 daily close/vol/amount
  - cn_m: M1/M2 money supply
  - cn_cpi: CPI
  - stk_limit: Up/down limit percentages

Output: data/tushare/regime_signals.csv
  Columns: week, csi300_close, m1_yoy, m2_yoy, m1m2_gap, up_pct_4w, down_pct_4w, cpi_3m_avg

Usage:
    python scripts/fetch_regime_data.py
"""

import urllib.request, json, csv, re, sys, time
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# ── Configuration ────────────────────────────────────────────────────────────

API_BASE = 'http://8.148.76.181:8686/'

# Extract token from build_pipeline_v2.py (same pattern as fetch_constituent_data.py)
BP = Path(__file__).resolve().parent.parent / 'scripts' / 'build_pipeline_v2.py'
if not BP.exists():
    BP = Path('/home/ubuntu/claw_etf_strategy/scripts/build_pipeline_v2.py')
with open(BP) as f:
    src = f.read()
m = re.search(r'TOKEN\s*=\s*"([^"]{20,})"', src)
if not m:
    print("ERROR: Token not found in build_pipeline_v2.py")
    sys.exit(1)
TOKEN = m.group(1)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'data' / 'tushare'
if not OUTPUT_DIR.exists():
    OUTPUT_DIR = Path('/home/ubuntu/claw_etf_strategy/data/tushare')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── API Helpers ──────────────────────────────────────────────────────────────

def post_json(url, data, timeout=60):
    """POST to Tushare API."""
    body = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(url, data=body,
        headers={'Content-Type': 'application/json'})
    resp = urllib.request.urlopen(req, timeout=timeout)
    result = json.loads(resp.read().decode('utf-8'))
    if result.get('code') != 0:
        print(f"  API error: code={result.get('code')}, msg={result.get('msg', '')[:100]}")
        return None
    return result

def parse_data(result):
    """Parse Tushare response data into list-of-dicts."""
    if result is None:
        return []
    data = result.get('data', [])
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        fields = data.get('fields', [])
        items = data.get('items', [])
        return [dict(zip(fields, item)) for item in items]
    return []

# ── Step 1: Fetch CSI300 Daily Index Data ───────────────────────────────────

def fetch_index_daily(ts_code: str, label: str) -> list:
    """Fetch daily index data for a given ts_code. Returns list of dicts."""
    print(f"  Fetching {label} (ts_code={ts_code})...", end=' ', flush=True)
    result = post_json(API_BASE + 'api', {
        'api_name': 'index_daily',
        'token': TOKEN,
        'params': {'ts_code': ts_code, 'start_date': '20130101', 'end_date': '20260615'},
        'fields': 'trade_date,close,vol,amount'
    })
    records = parse_data(result)
    if records:
        dates = sorted(set(r.get('trade_date', '') for r in records))
        print(f"{len(records)} records, {dates[0]}..{dates[-1]}")
        return records
    else:
        print(f"0 records")
        return []


def fetch_csi300_daily():
    """Fetch CSI300 daily close/vol/amount. Tries multiple codes. Returns list of dicts."""
    print("STEP 1: Fetch CSI300 index_daily")
    print("-" * 50)
    
    codes_to_try = ['000300.SH', '399300.SZ', '000016.SH', '000001.SH']
    
    for code in codes_to_try[:3]:  # First 3 are CSI300 variants
        records = fetch_index_daily(code, f"CSI300 ({code})")
        if records:
            return records
    
    # Fallback: SH Composite
    return fetch_index_daily('000001.SH', 'SH Composite fallback')


def fetch_sh_composite_daily():
    """Fetch SH Composite (000001.SH) daily data. Returns list of dicts."""
    print("\nSTEP 1b: Fetch SH Composite index_daily (000001.SH)")
    print("-" * 50)
    
    records = fetch_index_daily('000001.SH', 'SH Composite')
    if not records:
        print("  WARNING: No SH Composite data found.")
    return records

# ── Step 2: Fetch M1/M2 Money Supply ───────────────────────────────────────

def fetch_money_supply():
    """Fetch cn_m monthly M1/M2. Returns list of dicts."""
    print("\nSTEP 2: Fetch cn_m (M1/M2 money supply)")
    print("-" * 50)
    
    result = post_json(API_BASE + 'api', {
        'api_name': 'cn_m',
        'token': TOKEN,
        'params': {'m': 'm1,m2', 'start_m': '201301', 'end_m': '202606'},
        'fields': 'month,m1,m2'
    })
    records = parse_data(result)
    if records:
        months = sorted(set(r.get('month', '') for r in records))
        print(f"  {len(records)} records, {months[0]}..{months[-1]}")
        return records
    
    # Try alternative: no params filter
    print("  Retrying without m param...", end=' ', flush=True)
    result = post_json(API_BASE + 'api', {
        'api_name': 'cn_m',
        'token': TOKEN,
        'params': {'start_m': '201301', 'end_m': '202606'},
        'fields': 'month,m1,m2'
    })
    records = parse_data(result)
    if records:
        months = sorted(set(r.get('month', '') for r in records))
        print(f"{len(records)} records, {months[0]}..{months[-1]}")
        return records
    
    print("  WARNING: No M1/M2 data found.")
    return []

# ── Step 3: Fetch CPI ───────────────────────────────────────────────────────

def fetch_cpi():
    """Fetch cn_cpi monthly CPI. Returns list of dicts.
    
    CPI data format (list of dicts):
      {month, nt_val, nt_yoy, nt_mom, nt_accu, town_val, town_yoy, ...}
      nt_yoy = national CPI year-over-year (%)
    """
    print("\nSTEP 3: Fetch cn_cpi (CPI)")
    print("-" * 50)
    
    result = post_json(API_BASE + 'api', {
        'api_name': 'cn_cpi',
        'token': TOKEN,
        'params': {'start_m': '201301', 'end_m': '202606'},
        'fields': ''
    })
    records = parse_data(result)
    if records:
        months = sorted(set(r.get('month', '') for r in records))
        print(f"  {len(records)} records, {months[0]}..{months[-1]}")
        # Show sample
        sample = records[0]
        cpi_yoy_field = 'nt_yoy' if 'nt_yoy' in sample else 'town_yoy' if 'town_yoy' in sample else None
        if cpi_yoy_field:
            print(f"  CPI field used: {cpi_yoy_field} = {sample.get(cpi_yoy_field)}")
        return records
    
    print("  WARNING: No CPI data found.")
    return []

# ── Step 4: Fetch stk_limit (up/down limit percentages) ─────────────────────

def fetch_stk_limit():
    """Fetch stk_limit daily limit-up/limit-down percentages.
    
    This API returns per-stock rows. We'll fetch date range data
    (using start_date/end_date) and aggregate in processing.
    """
    print("\nSTEP 4: Fetch stk_limit (up/down limit percentages)")
    print("-" * 50)
    
    result = post_json(API_BASE + 'api', {
        'api_name': 'stk_limit',
        'token': TOKEN,
        'params': {'start_date': '20130101', 'end_date': '20260615'},
        'fields': 'trade_date,up_limit,down_limit'
    })
    records = parse_data(result)
    if records:
        # Count unique dates
        dates = set(r.get('trade_date', '') for r in records if r.get('trade_date'))
        print(f"  {len(records)} stock-rows across {len(dates)} dates")
        if dates:
            sorted_dates = sorted(dates)
            print(f"    Range: {sorted_dates[0]}..{sorted_dates[-1]}")
        return records
    
    print("  WARNING: No stk_limit data found.")
    return []

# ── Step 5: Process into weekly regime signals ──────────────────────────────

def process_to_weekly_signals(csi300_data, sh_data, money_data, cpi_data, stk_limit_data):
    """
    Process raw daily/monthly data into weekly regime signals CSV.

    Output columns:
      week, csi300_close, sh_close, m1_yoy, m2_yoy, m1m2_gap,
      up_pct_4w, down_pct_4w, cpi_3m_avg, breadth_score

    breadth_score = z-score of CSI300/SH Composite 4-week avg ratio over 52-week rolling window.
    Positive = broad participation (small/mid caps keeping pace with large caps).
    Negative = narrow leadership (large caps pulling away).
    """
    print("\nSTEP 5: Process data into weekly regime signals")
    print("-" * 50)
    
    # ── Build CSI300 daily dict: date_str -> close ──
    csi300_daily = {}
    for r in csi300_data:
        td = r.get('trade_date', '')
        close = r.get('close')
        if td and close is not None:
            try:
                csi300_daily[td] = float(close)
            except (ValueError, TypeError):
                pass
    
    print(f"  CSI300: {len(csi300_daily)} daily records")

    # ── Build SH Composite daily dict: date_str -> close ──
    sh_daily = {}
    for r in sh_data:
        td = r.get('trade_date', '')
        close = r.get('close')
        if td and close is not None:
            try:
                sh_daily[td] = float(close)
            except (ValueError, TypeError):
                pass

    print(f"  SH Composite: {len(sh_daily)} daily records")
    
    # ── Build M1/M2 monthly dict: month_str -> (m1, m2) ──
    money_monthly = {}
    for r in money_data:
        month = r.get('month', '')
        m1 = r.get('m1')
        m2 = r.get('m2')
        if month and m1 is not None and m2 is not None:
            try:
                money_monthly[month] = (float(m1), float(m2))
            except (ValueError, TypeError):
                pass
    
    print(f"  M1/M2: {len(money_monthly)} monthly records")
    
    # ── Build CPI monthly dict: month_str -> cpi_yoy ──
    # CPI data: list of dicts with fields like nt_yoy (national CPI YoY %)
    cpi_monthly = {}
    cpi_field = None
    if cpi_data:
        sample = cpi_data[0]
        if 'nt_yoy' in sample:
            cpi_field = 'nt_yoy'
        elif 'town_yoy' in sample:
            cpi_field = 'town_yoy'
        elif 'cnt_yoy' in sample:
            cpi_field = 'cnt_yoy'
    
    for r in cpi_data:
        month = r.get('month', '')
        if month and cpi_field:
            cpi_val = r.get(cpi_field)
            if cpi_val is not None:
                try:
                    cpi_monthly[month] = float(cpi_val)
                except (ValueError, TypeError):
                    pass
    
    print(f"  CPI: {len(cpi_monthly)} monthly records (field={cpi_field})")
    
    # ── Build stk_limit daily dict: date_str -> (avg_up_limit, avg_down_limit) ──
    # stk_limit data is per-stock rows; aggregate per date by averaging
    stk_by_date = defaultdict(lambda: {'up': [], 'down': []})
    for r in stk_limit_data:
        td = r.get('trade_date', '')
        up_l = r.get('up_limit')
        down_l = r.get('down_limit')
        if td and up_l is not None and down_l is not None:
            try:
                stk_by_date[td]['up'].append(float(up_l))
                stk_by_date[td]['down'].append(float(down_l))
            except (ValueError, TypeError):
                pass
    
    stk_daily = {}
    for td, vals in stk_by_date.items():
        if vals['up'] and vals['down']:
            avg_up = sum(vals['up']) / len(vals['up'])
            avg_down = sum(vals['down']) / len(vals['down'])
            stk_daily[td] = (round(avg_up, 4), round(avg_down, 4))
    
    print(f"  stk_limit: {len(stk_daily)} dates (aggregated from {len(stk_limit_data)} stock-rows)")
    
    # ── Generate weekly grid (Mondays) ──
    # Match backtest date range: 2013-05-20 to 2026-05-01
    all_dates = sorted(set(list(csi300_daily.keys())))
    if not all_dates:
        print("  ERROR: No CSI300 data available!")
        return []
    
    first_date = datetime.strptime(all_dates[0], '%Y%m%d')
    last_date = datetime.strptime(all_dates[-1], '%Y%m%d')
    
    # Find first Monday on or after first_date - go back to 2013-05-20 as anchor
    anchor = datetime(2013, 5, 20)  # W-MON from backtest config
    if anchor < first_date:
        anchor = first_date
    
    # Generate Mondays
    weeks = []
    d = anchor
    while d <= last_date:
        weeks.append(d.strftime('%Y%m%d'))
        d += timedelta(days=7)
    
    print(f"  Weekly grid: {len(weeks)} Mondays ({weeks[0]}..{weeks[-1]})")
    
    # ── For each week, compute signals ──
    rows = []
    for i, week_str in enumerate(weeks):
        week_dt = datetime.strptime(week_str, '%Y%m%d')
        
        # --- CSI300 close: find closest trading day <= week_dt ---
        csi300_close = None
        sh_close = None
        for d_str in reversed(all_dates):
            if d_str <= week_str:
                csi300_close = csi300_daily.get(d_str)
                sh_close = sh_daily.get(d_str)
                break

        # --- Breadth ratio: CSI300 / SH Composite 4-week avg ---
        # Compute 4-week rolling average of the ratio (CSI300/SH)
        breadth_ratio = None
        ratio_vals = []
        ratio_cutoff = week_dt - timedelta(days=28)
        for d_str in sorted(csi300_daily.keys()):
            d_dt = datetime.strptime(d_str, '%Y%m%d')
            if ratio_cutoff <= d_dt <= week_dt:
                c = csi300_daily.get(d_str)
                s = sh_daily.get(d_str)
                if c is not None and s is not None and s > 0:
                    ratio_vals.append(c / s)
        if ratio_vals:
            breadth_ratio = sum(ratio_vals) / len(ratio_vals)
        
        # --- M1/M2 YoY: find latest month <= week ---
        month_str = week_dt.strftime('%Y%m')
        m1_yoy = None
        m2_yoy = None
        m1m2_gap = None
        
        # Find latest available month
        latest_month = None
        for m_str in sorted(money_monthly.keys(), reverse=True):
            if m_str <= month_str:
                latest_month = m_str
                break
        
        if latest_month and latest_month in money_monthly:
            m1_val, m2_val = money_monthly[latest_month]
            # Find 12 months ago
            m_dt = datetime.strptime(latest_month, '%Y%m')
            m_prev = datetime(m_dt.year - 1, m_dt.month, 1).strftime('%Y%m')
            
            # Find closest available month <= m_prev
            best_prev = None
            for m_str in sorted(money_monthly.keys(), reverse=True):
                if m_str <= m_prev:
                    best_prev = m_str
                    break
            
            if best_prev and best_prev in money_monthly:
                m1_prev, m2_prev = money_monthly[best_prev]
                if m1_prev > 0:
                    m1_yoy = round(m1_val / m1_prev - 1, 6)
                if m2_prev > 0:
                    m2_yoy = round(m2_val / m2_prev - 1, 6)
                if m1_yoy is not None and m2_yoy is not None:
                    m1m2_gap = round(m1_yoy - m2_yoy, 6)
        
        # --- CPI 3-month average ---
        cpi_3m_avg = None
        cpi_vals = []
        # Look back up to 3 months
        for offset in range(3):
            m_dt = datetime(week_dt.year, week_dt.month, 1)
            # subtract months
            target_month = m_dt.month - offset
            target_year = m_dt.year
            while target_month <= 0:
                target_month += 12
                target_year -= 1
            target_str = f"{target_year}{target_month:02d}"
            
            best = None
            for m_str in sorted(cpi_monthly.keys(), reverse=True):
                if m_str <= target_str:
                    best = m_str
                    break
            if best and best in cpi_monthly:
                cpi_vals.append(cpi_monthly[best])
        
        if cpi_vals:
            cpi_3m_avg = round(sum(cpi_vals) / len(cpi_vals), 4)
        
        # --- Up/Down limit 4-week average ---
        up_pct_4w = None
        down_pct_4w = None
        
        # Get last 4 weeks of daily stk_limit data
        up_vals = []
        down_vals = []
        cutoff = week_dt - timedelta(days=28)
        for d_str in sorted(stk_daily.keys()):
            d_dt = datetime.strptime(d_str, '%Y%m%d')
            if cutoff <= d_dt <= week_dt and d_str in stk_daily:
                u, d = stk_daily[d_str]
                up_vals.append(u)
                down_vals.append(d)
        
        if up_vals:
            up_pct_4w = round(sum(up_vals) / len(up_vals), 4)
        if down_vals:
            down_pct_4w = round(sum(down_vals) / len(down_vals), 4)
        
        rows.append([
            week_str,
            csi300_close if csi300_close is not None else '',
            sh_close if sh_close is not None else '',
            m1_yoy if m1_yoy is not None else '',
            m2_yoy if m2_yoy is not None else '',
            m1m2_gap if m1m2_gap is not None else '',
            up_pct_4w if up_pct_4w is not None else '',
            down_pct_4w if down_pct_4w is not None else '',
            cpi_3m_avg if cpi_3m_avg is not None else '',
            breadth_ratio if breadth_ratio is not None else '',  # col 9, temporary
        ])

    # ── Compute breadth_score: 52-week rolling z-score of breadth_ratio ──
    # Then REPLACE the temporary breadth_ratio with breadth_score
    breadth_ratios = []
    for row in rows:
        val = row[9]  # breadth_ratio column
        breadth_ratios.append(float(val) if val != '' else None)

    # Compute rolling 52-week z-scores
    for i, br in enumerate(breadth_ratios):
        if br is None:
            rows[i][9] = ''  # replace with empty breadth_score
            continue
        # Look back up to 52 weeks
        window_start = max(0, i - 51)
        window_ratios = [breadth_ratios[j] for j in range(window_start, i + 1) if breadth_ratios[j] is not None]
        if len(window_ratios) >= 10:  # need at least 10 weeks of data
            mean = sum(window_ratios) / len(window_ratios)
            variance = sum((r - mean) ** 2 for r in window_ratios) / len(window_ratios)
            std = variance ** 0.5
            if std > 0:
                # INVERT: High CSI300/SH ratio = narrow breadth (only large caps)
                # Low ratio = wide breadth (broad participation)
                # Negate so positive z-score = WIDE, negative = THIN
                z = round(-(br - mean) / std, 4)
                rows[i][9] = z  # REPLACE breadth_ratio with breadth_score
            else:
                rows[i][9] = ''
        else:
            rows[i][9] = ''
    
    print(f"  Generated {len(rows)} weekly rows")
    return rows

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("T35: REGIME CLASSIFIER DATA PIPELINE")
    print("=" * 70)
    print(f"API: {API_BASE}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Fetch data
    csi300_data = fetch_csi300_daily()
    sh_data = fetch_sh_composite_daily()
    money_data = fetch_money_supply()
    cpi_data = fetch_cpi()
    stk_limit_data = fetch_stk_limit()

    # Process into weekly signals
    rows = process_to_weekly_signals(csi300_data, sh_data, money_data, cpi_data, stk_limit_data)
    
    if not rows:
        print("\nERROR: No weekly signals generated.")
        sys.exit(1)
    
    # Save CSV
    signals_path = OUTPUT_DIR / 'regime_signals.csv'
    with open(signals_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['week', 'csi300_close', 'sh_close', 'm1_yoy', 'm2_yoy', 'm1m2_gap',
                          'up_pct_4w', 'down_pct_4w', 'cpi_3m_avg', 'breadth_score'])
        writer.writerows(rows)

    print(f"\n✅ Saved regime signals: {signals_path}")
    print(f"   {len(rows)} weekly records")

    # Print data coverage summary
    valid_close = sum(1 for r in rows if r[1] != '')
    valid_sh = sum(1 for r in rows if r[2] != '')
    valid_m1 = sum(1 for r in rows if r[3] != '')
    valid_m2 = sum(1 for r in rows if r[4] != '')
    valid_up = sum(1 for r in rows if r[6] != '')
    valid_cpi = sum(1 for r in rows if r[8] != '')
    valid_breadth = sum(1 for r in rows if r[9] != '')

    print(f"\n  Data coverage:")
    print(f"    CSI300 close:  {valid_close}/{len(rows)} weeks")
    print(f"    SH Composite:   {valid_sh}/{len(rows)} weeks")
    print(f"    M1 YoY:        {valid_m1}/{len(rows)} weeks")
    print(f"    M2 YoY:        {valid_m2}/{len(rows)} weeks")
    print(f"    Up pct 4w:     {valid_up}/{len(rows)} weeks")
    print(f"    CPI 3m avg:    {valid_cpi}/{len(rows)} weeks")
    print(f"    Breadth score:  {valid_breadth}/{len(rows)} weeks")
    
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
