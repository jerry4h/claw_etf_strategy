#!/usr/bin/env python3
"""
T31: Constituent-Stock Signals Data Retrieval Pipeline

Fetches Tushare fund_portfolio + stock daily data to compute
constituent-derived signals (CWM, CONC) for the 5-ETF universe.

Extends the T27 pipeline.

Usage:
    python scripts/fetch_constituent_data.py

Output:
    data/tushare/constituent_portfolio.csv     — raw portfolio holdings
    data/tushare/constituent_signals.csv       — derived weekly signals
"""

import urllib.request, json, csv, re, sys, time
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# ── Configuration ────────────────────────────────────────────────────────────

API_BASE = 'http://8.148.76.181:8686/'

# Extract token from build_pipeline_v2.py
BP = Path(__file__).resolve().parent.parent / 'scripts' / 'build_pipeline_v2.py'
# Fallback: try relative to project root
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

# Baseline 5-ETF Tushare codes
BASELINE_ETFS = {
    '纳指ETF':      {'ts_code': '513100.SH', 'type': 'QDII',       'has_portfolio': False},
    '红利低波ETF':   {'ts_code': '512890.SH', 'type': 'A-share',    'has_portfolio': True},
    '沪深300ETF':    {'ts_code': '510300.SH', 'type': 'A-share',    'has_portfolio': True},
    '黄金ETF':       {'ts_code': '518880.SH', 'type': 'commodity',  'has_portfolio': False},
    '国债ETF':       {'ts_code': '511010.SH', 'type': 'bond',       'has_portfolio': False},
}

# ── API Helpers ──────────────────────────────────────────────────────────────

def post_json(url, data, timeout=30):
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

# ── Step 1: Fetch Fund Portfolio ─────────────────────────────────────────────

def fetch_portfolio(ts_code, label):
    """Fetch fund_portfolio for a single ETF. Returns list of holding records."""
    print(f"  [{label}] {ts_code}: ", end='', flush=True)
    result = post_json(API_BASE + 'api', {
        'api_name': 'fund_portfolio',
        'token': TOKEN,
        'params': {'ts_code': ts_code},
        'fields': ''
    })
    records = parse_data(result)
    if records:
        dates = sorted(set(r.get('end_date', '') for r in records))
        print(f"{len(records)} records, {len(dates)} periods ({dates[0]}..{dates[-1]})")
    else:
        print(f"0 records (no data)")
    return records

# ── Step 2: Fetch Stock Daily Prices ──────────────────────────────────────────

def fetch_stock_daily(symbols, start='20130101', end='20260616'):
    """Fetch daily close prices for a list of stock symbols.
    Returns dict: symbol -> {date: close_price}
    """
    print(f"  Fetching daily prices for {len(symbols)} stocks...", flush=True)
    prices = defaultdict(dict)
    
    for i, symbol in enumerate(symbols):
        if i % 50 == 0 and i > 0:
            print(f"    ... {i}/{len(symbols)}", flush=True)
        try:
            result = post_json(API_BASE + 'api', {
                'api_name': 'daily',
                'token': TOKEN,
                'params': {'ts_code': symbol, 'start_date': start, 'end_date': end},
                'fields': 'trade_date,close'
            }, timeout=30)
            records = parse_data(result)
            for r in records:
                prices[symbol][r.get('trade_date', '')] = float(r.get('close', 0) or 0)
            time.sleep(0.02)  # rate limit
        except Exception as e:
            pass  # skip stocks that fail
    
    print(f"    Done: {len(prices)} stocks with data")
    return dict(prices)

# ── Step 3: Compute Signals ───────────────────────────────────────────────────

def compute_cwm(etf_name, portfolio_records, stock_prices, window=12):
    """Compute Constituent Weighted Momentum.
    
    CWM(t) = Σ w_i(t) * (price_i(t) / price_i(t-window) - 1)
    
    For each quarter-end, compute momentum of top-10 holdings over window weeks.
    Between quarters, carry forward the previous quarter's CWM.
    """
    if not portfolio_records:
        return {}  # no data
    
    # Group by end_date
    periods = defaultdict(list)
    for r in portfolio_records:
        periods[r['end_date']].append(r)
    
    result = {}
    prev_cwm = 0.0
    
    for end_date in sorted(periods.keys()):
        holdings = periods[end_date]
        # Sort by weight descending
        sorted_holdings = sorted(holdings, 
            key=lambda x: float(x.get('stk_mkv_ratio', 0) or 0), reverse=True)
        top10 = sorted_holdings[:10]
        
        total_weight = sum(float(h.get('stk_mkv_ratio', 0) or 0) for h in top10)
        if total_weight == 0:
            result[end_date] = 0.0
            prev_cwm = 0.0
            continue
        
        # Compute weighted momentum
        weighted_mom = 0.0
        weight_sum = 0.0
        
        for h in top10:
            symbol = h.get('symbol', '')
            weight = float(h.get('stk_mkv_ratio', 0) or 0) / total_weight
            
            if symbol in stock_prices:
                prices = stock_prices[symbol]
                # Find price at end_date and price window weeks earlier
                p_now = prices.get(end_date)
                # Approximate window weeks ago by subtracting window*7 days
                dt_end = datetime.strptime(end_date, '%Y%m%d')
                dt_past = dt_end - timedelta(weeks=window)
                # Find closest trading day
                past_date = None
                for d_str in sorted(prices.keys(), reverse=True):
                    d = datetime.strptime(d_str, '%Y%m%d')
                    if d <= dt_past:
                        past_date = d_str
                        break
                
                p_past = prices.get(past_date) if past_date else None
                if p_now and p_past and p_past > 0:
                    mom = (p_now / p_past) - 1.0
                    weighted_mom += weight * mom
                    weight_sum += weight
        
        if weight_sum > 0:
            cwm = weighted_mom / weight_sum
        else:
            cwm = 0.0
        
        result[end_date] = cwm
        prev_cwm = cwm
    
    return result


def compute_conc(etf_name, portfolio_records):
    """Compute CONCentration change signal.
    
    CONC(t) = -(HHI(t) - HHI(t-4)) / HHI(t-4)  (YoY change)
    
    HHI(t) = sum(weight_i/100)^2 for top holdings at quarter-end t
    """
    if not portfolio_records:
        return {}
    
    # Group by end_date
    periods = defaultdict(list)
    for r in portfolio_records:
        periods[r['end_date']].append(r)
    
    sorted_dates = sorted(periods.keys())
    
    # Compute HHI for each period
    hhi_values = {}
    for end_date in sorted_dates:
        holdings = periods[end_date]
        ratios = [float(h.get('stk_mkv_ratio', 0) or 0) / 100.0 for h in holdings]
        hhi = sum(r * r for r in ratios if r > 0)
        hhi_values[end_date] = hhi
    
    # Compute CONC (YoY change)
    result = {}
    for i, end_date in enumerate(sorted_dates):
        # Find 4 quarters ago
        dt = datetime.strptime(end_date, '%Y%m%d')
        target_month = dt.month
        target_year = dt.year - 1
        
        # Find closest date ~4 quarters ago
        prev_date = None
        for d in reversed(sorted_dates[:i]):
            d_dt = datetime.strptime(d, '%Y%m%d')
            if d_dt.year == target_year and abs(d_dt.month - target_month) <= 1:
                prev_date = d
                break
        if not prev_date and i >= 4:
            prev_date = sorted_dates[i-4]  # fallback
        
        if prev_date and hhi_values.get(prev_date, 0) > 0:
            conc = -(hhi_values[end_date] - hhi_values[prev_date]) / hhi_values[prev_date]
        else:
            conc = 0.0
        
        result[end_date] = conc
    
    return result

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("T31: CONSTITUENT-STOCK SIGNALS DATA PIPELINE")
    print("=" * 70)
    print(f"API: {API_BASE}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Step 1: Fetch portfolio data
    print("STEP 1: Fetch fund_portfolio for 5 baseline ETFs")
    print("-" * 50)
    
    all_portfolio = {}
    all_stock_symbols = set()
    
    for etf_name, info in BASELINE_ETFS.items():
        ts_code = info['ts_code']
        records = fetch_portfolio(ts_code, etf_name)
        all_portfolio[etf_name] = {
            'ts_code': ts_code,
            'type': info['type'],
            'has_portfolio': info['has_portfolio'],
            'records': records,
        }
        
        if records:
            for r in records:
                all_stock_symbols.add(r.get('symbol', ''))
    
    # Save raw portfolio
    portfolio_path = OUTPUT_DIR / 'constituent_portfolio.csv'
    with open(portfolio_path, 'w', newline='', encoding='utf-8') as f:
        fields = ['etf_name', 'ts_code', 'ann_date', 'end_date', 'symbol', 
                   'mkv', 'amount', 'stk_mkv_ratio', 'stk_float_ratio']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for etf_name, info in all_portfolio.items():
            for r in info['records']:
                r['etf_name'] = etf_name
                writer.writerow(r)
    print(f"\n  ✅ Saved portfolio data: {portfolio_path}")
    print(f"     Unique stock symbols: {len(all_stock_symbols)}")
    
    # Step 2: Fetch stock daily prices
    print(f"\nSTEP 2: Fetch stock daily prices for {len(all_stock_symbols)} symbols")
    print("-" * 50)
    
    if all_stock_symbols:
        stock_prices = fetch_stock_daily(sorted(all_stock_symbols))
    else:
        stock_prices = {}
        print("  No stocks to fetch (no ETFs with portfolio data)")
    
    # Step 3: Compute signals
    print(f"\nSTEP 3: Compute derived signals (CWM, CONC)")
    print("-" * 50)
    
    all_signals = {}
    
    for etf_name, info in all_portfolio.items():
        records = info['records']
        
        if not records:
            print(f"  [{etf_name}] type={info['type']}: no portfolio data → signals = 0")
            all_signals[etf_name] = {'cwm': {}, 'conc': {}}
            continue
        
        print(f"  [{etf_name}] Computing CWM...", end=' ', flush=True)
        cwm = compute_cwm(etf_name, records, stock_prices)
        print(f"{len(cwm)} periods")
        
        print(f"  [{etf_name}] Computing CONC...", end=' ', flush=True)
        conc = compute_conc(etf_name, records)
        print(f"{len(conc)} periods")
        
        all_signals[etf_name] = {'cwm': cwm, 'conc': conc}
        
        # Print sample values
        if cwm:
            latest = sorted(cwm.keys())[-1]
            print(f"    Latest CWM ({latest}): {cwm[latest]:.4f}")
        if conc:
            latest = sorted(conc.keys())[-1]
            print(f"    Latest CONC ({latest}): {conc[latest]:.4f}")
    
    # Step 4: Save signals
    print(f"\nSTEP 4: Save derived signals")
    print("-" * 50)
    
    # Collect all dates
    all_dates = set()
    for etf_name, sigs in all_signals.items():
        all_dates.update(sigs['cwm'].keys())
        all_dates.update(sigs['conc'].keys())
    all_dates = sorted(all_dates)
    
    signals_path = OUTPUT_DIR / 'constituent_signals.csv'
    with open(signals_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['end_date', 'etf_name', 'cwm', 'conc'])
        
        for date in all_dates:
            for etf_name in BASELINE_ETFS:
                sigs = all_signals.get(etf_name, {})
                cwm_val = sigs.get('cwm', {}).get(date, 0.0)
                conc_val = sigs.get('conc', {}).get(date, 0.0)
                writer.writerow([date, etf_name, cwm_val, conc_val])
    
    print(f"  ✅ Saved signals: {signals_path}")
    print(f"     {len(all_dates)} dates × {len(BASELINE_ETFS)} ETFs")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    portfolios_with_data = sum(1 for v in all_portfolio.values() if v['records'])
    print(f"  ETFs with portfolio data: {portfolios_with_data}/{len(BASELINE_ETFS)}")
    print(f"  Unique constituent stocks: {len(all_stock_symbols)}")
    print(f"  Signal periods: {len(all_dates)}")
    print(f"\n  Output files:")
    print(f"    {portfolio_path}")
    print(f"    {signals_path}")
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
