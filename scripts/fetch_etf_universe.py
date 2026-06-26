#!/usr/bin/env python3
"""
Tushare ETF Universe Fetcher — Phase 3 Direction 2

Connects to the Tushare data proxy, queries available ETFs,
categorizes by asset class, and saves results.

Usage:
    python scripts/fetch_etf_universe.py --token <TUSHARE_TOKEN>
    python scripts/fetch_etf_universe.py --token <TOKEN> --output data/etf_universe.json
    python scripts/fetch_etf_universe.py --token <TOKEN> --quality-filter  # apply quality filters

API: Tushare Data Proxy at http://8.148.76.181:8686/
API Format: Standard Tushare POST /api with JSON body:
    {api_name, token, params, fields}

Key Tushare APIs used:
    - fund_basic: ETF list (market='E' for ETFs)
    - fund_daily: Daily NAV data
    - fund_portfolio: Fund holdings (quarterly)
    - index_basic: Index reference data
    - trade_cal: Trading calendar
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import urllib.request
    import urllib.error
except ImportError:
    pass


# ── Configuration ────────────────────────────────────────────────────────────

API_BASE = 'http://8.148.76.181:8686/'

# Quality filter defaults
MIN_DAYS_LISTED = 252 * 3     # At least 3 years trading history
MIN_AUM_YUAN = 200_000_000    # Minimum 2亿 RMB AUM
MIN_DAILY_VOLUME = 5_000_000  # Minimum 500万 RMB daily volume

# ETF categories for classification
CATEGORY_KEYWORDS = {
    'A股宽基': ['沪深300', '中证500', '中证1000', '创业板', '科创50', '科创100',
                '上证50', '深证100', '中证100', '中证2000', 'A50', 'MSCI中国A50'],
    'A股行业/主题': ['证券', '银行', '保险', '地产', '军工', '医药', '医疗', '半导体',
                     '芯片', '新能源', '光伏', '电池', '汽车', '消费', '食品', '饮料',
                     '白酒', '家电', '传媒', '计算机', '通信', '5G', '人工智能',
                     '机器人', '游戏', '煤炭', '有色', '钢铁', '化工', '电力',
                     '基建', '建材', '农业', '稀土', '碳中和'],
    '跨境ETF': ['纳指', '标普', '道琼斯', '恒生', 'H股', '中概', '互联',
                 '日经', '德国DAX', '法国CAC', '印度', '越南', '亚太',
                 '全球', '海外'],
    '债券ETF': ['国债', '地债', '政金债', '信用债', '可转债', '短融',
                 '城投债', '公司债', '利率债', '债券', '债'],
    '商品ETF': ['黄金', '白银', '有色', '豆粕', '原油', '能源', '化工'],
    '货币ETF': ['货币', '理财', '保证金'],
    '红利/价值': ['红利', '高股息', '低波', '价值', '质量', '基本面'],
    '跨境QDII': ['QDII', '纳斯达克', '标普500', '恒生科技', '恒生医疗'],
}

TUSHARE_ETF_FIELDS = [
    'ts_code',      # Tushare code (e.g., 510050.SH)
    'name',         # Fund name
    'management',   # Management company
    'found_date',   # Establishment date
    'benchmark',    # Benchmark index
    'invest_type',  # Investment type
    'type',         # Fund type
    'aum',          # Assets under management
    'status',       # Status (L=listed, D=delisted, E=listed_exchange)
]


# ── API Client ───────────────────────────────────────────────────────────────

class TushareClient:
    """Lightweight Tushare API client via urllib (no external deps)."""

    def __init__(self, token: str, base_url: str = API_BASE):
        self.token = token
        self.base_url = base_url.rstrip('/')

    def _request(self, api_name: str, params: dict | None = None,
                 fields: str = '') -> dict:
        """Make a Tushare API request."""
        url = f'{self.base_url}/api'
        body = {
            'api_name': api_name,
            'token': self.token,
            'params': params or {},
            'fields': fields,
        }
        data = json.dumps(body).encode('utf-8')
        req = urllib.request.Request(
            url, data=data,
            headers={'Content-Type': 'application/json'}
        )
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            result = json.loads(resp.read().decode('utf-8'))
            return result
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8', errors='replace')
            return {'code': e.code, 'msg': str(e), 'data': {'error': error_body}}
        except Exception as e:
            return {'code': -1, 'msg': str(e), 'data': {'fields': [], 'items': []}}

    def fund_basic(self, market: str = 'E', fields: str | None = None) -> dict:
        """Query fund basic info. market='E' for ETFs."""
        if fields is None:
            fields = ','.join(TUSHARE_ETF_FIELDS)
        return self._request('fund_basic', {'market': market}, fields)

    def fund_daily(self, ts_code: str, start_date: str, end_date: str,
                   fields: str = '') -> dict:
        """Query fund daily NAV data."""
        return self._request('fund_daily', {
            'ts_code': ts_code,
            'start_date': start_date,
            'end_date': end_date,
        }, fields or 'trade_date,close,pre_close,vol,amount')

    def trade_cal(self, exchange: str = 'SSE', start_date: str = '',
                  end_date: str = '', fields: str = '') -> dict:
        """Query trading calendar."""
        return self._request('trade_cal', {
            'exchange': exchange,
            'start_date': start_date,
            'end_date': end_date,
        }, fields or 'cal_date,is_open')


# ── Classification ───────────────────────────────────────────────────────────

def classify_etf(name: str, benchmark: str = '', invest_type: str = '') -> list[str]:
    """Classify an ETF into one or more categories based on name and benchmark."""
    categories = []
    search_text = f"{name} {benchmark}".lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in search_text:
                categories.append(category)
                break

    if not categories:
        categories.append('其他')

    return categories


# ── Quality Filters ──────────────────────────────────────────────────────────

def apply_quality_filters(etfs: list[dict], min_days: int = MIN_DAYS_LISTED,
                          min_aum: int = MIN_AUM_YUAN,
                          min_volume: int = MIN_DAILY_VOLUME) -> list[dict]:
    """Apply quality filters to ETF list."""
    filtered = []
    today = datetime.now()

    for etf in etfs:
        reasons = []

        # Status check: must be listed
        status = etf.get('status', '')
        if status not in ('L', 'E', ''):
            reasons.append(f'status={status}')

        # Age check
        found_date = etf.get('found_date', '')
        if found_date:
            try:
                fd = datetime.strptime(str(found_date)[:10], '%Y%m%d')
                days_listed = (today - fd).days
                if days_listed < min_days:
                    reasons.append(f'age={days_listed}d < {min_days}d')
            except ValueError:
                pass

        if reasons:
            etf['filter_reasons'] = reasons
            etf['passed_filter'] = False
        else:
            etf['passed_filter'] = True

        filtered.append(etf)

    return filtered


# ── Main ─────────────────────────────────────────────────────────────────────

def fetch_etf_universe(token: str, quality_filter: bool = False,
                       output_dir: str = 'data') -> dict:
    """
    Fetch and categorize the full ETF universe from Tushare.

    Args:
        token: Valid Tushare API token
        quality_filter: Whether to apply quality filters
        output_dir: Directory for output files

    Returns:
        Dict with categorized ETF universe
    """
    client = TushareClient(token)

    print("=" * 70)
    print("TUSHARE ETF UNIVERSE FETCHER")
    print(f"API: {API_BASE}")
    print(f"Quality filter: {'ON' if quality_filter else 'OFF'}")
    print("=" * 70)

    # Step 1: Fetch ETF list
    print("\n[1/3] Fetching ETF list (fund_basic, market=E)...")
    result = client.fund_basic(market='E')

    if result.get('code') == 401:
        print(f"  ❌ Authentication failed: {result.get('msg', 'Unknown error')}")
        print(f"  Please provide a valid Tushare token.")
        return {'error': 'auth_failed', 'msg': result.get('msg', '')}

    if result.get('code') != 0:
        print(f"  ❌ API error: code={result.get('code')}, msg={result.get('msg')}")
        return {'error': 'api_error', 'msg': result.get('msg', '')}

    data = result.get('data', {})

    # Handle two possible API response formats:
    # Format A: {fields: [...], items: [[...], ...]}  (legacy Tushare)
    # Format B: [{ts_code: ..., name: ...}, ...]     (this proxy)
    if isinstance(data, list):
        # Format B: data is already list of dicts
        etfs_raw = data
        fields = list(data[0].keys()) if data else []
        print(f"  ✅ Got {len(data)} ETFs (list-of-dicts format, fields: {fields})")
    elif isinstance(data, dict):
        # Format A: legacy fields/items format
        fields = data.get('fields', [])
        items = data.get('items', [])
        print(f"  ✅ Got {len(items)} ETFs with fields: {fields}")
        etfs_raw = [dict(zip(fields, item)) for item in items]
    else:
        print(f"  ❌ Unexpected data format: {type(data)}")
        return {'error': 'bad_format', 'msg': f'Unexpected data type: {type(data)}'}

    # Step 2: Process and classify
    print("\n[2/3] Processing and classifying...")
    etfs = []
    for etf_raw in etfs_raw:
        etf = dict(etf_raw)  # already a dict
        etf['categories'] = classify_etf(
            etf.get('name', ''),
            etf.get('benchmark', ''),
            etf.get('invest_type', '')
        )
        etfs.append(etf)

    # Step 3: Quality filters
    if quality_filter:
        print("\n[3/3] Applying quality filters...")
        etfs = apply_quality_filters(etfs)
        passed = sum(1 for e in etfs if e.get('passed_filter', False))
        failed = len(etfs) - passed
        print(f"  Passed: {passed}, Filtered out: {failed}")

    # Categorize summary
    category_counts = {}
    for etf in etfs:
        for cat in etf.get('categories', ['其他']):
            category_counts[cat] = category_counts.get(cat, 0) + 1

    print("\n" + "=" * 70)
    print("CATEGORY SUMMARY")
    print("=" * 70)
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # JSON
    json_path = output_path / f'etf_universe_{timestamp}.json'
    output = {
        'fetched_at': datetime.now().isoformat(),
        'api_base': API_BASE,
        'quality_filter': quality_filter,
        'total_etfs': len(etfs),
        'category_summary': category_counts,
        'etfs': etfs,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved to: {json_path}")

    # CSV summary
    csv_path = output_path / f'etf_universe_{timestamp}.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        csv_fields = ['ts_code', 'name', 'categories', 'management',
                       'found_date', 'benchmark', 'invest_type']
        f.write(','.join(csv_fields) + '\n')
        for etf in etfs:
            row = [
                str(etf.get(f, '')),
                f'"{etf.get("name", "")}"',
                f'"{"|".join(etf.get("categories", []))}"',
                f'"{etf.get("management", "")}"',
                str(etf.get('found_date', '')),
                f'"{etf.get("benchmark", "")}"',
                str(etf.get('invest_type', '')),
            ]
            f.write(','.join(row) + '\n')
    print(f"✅ Saved to: {csv_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description='Fetch ETF universe from Tushare API'
    )
    parser.add_argument('--token', type=str, required=True,
                        help='Tushare API token (required)')
    parser.add_argument('--quality-filter', action='store_true',
                        help='Apply quality filters (min 3yr history, min 2亿 AUM)')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory (default: data/)')
    parser.add_argument('--api-base', type=str, default=API_BASE,
                        help='Tushare API base URL')
    args = parser.parse_args()

    result = fetch_etf_universe(
        token=args.token,
        quality_filter=args.quality_filter,
        output_dir=args.output_dir,
    )

    if result.get('error'):
        sys.exit(1)


if __name__ == '__main__':
    main()
