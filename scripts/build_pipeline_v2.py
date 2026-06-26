#!/usr/bin/env python3
"""
Build Tushare production data files.
Selects 15 diversified ETF candidates, fetches NAV 2013-2026, builds weekly CSV.
"""
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

TOKEN = "6886937ca1f5f3a65fd59968588f26f9"
API_BASE = "http://8.148.76.181:8686/"
OUTPUT_DIR = Path("data/tushare")

# Hand-picked diversified expansion candidates (true ETFs, pre-2020, good liquidity)
EXPANSION_CANDIDATES = [
    # A-share broad — large cap value
    {"ts_code": "510050.SH", "name": "上证50ETF", "rationale": "大盘价值，低相关于沪深300"},
    # A-share broad — mid cap
    {"ts_code": "510500.SH", "name": "中证500ETF", "rationale": "中盘成长，补沪深300缺口"},
    # A-share broad — small cap
    {"ts_code": "512100.SH", "name": "中证1000ETF", "rationale": "小盘，最高波动补收益"},
    # A-share broad — ChiNext growth
    {"ts_code": "159915.SZ", "name": "创业板ETF", "rationale": "A股成长/科技，高弹性"},
    # A-share sector — healthcare
    {"ts_code": "159929.SZ", "name": "医药ETF", "rationale": "防御性行业，低相关于宽基"},
    # A-share sector — consumer staples
    {"ts_code": "159928.SZ", "name": "消费ETF", "rationale": "必选消费，稳定现金流"},
    # A-share sector — semiconductor
    {"ts_code": "512480.SH", "name": "半导体ETF", "rationale": "科技/成长，高弹性"},
    # A-share sector — securities/brokers
    {"ts_code": "512880.SH", "name": "证券ETF", "rationale": "牛市放大器，beta交易"},
    # A-share sector — defense/military
    {"ts_code": "512660.SH", "name": "军工ETF", "rationale": "政策驱动，低相关性"},
    # Cross-border — Hang Seng
    {"ts_code": "159920.SZ", "name": "恒生ETF", "rationale": "港股敞口，分散A股风险"},
    # Cross-border — S&P 500
    {"ts_code": "513500.SH", "name": "标普500ETF", "rationale": "美股大盘，与纳指互补"},
    # Cross-border — Nikkei 225
    {"ts_code": "513520.SH", "name": "日经ETF", "rationale": "日本敞口，亚太分散"},
    # Commodity — soybean meal futures
    {"ts_code": "159985.SZ", "name": "豆粕ETF", "rationale": "农产品商品，与黄金互补"},
    # Bond — convertible bond
    {"ts_code": "511380.SH", "name": "可转债ETF", "rationale": "固收+，股债双性"},
    # Dividend — SSE dividend
    {"ts_code": "510880.SH", "name": "红利ETF", "rationale": "高股息价值，补红利低波"},
]


def is_true_etf(ts_code):
    if ts_code.endswith('.SH'):
        code = ts_code.replace('.SH', '')
        return code.startswith('51') or code.startswith('56') or code.startswith('58')
    if ts_code.endswith('.SZ'):
        return ts_code.replace('.SZ', '').startswith('159')
    return False


def fetch_nav(ts_code, start_date, end_date):
    url = API_BASE + "api"
    body = {
        "api_name": "fund_daily",
        "token": TOKEN,
        "params": {"ts_code": ts_code, "start_date": start_date, "end_date": end_date},
        "fields": "trade_date,close,pre_close,vol,amount"
    }
    try:
        data = json.dumps(body).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read().decode('utf-8'))
        if result.get('code') == 0:
            return result.get('data', [])
        return []
    except Exception:
        return []


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TUSHARE DATA PIPELINE — Build Production Files")
    print("=" * 70)
    print(f"Candidates: {len(EXPANSION_CANDIDATES)} ETFs")
    print(f"Date range: 2013-01-01 to 2026-06-16")

    # Fetch NAV for all candidates
    print("\n--- Fetching NAV for all candidates ---")
    all_nav = {}
    api_calls = 0

    for i, etf in enumerate(EXPANSION_CANDIDATES, 1):
        ts_code = etf['ts_code']
        name = etf['name']
        print(f"[{i:>2}/{len(EXPANSION_CANDIDATES)}] {ts_code} {name}", end=' ', flush=True)

        nav_rows = []
        for year in range(2013, 2027):
            rows = fetch_nav(ts_code, f"{year}0101", f"{year}1231")
            nav_rows.extend(rows)
            api_calls += 1
            time.sleep(0.05)

        if nav_rows:
            dates = sorted(set(r['trade_date'] for r in nav_rows))
            all_nav[ts_code] = {
                'name': name,
                'rationale': etf['rationale'],
                'rows': nav_rows,
                'first_date': dates[0] if dates else '',
                'last_date': dates[-1] if dates else '',
                'count': len(nav_rows),
            }
            print(f"✅ {len(nav_rows)} rows [{dates[0]}..{dates[-1]}]")
        else:
            print(f"❌ NO DATA")

    print(f"\nNAV Fetch: {len(all_nav)}/{len(EXPANSION_CANDIDATES)} ETFs with data, {api_calls} API calls")

    # Save NAV
    nav_path = OUTPUT_DIR / "etf_daily_nav.json"
    with open(nav_path, 'w', encoding='utf-8') as f:
        json.dump(all_nav, f, ensure_ascii=False)
    print(f"✅ NAV saved: {nav_path}")

    # Save ETF metadata CSV
    meta_path = OUTPUT_DIR / "tushare_etf_universe.csv"
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write("ts_code,name,rationale,first_date,last_date,row_count\n")
        for ts_code, info in all_nav.items():
            row = f'{ts_code},"{info["name"]}","{info["rationale"]}",{info["first_date"]},{info["last_date"]},{info["count"]}'
            f.write(row + '\n')
    print(f"✅ Metadata saved: {meta_path}")

    return all_nav


if __name__ == '__main__':
    nav_data = main()
    print(f"\n{'='*70}")
    print(f"Done. {len(nav_data)} ETFs with NAV data.")
    print(f"{'='*70}")
