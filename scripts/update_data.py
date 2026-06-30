#!/usr/bin/env python3
"""数据更新脚本 — 新浪实时价 → 添加新行到 CSV"""
import urllib.request, csv, sys, os
from pathlib import Path
from datetime import datetime

PROJECT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT / 'data/all_etfs_nav_2013_20260622_scaled.csv'

# 新浪代码 + 名称
SYMBOLS = [
    ('sh513100', '纳指ETF'),
    ('sh512890', '红利低波ETF'),
    ('sh510300', '沪深300ETF'),
    ('sh518880', '黄金ETF'),
    ('sh511010', '国债ETF'),
]

def fetch_prices():
    """拉取5只ETF实时价"""
    prices = {}
    for code, name in SYMBOLS:
        url = f'https://hq.sinajs.cn/list={code}'
        req = urllib.request.Request(url, headers={'Referer': 'https://finance.sina.com.cn'})
        resp = urllib.request.urlopen(req, timeout=10)
        raw = resp.read().decode('gbk')
        parts = raw.split('"')[1].split(',')
        price = float(parts[3])  # 当前价
        prices[name] = price
        print(f"  {name} ({code}): {price}")
    return prices

def read_last_date():
    """读取CSV最后一行日期"""
    with open(CSV_PATH, 'r') as f:
        lines = f.readlines()
    if len(lines) < 2:
        return None
    last = lines[-1].strip()
    if not last:
        return None
    return last.split(',')[0]

def append_row(prices, date_str):
    """追加一行到CSV"""
    header = ['日期', '纳指ETF', '红利低波ETF', '沪深300ETF', '黄金ETF', '国债ETF']
    names = header[1:]

    # 读取现有CSV确认列顺序
    with open(CSV_PATH, 'r') as f:
        existing_header = f.readline().strip().split(',')

    row = [date_str]
    for n in names:
        row.append(str(prices.get(n, '')))

    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    print(f"\n  ✅ 已添加: {date_str}")

if __name__ == '__main__':
    print("=" * 50)
    print(" 数据更新 — 新浪实时价")
    print("=" * 50)

    last_date = read_last_date()
    today = datetime.now().strftime('%Y-%m-%d')
    print(f" CSV 最后日期: {last_date}")
    print(f" 今天: {today}")

    if last_date and last_date >= today:
        print(f"\n  ⚠️ CSV 已包含今天({today})的数据, 跳过更新")
        sys.exit(0)

    print("\n  拉取实时价...")
    prices = fetch_prices()
    append_row(prices, today)
    print("\n  下一步: python scripts/rebalance_live.py")