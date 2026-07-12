#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF 周线数据更新脚本（新数据用周五快照）

逻辑：
- 原 h20269_scaled 数据保持不变（日期和值均不动）
- 新增量数据从 Tushare 拉取，取每周五收盘价
- 新数据用原数据最后一周的比例因子缩放
- 文件名带最新数据日期标识

用法:
    python3 scripts/update_etf_data_tushare.py
"""

import tushare as ts
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# ---------- 配置 ----------
TUSHARE_TOKEN = '44b2cb657caaddd0c5c9ea6bdcfcbeed72f0a09470c7dbba54d16a4d'
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OLD_FILE = os.path.join(DATA_DIR, 'all_etfs_nav_2013_2026_h20269_scaled.csv')

# ETF 代码映射
ETF_MAP = {
    '纳指ETF': '513100.SH',
    '红利低波ETF': '512890.SH',
    '中证500ETF': '510500.SH',
    '黄金ETF': '518880.SH',
    '国债ETF': '511010.SH',
}

# ---------- 初始化 ----------
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

print("=" * 60)
print("ETF 周线数据更新（周五快照）")
print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ---------- 1. 读取原数据 ----------
print("\nStep 1: 读取原数据")
print("-" * 40)

if not os.path.exists(OLD_FILE):
    print(f"\u274c 原数据文件不存在: {OLD_FILE}")
    sys.exit(1)

old = pd.read_csv(OLD_FILE)
old['日期'] = pd.to_datetime(old['日期'])
old = old.sort_values('日期').reset_index(drop=True)

print(f"原数据: {len(old)} 行")
print(f"时间范围: {old['日期'].min().strftime('%Y-%m-%d')} ~ {old['日期'].max().strftime('%Y-%m-%d')}")
print(f"最后3行:")
print(old.tail(3).to_string(index=False))

last_old_date = old['日期'].max()

# ---------- 2. 计算每列的缩放比例 ----------
print("\nStep 2: 计算缩放比例（基于原数据最后日期）")
print("-" * 40)

ratios = {}
for name, code in ETF_MAP.items():
    old_val = old[old['日期'] == last_old_date][name].values[0]
    last_date_str = last_old_date.strftime('%Y%m%d')
    
    df = pro.fund_daily(ts_code=code, start_date=last_date_str, end_date=last_date_str)
    if df is not None and len(df) > 0:
        raw_close = df.iloc[0]['close']
        ratio = old_val / raw_close
        ratios[name] = ratio
        print(f"  {name}: 原值={old_val:.4f}, tushare={raw_close}, ratio={ratio:.6f}")
    else:
        # 该日没有交易数据，往前找
        df = pro.fund_daily(ts_code=code, start_date='20260401', end_date=last_date_str)
        if df is not None and len(df) > 0:
            df = df.sort_values('trade_date')
            last_row = df.iloc[-1]
            raw_close = last_row['close']
            ratio = old_val / raw_close
            ratios[name] = ratio
            print(f"  {name}: 原值={old_val:.4f}, tushare(最近{last_row['trade_date']})={raw_close}, ratio={ratio:.6f}")
        else:
            ratios[name] = 1.0
            print(f"  {name}: \u26a0\ufe0f 找不到原始数据，ratio=1.0")

# ---------- 3. 拉取增量日线数据 ----------
print("\nStep 3: 拉取增量日线")
print("-" * 40)

# 从原数据最后日期之后开始拉取
start_inc = (last_old_date + timedelta(days=1)).strftime('%Y%m%d')
today = datetime.now().strftime('%Y%m%d')
print(f"增量区间: {start_inc} ~ {today}")

all_inc = {}
for name, code in ETF_MAP.items():
    # 多拉取一些确保覆盖完整周
    fetch_start = (last_old_date - timedelta(days=14)).strftime('%Y%m%d')
    df = pro.fund_daily(ts_code=code, start_date=fetch_start, end_date=today)
    if df is not None and len(df) > 0:
        df = df.sort_values('trade_date').reset_index(drop=True)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        all_inc[name] = df
        print(f"  \u2705 {name}: {len(df)} 行, {df['trade_date'].min().strftime('%Y-%m-%d')} ~ {df['trade_date'].max().strftime('%Y-%m-%d')}")
    else:
        print(f"  \u26a0\ufe0f {name}: 无数据")

if not all_inc:
    print("\u274c 无增量数据")
    sys.exit(1)

# ---------- 4. 过滤增量（>原数据最后日期）并缩放 ----------
print("\nStep 4: 过滤增量并缩放")
print("-" * 40)

inc_filtered = {}
for name in ETF_MAP:
    if name not in all_inc:
        continue
    df = all_inc[name].copy()
    df_inc = df[df['trade_date'] > last_old_date].copy()
    if len(df_inc) == 0:
        print(f"  {name}: 无新的增量")
        inc_filtered[name] = pd.DataFrame()
        continue
    df_inc['close_scaled'] = df_inc['close'] * ratios[name]
    inc_filtered[name] = df_inc
    print(f"  \u2705 {name}: {len(df_inc)} 行增量")

# ---------- 5. 合并为周线（取每周五） ----------
print("\nStep 5: 聚合成周线（周五快照）")
print("-" * 40)

# 合并所有 ETF 的增量日线
merged_inc = None
for name in ETF_MAP:
    if name not in inc_filtered or len(inc_filtered[name]) == 0:
        continue
    d = inc_filtered[name][['trade_date', 'close_scaled']].rename(columns={'close_scaled': name})
    if merged_inc is None:
        merged_inc = d.copy()
    else:
        merged_inc = pd.merge(merged_inc, d, on='trade_date', how='outer')

if merged_inc is None or len(merged_inc) == 0:
    print("\u274c 无增量数据，数据已是最新")
    sys.exit(0)

merged_inc = merged_inc.sort_values('trade_date').reset_index(drop=True)

# 标记ISO周
merged_inc['isoyear'] = merged_inc['trade_date'].dt.isocalendar().year
merged_inc['isoweek'] = merged_inc['trade_date'].dt.isocalendar().week
merged_inc['weekday'] = merged_inc['trade_date'].dt.weekday  # Mon=0, Fri=4

# 每周取最后一个交易日
weekly_rows = []
for (year, week), group in merged_inc.groupby(['isoyear', 'isoweek']):
    # 优先取周五(weekday=4)，没有则取该周最后交易日
    friday = group[group['weekday'] == 4]
    if len(friday) > 0:
        weekly_rows.append(friday.sort_values('trade_date').iloc[-1])
    else:
        weekly_rows.append(group.sort_values('trade_date').iloc[-1])

weekly_new = pd.DataFrame(weekly_rows).sort_values('trade_date').reset_index(drop=True)
# 确保没有重叠（但原数据是周一，新数据是周五，理论上不会重叠）
weekly_new = weekly_new[weekly_new['trade_date'] > last_old_date].copy()

if len(weekly_new) == 0:
    print("\u274c 无新周数据")
    sys.exit(0)

print(f"\u2705 增量周线: {len(weekly_new)} 行")
cols_show = ['trade_date'] + list(ETF_MAP.keys())
print(weekly_new[cols_show].to_string(index=False))

# ---------- 6. 拼接 ----------
print("\nStep 6: 拼接新旧数据")
print("-" * 40)

old_out = old.copy()
old_out['日期'] = old_out['日期'].dt.strftime('%Y-%m-%d')

new_out = weekly_new[['trade_date'] + list(ETF_MAP.keys())].copy()
new_out = new_out.rename(columns={'trade_date': '日期'})
new_out['日期'] = new_out['日期'].dt.strftime('%Y-%m-%d')

combined = pd.concat([old_out, new_out], ignore_index=True)
combined = combined.sort_values('日期').reset_index(drop=True)

print(f"总行数: {len(combined)}")
print(f"时间范围: {combined['日期'].min()} ~ {combined['日期'].max()}")
print("\n最后5行:")
print(combined.tail(5).to_string(index=False))

# ---------- 7. 写出新文件 ----------
print("\nStep 7: 写出新文件")
print("-" * 40)

final_date = combined['日期'].max()
final_date_nodash = final_date.replace('-', '')
new_filename = f"all_etfs_nav_2013_{final_date_nodash}_scaled.csv"
new_filepath = os.path.join(DATA_DIR, new_filename)

combined.to_csv(new_filepath, index=False)
print(f"\u2705 新文件: {new_filepath}")
print(f"   行数: {len(combined)}")
print(f"   时间范围: {combined['日期'].min()} ~ {combined['日期'].max()}")

# ---------- 8. 一致性检查 ----------
print("\nStep 8: 一致性检查")
print("-" * 40)

# 8a. 空值
nulls = combined.isnull().sum()
if nulls.sum() > 0:
    print(f"\u26a0\ufe0f 空值:")
    print(nulls[nulls > 0].to_string())
else:
    print("\u2705 无空值")

# 8b. 原数据完整性
old_in_combined = combined[combined['日期'] <= last_old_date.strftime('%Y-%m-%d')]
if len(old_in_combined) == len(old):
    print(f"\u2705 原数据 {len(old)} 行完整保留")
else:
    print(f"\u26a0\ufe0f 原数据保留行数: {len(old_in_combined)}/{len(old)}")

# 8c. 日期连续
dates = pd.to_datetime(combined['日期'])
diffs = dates.diff().dt.days.iloc[1:]
big_gaps = diffs[diffs > 14]
if len(big_gaps) > 0:
    print(f"\u26a0\ufe0f 大间隔 ({len(big_gaps)}处):")
    for idx in big_gaps.index:
        print(f"    {dates.iloc[idx-1].strftime('%Y-%m-%d')} -> {dates.iloc[idx].strftime('%Y-%m-%d')} ({int(diffs.loc[idx])}天)")
else:
    print("\u2705 日期连续性通过")

# 8d. 旧数据最后值 vs 新数据第一个值 衔接合理性
old_last = old_out.tail(1)
new_first = new_out.head(1)
print(f"\n衔接检查:")
print(f"  原数据最后: {old_last.to_string(index=False)}")
print(f"  新数据第一: {new_first.to_string(index=False)}")

# ---------- 最终摘要 ----------
print(f"\n{'='*60}")
print(f"\U0001f4ca 最终文件摘要")
print(f"{'='*60}")
print(f"  \U0001f4c4 文件: {new_filename}")
print(f"  \U0001f4cf 行数: {len(combined)}")
print(f"  \U0001f4c5 时间范围: {combined['日期'].min()} ~ {combined['日期'].max()}")
print(f"  \U0001f4c8 ETF: {len(ETF_MAP)}只")
print(f"  \U0001f4cc 快照日: 存量=周一(历史), 增量=周五(最新)")

# 软链接
link_path = os.path.join(DATA_DIR, 'all_etfs_nav_latest.csv')
if os.path.exists(link_path) or os.path.islink(link_path):
    os.unlink(link_path)
os.symlink(new_filename, link_path)
print(f"  \U0001f517 软链接: {link_path} -> {new_filename}")

print(f"\n\u2705 数据更新完成!")