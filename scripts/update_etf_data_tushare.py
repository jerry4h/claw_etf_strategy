#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量拉取 ETF 日线行情，聚合成周线（每周一收盘价），
拼接到原数据文件后，输出带最后日期标识的新文件。

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
    '沪深300ETF': '510300.SH',
    '黄金ETF': '518880.SH',
    '国债ETF': '511010.SH',
}

# ---------- 初始化 ----------
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# ---------- 1. 读取原数据 ----------
print("=" * 60)
print("Step 1: 读取原数据")
print("=" * 60)

if not os.path.exists(OLD_FILE):
    print(f"❌ 原数据文件不存在: {OLD_FILE}")
    sys.exit(1)

old = pd.read_csv(OLD_FILE)
old['日期'] = pd.to_datetime(old['日期'])
old = old.sort_values('日期').reset_index(drop=True)

print(f"原数据: {len(old)} 行, {old['日期'].min().strftime('%Y-%m-%d')} ~ {old['日期'].max().strftime('%Y-%m-%d')}")
print(f"字段: {list(old.columns)}")
print(f"最后一行: {old.tail(1).to_string(index=False)}")

last_date = old['日期'].max()
last_date_str = last_date.strftime('%Y-%m-%d')

# ---------- 2. 拉取增量日线 ----------
print("\n" + "=" * 60)
print("Step 2: 拉取增量日线行情")
print("=" * 60)

# 从原数据最后日期+1天开始拉取
start_raw = (last_date + timedelta(days=1)).strftime('%Y%m%d')
today = datetime.now().strftime('%Y%m%d')

print(f"数据拉取区间: {start_raw} ~ {today}")

all_daily = {}
for name, code in ETF_MAP.items():
    # 拉取足够天数的日线（包含最后重叠周）
    fetch_start = (last_date - timedelta(days=14)).strftime('%Y%m%d')
    df = pro.fund_daily(ts_code=code, start_date=fetch_start, end_date=today)
    if df is None or len(df) == 0:
        print(f"  ⚠️ {name} ({code}): 无数据返回，跳过")
        continue
    df = df.sort_values('trade_date').reset_index(drop=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    print(f"  ✅ {name} ({code}): {len(df)} 行, {df['trade_date'].min().strftime('%Y-%m-%d')} ~ {df['trade_date'].max().strftime('%Y-%m-%d')}")
    all_daily[name] = df

if len(all_daily) == 0:
    print("❌ 所有ETF均无数据返回")
    sys.exit(1)

# ---------- 3. 计算缩放比例 ----------
print("\n" + "=" * 60)
print("Step 3: 计算缩放比例（以原数据最后周一的值为基准）")
print("=" * 60)

ratios = {}
for name in ETF_MAP.keys():
    if name not in all_daily:
        continue
    
    old_val = old[old['日期'] == last_date][name].values[0]
    last_date_str_fmt = last_date.strftime('%Y%m%d')
    df = all_daily[name]
    row = df[df['trade_date'].dt.strftime('%Y%m%d') == last_date_str_fmt]
    
    if len(row) > 0:
        raw_val = row.iloc[0]['close']
        ratio = old_val / raw_val
    else:
        # 如果原数据最后日期不是交易日，找最近的交易日
        df_after = df[df['trade_date'] >= last_date]
        if len(df_after) > 0:
            nearest = df_after.iloc[0]
            raw_val = nearest['close']
            ratio = old_val / raw_val
            print(f"  ⚠️ {name}: 原数据日期{last_date_str}无对应日线，使用最近交易日{nearest['trade_date'].strftime('%Y-%m-%d')}")
        else:
            ratio = 1.0
    
    ratios[name] = ratio
    print(f"  {name}: ratio={ratio:.6f} (原值={old_val:.4f}, tushare原始值=NA)")

# ---------- 4. 过滤增量数据 + 缩放 ----------
print("\n" + "=" * 60)
print("Step 4: 过滤增量日线并应用缩放")
print("=" * 60)

incremental = {}
for name in ETF_MAP.keys():
    if name not in all_daily:
        continue
    
    df = all_daily[name].copy()
    # 只保留大于原数据最后日期的数据
    df_inc = df[df['trade_date'] > last_date].copy()
    
    if len(df_inc) == 0:
        print(f"  {name}: 无增量数据")
        incremental[name] = df_inc
        continue
    
    df_inc['close_scaled'] = df_inc['close'] * ratios[name]
    incremental[name] = df_inc
    print(f"  ✅ {name}: {len(df_inc)} 行增量, {df_inc['trade_date'].min().strftime('%Y-%m-%d')} ~ {df_inc['trade_date'].max().strftime('%Y-%m-%d')}")

# ---------- 5. 合并所有 ETF 增量数据为周线 ----------
print("\n" + "=" * 60)
print("Step 5: 合并为周线（每周一收盘）")
print("=" * 60)

# 先把所有日线合并成一张表
merged_daily = None
for name in ETF_MAP.keys():
    if name not in incremental or len(incremental[name]) == 0:
        continue
    d = incremental[name][['trade_date', 'close_scaled']].rename(columns={'close_scaled': name})
    if merged_daily is None:
        merged_daily = d.copy()
    else:
        merged_daily = pd.merge(merged_daily, d, on='trade_date', how='outer')

if merged_daily is None or len(merged_daily) == 0:
    print("❌ 无增量数据可处理，数据已是最新")
    sys.exit(0)

merged_daily = merged_daily.sort_values('trade_date').reset_index(drop=True)

# 提取每周一的数据
merged_daily['weekday'] = merged_daily['trade_date'].dt.weekday
weekly_new = merged_daily[merged_daily['weekday'] == 0].copy()

if len(weekly_new) == 0:
    # 如果最新的周还没到周一，取最近的一个交易日
    print("  ⚠️ 增量数据中无周一数据，取每周最后一个交易日作为周线")
    merged_daily['week'] = merged_daily['trade_date'].dt.isocalendar().year.astype(str) + '-W' + \
                           merged_daily['trade_date'].dt.isocalendar().week.astype(str).str.zfill(2)
    weekly_new = merged_daily.groupby('week').last().reset_index()
    weekly_new = weekly_new.sort_values('trade_date').reset_index(drop=True)
    # 剔除与原数据重叠的周（原数据最后一周的周一）
    weekly_new = weekly_new[weekly_new['trade_date'] > last_date].copy()

print(f"增量周线: {len(weekly_new)} 行, {weekly_new['trade_date'].min().strftime('%Y-%m-%d')} ~ {weekly_new['trade_date'].max().strftime('%Y-%m-%d')}")
print(weekly_new[['trade_date'] + list(ETF_MAP.keys())].to_string(index=False))

# ---------- 6. 拼接 ----------
print("\n" + "=" * 60)
print("Step 6: 拼接新旧数据")
print("=" * 60)

old_out = old.copy()
old_out['日期'] = old_out['日期'].dt.strftime('%Y-%m-%d')

weekly_new_out = weekly_new[['trade_date'] + list(ETF_MAP.keys())].copy()
weekly_new_out = weekly_new_out.rename(columns={'trade_date': '日期'})
weekly_new_out['日期'] = weekly_new_out['日期'].dt.strftime('%Y-%m-%d')

combined = pd.concat([old_out, weekly_new_out], ignore_index=True)
combined = combined.drop_duplicates(subset=['日期'], keep='last').reset_index(drop=True)
combined = combined.sort_values('日期').reset_index(drop=True)

print(f"总行数: {len(combined)}")
print(f"日期范围: {combined['日期'].min()} ~ {combined['日期'].max()}")

# 验证最后几行
print(f"\n最后5行:")
print(combined.tail(5).to_string(index=False))

# ---------- 7. 写出 ----------
print("\n" + "=" * 60)
print("Step 7: 写出新文件")
print("=" * 60)

final_date = combined['日期'].max()
final_date_nodash = final_date.replace('-', '')
new_filename = f"all_etfs_nav_2013_{final_date_nodash}_scaled.csv"
new_filepath = os.path.join(DATA_DIR, new_filename)

combined.to_csv(new_filepath, index=False)
print(f"✅ 新文件: {new_filepath}")
print(f"   行数: {len(combined)}")
print(f"   日期范围: {combined['日期'].min()} ~ {combined['日期'].max()}")

# ---------- 8. 一致性检查 ----------
print("\n" + "=" * 60)
print("Step 8: 数据一致性检查")
print("=" * 60)

# 8a. 检查原数据与新数据重叠部分是否一致
old_last_5 = old_out.tail(5)
new_overlap = combined[combined['日期'].isin(old_last_5['日期'].values)].tail(5)
print(f"\n8a. 重叠日期行数: {len(new_overlap)}")
print(f"    原数据最后5行日期: {list(old_last_5['日期'].values)}")
print(f"    新数据相同日期行数: {len(new_overlap)}")

# 8b. 逐行比较重叠区域
if len(new_overlap) > 0:
    mismatches = []
    for _, o_row in old_last_5.iterrows():
        n_row = new_overlap[new_overlap['日期'] == o_row['日期']]
        if len(n_row) == 0:
            continue
        n_row = n_row.iloc[0]
        for col in ETF_MAP.keys():
            if abs(o_row[col] - n_row[col]) > 1e-4:
                mismatches.append(f"    日期={o_row['日期']}, 字段={col}: 原值={o_row[col]}, 新值={n_row[col]}")
    
    if mismatches:
        print(f"  ⚠️ 发现 {len(mismatches)} 处不一致:")
        for m in mismatches:
            print(m)
    else:
        print("  ✅ 重叠区域数据完全一致")

# 8c. 检查增量数据是否全部在原数据之后
assert combined[combined['日期'] <= last_date_str].shape[0] == len(old), \
    f"新数据在 {last_date_str} 之前的行数({combined[combined['日期'] <= last_date_str].shape[0]})与旧数据({len(old)})不一致"
print(f"  ✅ 新数据在 {last_date_str} 之前共 {len(old)} 行，与原数据一致")

# 8d. 检查新增行数
new_rows = combined[combined['日期'] > last_date_str]
print(f"  ✅ 新增行数: {len(new_rows)}")

# 8e. 检查是否有空值
null_counts = combined.isnull().sum()
if null_counts.sum() > 0:
    print(f"  ⚠️ 空值检查:")
    print(null_counts[null_counts > 0].to_string())
else:
    print("  ✅ 无空值")

# 8f. 检查日期连续性
dates = pd.to_datetime(combined['日期'])
date_diffs = dates.diff().dt.days.iloc[1:]
expected_gaps = [7]  # 周线，间隔应为7天
large_gaps = date_diffs[date_diffs > 14]
if len(large_gaps) > 0:
    print(f"  ⚠️ 日期间隔>14天的位置 ({len(large_gaps)}处):")
    for idx in large_gaps.index:
        print(f"    {dates.iloc[idx-1].strftime('%Y-%m-%d')} -> {dates.iloc[idx].strftime('%Y-%m-%d')} ({int(date_diffs.loc[idx])}天)")
else:
    print("  ✅ 日期间隔检查通过（所有间隔≤14天）")

# 8g. 最终统计摘要
print(f"\n{'='*60}")
print(f"📊 最终文件摘要")
print(f"{'='*60}")
print(f"  📄 文件: {new_filename}")
print(f"  📏 行数: {len(combined)}")
print(f"  📅 时间范围: {combined['日期'].min()} ~ {combined['日期'].max()}")
print(f"  📈 ETF数量: {len(ETF_MAP)}")
print(f"  📋 字段: {list(combined.columns)}")
print(f"\n✅ 数据更新完成!")