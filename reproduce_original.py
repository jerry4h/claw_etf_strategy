#!/usr/bin/env python3
"""
直接匹配原始 run_unified_search.py 的引擎逻辑
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path('/home/ubuntu/claw_eft_strategy')

# 参数 v2.3
mom_w = 0.35
vol_w = 0.30
val_w = 0.0
top_n = 2
def_alloc = 0.25
step_low = 0.20
step_high = 0.35
max_def = 0.95
hongli_ratio = 0.50
FEE_RATE = 0.00005

ETFS = ['纳指ETF', '红利低波ETF', '沪深300ETF', '黄金ETF', '国债ETF']
OFFENSIVE = [0, 2, 3]
DEFENSIVE = [1, 4]

# 加载数据
df = pd.read_csv(BASE / 'all_etfs_nav_2013_2026_h20269_scaled.csv', index_col=0, parse_dates=True)
df.columns = ETFS

# 使用原引擎的方式：不做额外ffill/bfill，原始数据已是周频
w_prices = df.values
w_index = df.index
n_weeks = len(w_index)
w_rets = np.diff(w_prices, axis=0) / w_prices[:-1]

print(f'Data: {n_weeks} weeks ({w_index[0].date()} to {w_index[-1].date()})')

# Precompute factors (匹配原始引擎)
w_mom4 = np.full((n_weeks, 5), np.nan)
for i in range(4, n_weeks):
    w_mom4[i] = np.prod(1 + w_rets[i-4:i], axis=0) - 1

w_vol20 = np.full((n_weeks, 5), np.nan)
for i in range(20, n_weeks):
    w_vol20[i] = w_rets[i-20:i].std(axis=0) * np.sqrt(52)

# 回测引擎（完全匹配原始引擎）
start_idx = 20
nav = 1.0
peak = 1.0
last_alloc = np.zeros(5)
max_dd = 0.0
in_stop_loss = False
stop_loss_weeks = 0

weekly_records = []

for i in range(start_idx, n_weeks - 1):
    # 评分
    scores = np.zeros(5)
    for j in OFFENSIVE:
        if not np.isnan(w_mom4[i, j]) and not np.isnan(w_vol20[i, j]):
            scores[j] = (
                mom_w * w_mom4[i, j] 
                - vol_w * w_vol20[i, j] 
                + val_w * (0.5 - 0.5)  # val_w=0
            )
    
    # 选top2
    off_scores = [(scores[j], j) for j in OFFENSIVE]
    off_scores.sort(reverse=True)
    selected_off = [j for _, j in off_scores[:top_n]]
    
    # 防御层
    avail_def = DEFENSIVE
    n_def = len(avail_def)
    
    nasdaq_vol = w_vol20[i, 0]
    if np.isnan(nasdaq_vol):
        def_ratio = def_alloc
    elif nasdaq_vol < step_low:
        def_ratio = def_alloc
    elif nasdaq_vol > step_high:
        def_ratio = max_def
    else:
        def_ratio = def_alloc + (max_def - def_alloc) * (nasdaq_vol - step_low) / (step_high - step_low)
    
    # 止损兜底
    if not in_stop_loss and nav < peak * 0.92:
        in_stop_loss = True
        stop_loss_weeks = 0
    
    if in_stop_loss:
        def_ratio = max(def_ratio, 0.95)
        stop_loss_weeks += 1
        if stop_loss_weeks >= 4:
            in_stop_loss = False
    
    # 构建仓位
    alloc = np.zeros(5)
    if n_def == 2:
        alloc[avail_def[0]] = def_ratio * hongli_ratio
        alloc[avail_def[1]] = def_ratio * (1 - hongli_ratio)
    else:
        alloc[avail_def[0]] = def_ratio
    
    for j in selected_off:
        alloc[j] = (1 - def_ratio) / len(selected_off)
    
    # 调仓费（原始引擎方式：无调仓阈值检查）
    turnover = np.sum(np.abs(alloc - last_alloc))
    fee_cost = turnover * FEE_RATE
    
    wret = sum(alloc[j] * w_rets[i, j] for j in range(5) if not np.isnan(w_rets[i, j]))
    nav *= (1 + wret - fee_cost)
    peak = max(peak, nav)
    
    dd = (peak - nav) / peak
    if dd > max_dd:
        max_dd = dd
    
    weekly_records.append({
        'date': w_index[i],
        'nav': nav,
        'peak': peak,
        'weekly_return': wret - fee_cost,
        'def_ratio': def_ratio,
        'in_stop_loss': in_stop_loss,
        'nasdaq_vol': nasdaq_vol,
    })
    
    last_alloc = alloc

# 绩效指标
n_weeks_used = len(weekly_records)
total_ret = nav - 1
annual_ret = (1 + total_ret) ** (52 / n_weeks_used) - 1

returns_arr = np.array([r['weekly_return'] for r in weekly_records])
sharpe_simple = returns_arr.mean() / returns_arr.std() * np.sqrt(52) if returns_arr.std() > 0 else 0

rfr_weekly = 0.025 / 52
excess_returns = returns_arr - rfr_weekly
sharpe_std = excess_returns.mean() / excess_returns.std() * np.sqrt(52) if excess_returns.std() > 0 else 0

print('=' * 60)
print('原始引擎复现结果 (v2.3, 无调仓阈值)')
print('=' * 60)
print(f'  年化收益:      {annual_ret*100:.2f}%  (目标: 14.06%)')
print(f'  最大回撤:      {max_dd*100:.2f}%  (目标: 8.21%)')
print(f'  标准夏普:      {sharpe_std:.3f}  (目标: 1.104)')
print(f'  回测周数:      {n_weeks_used}')
print()

# 年度收益
results_df = pd.DataFrame(weekly_records)
results_df['date'] = pd.to_datetime(results_df['date'])
results_df['year'] = results_df['date'].dt.year

print('年度收益:')
for year, group in results_df.groupby('year'):
    yr_ret = (1 + group['weekly_return']).prod() - 1
    avg_def = group['def_ratio'].mean()
    avg_vol = group['nasdaq_vol'].mean()
    print(f'  {year}: {yr_ret*100:+.1f}%  (def: {avg_def*100:.0f}%, nasdaq_vol: {avg_vol*100:.1f}%)')

print()
print('与文档目标对比:')
for k, v in [('年化收益', annual_ret*100), ('最大回撤', max_dd*100), ('标准夏普', sharpe_std)]:
    print(f'  {k}: {v:.2f}')
