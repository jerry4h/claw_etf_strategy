#!/usr/bin/env python3
"""市场状态分布分析 — 运行 P1 全开回测并提取状态统计"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.strategy import (
    StrategyConfig, load_config, MarketState,
    detect_market_state
)
from src.data_loader import (
    ETFS, OFFENSIVE_IDX, DEFENSIVE_IDX,
    load_nav_data, resample_weekly
)
from src.factors import compute_all_factors

# 加载 P1 全开配置
config = load_config(PROJECT_ROOT / "config/strategy_v2_5_p1_all_on.yaml")
print(f"Strategy: {config.name}")
print(f"stateful_stop_loss: {config.stateful_stop_loss}")
print(f"max_single_alloc: {config.max_single_alloc}")
print(f"overflow_to_defense_only: {config.overflow_to_defense_only}")
print()

# 先运行正式回测获取 nav 轨迹
from src.backtest import run_backtest
result = run_backtest(config)
records = result.weekly_records
print(f"Total weekly records: {len(records)}")

# 重新跑核心循环来记录市场状态
_nav_path = PROJECT_ROOT / config.nav_path
_pe_path = PROJECT_ROOT / config.pe_path if config.pe_path else None

nav_df_raw = load_nav_data(_nav_path)
weekly_nav = resample_weekly(nav_df_raw, anchor=config.anchor)

pe_df = None
if _pe_path and _pe_path.exists():
    from src.data_loader import load_pe_percentile
    pe_df = load_pe_percentile(_pe_path)

start = config.start_date
end = config.end_date
if start:
    weekly_nav = weekly_nav[weekly_nav.index >= pd.to_datetime(start)]
if end:
    weekly_nav = weekly_nav[weekly_nav.index <= pd.to_datetime(end)]

config_dict = {'factors': {
    'mom_window': config.mom_window,
    'vol_window': config.vol_window,
    'pe_window_years': config.pe_window_years,
}}
factors = compute_all_factors(weekly_nav, pe_df, config_dict)

w_prices = weekly_nav.values
w_index = weekly_nav.index
n_weeks = len(w_index)
w_rets = np.diff(w_prices, axis=0) / w_prices[:-1]

vol_values = factors['volatility'].values
NASDAQ_IDX = 0

start_idx = config.vol_window
state_history = []

for i in range(start_idx, n_weeks - 1):
    date = w_index[i]
    
    # 计算市场状态
    if i >= 12:
        nasdaq_12w_ret = np.prod(1 + w_rets[i-12:i, NASDAQ_IDX]) - 1
    else:
        nasdaq_12w_ret = 0.0
    
    if i >= 20:
        current_vol = np.std(w_rets[i-20:i, NASDAQ_IDX], ddof=0) * np.sqrt(52)
        vol_history = []
        for j in range(max(20, i-104), i+1):
            v = np.std(w_rets[max(0, j-20):j, NASDAQ_IDX], ddof=0) * np.sqrt(52)
            vol_history.append(v)
        nasdaq_vol_pct = sum(1 for v in vol_history if v < current_vol) / len(vol_history)
    else:
        nasdaq_vol_pct = 0.5
    
    # 从 records 获取回撤
    rec_idx = i - start_idx
    if rec_idx < len(records):
        dd_current = records[rec_idx]['drawdown']
    else:
        dd_current = 0.0
    
    state = detect_market_state(nasdaq_12w_ret, nasdaq_vol_pct, dd_current, config)
    state_history.append({
        'date': w_index[i + 1],
        'state': state.value,
        'nasdaq_12w_ret': round(nasdaq_12w_ret, 4),
        'nasdaq_vol_pct': round(nasdaq_vol_pct, 3),
        'drawdown': round(dd_current, 4),
    })

state_df = pd.DataFrame(state_history)
state_df['date'] = pd.to_datetime(state_df['date'])
state_df.set_index('date', inplace=True)

# 总体分布
print("=" * 60)
print("MARKET STATE DISTRIBUTION")
print("=" * 60)
state_counts = state_df['state'].value_counts()
state_pcts = state_df['state'].value_counts(normalize=True) * 100
for state in ['bull', 'normal', 'correction', 'crisis']:
    cnt = state_counts.get(state, 0)
    pct = state_pcts.get(state, 0)
    print(f"  {state.upper():12s}: {cnt:5d} weeks ({pct:5.1f}%)")

# 年度分布
print()
print("=" * 60)
print("ANNUAL STATE DISTRIBUTION")
print("=" * 60)
state_df['year'] = state_df.index.year
yearly = state_df.groupby('year')['state'].value_counts().unstack(fill_value=0)
for st in ['bull', 'normal', 'correction', 'crisis']:
    if st not in yearly.columns:
        yearly[st] = 0

print(f"{'Year':>6s}  {'BULL':>6s}  {'NORMAL':>7s}  {'CORR':>6s}  {'CRISIS':>6s}  {'TOTAL':>6s}")
print("-" * 56)
for yr in sorted(yearly.index):
    b = yearly.loc[yr, 'bull']
    n = yearly.loc[yr, 'normal']
    c = yearly.loc[yr, 'correction']
    cr = yearly.loc[yr, 'crisis']
    total = b + n + c + cr
    print(f"{yr:6d}  {b:6d}  {n:7d}  {c:6d}  {cr:6d}  {total:6d}")

# 状态转换矩阵
print()
print("=" * 60)
print("STATE TRANSITION MATRIX")
print("=" * 60)
states_order = ['bull', 'normal', 'correction', 'crisis']
prev_states = state_df['state'].shift(1).values[1:]
curr_states = state_df['state'].values[1:]
transitions = {}
for ps in states_order:
    for cs in states_order:
        cnt = ((prev_states == ps) & (curr_states == cs)).sum()
        transitions[(ps, cs)] = cnt

fmt_hdr = "From\\To"
header = f"{fmt_hdr:>12s}  {'BULL':>6s}  {'NORMAL':>7s}  {'CORR':>6s}  {'CRISIS':>6s}"
print(header)
print("-" * 49)
for ps in states_order:
    cols = []
    for cs in states_order:
        cols.append(f"{transitions.get((ps, cs), 0):6d}")
    print(f"{ps.upper():>12s}  {'  '.join(cols)}")

# 存储结果供报告使用
state_df.to_csv('/tmp/market_state_history.csv')
print("\nState history saved to /tmp/market_state_history.csv")
