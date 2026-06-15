#!/usr/bin/env python3
"""
虾池ETF轮动策略 - v2.3 基准复现脚本

直接实现技术方案定稿 v2.3 的参数，复现基准结果:
  年化 14.06%, 回撤 8.21%, 夏普 1.104
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path('/home/ubuntu/claw_eft_strategy')

# ============ 参数 (v2.3 定稿) ============
mom_w = 0.35
vol_w = 0.30
# val_w 已移除（不影响进攻层排序）
top_n = 2
def_alloc = 0.25
step_low = 0.20
step_high = 0.35
max_def = 0.95
hongli_ratio = 0.50
FEE_RATE = 0.00005  # 0.005% 双边

ETFS = ['纳指ETF', '红利低波ETF', '沪深300ETF', '黄金ETF', '国债ETF']
OFFENSIVE_IDX = [0, 2, 3]  # 纳指, 沪深300, 黄金
DEFENSIVE_IDX = [1, 4]      # 红利低波, 国债

# ============ 加载数据 ============
print('Loading data...')

# 使用 H20269 缩放后的周频数据（v2.3 基准使用此数据）
df_weekly = pd.read_csv(BASE / 'all_etfs_nav_2013_2026_h20269_scaled.csv', index_col=0, parse_dates=True)
df_weekly.columns = ETFS

w_prices = df_weekly.values
w_index = df_weekly.index
n_weeks = len(w_index)
w_rets = np.diff(w_prices, axis=0) / w_prices[:-1]

print(f'Data loaded: {n_weeks} weeks ({w_index[0].date()} to {w_index[-1].date()})')

# 对齐 PE 数据到周频
# PE 分位数（v2.3 中 val_w=0 已移除，不影响评分，仅保留以匹配原引擎）
# pe_df = pd.read_csv(BASE / '300etf_pe_percentile_weekly.csv', index_col=0, parse_dates=True)
# pe_weekly = pe_df.reindex(w_index, method='ffill').bfill().values.flatten()

# Precompute factors
w_mom4 = np.full((n_weeks, 5), np.nan)
for i in range(4, n_weeks):
    w_mom4[i] = np.prod(1 + w_rets[i-4:i], axis=0) - 1

w_vol20 = np.full((n_weeks, 5), np.nan)
for i in range(20, n_weeks):
    w_vol20[i] = w_rets[i-20:i].std(axis=0) * np.sqrt(52)

# 防御层动态：红利低波 2019+ 可用
# 数据中可能已有，检查实际数据
first_valid_row = ~np.isnan(w_prices).all(axis=1)
print(f'Data rows with any valid NAV: {first_valid_row.sum()}')

# ============ 回测引擎 ============
print('\nRunning backtest with v2.3 params...')

start_idx = 20
nav = 1.0
peak = 1.0
last_alloc = np.zeros(5)
max_dd = 0.0

# 止损兜底状态
in_stop_loss = False
stop_loss_weeks = 0

# 调仓阈值 7%
rebalance_threshold = 0.07

# 记录
weekly_records = []

for i in range(start_idx, n_weeks - 1):
    # ---- 评分: 进攻层ETF ----
    scores = np.zeros(5)
    for j in OFFENSIVE_IDX:
        if not np.isnan(w_mom4[i, j]) and not np.isnan(w_vol20[i, j]):
            # score = 0.35 * mom4 - 0.30 * vol20  (v2.3, 已移除 val_w)
            scores[j] = mom_w * w_mom4[i, j] - vol_w * w_vol20[i, j]

    # Select top_n offensive
    off_scores = [(scores[j], j) for j in OFFENSIVE_IDX]
    off_scores.sort(reverse=True)
    selected_off = [j for _, j in off_scores[:top_n]]

    # ---- 防御层 ----
    # vol 三段式防御
    nasdaq_vol = w_vol20[i, 0]  # 纳指
    if np.isnan(nasdaq_vol):
        def_ratio = def_alloc
    elif nasdaq_vol < step_low:
        def_ratio = def_alloc
    elif nasdaq_vol > step_high:
        def_ratio = max_def
    else:
        def_ratio = def_alloc + (max_def - def_alloc) * (nasdaq_vol - step_low) / (step_high - step_low)

    # 8% 止损兜底
    if not in_stop_loss and nav < peak * 0.92:
        in_stop_loss = True
        stop_loss_weeks = 0

    if in_stop_loss:
        def_ratio = max(def_ratio, 0.95)
        stop_loss_weeks += 1
        if stop_loss_weeks >= 4:
            in_stop_loss = False

    # ---- 构建仓位 ----
    alloc = np.zeros(5)
    # 防御层
    alloc[DEFENSIVE_IDX[0]] = def_ratio * hongli_ratio       # 红利低波
    alloc[DEFENSIVE_IDX[1]] = def_ratio * (1 - hongli_ratio)  # 国债
    # 进攻层
    for j in selected_off:
        alloc[j] = (1 - def_ratio) / len(selected_off)

    # ---- 调仓阈值检查 ----
    if i > start_idx:
        max_change = max(abs(alloc[j] - last_alloc[j]) for j in range(5))
        if max_change < rebalance_threshold:
            alloc = last_alloc.copy()

    # ---- 计算收益 ----
    turnover = np.sum(np.abs(alloc - last_alloc))
    fee_cost = turnover * FEE_RATE  # 双边已包含在 turnover 中

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
        'selected_off': [ETFS[j] for j in selected_off],
    })

    last_alloc = alloc

# ============ 绩效指标 ============
n_weeks_used = len(weekly_records)
total_ret = nav - 1
annual_ret = (1 + total_ret) ** (52 / n_weeks_used) - 1

returns_arr = np.array([r['weekly_return'] for r in weekly_records])
sharpe = returns_arr.mean() / returns_arr.std() * np.sqrt(52) if returns_arr.std() > 0 else 0
std_annual = returns_arr.std() * np.sqrt(52)

# 简化夏普（不扣无风险利率）
sharpe_simple = returns_arr.mean() / returns_arr.std() * np.sqrt(52)

# 标准夏普（扣2.5% 无风险利率）
rfr_weekly = 0.025 / 52
excess_returns = returns_arr - rfr_weekly
sharpe_std = excess_returns.mean() / excess_returns.std() * np.sqrt(52) if excess_returns.std() > 0 else 0

# 胜率
win_rate = (returns_arr > 0).mean()

print('=' * 60)
print('基准复现结果 (v2.3)')
print('=' * 60)
print(f'  年化收益:      {annual_ret*100:.2f}%  (目标: 14.06%)')
print(f'  最大回撤:      {max_dd*100:.2f}%  (目标: 8.21%)')
print(f'  标准夏普:      {sharpe_std:.3f}  (目标: 1.104)')
print(f'  简化夏普:      {sharpe_simple:.3f}')
print(f'  总收益(13年):  {total_ret*100:.1f}%')
print(f'  年化波动率:    {std_annual*100:.2f}%')
print(f'  周胜率:        {win_rate*100:.1f}%')
print(f'  回测周数:      {n_weeks_used}')
print(f'  数据区间:      {w_index[start_idx].date()} to {w_index[-2].date()}')
print(f'  止损触发:      {sum(1 for r in weekly_records if r["in_stop_loss"])} 周')
print()

# 年度收益分解
print('=' * 60)
print('年度收益分解')
print('=' * 60)
results_df = pd.DataFrame(weekly_records)
results_df['date'] = pd.to_datetime(results_df['date'])
results_df['year'] = results_df['date'].dt.year

for year, group in results_df.groupby('year'):
    yr_ret = (1 + group['weekly_return']).prod() - 1
    avg_def = group['def_ratio'].mean()
    print(f'  {year}: {yr_ret*100:+.1f}%  (avg def_ratio: {avg_def*100:.0f}%)')

# 与文档对比
print()
print('=' * 60)
print('与文档目标对比')
print('=' * 60)
targets = {'annual_return': 14.06, 'max_drawdown': 8.21, 'sharpe': 1.104}
actuals = {'annual_return': annual_ret*100, 'max_drawdown': max_dd*100, 'sharpe': sharpe_std}
for k in targets:
    diff = actuals[k] - targets[k]
    emoji = '✅' if abs(diff) < 0.5 else ('⚠️' if abs(diff) < 2 else '❌')
    print(f'  {k:20s}: 目标={targets[k]:.2f}, 实际={actuals[k]:.2f}, 差异={diff:+.2f}  {emoji}')
