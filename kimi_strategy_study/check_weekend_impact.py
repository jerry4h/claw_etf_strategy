import pandas as pd
from data_loader import load_nav_data, calculate_returns

df = load_nav_data()
ret = calculate_returns(df)

# 检查周末收益率
weekends = ret[ret.index.dayofweek >= 5]
print('周末收益率统计:')
print(weekends.describe())
print()
print('非零周末收益率数量:', (weekends != 0).sum().sum())
print('周末收益率总和:', weekends.sum().sum())
print()

# 对比：过滤周末后的20日动量 vs 含周末的20日动量
ret_no_weekend = ret[ret.index.dayofweek < 5]
mom_with = ret.rolling(20).sum() * 100
mom_no = ret_no_weekend.rolling(20).sum() * 100

vol_with = ret.rolling(20).std() * (252**0.5) * 100
vol_no = ret_no_weekend.rolling(20).std() * (252**0.5) * 100

# 取每周一对比
mondays = ret[ret.index.dayofweek == 0].index[-10:]
print('最近10个周一的因子对比（纳指ETF）:')
print(f"{'日期':<12} {'动量(含周末)':>12} {'动量(纯交易日)':>14} {'波动率(含周末)':>14} {'波动率(纯交易日)':>16}")
for d in mondays:
    m_w = mom_with.loc[d, "纳指ETF"] if d in mom_with.index else None
    m_n = mom_no.loc[d, "纳指ETF"] if d in mom_no.index else None
    v_w = vol_with.loc[d, "纳指ETF"] if d in vol_with.index else None
    v_n = vol_no.loc[d, "纳指ETF"] if d in vol_no.index else None
    print(f"{str(d.date()):<12} {m_w:>12.4f} {m_n:>14.4f} {v_w:>14.4f} {v_n:>16.4f}")

# 全量统计差异
common = mom_with.dropna().index.intersection(mom_no.dropna().index)
diff_mom = (mom_with.loc[common, "纳指ETF"] - mom_no.loc[common, "纳指ETF"]).describe()
diff_vol = (vol_with.loc[common, "纳指ETF"] - vol_no.loc[common, "纳指ETF"]).describe()
print()
print('动量差异统计 (含周末 - 纯交易日):')
print(diff_mom)
print()
print('波动率差异统计 (含周末 - 纯交易日):')
print(diff_vol)

# 计算周末占比
print()
print(f'总数据条数: {len(ret)}')
print(f'周末数据条数: {len(weekends)}')
print(f'周末占比: {len(weekends)/len(ret)*100:.1f}%')
print(f'实际交易日: {len(ret_no_weekend)}')
