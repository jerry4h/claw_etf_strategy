import pandas as pd

# 读取原始CSV，不做ffill
df_raw = pd.read_csv('meta_data/all_etfs_nav_2013_2026_merged.csv')
df_raw['日期'] = pd.to_datetime(df_raw['日期'])
df_raw = df_raw.set_index('日期').sort_index()

ETFS = ['纳指ETF', '红利低波ETF', '沪深300ETF', '黄金ETF', '国债ETF']
weekday_names = ['周一', '周二', '周三', '周四', '周五']

# 检查2024年春节（A股2月9日-2月16日休市）
print('=== 2024年春节期间原始数据 ===')
mask = (df_raw.index >= '2024-02-05') & (df_raw.index <= '2024-02-20')
print(df_raw.loc[mask, ETFS])
print()

# 检查NaN且是工作日的日期（即节假日）
print('=== 工作日但全NaN的日期（节假日） ===')
df_workday = df_raw[df_raw.index.dayofweek < 5]
all_nan = df_workday[ETFS].isna().all(axis=1)
holiday_dates = df_workday[all_nan].index
print('总数:', len(holiday_dates), '天')
print('最近20个:')
for d in holiday_dates[-20:]:
    print(' ', d.date(), weekday_names[d.dayofweek])
print()

# 检查周末NaN
print('=== 周末NaN统计 ===')
weekend_nan = df_raw[df_raw.index.dayofweek >= 5]
all_nan_weekend = weekend_nan[ETFS].isna().all(axis=1)
print('周末总天数:', len(weekend_nan))
print('周末全NaN天数:', all_nan_weekend.sum())
print('周末有数据天数:', (len(weekend_nan) - all_nan_weekend.sum()))
print()

# 对比：ffill后的数据
print('=== ffill后的数据（2024年春节同期） ===')
df_ffill = df_raw.copy()
for col in ETFS:
    df_ffill[col] = df_ffill[col].ffill()
print(df_ffill.loc[mask, ETFS])
