import pandas as pd

# The scaled file is already weekly, likely with H20269 data
scaled = pd.read_csv('/home/ubuntu/claw_eft_strategy/all_etfs_nav_2013_2026_h20269_scaled.csv', index_col=0, parse_dates=True)

# The merged file is daily, resample to weekly
merged_daily = pd.read_csv('/home/ubuntu/claw_eft_strategy/kimi_strategy_study/meta_data/all_etfs_nav_2013_2026_merged.csv', index_col=0, parse_dates=True)
merged_daily.columns = ['纳指ETF', '红利低波ETF', '沪深300ETF', '黄金ETF', '国债ETF']
merged_weekly = merged_daily.resample('W-MON').last().dropna(how='all').ffill().bfill()

# Compare 红利低波ETF values
print('红利低波ETF comparison at key dates:')
print('-' * 80)
print('{:<15} {:>18} {:>18} {:>18}'.format('Date', 'Scaled(H20269)', 'Merged', 'Diff'))
print('-' * 80)

# Check around 2019 when 512890 was listed
for dt_str in ['2018-12-31', '2019-01-07', '2019-03-04', '2019-06-03', '2019-12-30']:
    dt = pd.Timestamp(dt_str)
    s_val = scaled.loc[dt, '红利低波ETF'] if dt in scaled.index else None
    m_val = merged_weekly.loc[dt, '红利低波ETF'] if dt in merged_weekly.index else None
    s_str = '{:.6f}'.format(s_val) if s_val is not None else 'N/A'
    m_str = '{:.6f}'.format(m_val) if m_val is not None else 'N/A'
    diff_str = '{:.6f}'.format(s_val - m_val) if (s_val is not None and m_val is not None) else 'N/A'
    print('{:<15} {:>18} {:>18} {:>18}'.format(dt_str, s_str, m_str, diff_str))
print()

# Also compare full dataset correlation
s_hon = scaled['红利低波ETF']
m_hon = merged_weekly['红利低波ETF']
common = s_hon.index.intersection(m_hon.index)
print('Common dates:', len(common))
print('Correlation:', s_hon.loc[common].corr(m_hon.loc[common]))

s_nas = scaled['纳指ETF']
m_nas = merged_weekly['纳指ETF']
print('纳指 correlation:', s_nas.loc[common].corr(m_nas.loc[common]))
