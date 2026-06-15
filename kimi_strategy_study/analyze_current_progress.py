import pandas as pd

df = pd.read_csv('output/param_grid_full.csv')
print('当前已完成的组合数:', len(df))
print()

valid = df[df['max_drawdown'] > -0.15].copy()
print('满足回撤约束: %d / %d (%.1f%%)' % (len(valid), len(df), len(valid)/len(df)*100))
print('年化收益范围: %.2f%% ~ %.2f%%' % (df['annual_return'].min()*100, df['annual_return'].max()*100))
print('最大回撤范围: %.2f%% ~ %.2f%%' % (df['max_drawdown'].min()*100, df['max_drawdown'].max()*100))
print()

print('=== val_w 分析 ===')
for vw in sorted(df['val_w'].unique()):
    subset = df[df['val_w'] == vw]
    valid_sub = subset[subset['max_drawdown'] > -0.15]
    print('val_w=%.1f: 平均年化=%.2f%%, 满足约束=%d/%d' % (vw, subset['annual_return'].mean()*100, len(valid_sub), len(subset)))
print()

print('=== top_n 分析 ===')
for tn in sorted(df['top_n'].unique()):
    subset = df[df['top_n'] == tn]
    valid_sub = subset[subset['max_drawdown'] > -0.15]
    print('top_n=%d: 平均年化=%.2f%%, 满足约束=%d/%d' % (tn, subset['annual_return'].mean()*100, len(valid_sub), len(subset)))
print()

print('=== def_alloc 分析 (top 5) ===')
da = df.groupby('defensive_allocation')['annual_return'].mean().sort_values(ascending=False)
for k, v in da.head(5).items():
    print('  def_alloc=%.2f: 平均年化=%.2f%%' % (k, v*100))
print()

print('=== mom_w vs vol_w 交叉 (平均年化) ===')
pivot = df.pivot_table(values='annual_return', index='mom_w', columns='vol_w', aggfunc='mean')
print(pivot.round(4).to_string())
print()

valid['score'] = valid['annual_return'] / (-valid['max_drawdown'])
top10 = valid.nlargest(10, 'score')
print('=== Top 10 组合（按Calmar评分）===')
print(top10[['mom_w','vol_w','val_w','top_n','defensive_allocation','annual_return','max_drawdown','score']].to_string(index=False))
print()

# 与之前的中期分析对比
print('=== 与之前中期分析(8,500组)的对比 ===')
print('特征                | 之前(8,500组) | 当前(%d组)' % len(df))
print('--------------------|--------------|--------------')
print('val_w=0.0 平均年化  | 10.62%%        | %.2f%%' % (df[df['val_w']==0.0]['annual_return'].mean()*100))
print('val_w>0 平均年化    | ~9.0%%         | %.2f%%' % (df[df['val_w']>0]['annual_return'].mean()*100))
print('top_n=1 平均年化    | 8.20%%         | %.2f%%' % (df[df['top_n']==1]['annual_return'].mean()*100))
print('top_n=2 平均年化    | 10.16%%        | %.2f%%' % (df[df['top_n']==2]['annual_return'].mean()*100))
print('def_alloc=0.30 最优 | 10.45%%        | %.2f%%' % (df[df['defensive_allocation']==0.30]['annual_return'].mean()*100))
