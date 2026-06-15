import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 200)

df = pd.read_csv('output/param_grid_full.csv')
print('='*60)
print('全量网格搜索 中期分析报告 (8,500 / 11,000)')
print('='*60)
print()

# 1. 基础统计
print('【1. 基础统计】')
print(f"已完成的组合数: {len(df):,}")
print(f"年化收益范围: {df['annual_return'].min():.2%} ~ {df['annual_return'].max():.2%}")
print(f"最大回撤范围: {df['max_drawdown'].min():.2%} ~ {df['max_drawdown'].max():.2%}")
print(f"总收益范围: {df['total_return'].min():.2%} ~ {df['total_return'].max():.2%}")
print()

# 2. 回撤约束筛选
valid = df[df['max_drawdown'] > -0.15].copy()
print('【2. 回撤约束筛选 (max_drawdown > -15%)】')
print(f"满足约束的组合: {len(valid)} / {len(df)} ({len(valid)/len(df)*100:.1f}%)")
if len(valid) > 0:
    print(f"满足约束的年化收益范围: {valid['annual_return'].min():.2%} ~ {valid['annual_return'].max():.2%}")
    print()
    
    # 3. 评分排序（收益/回撤比）
    valid['score'] = valid['annual_return'] / (-valid['max_drawdown'])
    top10 = valid.nlargest(10, 'score')
    print('【3. Top 10 组合（按 年化/回撤 评分）】')
    print(top10[['mom_w','vol_w','val_w','top_n','defensive_allocation','annual_return','max_drawdown','score']].to_string(index=False))
    print()
    
    # 4. 各参数分布分析
    print('【4. 各参数最优区间分析】')
    for col in ['mom_w', 'vol_w', 'val_w', 'top_n', 'defensive_allocation']:
        grp = valid.groupby(col)['annual_return'].agg(['mean','max','count']).round(4)
        grp.columns = ['平均年化','最大年化','数量']
        print(f"\n{col}:")
        print(grp.to_string())
    
    print('\n【5. 参数交叉分析: mom_w vs vol_w (平均年化)】')
    pivot = valid.pivot_table(values='annual_return', index='mom_w', columns='vol_w', aggfunc='mean')
    print(pivot.round(4).to_string())
    
    print('\n【6. top_n 对比】')
    tn = valid.groupby('top_n')['annual_return'].agg(['mean','max','std','count'])
    print(tn.round(4).to_string())
    
    print('\n【7. val_w 有效性分析】')
    vw = valid.groupby('val_w')['annual_return'].agg(['mean','max','std','count'])
    print(vw.round(4).to_string())
    
    print('\n【8. defensive_allocation 分析】')
    da = valid.groupby('defensive_allocation')['annual_return'].agg(['mean','max','std','count'])
    print(da.round(4).to_string())
    
    print('\n【9. 最优组合的共同特征 (Top 50 by annual_return)】')
    best = valid.nlargest(50, 'annual_return')
    print("Top 50组合中:")
    print(f"  mom_w 分布: {dict(sorted(best['mom_w'].value_counts().sort_index().items()))}")
    print(f"  vol_w 分布: {dict(sorted(best['vol_w'].value_counts().sort_index().items()))}")
    print(f"  val_w 分布: {dict(sorted(best['val_w'].value_counts().sort_index().items()))}")
    print(f"  top_n 分布: {dict(sorted(best['top_n'].value_counts().sort_index().items()))}")
    print(f"  def_alloc 分布: {dict(sorted(best['defensive_allocation'].value_counts().sort_index().items()))}")
    print(f"  平均回撤: {best['max_drawdown'].mean():.2%}")
    print(f"  平均防御周数占比: {best['defensive_weeks'].mean()/best['total_weeks'].mean()*100:.1f}%")
