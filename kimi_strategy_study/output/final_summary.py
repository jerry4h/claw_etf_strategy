import pandas as pd
df = pd.read_csv('output/param_grid_full.csv')
valid = df[df['max_drawdown'] > -0.15].copy()
valid['score'] = valid['annual_return'] / (-valid['max_drawdown'])
valid['ret_pct'] = valid['annual_return']*100
valid['dd_pct'] = valid['max_drawdown']*100

print('=== Full Grid Search: 11,000 Combinations ===')
print(f'Total: {len(df)}')
print(f'Valid (DD<15%): {len(valid)} ({len(valid)/len(df)*100:.1f}%)')
print(f'Invalid: {len(df)-len(valid)} ({(len(df)-len(valid))/len(df)*100:.1f}%)')
print()

print('=== val_w Distribution ===')
for v in sorted(df['val_w'].unique()):
    sub = df[df['val_w']==v]
    valid_sub = sub[sub['max_drawdown'] > -0.15]
    print(f'  val_w={v}: total={len(sub)}, valid={len(valid_sub)}, avg_ret={sub["annual_return"].mean()*100:.2f}%')
print()

print('=== Calmar Top 10 (All top_n) ===')
for _,r in valid.nlargest(10,'score').iterrows():
    print(f'  mom={r["mom_w"]:.2f} vol={r["vol_w"]:.1f} val={r["val_w"]:.1f} n={int(r["top_n"])} def={r["defensive_allocation"]:.2f} | {r["ret_pct"]:.2f}% / {r["dd_pct"]:.2f}% | Calmar={r["score"]:.3f}')
print()

n2 = valid[valid['top_n']==2]
print('=== Calmar Top 10 (top_n=2) ===')
for _,r in n2.nlargest(10,'score').iterrows():
    print(f'  mom={r["mom_w"]:.2f} vol={r["vol_w"]:.1f} val={r["val_w"]:.1f} def={r["defensive_allocation"]:.2f} | {r["ret_pct"]:.2f}% / {r["dd_pct"]:.2f}% | Calmar={r["score"]:.3f}')
print()

print('=== Annual Return Top 10 (top_n=2) ===')
for _,r in n2.nlargest(10,'annual_return').iterrows():
    print(f'  mom={r["mom_w"]:.2f} vol={r["vol_w"]:.1f} val={r["val_w"]:.1f} def={r["defensive_allocation"]:.2f} | {r["ret_pct"]:.2f}% / {r["dd_pct"]:.2f}% | Calmar={r["score"]:.3f}')
