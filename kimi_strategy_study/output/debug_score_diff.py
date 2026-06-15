import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from data_loader import load_nav_data, ETFS
from factors import calculate_all_factors
from strategy import calculate_composite_score

nav_df = load_nav_data()
ret_df = nav_df.pct_change().dropna()
factors = calculate_all_factors(nav_df, ret_df, 20, 20, 60)

weekly_dates = nav_df.resample("W-MON").indices
weekly_dates = sorted(weekly_dates.keys())

OFFENSIVE = ["纳指ETF", "沪深300ETF", "黄金ETF"]

# 模拟3选2切换，统计评分差异
score_diffs = []  # 每次切换时，新进入标的 vs 退出标的的评分差
prev_top = []

for i, date in enumerate(weekly_dates[:-1]):
    scores = calculate_composite_score(
        factors["momentum"].loc[:date],
        factors["volatility"].loc[:date],
        factors["valuation"].loc[:date],
        ETFS, mom_w=0.30, vol_w=0.4, val_w=0.0
    )
    if scores.empty:
        continue

    score_row = scores.iloc[-1][OFFENSIVE]
    top2 = score_row.nlargest(2).index.tolist()

    if prev_top and set(top2) != set(prev_top):
        # 有切换发生
        new_entries = [e for e in top2 if e not in prev_top]
        exits = [e for e in prev_top if e not in top2]

        for new_etf, exit_etf in zip(new_entries, exits):
            new_score = score_row[new_etf]
            exit_score = score_row[exit_etf]
            diff_pct = (new_score - exit_score) / abs(exit_score) if exit_score != 0 else 999
            score_diffs.append({
                "date": date.strftime("%Y-%m-%d"),
                "exit": exit_etf,
                "exit_score": exit_score,
                "new": new_etf,
                "new_score": new_score,
                "diff_pct": diff_pct,
            })

    prev_top = top2

print(f"总共发生 {len(score_diffs)} 次标的切换")
print(f"评分差异统计:")
diffs = [d["diff_pct"] for d in score_diffs]
print(f"  平均: {sum(diffs)/len(diffs)*100:.1f}%")
print(f"  中位数: {sorted(diffs)[len(diffs)//2]*100:.1f}%")
print(f"  最小: {min(diffs)*100:.1f}%")
print(f"  最大: {max(diffs)*100:.1f}%")
print(f"  <5%: {sum(1 for d in diffs if d < 0.05)}次")
print(f"  <10%: {sum(1 for d in diffs if d < 0.10)}次")
print(f"  <20%: {sum(1 for d in diffs if d < 0.20)}次")
print(f"  <30%: {sum(1 for d in diffs if d < 0.30)}次")

print("\n前20次切换的评分差异（最小差异优先）:")
for d in sorted(score_diffs, key=lambda x: x["diff_pct"])[:20]:
    print(f"  {d['date']}: {d['exit']}({d['exit_score']:.2f}) → {d['new']}({d['new_score']:.2f}) 差异={d['diff_pct']*100:.1f}%")
