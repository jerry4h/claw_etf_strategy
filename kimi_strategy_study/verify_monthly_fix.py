import pandas as pd
result = pd.read_csv('output/nav_history.csv', parse_dates=['date'])
result['year'] = result['date'].dt.year
result['month'] = result['date'].dt.month

monthly = []
for y in sorted(result['year'].unique()):
    for m in range(1, 13):
        mask = (result['year'] == y) & (result['month'] == m)
        if mask.sum() > 0:
            wrong = result.loc[mask, 'weekly_return'].prod() - 1
            correct = (1 + result.loc[mask, 'weekly_return']).prod() - 1
            monthly.append((y, m, wrong, correct))

print('Monthly return comparison (last 10 months):')
print('Year-Month   Wrong       Correct')
for y, m, w, c in monthly[-10:]:
    print(f'{y}-{m:02d}       {w:>+8.2%}   {c:>+8.2%}')

print()
wrong_pos = sum(1 for _, _, w, _ in monthly if w > 0)
correct_pos = sum(1 for _, _, _, c in monthly if c > 0)
print(f'Wrong formula positive months:  {wrong_pos}/{len(monthly)}')
print(f'Correct formula positive months: {correct_pos}/{len(monthly)}')
