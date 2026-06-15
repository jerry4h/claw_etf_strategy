"""
快速网格搜索 - 覆盖关键参数区域，约600组，3分钟内完成
"""
import pandas as pd
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from backtest import run_backtest

# 精简但覆盖最优区域的参数空间
params = [
    (mom_w, vol_w, val_w, top_n, def_alloc)
    for mom_w in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    for vol_w in [0.3, 0.5, 0.7, 0.9]
    for val_w in [0.0, 0.1, 0.2]
    for top_n in [1, 2, 3]
    for def_alloc in [0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
]
params = list(set(params))
print(f"总组合数: {len(params)}")

def run_single(p):
    mom_w, vol_w, val_w, top_n, def_alloc = p
    result = run_backtest(
        mom_w=mom_w, vol_w=vol_w, val_w=val_w,
        top_n=top_n, defensive_allocation=def_alloc
    )
    if result.empty:
        return None
    final_val = result.iloc[-1]["portfolio_value"]
    total_return = final_val - 1
    annual_return = (1 + total_return) ** (52 / len(result)) - 1
    max_dd = ((result["peak_value"] - result["portfolio_value"]) / result["peak_value"]).max()
    return {
        "mom_w": mom_w, "vol_w": vol_w, "val_w": val_w,
        "top_n": top_n, "defensive_allocation": def_alloc,
        "total_return": total_return, "annual_return": annual_return,
        "max_drawdown": -max_dd,
        "defensive_weeks": int(result["in_defensive"].sum()),
        "total_weeks": len(result)
    }

n_workers = max(1, int(multiprocessing.cpu_count() * 0.8))
print(f"使用 {n_workers} 个进程")

with multiprocessing.Pool(n_workers) as pool:
    results = list(tqdm(pool.imap(run_single, params), total=len(params), desc="Grid Search"))

results = [r for r in results if r is not None]
df = pd.DataFrame(results)
df = df.drop_duplicates(subset=['mom_w', 'vol_w', 'val_w', 'top_n', 'defensive_allocation'])

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
df.to_csv(output_dir / "param_grid_search.csv", index=False)

valid = df[(df['annual_return'] > 0.10) & (df['max_drawdown'] > -0.15)]
print(f"\n有效结果: {len(df)}")
print(f"满足约束(年化>10%, 回撤<15%): {len(valid)} 个")
print("\nTop 5:")
print(df.nlargest(5, 'annual_return')[['mom_w','vol_w','val_w','top_n','defensive_allocation','annual_return','max_drawdown']].to_string(index=False))
