"""
全量参数网格搜索
覆盖完整参数空间，使用多进程并行
支持增量保存与断点续跑
"""
import pandas as pd
import numpy as np
import multiprocessing
from pathlib import Path
import time
import os
from tqdm import tqdm

from backtest import run_backtest

# 完整参数空间定义
MOM_W = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
VOL_W = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
VAL_W = [0.0, 0.1, 0.2, 0.3, 0.4]
TOP_N = [1, 2]
DEF_ALLOC = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

BATCH_SIZE = 500  # 每多少组增量保存一次

print("=" * 60)
print("全量网格搜索（支持增量保存 / 断点续跑）")
print("=" * 60)
print(f"参数空间:")
print(f"  mom_w:       {MOM_W}")
print(f"  vol_w:       {VOL_W}")
print(f"  val_w:       {VAL_W}")
print(f"  top_n:       {TOP_N}")
print(f"  def_alloc:   {DEF_ALLOC}")

all_params = [
    (mom_w, vol_w, val_w, top_n, def_alloc)
    for mom_w in MOM_W
    for vol_w in VOL_W
    for val_w in VAL_W
    for top_n in TOP_N
    for def_alloc in DEF_ALLOC
]
all_params = list(set(all_params))  # 去重
print(f"\n总组合数: {len(all_params)}")

n_workers = max(1, int(multiprocessing.cpu_count() * 0.5))
print(f"使用 {n_workers} 个进程 (总CPU核心: {multiprocessing.cpu_count()})")

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "param_grid_full.csv"

# ============================================================
# 断点续跑逻辑
# ============================================================
completed = set()
if output_path.exists():
    try:
        existing = pd.read_csv(output_path)
        # 检查是否是旧版本（含 top_n=3 或空文件）
        if existing.empty or (3 in existing['top_n'].values):
            print(f"\n发现旧版本/空结果文件，删除后重新开始...")
            os.remove(output_path)
        else:
            for _, row in existing.iterrows():
                completed.add((
                    float(row['mom_w']), float(row['vol_w']), float(row['val_w']),
                    int(row['top_n']), float(row['defensive_allocation'])
                ))
            print(f"\n断点续跑：已跳过 {len(completed)} 个已完成组合")
    except Exception as e:
        print(f"\n读取已有结果失败: {e}，删除后重新开始...")
        os.remove(output_path)

if completed:
    all_params = [p for p in all_params if p not in completed]
    print(f"实际需搜索: {len(all_params)} 种组合")

# ============================================================

def run_single(params):
    mom_w, vol_w, val_w, top_n, def_alloc = params
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
    defensive_days = result["in_defensive"].sum()
    total_days = len(result)
    return {
        "mom_w": mom_w,
        "vol_w": vol_w,
        "val_w": val_w,
        "top_n": top_n,
        "defensive_allocation": def_alloc,
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": -max_dd,
        "defensive_weeks": int(defensive_days),
        "total_weeks": total_days
    }


def save_batch(results_buffer, output_path, first_write):
    """将一批结果追加/写入 CSV"""
    if not results_buffer:
        return first_write
    df_batch = pd.DataFrame(results_buffer)
    if first_write:
        df_batch.to_csv(output_path, index=False)
        return False
    else:
        df_batch.to_csv(output_path, mode='a', header=False, index=False)
        return False


if __name__ == '__main__':
    start = time.time()
    results_buffer = []
    first_write = not output_path.exists()
    processed = 0
    saved_count = len(completed)

    try:
        with multiprocessing.Pool(n_workers) as pool:
            for i, r in enumerate(tqdm(
                pool.imap(run_single, all_params),
                total=len(all_params),
                desc="Grid Search"
            )):
                if r is not None:
                    results_buffer.append(r)
                processed += 1

                # 增量保存
                if processed % BATCH_SIZE == 0 and results_buffer:
                    first_write = save_batch(results_buffer, output_path, first_write)
                    saved_count += len(results_buffer)
                    results_buffer = []
                    elapsed = time.time() - start
                    print(f"\n  [增量保存] 已完成 {saved_count}/{len(all_params)+len(completed)} 组"
                          f" | 耗时 {elapsed/60:.1f}min | 速度 {saved_count/(elapsed/60):.0f} 组/min")

    except Exception as e:
        print(f"\n多进程失败: {e}，切换为串行")
        for i, params in enumerate(tqdm(all_params, desc="Grid Search")):
            r = run_single(params)
            if r is not None:
                results_buffer.append(r)
            processed += 1

            if processed % BATCH_SIZE == 0 and results_buffer:
                first_write = save_batch(results_buffer, output_path, first_write)
                saved_count += len(results_buffer)
                results_buffer = []

    # 保存最后一批
    if results_buffer:
        first_write = save_batch(results_buffer, output_path, first_write)
        saved_count += len(results_buffer)

    elapsed = time.time() - start
    print(f"\n搜索完成，本次耗时 {elapsed/60:.1f} 分钟")
    print(f"累计有效结果: {saved_count}")
    print(f"保存至: {output_path}")

    # ============================================================
    # 最终统计（从 CSV 加载完整数据）
    # ============================================================
    df = pd.read_csv(output_path)
    df = df.drop_duplicates(subset=['mom_w', 'vol_w', 'val_w', 'top_n', 'defensive_allocation'])

    # 统计
    valid = df[(df['annual_return'] > 0.10) & (df['max_drawdown'] > -0.15)]
    print(f"\n满足约束(年化>10%, 回撤<15%): {len(valid)} 个组合")

    # 按 top_n 分组统计
    print("\n" + "=" * 60)
    print("按 top_n 统计:")
    print("=" * 60)
    for n in sorted(df['top_n'].unique()):
        sub = df[df['top_n'] == n]
        sub_valid = sub[(sub['annual_return'] > 0.10) & (sub['max_drawdown'] > -0.15)]
        print(f"top_n={n}: 总组合 {len(sub)}, 满足约束 {len(sub_valid)},"
              f" 平均年化 {sub['annual_return'].mean()*100:.2f}%,"
              f" 平均回撤 {-sub['max_drawdown'].mean()*100:.2f}%")

    # Top 10
    print("\n" + "=" * 60)
    print("年化收益 Top 10:")
    print("=" * 60)
    top10 = df.nlargest(10, 'annual_return')
    print(top10[['mom_w', 'vol_w', 'val_w', 'top_n', 'defensive_allocation', 'annual_return', 'max_drawdown']].to_string(index=False))

    # Top 10 回撤最小
    print("\n" + "=" * 60)
    print("最大回撤 Top 10 (从小到大):")
    print("=" * 60)
    bot10 = df.nsmallest(10, 'max_drawdown')
    print(bot10[['mom_w', 'vol_w', 'val_w', 'top_n', 'defensive_allocation', 'annual_return', 'max_drawdown']].to_string(index=False))

    # 综合评分 Top 10 (年化/回撤)
    df['score'] = df['annual_return'] / (-df['max_drawdown'])
    print("\n" + "=" * 60)
    print("综合评分 Top 10 (年化/回撤):")
    print("=" * 60)
    top_score = df.nlargest(10, 'score')
    print(top_score[['mom_w', 'vol_w', 'val_w', 'top_n', 'defensive_allocation', 'annual_return', 'max_drawdown', 'score']].to_string(index=False))
