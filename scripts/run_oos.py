"""真样本外(OOS)验证 — 冻结训练窗选参，测试窗不重拟合。

方法：
  - 训练窗 2013-05-17 ~ 2023-12-31：在活跃自由参数网格上选最优（最高 DSR，按网格 N 试次矫正以去选择偏差；DD<15% 约束）
  - 测试窗 2024-01-01 ~ 2026-07-25：直接用训练窗选出的最优参数跑，绝不重拟合
  - 同时报告 FULL(全期, 最优参数) 与 BASELINE(生产参数, 全期) 作参照

注：网格刻意保持小（5 参数 × 3 水平 = 243 组合），以限制选择偏差。
"""
import sys
import itertools
import dataclasses
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.strategy import load_config
from src.backtest import run_backtest
from src.robustness import compute_dsr

cfg = load_config(PROJECT / 'config' / 'strategy_v4_3.yaml')

TRAIN_START, TRAIN_END = '2013-05-17', '2023-12-31'
TEST_START, TEST_END = '2024-01-01', cfg.end_date or '2026-07-25'

grid = {
    'mom_w': [0.8, 1.0, 1.2],
    'vol_w': [0.9, 1.1, 1.3],
    'def_alloc': [0.20, 0.25, 0.30],
    'step_low': [0.10, 0.15, 0.20],
    'step_high': [0.30, 0.35, 0.40],
}
keys = list(grid)
combos = list(itertools.product(*[grid[k] for k in keys]))
print(f"[OOS] grid={len(combos)} combos | TRAIN {TRAIN_START}..{TRAIN_END} | TEST {TEST_START}..{TEST_END}")

best = None
n_eval = 0
for vals in combos:
    kw = dict(zip(keys, vals))
    c = dataclasses.replace(cfg, **kw)
    res = run_backtest(c, start_date=TRAIN_START, end_date=TRAIN_END)
    if res.nav_series.empty:
        continue
    n_eval += 1
    m = res.metrics
    if m['max_drawdown'] >= 0.15:
        continue
    # 选参去偏：用网格 N 试次对 Sharpe 做 DSR 矫正，消除 243 选 1 的多重检验选择偏差
    dsr = compute_dsr(m['sharpe_ratio'], n_trials=len(combos), n_obs=len(res.nav_series), periods_per_year=52)
    if best is None or dsr > best['dsr']:
        best = {'kw': kw, 'sharpe': m['sharpe_ratio'], 'dsr': dsr,
                'ann': m['annual_return'], 'dd': m['max_drawdown'],
                'n': len(res.nav_series)}

# 兜底：若全部 DD>=0.15，取最高 Sharpe
if best is None:
    for vals in combos:
        kw = dict(zip(keys, vals))
        c = dataclasses.replace(cfg, **kw)
        res = run_backtest(c, start_date=TRAIN_START, end_date=TRAIN_END)
        if res.nav_series.empty:
            continue
        m = res.metrics
        dsr = compute_dsr(m['sharpe_ratio'], n_trials=len(combos), n_obs=len(res.nav_series), periods_per_year=52)
        if best is None or dsr > best['dsr']:
            best = {'kw': kw, 'sharpe': m['sharpe_ratio'], 'dsr': dsr,
                    'ann': m['annual_return'], 'dd': m['max_drawdown'],
                    'n': len(res.nav_series)}

print(f"[OOS] TRAIN best params: {best['kw']}")
print(f"[OOS] TRAIN best: Sharpe={best['sharpe']:.3f} ann={best['ann']*100:.2f}% "
      f"DD={best['dd']*100:.2f}% n={best['n']}")

# 测试窗：直接用训练窗最优参数，不重拟合
bo = dataclasses.replace(cfg, **best['kw'])
res_oos = run_backtest(bo, start_date=TEST_START, end_date=TEST_END)
mo = res_oos.metrics
print(f"[OOS] TEST (no refit): Sharpe={mo['sharpe_ratio']:.3f} "
      f"ann={mo['annual_return']*100:.2f}% DD={mo['max_drawdown']*100:.2f}% n={len(res_oos.nav_series)}")

# 参照
res_full = run_backtest(bo)
print(f"[OOS] FULL(best params, 全期): Sharpe={res_full.metrics['sharpe_ratio']:.3f} "
      f"ann={res_full.metrics['annual_return']*100:.2f}% DD={res_full.metrics['max_drawdown']*100:.2f}%")
res_base = run_backtest(cfg)
print(f"[OOS] BASELINE(prod params, 全期): Sharpe={res_base.metrics['sharpe_ratio']:.3f} "
      f"ann={res_base.metrics['annual_return']*100:.2f}% DD={res_base.metrics['max_drawdown']*100:.2f}%")

# 退化度
deg = best['sharpe'] - mo['sharpe_ratio']
print(f"[OOS] IS->OOS Sharpe 退化: {deg:+.3f}  "
      f"({'OOS 接近 IS，过拟合轻' if deg < 0.3 else 'OOS 明显低于 IS，存在过拟合信号'})")
