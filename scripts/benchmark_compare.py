#!/usr/bin/env python3
"""基准对比 — 策略 vs 每周再平衡等权 vs 真·买入持有(一次不调仓)。

三方净值全部对齐到策略有效区间、在共同起点归一化到 1.0，并用同一个
`src.backtest.compute_metrics` + 同一无风险利率计算，确保口径完全一致、可复现。

- 策略：官方引擎 run_backtest。
- 每周再平衡等权：每周把权重拉回 1/N（N=起点有效 ETF 数）。
- 真·买入持有：起点各 1/N，此后永不调仓，权重随价格自然漂移（停牌沿用上一价）。

用法:
  python scripts/benchmark_compare.py            # 全期(in-sample) + OOS(2024+) 两段
  python scripts/benchmark_compare.py --json     # JSON 输出（供程序化/回归测试）
  python scripts/benchmark_compare.py --start 2024-01-01   # 仅指定单段
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
_VENV = PROJECT / '.venv' / 'lib' / 'python3.12' / 'site-packages'
if _VENV.exists():
    sys.path.insert(0, str(_VENV))
sys.path.insert(0, str(PROJECT))

from src.strategy import load_config
from src.backtest import run_backtest, compute_metrics
from src.data_loader import ETFS, load_nav_data, resample_weekly


def _nav_frame(nav, idx):
    """把一维净值序列包成 compute_metrics 需要的 DataFrame（基准无防御/换手）。"""
    nav = np.asarray(nav, float)
    wr = np.zeros(len(nav))
    wr[1:] = nav[1:] / nav[:-1] - 1
    peak = np.maximum.accumulate(nav)
    dd = (peak - nav) / peak
    return pd.DataFrame({'nav': nav, 'weekly_return': wr, 'drawdown': dd,
                         'def_ratio': 0.0, 'turnover': 0.0}, index=idx)


def compute_benchmarks(cfg, start_date=None):
    """返回 {'window', 'strategy', 'ew_rebalanced', 'buy_hold'}，三方指标同口径。"""
    rf = cfg.risk_free_rate
    res = run_backtest(cfg, start_date=start_date)
    strat = res.nav_series.copy()
    start, end = strat.index[0], strat.index[-1]

    csv = cfg.nav_path if Path(cfg.nav_path).is_absolute() else str(PROJECT / cfg.nav_path)
    wn = resample_weekly(load_nav_data(csv), anchor=cfg.anchor)
    cols = [c for c in wn.columns if c in ETFS]
    pr = wn.loc[start:end, cols].astype(float)
    idx = pr.index
    valid = ~np.isnan(pr.iloc[0].values)
    n_valid = int(valid.sum())

    # 真·买入持有：权重漂移
    growth = pr.ffill().values / pr.ffill().iloc[0].values
    w0 = np.where(valid, 1.0 / n_valid, 0.0)
    bh = (growth * w0).sum(axis=1)
    bh = bh / bh[0]

    # 每周再平衡等权
    er = pr.ffill().pct_change().fillna(0.0).values
    rb = np.ones(len(idx))
    for i in range(1, len(idx)):
        rb[i] = rb[i - 1] * (1 + float(np.mean(er[i, valid])))

    return {
        'window': {'start': str(start.date()), 'end': str(end.date()),
                   'weeks': len(idx), 'n_valid_etf': n_valid},
        'strategy': compute_metrics(strat, rf),
        'ew_rebalanced': compute_metrics(_nav_frame(rb, idx), rf),
        'buy_hold': compute_metrics(_nav_frame(bh, idx), rf),
    }


def _fmt(label, m):
    return (f"{label:<26s} {m['total_return']*100:>8.1f}% {m['annual_return']*100:>7.2f}% "
            f"{m['max_drawdown']*100:>7.2f}% {m['sharpe_ratio']:>7.3f} {m['calmar_ratio']:>6.2f} "
            f"{m['annual_volatility']*100:>6.2f}% {m['win_rate']*100:>5.1f}%")


def _print_block(label, b):
    w = b['window']
    print(f"\n===== {label}: {w['start']} -> {w['end']} ({w['weeks']} weeks) =====")
    print(f"{'':<26s} {'CumRet':>9s} {'AnnRet':>7s} {'MaxDD':>8s} {'Sharpe':>7s} "
          f"{'Calmar':>6s} {'Vol':>7s} {'Win':>6s}")
    print('-' * 90)
    print(_fmt('Strategy (weekly rotate)', b['strategy']))
    print(_fmt('EW weekly-rebalanced', b['ew_rebalanced']))
    print(_fmt('TRUE buy&hold (no touch)', b['buy_hold']))


def main():
    p = argparse.ArgumentParser(description='策略 vs 每周再平衡等权 vs 真·买入持有 基准对比')
    p.add_argument('--json', action='store_true', help='JSON 输出')
    p.add_argument('--start', type=str, default=None,
                   help='指定单段回测起始日；缺省输出 全期 + OOS(2024-01-01) 两段')
    args = p.parse_args()

    cfg = load_config(PROJECT / 'config/strategy_v3_1.yaml')
    if args.start:
        blocks = {'CUSTOM': compute_benchmarks(cfg, start_date=args.start)}
    else:
        blocks = {
            'FULL (in-sample; strategy params fit on this window)': compute_benchmarks(cfg),
            'OOS (2024+; strategy params NOT refit here)': compute_benchmarks(cfg, start_date='2024-01-01'),
        }

    if args.json:
        print(json.dumps(blocks, ensure_ascii=False, indent=2, default=str))
    else:
        for label, b in blocks.items():
            _print_block(label, b)


if __name__ == '__main__':
    main()
