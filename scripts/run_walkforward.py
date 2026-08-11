#!/usr/bin/env python3
"""Walk-Forward 验证脚本 — 评估策略的样本外泛化能力。

四种模式，务必分清各自能证明什么：

  --reoptimize  【真·防过拟合样本外验证】每个滚动窗口用 anchored train 段
                (数据起点→窗口开始) 在参数网格上重新选参(DSR选优, DD<15%)，
                再拿选出的超参到 test 窗验证，绝不重拟合。train/test 时间不重叠、
                参数不含测试期信息。这是唯一能回答"参数是否过拟合"的模式。
  --benchmark   【固定参数稳定性对比，非防过拟合】用同一套生产参数在各 test 窗
                对比 策略 / 每周再平衡等权 / 真·买入持有。它衡量"固定策略相对
                基准的优势在不同市况稳不稳"，但因参数是在含这些窗口的全历史上
                调出的，不能证明参数未过拟合。
  --rolling     固定参数在各窗口的 train/test Sharpe 衰减(同样是固定参数)。
  (默认)         单次 train/test 切分(固定参数)。

用法:
  python scripts/run_walkforward.py --reoptimize --windows 10   # 真 OOS，每窗重选参
  python scripts/run_walkforward.py --benchmark --windows 10    # 固定参数 vs 基准
  python scripts/run_walkforward.py --rolling                   # 固定参数滚动衰减
  python scripts/run_walkforward.py --json                      # JSON 输出
"""

import argparse
import dataclasses
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.backtest import run_backtest
from src.strategy import load_config, StrategyConfig
from src.data_loader import load_nav_data, resample_weekly
from src.robustness import compute_dsr
from scripts.benchmark_compare import compute_benchmarks

# 参数选优网格（与 run_oos.py 一致：5 参数 × 3 水平 = 243 组合，刻意小以限制选择偏差）
DEFAULT_WF_GRID = {
    'mom_w': [0.8, 1.0, 1.2],
    'vol_w': [0.9, 1.1, 1.3],
    'def_alloc': [0.20, 0.25, 0.30],
    'step_low': [0.10, 0.15, 0.20],
    'step_high': [0.30, 0.35, 0.40],
}


def _run_segment(cfg: StrategyConfig, start: str, end: str) -> dict | None:
    """Run backtest on a date segment, return metrics or None."""
    try:
        r = run_backtest(cfg, start_date=start, end_date=end)
        if r.nav_series.empty:
            return None
        return {
            'start': start,
            'end': end,
            'sharpe': r.metrics['sharpe_ratio'],
            'annual_return': r.metrics['annual_return'],
            'max_drawdown': r.metrics['max_drawdown'],
            'calmar': r.metrics['calmar_ratio'],
            'win_rate': r.metrics['win_rate'],
            'weeks': r.metrics['total_weeks'],
            'final_nav': r.metrics['final_nav'],
        }
    except Exception as e:
        print(f"  [WARN] Backtest failed for {start} ~ {end}: {e}")
        return None


def single_split_wf(cfg: StrategyConfig, train_ratio: float = 0.6) -> dict:
    """Single train/test split walk-forward.

    Args:
        cfg: Strategy config.
        train_ratio: Fraction of data for training (default 0.6).

    Returns:
        Dict with train/test metrics and degradation analysis.
    """
    # Get full date range
    nav_path = cfg.nav_path if Path(cfg.nav_path).is_absolute() else str(PROJECT / cfg.nav_path)
    nav_df = load_nav_data(nav_path)
    weekly = resample_weekly(nav_df, anchor=cfg.anchor)
    if cfg.start_date:
        weekly = weekly[weekly.index >= pd.to_datetime(cfg.start_date)]

    dates = weekly.index
    n = len(dates)
    split_idx = int(n * train_ratio)
    split_date = dates[split_idx]

    full_start = str(dates[0].date())
    split_str = str(split_date.date())
    full_end = str(dates[-1].date())

    # Run train and test segments
    train = _run_segment(cfg, full_start, split_str)
    test = _run_segment(cfg, split_str, full_end)
    full = _run_segment(cfg, full_start, full_end)

    if not all([train, test, full]):
        return {'error': 'One or more segments failed'}

    # Compute degradation
    sharpe_degradation = train['sharpe'] - test['sharpe'] if train and test else 0
    sharpe_ratio_pct = (test['sharpe'] / train['sharpe'] * 100) if train['sharpe'] > 0 else 0

    return {
        'mode': 'single_split',
        'train_ratio': train_ratio,
        'split_date': split_str,
        'train': train,
        'test': test,
        'full': full,
        'sharpe_degradation': sharpe_degradation,
        'sharpe_test_over_train_pct': sharpe_ratio_pct,
        'overfitting_flag': sharpe_degradation > 0.5,  # > 0.5 Sharpe drop = warning
    }


def rolling_wf(cfg: StrategyConfig, n_windows: int = 5) -> dict:
    """Rolling walk-forward: split data into n_windows consecutive segments,
    train on all prior windows, test on current window.

    Args:
        cfg: Strategy config.
        n_windows: Number of consecutive windows (default 5).

    Returns:
        Dict with per-window results and aggregate statistics.
    """
    nav_path = cfg.nav_path if Path(cfg.nav_path).is_absolute() else str(PROJECT / cfg.nav_path)
    nav_df = load_nav_data(nav_path)
    weekly = resample_weekly(nav_df, anchor=cfg.anchor)
    if cfg.start_date:
        weekly = weekly[weekly.index >= pd.to_datetime(cfg.start_date)]

    dates = weekly.index
    n = len(dates)
    window_size = n // n_windows

    results = []
    for i in range(1, n_windows):
        train_end_idx = window_size * i
        train_start = str(dates[0].date())
        train_end = str(dates[train_end_idx].date())
        test_start = train_end
        test_end = str(dates[min(train_end_idx + window_size, n - 1)].date())

        train_m = _run_segment(cfg, train_start, train_end)
        test_m = _run_segment(cfg, test_start, test_end)

        if train_m and test_m:
            results.append({
                'window': i,
                'train_end': train_end,
                'train_sharpe': train_m['sharpe'],
                'test_sharpe': test_m['sharpe'],
                'train_return': train_m['annual_return'],
                'test_return': test_m['annual_return'],
                'train_dd': train_m['max_drawdown'],
                'test_dd': test_m['max_drawdown'],
                'sharpe_degradation': train_m['sharpe'] - test_m['sharpe'],
            })

    if not results:
        return {'error': 'All windows failed'}

    degradations = [r['sharpe_degradation'] for r in results]
    test_sharpes = [r['test_sharpe'] for r in results]

    return {
        'mode': 'rolling',
        'n_windows': n_windows,
        'windows': results,
        'avg_test_sharpe': float(np.mean(test_sharpes)),
        'std_test_sharpe': float(np.std(test_sharpes)),
        'avg_degradation': float(np.mean(degradations)),
        'max_degradation': float(np.max(degradations)),
        'n_negative_test': sum(1 for s in test_sharpes if s < 0),
    }


def benchmark_wf(cfg: StrategyConfig, n_windows: int = 5) -> dict:
    """Walk-forward 逐窗口 vs 两个「什么都不做」基准。

    把数据切为 n_windows 段；对每个 test 窗口（共 n_windows-1 个），
    用同一 [start, end] 跑策略 + 每周再平衡等权 + 真·买入持有，
    三方指标同口径（benchmark_compare.compute_benchmarks）。

    注意：本模式用【固定生产参数】，衡量的是"固定策略相对基准的稳定性"，
    不是防过拟合的样本外验证（参数在含这些窗口的全历史上调过）。
    要做真 OOS 请用 reoptimize_wf / --reoptimize。
    """
    nav_path = cfg.nav_path if Path(cfg.nav_path).is_absolute() else str(PROJECT / cfg.nav_path)
    nav_df = load_nav_data(nav_path)
    weekly = resample_weekly(nav_df, anchor=cfg.anchor)
    if cfg.start_date:
        weekly = weekly[weekly.index >= pd.to_datetime(cfg.start_date)]
    dates = weekly.index
    n = len(dates)
    window_size = n // n_windows

    results = []
    for i in range(1, n_windows):
        idx_end = window_size * i
        test_start = str(dates[idx_end].date())
        test_end = str(dates[min(idx_end + window_size, n - 1)].date())
        b = compute_benchmarks(cfg, start_date=test_start, end_date=test_end)
        sm = b['strategy']; rbm = b['ew_rebalanced']; bhm = b['buy_hold']
        results.append({
            'window': i,
            'test_start': test_start,
            'test_end': test_end,
            'weeks': b['window']['weeks'],
            'strategy_sharpe': sm['sharpe_ratio'],
            'ew_rebal_sharpe': rbm['sharpe_ratio'],
            'buyhold_sharpe': bhm['sharpe_ratio'],
            'strategy_annual': sm['annual_return'],
            'ew_rebal_annual': rbm['annual_return'],
            'buyhold_annual': bhm['annual_return'],
            'strategy_dd': sm['max_drawdown'],
            'ew_rebal_dd': rbm['max_drawdown'],
            'buyhold_dd': bhm['max_drawdown'],
            'vs_ew_rebal': sm['sharpe_ratio'] > rbm['sharpe_ratio'],
            'vs_buyhold': sm['sharpe_ratio'] > bhm['sharpe_ratio'],
        })

    vs_ew = sum(r['vs_ew_rebal'] for r in results)
    vs_bh = sum(r['vs_buyhold'] for r in results)
    return {
        'mode': 'benchmark_wf',
        'n_windows': n_windows,
        'n_test_windows': len(results),
        'windows': results,
        'wins_vs_ew_rebal': vs_ew,
        'wins_vs_buyhold': vs_bh,
        'avg_strategy_sharpe': float(np.mean([r['strategy_sharpe'] for r in results])),
        'avg_ew_rebal_sharpe': float(np.mean([r['ew_rebal_sharpe'] for r in results])),
        'avg_buyhold_sharpe': float(np.mean([r['buyhold_sharpe'] for r in results])),
    }


def _select_best_on_train(cfg: StrategyConfig, train_start: str, train_end: str,
                          grid: dict, dd_cap: float = 0.15) -> dict | None:
    """在 train 段网格搜索，用 DSR 选优（DD<dd_cap 约束），返回 best 参数。

    严格无泄露：只在 [train_start, train_end] 上回测选参，从不接触测试期。
    DSR(n_trials=网格大小) 对多重检验做矫正，抑制 243 选 1 的选择偏差。
    兜底：若所有组合都被 DD 约束排除，则退化为无约束取最高 Sharpe。
    """
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys]))
    n_trials = len(combos)

    def _eval(apply_dd_cap: bool):
        best = None
        for vals in combos:
            kw = dict(zip(keys, vals))
            c = dataclasses.replace(cfg, **kw)
            res = run_backtest(c, start_date=train_start, end_date=train_end)
            if res.nav_series.empty:
                continue
            m = res.metrics
            if apply_dd_cap and m['max_drawdown'] >= dd_cap:
                continue
            dsr = compute_dsr(m['sharpe_ratio'], n_trials=n_trials, n_obs=len(res.nav_series), periods_per_year=52)
            cand = {'kw': kw, 'train_sharpe': m['sharpe_ratio'],
                    'train_dd': m['max_drawdown'], 'dsr': dsr}
            if best is None or dsr > best['dsr']:
                best = cand
        return best

    return _eval(apply_dd_cap=True) or _eval(apply_dd_cap=False)


def reoptimize_wf(cfg: StrategyConfig, n_windows: int = 10, grid: dict | None = None) -> dict:
    """真·滚动 Walk-Forward（每窗重新选参）—— 唯一能回答"参数是否过拟合"的模式。

    对每个 test 窗 i：
      1. anchored train = [数据起点, test 窗开始]，在参数网格上 DSR 选优(DD<15%)选超参；
      2. 用选出的超参在 test 窗 [窗口开始, 窗口结束] 跑，绝不重拟合；
      3. test 窗同步对比每周再平衡等权 / 真·买入持有(与参数无关的两个基准)。
    train 与 test 时间不重叠、选参不含测试期信息 → 真正的样本外。
    """
    grid = grid or DEFAULT_WF_GRID
    nav_path = cfg.nav_path if Path(cfg.nav_path).is_absolute() else str(PROJECT / cfg.nav_path)
    nav_df = load_nav_data(nav_path)
    weekly = resample_weekly(nav_df, anchor=cfg.anchor)
    if cfg.start_date:
        weekly = weekly[weekly.index >= pd.to_datetime(cfg.start_date)]
    dates = weekly.index
    n = len(dates)
    window_size = n // n_windows

    results = []
    for i in range(1, n_windows):
        idx_end = window_size * i
        train_start = str(dates[0].date())
        train_end = str(dates[idx_end].date())
        test_start = train_end
        test_end = str(dates[min(idx_end + window_size, n - 1)].date())

        best = _select_best_on_train(cfg, train_start, train_end, grid)
        if best is None:
            continue
        bo = dataclasses.replace(cfg, **best['kw'])
        b = compute_benchmarks(bo, start_date=test_start, end_date=test_end)
        sm, rbm, bhm = b['strategy'], b['ew_rebalanced'], b['buy_hold']
        results.append({
            'window': i,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'weeks': b['window']['weeks'],
            'best_params': best['kw'],
            'train_sharpe': best['train_sharpe'],
            'test_strategy_sharpe': sm['sharpe_ratio'],
            'sharpe_degradation': best['train_sharpe'] - sm['sharpe_ratio'],
            'ew_rebal_sharpe': rbm['sharpe_ratio'],
            'buyhold_sharpe': bhm['sharpe_ratio'],
            'strategy_annual': sm['annual_return'],
            'ew_rebal_annual': rbm['annual_return'],
            'buyhold_annual': bhm['annual_return'],
            'strategy_dd': sm['max_drawdown'],
            'vs_ew_rebal': sm['sharpe_ratio'] > rbm['sharpe_ratio'],
            'vs_buyhold': sm['sharpe_ratio'] > bhm['sharpe_ratio'],
        })

    if not results:
        return {'error': 'All windows failed', 'mode': 'reoptimize_wf'}
    nw = len(results)
    return {
        'mode': 'reoptimize_wf',
        'n_windows': n_windows,
        'n_test_windows': nw,
        'windows': results,
        'wins_vs_ew_rebal': sum(r['vs_ew_rebal'] for r in results),
        'wins_vs_buyhold': sum(r['vs_buyhold'] for r in results),
        'avg_train_sharpe': float(np.mean([r['train_sharpe'] for r in results])),
        'avg_test_strategy_sharpe': float(np.mean([r['test_strategy_sharpe'] for r in results])),
        'avg_sharpe_degradation': float(np.mean([r['sharpe_degradation'] for r in results])),
        'avg_ew_rebal_sharpe': float(np.mean([r['ew_rebal_sharpe'] for r in results])),
        'avg_buyhold_sharpe': float(np.mean([r['buyhold_sharpe'] for r in results])),
    }



def fmt_report(result: dict) -> str:
    """Format walk-forward result as human-readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append(" Walk-Forward 验证报告")
    lines.append("=" * 60)

    if 'error' in result:
        lines.append(f" ERROR: {result['error']}")
        return '\n'.join(lines)

    if result['mode'] == 'single_split':
        tr = result['train']
        te = result['test']
        fu = result['full']
        lines.append(f" 切分比例: {result['train_ratio']*100:.0f}% / {(1-result['train_ratio'])*100:.0f}%")
        lines.append(f" 切分日期: {result['split_date']}")
        lines.append("")
        lines.append(f" {'指标':<12s} {'Train':>10s} {'Test':>10s} {'Full':>10s}")
        lines.append(f" {'─'*45}")
        lines.append(f" {'Sharpe':<12s} {tr['sharpe']:>10.4f} {te['sharpe']:>10.4f} {fu['sharpe']:>10.4f}")
        lines.append(f" {'年化收益':<10s} {tr['annual_return']*100:>9.2f}% {te['annual_return']*100:>9.2f}% {fu['annual_return']*100:>9.2f}%")
        lines.append(f" {'最大回撤':<10s} {tr['max_drawdown']*100:>9.2f}% {te['max_drawdown']*100:>9.2f}% {fu['max_drawdown']*100:>9.2f}%")
        lines.append(f" {'Calmar':<12s} {tr['calmar']:>10.2f} {te['calmar']:>10.2f} {fu['calmar']:>10.2f}")
        lines.append(f" {'周数':<12s} {tr['weeks']:>10d} {te['weeks']:>10d} {fu['weeks']:>10d}")
        lines.append("")

        deg = result['sharpe_degradation']
        pct = result['sharpe_test_over_train_pct']
        flag = result['overfitting_flag']
        lines.append(f" Sharpe 衰减: {deg:+.4f} (Test/Train = {pct:.1f}%)")
        if flag:
            lines.append(f" ⚠️  过拟合警告: Sharpe 衰减 > 0.5，样本外表现显著下降")
        else:
            lines.append(f" ✅ 无明显过拟合 (衰减 < 0.5)")

    elif result['mode'] == 'rolling':
        lines.append(f" 滚动窗口数: {result['n_windows']}")
        lines.append("")
        lines.append(f" {'Window':<8s} {'Train End':>12s} {'Train Sh':>10s} {'Test Sh':>10s} {'Degrad':>10s}")
        lines.append(f" {'─'*55}")
        for w in result['windows']:
            lines.append(
                f" {w['window']:<8d} {w['train_end']:>12s} "
                f"{w['train_sharpe']:>10.4f} {w['test_sharpe']:>10.4f} "
                f"{w['sharpe_degradation']:>+10.4f}"
            )
        lines.append("")
        lines.append(f" 平均 Test Sharpe: {result['avg_test_sharpe']:.4f} ± {result['std_test_sharpe']:.4f}")
        lines.append(f" 平均 Sharpe 衰减: {result['avg_degradation']:+.4f}")
        lines.append(f" 最大 Sharpe 衰减: {result['max_degradation']:+.4f}")
        if result['n_negative_test'] > 0:
            lines.append(f" ⚠️  {result['n_negative_test']}/{len(result['windows'])} 个窗口 Test Sharpe < 0")
        else:
            lines.append(f" ✅ 所有窗口 Test Sharpe > 0")

    elif result.get('mode') == 'benchmark_wf':
        lines.append(f" 滚动窗口数: {result['n_windows']} (test windows = {result['n_test_windows']})")
        lines.append(f" 基准：每周再平衡等权 (rebal) / 真·买入持有 (buyhold)")
        lines.append("")
        lines.append(f" {'Win':<5s} {'End':>12s} {'Wks':>5s} {'Strat':>7s} {'Rebal':>7s} {'BH':>7s} | {'Strat':>7s} {'Rebal':>7s} {'BH':>7s} | {'Strat':>6s} {'Rebal':>6s} {'BH':>6s}")
        lines.append(f" {'':<5s} {'(test)':>12s} {'':>5s} {'Sharpe':>7s} {'Sharpe':>7s} {'Sharpe':>7s} | {'AnnRet':>7s} {'AnnRet':>7s} {'AnnRet':>7s} | {'MaxDD':>6s} {'MaxDD':>6s} {'MaxDD':>6s}")
        lines.append(" " + "-" * 120)
        for w in result['windows']:
            win_e = 'Y' if w['vs_ew_rebal'] else '-'
            win_b = 'Y' if w['vs_buyhold'] else '-'
            win = f"{win_e}/{win_b}"
            lines.append(
                f" {win:<5s} {w['test_end']:>12s} {w['weeks']:>5d} "
                f"{w['strategy_sharpe']:>7.3f} {w['ew_rebal_sharpe']:>7.3f} {w['buyhold_sharpe']:>7.3f} | "
                f"{w['strategy_annual']*100:>+6.2f}% {w['ew_rebal_annual']*100:>+6.2f}% {w['buyhold_annual']*100:>+6.2f}% | "
                f"{w['strategy_dd']*100:>5.2f}% {w['ew_rebal_dd']*100:>5.2f}% {w['buyhold_dd']*100:>5.2f}%"
            )
        lines.append(" " + "-" * 120)
        nw = result['n_test_windows']
        lines.append(f" vs 每周再平衡 (rebal):   {result['wins_vs_ew_rebal']}/{nw} 胜 "
                     f"({result['wins_vs_ew_rebal']/nw*100:.1f}%)    "
                     f"avg Sharpe: strat {result['avg_strategy_sharpe']:.3f} vs rebal {result['avg_ew_rebal_sharpe']:.3f}")
        lines.append(f" vs 真·买入持有 (buyhold): {result['wins_vs_buyhold']}/{nw} 胜 "
                     f"({result['wins_vs_buyhold']/nw*100:.1f}%)    "
                     f"avg Sharpe: strat {result['avg_strategy_sharpe']:.3f} vs buyhold {result['avg_buyhold_sharpe']:.3f}")

    elif result.get('mode') == 'reoptimize_wf':
        lines.append(f" 真·滚动 WF（每窗 anchored 训练重新选参 → test 验证，不重拟合）")
        lines.append(f" 滚动窗口数: {result['n_windows']} (test windows = {result['n_test_windows']})")
        lines.append("")
        lines.append(f" {'Win':<5s} {'TrainEnd':>11s} {'TestEnd':>11s} {'Wks':>4s} "
                     f"{'TrainSh':>8s} {'TestSh':>7s} {'Degr':>6s} {'Rebal':>7s} {'BH':>7s}  {'BestParams'}")
        lines.append(" " + "-" * 118)
        for w in result['windows']:
            win_e = 'Y' if w['vs_ew_rebal'] else '-'
            win_b = 'Y' if w['vs_buyhold'] else '-'
            bp = w['best_params']
            bp_str = (f"mom={bp['mom_w']} vol={bp['vol_w']} def={bp['def_alloc']} "
                      f"lo={bp['step_low']} hi={bp['step_high']}")
            lines.append(
                f" {win_e}/{win_b:<3s} {w['train_end']:>11s} {w['test_end']:>11s} {w['weeks']:>4d} "
                f"{w['train_sharpe']:>8.3f} {w['test_strategy_sharpe']:>7.3f} "
                f"{w['sharpe_degradation']:>+6.2f} {w['ew_rebal_sharpe']:>7.3f} {w['buyhold_sharpe']:>7.3f}  {bp_str}"
            )
        lines.append(" " + "-" * 118)
        nw = result['n_test_windows']
        lines.append(f" vs 每周再平衡 (rebal):   {result['wins_vs_ew_rebal']}/{nw} 胜 "
                     f"({result['wins_vs_ew_rebal']/nw*100:.1f}%)    "
                     f"avg TestSharpe: strat {result['avg_test_strategy_sharpe']:.3f} vs rebal {result['avg_ew_rebal_sharpe']:.3f}")
        lines.append(f" vs 真·买入持有 (buyhold): {result['wins_vs_buyhold']}/{nw} 胜 "
                     f"({result['wins_vs_buyhold']/nw*100:.1f}%)    "
                     f"avg TestSharpe: strat {result['avg_test_strategy_sharpe']:.3f} vs buyhold {result['avg_buyhold_sharpe']:.3f}")
        lines.append(f" 平均 IS→OOS Sharpe 退化: {result['avg_sharpe_degradation']:+.3f} "
                     f"(train {result['avg_train_sharpe']:.3f} → test {result['avg_test_strategy_sharpe']:.3f})")

    lines.append("=" * 60)
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description='Walk-Forward 验证')
    p.add_argument('--ratio', type=float, default=0.6, help='Train 比例 (default: 0.6)')
    p.add_argument('--rolling', action='store_true', help='滚动 WF 模式(固定参数)')
    p.add_argument('--benchmark', action='store_true', help='固定参数 vs 基准(每周再平衡/买入持有)对比，非防过拟合')
    p.add_argument('--reoptimize', action='store_true', help='真·滚动WF：每窗重新选参再验证(唯一防过拟合样本外)')
    p.add_argument('--windows', type=int, default=5, help='滚动窗口数 (default: 5)')
    p.add_argument('--json', action='store_true', help='JSON 输出')
    args = p.parse_args()

    cfg = load_config(PROJECT / 'config/strategy_v4_6.yaml')

    if args.reoptimize:
        result = reoptimize_wf(cfg, n_windows=args.windows)
    elif args.benchmark:
        result = benchmark_wf(cfg, n_windows=args.windows)
    elif args.rolling:
        result = rolling_wf(cfg, n_windows=args.windows)
    else:
        result = single_split_wf(cfg, train_ratio=args.ratio)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(fmt_report(result))


if __name__ == '__main__':
    main()
