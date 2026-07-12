#!/usr/bin/env python3
"""快速绩效计算 — 策略 vs 等权持有 (5 ETF 各 20%，每周再均衡)
直接复用 backtest.py 引擎，不重复实现策略逻辑。
策略参数变化时只需改 config yaml，本脚本无需修改。

输出: 当年收益、近1年收益、当前回撤及起始日

用法:
  python scripts/calc_performance.py              # 全部输出
  python scripts/calc_performance.py --ytd        # 仅当年
  python scripts/calc_performance.py --json       # JSON 格式
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
_VENV = PROJECT / '.venv' / 'lib' / 'python3.12' / 'site-packages'
if _VENV.exists():
    sys.path.insert(0, str(_VENV))
sys.path.insert(0, str(PROJECT))

from src.backtest import run_backtest
from src.strategy import load_config
from src.data_loader import ETFS


def compute_navs():
    cfg = load_config(PROJECT / 'config/strategy_v3_0_final.yaml')

    # 策略净值（官方引擎）
    r = run_backtest(cfg)
    strat_nav = r.nav_series['nav']

    # 等权基准 — 每周再均衡到 20%（不是买入持有让权重漂移）
    csv_path = cfg.nav_path if Path(cfg.nav_path).is_absolute() else str(PROJECT / cfg.nav_path)
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    cols = [c for c in df.columns if c in ETFS]
    w_prices = df[cols].values
    w_rets = np.diff(w_prices, axis=0) / w_prices[:-1]
    bn = np.ones(len(w_prices))
    for i in range(1, len(w_prices)):
        wret = sum(1.0/len(cols) * w_rets[i-1, j] for j in range(len(cols))
                   if not np.isnan(w_rets[i-1, j]))
        bn[i] = bn[i-1] * (1 + wret)
    bench_nav = pd.Series(bn, index=df.index)

    # 对齐两个序列到一个公共索引
    common = strat_nav.index.intersection(bench_nav.index)
    strat_nav = strat_nav.loc[common]
    bench_nav = bench_nav.loc[common]

    return strat_nav, bench_nav


def compute_perf(strat_nav, bench_nav):
    idx = strat_nav.index
    last = idx[-1]
    cur_year = last.year

    ytd_start = idx[idx.year == cur_year][0]
    oney_start = idx[-52] if len(idx) >= 52 else idx[0]

    def _metrics(nav):
        ytd_ret = float(nav.loc[last] / nav.loc[ytd_start] - 1)
        oney_ret = float(nav.loc[last] / nav.loc[oney_start] - 1)
        recent = nav.loc[ytd_start:last]
        peak = recent.cummax()
        dd = (peak - recent) / peak
        ddn = float(dd.iloc[-1])
        dd_start = None
        if ddn > 0.001:
            dd_start = str(peak[peak == peak.iloc[-1]].index[0].date())
        return ytd_ret, oney_ret, ddn, dd_start

    s = _metrics(strat_nav)
    b = _metrics(bench_nav)

    return {
        'strategy': dict(zip(['ytd', '1y', 'dd', 'dd_start'], s)),
        'benchmark': dict(zip(['ytd', '1y', 'dd', 'dd_start'], b)),
        'last_date': str(last.date()),
        'ytd_start': str(ytd_start.date()),
        'oney_start': str(oney_start.date()),
    }


def fmt_table(perf):
    lines = [
        "📊 绩效对比: 策略 vs 等权持有 (各 1/N，每周再均衡)",
        f"  数据截至 {perf['last_date']}",
        f"{'─'*50}",
        f"  {'':>22s} {'策略':>10s} {'等权持有':>10s}",
        f"{'─'*50}",
    ]
    s, b = perf['strategy'], perf['benchmark']
    yr = perf['last_date'][:4]
    lines.append(f"  今年收益({yr})          {s['ytd']*100:>+8.2f}% {b['ytd']*100:>+9.2f}%")
    if '1y' in s and s['1y'] is not None:
        lines.append(f"  近1年收益           {s['1y']*100:>+8.2f}% {b['1y']*100:>+9.2f}%")
    lines.append(f"  当前回撤            {s['dd']*100:>8.2f}% {b['dd']*100:>9.2f}%")
    ds_s = s.get('dd_start', '') or ''
    ds_b = b.get('dd_start', '') or ''
    lines.append(f"  回撤起始            {ds_s[:10] if ds_s else '-':>8s} {ds_b[:10] if ds_b else '-':>9s}")
    lines.append(f"{'─'*50}")
    return '\n'.join(lines)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='快速绩效对比')
    p.add_argument('--json', action='store_true', help='JSON 输出')
    p.add_argument('--ytd', action='store_true', help='仅当年')
    args = p.parse_args()

    strat_nav, bench_nav = compute_navs()
    perf = compute_perf(strat_nav, bench_nav)

    if args.ytd:
        perf_ytd = {
            'last_date': perf['last_date'], 'ytd_start': perf['ytd_start'],
            'strategy': {k: perf['strategy'][k] for k in ['ytd', 'dd', 'dd_start']},
            'benchmark': {k: perf['benchmark'][k] for k in ['ytd', 'dd', 'dd_start']},
        }
        if args.json:
            print(json.dumps(perf_ytd, ensure_ascii=False, indent=2, default=str))
        else:
            raw = fmt_table(perf_ytd)
            lines = [l for l in raw.split('\n') if '近1年' not in l]
            print('\n'.join(lines))
    elif args.json:
        print(json.dumps(perf, ensure_ascii=False, indent=2, default=str))
    else:
        print(fmt_table(perf))