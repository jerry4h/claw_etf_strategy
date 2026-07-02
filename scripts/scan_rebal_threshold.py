#!/usr/bin/env python3
"""调仓阈值扫描 0~10% (步长1%), 含万分之0.5费率"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import run_backtest
from src.strategy import load_config

cfg = load_config('config/strategy_v3_0_final.yaml')
cfg.nav_path = 'data/all_etfs_nav_2013_20260626.csv'
cfg.end_date = None; cfg.start_date = None

print()
print(f"{'阈值':>5} {'Sharpe':>8} {'年化%':>8} {'DD%':>7} {'波动%':>7} {'胜率%':>7} {'防御周':>6} {'交易数':>7} {'Calmar':>7}")
print('-' * 62)

for pct in range(0, 11):
    cfg.rebalance_threshold = pct / 100.0
    r = run_backtest(cfg)
    m = r.metrics
    nt = m.get('rebalance_count', 'N/A')
    ca = m.get('calmar_ratio', 0)
    print(f"{pct:>4}% {m['sharpe_ratio']:>8.4f} {m['annual_return']*100:>7.2f}  {m['max_drawdown']*100:>6.2f}  {m['annual_volatility']*100:>6.2f} {m['win_rate']*100:>6.1f}  {m.get('defensive_weeks',0):>5d} {str(nt):>6s} {ca:>7.2f}")
print()
