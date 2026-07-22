#!/usr/bin/env python3
"""压力测试 — 策略在极端市场环境下的表现分析。

测试窗口:
  - 2015 股灾 (2015-06 ~ 2015-09)
  - 2016 熔断 (2016-01 ~ 2016-02)
  - 2018 熊市 (2018-01 ~ 2018-12)
  - 2020 疫情 (2020-02 ~ 2020-03)
  - 2022 下跌 (2022-01 ~ 2022-10)
  - 2024 调整 (2024-07 ~ 2024-09)

用法:
    python scripts/stress_test.py
"""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import numpy as np
import pandas as pd
from src.strategy import load_config
from src.backtest import run_backtest
from src.data_loader import load_nav_data, resample_weekly, ETFS


STRESS_WINDOWS = [
    ("2015 股灾",   "2015-06-01", "2015-09-30"),
    ("2016 熔断",   "2016-01-01", "2016-02-29"),
    ("2018 熊市",   "2018-01-01", "2018-12-31"),
    ("2020 疫情",   "2020-02-01", "2020-03-31"),
    ("2022 下跌",   "2022-01-01", "2022-10-31"),
    ("2024 调整",   "2024-07-01", "2024-09-30"),
]


def main():
    cfg = load_config(PROJECT / "config" / "strategy_v3_0_final.yaml")
    result = run_backtest(cfg)
    nav_series = result.nav_series

    # Load raw data for equal-weight benchmark
    nav_path = PROJECT / cfg.nav_path
    df = load_nav_data(nav_path)
    weekly = resample_weekly(df, anchor=cfg.anchor)
    if cfg.start_date:
        weekly = weekly[weekly.index >= pd.to_datetime(cfg.start_date)]

    print("=" * 75)
    print(" 压力测试 — 策略在极端市场环境下的表现")
    print("=" * 75)
    print()
    header = " {:<12s} {:<24s} {:>8s} {:>8s} {:>8s} {:>8s} {:>6s}".format(
        "窗口", "区间", "策略", "等权", "超额", "策略DD", "防御%")
    print(header)
    print(" " + "-" * 72)

    for name, start, end in STRESS_WINDOWS:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        # Strategy NAV in window
        mask = (nav_series.index >= start_dt) & (nav_series.index <= end_dt)
        window_nav = nav_series.loc[mask]
        if len(window_nav) < 2:
            print(" {:<12s} {}~{}  数据不足".format(name, start, end))
            continue

        strat_ret = window_nav["nav"].iloc[-1] / window_nav["nav"].iloc[0] - 1
        strat_peak = window_nav["nav"].cummax()
        strat_dd = ((strat_peak - window_nav["nav"]) / strat_peak).max()
        avg_def = window_nav["def_ratio"].mean()

        # Equal-weight benchmark in window
        wmask = (weekly.index >= start_dt) & (weekly.index <= end_dt)
        w_prices = weekly.loc[wmask].values
        if len(w_prices) < 2:
            ew_ret = 0.0
        else:
            ew_rets = np.diff(w_prices, axis=0) / w_prices[:-1]
            ew_ret = np.prod(1 + np.nanmean(ew_rets, axis=1)) - 1

        excess = strat_ret - ew_ret
        print(" {:<12s} {}~{}  {:>+7.1f}% {:>+7.1f}% {:>+7.1f}% {:>7.2f}% {:>5.0f}%".format(
            name, start, end, strat_ret*100, ew_ret*100, excess*100, strat_dd*100, avg_def*100))

    print()
    print(" 说明:")
    print("   - 策略DD: 窗口内最大回撤")
    print("   - 防御%: 窗口内平均防御比例")
    print("   - 超额 > 0 表示策略跑赢等权持有")
    print()

    # Full-period context
    m = result.metrics
    print(" 全区间参考: Sharpe={:.3f}, 年化={:.1f}%, DD={:.2f}%".format(
        m['sharpe_ratio'], m['annual_return']*100, m['max_drawdown']*100))
    print()

    # Per-ETF performance in stress windows
    print(" 各 ETF 在压力窗口的表现:")
    line = " {:<12s}".format("窗口")
    for etf in ETFS:
        line += " {:>8s}".format(etf[:4])
    print(line)
    print(" " + "-" * (12 + 9 * len(ETFS)))

    for name, start, end in STRESS_WINDOWS:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        wmask = (weekly.index >= start_dt) & (weekly.index <= end_dt)
        w_prices = weekly.loc[wmask].values
        if len(w_prices) < 2:
            continue
        line = " {:<12s}".format(name)
        for j in range(len(ETFS)):
            p0, p1 = w_prices[0, j], w_prices[-1, j]
            if p0 > 0 and not np.isnan(p0) and not np.isnan(p1):
                ret = p1 / p0 - 1
                line += " {:>+7.1f}%".format(ret*100)
            else:
                line += " {:>8s}".format("N/A")
        print(line)


if __name__ == "__main__":
    main()
