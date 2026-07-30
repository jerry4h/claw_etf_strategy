"""实验C辅助分析：统计 ashare_vol_boost（M3 中证500 vol 危机加成）实际触发的周数占比。

只读复用 src 模块，完全复刻 src/backtest.py run_backtest 中的数据管线
（load_nav_data → resample_weekly → 日期过滤 → compute_all_factors → 逐周循环），
对每个回测周 i 调用 engine_core.compute_ashare_vol_boost，统计 boost>0 的周数。

用法:
    .venv/bin/python scripts/_exp_ashare_trigger.py [--config config/experiments/v4_3_ashare_boost.yaml]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_nav_data, load_pe_percentile, resample_weekly, classify_etfs
from src.factors import compute_all_factors
from src.engine_core import compute_ashare_vol_boost
from src.strategy import load_config


def main():
    parser = argparse.ArgumentParser(description='统计 ashare_vol_boost 触发周占比')
    parser.add_argument('--config', default='config/experiments/v4_3_ashare_boost.yaml')
    args = parser.parse_args()

    config = load_config(project_root / args.config)
    print(f"配置: {args.config}")
    print(f"ashare_vol_boost_enabled = {config.ashare_vol_boost_enabled}")
    print(f"threshold={config.ashare_vol_crisis_threshold}, max_boost={config.ashare_vol_max_boost}, "
          f"slope={config.ashare_vol_slope}, pct_window={config.ashare_vol_pct_window}")

    # === 数据管线（与 run_backtest 一致）===
    _nav_path = config.nav_path
    if _nav_path and not Path(_nav_path).is_absolute():
        _nav_path = project_root / _nav_path
    _pe_path = config.pe_path
    if _pe_path and not Path(_pe_path).is_absolute():
        _pe_path = project_root / _pe_path

    nav_df = load_nav_data(_nav_path)
    weekly_nav = resample_weekly(nav_df, anchor=config.anchor)
    pe_df = load_pe_percentile(_pe_path) if _pe_path and Path(_pe_path).exists() else None

    start = config.start_date
    end = config.end_date
    if start:
        weekly_nav = weekly_nav[weekly_nav.index >= pd.to_datetime(start)]
    if end:
        weekly_nav = weekly_nav[weekly_nav.index <= pd.to_datetime(end)]

    config_dict = {
        'factors': {
            'mom_window': config.mom_window,
            'vol_window': config.vol_window,
            'vol_ddof': config.vol_ddof,
            'pe_window_years': config.pe_window_years,
            'ewma_factors_enabled': config.ewma_factors_enabled,
            'ewma_mom_halflife': config.ewma_mom_halflife,
            'ewma_vol_halflife': config.ewma_vol_halflife,
            'vol_taper_enabled': config.vol_taper_enabled,
            'vol_taper_window': config.vol_taper_window,
            'vol_taper_len': config.vol_taper_len,
        }
    }
    factors = compute_all_factors(weekly_nav, pe_df, config_dict)
    vol_values = factors['volatility'].values

    w_index = weekly_nav.index
    n_weeks = len(w_index)
    etf_names = list(weekly_nav.columns)
    CSI500_IDX = etf_names.index('中证500ETF') if '中证500ETF' in etf_names else -1
    print(f"CSI500_IDX = {CSI500_IDX} ({'中证500ETF' if CSI500_IDX >= 0 else '未找到!'})")

    # 起始索引（与 run_backtest 一致）
    if config.ewma_factors_enabled:
        start_idx = max(config.ewma_mom_halflife * 2, config.ewma_vol_halflife * 2,
                        config.vol_window, config.mom_window)
    elif config.vol_taper_enabled:
        start_idx = max(config.vol_taper_window, config.mom_window)
    else:
        start_idx = max(config.vol_window, config.mom_window)

    # === 逐周计算 boost（与 backtest.py L376 完全一致的调用）===
    triggered = []  # (date, boost)
    total = 0
    for i in range(start_idx, n_weeks - 1):
        total += 1
        boost = compute_ashare_vol_boost(vol_values, i, CSI500_IDX, config)
        if boost > 0:
            triggered.append((w_index[i], boost))

    n_trig = len(triggered)
    print(f"\n回测周数(循环范围): {total}")
    print(f"触发周数(boost>0): {n_trig}")
    print(f"触发占比: {n_trig / total * 100:.2f}%")
    if triggered:
        boosts = np.array([b for _, b in triggered])
        print(f"boost 均值: {boosts.mean():.4f}, 最大: {boosts.max():.4f}, "
              f"达到 max_boost({config.ashare_vol_max_boost}) 的周数: {int((boosts >= config.ashare_vol_max_boost - 1e-12).sum())}")
        # 按年份分布
        years = pd.Series([d.year for d, _ in triggered]).value_counts().sort_index()
        print("\n触发周按年份分布:")
        for y, c in years.items():
            print(f"  {y}: {c} 周")
        print("\n触发明细(前 30 条):")
        for d, b in triggered[:30]:
            print(f"  {d.date()}  boost={b:.4f}")


if __name__ == '__main__':
    main()
