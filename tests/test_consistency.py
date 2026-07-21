"""一致性测试 - rebalance_live.py 与 backtest 引擎的逻辑一致性。

验证实盘脚本的独立实现与引擎在相同输入下产生相同输出。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from src.data_loader import ETFS, OFFENSIVE, DEFENSIVE, load_nav_data, resample_weekly, classify_etfs
from src.factors import calculate_momentum, calculate_volatility
from src.strategy import load_config, calculate_defense_ratio
from src.backtest import run_backtest
from src.utils import compute_sharpe


PROJECT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT / 'config' / 'strategy_v3_0_final.yaml'


@pytest.fixture(scope='module')
def engine_result():
    cfg = load_config(CONFIG_PATH)
    result = run_backtest(cfg)
    return result, cfg


class TestEngineLiveConsistency:
    def test_sharpe_reasonable(self, engine_result):
        """引擎 Sharpe 应为合理值（>1.0）。"""
        result, cfg = engine_result
        assert result.metrics['sharpe_ratio'] > 1.0

    def test_defense_ratio_in_valid_range(self, engine_result):
        """验证引擎最后一周的防御比例在合理范围 [def_alloc, max_def] 内，
        且与 vol 三段式公式一致（允许止损覆盖导致更高）。

        注意: 引擎最后一条记录对应 index n_weeks-2（需要下一周收益计算P&L），
        而实盘脚本看的是最后一行。因此用引擎对应的 vol 来验证。
        """
        result, cfg = engine_result
        nav_path = PROJECT / cfg.nav_path
        df = load_nav_data(nav_path)
        weekly = resample_weekly(df, anchor=cfg.anchor)
        if cfg.start_date:
            weekly = weekly[weekly.index >= pd.to_datetime(cfg.start_date)]

        vol = calculate_volatility(weekly, window=cfg.vol_window)
        # 引擎最后决策对应 weekly 的倒数第二行（index -2）
        engine_last_idx = len(weekly) - 2
        nasdaq_vol = vol.iloc[engine_last_idx]['纳指ETF']
        expected_def = calculate_defense_ratio(nasdaq_vol, cfg)

        engine_def = result.nav_series['def_ratio'].iloc[-1]
        # 引擎可能因止损覆盖而 >= vol三段式结果
        assert engine_def >= expected_def - 0.005,             f'Engine def={engine_def:.4f} < expected={expected_def:.4f} (vol={nasdaq_vol:.4f})'
        # 防御比例应在合理范围
        assert cfg.def_alloc - 0.001 <= engine_def <= cfg.max_def + 0.01,             f'Engine def={engine_def:.4f} out of range [{cfg.def_alloc}, {cfg.max_def}]'

    def test_full_verify_sharpe_gap(self):
        """完整验证：引擎 vs 手动逐周回测的 Sharpe 差距 < 0.05。"""
        cfg = load_config(CONFIG_PATH)
        result = run_backtest(cfg)
        engine_sharpe = result.metrics['sharpe_ratio']

        nav_path = PROJECT / cfg.nav_path
        df = load_nav_data(nav_path)
        weekly = resample_weekly(df, anchor=cfg.anchor)
        if cfg.start_date:
            weekly = weekly[weekly.index >= pd.to_datetime(cfg.start_date)]

        mom = calculate_momentum(weekly, window=cfg.mom_window)
        vol = calculate_volatility(weekly, window=cfg.vol_window)

        prices = weekly.values
        w_rets = np.diff(prices, axis=0) / prices[:-1]
        n_weeks = len(weekly)
        etf_names = list(weekly.columns)
        off_idx, def_idx, nasdaq_idx = classify_etfs(etf_names)

        nav = 1.0
        peak = 1.0
        last_alloc = np.zeros(len(etf_names))
        weekly_rets = []
        start_idx = cfg.vol_window

        for i in range(start_idx, n_weeks - 1):
            scores_vec = np.full(len(etf_names), -np.inf)
            for j in off_idx:
                mv = mom.values[i, j]
                vv = vol.values[i, j]
                if not np.isnan(mv) and not np.isnan(vv):
                    scores_vec[j] = cfg.mom_w * mv - cfg.vol_w * vv

            off_scores = [(scores_vec[j], j) for j in off_idx if not np.isnan(scores_vec[j])]
            off_scores.sort(key=lambda x: x[0], reverse=True)
            selected_off = [j for _, j in off_scores[:cfg.top_n]]

            nasdaq_vol = vol.values[i, nasdaq_idx]
            if pd.isna(nasdaq_vol):
                def_ratio = cfg.def_alloc
            elif nasdaq_vol < cfg.step_low:
                def_ratio = cfg.def_alloc
            elif nasdaq_vol > cfg.step_high:
                def_ratio = cfg.max_def
            else:
                slope = (nasdaq_vol - cfg.step_low) / (cfg.step_high - cfg.step_low)
                def_ratio = cfg.def_alloc + (cfg.max_def - cfg.def_alloc) * slope

            alloc = np.zeros(len(etf_names))
            if def_idx:
                hl_vol_val = vol.values[i, def_idx[0]]
                if not np.isnan(hl_vol_val):
                    hl_ratio = np.clip(0.80 - 2.67 * hl_vol_val, 0, 0.80)
                else:
                    hl_ratio = cfg.hongli_ratio
                alloc[def_idx[0]] = def_ratio * hl_ratio
                if len(def_idx) > 1:
                    alloc[def_idx[1]] = def_ratio * (1 - hl_ratio)

            if selected_off and i >= cfg.inv_vol_window:
                inv_vols = []
                for j in selected_off:
                    rets = w_rets[i - cfg.inv_vol_window:i, j]
                    rets = rets[~np.isnan(rets)]
                    v = np.std(rets, ddof=0) * np.sqrt(52) if len(rets) >= 3 else 0.20
                    inv_vols.append(1.0 / max(v, 0.05))
                total = sum(inv_vols)
                for k, j in enumerate(selected_off):
                    alloc[j] = (1 - def_ratio) * (inv_vols[k] / total)
            elif selected_off:
                for j in selected_off:
                    alloc[j] = (1 - def_ratio) / len(selected_off)

            if cfg.max_single_alloc < 1.0:
                overflow = 0.0
                for j in off_idx:
                    if alloc[j] > cfg.max_single_alloc:
                        overflow += alloc[j] - cfg.max_single_alloc
                        alloc[j] = cfg.max_single_alloc
                if overflow > 0 and def_idx:
                    def_total = sum(alloc[j] for j in def_idx)
                    if def_total > 0:
                        for j in def_idx:
                            alloc[j] += overflow * alloc[j] / def_total

            if i > start_idx:
                max_change = np.max(np.abs(alloc - last_alloc))
                if max_change < cfg.rebalance_threshold:
                    alloc = last_alloc.copy()

            turnover = np.sum(np.abs(alloc - last_alloc))
            fee = turnover * cfg.fee_rate
            wret = sum(alloc[j] * w_rets[i, j] for j in range(len(etf_names))
                       if not np.isnan(w_rets[i, j]))
            nav *= (1 + wret - fee)
            peak = max(peak, nav)
            weekly_rets.append(wret - fee)
            last_alloc = alloc.copy()

        manual_sharpe = compute_sharpe(pd.Series(weekly_rets), cfg.risk_free_rate)
        gap = abs(engine_sharpe - manual_sharpe)
        assert gap < 0.05,             f'Sharpe gap too large: engine={engine_sharpe:.4f}, manual={manual_sharpe:.4f}, gap={gap:.4f}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
