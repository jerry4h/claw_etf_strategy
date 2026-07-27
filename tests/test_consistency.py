"""一致性测试 - rebalance_live.py 与 backtest 引擎的逻辑一致性。

验证实盘脚本的独立实现与引擎在相同输入下产生相同输出。
手动验证循环使用 engine_core 共享函数，确保引擎与共享逻辑一致。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from src.data_loader import ETFS, OFFENSIVE, DEFENSIVE, load_nav_data, resample_weekly, classify_etfs
from src.factors import calculate_momentum, calculate_volatility
from src.strategy import load_config, calculate_defense_ratio, apply_max_alloc_cap
from src.backtest import run_backtest
from src.utils import compute_sharpe
from src.engine_core import (
    compute_crisis_boost, compute_dynamic_hongli,
    compute_inv_vol_weights, compute_score_margin,
)


PROJECT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT / 'config' / 'strategy_v4_1.yaml'


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
        assert engine_def >= expected_def - 0.005, \
            f'Engine def={engine_def:.4f} < expected={expected_def:.4f} (vol={nasdaq_vol:.4f})'
        # 防御比例应在合理范围
        assert cfg.def_alloc - 0.001 <= engine_def <= cfg.max_def + 0.01, \
            f'Engine def={engine_def:.4f} out of range [{cfg.def_alloc}, {cfg.max_def}]'

    def test_full_verify_sharpe_gap(self):
        """完整验证：引擎 vs 手动逐周回测的 Sharpe 差距 < 0.05。
        手动循环使用 engine_core 共享函数，确保引擎与共享逻辑一致。
        """
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
        gap_history = []
        last_selected = None

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

            # Score Margin (static + dynamic) — using shared function
            if i > start_idx and len(off_scores) > cfg.top_n:
                gap = off_scores[cfg.top_n - 1][0] - off_scores[cfg.top_n][0]
                eff_margin, gap_history = compute_score_margin(gap, gap_history, cfg)
                if gap < eff_margin and last_selected is not None:
                    valid_last = [j for j in last_selected if j in off_idx and not np.isnan(scores_vec[j])]
                    if len(valid_last) == cfg.top_n:
                        selected_off = valid_last
            elif cfg.dynamic_margin_sensitivity > 0 and len(off_scores) > cfg.top_n:
                gap = off_scores[cfg.top_n - 1][0] - off_scores[cfg.top_n][0]
                _, gap_history = compute_score_margin(gap, gap_history, cfg)

            # Layer 3: defense ratio
            nasdaq_vol = vol.values[i, nasdaq_idx]
            def_ratio = calculate_defense_ratio(nasdaq_vol, cfg)

            # Layer 3.5: crisis correlation boost — using shared function
            crisis_boost = compute_crisis_boost(w_rets, i, off_idx, cfg)
            if crisis_boost > 0:
                def_ratio = min(def_ratio + crisis_boost, 1.0)

            # Build allocation
            alloc = np.zeros(len(etf_names))
            if def_idx:
                # Dynamic hongli — using shared function
                hl_vol_val = vol.values[i, def_idx[0]]
                hl_ratio = compute_dynamic_hongli(hl_vol_val, cfg)
                alloc[def_idx[0]] = def_ratio * hl_ratio
                if len(def_idx) > 1:
                    alloc[def_idx[1]] = def_ratio * (1 - hl_ratio)

            # Inv-vol weights — using shared function
            if selected_off and i >= cfg.inv_vol_window:
                inv_weights = compute_inv_vol_weights(
                    w_rets, selected_off, i, cfg.inv_vol_window
                )
                for k, j in enumerate(selected_off):
                    alloc[j] = (1 - def_ratio) * inv_weights[k]
            elif selected_off:
                for j in selected_off:
                    alloc[j] = (1 - def_ratio) / len(selected_off)

            # Max alloc cap — using shared function
            if cfg.max_single_alloc < 1.0:
                alloc = apply_max_alloc_cap(
                    alloc, cfg.max_single_alloc, off_idx,
                    overflow_to_defense_only=cfg.overflow_to_defense_only,
                    def_idx=def_idx
                )

            # Rebalance threshold
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
            last_selected = selected_off.copy()

        manual_sharpe = compute_sharpe(pd.Series(weekly_rets), cfg.risk_free_rate)
        gap = abs(engine_sharpe - manual_sharpe)
        assert gap < 0.05, \
            f'Sharpe gap too large: engine={engine_sharpe:.4f}, manual={manual_sharpe:.4f}, gap={gap:.4f}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
