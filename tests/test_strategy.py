"""单元测试 — src/strategy.py 策略逻辑正确性验证。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from src.strategy import (
    StrategyConfig, score_offensive, select_top,
    calculate_defense_ratio, allocate, check_rebalance, check_stop_loss
)


@pytest.fixture
def config():
    """v3.0 最终版参数。"""
    return StrategyConfig(
        mom_w=1.0, vol_w=1.10, top_n=2, score_margin=0.02,
        mom_window=4, vol_window=11,
        def_alloc=0.25, step_low=0.15, step_high=0.35, max_def=0.95,
        hongli_ratio=0.50, rebalance_threshold=0.025, fee_rate=0.00005,
        max_single_alloc=0.40, inv_vol_enabled=True, inv_vol_window=10,
    )


@pytest.fixture
def sample_factors():
    """构造评分用的动量和波动率 DataFrame。"""
    dates = pd.date_range('2024-01-01', periods=5, freq='W-MON')
    cols = ['纳指ETF', '红利低波ETF', '中证500ETF', '黄金ETF', '国债ETF']
    mom_data = {
        '纳指ETF': [0.05, 0.03, -0.02, 0.08, 0.01],
        '红利低波ETF': [0.02, 0.01, 0.03, 0.01, 0.02],
        '中证500ETF': [0.04, 0.06, 0.01, -0.01, 0.03],
        '黄金ETF': [0.03, 0.02, 0.05, 0.04, 0.06],
        '国债ETF': [0.001, 0.002, 0.001, 0.001, 0.002],
    }
    vol_data = {
        '纳指ETF': [0.20, 0.18, 0.25, 0.12, 0.15],
        '红利低波ETF': [0.10, 0.12, 0.08, 0.11, 0.09],
        '中证500ETF': [0.22, 0.19, 0.30, 0.16, 0.18],
        '黄金ETF': [0.15, 0.14, 0.18, 0.13, 0.16],
        '国债ETF': [0.02, 0.03, 0.02, 0.02, 0.03],
    }
    momentum = pd.DataFrame(mom_data, index=dates)
    volatility = pd.DataFrame(vol_data, index=dates)
    return momentum, volatility, dates


class TestScoreOffensive:
    def test_basic_scoring(self, config, sample_factors):
        momentum, volatility, dates = sample_factors
        scores = score_offensive(momentum, volatility, dates[0], config)
        # 进攻层: 纳指(0), 中证500(2), 黄金(3)
        assert '纳指ETF' in scores
        assert '中证500ETF' in scores
        assert '黄金ETF' in scores
        # 防御层不应出现
        assert '红利低波ETF' not in scores
        assert '国债ETF' not in scores

    def test_score_formula(self, config, sample_factors):
        """验证 score = mom_w * mom - vol_w * vol。"""
        momentum, volatility, dates = sample_factors
        scores = score_offensive(momentum, volatility, dates[0], config)
        # 纳指: 1.0 * 0.05 - 1.10 * 0.20 = 0.05 - 0.22 = -0.17
        expected = 1.0 * 0.05 - 1.10 * 0.20
        assert abs(scores['纳指ETF'] - expected) < 1e-10

    def test_nan_handling(self, config, sample_factors):
        momentum, volatility, dates = sample_factors
        # 插入 NaN
        momentum.iloc[2, 0] = np.nan
        scores = score_offensive(momentum, volatility, dates[2], config)
        assert scores['纳指ETF'] == float('-inf')

    def test_missing_date(self, config, sample_factors):
        momentum, volatility, dates = sample_factors
        scores = score_offensive(momentum, volatility, pd.Timestamp('2099-01-01'), config)
        assert scores == {}


class TestSelectTop:
    def test_selects_highest(self):
        scores = {'纳指ETF': 0.10, '中证500ETF': 0.05, '黄金ETF': 0.20}
        result = select_top(scores, top_n=2)
        assert result == ['黄金ETF', '纳指ETF']

    def test_excludes_inf(self):
        scores = {'纳指ETF': float('-inf'), '中证500ETF': 0.05, '黄金ETF': 0.20}
        result = select_top(scores, top_n=2)
        assert result == ['黄金ETF', '中证500ETF']

    def test_top_n_larger_than_valid(self):
        scores = {'纳指ETF': 0.10, '中证500ETF': float('-inf')}
        result = select_top(scores, top_n=2)
        assert result == ['纳指ETF']


class TestDefenseRatio:
    def test_below_step_low(self, config):
        # vol < step_low(0.15) → def_alloc(0.25)
        assert calculate_defense_ratio(0.10, config) == 0.25

    def test_above_step_high(self, config):
        # vol > step_high(0.35) → max_def(0.95)
        assert calculate_defense_ratio(0.40, config) == 0.95

    def test_linear_interpolation(self, config):
        # vol = 0.25 (midpoint of [0.15, 0.35])
        # ratio = (0.25-0.15)/(0.35-0.15) = 0.5
        # def = 0.25 + 0.5*(0.95-0.25) = 0.25 + 0.35 = 0.60
        result = calculate_defense_ratio(0.25, config)
        assert abs(result - 0.60) < 1e-10

    def test_at_boundaries(self, config):
        assert calculate_defense_ratio(0.15, config) == 0.25
        assert calculate_defense_ratio(0.35, config) == 0.95

    def test_nan_returns_default(self, config):
        assert calculate_defense_ratio(float('nan'), config) == 0.25


class TestAllocate:
    def test_weights_sum_to_one(self, config):
        selected = ['纳指ETF', '黄金ETF']
        alloc = allocate(selected, defense_ratio=0.30, config=config)
        assert abs(alloc.sum() - 1.0) < 1e-10

    def test_defense_split(self, config):
        selected = ['纳指ETF', '黄金ETF']
        alloc = allocate(selected, defense_ratio=0.40, config=config)
        # 红利低波(idx=1): 0.40 * 0.50 = 0.20
        # 国债(idx=4): 0.40 * 0.50 = 0.20
        assert abs(alloc[1] - 0.20) < 1e-10
        assert abs(alloc[4] - 0.20) < 1e-10

    def test_offensive_equal_split(self, config):
        selected = ['纳指ETF', '黄金ETF']
        alloc = allocate(selected, defense_ratio=0.25, config=config)
        # 进攻层: (1-0.25)/2 = 0.375 each
        assert abs(alloc[0] - 0.375) < 1e-10  # 纳指
        assert abs(alloc[3] - 0.375) < 1e-10  # 黄金

    def test_full_defense(self, config):
        selected = []
        alloc = allocate(selected, defense_ratio=0.95, config=config)
        assert abs(alloc[0]) < 1e-10  # 纳指=0
        assert abs(alloc[2]) < 1e-10  # 中证500=0
        assert abs(alloc[3]) < 1e-10  # 黄金=0


class TestCheckRebalance:
    def test_triggers_above_threshold(self):
        current = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        new = np.array([0.4, 0.1, 0.2, 0.2, 0.1])
        assert bool(check_rebalance(current, new, threshold=0.025)) == True

    def test_no_trigger_below_threshold(self):
        current = np.array([0.30, 0.20, 0.20, 0.20, 0.10])
        new = np.array([0.31, 0.19, 0.20, 0.20, 0.10])
        assert bool(check_rebalance(current, new, threshold=0.025)) == False


class TestCheckStopLoss:
    def test_triggers(self):
        assert check_stop_loss(0.90, 1.0, threshold=0.08) is True

    def test_no_trigger(self):
        assert check_stop_loss(0.95, 1.0, threshold=0.08) is False

    def test_zero_peak(self):
        assert check_stop_loss(0.95, 0.0, threshold=0.08) is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
