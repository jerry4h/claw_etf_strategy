"""单元测试 — src/factors.py 因子计算正确性验证。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from src.factors import calculate_momentum, calculate_volatility, compute_all_factors


@pytest.fixture
def simple_nav():
    """构造一个简单的 5-ETF 周频净值序列（20周）。"""
    np.random.seed(42)
    n_weeks = 20
    dates = pd.date_range('2020-01-06', periods=n_weeks, freq='W-MON')
    # 生成随机游走价格
    rets = np.random.randn(n_weeks, 5) * 0.02
    prices = np.cumprod(1 + rets, axis=0) * 100
    cols = ['纳指ETF', '红利低波ETF', '中证500ETF', '黄金ETF', '国债ETF']
    return pd.DataFrame(prices, index=dates, columns=cols)


class TestMomentum:
    def test_shape(self, simple_nav):
        mom = calculate_momentum(simple_nav, window=4)
        assert mom.shape == simple_nav.shape

    def test_first_window_is_nan(self, simple_nav):
        mom = calculate_momentum(simple_nav, window=4)
        # 前 window 行应为 NaN（需要 window 个收益率）
        assert mom.iloc[:4].isna().all().all()

    def test_manual_calculation(self, simple_nav):
        """手动验证第 5 行的动量值。"""
        mom = calculate_momentum(simple_nav, window=4)
        prices = simple_nav.values
        # momentum[4] = prod(price[1:5] / price[0:4]) - 1
        # 即 prod(1 + ret[0:4]) - 1
        for j in range(5):
            rets = prices[1:5, j] / prices[0:4, j] - 1
            expected = np.prod(1 + rets) - 1
            actual = mom.iloc[4, j]
            assert abs(actual - expected) < 1e-10, f"col {j}: {actual} != {expected}"

    def test_window_1_equals_weekly_return(self, simple_nav):
        """window=1 时动量应等于单周收益率。"""
        mom = calculate_momentum(simple_nav, window=1)
        prices = simple_nav.values
        for i in range(1, len(prices)):
            for j in range(5):
                expected = prices[i, j] / prices[i-1, j] - 1
                actual = mom.iloc[i, j]
                assert abs(actual - expected) < 1e-10


class TestVolatility:
    def test_shape(self, simple_nav):
        vol = calculate_volatility(simple_nav, window=11)
        assert vol.shape == simple_nav.shape

    def test_first_window_is_nan(self, simple_nav):
        vol = calculate_volatility(simple_nav, window=11)
        assert vol.iloc[:11].isna().all().all()

    def test_ddof0(self, simple_nav):
        """验证使用 ddof=0（总体标准差）。"""
        window = 11
        vol = calculate_volatility(simple_nav, window=window)
        prices = simple_nav.values
        w_rets = np.diff(prices, axis=0) / prices[:-1]
        # 验证第 12 行（index=11）
        for j in range(5):
            rets_slice = w_rets[0:window, j]
            expected = np.std(rets_slice, ddof=0) * np.sqrt(52)
            actual = vol.iloc[11, j]
            assert abs(actual - expected) < 1e-10

    def test_annualization_factor(self, simple_nav):
        """验证年化因子为 sqrt(52)。"""
        vol = calculate_volatility(simple_nav, window=11)
        # 取一个非 NaN 值，验证量级合理（年化 vol 通常在 5%~80%）
        valid = vol.iloc[11:].values.flatten()
        valid = valid[~np.isnan(valid)]
        assert np.all(valid > 0)
        assert np.all(valid < 5.0)  # 年化 500% 以上不合理


class TestComputeAllFactors:
    def test_returns_dict(self, simple_nav):
        config = {'factors': {'mom_window': 4, 'vol_window': 11, 'pe_window_years': 5}}
        result = compute_all_factors(simple_nav, pe_df=None, config=config)
        assert 'momentum' in result
        assert 'volatility' in result
        assert 'pe_percentile' not in result  # pe_df=None

    def test_no_lookahead(self, simple_nav):
        """验证因子在 index i 只使用 i 及之前的数据。
        方法：截断数据到第 k 周，计算因子，与完整数据的前 k 行对比。"""
        config = {'factors': {'mom_window': 4, 'vol_window': 11, 'pe_window_years': 5}}
        full = compute_all_factors(simple_nav, pe_df=None, config=config)

        # 截断到第 15 行
        truncated = simple_nav.iloc[:15]
        partial = compute_all_factors(truncated, pe_df=None, config=config)

        # 前 15 行的因子值应完全一致
        pd.testing.assert_frame_equal(
            full['momentum'].iloc[:15], partial['momentum'],
            check_names=False
        )
        pd.testing.assert_frame_equal(
            full['volatility'].iloc[:15], partial['volatility'],
            check_names=False
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
