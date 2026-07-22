"""单元测试 — src/robustness.py 鲁棒性评估函数验证。

覆盖:
  - _norm_cdf: 标准正态分布 CDF 值验证
  - compute_dsr: Deflated Sharpe Ratio 计算验证
  - compute_pss: 参数稳定性评分验证
"""
import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from src.robustness import _norm_cdf, compute_dsr, compute_pss


# ── _norm_cdf 测试 ───────────────────────────────────────────────────────────

class TestNormCdf:
    def test_zero_is_half(self):
        """Phi(0) = 0.5"""
        assert abs(_norm_cdf(0.0) - 0.5) < 1e-10

    def test_positive_1_96(self):
        """Phi(1.96) ≈ 0.975"""
        assert abs(_norm_cdf(1.96) - 0.975) < 0.001

    def test_negative_1_96(self):
        """Phi(-1.96) ≈ 0.025"""
        assert abs(_norm_cdf(-1.96) - 0.025) < 0.001

    def test_large_positive(self):
        """Phi(5) 应非常接近 1.0"""
        assert _norm_cdf(5.0) > 0.9999

    def test_large_negative(self):
        """Phi(-5) 应非常接近 0.0"""
        assert _norm_cdf(-5.0) < 0.0001

    def test_symmetry(self):
        """Phi(x) + Phi(-x) = 1.0"""
        for x in [0.5, 1.0, 1.5, 2.0, 3.0]:
            total = _norm_cdf(x) + _norm_cdf(-x)
            assert abs(total - 1.0) < 1e-10, f"对称性不满足: x={x}, sum={total}"

    def test_monotonicity(self):
        """Phi 应为单调递增函数。"""
        xs = np.linspace(-3, 3, 50)
        values = [_norm_cdf(x) for x in xs]
        for i in range(1, len(values)):
            assert values[i] >= values[i-1], f"非单调: x={xs[i-1]:.2f}->{xs[i]:.2f}"

    def test_known_value_1(self):
        """Phi(1.0) ≈ 0.8413"""
        assert abs(_norm_cdf(1.0) - 0.8413) < 0.001

    def test_known_value_negative_1(self):
        """Phi(-1.0) ≈ 0.1587"""
        assert abs(_norm_cdf(-1.0) - 0.1587) < 0.001


# ── compute_dsr 测试 ─────────────────────────────────────────────────────────

class TestComputeDsr:
    def test_high_sharpe_few_trials(self):
        """Sharpe=1.5, n_trials=30, n_obs=664, skew=0, kurtosis=3 → DSR 应接近 1.0。

        高 Sharpe 且中等试验次数，DSR 应非常高。
        """
        dsr = compute_dsr(
            sharpe=1.5,
            n_trials=30,
            n_obs=664,
            skew=0.0,
            kurtosis=3.0
        )
        assert dsr > 0.85, f"DSR 应接近 1.0，实际: {dsr:.4f}"
        assert dsr <= 1.0, f"DSR 应 <= 1.0，实际: {dsr:.4f}"

    def test_low_sharpe_many_trials(self):
        """Sharpe=0.3, n_trials=100 → DSR 应较低。"""
        dsr = compute_dsr(
            sharpe=0.3,
            n_trials=100,
            n_obs=200,
            skew=0.0,
            kurtosis=3.0
        )
        assert dsr < 0.5, f"低 Sharpe 多试验 DSR 应较低，实际: {dsr:.4f}"

    def test_dsr_in_unit_interval(self):
        """DSR 结果应在 [0, 1] 区间。"""
        for sharpe in [0.0, 0.5, 1.0, 2.0]:
            for n_trials in [1, 10, 50, 200]:
                dsr = compute_dsr(sharpe, n_trials, 500, 0.0, 3.0)
                assert 0.0 <= dsr <= 1.0, (
                    f"DSR 超出 [0,1]: sharpe={sharpe}, n_trials={n_trials}, dsr={dsr:.4f}"
                )

    def test_more_trials_lower_dsr(self):
        """相同 Sharpe 下，试验次数越多，DSR 应越低（多重测试惩罚）。"""
        dsr_10 = compute_dsr(1.0, 10, 500, 0.0, 3.0)
        dsr_100 = compute_dsr(1.0, 100, 500, 0.0, 3.0)
        assert dsr_10 > dsr_100, (
            f"更多试验应降低 DSR: dsr(n=10)={dsr_10:.4f}, dsr(n=100)={dsr_100:.4f}"
        )

    def test_higher_sharpe_higher_dsr(self):
        """相同试验次数下，Sharpe 越高，DSR 应越高。"""
        dsr_low = compute_dsr(0.5, 20, 500, 0.0, 3.0)
        dsr_high = compute_dsr(2.0, 20, 500, 0.0, 3.0)
        assert dsr_high > dsr_low, (
            f"更高 Sharpe 应提高 DSR: dsr(SR=0.5)={dsr_low:.4f}, dsr(SR=2.0)={dsr_high:.4f}"
        )

    def test_skew_effect(self):
        """负偏度应降低 DSR（增加风险估计）。"""
        dsr_no_skew = compute_dsr(1.0, 20, 500, skew=0.0, kurtosis=3.0)
        dsr_neg_skew = compute_dsr(1.0, 20, 500, skew=-1.0, kurtosis=3.0)
        # 负偏度增大 SE → 降低 z_stat → 降低 DSR
        # 注意：公式中 skew*sharpe 项为负，使 variance 增大
        assert dsr_no_skew >= dsr_neg_skew, (
            f"负偏度应降低 DSR: no_skew={dsr_no_skew:.4f}, neg_skew={dsr_neg_skew:.4f}"
        )

    def test_single_trial(self):
        """n_trials=1 时（无多重测试），DSR 应较高。"""
        dsr = compute_dsr(1.0, 1, 500, 0.0, 3.0)
        assert dsr > 0.8, f"单次试验 DSR 应较高，实际: {dsr:.4f}"


# ── compute_pss 测试 ─────────────────────────────────────────────────────────

class TestComputePss:
    def test_empty_details(self):
        """空 mc_details 应返回全零字典。"""
        result = compute_pss([])
        assert result['n_total'] == 0
        assert result['return_p50'] == 0.0
        assert result['sharpe_p50'] == 0.0

    def test_basic_statistics(self):
        """验证基础统计量计算正确。"""
        mc_details = [
            {'annual_return': 0.10, 'max_drawdown': 0.05, 'sharpe_ratio': 1.0},
            {'annual_return': 0.15, 'max_drawdown': 0.08, 'sharpe_ratio': 1.5},
            {'annual_return': 0.20, 'max_drawdown': 0.10, 'sharpe_ratio': 2.0},
            {'annual_return': 0.25, 'max_drawdown': 0.12, 'sharpe_ratio': 2.5},
            {'annual_return': 0.30, 'max_drawdown': 0.15, 'sharpe_ratio': 3.0},
        ]
        result = compute_pss(mc_details)

        assert result['n_total'] == 5
        # 中位数
        assert abs(result['return_p50'] - 0.20) < 1e-10
        assert abs(result['dd_p50'] - 0.10) < 1e-10
        assert abs(result['sharpe_p50'] - 2.0) < 1e-10

    def test_percentiles(self):
        """验证 p10 和 p90 计算。"""
        np.random.seed(42)
        n = 100
        mc_details = [
            {
                'annual_return': float(np.random.normal(0.15, 0.05)),
                'max_drawdown': float(np.random.uniform(0.03, 0.20)),
                'sharpe_ratio': float(np.random.normal(1.5, 0.3)),
            }
            for _ in range(n)
        ]
        result = compute_pss(mc_details)

        # 手动计算百分位
        returns = np.array([r['annual_return'] for r in mc_details])
        assert abs(result['return_p10'] - np.percentile(returns, 10)) < 1e-10
        assert abs(result['return_p90'] - np.percentile(returns, 90)) < 1e-10

    def test_coefficient_of_variation(self):
        """验证变异系数 (CV) 计算。"""
        mc_details = [
            {'annual_return': 0.10, 'max_drawdown': 0.05, 'sharpe_ratio': 1.0},
            {'annual_return': 0.20, 'max_drawdown': 0.10, 'sharpe_ratio': 2.0},
            {'annual_return': 0.30, 'max_drawdown': 0.15, 'sharpe_ratio': 3.0},
        ]
        result = compute_pss(mc_details)

        # return CV = std / mean
        returns = np.array([0.10, 0.20, 0.30])
        expected_cv = np.std(returns) / np.mean(returns)
        assert abs(result['return_cv'] - expected_cv) < 1e-10

    def test_single_entry(self):
        """单条 MC 结果应能正常计算。"""
        mc_details = [
            {'annual_return': 0.15, 'max_drawdown': 0.08, 'sharpe_ratio': 1.2},
        ]
        result = compute_pss(mc_details)
        assert result['n_total'] == 1
        assert result['return_p50'] == 0.15
        assert result['sharpe_p50'] == 1.2
        # 单条记录 std=0 → CV=0
        assert result['return_cv'] == 0.0

    def test_all_keys_present(self):
        """结果应包含所有预期键。"""
        mc_details = [
            {'annual_return': 0.15, 'max_drawdown': 0.08, 'sharpe_ratio': 1.2},
        ]
        result = compute_pss(mc_details)
        expected_keys = {
            'return_p10', 'return_p50', 'return_p90',
            'dd_p10', 'dd_p50', 'dd_p90',
            'sharpe_p10', 'sharpe_p50', 'sharpe_p90',
            'return_cv', 'dd_cv', 'sharpe_cv',
            'n_total',
        }
        assert set(result.keys()) == expected_keys


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
