"""单元测试 — src/engine_core.py 共享策略函数 + apply_max_alloc_cap 边界情况。

覆盖:
- compute_crisis_boost: Layer 3.5 危机相关性收敛
- compute_dynamic_hongli: 动态红利低波配比
- compute_inv_vol_weights: 波动率倒数加权
- compute_score_margin: 静态+动态 score margin
- apply_trend_confirmation: 趋势确认过滤
- apply_max_alloc_cap: 权重上限（边界情况）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from src.strategy import StrategyConfig, apply_max_alloc_cap
from src.engine_core import (
    compute_crisis_boost,
    compute_dynamic_hongli,
    compute_inv_vol_weights,
    compute_score_margin,
    apply_trend_confirmation,
)


@pytest.fixture
def config():
    """v3.0 最终版参数（与 YAML 一致）。"""
    return StrategyConfig(
        mom_w=1.0, vol_w=1.10, top_n=2, score_margin=0.02,
        dynamic_margin_sensitivity=1.0, dynamic_margin_window=3,
        mom_window=6, vol_window=11,
        def_alloc=0.25, step_low=0.15, step_high=0.35, max_def=0.95,
        hongli_ratio=0.50,
        crisis_corr_window=26, crisis_corr_threshold=0.60,
        crisis_corr_slope=1.875, crisis_corr_max_boost=0.15,
        hongli_intercept=0.80, hongli_vol_coeff=2.67,
        rebalance_threshold=0.025, fee_rate=0.00005,
        max_single_alloc=0.40, inv_vol_enabled=True, inv_vol_window=10,
    )


# ===================================================================
# Layer 3.5: 危机相关性收敛
# ===================================================================

class TestCrisisCorrelationBoost:
    def test_no_boost_below_threshold(self, config):
        """低相关性 → boost = 0。"""
        np.random.seed(42)
        n = 50
        # 生成低相关性的随机收益 (corr ≈ 0)
        w_rets = np.random.randn(n, 5) * 0.02
        boost = compute_crisis_boost(w_rets, n - 1, [0, 2, 3], config)
        assert boost == 0.0

    def test_boost_when_high_correlation(self, config):
        """高相关性 → boost > 0。"""
        n = 50
        # 构造高度相关的收益序列
        base = np.random.randn(n) * 0.02
        w_rets = np.zeros((n, 5))
        w_rets[:, 0] = base + np.random.randn(n) * 0.001  # 纳指
        w_rets[:, 2] = base + np.random.randn(n) * 0.001  # 中证500
        w_rets[:, 3] = base + np.random.randn(n) * 0.001  # 黄金
        w_rets[:, 1] = np.random.randn(n) * 0.01  # 红利低波（不相关）
        w_rets[:, 4] = np.random.randn(n) * 0.01  # 国债（不相关）

        boost = compute_crisis_boost(w_rets, n - 1, [0, 2, 3], config)
        assert boost > 0.0
        assert boost <= config.crisis_corr_max_boost

    def test_boost_capped_at_max(self, config):
        """boost 不超过 max_boost。"""
        n = 50
        # 完全相关
        base = np.random.randn(n) * 0.02
        w_rets = np.zeros((n, 5))
        w_rets[:, 0] = base
        w_rets[:, 2] = base
        w_rets[:, 3] = base
        boost = compute_crisis_boost(w_rets, n - 1, [0, 2, 3], config)
        assert boost == config.crisis_corr_max_boost

    def test_insufficient_data_returns_zero(self, config):
        """窗口不足 → boost = 0。"""
        w_rets = np.random.randn(10, 5) * 0.02
        boost = compute_crisis_boost(w_rets, 5, [0, 2, 3], config)
        assert boost == 0.0

    def test_single_offensive_returns_zero(self, config):
        """只有一只进攻ETF → 无法算相关性 → boost = 0。"""
        w_rets = np.random.randn(50, 5) * 0.02
        boost = compute_crisis_boost(w_rets, 49, [0], config)
        assert boost == 0.0

    def test_window_alignment(self, config):
        """验证使用 w_rets[i-window:i]（不含当前周未来收益）。"""
        n = 50
        base = np.random.randn(n) * 0.02
        w_rets = np.zeros((n, 5))
        w_rets[:, 0] = base
        w_rets[:, 2] = base
        w_rets[:, 3] = base
        # i=30, window=26 → 使用 w_rets[4:30]
        boost_30 = compute_crisis_boost(w_rets, 30, [0, 2, 3], config)
        # i=49 → 使用 w_rets[23:49]
        boost_49 = compute_crisis_boost(w_rets, 49, [0, 2, 3], config)
        # 两处都是完全相关，boost 应该相同
        assert boost_30 == boost_49


# ===================================================================
# 动态红利低波配比
# ===================================================================

class TestDynamicHongli:
    def test_low_vol_high_ratio(self, config):
        """低波动率 → 高配比（接近 intercept）。"""
        # vol = 0.05 → 0.80 - 2.67*0.05 = 0.80 - 0.1335 = 0.6665
        result = compute_dynamic_hongli(0.05, config)
        expected = np.clip(0.80 - 2.67 * 0.05, 0, 0.80)
        assert abs(result - expected) < 1e-10

    def test_high_vol_clipped_to_zero(self, config):
        """高波动率 → 配比被 clip 到 0。"""
        # vol = 0.50 → 0.80 - 2.67*0.50 = 0.80 - 1.335 = -0.535 → clip → 0
        result = compute_dynamic_hongli(0.50, config)
        assert result == 0.0

    def test_nan_returns_default(self, config):
        """NaN 波动率 → 返回默认 hongli_ratio。"""
        result = compute_dynamic_hongli(float('nan'), config)
        assert result == config.hongli_ratio

    def test_moderate_vol(self, config):
        """中等波动率 → 线性公式。"""
        # vol = 0.15 → 0.80 - 2.67*0.15 = 0.80 - 0.4005 = 0.3995
        result = compute_dynamic_hongli(0.15, config)
        expected = 0.80 - 2.67 * 0.15
        assert abs(result - expected) < 1e-10

    def test_uses_config_params(self):
        """验证使用 config 参数而非硬编码值。"""
        cfg = StrategyConfig(
            hongli_intercept=0.70, hongli_vol_coeff=3.0, hongli_ratio=0.40
        )
        # vol=0.10 → 0.70 - 3.0*0.10 = 0.40
        result = compute_dynamic_hongli(0.10, cfg)
        assert abs(result - 0.40) < 1e-10


# ===================================================================
# 波动率倒数加权
# ===================================================================

class TestInvVolWeights:
    def test_weights_sum_to_one(self, config):
        """权重之和 = 1.0。"""
        np.random.seed(42)
        w_rets = np.random.randn(30, 5) * 0.02
        weights = compute_inv_vol_weights(w_rets, [0, 3], 25, 10)
        assert abs(sum(weights) - 1.0) < 1e-10

    def test_lower_vol_gets_higher_weight(self):
        """低波动率 ETF 获得更高权重。"""
        np.random.seed(42)
        w_rets = np.zeros((30, 5))
        # ETF 0: 低波动率
        w_rets[:, 0] = np.random.randn(30) * 0.005
        # ETF 3: 高波动率
        w_rets[:, 3] = np.random.randn(30) * 0.03
        weights = compute_inv_vol_weights(w_rets, [0, 3], 25, 10)
        assert weights[0] > weights[1]  # ETF 0 > ETF 3

    def test_insufficient_data_equal_weights(self):
        """数据不足 → 等权重。"""
        w_rets = np.random.randn(5, 5) * 0.02
        weights = compute_inv_vol_weights(w_rets, [0, 3], 3, 10)
        assert abs(weights[0] - 0.5) < 1e-10
        assert abs(weights[1] - 0.5) < 1e-10

    def test_empty_selection(self):
        """空选择 → 空列表。"""
        w_rets = np.random.randn(30, 5) * 0.02
        weights = compute_inv_vol_weights(w_rets, [], 25, 10)
        assert weights == []


# ===================================================================
# Score Margin (静态 + 动态)
# ===================================================================

class TestScoreMargin:
    def test_static_margin_only(self, config):
        """dynamic_sensitivity=0 → 纯静态 margin。"""
        cfg = StrategyConfig(score_margin=0.02, dynamic_margin_sensitivity=0.0)
        margin, hist = compute_score_margin(0.01, [], cfg)
        assert margin == 0.02
        assert hist == []  # 不追踪 history

    def test_dynamic_margin_increases_with_volatility(self, config):
        """gap 波动率大 → margin 增大。"""
        gap_history = [0.01, 0.05, 0.02]
        margin, hist = compute_score_margin(0.03, gap_history.copy(), config)
        # std([0.01, 0.05, 0.02, 0.03]) > 0 → margin > base
        assert margin > config.score_margin

    def test_dynamic_margin_window_limit(self, config):
        """gap_history 保持 window 大小（追加后裁剪到 window）。"""
        gap_history = [0.01, 0.02, 0.03]  # 3 items = window size
        margin, hist = compute_score_margin(0.05, gap_history.copy(), config)
        # 追加 0.05 → 4 items → pop oldest → 3 items
        assert len(hist) == config.dynamic_margin_window  # = 3
        assert hist == [0.02, 0.03, 0.05]  # oldest removed

    def test_single_gap_no_std(self, config):
        """只有 1 个 gap → std 无法计算 → 仅用 static margin。"""
        margin, hist = compute_score_margin(0.01, [], config)
        assert len(hist) == 1
        assert margin == config.score_margin  # 只有 1 个点，std=0

    def test_two_gaps_computes_std(self, config):
        """2 个 gap → 可以计算 std。"""
        margin, hist = compute_score_margin(0.05, [0.01], config)
        assert len(hist) == 2
        # std([0.01, 0.05]) > 0 → margin > base
        assert margin > config.score_margin


# ===================================================================
# Trend Confirmation
# ===================================================================

class TestTrendConfirmation:
    def test_disabled_returns_candidate(self, config):
        """trend_confirm_weeks=0 → 直接返回 candidate。"""
        cfg = StrategyConfig(trend_confirm_weeks=0)
        sel, pend, cnt = apply_trend_confirmation([0, 2], [0, 3], None, 0, cfg)
        assert sel == [0, 2]

    def test_first_week_starts_pending(self, config):
        """首次出现新候选 → pending_count=1, 不切换。"""
        cfg = StrategyConfig(trend_confirm_weeks=2)
        sel, pend, cnt = apply_trend_confirmation([0, 2], [0, 3], None, 0, cfg)
        assert sel == [0, 3]  # 保持旧选择
        assert cnt == 1

    def test_confirms_after_n_weeks(self, config):
        """连续 N 周一致 → 确认切换。"""
        cfg = StrategyConfig(trend_confirm_weeks=2)
        # Week 1: first appearance
        sel1, pend1, cnt1 = apply_trend_confirmation([0, 2], [0, 3], None, 0, cfg)
        assert sel1 == [0, 3]
        # Week 2: same candidate
        sel2, pend2, cnt2 = apply_trend_confirmation([0, 2], [0, 3], pend1, cnt1, cfg)
        assert sel2 == [0, 2]  # 确认！切换
        assert cnt2 == 2

    def test_reset_on_candidate_change(self, config):
        """候选变化 → 重置计数。"""
        cfg = StrategyConfig(trend_confirm_weeks=3)
        # Week 1: candidate A
        sel1, pend1, cnt1 = apply_trend_confirmation([0, 2], [0, 3], None, 0, cfg)
        # Week 2: candidate B (different)
        sel2, pend2, cnt2 = apply_trend_confirmation([2, 3], [0, 3], pend1, cnt1, cfg)
        assert cnt2 == 1  # 重置
        assert sel2 == [0, 3]  # 保持旧选择


# ===================================================================
# apply_max_alloc_cap — 边界情况
# ===================================================================

class TestMaxAllocCap:
    def test_no_cap_when_max_is_one(self):
        """max_single >= 1.0 → 不做任何修改。"""
        alloc = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        result = apply_max_alloc_cap(alloc, 1.0, [0, 2, 3])
        np.testing.assert_array_almost_equal(result, alloc)

    def test_basic_cap_overflow_to_defense(self):
        """基本 cap: 超出部分按防御层比例分配。"""
        alloc = np.array([0.50, 0.15, 0.0, 0.0, 0.35])
        # max_single=0.40, offensive_idx=[0,2,3]
        # ETF 0: 0.50 > 0.40, excess=0.10
        # defense total = 0.15 + 0.35 = 0.50
        # ETF 1 gets: 0.10 * 0.15/0.50 = 0.03
        # ETF 4 gets: 0.10 * 0.35/0.50 = 0.07
        result = apply_max_alloc_cap(
            alloc, 0.40, [0, 2, 3],
            overflow_to_defense_only=True
        )
        assert abs(result[0] - 0.40) < 1e-10
        assert abs(result[1] - 0.18) < 1e-10  # 0.15 + 0.03
        assert abs(result[4] - 0.42) < 1e-10  # 0.35 + 0.07
        assert abs(result.sum() - 1.0) < 1e-10

    def test_sum_preserved(self):
        """cap 后权重总和保持为 1.0。"""
        alloc = np.array([0.45, 0.10, 0.0, 0.35, 0.10])
        result = apply_max_alloc_cap(alloc, 0.40, [0, 2, 3])
        assert abs(result.sum() - 1.0) < 1e-10

    def test_multiple_etfs_capped(self):
        """多只 ETF 同时超 cap。"""
        alloc = np.array([0.45, 0.10, 0.0, 0.45, 0.0])
        result = apply_max_alloc_cap(alloc, 0.40, [0, 2, 3])
        assert result[0] <= 0.40 + 1e-10
        assert result[3] <= 0.40 + 1e-10
        assert abs(result.sum() - 1.0) < 1e-10

    def test_zero_defense_overflow_even_split(self):
        """防御层为 0 → overflow 均分。"""
        alloc = np.array([0.50, 0.0, 0.50, 0.0, 0.0])
        result = apply_max_alloc_cap(
            alloc, 0.40, [0, 2],
            overflow_to_defense_only=True,
            def_idx=[1, 4]
        )
        assert abs(result[0] - 0.40) < 1e-10
        assert abs(result[2] - 0.40) < 1e-10
        # overflow = 0.20, split evenly to def_idx
        assert abs(result[1] - 0.10) < 1e-10
        assert abs(result[4] - 0.10) < 1e-10
        assert abs(result.sum() - 1.0) < 1e-10

    def test_does_not_modify_original(self):
        """不修改输入数组。"""
        alloc = np.array([0.50, 0.15, 0.0, 0.0, 0.35])
        original = alloc.copy()
        apply_max_alloc_cap(alloc, 0.40, [0, 2, 3])
        np.testing.assert_array_equal(alloc, original)

    def test_v24_overflow_to_other_offensive(self):
        """overflow_to_defense_only=False → 先分给其他进攻ETF。"""
        alloc = np.array([0.50, 0.10, 0.20, 0.20, 0.0])
        result = apply_max_alloc_cap(
            alloc, 0.40, [0, 2, 3],
            overflow_to_defense_only=False
        )
        assert abs(result[0] - 0.40) < 1e-10
        # excess 0.10 distributed to other offensive ETFs [2,3]
        assert result[2] > 0.20  # got some overflow
        assert result[3] > 0.20
        assert abs(result.sum() - 1.0) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
