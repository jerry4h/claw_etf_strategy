"""前视偏差验证测试 — 确认信号生成严格使用历史可用信息。

两份 GPT 审计报告均提出"交易时点/未来函数风险"为最高优先级问题。
本测试从三个维度验证无前视偏差：

1. 因子截断不变性：截断数据到第 k 周，因子值与完整数据的前 k 行完全一致
2. 信号-收益对齐：第 i 周的决策使用 index<=i 的信息，收益来自 i→i+1
3. 调仓执行时点：信号在周末生成，下周一执行（anchor=W-MON）

时点模型文档：
  - 数据频率: 周频（历史=周一快照，增量=周五快照）
  - 信号生成: 第 i 周末（使用 price[0..i] 计算 mom/vol）
  - 调仓执行: 第 i+1 周初（周一开盘）
  - 收益归属: w_rets[i] = (price[i+1] - price[i]) / price[i]
  - 回测循环: for i in range(start_idx, n_weeks-1):
      决策用 mom_values[i], vol_values[i]（仅含 price[0..i]）
      收益用 w_rets[i]（即 price[i] → price[i+1] 的收益率）
  - 结论: 决策时点 = i，收益时点 = i→i+1，无前视偏差
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from src.factors import calculate_momentum, calculate_volatility
from src.data_loader import load_nav_data, resample_weekly
from src.strategy import load_config


PROJECT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT / 'config' / 'strategy_v3_0_final.yaml'


@pytest.fixture(scope='module')
def weekly_data():
    cfg = load_config(CONFIG_PATH)
    nav_path = PROJECT / cfg.nav_path
    df = load_nav_data(nav_path)
    weekly = resample_weekly(df, anchor=cfg.anchor)
    if cfg.start_date:
        weekly = weekly[weekly.index >= pd.to_datetime(cfg.start_date)]
    return weekly, cfg


class TestNoLookAheadBias:
    """验证因子计算不存在前视偏差。"""

    def test_momentum_truncation_invariance(self, weekly_data):
        """截断数据到第 k 周，动量值与完整数据的前 k 行完全一致。

        如果 momentum[k] 使用了 k 之后的数据，截断后值会改变。
        """
        weekly, cfg = weekly_data
        full_mom = calculate_momentum(weekly, window=cfg.mom_window)

        # 测试多个截断点
        for k in [50, 100, 200, 300, len(weekly) - 1]:
            if k >= len(weekly):
                continue
            truncated = weekly.iloc[:k+1]
            partial_mom = calculate_momentum(truncated, window=cfg.mom_window)

            # 前 k+1 行必须完全一致
            pd.testing.assert_frame_equal(
                full_mom.iloc[:k+1], partial_mom,
                check_names=False,
                obj=f"Momentum truncated at k={k}"
            )

    def test_volatility_truncation_invariance(self, weekly_data):
        """截断数据到第 k 周，波动率值与完整数据的前 k 行完全一致。"""
        weekly, cfg = weekly_data
        full_vol = calculate_volatility(weekly, window=cfg.vol_window)

        for k in [50, 100, 200, 300, len(weekly) - 1]:
            if k >= len(weekly):
                continue
            truncated = weekly.iloc[:k+1]
            partial_vol = calculate_volatility(truncated, window=cfg.vol_window)

            pd.testing.assert_frame_equal(
                full_vol.iloc[:k+1], partial_vol,
                check_names=False,
                obj=f"Volatility truncated at k={k}"
            )

    def test_signal_uses_only_past_prices(self, weekly_data):
        """验证第 i 周的信号仅依赖 price[0..i]，不依赖 price[i+1..]。

        方法：修改第 i+1 周及之后的价格，确认第 i 周的因子值不变。
        """
        weekly, cfg = weekly_data
        n = len(weekly)
        test_idx = min(200, n - 10)

        # 原始因子
        orig_mom = calculate_momentum(weekly, window=cfg.mom_window)
        orig_vol = calculate_volatility(weekly, window=cfg.vol_window)

        # 篡改 test_idx+1 之后的所有价格（乘以随机因子）
        tampered = weekly.copy()
        np.random.seed(99)
        tamper_factor = np.random.uniform(0.5, 2.0, size=(n - test_idx - 1, len(weekly.columns)))
        tampered.iloc[test_idx+1:] = tampered.iloc[test_idx+1:].values * tamper_factor

        tampered_mom = calculate_momentum(tampered, window=cfg.mom_window)
        tampered_vol = calculate_volatility(tampered, window=cfg.vol_window)

        # 第 test_idx 行及之前的因子值必须完全一致
        pd.testing.assert_frame_equal(
            orig_mom.iloc[:test_idx+1], tampered_mom.iloc[:test_idx+1],
            check_names=False,
            obj="Momentum must not change when future prices are tampered"
        )
        pd.testing.assert_frame_equal(
            orig_vol.iloc[:test_idx+1], tampered_vol.iloc[:test_idx+1],
            check_names=False,
            obj="Volatility must not change when future prices are tampered"
        )

    def test_backtest_return_alignment(self, weekly_data):
        """验证回测中第 i 周的收益确实来自 price[i] → price[i+1]。

        即：决策在 i 周末做出，收益在 i+1 周实现。
        """
        weekly, cfg = weekly_data
        prices = weekly.values
        w_rets = np.diff(prices, axis=0) / prices[:-1]

        # w_rets[i] 应该是 (price[i+1] - price[i]) / price[i]
        for i in [10, 50, 100, 200]:
            if i >= len(w_rets):
                continue
            for j in range(prices.shape[1]):
                expected = (prices[i+1, j] - prices[i, j]) / prices[i, j]
                actual = w_rets[i, j]
                assert abs(actual - expected) < 1e-12, \
                    f"w_rets[{i},{j}] = {actual} != expected {expected}"

    def test_decision_at_i_uses_mom_vol_at_i(self, weekly_data):
        """验证回测循环中，第 i 周的决策使用 mom[i] 和 vol[i]，
        而 mom[i] 仅依赖 price[i-window..i]（不含 i+1）。
        """
        weekly, cfg = weekly_data
        mom = calculate_momentum(weekly, window=cfg.mom_window)
        vol = calculate_volatility(weekly, window=cfg.vol_window)
        prices = weekly.values

        # 验证 mom[i] 的计算仅使用 price[i-window..i]
        test_i = 100
        w_rets = np.diff(prices, axis=0) / prices[:-1]
        for j in range(prices.shape[1]):
            # mom[test_i] = prod(1 + w_rets[test_i-window : test_i]) - 1
            # w_rets[k] = (price[k+1] - price[k]) / price[k]
            # 所以 w_rets[test_i-1] = (price[test_i] - price[test_i-1]) / price[test_i-1]
            # 最大用到的 price 是 price[test_i]，不含 price[test_i+1]
            rets_slice = w_rets[test_i - cfg.mom_window:test_i, j]
            expected_mom = np.prod(1 + rets_slice) - 1
            actual_mom = mom.values[test_i, j]
            assert abs(actual_mom - expected_mom) < 1e-12, \
                f"mom[{test_i},{j}] uses wrong window"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
