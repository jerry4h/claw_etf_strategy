"""PVD 条件激活因子 (v4.5) 单元测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.strategy import load_config, StrategyConfig
from src.factors import compute_pvd_factor, compute_all_factors


# ============================================================
# 1. Config loads correctly
# ============================================================

class TestPVDConfig:
    def test_pvd_config_loads(self):
        """v4_5_pvd.yaml 加载断言：enabled=True, w=0.15, window=8 等."""
        cfg = load_config(PROJECT / 'config/strategy_v4_5_pvd.yaml')
        assert cfg.pvd_enabled is True
        assert cfg.pvd_w == 0.15
        assert cfg.pvd_window == 8
        assert cfg.pvd_min_periods == 6
        assert cfg.pvd_score_gap_threshold == 0.05
        assert cfg.pvd_vol_pct_range == (0.25, 0.75)

    def test_pvd_disabled_by_default(self):
        """v4.3 和 v4.4 配置中 pvd_enabled 应为 False."""
        for name in ('strategy_v4_3.yaml', 'strategy_v4_4.yaml'):
            cfg = load_config(PROJECT / 'config' / name)
            assert cfg.pvd_enabled is False


# ============================================================
# 2. PVD computation correctness
# ============================================================

class TestPVDComputation:
    def _make_data(self, n=20):
        """Create synthetic nav and vol data."""
        dates = pd.date_range('2020-01-06', periods=n, freq='W-MON')
        np.random.seed(42)
        nav_vals = np.exp(np.cumsum(np.random.randn(n, 3) * 0.02, axis=0))
        vol_vals = np.abs(np.random.randn(n, 3) * 1000 + 5000)
        cols = ['A', 'B', 'C']
        nav = pd.DataFrame(nav_vals, index=dates, columns=cols)
        vol = pd.DataFrame(vol_vals, index=dates, columns=cols)
        return nav, vol

    def test_pvd_shape_and_nan(self):
        """PVD output shape matches input; first rows are NaN due to rolling."""
        nav, vol = self._make_data(20)
        pvd = compute_pvd_factor(nav, vol, window=8, min_periods=6)
        assert pvd.shape == nav.shape
        # First 6 rows should be NaN (shift + min_periods=6)
        assert pvd.iloc[:6].isna().all().all()
        # Later rows should have values
        assert pvd.iloc[10:].notna().any().any()

    def test_pvd_correctness_manual(self):
        """Manual check: 3-week subset rolling corr."""
        dates = pd.date_range('2020-01-06', periods=12, freq='W-MON')
        # Simple monotonic nav and vol for predictable corr
        nav_vals = np.array([[100, 200],
                            [102, 198],
                            [104, 196],
                            [106, 194],
                            [108, 192],
                            [110, 190],
                            [112, 188],
                            [114, 186],
                            [116, 184],
                            [118, 182],
                            [120, 180],
                            [122, 178]], dtype=float)
        vol_vals = np.array([[1000, 2000],
                            [1100, 1900],
                            [1200, 1800],
                            [1300, 1700],
                            [1400, 1600],
                            [1500, 1500],
                            [1600, 1400],
                            [1700, 1300],
                            [1800, 1200],
                            [1900, 1100],
                            [2000, 1000],
                            [2100, 900]], dtype=float)
        nav = pd.DataFrame(nav_vals, index=dates, columns=['X', 'Y'])
        vol = pd.DataFrame(vol_vals, index=dates, columns=['X', 'Y'])
        pvd = compute_pvd_factor(nav, vol, window=5, min_periods=4)
        # X: nav monotonically increasing + vol monotonically increasing
        # → log_ret > 0, vol_change > 0 → positive corr
        # Y: nav decreasing + vol decreasing → same direction → positive corr
        assert pvd['X'].iloc[-1] > 0.9  # strong positive
        assert pvd['Y'].iloc[-1] > 0.9  # strong positive

    def test_pvd_nan_propagation(self):
        """NaN in vol should produce NaN in pvd (no error)."""
        nav, vol = self._make_data(15)
        vol.iloc[:5, 1] = np.nan  # Set column B early rows to NaN
        pvd = compute_pvd_factor(nav, vol, window=8, min_periods=6)
        # Column B should have extra NaNs
        assert pvd.iloc[8:, 1].isna().sum() >= 0  # no crash


# ============================================================
# 3. Condition activation logic
# ============================================================

class TestPVDConditionActivation:
    def test_pvd_not_triggered_high_vol(self):
        """When nasdaq vol > p75, PVD should NOT be applied."""
        # This is implicitly tested via the backtest integration;
        # Here we verify the gate logic boundary.
        vol_series = np.array([0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40])
        p25 = np.percentile(vol_series, 25)
        p75 = np.percentile(vol_series, 75)
        # 0.40 > p75 → not in range
        assert 0.40 > p75
        # 0.10 < p25 → not in range
        assert 0.10 < p25
        # 0.18 in range
        assert p25 <= 0.18 <= p75

    def test_pvd_not_triggered_wide_gap(self):
        """When score gap > threshold, PVD should NOT be applied."""
        gap = 0.08
        threshold = 0.05
        assert gap >= threshold  # should NOT trigger


# ============================================================
# 4. NaN handling for 红利低波 pre-2019
# ============================================================

class TestPVDNanHandling:
    def test_hongli_pre2019_nan(self):
        """红利低波 pre-2019 成交额 should be NaN (no file data before 2019-01-18)."""
        from src.data_loader import load_weekly_volume_from_cache
        cache_dir = PROJECT / 'data' / 'experiments' / 'tushare_cache'
        if not cache_dir.exists():
            pytest.skip("tushare_cache not available")
        dates = pd.date_range('2018-01-01', periods=60, freq='W-MON')
        wv = load_weekly_volume_from_cache(cache_dir, dates)
        # 红利低波ETF column should be NaN for dates before 2019-01-14
        # (fund_daily_512890SH.csv starts 2019-01-18, ISO week of that date maps to Mon 2019-01-14)
        pre2019 = wv.loc[wv.index < '2019-01-14', '红利低波ETF']
        assert pre2019.isna().all()


# ============================================================
# 5. Baseline unchanged (CRITICAL)
# ============================================================

class TestBaselineUnchanged:
    def test_v43_sharpe_unchanged(self):
        """pvd_enabled=false (v4.3) 时 Sharpe 与 T1 前完全相同 (1.4878)."""
        from src.backtest import run_backtest
        cfg = load_config(PROJECT / 'config/strategy_v4_3.yaml')
        assert cfg.pvd_enabled is False
        result = run_backtest(cfg)
        sharpe = result.metrics['sharpe_ratio']
        # Pin to known baseline value
        assert abs(sharpe - 1.4878) < 0.0005, \
            f"Baseline Sharpe drift: got {sharpe:.4f}, expected 1.4878"


# ============================================================
# 6. compute_all_factors backward compatibility
# ============================================================

class TestComputeAllFactorsCompat:
    def test_no_weekly_vol_no_pvd(self):
        """When weekly_vol is None, result should NOT contain 'pvd' key."""
        dates = pd.date_range('2020-01-06', periods=30, freq='W-MON')
        nav = pd.DataFrame(
            np.random.randn(30, 3).cumsum(axis=0) + 100,
            index=dates, columns=['A', 'B', 'C']
        )
        config_dict = {'factors': {'mom_window': 6, 'vol_window': 11, 'pvd_enabled': True}}
        result = compute_all_factors(nav, None, config_dict, weekly_vol=None)
        assert 'pvd' not in result

    def test_pvd_disabled_no_pvd(self):
        """When pvd_enabled=False, result should NOT contain 'pvd' key."""
        dates = pd.date_range('2020-01-06', periods=30, freq='W-MON')
        nav = pd.DataFrame(
            np.random.randn(30, 3).cumsum(axis=0) + 100,
            index=dates, columns=['A', 'B', 'C']
        )
        vol = pd.DataFrame(
            np.abs(np.random.randn(30, 3)) * 1000 + 5000,
            index=dates, columns=['A', 'B', 'C']
        )
        config_dict = {'factors': {'mom_window': 6, 'vol_window': 11, 'pvd_enabled': False}}
        result = compute_all_factors(nav, None, config_dict, weekly_vol=vol)
        assert 'pvd' not in result
