"""单元测试 — load_config()、字段完整性、止损触发、增强 allocate()、defense_ratio override。"""
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.strategy import (
    StrategyConfig, load_config, allocate, check_stop_loss,
    calculate_defense_ratio, apply_max_alloc_cap,
)


# ── P1-5: load_config() 测试 ─────────────────────────────────────────────────

@pytest.fixture
def yaml_config_path():
    """Path to the real YAML config file."""
    return Path(__file__).resolve().parent.parent / 'config' / 'strategy_v4_1.yaml'


class TestLoadConfig:
    """Verify load_config() correctly parses YAML into StrategyConfig."""

    def test_loads_scorer_weights(self, yaml_config_path):
        cfg = load_config(yaml_config_path)
        assert cfg.mom_w == 1.0
        assert cfg.vol_w == 1.10

    def test_loads_selection_params(self, yaml_config_path):
        cfg = load_config(yaml_config_path)
        assert cfg.top_n == 2
        assert cfg.score_margin == 0.02
        assert cfg.dynamic_margin_sensitivity == 1.0
        assert cfg.dynamic_margin_window == 3
        assert cfg.trend_confirm_weeks == 0

    def test_loads_factor_windows(self, yaml_config_path):
        cfg = load_config(yaml_config_path)
        assert cfg.mom_window == 6
        assert cfg.vol_window == 11
        assert cfg.pe_window_years == 5

    def test_loads_defense_params(self, yaml_config_path):
        cfg = load_config(yaml_config_path)
        assert cfg.def_alloc == 0.25
        assert cfg.step_low == 0.15
        assert cfg.step_high == 0.35
        assert cfg.max_def == 0.95

    def test_loads_crisis_correlation(self, yaml_config_path):
        cfg = load_config(yaml_config_path)
        assert cfg.crisis_corr_window == 26
        assert cfg.crisis_corr_threshold == 0.60
        assert cfg.crisis_corr_slope == 1.875
        assert cfg.crisis_corr_max_boost == 0.15

    def test_loads_hongli_formula(self, yaml_config_path):
        cfg = load_config(yaml_config_path)
        assert cfg.hongli_intercept == 0.80
        assert cfg.hongli_vol_coeff == 2.67

    def test_loads_risk_control(self, yaml_config_path):
        cfg = load_config(yaml_config_path)
        assert cfg.stop_loss == 0.08
        assert cfg.recovery_weeks == 4

    def test_loads_allocation(self, yaml_config_path):
        cfg = load_config(yaml_config_path)
        assert cfg.max_single_alloc == 0.40
        assert cfg.overflow_to_defense_only is True
        assert cfg.dynamic_weight_cap is False

    def test_disabled_features_have_correct_defaults(self, yaml_config_path):
        """All DISABLED features should be False after loading."""
        cfg = load_config(yaml_config_path)
        assert cfg.d4_enabled is False
        assert cfg.softmax_enabled is False
        assert cfg.d1_enabled is False
        assert cfg.tiered_stop_loss is False
        assert cfg.ptiered_stop_loss is False
        assert cfg.stateful_stop_loss is False
        assert cfg.regime_enabled is False
        assert cfg.constituent_signals_enabled is False

    def test_inv_vol_enabled(self, yaml_config_path):
        cfg = load_config(yaml_config_path)
        assert cfg.inv_vol_enabled is True
        assert cfg.inv_vol_window == 10


# ── P1-6: 字段完整性回归测试 ─────────────────────────────────────────────────

class TestFieldCompleteness:
    """Regression test: load_config() must populate ALL StrategyConfig fields.

    Catches the bug where new fields added to StrategyConfig but missing from
    load_config() silently fall back to dataclass defaults (which may differ
    from YAML values). This caused dynamic_margin_sensitivity=1.0 in YAML to
    become 0.0 in MC/OAT/Grid tests.
    """

    def test_yaml_populates_all_known_fields(self, yaml_config_path):
        """Every field in StrategyConfig should have a non-default value
        set by load_config() OR be explicitly disabled/defaulted in YAML."""
        cfg = load_config(yaml_config_path)
        default_cfg = StrategyConfig()

        # These fields are expected to differ from defaults because YAML sets them
        critical_fields = {
            'mom_w': 1.0,
            'vol_w': 1.10,
            'mom_window': 6,
            'vol_window': 11,
            'dynamic_margin_sensitivity': 1.0,
            'dynamic_margin_window': 3,
            'score_margin': 0.02,
            'crisis_corr_window': 26,
            'crisis_corr_threshold': 0.60,
            'crisis_corr_slope': 1.875,
            'crisis_corr_max_boost': 0.15,
            'hongli_intercept': 0.80,
            'hongli_vol_coeff': 2.67,
            'rebalance_threshold': 0.025,
            'max_single_alloc': 0.40,
            'inv_vol_enabled': True,
            'inv_vol_window': 10,
        }
        for field_name, expected_val in critical_fields.items():
            actual = getattr(cfg, field_name)
            assert actual == expected_val, (
                f"{field_name}: expected {expected_val} from YAML, got {actual}. "
                f"load_config() may be missing this field."
            )

    def test_dataclass_has_no_new_unmapped_fields(self, yaml_config_path):
        """Detect if StrategyConfig has fields that load_config() doesn't set.

        Compares loaded config against a default-constructed StrategyConfig.
        Fields that are identical to defaults AND are not trivially safe
        (like legacy disabled features) are flagged.
        """
        cfg = load_config(yaml_config_path)
        default_cfg = StrategyConfig()

        # Fields that are safe to match defaults:
        # - Legacy/disabled features (expected to use defaults)
        # - Fields where YAML value == dataclass default (ambiguous:
        #   correctly loaded from YAML, or never loaded — both look the same)
        # Critical fields with distinct values are tested in test_yaml_populates_all_known_fields.
        safe_defaults = {
            'name', 'version',  # identity
            'nav_path', 'pe_path', 'start_date', 'end_date',  # data paths
            'risk_free_rate',  # reporting
            # Ambiguous: YAML value == dataclass default (verified in other test)
            'top_n', 'trend_confirm_weeks', 'mom_window', 'vol_window',
            'pe_window_years', 'def_alloc', 'step_high', 'max_def',
            'crisis_corr_window', 'crisis_corr_threshold',
            'crisis_corr_slope', 'crisis_corr_max_boost',
            'hongli_intercept', 'hongli_vol_coeff',
            'fee_rate', 'anchor', 'inv_vol_window',
            'step_low',  # YAML=0.15, but dataclass was updated to match
            # Legacy disabled features — expected to match defaults
            'tiered_stop_loss', 'l1_drawdown', 'l1_defense', 'l2_drawdown',
            'l2_defense', 'l3_weekly_drop', 'l3_down_weeks', 'l3_window',
            'l2_recovery_weeks', 'l3_recovery_weeks',
            'ptiered_stop_loss', 'p_recovery_weeks',
            'p_l1_dd_low', 'p_l1_dd_high', 'p_l1_position',
            'p_l2_dd_low', 'p_l2_dd_high', 'p_l2_position',
            'p_l3_dd_threshold', 'p_l3_position',
            'stateful_stop_loss', 'ms_bull_mom', 'ms_correction_mom',
            'ms_crisis_mom', 'ms_low_vol_pct', 'ms_mid_vol_pct',
            'ms_high_vol_pct', 'ms_shallow_dd', 'ms_moderate_dd',
            'ms_deep_dd', 'ss_bull_l1', 'ss_bull_l1_def', 'ss_bull_l2',
            'ss_bull_l2_def', 'ss_bull_recovery', 'ss_normal_l1',
            'ss_normal_l1_def', 'ss_normal_l2', 'ss_normal_l2_def',
            'ss_normal_recovery', 'ss_correction_l1', 'ss_correction_l1_def',
            'ss_correction_l2', 'ss_correction_l2_def',
            'ss_correction_recovery', 'ss_crisis_l1', 'ss_crisis_l1_def',
            'ss_crisis_l2', 'ss_crisis_l2_def', 'ss_crisis_recovery',
            'dynamic_weight_cap', 'dc_bull_cap', 'dc_normal_cap',
            'dc_correction_cap', 'dc_crisis_cap',
            'd4_enabled', 'd4_momentum_window', 'd4_momentum_threshold',
            'd4_action', 'd4_min_candidates',
            'softmax_enabled', 'softmax_temperature',
            'softmax_hard_top_n_fallback', 'softmax_min_candidates',
            'softmax_regime_enabled', 'softmax_regime_temperature',
            'd1_enabled', 'd1_lookback', 'd1_tq_low', 'd1_tq_high',
            'd1_mom_w_low', 'd1_mom_w_high', 'd1_vol_w_low',
            'd1_vol_w_high', 'd1_weight_sum',
            'constituent_signals_enabled', 'constituent_signals_path',
            'cwm_weight', 'conc_weight', 'cwm_window',
            'regime_enabled', 'regime_data_path', 'regime_overrides',
            'regime_3state',
            'hongli_ratio',  # kept at default 0.50 (dynamic formula supersedes)
            'stop_loss', 'recovery_weeks',
            'overflow_to_defense_only',
            'vol_ddof', 'hedge_cost_weekly',
            # 已对齐到生产 YAML 值：YAML==default，属模糊情况（在 test_yaml_populates_all_known_fields 验证）
            'mom_w', 'vol_w', 'score_margin', 'dynamic_margin_sensitivity',
            'rebalance_threshold', 'max_single_alloc',
            'snr_adaptive_enabled', 'snr_ewma_halflife', 'snr_vol_baseline',  # v4.0 SNR
            'ewma_factors_enabled', 'ewma_mom_halflife', 'ewma_vol_halflife',  # v4.0 EWMA
            'vol_taper_enabled', 'vol_taper_window', 'vol_taper_len',  # v4.0 Vol Taper
            # M3: 中证500 vol crisis boost — 默认关；YAML 段 ashare_vol 可启用
            'ashare_vol_boost_enabled', 'ashare_vol_crisis_threshold', 'ashare_vol_max_boost',
            'ashare_vol_slope', 'ashare_vol_pct_window',
            # v4.4: Layer 3.5 EWMA 相关估计 — 默认关；YAML 段 crisis_correlation_ewma 可启用
            # (v4_1 YAML 无此段 → 默认值；load_config 接线在 test_v44_crisis_corr 验证)
            'crisis_corr_ewma_enabled', 'crisis_corr_ewma_halflife',
        }

        mismatches = []
        for f in fields(StrategyConfig):
            loaded = getattr(cfg, f.name)
            default = getattr(default_cfg, f.name)
            if loaded == default and f.name not in safe_defaults:
                # Flag: this field was NOT changed by load_config()
                # and is not in the safe-defaults list
                mismatches.append(f.name)

        # Allow some fields to match defaults — but assert the list is small
        assert len(mismatches) == 0, (
            f"Fields that match defaults but aren't in safe_defaults: {mismatches}. "
            f"If intentional, add them to safe_defaults. Otherwise, fix load_config()."
        )


# ── P2: 止损触发测试 ──────────────────────────────────────────────────────────

class TestStopLoss:
    """Test check_stop_loss() behavior."""

    def test_not_triggered_when_no_drawdown(self):
        assert check_stop_loss(1.0, 1.0, 0.08) is False

    def test_triggered_at_threshold(self):
        # DD = (1.0 - 0.92) / 1.0 ≈ 0.08 (use 0.919 to avoid float precision edge)
        assert check_stop_loss(0.919, 1.0, 0.08) is True

    def test_not_triggered_below_threshold(self):
        # DD = (1.0 - 0.93) / 1.0 = 0.07 < 0.08 → False
        assert check_stop_loss(0.93, 1.0, 0.08) is False

    def test_triggered_above_threshold(self):
        # DD = (1.0 - 0.90) / 1.0 = 0.10 > 0.08 → True
        assert check_stop_loss(0.90, 1.0, 0.08) is True

    def test_peak_zero_returns_false(self):
        assert check_stop_loss(0.5, 0.0, 0.08) is False

    def test_higher_peak_same_nav_more_likely_trigger(self):
        # DD1 = (1.10 - 1.0) / 1.10 = 0.0909 > 0.08 → True
        assert check_stop_loss(1.0, 1.10, 0.08) is True
        # DD2 = (1.05 - 1.0) / 1.05 = 0.0476 < 0.08 → False
        assert check_stop_loss(1.0, 1.05, 0.08) is False


# ── P1-3: 增强 allocate() 测试 ───────────────────────────────────────────────

class TestAllocateEnhanced:
    """Test enhanced allocate() with dynamic hongli, inv-vol, and max cap."""

    @pytest.fixture
    def base_config(self):
        return StrategyConfig(
            mom_w=1.0, vol_w=1.10, top_n=2,
            def_alloc=0.25, step_low=0.15, step_high=0.35, max_def=0.95,
            hongli_ratio=0.50,
            max_single_alloc=0.40,
            overflow_to_defense_only=True,
        )

    def test_eff_hl_ratio_override(self, base_config):
        """Dynamic hongli ratio should override static config value."""
        selected = ['纳指ETF', '黄金ETF']
        # Static: hl_ratio=0.50 → 红利低波=0.25*0.50=0.125
        alloc_static = allocate(selected, 0.25, base_config)
        assert abs(alloc_static[1] - 0.125) < 1e-9  # 红利低波ETF

        # Override: hl_ratio=0.70 → 红利低波=0.25*0.70=0.175
        alloc_dyn = allocate(selected, 0.25, base_config, eff_hl_ratio=0.70)
        assert abs(alloc_dyn[1] - 0.175) < 1e-9

    def test_off_weights_inv_vol(self, base_config):
        """Inv-vol weights should replace equal-weight offensive allocation."""
        selected = ['纳指ETF', '黄金ETF']
        # Equal weight: each gets (1-0.25)/2 = 0.375
        alloc_eq = allocate(selected, 0.25, base_config)
        assert abs(alloc_eq[0] - 0.375) < 1e-9  # 纳指ETF
        assert abs(alloc_eq[3] - 0.375) < 1e-9  # 黄金ETF

        # Inv-vol: 70/30 split
        alloc_iv = allocate(selected, 0.25, base_config, off_weights=[0.7, 0.3])
        assert abs(alloc_iv[0] - 0.75 * 0.7) < 1e-9  # 纳指ETF
        assert abs(alloc_iv[3] - 0.75 * 0.3) < 1e-9  # 黄金ETF

    def test_apply_cap(self, base_config):
        """apply_cap=True should enforce max_single_alloc."""
        selected = ['纳指ETF', '黄金ETF']
        base_config.max_single_alloc = 0.30
        # Without cap: each gets 0.375 (exceeds 0.30)
        alloc_no_cap = allocate(selected, 0.25, base_config, apply_cap=False)
        assert alloc_no_cap[0] > 0.30

        # With cap: capped at 0.30, excess goes to defense
        alloc_cap = allocate(selected, 0.25, base_config, apply_cap=True)
        assert alloc_cap[0] <= 0.30 + 1e-9
        assert alloc_cap[3] <= 0.30 + 1e-9
        # Sum should still be ~1.0
        assert abs(alloc_cap.sum() - 1.0) < 1e-9

    def test_backward_compatible_no_new_params(self, base_config):
        """Old callers (without new params) should work identically."""
        selected = ['纳指ETF', '中证500ETF']
        alloc = allocate(selected, 0.25, base_config)
        assert abs(alloc.sum() - 1.0) < 1e-9
        assert alloc[1] > 0  # 红利低波ETF gets defense allocation
        assert alloc[4] > 0  # 国债ETF gets defense allocation


# ── P2: calculate_defense_ratio 带 base_def_alloc override ────────────────────

class TestDefenseRatioOverride:
    """Test calculate_defense_ratio() with optional base_def_alloc parameter."""

    @pytest.fixture
    def config(self):
        return StrategyConfig(
            def_alloc=0.25, step_low=0.15, step_high=0.35, max_def=0.95
        )

    def test_default_uses_config_def_alloc(self, config):
        # Low vol → returns base (config.def_alloc = 0.25)
        assert calculate_defense_ratio(0.10, config) == 0.25

    def test_override_base_def_alloc(self, config):
        # Low vol with override → returns override value
        result = calculate_defense_ratio(0.10, config, base_def_alloc=0.40)
        assert result == 0.40

    def test_override_in_interpolation(self, config):
        # Mid vol (0.25, midpoint between 0.15 and 0.35)
        # Without override: 0.25 + (0.95-0.25) * 0.5 = 0.60
        result_default = calculate_defense_ratio(0.25, config)
        assert abs(result_default - 0.60) < 1e-9

        # With override (base=0.40): 0.40 + (0.95-0.40) * 0.5 = 0.675
        result_override = calculate_defense_ratio(0.25, config, base_def_alloc=0.40)
        assert abs(result_override - 0.675) < 1e-9

    def test_high_vol_ignores_base(self, config):
        # High vol → always max_def regardless of base
        result_default = calculate_defense_ratio(0.40, config)
        result_override = calculate_defense_ratio(0.40, config, base_def_alloc=0.40)
        assert result_default == 0.95
        assert result_override == 0.95

    def test_nan_vol_returns_base(self, config):
        assert calculate_defense_ratio(float('nan'), config) == 0.25
        assert calculate_defense_ratio(float('nan'), config, base_def_alloc=0.40) == 0.40


# ── v4.4: 配置加载回归 ────────────────────────────────────────────────────────

class TestV44ConfigLoad:
    """v4.4 配置加载回归（与 tests/test_v44_crisis_corr.py 的深度覆盖形成双保险）。"""

    def test_v4_4_config_loads(self):
        cfg = load_config(
            Path(__file__).resolve().parent.parent / "config" / "strategy_v4_4.yaml"
        )
        assert cfg.def_alloc == 0.35
        assert cfg.step_low == 0.075
        assert cfg.step_high == 0.38
        assert cfg.max_def == 0.83
        assert cfg.crisis_corr_ewma_enabled is True
        assert cfg.crisis_corr_ewma_halflife == 8
