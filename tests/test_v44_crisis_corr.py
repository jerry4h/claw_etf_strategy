"""单元测试 — v4.4 相关性危机轴闭环（T3 测试套件）。

覆盖:
1. classic 路径逐位不变（flag=False 时 compute_crisis_boost ≡ _compute_crisis_boost_classic + pin 值防漂移）
2. EWMA 版对相关性阶跃的响应加速（halflife=8 vs 26 周均匀窗）
3. EWMA 加权相关数学自洽（halflife→∞ 收敛到等权 Pearson）
4. gen_regime_corr 两状态 Markov DGP（形状/有限性/可复现/危机-平常态相关分离）
5. robustness_score 向后兼容（include_corr_scenarios 开关控制情景键集合）
6. v4_3 / v4_4 YAML 配置加载（crisis_correlation_ewma 段接线 + 圆整防御参数）

约束: 无网络、固定 seed（无 flaky）、单测试 <30s。
"""
import importlib.util
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import numpy as np
import pandas as pd
import pytest

from src.strategy import StrategyConfig, load_config
from src.engine_core import (
    compute_crisis_boost,
    _compute_crisis_boost_classic,
    _compute_crisis_boost_ewma,
)


# ===================================================================
# 工具: 配置与合成收益构造
# ===================================================================

def make_cfg(**kw):
    """v4.3 生产 Layer 3.5 参数（window=26/threshold=0.60/slope=1.875/max=0.15）。"""
    base = dict(
        crisis_corr_window=26, crisis_corr_threshold=0.60,
        crisis_corr_slope=1.875, crisis_corr_max_boost=0.15,
        crisis_corr_ewma_enabled=False, crisis_corr_ewma_halflife=8,
    )
    base.update(kw)
    return StrategyConfig(**base)


def make_rets(seed, n=60, n_cols=5, corr=0.0, nan_frac=0.0, scale=0.02):
    """单因子结构合成周收益: 进攻对两两相关 ≈ corr（population），可注 NaN。"""
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(n)
    noise = rng.standard_normal((n, n_cols))
    load = np.sqrt(corr)
    rets = (load * common[:, None] + np.sqrt(1.0 - corr) * noise) * scale
    if nan_frac > 0:
        mask = rng.random((n, n_cols)) < nan_frac
        rets[mask] = np.nan
    return rets


def make_step_rets(seed, n=60, step=40, corr_lo=0.2, corr_hi=0.9):
    """相关性阶跃: 前 step 周进攻对相关≈corr_lo, 之后共同因子驱动相关≈corr_hi。"""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n)
    e = rng.standard_normal((n, 5))
    load = np.where(np.arange(n)[:, None] < step, np.sqrt(corr_lo), np.sqrt(corr_hi))
    return (load * f[:, None] + np.sqrt(1.0 - load ** 2) * e) * 0.02


OFF_IDX = [0, 2, 3]  # 与生产一致: 纳指/中证500/黄金

# ---- scripts/adversarial_robustness.py 惰性加载（仅测试 4/5 需要，避免拖慢 1-3/6）----
_ARM_CACHE = {}


def _load_arm():
    if "m" not in _ARM_CACHE:
        spec = importlib.util.spec_from_file_location(
            "arm_t3", PROJECT / "scripts" / "adversarial_robustness.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _ARM_CACHE["m"] = m
    return _ARM_CACHE["m"]


# ===================================================================
# 1. classic 路径逐位不变
# ===================================================================

class TestClassicPathBitwiseUnchanged:
    def test_classic_path_bitwise_unchanged(self):
        """flag=False 时分发器与 classic 实现逐位一致（含 NaN/高低相关多组随机输入）。"""
        cfg = make_cfg(crisis_corr_ewma_enabled=False)
        for seed in (1, 2, 3, 4):
            for corr in (0.0, 0.81):          # 低相关(boost=0) / 高相关(boost>0)
                for nan_frac in (0.0, 0.15):  # 无缺失 / 15% NaN
                    w_rets = make_rets(seed, n=60, corr=corr, nan_frac=nan_frac)
                    for i in (26, 40, 59):
                        via_dispatch = compute_crisis_boost(w_rets, i, OFF_IDX, cfg)
                        direct = _compute_crisis_boost_classic(w_rets, i, OFF_IDX, cfg)
                        # 逐位一致（同一代码路径，允许严格 ==）
                        assert via_dispatch == direct, (
                            f"seed={seed} corr={corr} nan={nan_frac} i={i}: "
                            f"dispatch={via_dispatch!r} != classic={direct!r}"
                        )

    def test_classic_pin_value(self):
        """pin 固定 seed 输入的具体数值，防未来 classic 实现漂移。

        构造: make_rets(seed=20260730, n=60, corr=0.64), i=59, off_idx=[0,2,3]
        期望值 0.017487802768011568 于 2026-07-30 由当前实现跑出（线性区，未触 cap）。
        """
        cfg = make_cfg(crisis_corr_ewma_enabled=False)
        w_rets = make_rets(20260730, n=60, corr=0.64)
        boost = compute_crisis_boost(w_rets, 59, OFF_IDX, cfg)
        PIN = 0.017487802768011568
        assert abs(boost - PIN) < 1e-12, f"classic pin 漂移: got {boost!r}, want {PIN!r}"
        # 附带健全性: 处于线性区 (0, max_boost)
        assert 0.0 < boost < cfg.crisis_corr_max_boost


# ===================================================================
# 2. EWMA 版响应加速
# ===================================================================

class TestEwmaBoostActivation:
    def test_ewma_boost_activation(self):
        """60 周合成: 前 40 周相关≈0.2、后 20 周≈0.9。

        halflife=8 时 EWMA 版在阶跃后 ≤10 周内 boost>0；同时点 classic 版
        （26 周均匀窗）boost 显著更小（<50%）——证明 EWMA 响应加速。
        """
        cfg_classic = make_cfg(crisis_corr_ewma_enabled=False)
        cfg_ewma = make_cfg(crisis_corr_ewma_enabled=True, crisis_corr_ewma_halflife=8)
        step = 40
        # 多 seed 防单一路径侥幸；全部固定 seed 无 flaky
        # (probe 实测激活周: seed 7→+7w, 2026→+7w, 99→+2w)
        for seed in (7, 2026, 99):
            w_rets = make_step_rets(seed, n=60, step=step)
            activation = None
            for i in range(step + 1, 60):
                if _compute_crisis_boost_ewma(w_rets, i, OFF_IDX, cfg_ewma) > 0:
                    activation = i
                    break
            assert activation is not None, f"seed={seed}: EWMA 60 周内未激活"
            weeks_after_step = activation - step
            assert weeks_after_step <= 10, (
                f"seed={seed}: EWMA 激活耗时 {weeks_after_step} 周 (>10)"
            )
            b_ewma = _compute_crisis_boost_ewma(w_rets, activation, OFF_IDX, cfg_ewma)
            b_classic = _compute_crisis_boost_classic(w_rets, activation, OFF_IDX, cfg_classic)
            # 同时点 classic 显著更小或为 0
            assert b_classic < 0.5 * b_ewma, (
                f"seed={seed} i={activation}: classic={b_classic:.4f} "
                f"未显著小于 ewma={b_ewma:.4f}"
            )


# ===================================================================
# 3. EWMA 加权相关数学自洽（等权极限收敛）
# ===================================================================

class TestEwmaWeightedCorrMath:
    def test_ewma_weighted_corr_math(self):
        """halflife=10000 → 权重≈等权 → EWMA 版与 classic 版 |Δ|<1e-3。"""
        cfg_classic = make_cfg(crisis_corr_ewma_enabled=False)
        cfg_inf = make_cfg(crisis_corr_ewma_enabled=True, crisis_corr_ewma_halflife=10000)
        nontrivial = 0
        for seed in (5, 8, 13, 21, 34):
            w_rets = make_rets(seed, n=40, corr=0.62)  # 无 NaN 随机输入
            b_classic = _compute_crisis_boost_classic(w_rets, 39, OFF_IDX, cfg_classic)
            b_ewma = _compute_crisis_boost_ewma(w_rets, 39, OFF_IDX, cfg_inf)
            assert abs(b_ewma - b_classic) < 1e-3, (
                f"seed={seed}: |Δ|={abs(b_ewma - b_classic):.2e} ≥ 1e-3 "
                f"(classic={b_classic!r}, ewma={b_ewma!r})"
            )
            if 0.0 < b_classic < cfg_classic.crisis_corr_max_boost:
                nontrivial += 1  # 线性区样本（非 0 也非 cap，收敛断言非平凡）
        # probe 实测 seed=8 落在线性区 (classic≈0.03047, |Δ|≈3.9e-5)
        assert nontrivial >= 1, "所有样本都在 0/cap 平凡区，收敛断言失去意义"


# ===================================================================
# 4. gen_regime_corr 两状态 Markov DGP
# ===================================================================

class TestRegimeCorrDgp:
    @pytest.fixture
    def fit_params(self):
        """最小合法拟合参数 (K=5)，不跑真实 fit_var_t/fit_garch。"""
        K = 5
        mu = np.full(K, 0.001)
        A = np.eye(K) * 0.05                              # 谱半径 0.05 << 0.99
        R = np.full((K, K), 0.15) + np.eye(K) * 0.85      # 平常态相关 0.15 (PSD)
        nu = 8.0
        gp = [{"omega": 1e-5, "alpha": 0.05, "beta": 0.90} for _ in range(K)]
        return mu, A, R, nu, gp

    def test_regime_corr_dgp(self, fit_params):
        arm = _load_arm()
        mu, A, R, nu, gp = fit_params

        r1 = arm.gen_regime_corr(mu, A, R, nu, gp, {}, 1000, 42)
        # 形状与有限性
        assert r1.shape == (1000, 5)
        assert np.isfinite(r1).all(), "存在 NaN/Inf"
        # 同 seed 两次调用逐位一致（可复现）
        r2 = arm.gen_regime_corr(mu, A, R, nu, gp, {}, 1000, 42)
        assert np.array_equal(r1, r2), "同 seed 不可复现"

        # 危机/平常态相关分离: 滚动 8 周样本相关分布的分位差
        # (危机态 rho_crisis=0.85 vs 平常态 0.15; P90-P10 阈值放宽到 0.3 防 flaky)
        win = 8
        a, b = OFF_IDX[0], OFF_IDX[1]
        cors = np.array([
            np.corrcoef(r1[t:t + win, a], r1[t:t + win, b])[0, 1]
            for t in range(1000 - win)
        ])
        cors = cors[np.isfinite(cors)]
        p10, p90 = np.percentile(cors, [10, 90])
        assert p90 - p10 > 0.3, (
            f"危机/平常态相关未分离: P90={p90:.3f} P10={p10:.3f} 差={p90 - p10:.3f}"
        )  # probe 实测 seed=42: P90-P10≈1.12


# ===================================================================
# 5. robustness_score 向后兼容（情景键集合）
# ===================================================================

class TestRobustnessScoreBackwardCompat:
    def test_robustness_score_backward_compat(self, monkeypatch):
        """mock 掉数据加载/拟合/回测评估，仅验证情景集合组装逻辑。"""
        arm = _load_arm()
        K = 5
        mu = np.full(K, 0.001)
        A = np.eye(K) * 0.05
        R = np.full((K, K), 0.15) + np.eye(K) * 0.85
        nu = 8.0
        gp = [{"omega": 1e-5, "alpha": 0.05, "beta": 0.90} for _ in range(K)]

        idx = pd.date_range("2020-01-06", periods=31, freq="W-MON")
        wk = pd.DataFrame(np.ones((31, K)), index=idx)
        w_rets = np.zeros((30, K))

        calls = []

        def fake_eval(mu, A, R, nu, gp, params, T, real_dates, first_nav, cfg,
                      seeds=(11, 22, 33)):
            calls.append(dict(params))
            return {"strat_sharpe": 1.0, "ew_sharpe": 0.5,
                    "strat_maxdd": 0.10, "ew_maxdd": 0.15,
                    "strat_annual": 0.12, "ew_annual": 0.06}

        monkeypatch.setattr(arm.dm, "load_real", lambda: (None, wk, w_rets))
        monkeypatch.setattr(arm.dm, "fit_var_t",
                            lambda x: (mu, A, None, nu, None, {}))
        monkeypatch.setattr(arm, "fit_garch", lambda resid: (gp, R))
        monkeypatch.setattr(arm, "_eval_strat_ew", fake_eval)

        cfg = StrategyConfig()

        # False（默认）: 情景键集合 == STRESS_SCENARIOS 键集合（v4.3 行为）
        sc0 = arm.robustness_score(cfg, seeds=(1,), include_corr_scenarios=False)
        assert set(sc0["scenarios"].keys()) == set(arm.STRESS_SCENARIOS.keys())
        assert not any(p.get("dgp") == "regime_corr" for p in calls), (
            "默认路径不应触发 regime_corr DGP"
        )

        # True: 情景键集合 == 并集，corr 情景走 regime_corr DGP 且机制标注 corr_crisis
        calls.clear()
        sc1 = arm.robustness_score(cfg, seeds=(1,), include_corr_scenarios=True)
        expected = set(arm.STRESS_SCENARIOS.keys()) | set(arm.CORR_STRESS_SCENARIOS.keys())
        assert set(sc1["scenarios"].keys()) == expected
        n_regime = sum(1 for p in calls if p.get("dgp") == "regime_corr")
        assert n_regime == len(arm.CORR_STRESS_SCENARIOS)
        for name in arm.CORR_STRESS_SCENARIOS:
            assert sc1["scenarios"][name]["mechanism"] == "corr_crisis"


# ===================================================================
# 6. v4_3 / v4_4 YAML 配置加载
# ===================================================================

class TestConfigLoads:
    def test_v4_3_and_v4_4_config_loads(self):
        """v4_3 无 crisis_correlation_ewma 段 → 默认关; v4_4 开启且圆整参数就位。"""
        cfg43 = load_config(PROJECT / "config" / "strategy_v4_3.yaml")
        assert cfg43.crisis_corr_ewma_enabled is False
        assert cfg43.crisis_corr_ewma_halflife == 8

        cfg44 = load_config(PROJECT / "config" / "strategy_v4_4.yaml")
        assert cfg44.crisis_corr_ewma_enabled is True
        assert cfg44.crisis_corr_ewma_halflife == 8
        # v4.4 圆整防御参数
        assert cfg44.def_alloc == 0.35
        assert cfg44.max_def == 0.83


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
