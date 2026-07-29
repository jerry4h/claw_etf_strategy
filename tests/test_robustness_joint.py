"""联合鲁棒性检验 pin 测试 — 保护 scripts/robustness_joint.py 与方法学一致性。

不是完整回归（不会跑 445 次回测），只固定核心 API 与关键数字骨架：
  1. ACTIVE_PARAMS 定义 8 个活参（避免误删）
  2. perturb_single 对连续/离散参的扰动逻辑正确
  3. lhs_signed 采样落在 [-1, +1]^dim 且每维每层均匀
  4. v4.3 首次基线（2026-07-29）关键数字 pin，防止上游漂移
     基线 Sharpe/MaxDD/annual、Test 1 max drop、Test 3 联合/边缘比率、alpha 胜率

复跑完整检验：`python scripts/robustness_joint.py --test all --n 200`
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


def _load_rj():
    spec = importlib.util.spec_from_file_location("rj", PROJECT / "scripts" / "robustness_joint.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_active_params_has_eight_taper_knobs():
    """ACTIVE_PARAMS 必须包含 SPACE_TAPER 7 主控 + mom_window。"""
    rj = _load_rj()
    names = [t[0] for t in rj.ACTIVE_PARAMS]
    expected = {"max_def", "def_alloc", "top_n", "step_low", "step_high",
                "vol_taper_window", "vol_taper_len", "mom_window"}
    assert set(names) == expected, f"ACTIVE_PARAMS 结构漂移: {set(names) ^ expected}"


def test_perturb_single_rel_continuous():
    """连续参数 rel 扰动: cur * (1 + delta)，命中边界返回 (None, cur, False)。"""
    rj = _load_rj()
    from src.strategy import load_config
    cfg = load_config(PROJECT / "config/strategy_v4_3.yaml")

    # def_alloc 0.3492 * 1.10 = 0.38412, 在 [0.10, 0.50] 内
    new_cfg, new_val, applied = rj.perturb_single(cfg, "def_alloc", 0.10, 0.50, False, "rel", 0.10)
    assert applied is True
    assert abs(new_val - 0.3492 * 1.10) < 1e-9

    # 0.10 * (1+0) = 0.10 命中下界应等于 cur ⇒ 无变化
    # 但 def_alloc = 0.3492 ≠ 0.10, delta=0 会导致 rel_new = 0.3492, 与 cur 相等
    _, _, applied2 = rj.perturb_single(cfg, "def_alloc", 0.10, 0.50, False, "rel", 0.0)
    assert applied2 is False


def test_perturb_single_abs_int_discrete():
    """离散参数 abs_int: cur + delta, delta=0 不生效。"""
    rj = _load_rj()
    from src.strategy import load_config
    cfg = load_config(PROJECT / "config/strategy_v4_3.yaml")

    # vol_taper_window=14, +2 = 16, 在 [8, 20] 内
    new_cfg, new_val, applied = rj.perturb_single(cfg, "vol_taper_window", 8, 20, True, "abs_int", 2)
    assert applied is True
    assert int(new_val) == 16
    assert new_cfg.inv_vol_window == 16  # 联动

    # 扰动到大值时 taper_len 需自动收
    new_cfg2, _, _ = rj.perturb_single(cfg, "vol_taper_window", 8, 20, True, "abs_int", -6)  # window=8
    assert new_cfg2.vol_taper_len <= new_cfg2.vol_taper_window - 2


def test_lhs_signed_range_and_shape():
    """LHS 采样 [-1, +1]^dim, 每列均值 ≈ 0。"""
    rj = _load_rj()
    U = rj.lhs_signed(200, 8, seed=2026)
    assert U.shape == (200, 8)
    assert U.min() >= -1.0 - 1e-9
    assert U.max() <= 1.0 + 1e-9
    # 每列均值应接近 0（LHS 均匀采样）
    for j in range(8):
        assert abs(U[:, j].mean()) < 0.10, f"col {j} mean drift {U[:, j].mean()}"


def test_v43_joint_robustness_baseline_pinned():
    """v4.3 联合鲁棒性首次基线 (2026-07-29) 关键数字 pin。

    数字取自 output/robustness/robustness_joint_all_20260729_114702.json
    (n=200, block=13, eps=0.10, seed-base=8000, seed=2026)。
    此测试不重新回测，仅从落盘 JSON 校验（若 JSON 缺失则跳过）。
    """
    import json
    fp = PROJECT / "output" / "robustness" / "robustness_joint_all_20260729_114702.json"
    if not fp.exists():
        pytest.skip(f"baseline snapshot 不存在: {fp}")
    d = json.load(open(fp))

    # 基线
    bm = d["base_metrics"]
    assert abs(bm["sharpe"] - 1.4878) < 0.01, f"v4.3 Sharpe 漂移: {bm['sharpe']:.4f}"
    assert abs(bm["maxdd"] - 0.0584) < 0.005
    assert abs(bm["annual"] - 0.1452) < 0.005

    # Test 1 全参 PASS
    for p, v in d["test1_verdict"].items():
        assert v["pass_all"] is True, f"Test 1 {p} FAIL 漂移: {v}"

    # Test 2 相对判据: 96% 胜率、alpha P10 > 0
    t2 = d["test2_rows"]
    diffs = [r["sharpe"] - r["ew_sharpe"] for r in t2]
    win_rate = sum(1 for x in diffs if x > 0) / len(diffs)
    alpha_p10 = float(np.quantile(diffs, 0.10))
    assert win_rate >= 0.90, f"策略跑赢 EW 比率漂移: {win_rate:.3f}"
    assert alpha_p10 > 0, f"alpha P10 漂移: {alpha_p10:.4f}"

    # Test 3 无薄峰
    cmp = d["marginal_vs_joint"]
    assert cmp["no_thin_ridge"] is True
    assert cmp["joint_over_linear_ratio"] <= 1.30, \
        f"薄峰警告: joint/linear = {cmp['joint_over_linear_ratio']:.3f}"

    # 参数×Sharpe 皮尔逊相关全部 |ρ| ≤ 0.30
    t3 = d["test3_rows"]
    y = np.array([r["sharpe"] for r in t3])
    for name, *_ in _load_rj().ACTIVE_PARAMS:
        x = np.array([r["deltas"][name] for r in t3])
        if x.std() > 0:
            rho = float(np.corrcoef(x, y)[0, 1])
            assert abs(rho) <= 0.30, f"{name} × Sharpe |ρ|={abs(rho):.3f} 超阈值 0.30"
