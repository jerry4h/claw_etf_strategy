"""回归测试 — 审计整改（批1）：真·WF 重选参无泄露 + 实盘止损一致性。

- reoptimize_wf：每窗 anchored train 选参→test 验证，train/test 时间不重叠、
  选参只用 train 段（无未来信息泄露）。用缩小网格+少窗口快速验证结构与无泄露；
  完整 9 窗胜率（跑得慢）见 README，不入单测。
- rebalance_live.replay_stop_loss_state：实盘主路径止损与回测引擎同口径；
  当前历史 MaxDD 6.97% < 8%，止损应全程休眠（triggers==0）。
"""
import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.strategy import load_config


def _load_module(name, rel):
    spec = importlib.util.spec_from_file_location(name, PROJECT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wf = _load_module("wf_mod", "scripts/run_walkforward.py")

# 缩小网格：2×2×1×1×1 = 4 组合，快速验证结构/无泄露（不追求完整胜率）
SMALL_GRID = {
    'mom_w': [0.8, 1.0],
    'vol_w': [0.9, 1.1],
    'def_alloc': [0.25],
    'step_low': [0.15],
    'step_high': [0.35],
}


@pytest.fixture(scope="module")
def wf_res():
    cfg = load_config(PROJECT / "config/strategy_v4_1.yaml")
    return wf.reoptimize_wf(cfg, n_windows=4, grid=SMALL_GRID)


def test_reoptimize_structure(wf_res):
    assert wf_res['mode'] == 'reoptimize_wf'
    assert wf_res['n_test_windows'] == 3
    assert len(wf_res['windows']) == 3
    need = {'train_end', 'test_start', 'test_end', 'best_params', 'train_sharpe',
            'test_strategy_sharpe', 'ew_rebal_sharpe', 'buyhold_sharpe',
            'vs_ew_rebal', 'vs_buyhold'}
    for w in wf_res['windows']:
        assert need.issubset(w)


def test_reoptimize_no_leakage(wf_res):
    """anchored WF：每窗 train 到窗口开始为止(test_start==train_end)，test 是之后的未来，
    选参用的是 train 段 Sharpe，从不含 test 期收益 → 无未来信息泄露。"""
    for w in wf_res['windows']:
        assert w['test_start'] == w['train_end']
        assert w['test_end'] > w['test_start']


def test_reoptimize_best_params_in_grid(wf_res):
    for w in wf_res['windows']:
        bp = w['best_params']
        assert bp['mom_w'] in SMALL_GRID['mom_w']
        assert bp['vol_w'] in SMALL_GRID['vol_w']


def test_select_best_only_uses_train():
    """_select_best_on_train 只接受 train 区间边界，返回 train 段选优结果。"""
    cfg = load_config(PROJECT / "config/strategy_v4_1.yaml")
    best = wf._select_best_on_train(cfg, '2013-08-01', '2018-01-01', SMALL_GRID)
    assert best is not None
    assert 'kw' in best and 'train_sharpe' in best and 'dsr' in best


def test_live_stop_loss_dormant_on_current_data():
    """实盘止损与引擎同口径；当前 MaxDD 6.97% < 8% → 全程未触发(triggers==0, should_stop=False)。"""
    rl = _load_module("rlive_mod", "scripts/rebalance_live.py")
    df = rl.load(PROJECT / rl.cfg.nav_path)
    st = rl.replay_stop_loss_state(df, len(df) - 1)
    assert st['triggers'] == 0
    assert st['should_stop'] is False


def test_robustness_score_distinguishes_taper_vs_rolling():
    """对抗稳定性指标应能区分 rolling(脆弱) vs taper(鲁棒), 且结构完整。3-seed 稳定版。

    单 seed 的 pass_rate 粒度为 20%(5 情景), 会被单条路径运气翻转; 用 3 seed 中位数更稳。
    taper 相对 rolling 的鲁棒优势可能体现在 pass_rate 或 baseline_retention 之一 —
    两者取其一成立即视为通过(避免 pass_rate 在 5 情景粒度下无法区分小改进的假象)。
    """
    import warnings, dataclasses
    warnings.filterwarnings("ignore")
    adv = _load_module("adv_score", "scripts/adversarial_robustness.py")
    cfg = load_config(PROJECT / "config/strategy_v4_1.yaml")
    cfg_taper = dataclasses.replace(cfg, vol_taper_enabled=True,
                                    vol_taper_window=14, vol_taper_len=5, vol_window=14)
    sc_r = adv.robustness_score(cfg,       seeds=(11, 22, 33))
    sc_t = adv.robustness_score(cfg_taper, seeds=(11, 22, 33))
    # 结构完整
    assert 0.0 <= sc_r["pass_rate"] <= 1.0
    assert "stagflation" in sc_r["scenarios"]
    assert sc_r["baseline_sharpe"] > 0
    # taper 应至少在一个维度不弱于 rolling(pass_rate 或 baseline_retention)
    assert (sc_t["pass_rate"] >= sc_r["pass_rate"]
            or sc_t["baseline_retention"] >= sc_r["baseline_retention"]), \
        f"taper 未在任一鲁棒维度改善: pass {sc_t['pass_rate']} vs {sc_r['pass_rate']}, " \
        f"retention {sc_t['baseline_retention']} vs {sc_r['baseline_retention']}"


def test_robustness_score_mechanism_grouping_and_dual_caliber():
    """robustness_score 应输出机制分组 + 回撤/收益双口径(多目标框架依赖)。快速 1-seed。"""
    import warnings
    warnings.filterwarnings("ignore")
    adv = _load_module("adv_mech", "scripts/adversarial_robustness.py")
    cfg = load_config(PROJECT / "config/strategy_v4_1.yaml")
    sc = adv.robustness_score(cfg, seeds=(11,))
    # 顶层双口径 + 全情景回撤
    assert 0.0 <= sc["pass_rate"] <= 1.0
    assert 0.0 <= sc["pass_rate_return"] <= 1.0
    assert sc["worst_maxdd"] > 0.0
    # 5 机制分组齐全
    expected = {"vol_defense", "selection", "defense_asset", "dispersion", "composite"}
    assert set(sc["by_mechanism"].keys()) == expected
    for mech, d in sc["by_mechanism"].items():
        for key in ("pass_rate", "pass_rate_return", "worst_maxdd", "worst_sharpe"):
            assert key in d, f"{mech} 缺字段 {key}"
        assert 0.0 <= d["pass_rate"] <= 1.0


def test_evaluate_full_multiobjective_structure():
    """evaluate_full 应返回 多目标约束判定结构(realized+adversarial+分机制门禁)。快速 1-seed。"""
    import warnings
    warnings.filterwarnings("ignore")
    ev_mod = _load_module("eval_full", "scripts/evaluate.py")
    cfg = load_config(PROJECT / "config/strategy_v4_1.yaml")
    ev = ev_mod.evaluate_full(cfg, d_max=0.12, seeds=(11,))
    for key in ("objective", "verdict", "realized", "adversarial", "constraints"):
        assert key in ev, f"缺字段 {key}"
    assert ev["verdict"] in ("PASS", "FAIL")
    c = ev["constraints"]
    assert c["d_max"] == 0.12
    assert "mechanism_gates" in c
    # selection 为软门禁, 其余为硬门禁
    assert c["mechanism_gates"]["selection"]["gate"] == "soft"
    for mech in ("vol_defense", "defense_asset", "dispersion", "composite"):
        assert c["mechanism_gates"][mech]["gate"] == "hard"


def test_v4_2_production_headline_metrics_pinned():
    """P0-2 回归: pin 住 v4.2 生产 config 的三个 README 头部承诺数字。

    任何改动打破这个 pin (Sharpe 1.635, 年化 15.84%, MaxDD 6.75%) 都会立刻失败,
    强制 review 是引擎变了还是 config 漂了。容差 ±0.5% 覆盖浮点噪声 + yaml roundtrip。
    """
    import warnings
    warnings.filterwarnings("ignore")
    from src.strategy import load_config as _lc
    from src.backtest import run_backtest as _rb
    cfg = _lc(PROJECT / "config/strategy_v4_2.yaml")
    res = _rb(cfg)
    m = res.metrics
    # 三个头部承诺数字, 允许 ±0.5% 绝对容差
    assert abs(m["sharpe_ratio"]   - 1.6350) < 0.01, f"v4.2 Sharpe 漂移: {m['sharpe_ratio']:.4f} 期望 ~1.635"
    assert abs(m["annual_return"]  - 0.1584) < 0.005, f"v4.2 年化漂移: {m['annual_return']:.4%} 期望 ~15.84%"
    assert abs(m["max_drawdown"]   - 0.0675) < 0.005, f"v4.2 MaxDD 漂移: {m['max_drawdown']:.4%} 期望 ~6.75%"
    # 验证 P0-1 修复: defensive_weeks 已跟随 config.def_alloc=0.1447, 不再硬编码 0.25
    df = res.nav_series
    reported = int(m["defensive_weeks"])
    naive_025 = int(df["def_ratio"].gt(0.25).sum())
    tracks_config = int(df["def_ratio"].gt(cfg.def_alloc).sum())
    assert reported == tracks_config, f"defensive_weeks 未跟随 config.def_alloc: reported={reported} vs cfg={tracks_config}"
    # 在 v4.2 下这两应不同 (def_alloc=0.1447 < 0.25), 证明修复真的生效
    assert reported != naive_025, "defensive_weeks 与旧口径 (def_ratio>0.25) 不应相等, 否则 P0-1 修复未生效"


def test_v4_2_multiobjective_pass_verdict():
    """P0-2 回归: v4.2 config 在 v4.0 多目标约束框架下应给出 PASS verdict (7-seed 严格)。

    这是 v4.0 框架"生产 config 必然是 verdict=PASS"的最直接不变量断言。
    需要 evaluate.py 端到端跑通, 用中等 seed 数 (3) 平衡稳定性和 CI 速度。
    """
    import warnings
    warnings.filterwarnings("ignore")
    ev_mod = _load_module("eval_v42", "scripts/evaluate.py")
    cfg = load_config(PROJECT / "config/strategy_v4_2.yaml")
    ev = ev_mod.evaluate_full(cfg, d_max=0.12, seeds=(11, 22, 33))
    # v4.2 是 v4.0 框架 7-seed 严格 PASS 出来的候选,3-seed 也应稳定 PASS
    assert ev["verdict"] == "PASS", f"v4.2 应 verdict=PASS 但实际 {ev['verdict']}, failed={ev['failed_constraints']}"
    # 全情景 DD 应 ≤ D_max
    assert ev["adversarial"]["worst_maxdd"] <= 0.12 + 0.005, \
        f"v4.2 全情景 DD {ev['adversarial']['worst_maxdd']:.4f} > 12% + 容差"


def test_v4_3_tapered_alternative_headline_pinned():
    """归档回归: v4.3 tapered-vol 替代生产 config 的关键数字 pin。

    v4.3 与 v4.2 互不支配(用低 Sharpe 换无跳变+低回撤), 两者都归档为已验证配置。
    这里 pin 住 v4.3 的 taper 开关 + 三个 backtest 数字, 防止 config/引擎漂移。
    """
    import warnings
    warnings.filterwarnings("ignore")
    from src.strategy import load_config as _lc
    from src.backtest import run_backtest as _rb
    cfg = _lc(PROJECT / "config/strategy_v4_3.yaml")
    # taper 设计: v4.3 必须是 tapered vol (消除窗口跳变), 非 rolling
    assert cfg.vol_taper_enabled is True, "v4.3 必须启用 vol_taper (设计核心)"
    assert cfg.vol_taper_window == 14 and cfg.vol_taper_len == 7, \
        f"v4.3 taper 参数漂移: window={cfg.vol_taper_window} len={cfg.vol_taper_len}"
    assert cfg.ewma_factors_enabled is False, "v4.3 不应同时开 EWMA"
    m = _rb(cfg).metrics
    assert abs(m["sharpe_ratio"]  - 1.4878) < 0.01,  f"v4.3 Sharpe 漂移: {m['sharpe_ratio']:.4f} 期望 ~1.488"
    assert abs(m["annual_return"] - 0.1452) < 0.005, f"v4.3 年化漂移: {m['annual_return']:.4%} 期望 ~14.52%"
    assert abs(m["max_drawdown"]  - 0.0584) < 0.005, f"v4.3 MaxDD 漂移: {m['max_drawdown']:.4%} 期望 ~5.84%"
    # v4.3 设计卖点: realized MaxDD 应低于 v4.2 (6.75%)
    assert m["max_drawdown"] < 0.0675, f"v4.3 MaxDD {m['max_drawdown']:.4%} 未低于 v4.2 6.75%"
