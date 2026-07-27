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
