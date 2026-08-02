"""回归测试 — scripts/benchmark_compare.py 三方基准净值口径锁定。

目的：把"策略 / 每周再平衡等权 / 真·买入持有"三方在全期窗口的关键指标钉住，
防止引擎、数据清洗或基准构造被改动后悄悄改变基准口径而无人察觉。
同时把核心结论（买入持有收益更高但风险更差）编码为断言。
"""
import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

# 从文件路径加载 scripts/benchmark_compare.py（scripts 非包，避免 import 结构依赖）
_spec = importlib.util.spec_from_file_location(
    "benchmark_compare", PROJECT / "scripts" / "benchmark_compare.py")
_bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bc)

from src.strategy import load_config


@pytest.fixture(scope="module")
def full():
    cfg = load_config(PROJECT / "config/strategy_v4_1.yaml")
    return _bc.compute_benchmarks(cfg)


def test_window_is_666_weeks(full):
    # 数据窗口至 2026-07-31 (678 周 NAV, 666 有效周)
    assert full["window"]["weeks"] == 666, "有效周数变化"
    # start date shifts with EWMA (longer warmup)
    assert full["window"]["start"] == "2013-08-09"
    assert full["window"]["n_valid_etf"] == 5


def test_strategy_metrics_reproduce(full):
    m = full["strategy"]
    assert m["sharpe_ratio"] == pytest.approx(1.610, abs=0.01)
    assert m["annual_return"] == pytest.approx(0.1705, abs=0.003)
    assert m["max_drawdown"] == pytest.approx(0.0697, abs=0.002)


def test_buyhold_higher_return_worse_risk_than_rebalance(full):
    """核心结论：真·买入持有收益更高，但回撤更深、Sharpe 更低。"""
    bh, rb = full["buy_hold"], full["ew_rebalanced"]
    assert bh["total_return"] > rb["total_return"]      # 收益：买入持有更高
    assert bh["max_drawdown"] > rb["max_drawdown"]      # 回撤：买入持有更深
    assert bh["sharpe_ratio"] < rb["sharpe_ratio"]      # 风险调整：买入持有更差


def test_strategy_dominates_both_risk_adjusted(full):
    s = full["strategy"]
    assert s["sharpe_ratio"] > full["ew_rebalanced"]["sharpe_ratio"]
    assert s["sharpe_ratio"] > full["buy_hold"]["sharpe_ratio"]
    assert s["max_drawdown"] < full["buy_hold"]["max_drawdown"]


def test_benchmark_absolute_levels(full):
    """锁定两个基准的关键绝对值（宽容差，防口径漂移）。"""
    rb, bh = full["ew_rebalanced"], full["buy_hold"]
    assert rb["sharpe_ratio"] == pytest.approx(0.918, abs=0.03)
    assert rb["max_drawdown"] == pytest.approx(0.202, abs=0.02)
    assert bh["sharpe_ratio"] == pytest.approx(0.818, abs=0.03)
    assert bh["max_drawdown"] == pytest.approx(0.301, abs=0.02)


# --- WF 基准对比：策略 vs 9 个滚动窗口的 rebal / buyhold ---

@pytest.fixture(scope="module")
def wf_result():
    cfg = load_config(PROJECT / "config/strategy_v4_1.yaml")
    import importlib.util as _iu
    _s = _iu.spec_from_file_location("wf", PROJECT / "scripts" / "run_walkforward.py")
    _m = _iu.module_from_spec(_s)
    _s.loader.exec_module(_m)
    return _m.benchmark_wf(cfg, n_windows=10)


def test_wf_vs_ew_rebal_wins(wf_result):
    """对每周再平衡等权：9 窗至少 6 窗 Sharpe 胜出。"""
    assert wf_result["wins_vs_ew_rebal"] >= 6
    assert wf_result["n_test_windows"] == 9


def test_wf_vs_buyhold_wins(wf_result):
    """对真·买入持有：9 窗至少 7 窗 Sharpe 胜出（当前实测 8/9）。"""
    assert wf_result["wins_vs_buyhold"] >= 7


def test_wf_strategy_avg_sharpe_dominates(wf_result):
    s = wf_result["avg_strategy_sharpe"]
    assert s > wf_result["avg_ew_rebal_sharpe"]
    assert s > wf_result["avg_buyhold_sharpe"]
