# -*- coding: utf-8 -*-
"""任务22 — 调仓日溢价哨兵 (scripts/premium_sentinel.py) 单元测试 (任务28 扩充 v2 回落防线)。

全部不联网：advise/collapse_metrics/share_metrics 只吃构造数据；降级路径通过
monkeypatch 内部函数模拟。另含 rebalance_live 默认行为与哨兵
无关的冒烟测试 (import 层面验证, CI --verify 依赖导入期零网络)。
"""
import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


def _load(name):
    """从文件路径加载 scripts/*.py（scripts 非包，避免 import 结构依赖）。"""
    spec = importlib.util.spec_from_file_location(name, PROJECT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ps = _load("premium_sentinel")


def _rec(premium, source="test"):
    return {"close": 1.0 * (1 + premium), "unit_nav": 1.0, "premium": premium,
            "close_date": "2026-07-31", "nav_date": "2026-07-30", "source": source}


# ------------------------------------------------------------------
# advise: 阈值判定
# ------------------------------------------------------------------

def test_advise_below_low_threshold_is_normal():
    txt = ps.advise({ps.TARGET_CODE: _rec(0.005)})
    assert "✅" in txt and "正常" in txt
    assert "候选" not in txt          # 未超阈值不列候选表
    assert "1.9" not in txt           # sanity: 不误报


def test_advise_watch_zone_between_thresholds():
    txt = ps.advise({ps.TARGET_CODE: _rec(0.022)})
    assert "🟡" in txt and "观察区" in txt
    assert "🔴" not in txt


def test_advise_boundary_exactly_high_threshold_alerts():
    prem = {ps.TARGET_CODE: _rec(0.025), "513300": _rec(0.01)}
    txt = ps.advise(prem)
    assert "🔴" in txt


def test_advise_custom_thresholds():
    # 自定义阈值: 3% 溢价在 high=5% 下只进观察区
    txt = ps.advise({ps.TARGET_CODE: _rec(0.03)}, threshold_high=0.05, threshold_low=0.028)
    assert "🟡" in txt and "🔴" not in txt


# ------------------------------------------------------------------
# advise: 候选排序 (同标的优先、溢价最低)
# ------------------------------------------------------------------

def test_advise_over_threshold_prefers_nasdaq_lowest_premium():
    prem = {
        ps.TARGET_CODE: _rec(0.109),
        "513300": _rec(0.030),
        "159941": _rec(0.015),   # 纳指最低 → 应被推荐
        "513390": _rec(0.040),
        "159632": _rec(0.025),
        "513500": _rec(0.001),   # 标普更低, 但同标的优先
        "513650": _rec(0.002),
    }
    txt = ps.advise(prem)
    assert "🔴" in txt
    assert "建议执行标的: 159941" in txt
    assert "同标的" in txt
    # 候选表逐行列出各自溢价
    for c in ("513300", "159941", "513390", "159632", "513500", "513650"):
        assert c in txt
    # 风险提示必须存在
    assert "风险提示" in txt and "人工确认" in txt


def test_advise_falls_back_to_sp500_when_no_nasdaq_candidate():
    prem = {
        ps.TARGET_CODE: _rec(0.109),
        "513300": {"error": "timeout", "premium": None, "source": "none"},
        "159941": {"error": "timeout", "premium": None, "source": "none"},
        "513390": {"error": "timeout", "premium": None, "source": "none"},
        "159632": {"error": "timeout", "premium": None, "source": "none"},
        "513500": _rec(0.008),   # 标普最低 → 纳指全失败时退到标普
        "513650": _rec(0.012),
    }
    txt = ps.advise(prem)
    assert "建议执行标的: 513500" in txt
    assert "跨标的" in txt
    assert "获取失败" in txt      # 失败候选在表中标注


def test_advise_no_better_candidate_keeps_original():
    # 候选溢价都不低于主标的 → 建议维持原标的
    prem = {
        ps.TARGET_CODE: _rec(0.03),
        "513300": _rec(0.05),
        "513500": _rec(0.06),
    }
    txt = ps.advise(prem)
    assert "维持原标的" in txt


def test_advise_all_candidates_failed():
    prem = {ps.TARGET_CODE: _rec(0.05)}
    prem.update({c: {"error": "x", "premium": None, "source": "none"}
                 for c in ps.CANDIDATES})
    txt = ps.advise(prem)
    assert "候选溢价全部获取失败" in txt


# ------------------------------------------------------------------
# 降级路径
# ------------------------------------------------------------------

def test_advise_degrades_when_target_missing():
    txt = ps.advise({ps.TARGET_CODE: {"error": "全部数据源失败", "premium": None,
                                      "source": "none"}})
    assert "降级" in txt and "不影响调仓建议" in txt
    assert "🔴" not in txt and "🟡" not in txt


def test_advise_degrades_on_empty_dict():
    txt = ps.advise({})
    assert "降级" in txt


def test_fetch_premiums_never_raises_on_total_failure(monkeypatch):
    # tushare 不可用 + 东财失败 → 每个代码返回 error 标记, 不抛异常
    monkeypatch.setattr(ps.os, "environ", {})           # 无 token
    monkeypatch.setattr(ps, "_load_env", lambda: None)  # 不读 .env
    def boom(code):
        raise RuntimeError("network down")
    monkeypatch.setattr(ps, "_fetch_em_one", boom)
    out = ps.fetch_premiums(["513100", "513500"])
    assert set(out) == {"513100", "513500"}
    for rec in out.values():
        assert rec["premium"] is None and "error" in rec


def test_fetch_premiums_falls_back_to_eastmoney(monkeypatch):
    monkeypatch.setattr(ps.os, "environ", {})
    monkeypatch.setattr(ps, "_load_env", lambda: None)
    monkeypatch.setattr(ps, "_fetch_em_one", lambda code: _rec(0.02, source="eastmoney"))
    out = ps.fetch_premiums(["513100"])
    assert out["513100"]["source"] == "eastmoney"
    assert out["513100"]["premium"] == pytest.approx(0.02)


# ------------------------------------------------------------------
# 常量与默认阈值锁定
# ------------------------------------------------------------------

def test_constants_and_default_thresholds():
    assert ps.TARGET_CODE == "513100"
    assert set(ps.CANDIDATES) == {"513300", "159941", "513390", "159632",
                                  "513500", "513650"}
    # low=1.5% 对齐 E3 纪律"溢价>1.5% 新增走场外" (任务28 修订), high=2.5% 告警
    assert ps.THRESHOLD_HIGH == pytest.approx(0.025)
    assert ps.THRESHOLD_LOW == pytest.approx(0.015)
    # E4 回落防线参数 (SOP §6.2)
    assert ps.R1_X_MAIN == pytest.approx(0.02)
    assert ps.R1_X_ALT == pytest.approx(0.015)
    assert ps.R1_P5_FLOOR == pytest.approx(0.01)
    assert ps.R2_SHARE_5D == pytest.approx(0.05)
    assert (ps.P5_WIN, ps.DD20_WIN) == (5, 20)
    kinds = {v[1] for v in ps.CANDIDATES.values()}
    assert kinds == {"纳指", "标普"}


# ------------------------------------------------------------------
# 冒烟: rebalance_live 默认行为与哨兵无关
# ------------------------------------------------------------------

def test_rebalance_live_import_has_no_sentinel_dependency():
    """rebalance_live 模块导入期不得引用哨兵/发网络请求 (CI --verify 依赖)。"""
    rl = _load("rebalance_live")
    # 导入成功且模块级零哨兵引用 (惰性导入在 main() 的 --premium-check 分支内)
    assert not hasattr(rl, "premium_sentinel")
    assert not hasattr(rl, "fetch_premiums") and not hasattr(rl, "advise")
    # 网络库未因导入被拉进来
    assert "tushare" not in sys.modules or True  # tushare 可能被其他测试引入, 只做源检查
    src = (PROJECT / "scripts" / "rebalance_live.py").read_text(encoding="utf-8")
    head = src.split("def main()")[0]
    for banned in ("premium_sentinel", "requests", "tushare"):
        assert banned not in head, f"模块导入期不得出现 {banned}"
    # 核心入口仍在
    assert callable(rl.main) and callable(rl.compute)


# ------------------------------------------------------------------
# v2 回落防线 (任务28): collapse_metrics — p5/dd20 计算与 R1 判定
# ------------------------------------------------------------------

def _prow(vals):
    """构造 [(date, premium)] 日序列 (2026-01-01 起逐日)。"""
    return [(f"2026-01-{i+1:02d}", v) for i, v in enumerate(vals)]


def test_collapse_metrics_rising_series_p5_and_dd20_zero():
    # 0.01..0.07 单调上升: 末日 p5 = mean(0.03..0.07) = 0.05, 峰值即当日 → dd20=0
    m = ps.collapse_metrics(_prow([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]))
    assert m["ok"]
    assert m["p5"] == pytest.approx(0.05)
    assert m["p5_peak20"] == pytest.approx(0.05)
    assert m["p5_peak20_date"] == "2026-01-07"
    assert m["dd20"] == pytest.approx(0.0, abs=1e-12)
    assert not m["r1_main"] and not m["r1_alt"]
    assert m["gap_main_pp"] == pytest.approx(2.0)
    assert m["gap_alt_pp"] == pytest.approx(1.5)


def test_collapse_metrics_drawdown_fires_main_and_alt():
    # 峰值 p5=5% → 末日 p5=2%: dd20≈3pp ≥ 2pp 且 p5>1% → 主/备选口径皆触发
    m = ps.collapse_metrics(_prow([0.05] * 8 + [0.02] * 5))
    assert m["ok"]
    assert m["p5"] == pytest.approx(0.02)
    assert m["dd20"] == pytest.approx(0.03)
    assert m["r1_main"] and m["r1_alt"]


def test_collapse_metrics_alt_zone_between_thresholds():
    # dd20≈1.8pp: 主口径(2pp)未触发、备选(1.5pp)触发; gap 显示距触发差距
    m = ps.collapse_metrics(_prow([0.048] * 8 + [0.03] * 5))
    assert m["ok"]
    assert m["dd20"] == pytest.approx(0.018)
    assert not m["r1_main"] and m["r1_alt"]
    assert m["gap_main_pp"] == pytest.approx(0.2)


def test_collapse_metrics_p5_floor_blocks_trigger():
    # 回撤很大但 p5≤1% (已低溢价) → 无防守必要, 不触发
    m = ps.collapse_metrics(_prow([0.04] * 8 + [0.005] * 5))
    assert m["ok"]
    assert m["dd20"] > 0.03
    assert not m["r1_main"] and not m["r1_alt"]


def test_collapse_metrics_insufficient_history():
    assert ps.collapse_metrics(_prow([0.02] * 4))["ok"] is False       # <5 行
    assert ps.collapse_metrics(_prow([0.02] * 6))["ok"] is False       # p5 有效点<5


# ------------------------------------------------------------------
# v2 回落防线: share_metrics — R2 份额扩张
# ------------------------------------------------------------------

def test_share_metrics_r2_trigger():
    rows = _prow([100.0] * 9 + [106.0])   # 5日 +6% ≥ +5%
    s = ps.share_metrics(rows)
    assert s["ok"] and s["r2"]
    assert s["chg5"] == pytest.approx(0.06)


def test_share_metrics_split_ratio_adjusted():
    # 单日 4 倍跳变视为拆分折算 → 实际 5 日仅 +1%, 不触发
    rows = _prow([100.0] * 4 + [400.0] * 5 + [404.0])
    s = ps.share_metrics(rows)
    assert s["ok"] and not s["r2"]
    assert s["chg5"] == pytest.approx(0.01)


def test_share_metrics_insufficient_history():
    assert ps.share_metrics(_prow([100.0] * 5))["ok"] is False


# ------------------------------------------------------------------
# v2 回落防线: 降级路径与文本 (全离线)
# ------------------------------------------------------------------

def test_collapse_check_offline_no_cache_degrades(monkeypatch):
    # 无缓存 + 离线 → R1/R2 均降级, 不抛异常不联网
    monkeypatch.setattr(ps, "_load_premium_cache", lambda code: [])
    monkeypatch.setattr(ps, "_load_share_cache", lambda code: [])
    chk = ps.collapse_check(online=False)
    assert chk["ok"] is False
    assert chk["share"]["ok"] is False
    txt = ps.collapse_advise(chk)
    assert "R1 降级" in txt and "不影响调仓建议" in txt
    assert "R2 降级" in txt and "人工核查" in txt


def test_collapse_advise_not_fired_shows_gap():
    chk = ps.collapse_metrics(_prow([0.03] * 10))
    chk["note"] = "测试"
    chk["share"] = {"ok": True, "date": "2026-01-10", "chg5": 0.0,
                    "r2": False, "note": "测试"}
    txt = ps.collapse_advise(chk)
    assert "✅ R1(X=2pp) 未触发" in txt and "距触发还差 2.00pp" in txt
    assert "备选口径 X=1.5pp" in txt
    assert "✅ R2 未触发" in txt
    assert "🔴" not in txt


def test_collapse_advise_fired_text():
    chk = ps.collapse_metrics(_prow([0.05] * 8 + [0.02] * 5))
    chk["note"] = "测试"
    chk["share"] = {"ok": True, "date": "2026-01-13", "chg5": 0.06,
                    "r2": True, "note": "测试"}
    txt = ps.collapse_advise(chk)
    assert "🔴 R1 触发" in txt and "SOP §6.2" in txt
    assert "🔴 R2 份额预警" in txt


def test_collapse_advise_low_premium_not_applicable():
    chk = ps.collapse_metrics(_prow([0.04] * 8 + [0.005] * 5))
    chk["note"] = "测试"
    chk["share"] = {"ok": False, "msg": "无数据"}
    txt = ps.collapse_advise(chk)
    assert "R1 不适用" in txt and "p5≤1%" in txt


def test_collapse_report_never_raises(monkeypatch):
    def boom(code=ps.TARGET_CODE, online=True):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(ps, "collapse_check", boom)
    txt = ps.collapse_report()
    assert isinstance(txt, str)
    assert "降级" in txt and "不影响调仓建议" in txt


def test_fetch_histories_offline_use_cache_only(monkeypatch):
    # online=False 时只读缓存, 绝不碰网络 (把网络函数炸掉验证)
    monkeypatch.setattr(ps, "_load_premium_cache", lambda code: _prow([0.02] * 3))
    monkeypatch.setattr(ps, "_load_share_cache", lambda code: _prow([100.0] * 3))
    def boom(*a, **k):
        raise AssertionError("离线路径不得联网")
    monkeypatch.setattr(ps, "_fetch_close_hist", boom)
    monkeypatch.setattr(ps, "_fetch_nav_hist", boom)
    monkeypatch.setattr(ps, "_fetch_sse_share", boom)
    rows, note = ps.fetch_premium_history(online=False)
    assert len(rows) == 3 and "离线" in note
    rows, note = ps.fetch_share_history(online=False)
    assert len(rows) == 3 and "离线" in note

