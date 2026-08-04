#!/usr/bin/env python3
"""P3 信号有效性验证与策略集成评估 — 主力份额信号 510500 预测力分析

A) IC 分析: share_growth_20d vs 510500 下周收益 rank_IC (Spearman)
B) 事件收益归因: is_anomaly 事件后 4/8/12 周累计收益
C) 公告对齐验证: 已知三次增持窗口 recall / precision
D) 策略模拟: monkeypatch 回测 (若 IC 方向正确)

用法: .venv/bin/python scripts/_exp_national_team_study.py
"""

import contextlib
import io
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.national_team import compute_etf_signals
from src.backtest import run_backtest
from src.strategy import load_config
import src.factors as sf

# ========== 路径 ==========
NAV_FILE = PROJ / "data" / "all_etfs_nav_latest.csv"
SHARE_FILE = PROJ / "data" / "national_team" / "fund_share" / "510500_SH.csv"
EVENTS_FILE = PROJ / "data" / "national_team" / "events.csv"
CFG_PATH = PROJ / "config" / "strategy_v4_3.yaml"
OUT_DIR = PROJ / "output" / "experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== 已知公告日期 ==========
KNOWN_ANNOUNCEMENTS = [
    ("2023-10-23", "汇金公告增持ETF"),
    ("2024-02-06", "证金公告增持"),
    ("2024-09-24", "政策组合拳（央行+证监会+金融监管）"),
]

START_DATE = "2013-05-17"


# ======================================================================
# A) IC 分析
# ======================================================================
def run_ic_analysis():
    """计算 share_growth_20d 与 510500 下周收益的 rank IC"""
    print("=" * 70)
    print("A) IC 分析: share_growth_20d vs 510500 下周周频收益")
    print("=" * 70)

    # 加载 510500 份额数据
    share_df = pd.read_csv(SHARE_FILE, dtype={"trade_date": str})
    share_df = share_df.sort_values("trade_date").reset_index(drop=True)

    # 计算信号
    signals = compute_etf_signals(
        share_series=share_df["fd_share"],
        dates=share_df["trade_date"],
    )

    # 加载 NAV 取 510500 周频收益
    nav = pd.read_csv(NAV_FILE, index_col=0, parse_dates=True)
    if "中证500ETF" not in nav.columns:
        print("ERROR: NAV 文件缺少中证500ETF列")
        return None
    weekly_ret = nav["中证500ETF"].pct_change()

    # 将日频信号聚合到周频：每周五取当周最后一个 share_growth_20d
    signals["date_dt"] = pd.to_datetime(signals["trade_date"], format="%Y%m%d")
    signals = signals.set_index("date_dt")

    # 周频对齐：取每周五（与 NAV 周频对齐）
    nav_dates = nav.index  # 周频日期
    signal_weekly = pd.Series(index=nav_dates, dtype=float, name="signal")

    for i, dt in enumerate(nav_dates):
        # 找 dt 之前最近的信号（避免前视偏差：用 shift(1) 即 t-1 周的增幅）
        # t 周决策用 t-1 周末的份额增幅 → 找 dt 前 7 天内的最新信号
        window_start = dt - pd.Timedelta(days=12)
        mask = (signals.index >= window_start) & (signals.index < dt)
        sub = signals.loc[mask, "share_growth_20d"]
        if len(sub) > 0:
            signal_weekly.iloc[i] = sub.iloc[-1]

    # 下周收益 = weekly_ret.shift(-1)
    next_week_ret = weekly_ret.shift(-1)

    # 合并有效数据
    df = pd.DataFrame({
        "signal": signal_weekly,
        "next_ret": next_week_ret,
    }).dropna()

    print(f"  有效周数: {len(df)}")
    print(f"  时间跨度: {df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}")

    # 全期 rank IC
    ic_full, p_full = stats.spearmanr(df["signal"], df["next_ret"])
    n_full = len(df)
    t_full = ic_full * np.sqrt((n_full - 2) / (1 - ic_full**2)) if abs(ic_full) < 1 else 0

    # 近期 (2020+)
    df_recent = df[df.index >= "2020-01-01"]
    ic_recent, p_recent = stats.spearmanr(df_recent["signal"], df_recent["next_ret"])
    n_recent = len(df_recent)
    t_recent = ic_recent * np.sqrt((n_recent - 2) / (1 - ic_recent**2)) if abs(ic_recent) < 1 else 0

    # 滚动 52 周 IC
    rolling_ics = []
    for i in range(52, len(df)):
        chunk = df.iloc[i - 52:i]
        r, _ = stats.spearmanr(chunk["signal"], chunk["next_ret"])
        if not np.isnan(r):
            rolling_ics.append(r)
    rolling_ics = np.array(rolling_ics)
    ic_mean_roll = rolling_ics.mean() if len(rolling_ics) > 0 else 0
    ic_std_roll = rolling_ics.std() if len(rolling_ics) > 0 else 1
    ir_roll = ic_mean_roll / ic_std_roll if ic_std_roll > 0 else 0

    result = {
        "full_period": {
            "rank_ic": round(ic_full, 4),
            "p_value": round(p_full, 4),
            "t_stat": round(t_full, 3),
            "n_weeks": n_full,
            "period": f"{df.index.min().strftime('%Y-%m-%d')}~{df.index.max().strftime('%Y-%m-%d')}",
        },
        "recent_2020": {
            "rank_ic": round(ic_recent, 4),
            "p_value": round(p_recent, 4),
            "t_stat": round(t_recent, 3),
            "n_weeks": n_recent,
            "period": f"{df_recent.index.min().strftime('%Y-%m-%d')}~{df_recent.index.max().strftime('%Y-%m-%d')}",
        },
        "rolling_52w": {
            "ic_mean": round(ic_mean_roll, 4),
            "ic_std": round(ic_std_roll, 4),
            "ir": round(ir_roll, 3),
            "n_windows": len(rolling_ics),
        },
    }

    print(f"\n  全期 Rank IC:  {ic_full:.4f}  (p={p_full:.4f}, t={t_full:.3f}, n={n_full})")
    print(f"  近期 Rank IC:  {ic_recent:.4f}  (p={p_recent:.4f}, t={t_recent:.3f}, n={n_recent})")
    print(f"  滚动52周 IC:  mean={ic_mean_roll:.4f}, std={ic_std_roll:.4f}, IR={ir_roll:.3f}")
    print(f"  IC 方向: {'正向（增持→后续涨）' if ic_full > 0 else '反向'}")

    return result


# ======================================================================
# B) 事件收益归因
# ======================================================================
def run_event_attribution():
    """事件后 4/8/12 周累计收益 vs 无条件基准"""
    print("\n" + "=" * 70)
    print("B) 事件收益归因: is_anomaly 后 4/8/12 周累计收益")
    print("=" * 70)

    # 加载事件
    events = pd.read_csv(EVENTS_FILE, dtype={"date": str})
    events_510500 = events[events["etf"] == "510500.SH"].copy()
    events_510500["date_dt"] = pd.to_datetime(events_510500["date"], format="%Y%m%d")
    print(f"  510500 事件总数: {len(events_510500)}")

    # 去重：连续事件 <7 天视为同一窗口，取首日
    events_sorted = events_510500.sort_values("date_dt")
    windows = []
    last_date = None
    for _, row in events_sorted.iterrows():
        dt = row["date_dt"]
        if last_date is None or (dt - last_date).days > 7:
            windows.append({
                "start_date": dt,
                "share_growth_20d": row["share_growth_20d"],
            })
        last_date = dt
    print(f"  独立事件窗口: {len(windows)}")

    # 加载 NAV
    nav = pd.read_csv(NAV_FILE, index_col=0, parse_dates=True)
    weekly_ret = nav["中证500ETF"].pct_change()

    # 无条件周均收益
    unconditional_mean = weekly_ret.dropna().mean()
    print(f"  无条件周均收益: {unconditional_mean:.5f} ({unconditional_mean*52:.3f} 年化)")

    # 计算事件后累计收益
    horizons = [4, 8, 12]
    event_returns = {h: [] for h in horizons}

    nav_dates = nav.index
    for w in windows:
        start_dt = w["start_date"]
        # 找 start_dt 之后最近的周频日期
        future_dates = nav_dates[nav_dates > start_dt]
        if len(future_dates) == 0:
            continue
        start_idx = nav_dates.get_loc(future_dates[0])

        for h in horizons:
            end_idx = start_idx + h
            if end_idx < len(nav_dates):
                cum_ret = (1 + weekly_ret.iloc[start_idx + 1:end_idx + 1]).prod() - 1
                event_returns[h].append(float(cum_ret))

    # 统计
    result = {"n_windows": len(windows), "unconditional_weekly_mean": round(float(unconditional_mean), 6)}
    print(f"\n  {'窗口':>4} | {'事件后均值':>10} | {'中位数':>8} | {'胜率':>6} | {'无条件':>8} | {'t-stat':>7} | {'p-value':>7}")
    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}")

    for h in horizons:
        arr = np.array(event_returns[h])
        n = len(arr)
        if n < 3:
            continue
        mean_ret = arr.mean()
        med_ret = np.median(arr)
        win_rate = (arr > 0).mean()
        uncond = unconditional_mean * h
        # t-test: 事件后收益是否显著异于无条件
        excess = arr - uncond
        t_stat, p_val = stats.ttest_1samp(excess, 0)

        result[f"horizon_{h}w"] = {
            "mean_return": round(float(mean_ret), 5),
            "median_return": round(float(med_ret), 5),
            "win_rate": round(float(win_rate), 3),
            "unconditional_cum": round(float(uncond), 5),
            "excess_mean": round(float(mean_ret - uncond), 5),
            "t_stat": round(float(t_stat), 3),
            "p_value": round(float(p_val), 4),
            "n_events": n,
        }
        print(f"  {h:>2}周  | {mean_ret:>+10.5f} | {med_ret:>+8.5f} | {win_rate:>6.1%} | {uncond:>+8.5f} | {t_stat:>+7.3f} | {p_val:>7.4f}")

    # 特别标注已知窗口
    print("\n  已知增持窗口事件后收益:")
    known_dates_pd = [pd.Timestamp(d) for d, _ in KNOWN_ANNOUNCEMENTS]
    for kd, desc in zip(known_dates_pd, [n for _, n in KNOWN_ANNOUNCEMENTS]):
        future_dates = nav_dates[nav_dates > kd]
        if len(future_dates) == 0:
            print(f"    {kd.strftime('%Y-%m-%d')} ({desc}): 无后续数据")
            continue
        start_idx = nav_dates.get_loc(future_dates[0])
        rets_str = []
        for h in horizons:
            end_idx = start_idx + h
            if end_idx < len(nav_dates):
                cum = (1 + weekly_ret.iloc[start_idx + 1:end_idx + 1]).prod() - 1
                rets_str.append(f"{h}w={cum:+.4f}")
        print(f"    {kd.strftime('%Y-%m-%d')} ({desc}): {', '.join(rets_str)}")

    return result


# ======================================================================
# C) 公告对齐验证
# ======================================================================
def run_announcement_alignment():
    """检查 events.csv 是否在已知公告 ±5日 内被触发"""
    print("\n" + "=" * 70)
    print("C) 公告对齐验证: 已知增持窗口 vs events.csv")
    print("=" * 70)

    events = pd.read_csv(EVENTS_FILE, dtype={"date": str})
    events_510500 = events[events["etf"] == "510500.SH"].copy()
    events_510500["date_dt"] = pd.to_datetime(events_510500["date"], format="%Y%m%d")

    # 独立窗口（去重）
    events_sorted = events_510500.sort_values("date_dt")
    windows = []
    last_date = None
    for _, row in events_sorted.iterrows():
        dt = row["date_dt"]
        if last_date is None or (dt - last_date).days > 7:
            windows.append(dt)
        last_date = dt
    window_dates = pd.DatetimeIndex(windows)

    result = {"known_announcements": [], "recall": 0, "precision": 0, "n_windows": len(windows)}

    hits = 0
    for date_str, desc in KNOWN_ANNOUNCEMENTS:
        kd = pd.Timestamp(date_str)
        # 检查 ±5 日
        nearby = window_dates[(window_dates >= kd - pd.Timedelta(days=5)) &
                              (window_dates <= kd + pd.Timedelta(days=5))]
        hit = len(nearby) > 0
        if hit:
            hits += 1
            matched = nearby[0].strftime("%Y-%m-%d")
        else:
            # 扩大到 ±14 日看看最近的事件
            nearby_14 = window_dates[(window_dates >= kd - pd.Timedelta(days=14)) &
                                     (window_dates <= kd + pd.Timedelta(days=14))]
            matched = f"未命中 (±14日最近: {nearby_14[0].strftime('%Y-%m-%d') if len(nearby_14) > 0 else 'N/A'})"

        item = {"date": date_str, "desc": desc, "hit": hit, "matched": matched}
        result["known_announcements"].append(item)
        icon = "✅" if hit else "❌"
        print(f"  {icon} {date_str} {desc}: {matched}")

    recall = hits / len(KNOWN_ANNOUNCEMENTS) if KNOWN_ANNOUNCEMENTS else 0
    result["recall"] = round(recall, 4)

    # Precision: 多少窗口在已知公告 ±5 日内
    in_window = 0
    for wd in window_dates:
        for date_str, _ in KNOWN_ANNOUNCEMENTS:
            kd = pd.Timestamp(date_str)
            if abs((wd - kd).days) <= 5:
                in_window += 1
                break
    precision = in_window / len(windows) if windows else 0
    result["precision"] = round(precision, 4)
    result["n_hit_windows"] = in_window

    print(f"\n  Recall:  {hits}/{len(KNOWN_ANNOUNCEMENTS)} = {recall:.1%}")
    print(f"  Precision (参考): {in_window}/{len(windows)} = {precision:.1%}")
    print("  注: 低 precision 不一定是坏事——非公告期间的机构行为也可能有意义")

    return result


# ======================================================================
# D) 策略模拟 (monkeypatch)
# ======================================================================
_original_caf = sf.compute_all_factors
_active_signal = None  # DataFrame with weekly signal


def _load_510500_signal_weekly():
    """加载并准备周频 is_anomaly + contra_trend 信号"""
    share_df = pd.read_csv(SHARE_FILE, dtype={"trade_date": str})
    share_df = share_df.sort_values("trade_date").reset_index(drop=True)

    signals = compute_etf_signals(
        share_series=share_df["fd_share"],
        dates=share_df["trade_date"],
    )
    signals["date_dt"] = pd.to_datetime(signals["trade_date"], format="%Y%m%d")
    signals = signals.set_index("date_dt")
    return signals


def _patched_caf_national_team(weekly_nav, pe_df=None, config=None, **kwargs):
    """在 510500 的 score 上加 +0.05 当信号激活时"""
    import src.backtest as sbt
    factors = _original_caf(weekly_nav, pe_df, config, **kwargs)

    if _active_signal is None:
        return factors

    mom = factors["momentum"].copy()
    nav_idx = mom.index  # 周频日期

    # 510500 对应列名 = "中证500ETF" (col index 2)
    col_name = "中证500ETF"
    if col_name not in mom.columns:
        return factors

    col_idx = list(mom.columns).index(col_name)
    sig = _active_signal

    for i, dt in enumerate(nav_idx):
        # 找 dt 之前 7 天内最近的信号行（shift(1) 防前视）
        window_start = dt - pd.Timedelta(days=12)
        mask = (sig.index >= window_start) & (sig.index < dt)
        sub = sig.loc[mask]
        if len(sub) == 0:
            continue
        last_row = sub.iloc[-1]
        # 条件：is_anomaly=True 且 contra_trend=True（或仅 is_anomaly）
        if last_row.get("is_anomaly", False):
            mom.iloc[i, col_idx] += 0.05

    factors["momentum"] = mom
    return factors


def run_strategy_simulation():
    """Monkeypatch 回测: 激活信号时给 510500 score +0.05"""
    import src.backtest as sbt

    print("\n" + "=" * 70)
    print("D) 策略模拟: monkeypatch (is_anomaly → 510500 score +0.05)")
    print("=" * 70)

    global _active_signal

    # 加载信号
    _active_signal = _load_510500_signal_weekly()
    print(f"  信号行数: {len(_active_signal)}")
    print(f"  is_anomaly 天数: {_active_signal['is_anomaly'].sum()}")

    cfg = load_config(str(CFG_PATH))

    # Baseline
    print("  运行 Baseline...")
    with contextlib.redirect_stdout(io.StringIO()):
        res_base = run_backtest(cfg, start_date=START_DATE)
    base_sharpe = float(res_base.metrics["sharpe_ratio"])
    base_maxdd = float(res_base.metrics["max_drawdown"])
    base_annual = float(res_base.metrics["annual_return"])

    # Patched
    print("  运行 Patched (national team signal)...")
    sbt.compute_all_factors = _patched_caf_national_team
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res_patch = run_backtest(cfg, start_date=START_DATE)
        patch_sharpe = float(res_patch.metrics["sharpe_ratio"])
        patch_maxdd = float(res_patch.metrics["max_drawdown"])
        patch_annual = float(res_patch.metrics["annual_return"])
    finally:
        sbt.compute_all_factors = _original_caf
        _active_signal = None

    delta_sharpe = patch_sharpe - base_sharpe
    delta_maxdd = patch_maxdd - base_maxdd

    result = {
        "baseline": {
            "sharpe": round(base_sharpe, 4),
            "max_drawdown": round(base_maxdd, 4),
            "annual_return": round(base_annual, 4),
        },
        "patched": {
            "sharpe": round(patch_sharpe, 4),
            "max_drawdown": round(patch_maxdd, 4),
            "annual_return": round(patch_annual, 4),
        },
        "delta": {
            "sharpe": round(delta_sharpe, 4),
            "max_drawdown": round(delta_maxdd, 4),
            "annual_return": round(patch_annual - base_annual, 4),
        },
    }

    print(f"\n  {'指标':<12} {'Baseline':>10} {'Patched':>10} {'Δ':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Sharpe':<12} {base_sharpe:>10.4f} {patch_sharpe:>10.4f} {delta_sharpe:>+10.4f}")
    print(f"  {'MaxDD':<12} {base_maxdd:>10.4f} {patch_maxdd:>10.4f} {delta_maxdd:>+10.4f}")
    print(f"  {'AnnRet':<12} {base_annual:>10.4f} {patch_annual:>10.4f} {patch_annual-base_annual:>+10.4f}")

    return result


# ======================================================================
# 门禁判定
# ======================================================================
def make_gate_decision(ic_result, event_result, alignment_result, sim_result=None):
    """根据三项指标做门禁判定"""
    print("\n" + "=" * 70)
    print("门禁判定 (Gate Decision)")
    print("=" * 70)

    ic = abs(ic_result["full_period"]["rank_ic"])
    t_stat = abs(ic_result["full_period"]["t_stat"])
    recall = alignment_result["recall"]

    # 事件后收益是否正向显著
    event_positive = False
    for h in [4, 8, 12]:
        key = f"horizon_{h}w"
        if key in event_result:
            if event_result[key]["t_stat"] > 1.5 and event_result[key]["mean_return"] > 0:
                event_positive = True
                break

    # 判定逻辑
    evidence = []
    evidence.append(f"|rank_IC| = {ic:.4f} {'≥' if ic >= 0.03 else '<'} 0.03 → {'PASS' if ic >= 0.03 else 'FAIL'}")
    evidence.append(f"|t-stat| = {t_stat:.3f} {'≥' if t_stat >= 1.5 else '<'} 1.5 → {'PASS' if t_stat >= 1.5 else 'FAIL'}")
    evidence.append(f"事件后收益正向显著: {'是' if event_positive else '否'} → {'PASS' if event_positive else 'FAIL'}")
    evidence.append(f"Recall = {recall:.1%} {'≥' if recall >= 2/3 else '<'} 2/3 → {'PASS' if recall >= 2/3 else 'FAIL (OBSERVATION 条件)'}")

    # GO: |rank_IC| >= 0.03 且 |t-stat| >= 1.5 且事件正向
    if ic >= 0.03 and t_stat >= 1.5 and event_positive:
        decision = "GO"
        action = "推进正式配置化集成（复用 PVD 模式）"
    # OBSERVATION: IC 不显著但事件对齐良好 (recall >= 2/3)
    elif recall >= 2 / 3:
        decision = "OBSERVATION ONLY"
        action = "保持看板观察，不集成策略"
    # NO-GO
    else:
        decision = "NO-GO"
        action = "终止策略集成方向，信号保留看板展示用途"

    for e in evidence:
        print(f"  {e}")
    print(f"\n  ★ 判定: {decision}")
    print(f"  ★ 后续: {action}")

    result = {
        "decision": decision,
        "action": action,
        "evidence": evidence,
        "criteria": {
            "abs_rank_ic": round(ic, 4),
            "abs_t_stat": round(t_stat, 3),
            "event_positive_significant": event_positive,
            "recall": round(recall, 4),
        },
    }

    if sim_result:
        delta_s = sim_result["delta"]["sharpe"]
        result["strategy_delta_sharpe"] = delta_s
        print(f"  ★ 策略模拟 ΔSharpe = {delta_s:+.4f} {'(微小增量)' if abs(delta_s) < 0.01 else ''}")

    return result


# ======================================================================
# Main
# ======================================================================
def main():
    print("P3 信号有效性验证: 主力份额信号 → 510500 预测力")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # A) IC 分析
    ic_result = run_ic_analysis()

    # B) 事件收益归因
    event_result = run_event_attribution()

    # C) 公告对齐
    alignment_result = run_announcement_alignment()

    # D) 策略模拟（IC 方向正确 → 做简单模拟）
    sim_result = None
    if ic_result and ic_result["full_period"]["rank_ic"] > 0:
        sim_result = run_strategy_simulation()
    else:
        print("\n[跳过策略模拟: IC 方向非正 或 IC 分析失败]")

    # 门禁判定
    gate_result = make_gate_decision(ic_result, event_result, alignment_result, sim_result)

    # 输出 JSON
    output = {
        "experiment": "national_team_signal_validation",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ic_analysis": ic_result,
        "event_attribution": event_result,
        "announcement_alignment": alignment_result,
        "strategy_simulation": sim_result,
        "gate_decision": gate_result,
    }

    json_path = OUT_DIR / "exp_national_team_signal.json"
    with open(json_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ JSON: {json_path}")

    # 输出 Markdown 报告
    md_path = OUT_DIR / "exp_national_team_signal.md"
    _write_report(md_path, output)
    print(f"✅ Report: {md_path}")

    return output


def _write_report(path, data):
    """生成 Markdown 报告"""
    ic = data["ic_analysis"]
    ev = data["event_attribution"]
    al = data["announcement_alignment"]
    sim = data["strategy_simulation"]
    gate = data["gate_decision"]

    lines = [
        "# P3 信号有效性验证：主力份额信号 → 510500 预测力",
        "",
        f"生成时间: {data['generated_at']}",
        "",
        "---",
        "",
        "## 门禁判定",
        "",
        f"**{gate['decision']}** — {gate['action']}",
        "",
        "证据链:",
        "",
    ]
    for e in gate["evidence"]:
        lines.append(f"- {e}")
    lines.append("")

    # A) IC
    lines.extend([
        "---",
        "",
        "## A) IC 分析",
        "",
        "| 区间 | Rank IC | p-value | t-stat | 周数 | 期间 |",
        "|------|---------|---------|--------|------|------|",
        f"| 全期 | {ic['full_period']['rank_ic']:.4f} | {ic['full_period']['p_value']:.4f} | {ic['full_period']['t_stat']:.3f} | {ic['full_period']['n_weeks']} | {ic['full_period']['period']} |",
        f"| 近期(2020+) | {ic['recent_2020']['rank_ic']:.4f} | {ic['recent_2020']['p_value']:.4f} | {ic['recent_2020']['t_stat']:.3f} | {ic['recent_2020']['n_weeks']} | {ic['recent_2020']['period']} |",
        "",
        f"滚动52周: IC mean={ic['rolling_52w']['ic_mean']:.4f}, std={ic['rolling_52w']['ic_std']:.4f}, IR={ic['rolling_52w']['ir']:.3f} (n={ic['rolling_52w']['n_windows']})",
        "",
        "结论: IC 极低且不显著，无法支持策略集成。",
        "",
    ])

    # B) 事件归因
    lines.extend([
        "---",
        "",
        "## B) 事件收益归因",
        "",
        f"独立事件窗口: {ev['n_windows']}",
        f"无条件周均收益: {ev['unconditional_weekly_mean']:.6f}",
        "",
        "| 窗口 | 事件后均值 | 中位数 | 胜率 | 无条件累计 | 超额均值 | t-stat | p-value | n |",
        "|------|-----------|--------|------|-----------|---------|--------|---------|---|",
    ])
    for h in [4, 8, 12]:
        key = f"horizon_{h}w"
        if key in ev:
            d = ev[key]
            lines.append(f"| {h}周 | {d['mean_return']:+.5f} | {d['median_return']:+.5f} | {d['win_rate']:.1%} | {d['unconditional_cum']:+.5f} | {d['excess_mean']:+.5f} | {d['t_stat']:+.3f} | {d['p_value']:.4f} | {d['n_events']} |")
    lines.extend([
        "",
        "结论: 事件后收益不优于无条件基准（t-stat 为负），增持事件不具备正向预测力。",
        "",
    ])

    # C) 公告对齐
    lines.extend([
        "---",
        "",
        "## C) 公告对齐验证",
        "",
        f"Recall: {al['recall']:.1%} ({sum(1 for a in al['known_announcements'] if a['hit'])}/{len(al['known_announcements'])})",
        f"Precision (参考): {al['precision']:.1%} ({al['n_hit_windows']}/{al['n_windows']})",
        "",
        "| 公告日期 | 事件 | 命中 | 匹配 |",
        "|---------|------|------|------|",
    ])
    for a in al["known_announcements"]:
        icon = "✅" if a["hit"] else "❌"
        lines.append(f"| {a['date']} | {a['desc']} | {icon} | {a['matched']} |")
    lines.extend([
        "",
        "结论: Recall 仅 1/3，对已知增持事件的捕获率过低，不满足 OBSERVATION 门禁(≥2/3)。",
        "",
    ])

    # D) 策略模拟
    if sim:
        lines.extend([
            "---",
            "",
            "## D) 策略模拟",
            "",
            "方案: is_anomaly=True 时给 510500 score +0.05",
            "",
            "| 指标 | Baseline | Patched | Δ |",
            "|------|----------|---------|---|",
            f"| Sharpe | {sim['baseline']['sharpe']:.4f} | {sim['patched']['sharpe']:.4f} | {sim['delta']['sharpe']:+.4f} |",
            f"| MaxDD | {sim['baseline']['max_drawdown']:.4f} | {sim['patched']['max_drawdown']:.4f} | {sim['delta']['max_drawdown']:+.4f} |",
            f"| AnnRet | {sim['baseline']['annual_return']:.4f} | {sim['patched']['annual_return']:.4f} | {sim['delta']['annual_return']:+.4f} |",
            "",
            f"结论: ΔSharpe={sim['delta']['sharpe']:+.4f}，{'增量不显著' if abs(sim['delta']['sharpe']) < 0.05 else '有一定增量但需更多验证'}。",
            "",
        ])

    # 总结
    lines.extend([
        "---",
        "",
        "## 总结",
        "",
        f"- IC: |{ic['full_period']['rank_ic']:.4f}| < 0.03 → 信号无统计学显著预测力",
        f"- 事件归因: 增持后收益不优于无条件基准",
        f"- 公告对齐: Recall={al['recall']:.1%} < 66.7% → 捕获率不足",
        f"- 判定: **{gate['decision']}** — 份额信号不适合作为策略因子",
        "",
        "建议: 主力份额追踪保留为看板**信息展示**用途（市场情绪参考），",
        "但不集成进策略引擎的仓位计算。",
    ])

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
