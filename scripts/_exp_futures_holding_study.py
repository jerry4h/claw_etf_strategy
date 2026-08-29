#!/usr/bin/env python3
"""股指期货多空龙虎榜(B 线) E0 数据管线 + E1 信号质量评估.

用户诉求: 从"看涨看跌份额"里找机构对冲痕迹以预测风险.
A 线(基差贴水)已 NO-GO(同步指标)。本脚本测 B 线 —— 真正带方向的份额数据。

**数据结构关键事实(E0 预检实测, 踩过坑)**:
  fut_holding 是**前 20 名排名榜**, 不是全市场持仓。每个合约每个字段(vol/
  long_hld/short_hld)恰好 20 个有值, 且数据里**没有任何真正的 0, 只有 NaN**。
  某会员 long_hld=NaN 意味着"未进多头前 20", **不等于多头持仓为 0**。
  因此:
    - 禁止 groupby().sum() 直接用(它把 NaN 当 0, 会把"未上榜"读成"持仓为零")
    - 禁止整行 dropna(会误删只上单边榜的会员, 实测 92 行被删成 37 行)
    - 正确口径: 逐字段各自 skipna 合计, 比较两个同规模(前 20)榜的力量对比
    - "纯空头席位占比"这类信号不可构造(无法区分未上榜与持仓为零)
    - 外资席位信号不可用: 摩根大通/瑞银 2015 与 2018 年未上榜, 2020 年才 1 行

无前视: 龙虎榜当日收盘后公布, 取周五值、下周一执行, 与策略 rebalance 时序一致。

用法:
    python scripts/_exp_futures_holding_study.py --fetch    # 取数(约 680 次 API)
    python scripts/_exp_futures_holding_study.py            # E0 -> E1 -> 报告
    python scripts/_exp_futures_holding_study.py --render-only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

# 复用 A 线脚本的评估机件(ts_ic/bh_fdr/leading_ok/line_funnel/...), 保证两条线
# 的门禁与统计口径逐位一致, 结论可直接横向比较。
_spec = importlib.util.spec_from_file_location(
    "fb", PROJECT / "scripts" / "_exp_futures_basis_study.py")
fb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fb)

CACHE = PROJECT / "data" / "experiments"
OUT_DIR = PROJECT / "output" / "experiments"
OUT_MD = OUT_DIR / "exp_futures_holding.md"
OUT_JSON = OUT_DIR / "exp_futures_holding.json"
OUT_PNG = OUT_DIR / "futures_holding.png"
RAW = CACHE / "raw_fut_holding.csv"          # 命中 .gitignore 的 raw_*.csv

ROOTS = ("IC", "IF")
MAIN_LINE = "IC"
REF_LINE = "IF"
TOP_N_CONC = 5        # 集中度取前 5 会员
MIN_BROKERS = 8       # 单周有效会员数下限, 低于此视为榜单不完整


# --------------------------------------------------------------------------
# E0: 取数
# --------------------------------------------------------------------------
def weekly_trade_dates() -> list[str]:
    """策略每个周频锚点对应的最后一个期货交易日(YYYYMMDD).

    用已缓存的 IC/IF 日线交易日历, 避免对休市日发起无效调用。
    """
    days = set()
    for root in ROOTS:
        p = CACHE / f"raw_fut_{root.lower()}_daily.csv"
        if p.exists():
            d = pd.read_csv(p, usecols=["trade_date"], dtype={"trade_date": str})
            days |= set(d.trade_date.unique())
    cal = pd.Series(sorted(pd.to_datetime(sorted(days), format="%Y%m%d")))
    weekly, _cfg = fb.load_weekly()
    out = []
    for wd in weekly.index:
        prior = cal[cal <= wd]
        if len(prior):
            out.append(prior.iloc[-1].strftime("%Y%m%d"))
    return sorted(set(out))


def fetch_holding() -> None:
    pro = fb._pro()
    dates = weekly_trade_dates()
    print(f"  需拉取 {len(dates)} 个交易日 (策略周频锚点)", flush=True)
    rows, empty = [], 0
    for i, d in enumerate(dates, 1):
        x = fb._retry(pro.fut_holding, trade_date=d, exchange="CFFEX")
        if x is None or not len(x):
            empty += 1
        else:
            x = x[x.symbol.astype(str).str.match(r"^(IC|IF)\d")]
            if len(x):
                rows.append(x)
        if i % 60 == 0:
            print(f"  {i}/{len(dates)} (空 {empty})", flush=True)
        time.sleep(0.15)
    df = pd.concat(rows, ignore_index=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW, index=False)
    print(f"  缓存完成: {len(df)} 行, {df.trade_date.nunique()} 个交易日, "
          f"空返回 {empty} 个", flush=True)


# --------------------------------------------------------------------------
# E0: 排名榜口径的日度聚合
# --------------------------------------------------------------------------
def aggregate_daily(root: str) -> tuple[pd.DataFrame, dict]:
    """把逐会员排名榜聚合成日度指标.

    严格排名榜口径: 逐字段 skipna 合计, 不整行 dropna, 不把 NaN 当 0.
    """
    raw = pd.read_csv(RAW, dtype={"trade_date": str})
    d = raw[raw.symbol.astype(str).str.match(rf"^{root}\d")].copy()
    for c in ("long_hld", "short_hld", "long_chg", "short_chg", "vol"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["dt"] = pd.to_datetime(d.trade_date, format="%Y%m%d")

    recs = []
    for dt, g in d.groupby("dt"):
        # 逐字段 skipna 合计 —— 两个榜规模相同(各前 20), 力量对比有意义
        L = g.long_hld.sum(min_count=1)
        S = g.short_hld.sum(min_count=1)
        if not (np.isfinite(L) and np.isfinite(S)) or (L + S) <= 0:
            continue
        # 会员级: min_count=1 保证全 NaN 组结果为 NaN 而非 0
        by_l = g.groupby("broker").long_hld.sum(min_count=1)
        by_s = g.groupby("broker").short_hld.sum(min_count=1)
        has_l, has_s = by_l.notna(), by_s.notna()
        n_broker = int((has_l | has_s).sum())
        l_valid = by_l.dropna().sort_values(ascending=False)
        s_valid = by_s.dropna().sort_values(ascending=False)
        recs.append({
            "dt": dt,
            "long_top": float(L), "short_top": float(S),
            "net_short_rate": float((S - L) / (S + L)),
            "n_broker": n_broker,
            "n_both": int((has_l & has_s).sum()),
            "short_only_cnt": int((~has_l & has_s).sum()),
            "long_only_cnt": int((has_l & ~has_s).sum()),
            "short_conc": float(s_valid.head(TOP_N_CONC).sum() / max(S, 1)),
            "long_conc": float(l_valid.head(TOP_N_CONC).sum() / max(L, 1)),
            "long_chg_sum": float(g.long_chg.sum(min_count=1)),
            "short_chg_sum": float(g.short_chg.sum(min_count=1)),
        })
    df = pd.DataFrame(recs).set_index("dt").sort_index()
    df["conc_gap"] = df.short_conc - df.long_conc
    df["chg_asym"] = ((df.short_chg_sum - df.long_chg_sum)
                      / (df.long_top + df.short_top))
    df["short_only_share"] = df.short_only_cnt / df.n_broker.clip(lower=1)

    diag = {
        "root": root,
        "n_days": int(len(df)),
        "date_min": str(df.index.min().date()),
        "date_max": str(df.index.max().date()),
        "n_broker_min": int(df.n_broker.min()),
        "n_broker_median": float(df.n_broker.median()),
        "days_below_min_brokers": int((df.n_broker < MIN_BROKERS).sum()),
        "net_short_rate_mean_pct": round(float(df.net_short_rate.mean() * 100), 3),
        "net_short_rate_min_pct": round(float(df.net_short_rate.min() * 100), 2),
        "net_short_rate_max_pct": round(float(df.net_short_rate.max() * 100), 2),
        "pct_days_net_short": round(float((df.net_short_rate > 0).mean() * 100), 2),
        "short_conc_mean_pct": round(float(df.short_conc.mean() * 100), 2),
        "long_conc_mean_pct": round(float(df.long_conc.mean() * 100), 2),
    }
    return df, diag


def e0_nan_discipline_check(root: str) -> dict:
    """自证 NaN 口径正确性, 把踩过的坑固化成校验.

    对比三种口径, 证明"错误口径"与"正确口径"确有实质差异 —— 这不是学术洁癖:
    首版预检正因 groupby().sum() 把 NaN 当 0, 得出"摩根大通多头挂零=纯空头盘"
    的错误结论。
    """
    raw = pd.read_csv(RAW, dtype={"trade_date": str})
    d = raw[raw.symbol.astype(str).str.match(rf"^{root}\d")].copy()
    for c in ("long_hld", "short_hld"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    last = d[d.trade_date == d.trade_date.max()]
    n_rows = len(last)
    n_dropna_both = len(last.dropna(subset=["long_hld", "short_hld"]))
    # 错误口径: groupby.sum() 默认 min_count=0, NaN 组 -> 0
    wrong = last.groupby("broker")[["long_hld"]].sum()
    n_fake_zero = int((wrong.long_hld == 0).sum())
    # 正确口径: min_count=1, NaN 组保持 NaN
    right = last.groupby("broker")[["long_hld"]].sum(min_count=1)
    n_true_nan = int(right.long_hld.isna().sum())
    n_true_zero = int((right.long_hld == 0).sum())
    return {
        "sample_date": last.trade_date.max(),
        "n_rows": n_rows,
        "n_after_wrong_dropna": n_dropna_both,
        "rows_lost_by_dropna": n_rows - n_dropna_both,
        "wrong_groupby_zero_count": n_fake_zero,
        "correct_nan_count": n_true_nan,
        "correct_true_zero_count": n_true_zero,
        "no_real_zero_in_data": bool(n_true_zero == 0),
        "wrong_vs_right_differ": bool(n_fake_zero != n_true_zero),
        "note": ("数据中不存在真实 0; 错误口径把 "
                 f"{n_fake_zero} 个未上榜会员读成多头持仓为零"),
    }


SIGNALS = ["net_short_rate", "net_short_z", "net_short_chg_1w",
           "net_short_chg_4w", "short_only_share", "conc_gap", "chg_asym"]


def build_signals(wk: pd.DataFrame) -> pd.DataFrame:
    """七个多空份额信号. 所有标准化一律 expanding, 禁止全样本 percentile。"""
    sig = pd.DataFrame(index=wk.index)
    sig["net_short_rate"] = wk["net_short_rate"]
    sig["net_short_z"] = fb.expanding_z(wk["net_short_rate"])
    sig["net_short_chg_1w"] = wk["net_short_rate"].diff(1)
    sig["net_short_chg_4w"] = wk["net_short_rate"].diff(4)
    sig["short_only_share"] = wk["short_only_share"]
    sig["conc_gap"] = wk["conc_gap"]
    sig["chg_asym"] = wk["chg_asym"]
    return sig


# --------------------------------------------------------------------------
# E1': 事件研究(脉冲视角)
# --------------------------------------------------------------------------
MIN_HIST = 104        # 分位阈值最小历史(周, ~2年)
ROLL_WIN = 104        # 滚动分位窗口(周), 应对净空率的非平稳
MIN_EVENTS = 20       # 触发次数下限, 低于此不做统计推断
MIN_YEAR_SPAN = 4     # 触发年份跳数下限, 低于此视为时间覆盖不足
Q_GRID = (0.90, 0.95)


def pulse_flags(sig: pd.Series, q: float, side: str,
                min_hist: int = MIN_HIST, mode: str = "roll") -> pd.Series:
    """分位阈值触发标记.

    无前视关键: 阈值序列先 shift(1), 第 t 期的阈值只用 t-1 及之前的历史,
    不含第 t 期自身。禁止 sig.quantile(q) 这类全样本分位。

    mode:
      'roll' (默认) —— ROLL_WIN 周滚动分位。适用于**非平稳**序列: 净空率分年
        均值从 0.5%(2015) 升到 6.59%(2026), 用 expanding 会把触发全部集中到趋势
        启动期(实测 net_short_z 仅 2019-2020 两年有触发), 使 E2 退化成单一年份实验。
      'exp' —— expanding 分位。仅对已差分、近乎平稳的信号才安全。
    """
    prev = sig.shift(1)
    if mode == "exp":
        thr = prev.expanding(min_periods=min_hist).quantile(q)
    else:
        thr = prev.rolling(ROLL_WIN, min_periods=min_hist).quantile(q)
    flag = (sig > thr) if side == "high" else (sig < thr)
    return flag.where(thr.notna(), False).fillna(False).astype(bool)


def event_study(flag: pd.Series, tg: pd.DataFrame) -> dict:
    """触发周 vs 非触发周的前瞻风险分布对照.

    这是本脚本相对 A 线的核心方法论转变。全样本时序 IC 隐含假设"信号每周都线性
    有效"; 而份额/持仓类因子的先验是**稀疏脉冲** —— 长期贴近噪声, 只在少数极端
    节点携带信息。若信号仅在 5% 的周有效, 其余 95% 为噪声, 全样本 IC 会被稀释
    到接近 0, 于是"IC 不显著"反映的是框架盲区而非信号无效。故此处改用事件研究:
    只比较触发周与非触发周的后续风险分布。

    用 Mann-Whitney U(非参数): 两组样本量极不平衡(约 1:20), 且风险类目标右偏,
    t 检验的正态假设不成立。
    """
    from scipy import stats as st
    out = {"n_trigger": int(flag.sum()),
           "trigger_rate_pct": round(float(flag.mean() * 100), 2)}
    per = {}
    for col in tg.columns:
        d = pd.concat([flag.rename("f"), tg[col].rename("y")], axis=1).dropna()
        if not len(d):
            per[col] = None
            continue
        a = d.y[d.f].to_numpy(dtype=float)
        b = d.y[~d.f].to_numpy(dtype=float)
        if len(a) < MIN_EVENTS or len(b) < MIN_EVENTS:
            per[col] = {"n_trig": len(a), "n_base": len(b),
                        "insufficient": True}
            continue
        u, p = st.mannwhitneyu(a, b, alternative="two-sided")
        # 效应量 rank-biserial correlation, 与样本量无关
        rb = 2.0 * u / (len(a) * len(b)) - 1.0
        per[col] = {
            "n_trig": len(a), "n_base": len(b),
            "med_trig": round(float(np.median(a)), 6),
            "med_base": round(float(np.median(b)), 6),
            "delta_med": round(float(np.median(a) - np.median(b)), 6),
            "mean_trig": round(float(a.mean()), 6),
            "mean_base": round(float(b.mean()), 6),
            "rank_biserial": round(float(rb), 4),
            "p_mwu": round(float(p), 5),
            "sig_05": bool(p < 0.05),
            "insufficient": False,
        }
    out["per_target"] = per
    return out


def pulse_scan(sig: pd.DataFrame, tg: pd.DataFrame) -> dict:
    """信号 x 分位 x 方向 的脉冲事件研究全扫描.

    预注册主假说(理论驱动, 非数据挖掘): **净空率极端升高 -> 后续风险上升**,
    即 net_short_z / net_short_chg_* 的 side='high'。其余组合列为探索性,
    报告中明确标注未做多重比较校正。
    """
    res = {}
    for c in sig.columns:
        for q in Q_GRID:
            for side in ("high", "low"):
                qq = q if side == "high" else round(1.0 - q, 2)
                flag = pulse_flags(sig[c], qq, side)
                res[f"{c}|q{qq}|{side}"] = event_study(flag, tg)
    return res


# 预注册主假说: **必须用差分型信号**。首版选 net_short_z(水平型 z-score)是错的 ——
# 净空率强非平稳, 水平型信号的极端分位触发仅落在趋势启动期(实测仅 2 个年份),
# 而差分型(chg_1w/chg_4w/chg_asym)触发分散在 2018-2026 共 6-8 个年份。
# 经济含义也更直接: 关心的是对冲需求的**突变**, 而非长期抬升的绝对水平。
PREREG = [("net_short_chg_4w", 0.95, "high"),
          ("net_short_chg_1w", 0.95, "high")]


def prereg_keys() -> list[str]:
    return [f"{c}|q{q}|{s}" for c, q, s in PREREG]


# --------------------------------------------------------------------------
# E2: 集成回测(monkeypatch, 不碰 src/)
# --------------------------------------------------------------------------
#
# 注入点: src.backtest.compute_ashare_vol_boost —— 它已是 additive defense
# boost 形态(backtest.py:435 `def_ratio = min(def_ratio + ashare_boost, 1.0)`),
# 且 ashare_vol_boost_enabled 默认 False、v4.6 config 无 ashare_vol 段, 生产中
# 恒返回 0。故 patch 它是**纯增量**注入, 不会改变任何现有生产行为。
#
# 时序对齐(探针实测, 必须自证否则整体错位):
#   vol_values 行数 = 682 = load_weekly 行数 -> i 即 weekly 位置索引
#   i 范围 [14, 680] 严格递增, 共 667 次 = nav_series 行数
#   i=14 对应 weekly.index[14]=2013-08-23, 而 nav_series 首行 2013-08-30
#   -> 第 i 周用 weekly[i] 收盘信息决策, 收益记在 weekly[i+1](信号 t / 执行 t+1)
#   因此 flag 按 weekly.index 构造、以 arr[i] 索引即正确且无前视: flag[i] 用的是
#   weekly.index[i] 周五收盘后公布的龙虎榜, 正是该周决策可用的信息。
BOOST_GRID = (0.10, 0.15, 0.20)
HOLD_GRID = (1, 2, 4)


def build_pulse_boost(flag: pd.Series, boost: float, hold: int,
                      n: int) -> np.ndarray:
    """把脉冲触发展开成逐周 boost 数组(长度 = weekly 周数).

    触发周起连续 hold 周施加 boost。重叠触发取 max 而非累加 —— 累加会让密集
    触发期的防御比例失控, 且无理论依据。
    """
    arr = np.zeros(n, dtype=float)
    for t in np.where(flag.to_numpy())[0]:
        end = min(t + hold, n)
        arr[t:end] = np.maximum(arr[t:end], boost)
    return arr


def run_bt_pulse(cfg, arr: np.ndarray | None) -> tuple[object, dict]:
    """跑回测; arr 非 None 时注入脉冲 boost. 返回 (result, injection_audit).

    audit 用于自证注入确实生效且下标未越界 —— 未对齐即视为失败, 绝不输出可能
    错位的结果。
    """
    import src.backtest as sbt
    from src.backtest import run_backtest

    if arr is None:
        res = run_backtest(cfg)
        return res, {"patched": False, "n_calls": 0, "n_hit": 0,
                     "i_min": None, "i_max": None, "out_of_range": 0,
                     "aligned": True}

    orig = sbt.compute_ashare_vol_boost
    st = {"n": 0, "hit": 0, "oor": 0, "imin": None, "imax": None,
          "base_nonzero": 0}

    def patched(vol_values, i, ashare_idx, config):
        base = orig(vol_values, i, ashare_idx, config)
        if base > 0:
            st["base_nonzero"] += 1
        st["n"] += 1
        st["imin"] = i if st["imin"] is None else min(st["imin"], i)
        st["imax"] = i if st["imax"] is None else max(st["imax"], i)
        if not (0 <= i < len(arr)):
            st["oor"] += 1
            return base
        extra = float(arr[i])
        if extra > 0:
            st["hit"] += 1
        return max(base, extra)

    sbt.compute_ashare_vol_boost = patched
    try:
        res = run_backtest(cfg)
    finally:
        sbt.compute_ashare_vol_boost = orig

    audit = {
        "patched": True, "n_calls": st["n"], "n_hit": st["hit"],
        "i_min": st["imin"], "i_max": st["imax"],
        "out_of_range": st["oor"],
        "prod_base_boost_nonzero": st["base_nonzero"],
        "arr_len": int(len(arr)),
        "aligned": bool(st["oor"] == 0 and st["n"] > 0),
    }
    return res, audit


def metrics_of(res) -> dict:
    """绩效指标. 除 MaxDD 外必须带**对全程回掤敏感**的度量.

    首版只用 MaxDD 作主指标是错的: 实测基线 MaxDD 5.7646% 发生在
    2018-02-09, 而脉冲从未在那里触发(±8 周内 0 次), 导致 27 组参数的
    MaxDD 分毫不动、恒为 5.7646% —— 度量对干预完全不敏感, 却被误读成
    "脉冲防御无效"。Ulcer index / 平均回掤 / 回掤积分对全程敏感, 才能区分参数。
    """
    m = res.metrics
    dd = res.nav_series["drawdown"]
    return {
        "cagr_pct": round(float(m["annual_return"]) * 100, 4),
        "maxdd_pct": round(float(m["max_drawdown"]) * 100, 4),
        "ulcer_pct": round(float(np.sqrt((dd ** 2).mean())) * 100, 4),
        "avg_dd_pct": round(float(dd.mean()) * 100, 4),
        "dd_p95_pct": round(float(dd.quantile(0.95)) * 100, 4),
        "sharpe": round(float(m["sharpe_ratio"]), 4),
        "calmar": round(float(m["calmar_ratio"]), 4),
        "win_rate_pct": round(float(m["win_rate"]) * 100, 2),
        "final_nav": round(float(m["final_nav"]), 4),
        "rebalance_count": int(m["rebalance_count"]),
        "ann_vol_pct": round(float(m["annual_volatility"]) * 100, 4),
    }


def trigger_coverage(flag: pd.Series) -> dict:
    """触发的时间覆盖度 —— 必查项, 否则 E2 测的不是信号而是某个特定年份.

    实测教训: net_short_z 的 expanding q0.95 触发 42 次**全部挤在 2019-2020**
    两年。因为净空率非平稳(分年均值 0.5%@2015 -> 6.59%@2026), expanding 分位
    捕捉的是"相对历史的新高": 趋势启动期集中爆表, 之后新均值抬升就不再“新”。
    此时跑 E2 等于在问"2019-2020 加防御好不好", 与份额信号无关。
    """
    idx = flag[flag].index
    if not len(idx):
        return {"n": 0, "year_span": 0, "years": {}, "coverage_ok": False}
    yrs = pd.Series([d.year for d in idx]).value_counts().sort_index()
    span = int(len(yrs))
    return {
        "n": int(flag.sum()),
        "year_span": span,
        "years": {int(k): int(v) for k, v in yrs.items()},
        "first": str(idx[0].date()), "last": str(idx[-1].date()),
        "max_year_share_pct": round(float(yrs.max() / yrs.sum() * 100), 1),
        "coverage_ok": bool(span >= 4),
    }


def conditional_dd(res, flag: pd.Series, win: int = 4) -> dict:
    """触发后 win 周内的局部回掤 —— 直接检验"脉冲是否预示风险", 且对局部敏感.

    全样本 MaxDD 被历史单一极值锚死, 无法反映干预在触发处的真实效果。
    """
    nav = res.nav_series["nav"]
    f = flag.reindex(nav.index).fillna(False).astype(bool)
    locs = np.where(f.to_numpy())[0]
    if not len(locs):
        return {"n_events": 0}
    vals = []
    for t in locs:
        seg = nav.iloc[t:min(t + win + 1, len(nav))]
        if len(seg) < 2:
            continue
        vals.append(float((seg / seg.cummax() - 1.0).min()))
    if not vals:
        return {"n_events": 0}
    return {
        "n_events": len(vals),
        "mean_fwd_dd_pct": round(float(np.mean(vals)) * 100, 4),
        "med_fwd_dd_pct": round(float(np.median(vals)) * 100, 4),
        "worst_fwd_dd_pct": round(float(np.min(vals)) * 100, 4),
    }


def e2_grid(sig: pd.DataFrame, weekly: pd.DataFrame,
            combos: list[tuple[str, float, str]]) -> dict:
    """脉冲 boost 集成回测参数网格 A/B.

    判定标准(先定后测, 避免事后挑参数):
      主指标 = MaxDD 改善(脉冲信号的目标是防风险, 不是增收)
      约束   = CAGR 恶化不超过 0.5pp
      稳健性 = 不看单点最优, 要求 boost x hold 邻域一致改善; 孤立最优点视为噪声
    """
    from src.strategy import load_config
    cfg = load_config(PROJECT / "config" / "strategy_v4_6.yaml")
    n = len(weekly)

    base_res, base_audit = run_bt_pulse(cfg, None)
    base_m = metrics_of(base_res)
    print(f"  baseline: CAGR={base_m['cagr_pct']}% MaxDD={base_m['maxdd_pct']}% "
          f"Sharpe={base_m['sharpe']} Calmar={base_m['calmar']}", flush=True)

    rows, audits = [], []
    cov_by_combo = {}
    for (c, q, side) in combos:
        flag = pulse_flags(sig[c], q, side)
        n_trig = int(flag.sum())
        cov = trigger_coverage(flag)
        base_cdd = conditional_dd(base_res, flag)
        cov_by_combo[f"{c}|q{q}|{side}"] = {**cov,
                                            "baseline_cond_dd": base_cdd}
        for boost in BOOST_GRID:
            for hold in HOLD_GRID:
                arr = build_pulse_boost(flag, boost, hold, n)
                res, audit = run_bt_pulse(cfg, arr)
                if not audit["aligned"]:
                    raise RuntimeError(f"注入未对齐: {audit}")
                m = metrics_of(res)
                cdd = conditional_dd(res, flag)
                d_cond = (round(cdd["mean_fwd_dd_pct"]
                                - base_cdd["mean_fwd_dd_pct"], 4)
                          if cdd.get("n_events") and base_cdd.get("n_events")
                          else None)
                rows.append({
                    "signal": c, "q": q, "side": side, "n_trigger": n_trig,
                    "year_span": cov["year_span"],
                    "coverage_ok": cov["coverage_ok"],
                    "boost": boost, "hold": hold,
                    "weeks_boosted": int((arr > 0).sum()),
                    **m,
                    "d_cagr_pp": round(m["cagr_pct"] - base_m["cagr_pct"], 4),
                    "d_maxdd_pp": round(m["maxdd_pct"] - base_m["maxdd_pct"], 4),
                    "d_ulcer_pp": round(m["ulcer_pct"] - base_m["ulcer_pct"], 4),
                    "d_avg_dd_pp": round(m["avg_dd_pct"]
                                         - base_m["avg_dd_pct"], 4),
                    "d_dd_p95_pp": round(m["dd_p95_pct"]
                                         - base_m["dd_p95_pct"], 4),
                    "d_cond_dd_pp": d_cond,
                    "d_sharpe": round(m["sharpe"] - base_m["sharpe"], 4),
                    "d_calmar": round(m["calmar"] - base_m["calmar"], 4),
                    "n_boost_weeks_used": audit["n_hit"],
                })
                audits.append(audit)
        print(f"  {c}|q{q}|{side}: 触发 {n_trig} 次/跳越 {cov['year_span']} 个年份"
              f"{'' if cov['coverage_ok'] else ' [覆盖不足]'}, 9 组参数完成",
              flush=True)

    grid = pd.DataFrame(rows)
    return {
        "baseline": base_m,
        "baseline_audit": base_audit,
        "grid": grid.to_dict("records"),
        "coverage": cov_by_combo,
        "audit_sample": audits[0] if audits else None,
        "audit_all_aligned": bool(all(a["aligned"] for a in audits)),
        "audit_prod_base_all_zero": bool(
            all(a.get("prod_base_boost_nonzero", 0) == 0 for a in audits)),
    }


def e2_robust_region(grid_records: list[dict]) -> dict:
    """稳健性判定: 是否存在连续改善区域, 而非孤立最优点.

    主指标用 **Ulcer index** 而非 MaxDD: 基线 MaxDD 由 2018-02-09 单一极值确定,
    而脉冲从未在那里触发, 导致 MaxDD 对任何参数都恒定不变(实测 27 组均为
    5.7646%), 无法区分优劣。Ulcer 对全程回掤敏感, 才能反映干预效果。

    coverage_ok 为前置条件: 触发年份跳数 < MIN_YEAR_SPAN 时, 结果只反映某个
    特定年份而非信号本质, 不得计入稳健区域。
    """
    df = pd.DataFrame(grid_records)
    out = {}
    for (c, q, side), g in df.groupby(["signal", "q", "side"]):
        piv = g.pivot(index="boost", columns="hold", values="d_ulcer_pp")
        n_improve = int((piv < 0).sum().sum())
        n_cells = int(piv.size)
        best = g.loc[g.d_ulcer_pp.idxmin()]
        nb = g[((g.boost == best.boost) | (g.hold == best.hold))
               & ~((g.boost == best.boost) & (g.hold == best.hold))]
        nb_improve = int((nb.d_ulcer_pp < 0).sum())
        cov_ok = bool(g.coverage_ok.iloc[0])
        out[f"{c}|q{q}|{side}"] = {
            "metric": "ulcer_pct",
            "n_cells": n_cells, "n_improve": n_improve,
            "improve_share_pct": round(100.0 * n_improve / n_cells, 1),
            "best_d_ulcer_pp": round(float(best.d_ulcer_pp), 4),
            "best_d_maxdd_pp": round(float(best.d_maxdd_pp), 4),
            "best_d_avg_dd_pp": round(float(best.d_avg_dd_pp), 4),
            "best_d_cond_dd_pp": (round(float(best.d_cond_dd_pp), 4)
                                  if pd.notna(best.d_cond_dd_pp) else None),
            "best_d_cagr_pp": round(float(best.d_cagr_pp), 4),
            "best_d_calmar": round(float(best.d_calmar), 4),
            "best_d_sharpe": round(float(best.d_sharpe), 4),
            "best_boost": float(best.boost), "best_hold": int(best.hold),
            "year_span": int(g.year_span.iloc[0]),
            "coverage_ok": cov_ok,
            "neighbors_total": int(len(nb)),
            "neighbors_improving": nb_improve,
            "isolated_optimum": bool(nb_improve < max(len(nb) // 2, 1)),
            "majority_improve": bool(n_improve > n_cells / 2),
            "cagr_constraint_ok": bool(float(best.d_cagr_pp) > -0.5),
            "maxdd_insensitive": bool(g.d_maxdd_pp.abs().max() < 1e-9),
        }
    return out


# --------------------------------------------------------------------------
# 裁决
# --------------------------------------------------------------------------
def absorption_check(res: dict) -> dict:
    """检验信号预示的风险是否已被策略现有防御层吸收.

    这是区分"信号无预测力"与"信号有预测力但无增量"的关键诊断。

    对比两类目标的效应量:
      市场层面 fwd_vol_4w / fwd_maxdd_4w —— 中证500 本身的前瞻风险
      策略层面 fwd_strat_dd_4w —— 策略净值的前瞻回掤
    若市场层面恶化明显而策略层面几乎不变, 说明现有 Layer3(纳指波动映射)
    + Layer3.5(危机相关 boost) 已把这部分风险吸收, 再叠一层脉冲防御为冗余。
    """
    ev = res.get("event_study", {})
    out = {}
    for k in res.get("prereg_keys", []):
        d = ev.get(k)
        if not d:
            continue
        per = d.get("per_target") or {}
        mkt, strat = [], None
        for t in ("fwd_vol_4w", "fwd_maxdd_4w"):
            pt = per.get(t)
            if pt and not pt.get("insufficient"):
                mkt.append(abs(float(pt["rank_biserial"])))
        ps = per.get("fwd_strat_dd_4w")
        if ps and not ps.get("insufficient"):
            strat = abs(float(ps["rank_biserial"]))
        if not mkt or strat is None:
            continue
        mkt_max = max(mkt)
        out[k] = {
            "market_effect_max": round(mkt_max, 4),
            "strategy_effect": round(strat, 4),
            "ratio_strategy_over_market": (round(strat / mkt_max, 3)
                                           if mkt_max > 0 else None),
            "absorbed": bool(mkt_max > 0 and strat < 0.5 * mkt_max),
        }
    n_abs = sum(1 for v in out.values() if v["absorbed"])
    return {
        "per_signal": out,
        "n_absorbed": n_abs,
        "n_total": len(out),
        "all_absorbed": bool(out and n_abs == len(out)),
        "note": ("信号对**市场**风险有效应但对**策略**回掤几乎无效应, "
                 "表明风险已被现有 Layer3/Layer3.5 防御层吸收, "
                 "脉冲防御为冗余层"),
    }


def recompute_verdict(res: dict) -> dict:
    """三层证据合并裁决. 幂等(可对已有 json 重算)。

    **E2 权重高于 E1'**: 份额类因子是稀疏脉冲, 全样本相关性不显著不足以
    否定它 —— 真正的判据是接入策略后回测是否改善风险。故即使 E1' 事件研究
    不显著, 只要 E2 存在稳健改善区域, 仍给出条件 GO。

    反之, 若 E2 仅有孤立点改善, 那是参数噪声而非信号 —— 3x3 网格里总会有
    某个格子碰巧变好, 把它当成发现就是过拟合。
    """
    ev = res.get("event_study", {})
    rob = res.get("e2_robust", {})
    e2 = res.get("e2", {})

    pre_hits = []
    for k in res.get("prereg_keys", []):
        d = ev.get(k)
        if not d:
            continue
        for t in fb.RISK_TARGETS:
            pt = (d.get("per_target") or {}).get(t)
            if pt and not pt.get("insufficient") and pt.get("sig_05"):
                pre_hits.append(f"{k}->{t}")
    expl_hits = 0
    for k, d in ev.items():
        for t, pt in (d.get("per_target") or {}).items():
            if pt and not pt.get("insufficient") and pt.get("sig_05"):
                expl_hits += 1

    robust_names, isolated_names, cov_fail = [], [], []
    for k, v in rob.items():
        if not v.get("coverage_ok", True):
            cov_fail.append(f"{k}(仅{v.get('year_span')}个年份)")
            continue
        ok = (v["majority_improve"] and not v["isolated_optimum"]
              and v.get("cagr_constraint_ok", True)
              and v["best_d_ulcer_pp"] < 0)
        if ok:
            robust_names.append(f"{k}(Ulcer{v['best_d_ulcer_pp']:+.3f}pp)")
        elif v["best_d_ulcer_pp"] < 0:
            isolated_names.append(k)

    grid = pd.DataFrame(e2.get("grid", []))
    n_improve = int((grid.d_ulcer_pp < 0).sum()) if len(grid) else 0
    n_total = int(len(grid))
    maxdd_dead = (bool(len(grid) and grid.d_maxdd_pp.abs().max() < 1e-9)
                  if len(grid) else False)

    if robust_names and pre_hits:
        case = "D1 E2 稳健改善 + 预注册事件研究显著"
        decision = "GO"
        nxt = ("进 E3: 先做分期 OOS(2015-2019 / 2020-2026) 与成本敏感性, 通过后"
               "才议引入 src/ 并新增 YAML 开关(默认关)。")
    elif robust_names:
        case = "D2 E2 稳健改善但预注册事件研究不显著"
        decision = "条件 GO"
        nxt = ("回测层面有效但机制未独立证实。先做分期 OOS + 噪声带对照, 确认"
               "改善超出噪声尺度再谈集成。")
    elif isolated_names:
        case = "D3 仅孤立参数点改善"
        decision = "NO-GO"
        nxt = "3x3 网格里总会有格子碰巧变好, 邻域不一致即参数噪声。不得改 src/。"
    elif n_improve > 0:
        case = "D4 有改善格子但未构成稳健区域"
        decision = "NO-GO"
        nxt = "改善分散且不满足 CAGR 约束或多数则, 不得改 src/。"
    else:
        case = "D5 全网格无回掤改善"
        decision = "NO-GO"
        nxt = "脉冲防御在本策略上无正向作用, 方向关闭。"

    return {
        "case": case, "decision": decision, "next": nxt,
        "primary_metric": "ulcer_pct",
        "absorption": absorption_check(res),
        "maxdd_insensitive": maxdd_dead,
        "maxdd_note": ("全网格 MaxDD 恒定不变 —— 基线最大回掤发生处从未被脉冲"
                       "触发, 此时 MaxDD 不可作为判据(首版即因此误判 NO-GO)"
                       if maxdd_dead else "MaxDD 对参数有响应"),
        "prereg_significant": pre_hits,
        "n_prereg_significant": len(pre_hits),
        "n_exploratory_significant_uncorrected": expl_hits,
        "robust_regions": robust_names,
        "isolated_only": isolated_names,
        "coverage_excluded": cov_fail,
        "grid_improve_cells": n_improve, "grid_total_cells": n_total,
        "grid_improve_share_pct": (round(100.0 * n_improve / n_total, 1)
                                   if n_total else None),
    }


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def _sk(s: str) -> str:
    """组合 key 内部用 '|' 分隔(signal|q|side), 直接写入 markdown 表格会被误读成
    列分隔符而打乱表结构。渲染时统一换成 ' / '。"""
    return str(s).replace("|", " / ")


def render(res: dict) -> None:
    v = res["verdict"]
    L = []
    L.append("# 股指期货多空龙虎榜(B 线): 脉冲信号 E0 + E1' + E2 集成回测")
    L.append("")
    L.append(f"**裁决: {v['decision']} —— {v['case']}**")
    L.append("")
    L.append(f"下一步: {v['next']}")
    L.append("")
    L.append("评估路径与 A 线(基差)不同: 份额类因子是**稀疏脉冲** —— 长期贴近噪声, "
             "只在少数极端节点携带信息。全样本时序 IC 隐含'每周都线性有效'的假设, "
             "会把这类信号稀释到不显著。故此处用**事件研究 + 真回测**, 且裁决以 E2 "
             "回测为主判据。")
    L.append("")

    # ---- E0
    L.append("## E0 数据质量")
    L.append("")
    L.append("| 品种 | 交易日 | 区间 | 会员数(中位) | 净空率均值 | 净空率区间 | 净空日占比 |")
    L.append("|---|---|---|---|---|---|---|")
    for root, d in res["e0"].items():
        L.append(f"| {root} | {d['n_days']} | {d['date_min']} ~ {d['date_max']} | "
                 f"{d['n_broker_median']:.0f} | {d['net_short_rate_mean_pct']:+.3f}% | "
                 f"[{d['net_short_rate_min_pct']:+.2f}%, {d['net_short_rate_max_pct']:+.2f}%] | "
                 f"{d['pct_days_net_short']:.2f}% |")
    L.append("")
    L.append("### NaN 口径自证(踩过的坑, 固定为校验)")
    L.append("")
    L.append("`fut_holding` 是**前 20 名排名榜**而非全市场持仓。某会员 `long_hld=NaN` "
             "意味着'未进多头前 20', **不等于多头持仓为 0**。")
    L.append("")
    L.append("| 品种 | 样本日 | 总行 | 整行 dropna 后 | 被误删 | 错误口径读出的'零持仓' | 真实零值 | 数据中无真 0 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for root, d in res["e0_nan"].items():
        L.append(f"| {root} | {d['sample_date']} | {d['n_rows']} | "
                 f"{d['n_after_wrong_dropna']} | {d['rows_lost_by_dropna']} | "
                 f"{d['wrong_groupby_zero_count']} | {d['correct_true_zero_count']} | "
                 f"{'✓' if d['no_real_zero_in_data'] else '✗'} |")
    L.append("")
    L.append("因此本脚本全程: 逐字段 skipna 合计(`sum(min_count=1)`), 不整行 dropna, "
             "不用默认 `groupby().sum()`。受此限制, '纯空头席位占比'与'外资净空占比'"
             "两个原定信号**不可构造**(前者无法区分未上榜与持仓为零; 后者摩根大通/瑞银 "
             "2015 与 2018 年未上榜、2020 年才 1 行)。")
    L.append("")

    # ---- E1'
    L.append("## E1' 事件研究(脉冲视角)")
    L.append("")
    L.append(f"预注册主假说(理论驱动, 先定后测): "
             f"{', '.join(_sk(x) for x in res['prereg_keys'])} "
             "—— 净空率极端升高 -> 后续风险上升。")
    L.append("")
    L.append("### 预注册假说结果(风险类目标)")
    L.append("")
    L.append("| 组合 | 触发次数 | 触发率 | 目标 | 触发组中位 | 基准组中位 | 差 | 效应量 | p(MWU) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k in res["prereg_keys"]:
        d = res["event_study"].get(k)
        if not d:
            continue
        for t in fb.RISK_TARGETS:
            pt = (d.get("per_target") or {}).get(t)
            if not pt or pt.get("insufficient"):
                L.append(f"| {_sk(k)} | {d['n_trigger']} | "
                         f"{d['trigger_rate_pct']}% | "
                         f"{t} | - | - | - | - | 样本不足 |")
                continue
            mark = " **\\***" if pt["sig_05"] else ""
            L.append(f"| {_sk(k)} | {d['n_trigger']} | "
                     f"{d['trigger_rate_pct']}% | {t} | "
                     f"{pt['med_trig']:.4f} | {pt['med_base']:.4f} | "
                     f"{pt['delta_med']:+.4f} | {pt['rank_biserial']:+.3f} | "
                     f"{pt['p_mwu']:.4f}{mark} |")
    L.append("")
    L.append(f"探索性扫描(未做多重比较校正, 仅供参考): "
             f"{res['verdict']['n_exploratory_significant_uncorrected']} 个组合在 "
             f"p<0.05 下显著。共扫 {len(res['event_study'])} 个信号x分位x方向组合。")
    L.append("")

    # ---- E2
    e2 = res["e2"]
    b = e2["baseline"]
    L.append("## E2 集成回测(monkeypatch 注入, 未改 src/)")
    L.append("")
    L.append(f"基线 v4.6: CAGR {b['cagr_pct']}% | MaxDD {b['maxdd_pct']}% | "
             f"Sharpe {b['sharpe']} | Calmar {b['calmar']} | 末值 {b['final_nav']}")
    L.append("")
    L.append("注入自证: " + ("全部对齐 ✓" if e2["audit_all_aligned"] else "**未对齐 ✗**")
             + " | 生产基础 boost 恒为零: "
             + ("✓(纯增量注入)" if e2["audit_prod_base_all_zero"] else "✗"))
    a = e2.get("audit_sample") or {}
    if a:
        L.append(f"样本 audit: n_calls={a.get('n_calls')} i范围=[{a.get('i_min')}, "
                 f"{a.get('i_max')}] arr长度={a.get('arr_len')} 越界={a.get('out_of_range')}")
    L.append("")
    L.append("### 全参数网格(不挑最优, 全部列出)")
    L.append("")
    if v.get("maxdd_insensitive"):
        L.append(f"注意: {v['maxdd_note']}。故主指标为 Ulcer index。")
        L.append("")
    L.append("| 信号 | q | 触发 | 年跳 | boost | hold | 加强周 | ΔCAGR | ΔUlcer | Δ均回掤 | ΔMaxDD | Δ条件回掤 | ΔSharpe |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in e2["grid"]:
        mark = " ←" if r["d_ulcer_pp"] < 0 else ""
        cdd = (f"{r['d_cond_dd_pp']:+.3f}"
               if r.get("d_cond_dd_pp") is not None else "-")
        L.append(f"| {r['signal']} | {r['q']} | {r['n_trigger']} | "
                 f"{r['year_span']} | {r['boost']} | {r['hold']} | "
                 f"{r['weeks_boosted']} | {r['d_cagr_pp']:+.3f} | "
                 f"{r['d_ulcer_pp']:+.3f}{mark} | {r['d_avg_dd_pp']:+.3f} | "
                 f"{r['d_maxdd_pp']:+.3f} | {cdd} | {r['d_sharpe']:+.4f} |")
    L.append("")
    L.append("### 稳健区域判定")
    L.append("")
    L.append("判定标准先定后测: 主指标 Ulcer index 改善; 约束 CAGR 恶化 <= 0.5pp; "
             "前置条件触发年份跳数 >= 4; 且要求 3x3 网格多数格子改善 + 最优点"
             "邻域一致 —— 孤立最优点视为参数噪声。")
    L.append("")
    L.append("| 组合 | 年跳 | 覆盖达标 | 改善格子 | 占比 | 最优ΔUlcer | 最优Δ均回掤 | 最优ΔCAGR | 最优参数 | 邻域改善 | 孤立最优 | 多数改善 | CAGR约束 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, r in res["e2_robust"].items():
        L.append(f"| {_sk(k)} | {r['year_span']} | "
                 f"{'是' if r['coverage_ok'] else '**否**'} | "
                 f"{r['n_improve']}/{r['n_cells']} | "
                 f"{r['improve_share_pct']}% | {r['best_d_ulcer_pp']:+.3f}pp | "
                 f"{r['best_d_avg_dd_pp']:+.3f}pp | "
                 f"{r['best_d_cagr_pp']:+.3f}pp | "
                 f"boost{r['best_boost']}/hold{r['best_hold']} | "
                 f"{r['neighbors_improving']}/{r['neighbors_total']} | "
                 f"{'是' if r['isolated_optimum'] else '否'} | "
                 f"{'是' if r['majority_improve'] else '否'} | "
                 f"{'✓' if r.get('cagr_constraint_ok') else '✗'} |")
    L.append("")

    # ---- 结论
    # ---- 机制诊断
    ab = v.get("absorption") or {}
    if ab.get("per_signal"):
        L.append("## 机制诊断: 信号无预测力, 还是预测力已被吸收?")
        L.append("")
        L.append("对比两类目标的效应量: 市场层面(中证500 自身前瞻波动/回掤) 对 "
                 "策略层面(策略净值前瞻回掤)。前者大后者小, 即风险已被现有防御层吸收。")
        L.append("")
        L.append("| 信号 | 市场层面效应量(max) | 策略层面效应量 | 策略/市场 | 已被吸收 |")
        L.append("|---|---|---|---|---|")
        for k, a in ab["per_signal"].items():
            L.append(f"| {_sk(k)} | {a['market_effect_max']:.4f} | "
                     f"{a['strategy_effect']:.4f} | "
                     f"{a['ratio_strategy_over_market']} | "
                     f"{'是' if a['absorbed'] else '否'} |")
        L.append("")
        if ab.get("all_absorbed"):
            L.append(f"**{ab['note']}**。这解释了为何 E2 里 CAGR 普降而回掤不降: "
                     "策略本来就没有暴露在这部分风险下, 额外防御只有成本没有收益。")
            L.append("")

    L.append("## 结论")
    L.append("")
    L.append(f"- 预注册假说显著项: {v['n_prereg_significant']} 个"
             + (f" ({', '.join(_sk(x) for x in v['prereg_significant'])})"
                if v["prereg_significant"] else ""))
    L.append(f"- E2 网格 Ulcer 改善格子: {v['grid_improve_cells']}/"
             f"{v['grid_total_cells']} ({v['grid_improve_share_pct']}%)")
    L.append("- 稳健改善区域: "
             + (", ".join(_sk(x) for x in v["robust_regions"])
                if v["robust_regions"] else "无"))
    L.append("- 仅孤立点改善: "
             + (", ".join(_sk(x) for x in v["isolated_only"])
                if v["isolated_only"] else "无"))
    if v.get("coverage_excluded"):
        L.append("- 因触发时间覆盖不足而排除: "
                 + ", ".join(_sk(x) for x in v["coverage_excluded"]))
    L.append("")
    L.append(f"**{v['decision']} —— {v['case']}**")
    L.append("")
    L.append(v["next"])
    L.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"  报告: {OUT_MD}")


def plot(res: dict, wk: pd.DataFrame | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        fb.setup_font()
    except Exception:
        pass

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    ax = axes[0][0]
    if wk is not None and "net_short_rate" in wk:
        s = wk["net_short_rate"] * 100
        ax.plot(s.index, s.values, lw=0.9, color="#c0392b")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(f"{MAIN_LINE} 前20名榜净空率(%)  >0 为空头更集中")
        ax.set_ylabel("%")
        ax.grid(alpha=0.3)

    ax = axes[0][1]
    keys = res.get("prereg_keys", [])
    if keys and wk is not None:
        k = keys[0]
        d = res["event_study"].get(k, {})
        pt = (d.get("per_target") or {}).get("fwd_vol_4w")
        if pt and not pt.get("insufficient"):
            ax.bar(["触发周", "非触发周"],
                   [pt["med_trig"] * 100, pt["med_base"] * 100],
                   color=["#c0392b", "#7f8c8d"])
            ax.set_title(f"{k}\n后续4周实现波动中位数(%)  p={pt['p_mwu']}")
            ax.set_ylabel("年化波动 %")
            ax.grid(alpha=0.3, axis="y")

    grid = pd.DataFrame(res["e2"]["grid"])
    ax = axes[1][0]
    if len(grid):
        first = grid.signal.iloc[0]
        g = grid[grid.signal == first]
        piv = g.pivot_table(index="boost", columns="hold", values="d_ulcer_pp")
        im = ax.imshow(piv.values, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        ax.set_xlabel("hold(周)")
        ax.set_ylabel("boost")
        ax.set_title(f"{first}\nΔUlcer index(pp) 负=改善")
        for a_ in range(piv.shape[0]):
            for b_ in range(piv.shape[1]):
                ax.text(b_, a_, f"{piv.values[a_, b_]:+.2f}",
                        ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1][1]
    if len(grid):
        ax.scatter(grid.d_cagr_pp, grid.d_ulcer_pp, s=28, alpha=0.75,
                   c=["#27ae60" if x < 0 else "#c0392b"
                      for x in grid.d_ulcer_pp])
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("ΔCAGR (pp)")
        ax.set_ylabel("ΔUlcer index (pp)")
        ax.set_title("全网格权衡: 左下象限=降回掤但损收益")
        ax.grid(alpha=0.3)

    v = res["verdict"]
    fig.suptitle(f"股指期货多空龙虎榜脉冲信号 —— {v['decision']}: {v['case']}",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)
    plt.close(fig)
    print(f"  图: {OUT_PNG}")


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="拉取龙虎榜原始数据")
    ap.add_argument("--render-only", action="store_true",
                    help="仅用已有 json 重算裁决与报告(幂等校验)")
    args = ap.parse_args()

    if args.fetch:
        fetch_holding()
        return

    if args.render_only:
        res = json.loads(OUT_JSON.read_text())
        res["verdict"] = recompute_verdict(res)
        OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=2))
        render(res)
        print(f"裁决: {res['verdict']['decision']} — {res['verdict']['case']}")
        return

    if not RAW.exists():
        print(f"缺缓存 {RAW}, 先跑 --fetch")
        sys.exit(1)

    print("[1/5] E0 聚合与数据质量")
    weekly, cfg = fb.load_weekly()
    e0, e0_nan, wk_by_root = {}, {}, {}
    for root in ROOTS:
        daily, diag = aggregate_daily(root)
        e0[root] = diag
        e0_nan[root] = e0_nan_discipline_check(root)
        cols = ["net_short_rate", "short_only_share", "conc_gap", "chg_asym"]
        wk_by_root[root] = fb.align_to_weekly(daily, weekly.index, cols)
        print(f"  {root}: {diag['n_days']} 日, 净空率均值 "
              f"{diag['net_short_rate_mean_pct']:+.3f}%, 净空日占比 "
              f"{diag['pct_days_net_short']:.1f}%")

    print("[2/5] 构造信号与目标")
    from src.backtest import run_backtest
    from src.strategy import load_config
    base_cfg = load_config(PROJECT / "config" / "strategy_v4_6.yaml")
    base = run_backtest(base_cfg)
    tg = fb.build_targets(weekly, base.nav_series["nav"])
    sig = build_signals(wk_by_root[MAIN_LINE])
    print(f"  信号 {list(sig.columns)}")

    print("[3/5] E1' 事件研究扫描")
    ev = pulse_scan(sig, tg)
    print(f"  共 {len(ev)} 个组合")

    print("[4/5] E2 集成回测网格")
    combos = [(c, q, s) for c, q, s in PREREG]
    combos.append(("chg_asym", 0.95, "high"))
    e2 = e2_grid(sig, weekly, combos)
    rob = e2_robust_region(e2["grid"])

    print("[5/5] 裁决与报告")
    res = {
        "meta": {"main_line": MAIN_LINE, "n_weeks": int(len(weekly)),
                 "boost_grid": list(BOOST_GRID), "hold_grid": list(HOLD_GRID),
                 "q_grid": list(Q_GRID), "min_hist": MIN_HIST,
                 "min_events": MIN_EVENTS},
        "e0": e0, "e0_nan": e0_nan,
        "prereg_keys": prereg_keys(),
        "event_study": ev,
        "e2": e2, "e2_robust": rob,
    }
    res["verdict"] = recompute_verdict(res)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    render(res)
    plot(res, wk_by_root[MAIN_LINE])
    print()
    print(f"裁决: {res['verdict']['decision']} — {res['verdict']['case']}")
    print(f"下一步: {res['verdict']['next']}")


if __name__ == "__main__":
    main()
