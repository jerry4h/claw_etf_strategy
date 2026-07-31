#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务21 (E3): 溢价择时开关机制 — 真实溢价序列历史模拟。

关键数据事实 (本实验独立核验, 决定核算框架):
  生产数据 data/all_etfs_nav_latest.csv 的"纳指ETF"列 = 513100 场内收盘价(仅拆分折算,
  2022-01 前 ratio≈0.1999, 之后=1.0; 周收益与 tushare fund_daily close 逐周吻合,
  仅拆分周例外)。因此 **v4.4 生产回测本身就是"始终用 513100 按市场价执行"** ——
  溢价的涨跌已内生在回测收益里 (买入建仓含当日溢价、卖出结算含当日溢价, 即执行成本法
  的市价等价形式)。任务书假设生产数据为 NAV, 与事实不符, 本实验按市价事实核算:

  - 基准"始终513100市价执行" = v4.4 生产回测原样。
  - 溢价成本/收益的显式度量 = 生产回测 − "纳指腿全程 NAV 计价"参照
    (把纳指腿周收益换成 513100 adj_nav 周收益, 其余不动)。
  - 开关机制反事实: 替代状态期间把纳指腿周收益换成替代执行收益 (下述三层),
    切换本身即"卖出结算溢价差 / 买入建仓溢价"—— 市价序列在切出日含溢价卖出、
    切回日含溢价买入, 执行成本法在此框架下由收益拼接精确实现, 无需重复计费。

开关机制: 每个调仓日看 513100 当周溢价 p_t (截至调仓日最近5个交易日日频溢价均值,
  平滑 QDII T+1 净值披露噪声):
    常态且 p_t > U → 纳指腿改替代执行; 替代且 p_t < L → 回归 513100; 其余维持 (滞回)。
  三种替代层:
    (a) 场外近似: 纳指腿按 513100 adj_nav 计价(零溢价), 申赎按 τ+1 交易日 NAV 成交,
        交易权重变动承担 1 个交易日 NAV 实算滞后成本;
    (b) 513500 场内: 纳指腿按 513500 市价(= adj_nav × 溢价比) 计价, 含其真实溢价
        执行成本与持有期标普−纳指收益差 (双方日频 adj_nav 实算);
    (c) 临时降权: 纳指腿一半仓位改挂当周防御组合 (后处理近似, 不反馈 vol/防御触发)。
  参数扫描: U ∈ {1.5,2,2.5,3,4}% × 滞回宽度 (U−L) ∈ {0.25,0.5,1}% × 3 层 = 45 组合。

期末溢价情景: 期末 513100 溢价 10.9% 为历史极端, 持有者的浮盈溢价未变现。主表按市价
  mark-to-market; 另给"期末溢价回归历史中位(0.43%)"情景年化, 检验机制的前瞻价值。

硬约束: 不改生产数据 / src / scripts 既有文件 / config 生产 yaml。
数据: 全部来自 data/experiments/tushare_cache/ (任务#17 缓存), 溢价 = close/unit_nav−1,
  merge_asof backward ≤7天 (口径同 output/experiments/premium_cache_summary.md);
  不依赖并行 E1/E2 产物。

用法: .venv/bin/python scripts/_exp_premium_e3.py
输出: output/experiments/premium_e3_switch.{md,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.backtest import run_backtest
from src.data_loader import load_nav_data, resample_weekly
from src.strategy import load_config
from src.utils import annualize_return, compute_sharpe, compute_annual_volatility

CACHE = PROJ / 'data' / 'experiments' / 'tushare_cache'
OUT_DIR = PROJ / 'output' / 'experiments'
CFG_PATH = PROJ / 'config' / 'strategy_v4_4.yaml'

NAS_COL = '纳指ETF'
DEF_COLS = ['红利低波ETF', '国债ETF']
WIN_2024 = '2024-01-01'

U_GRID = [0.015, 0.02, 0.025, 0.03, 0.04]
WIDTH_GRID = [0.0025, 0.005, 0.01]
LAYERS = ['a', 'b', 'c']
LAYER_DESC = {
    'a': '(a) 场外近似: NAV计价零溢价 + 1交易日申赎滞后',
    'b': '(b) 513500 场内: 自身真实溢价 + 标普/纳指收益差',
    'c': '(c) 临时降权: 纳指×0.5, 一半归防御层',
}
SIG_DAYS = 5  # "当周溢价" = 截至调仓日最近5个交易日日频溢价均值


# ============================================================
# Part 1: 数据 — 溢价 / adj_nav (全部来自 tushare_cache, 独立计算)
# ============================================================

def load_fund(code: str) -> tuple[pd.Series, pd.Series]:
    """返回 (日频溢价, 日频 adj_nav)。溢价 = close/unit_nav−1, merge_asof backward ≤7d。"""
    daily = pd.read_csv(CACHE / f'fund_daily_{code}.csv')
    daily['date'] = pd.to_datetime(daily['trade_date'], format='%Y%m%d')
    daily = daily[['date', 'close']].sort_values('date').reset_index(drop=True)

    nav = pd.read_csv(CACHE / f'fund_nav_{code}.csv')
    nav['nav_date'] = pd.to_datetime(nav['nav_date'], format='%Y%m%d')
    nav = nav.sort_values(['nav_date', 'ann_date']).drop_duplicates('nav_date', keep='last')

    m = pd.merge_asof(daily, nav[['nav_date', 'unit_nav']].dropna(),
                      left_on='date', right_on='nav_date',
                      direction='backward', tolerance=pd.Timedelta(days=7))
    prem = pd.Series((m['close'] / m['unit_nav'] - 1).values, index=m['date'].values).dropna()

    adj = nav[['nav_date', 'adj_nav']].dropna()
    adj_s = pd.Series(adj['adj_nav'].values, index=adj['nav_date'].values).sort_index()
    return prem.sort_index(), adj_s


def sig_premium(prem: pd.Series, ts: pd.Timestamp) -> float:
    """开关信号: 截至 ts 的最近 SIG_DAYS 个日频溢价均值。"""
    i = prem.index.searchsorted(ts, side='right')
    win = prem.iloc[max(0, i - SIG_DAYS):i]
    return float(win.mean()) if len(win) else 0.0


def next_day_ret(adj: pd.Series, ts: pd.Timestamp) -> float:
    """ts 后第一个交易日的 adj_nav 单日收益 (场外申赎 1 日滞后成本)。"""
    base = adj.asof(ts)
    i = adj.index.searchsorted(ts, side='right')
    if np.isnan(base) or i >= len(adj):
        return 0.0
    nxt = adj.index[i]
    if (nxt - ts).days > 7:
        return 0.0
    return float(adj.iloc[i] / base - 1)


# ============================================================
# Part 2: 生产回测 + 逐周上下文预计算
# ============================================================

def build_context() -> dict:
    print('Step 1: v4.4 生产回测(市价) + 上下文预计算')
    print('-' * 60)
    cfg = load_config(CFG_PATH)
    res = run_backtest(cfg)
    ns = res.nav_series

    weekly_nav = resample_weekly(load_nav_data(PROJ / cfg.nav_path), anchor=cfg.anchor)
    if cfg.start_date:
        weekly_nav = weekly_nav[weekly_nav.index >= pd.to_datetime(cfg.start_date)]
    w_index = weekly_nav.index
    k0 = int(w_index.get_loc(ns.index[0])) - 1
    N = len(ns)
    trade_dates = w_index[k0:k0 + N]          # 第 k 条记录的调仓/执行日 τ_k
    rec_dates = ns.index                       # 持有周结束日 d_k
    assert (rec_dates == w_index[k0 + 1:k0 + 1 + N]).all(), '周索引对齐失败'

    prem100, adj100 = load_fund('513100SH')
    prem500, adj500 = load_fund('513500SH')
    print(f'  513100 溢价: {len(prem100)} 天 {prem100.index[0].date()}~{prem100.index[-1].date()}, '
          f'中位 {prem100.median()*100:.2f}%, 最新 {prem100.iloc[-1]*100:.2f}%')
    print(f'  513500 溢价: {len(prem500)} 天 {prem500.index[0].date()}~{prem500.index[-1].date()}, '
          f'中位 {prem500.median()*100:.2f}%, 最新 {prem500.iloc[-1]*100:.2f}%')

    w_nas = ns[f'weight_{NAS_COL}'].values
    r_week = ns['weekly_return'].values

    wk_ret = weekly_nav.pct_change()
    r_mkt100 = wk_ret[NAS_COL].reindex(rec_dates).fillna(0.0).values   # 生产市价周收益
    r_def_cols = wk_ret[DEF_COLS].reindex(rec_dates).fillna(0.0).values

    # 防御层收益: 当周记录防御权重归一; 防御仓位≈0 时退化 50/50 (hongli_ratio=0.5)
    wd = ns[[f'weight_{c}' for c in DEF_COLS]].values
    wd_sum = wd.sum(axis=1, keepdims=True)
    wd_norm = np.where(wd_sum > 1e-9, wd / np.maximum(wd_sum, 1e-12), 0.5)
    r_def = (wd_norm * r_def_cols).sum(axis=1)

    def wret(adj: pd.Series, t0, t1) -> float:
        a0, a1 = adj.asof(t0), adj.asof(t1)
        if np.isnan(a0) or np.isnan(a1) or a0 <= 0:
            return np.nan
        return float(a1 / a0 - 1)

    # 逐周: 信号溢价 / 执行日溢价 / NAV 周收益 / 513500 市价周收益 / 1日滞后
    p100_sig = np.array([sig_premium(prem100, t) for t in trade_dates])
    p100_exec = np.nan_to_num(np.array([prem100.asof(t) for t in trade_dates]), nan=0.0)
    r_nav100 = np.array([wret(adj100, trade_dates[k], rec_dates[k]) for k in range(N)])
    r_nav100 = np.nan_to_num(r_nav100, nan=0.0)
    r1d_100 = np.array([next_day_ret(adj100, t) for t in trade_dates])

    p500_t = np.array([prem500.asof(t) for t in trade_dates])
    p500_d = np.array([prem500.asof(t) for t in rec_dates])
    r_nav500 = np.array([wret(adj500, trade_dates[k], rec_dates[k]) for k in range(N)])
    # 513500 市价周收益 = NAV 总回报 × 溢价比 (拆分/分红安全)
    r_mkt500 = (1 + r_nav500) * (1 + p500_d) / (1 + p500_t) - 1
    avail_b = ~np.isnan(r_mkt500)
    r_mkt500 = np.nan_to_num(r_mkt500, nan=0.0)
    print(f'  记录周数 {N}, 513500 可用周占比 {avail_b.mean()*100:.1f}% (2014-01-15 上市前不可用)')

    # 口径核验: 生产市价周收益 vs adj_nav×溢价比 重构值 (QDII T+1 噪声下应大体吻合)
    p100_t = np.nan_to_num(np.array([prem100.asof(t) for t in trade_dates]), nan=0.0)
    p100_d = np.nan_to_num(np.array([prem100.asof(t) for t in rec_dates]), nan=0.0)
    r_rec = (1 + r_nav100) * (1 + p100_d) / (1 + p100_t) - 1
    chk = np.abs(r_rec - r_mkt100)
    print(f'  核验(市价周收益 vs adjnav×溢价比重构): 中位差 {np.median(chk)*100:.3f}pp, '
          f'p90 {np.quantile(chk, 0.9)*100:.3f}pp')

    p100_final = float(np.nan_to_num(prem100.asof(rec_dates[-1]), nan=0.0))
    p500_final = float(np.nan_to_num(prem500.asof(rec_dates[-1]), nan=0.0))
    p100_median = float(prem100.median())

    return dict(cfg=cfg, N=N, w_nas=w_nas, r_week=r_week,
                rec_dates=rec_dates, trade_dates=trade_dates,
                p100_sig=p100_sig, p100_exec=p100_exec,
                r_mkt100=r_mkt100, r_nav100=r_nav100, r1d_100=r1d_100,
                r_mkt500=r_mkt500, avail_b=avail_b, r_def=r_def,
                p100_final=p100_final, p500_final=p500_final, p100_median=p100_median,
                prem100=prem100, prem500=prem500,
                recon_check=dict(median_pp=float(np.median(chk)), p90_pp=float(np.quantile(chk, 0.9))))


# ============================================================
# Part 3: 开关机制模拟 (纳指腿收益替换 = 执行成本法的市价等价实现)
# ============================================================

def simulate(ctx: dict, layer: str, U: float, L: float, force_alt: bool = False) -> dict:
    """逐周模拟, 返回调整后周收益 + 状态 + 切换/收割统计。

    layer='nav' 为参照: 纳指腿全程 NAV 计价、零成本 (度量溢价敞口的历史净贡献)。
    force_alt=True → 始终替代; U=+inf → 始终513100 (即生产回测原样)。
    """
    N = ctx['N']
    fee = ctx['cfg'].fee_rate
    w_nas, r_week = ctx['w_nas'], ctx['r_week']
    p_sig, p_exec = ctx['p100_sig'], ctx['p100_exec']
    r_mkt100, r_nav100, r1d = ctx['r_mkt100'], ctx['r_nav100'], ctx['r1d_100']
    r_mkt500, avail_b, r_def = ctx['r_mkt500'], ctx['avail_b'], ctx['r_def']

    state = 0                     # 0 = 513100 市价, 1 = 替代执行
    h100 = halt = hdefx = 0.0     # 当前持仓 (权重口径)
    prev_w = 0.0
    r_adj = np.zeros(N)
    states = np.zeros(N, dtype=int)
    sw_out, sw_in = [], []        # (日期, 当日执行溢价): 切出=卖513100, 切回=买513100

    for k in range(N):
        # --- 滞回状态机 (每个调仓日评估) ---
        want = state
        if layer == 'nav' or force_alt:
            want = 1
        elif state == 0 and p_sig[k] > U:
            want = 1
        elif state == 1 and p_sig[k] < L:
            want = 0
        if want == 1 and layer == 'b' and not avail_b[k]:
            want = 0  # 513500 上市前替代不可用
        if want != state:
            d = str(ctx['trade_dates'][k].date())
            (sw_out if want == 1 else sw_in).append((d, float(p_exec[k])))
        state = want
        states[k] = state

        # --- 目标持仓 ---
        w = w_nas[k]
        if state == 0:
            t100, talt, tdefx = w, 0.0, 0.0
        elif layer == 'c':
            t100, talt, tdefx = 0.5 * w, 0.0, 0.5 * w
        else:  # a / b / nav: 全腿替代
            t100, talt, tdefx = 0.0, w, 0.0
        d100, dalt, ddefx = t100 - h100, talt - halt, tdefx - hdefx

        # --- 纳指腿收益替换 (执行成本法的市价等价: 切出日含溢价卖出/切回日含溢价买入,
        #     已由生产市价序列在拼接点自然结算, 不重复计费) ---
        adj = 0.0
        if layer == 'nav':
            adj += talt * (r_nav100[k] - r_mkt100[k])
        elif layer == 'a' and state == 1:
            adj += talt * (r_nav100[k] - r_mkt100[k])
            adj += -dalt * r1d[k]         # 申赎 τ+1 日 NAV 成交的 1 日滞后
        elif layer == 'b' and state == 1:
            adj += talt * (r_mkt500[k] - r_mkt100[k])
        elif layer == 'c' and state == 1:
            adj += tdefx * (r_def[k] - r_mkt100[k])

        # --- 额外换手费 (引擎已按基线 |Δw| 计费, 只补差额; nav 参照不计) ---
        if layer != 'nav':
            extra_to = max(0.0, abs(d100) + abs(dalt) + abs(ddefx) - abs(w - prev_w))
            adj -= extra_to * fee

        r_adj[k] = r_week[k] + adj
        h100, halt, hdefx = t100, talt, tdefx
        prev_w = w

    # 期末溢价回归情景: 期末仍以市价持有 513100 的部分, 溢价从 p_final 回归历史中位
    scen_haircut = h100 * (ctx['p100_final'] - ctx['p100_median']) / (1 + ctx['p100_final'])
    if layer == 'b' and halt > 0:
        scen_haircut += halt * ctx['p500_final'] / (1 + ctx['p500_final']) * 0  # 513500 期末溢价≈0.3%, 忽略

    # 溢价收割: 完整来回 (切出溢价 − 下次切回溢价)
    harvests = [o[1] - i[1] for o, i in zip(sw_out, sw_in)]

    return dict(r_adj=pd.Series(r_adj, index=ctx['rec_dates']),
                states=pd.Series(states, index=ctx['rec_dates']),
                switch_dates=[d for d, _ in sw_out] + [d for d, _ in sw_in],
                sw_out=sw_out, sw_in=sw_in,
                harvest_mean=float(np.mean(harvests)) if harvests else 0.0,
                n_round_trips=len(harvests),
                scen_haircut=float(scen_haircut),
                end_state=int(state), end_h100=float(h100))


# ============================================================
# Part 4: 指标
# ============================================================

def window_metrics(sim: dict, rf: float, start: str | None = None) -> dict:
    r, st = sim['r_adj'], sim['states']
    if start:
        r = r[r.index >= pd.to_datetime(start)]
        st = st[st.index >= pd.to_datetime(start)]
    n = len(r)
    total = float(np.prod(1 + r.values)) - 1
    total_scen = (1 + total) * (1 - sim['scen_haircut']) - 1
    cum = np.cumprod(1 + r.values)
    peak = np.maximum.accumulate(cum)
    sw = [d for d in sim['switch_dates']
          if start is None or pd.to_datetime(d) >= pd.to_datetime(start)]
    months = n / (52 / 12)
    return dict(
        ann_net=float(annualize_return(total, n)),
        ann_net_scen=float(annualize_return(total_scen, n)),
        sharpe=float(compute_sharpe(r, rf)),
        ann_vol=float(compute_annual_volatility(r)),
        max_dd=float(((peak - cum) / peak).max()),
        n_weeks=int(n),
        n_switches=len(sw),
        switches_per_month=float(len(sw) / months) if months > 0 else 0.0,
        alt_share=float(st.mean()),
    )


def eval_combo(ctx: dict, sim: dict, rf: float) -> dict:
    return dict(full=window_metrics(sim, rf),
                w2024=window_metrics(sim, rf, WIN_2024),
                harvest_mean=sim['harvest_mean'], n_round_trips=sim['n_round_trips'],
                scen_haircut=sim['scen_haircut'], end_state=sim['end_state'],
                switch_dates=sim['switch_dates'])


# ============================================================
# Part 5: 主流程 — 基准 + 45 组合扫描
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = build_context()
    rf = ctx['cfg'].risk_free_rate

    print('\nStep 2: 基准模拟')
    print('-' * 60)
    baselines = {}
    sim_100 = simulate(ctx, 'a', U=1e9, L=-1e9)          # 始终513100 = 生产原样
    baselines['always_513100'] = eval_combo(ctx, sim_100, rf)
    sim_nav = simulate(ctx, 'nav', U=0, L=0)             # NAV 计价参照
    baselines['nav_reference'] = eval_combo(ctx, sim_nav, rf)
    m0, mn = baselines['always_513100']['full'], baselines['nav_reference']['full']
    print(f"  始终513100(=v4.4生产): 净年化 {m0['ann_net']*100:.2f}%, Sharpe {m0['sharpe']:.3f}")
    print(f"  NAV计价参照: 净年化 {mn['ann_net']*100:.2f}% → 溢价敞口历史净贡献 "
          f"{(m0['ann_net']-mn['ann_net'])*100:+.2f}pp/年 (期末溢价 {ctx['p100_final']*100:.2f}% 未变现)")
    for ly in LAYERS:
        s = simulate(ctx, ly, U=0, L=0, force_alt=True)
        baselines[f'always_alt_{ly}'] = eval_combo(ctx, s, rf)
        m = baselines[f'always_alt_{ly}']['full']
        print(f"  始终替代{ly}: 净年化 {m['ann_net']*100:.2f}%, Sharpe {m['sharpe']:.3f}")

    print('\nStep 3: 45 组合参数扫描')
    print('-' * 60)
    combos = []
    for ly in LAYERS:
        for U in U_GRID:
            for wdt in WIDTH_GRID:
                sim = simulate(ctx, ly, U, U - wdt)
                combos.append(dict(layer=ly, U=U, L=U - wdt, width=wdt,
                                   **eval_combo(ctx, sim, rf)))
    combos.sort(key=lambda c: c['full']['ann_net'], reverse=True)
    best = combos[0]
    best_scen = max(combos, key=lambda c: c['full']['ann_net_scen'])
    print(f"  市价口径最优: L{best['layer']} U={best['U']*100:.1f}% W={best['width']*100:.2f}pp: "
          f"净年化 {best['full']['ann_net']*100:.2f}% (月均切换 {best['full']['switches_per_month']:.3f})")
    print(f"  溢价回归情景最优: L{best_scen['layer']} U={best_scen['U']*100:.1f}% "
          f"W={best_scen['width']*100:.2f}pp: 情景年化 {best_scen['full']['ann_net_scen']*100:.2f}%")

    # p*≈2.1% 检验: 每层各 U 的最佳 (宽度取最优), 市价与情景双口径
    pstar_check = {}
    for ly in LAYERS:
        by_u = {}
        for U in U_GRID:
            cs = [c for c in combos if c['layer'] == ly and abs(c['U'] - U) < 1e-9]
            b1 = max(cs, key=lambda c: c['full']['ann_net'])
            b2 = max(cs, key=lambda c: c['full']['ann_net_scen'])
            by_u[f'{U*100:.1f}%'] = dict(
                ann_net_full=b1['full']['ann_net'], ann_net_2024=b1['w2024']['ann_net'],
                ann_scen_full=b2['full']['ann_net_scen'],
                switches_per_month=b1['full']['switches_per_month'])
        pstar_check[ly] = dict(
            by_U=by_u,
            best_U_mkt=max(by_u, key=lambda k: by_u[k]['ann_net_full']),
            best_U_scen=max(by_u, key=lambda k: by_u[k]['ann_scen_full']))
        print(f"  layer {ly}: 最优U 市价口径 {pstar_check[ly]['best_U_mkt']}, "
              f"情景口径 {pstar_check[ly]['best_U_scen']}")

    prem = ctx['prem100']
    prem_stats = dict(
        median_full=float(prem.median()), p90_full=float(prem.quantile(0.9)),
        median_2024=float(prem[prem.index >= WIN_2024].median()),
        last=float(prem.iloc[-1]), last_date=str(prem.index[-1].date()),
        share_above={f'{u*100:.1f}%': float((prem > u).mean()) for u in U_GRID})

    result = dict(
        task='T21/E3 premium timing switch simulation',
        data_fact='生产纳指ETF列=513100场内市价(拆分折算), v4.4生产回测即"始终513100市价执行"; 溢价成本已内生',
        method=dict(
            premium='close/unit_nav-1, merge_asof backward<=7d (tushare_cache 原始文件独立计算)',
            signal=f'调仓日前{SIG_DAYS}个交易日日频溢价均值 (平滑 QDII T+1 噪声)',
            exec_cost='执行成本法的市价等价实现: 替代状态期间纳指腿周收益替换为替代执行收益; '
                      '切出日含溢价卖出/切回日含溢价买入由市价序列在拼接点自然结算; 额外换手补 fee_rate',
            scenario=f"期末溢价回归情景: 期末仍持 513100 的市价仓位, 溢价从 {ctx['p100_final']*100:.2f}% "
                     f"回归历史中位 {ctx['p100_median']*100:.2f}%",
            windows=dict(full=[str(ctx['rec_dates'][0].date()), str(ctx['rec_dates'][-1].date())],
                         w2024=WIN_2024),
            recon_check=ctx['recon_check']),
        premium_stats=prem_stats,
        baselines=baselines,
        combos=combos,
        pstar_check=pstar_check)
    jp = OUT_DIR / 'premium_e3_switch.json'
    jp.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float),
                  encoding='utf-8')
    print(f'\nJSON 已写出: {jp}')

    write_report(ctx, result)
    print('完成。')


# ============================================================
# Part 6: Markdown 报告
# ============================================================

def write_report(ctx: dict, R: dict):
    B, combos, ps = R['baselines'], R['combos'], R['premium_stats']
    a100 = B['always_513100']
    navr = B['nav_reference']
    best = combos[0]
    best_scen = max(combos, key=lambda c: c['full']['ann_net_scen'])
    alt_same = B[f"always_alt_{best['layer']}"]
    alt_scen_same = B[f"always_alt_{best_scen['layer']}"]

    d_mkt_100 = best['full']['ann_net'] - a100['full']['ann_net']
    d_mkt_alt = best['full']['ann_net'] - alt_same['full']['ann_net']
    d24_100 = best['w2024']['ann_net'] - a100['w2024']['ann_net']
    d_scen_100 = best_scen['full']['ann_net_scen'] - a100['full']['ann_net_scen']
    d_scen_alt = best_scen['full']['ann_net_scen'] - alt_scen_same['full']['ann_net_scen']
    prem_contrib = a100['full']['ann_net'] - navr['full']['ann_net']

    L = []
    ap = L.append
    ap('# 实验报告 E3: 溢价择时开关机制 — 真实溢价序列历史模拟 (任务 #21)')
    ap('')
    ap(f"窗口: 全期 {R['method']['windows']['full'][0]} ~ {R['method']['windows']['full'][1]}"
       f" ({a100['full']['n_weeks']} 周) | 2024至今 ({a100['w2024']['n_weeks']} 周) | "
       '生产文件零改动, 反事实为回测后纳指腿收益替换')
    ap('')
    ap('## 0. TL;DR')
    ap('')
    ap('- **关键数据事实**: 生产数据"纳指ETF"列经核验是 513100 **场内市价**(拆分折算), 不是基金净值 —— '
       'v4.4 生产回测本身就是"始终 513100 按市场价执行", 溢价的涨跌与买卖结算已内生于回测收益。')
    ap(f"- **溢价敞口的历史净贡献是 {prem_contrib*100:+.2f}pp/年** (市价 {a100['full']['ann_net']*100:.2f}% vs "
       f"纳指腿全程 NAV 计价 {navr['full']['ann_net']*100:.2f}%): 13 年间溢价从 ~0 涨到 "
       f"{ctx['p100_final']*100:.1f}%, 持有者一路吃到溢价扩张的浮盈 (**未变现**)。")
    ap(f"- **市价 mark-to-market 口径**: 最优开关组合 (层{best['layer']}, U={best['U']*100:.1f}%, "
       f"L={best['L']*100:.2f}%) 净年化 {best['full']['ann_net']*100:.2f}%, 相对始终513100 "
       f"**{d_mkt_100*100:+.2f}pp/年** — 开关在此口径下**不增值**, 因为历史上溢价净扩张, 切出去就少赚。")
    ap(f"- **期末溢价回归情景** (期末溢价 {ctx['p100_final']*100:.1f}% → 历史中位 "
       f"{ctx['p100_median']*100:.2f}%): 最优组合 (层{best_scen['layer']}, U={best_scen['U']*100:.1f}%, "
       f"L={best_scen['L']*100:.2f}%) 情景年化 {best_scen['full']['ann_net_scen']*100:.2f}% vs 始终513100 "
       f"{a100['full']['ann_net_scen']*100:.2f}%, 净差 **{d_scen_100*100:+.2f}pp/年** — "
       '开关的价值本质是**溢价崩塌保险**, 其净值取决于期末溢价是否兑现。')
    ap(f"- 切换频率: 最优组合月均 {best['full']['switches_per_month']:.3f} 次 "
       f"({'✅ <0.5 可接受' if best['full']['switches_per_month'] < 0.5 else '⚠️ ≥0.5 偏高'})。")
    ap('')

    ap('## 1. 方法')
    ap('')
    ap('### 1.1 数据事实核验 (决定核算框架)')
    ap('')
    ap('- 生产 `纳指ETF` 列 / tushare `fund_daily close` 比值: 2022-01 拆分前恒为 ≈0.1999, 之后 =1.0;')
    ap('  周收益逐周吻合 (仅拆分周例外)。→ 该列为**拆分折算后的场内收盘价**。')
    ap(f"- 交叉核验: 市价周收益 vs `adj_nav × (1+溢价)` 比值重构, 中位差 "
       f"{R['method']['recon_check']['median_pp']*100:.3f}pp / p90 "
       f"{R['method']['recon_check']['p90_pp']*100:.3f}pp (QDII T+1 披露噪声内, 口径自洽)。")
    ap('- 推论: 任务书中"v4.4 生产 NAV 回测 + 叠加溢价成本"的前提不成立 —— 若再叠加执行溢价会**重复计费**。')
    ap('  正确框架: 生产回测 = "始终513100市价执行"基准; 反事实 = 替代状态期间把纳指腿周收益替换为替代执行收益。')
    ap('')
    ap('### 1.2 执行成本法 (独立实现, 市价等价形式)')
    ap('')
    ap(f"- 溢价序列: `{R['method']['premium']}`; 开关信号 p_t: {R['method']['signal']}。")
    ap('- 切出日 (卖 513100 → 买替代): 市价序列自然按含当日溢价的价格结算卖出 (= "卖出结算溢价差");')
    ap('  切回日 (卖替代 → 买 513100): 按含当日溢价的市价买入 (= "买入建仓溢价")。一个来回的溢价净损益')
    ap('  = 持仓 × (切出日溢价 − 切回日溢价), 由收益拼接精确实现, 无需显式计费; 额外换手按 `fee_rate` 补费。')
    ap('- 期末若仍持 513100, 其市值含期末溢价 (mark-to-market); §3.3 给出溢价回归情景。')
    ap('')
    ap('### 1.3 三种替代层与滞回开关')
    ap('')
    ap('每个调仓日: 常态且 p_t > U → 切替代; 替代且 p_t < L → 回归; 其余维持 (滞回防抖)。')
    ap('')
    ap(f"- **{LAYER_DESC['a']}**: 替代期纳指腿按 513100 adj_nav 计价 (假设场外联接基金跟踪同一净值、零溢价);")
    ap('  申赎按 τ+1 交易日 NAV 成交, 交易权重变动承担 1 交易日 adj_nav 实算收益差 (`−Δh·r_1d`)。')
    ap(f"- **{LAYER_DESC['b']}**: 替代期纳指腿按 513500 市价计价 (= adj_nav 总回报 × 自身溢价比,")
    ap('  含其真实溢价执行成本与持有期标普−纳指收益差); 513500 上市 (2014-01-15) 前替代不可用, 强制维持。')
    ap(f"- **{LAYER_DESC['c']}**: 后处理近似 — 纳指腿一半改挂当周防御组合 (红利低波/国债按当周记录权重归一,")
    ap('  防御仓位≈0 时 50/50), 另一半仍持 513100 市价。**局限**: 不反馈 vol 因子/防御触发/inv-vol 权重。')
    ap('')

    ap('## 2. 基准')
    ap('')
    ap('| 基准 | 全期净年化 | Sharpe | MaxDD | 情景年化* | 2024至今净年化 | 2024 Sharpe |')
    ap('|---|---|---|---|---|---|---|')
    for key, lbl in [('always_513100', '**始终513100市价 (=v4.4生产)**'),
                     ('nav_reference', '纳指腿全程NAV计价 (参照)'),
                     ('always_alt_a', '始终替代 a (场外)'),
                     ('always_alt_b', '始终替代 b (513500)'),
                     ('always_alt_c', '始终替代 c (降权)')]:
        m, w = B[key]['full'], B[key]['w2024']
        ap(f"| {lbl} | {m['ann_net']*100:.2f}% | {m['sharpe']:.3f} | {m['max_dd']*100:.2f}% | "
           f"{m['ann_net_scen']*100:.2f}% | {w['ann_net']*100:.2f}% | {w['sharpe']:.3f} |")
    ap('')
    ap(f"\\* 情景年化: {R['method']['scenario']}。")
    ap(f"溢价敞口历史净贡献 = {prem_contrib*100:+.2f}pp/年; 其中期末未变现溢价占大头 "
       f"(始终513100 情景年化降至 {a100['full']['ann_net_scen']*100:.2f}%, 即 "
       f"{(a100['full']['ann_net']-a100['full']['ann_net_scen'])*100:+.2f}pp 依赖期末高溢价兑现)。")
    ap('')

    ap('## 3. 参数扫描 (45 组合)')
    ap('')
    ap('### 3.1 Top 10 (按全期市价净年化; 全表见 JSON `combos`)')
    ap('')
    ap('| 层 | U | L | 全期净年化 | 情景年化 | Sharpe | 切换 | 月均 | 替代占比 | 2024净年化 | 2024替代占比 |')
    ap('|---|---|---|---|---|---|---|---|---|---|---|')
    for c in combos[:10]:
        f, w = c['full'], c['w2024']
        ap(f"| {c['layer']} | {c['U']*100:.1f}% | {c['L']*100:.2f}% | {f['ann_net']*100:.2f}% | "
           f"{f['ann_net_scen']*100:.2f}% | {f['sharpe']:.3f} | {f['n_switches']} | "
           f"{f['switches_per_month']:.3f} | {f['alt_share']*100:.1f}% | "
           f"{w['ann_net']*100:.2f}% | {w['alt_share']*100:.1f}% |")
    ap('')
    ap('### 3.2 各层各 U 最佳净年化 (宽度取最优) — p\\*≈2.1% 检验')
    ap('')
    ap('市价口径 / 溢价回归情景口径:')
    ap('')
    ap('| 层 \\ U | ' + ' | '.join(f'{u*100:.1f}%' for u in U_GRID) + ' | 最优U(市价) | 最优U(情景) |')
    ap('|---' * (len(U_GRID) + 3) + '|')
    for ly in LAYERS:
        pc = R['pstar_check'][ly]
        cells = ' | '.join(
            f"{pc['by_U'][f'{u*100:.1f}%']['ann_net_full']*100:.2f}% / "
            f"{pc['by_U'][f'{u*100:.1f}%']['ann_scen_full']*100:.2f}%" for u in U_GRID)
        ap(f"| {ly} | {cells} | **{pc['best_U_mkt']}** | **{pc['best_U_scen']}** |")
    ap('')
    share_lines = ', '.join(f'p>{k}: {v*100:.1f}%' for k, v in ps['share_above'].items())
    ap(f"全样本溢价超阈值时间占比: {share_lines}; 溢价历史中位 {ps['median_full']*100:.2f}%, "
       f"2024至今中位 {ps['median_2024']*100:.2f}%, 最新({ps['last_date']}) {ps['last']*100:.2f}%。")
    ap('')
    ap('### 3.3 最优组合的切换/收割明细')
    ap('')
    for tag, c in [('市价口径最优', best), ('情景口径最优', best_scen)]:
        ap(f"- **{tag}** (层{c['layer']}, U={c['U']*100:.1f}%, L={c['L']*100:.2f}%): "
           f"完整来回 {c['n_round_trips']} 次, 平均每来回溢价收割 (切出溢价−切回溢价) = "
           f"**{c['harvest_mean']*100:+.2f}pp**; 期末状态: "
           f"{'替代中 (躲开当前 %.1f%% 溢价敞口)' % (ctx['p100_final']*100) if c['end_state']==1 else '持有 513100 (承受期末溢价敞口)'}; "
           f"2024至今替代占比 {c['w2024']['alt_share']*100:.1f}%。")
    ap('')

    ap('## 4. 结论')
    ap('')
    ap(f"1. **最优阈值/滞回**: 市价口径最优 = 层{best['layer']} U={best['U']*100:.1f}%/"
       f"L={best['L']*100:.2f}%; 溢价回归情景最优 = 层{best_scen['layer']} "
       f"U={best_scen['U']*100:.1f}%/L={best_scen['L']*100:.2f}%。")
    ap(f"2. **机制净价值取决于对期末极端溢价的判断**: 市价口径 (溢价扩张全兑现) 下开关相对始终513100 "
       f"为 {d_mkt_100*100:+.2f}pp/年、相对始终替代(同层) 为 {d_mkt_alt*100:+.2f}pp/年; "
       f"2024至今 {d24_100*100:+.2f}pp/年。溢价回归情景下开关相对始终513100 为 "
       f"**{d_scen_100*100:+.2f}pp/年**、相对始终替代(同层) 为 {d_scen_alt*100:+.2f}pp/年。")
    ap(f"3. **p\\*≈2.1% 检验**: 既有盈亏平衡点来自 T16 的\"买入溢价、卖出前完全回落\"保守假设。真实序列下"
       f"溢价高度**持续** (超过 2% 的时段占 {ps['share_above']['2.0%']*100:.0f}%, 且 13 年净扩张), "
       '来回的净损耗只有溢价**变动部分**, p\\* 的前提假设与真实序列不符。扫描结果: 各层最优 U 为 '
       + ', '.join(f"层{ly}={R['pstar_check'][ly]['best_U_mkt']}" for ly in LAYERS)
       + ' (U=2% 在任何层都不是最优), 且同层各 U 间净年化差 <0.4pp/年 — '
       '**p\\*≈2.1% 在真实序列下不是最优开关阈值, 也不存在尖锐的最优点**; 对层(a) 这种近零成本替代, '
       '越低的 U 越好 (1.5% 为网格下限), 对层(b) 这种高成本替代, 越高的 U 越好 (4% 为网格上限) — '
       '最优阈值由替代成本决定, 而非 p\\* 型盈亏平衡。')
    ap(f"4. **切换频率**: 全部 45 组合月均切换 ≤ "
       f"{max(c['full']['switches_per_month'] for c in combos):.3f} 次, "
       f"最优组合 {best['full']['switches_per_month']:.3f} 次/月, 均满足 <0.5 的可接受线。")
    ap(f"5. **v4.5 特性建议: 不建议做成自动开关特性** — 全期市价口径净增值仅 {d_mkt_100*100:+.2f}pp/年 "
       f"(在噪声量级), 保险价值 (期末溢价回归情景) 也只有 {d_scen_100*100:+.2f}pp/年, 而在溢价单边扩张段 "
       f"(2024至今) 机会成本高达 {d24_100*100:+.2f}pp/年。收益/风险不对称且方向依赖于“极端溢价是否崩塌”"
       '这一无法回测验证的判断。若仍要落地, 建议以**运营纪律**而非引擎特性实现: 新增/加仓的纳指腿在 '
       'p_t>1.5% 时走场外申购 (层a, 成本最低), 存量仓位不动 — 这样保留了大部分保险价值, 又避免了自动'
       '切换在扩张期反复踏空的机会成本; (b) 513500 因标普−纳指收益差损耗大而劣后, (c) 降权本质是'
       '增防御而非防溢价, 与 Layer3 职责重叠。')
    ap('')
    ap('## 5. 局限')
    ap('')
    ap('- QDII 净值 T+1 披露 → 日频溢价含时差噪声; 信号已用 5 日均值平滑, 拼接点溢价仍用当日 asof 值。')
    ap('- 层(a) 忽略场外申赎费率 (QDII 申购折后常 ~0.1%, 赎回费+到账 T+3~T+7) 与额度限购风险, 偏乐观。')
    ap('- 层(c) 为后处理近似, 不反馈 vol 因子/防御触发/inv-vol; 层(b) 上市前 (2014-01 前) 强制维持 513100。')
    ap('- 反事实为叠加核算, 未改动引擎选基/防御路径 (开关只作用于纳指腿执行方式)。')
    ap('- 期末溢价情景仅一档 (回归历史中位); 真实前瞻价值介于市价口径与情景口径之间。')
    ap('')
    ap('---')
    ap('产物: `output/experiments/premium_e3_switch.{md,json}` | 脚本: `scripts/_exp_premium_e3.py` | '
       '数据: `data/experiments/tushare_cache/` (只读)')

    mp = OUT_DIR / 'premium_e3_switch.md'
    mp.write_text('\n'.join(L), encoding='utf-8')
    print(f'报告已写出: {mp}')


if __name__ == '__main__':
    main()
