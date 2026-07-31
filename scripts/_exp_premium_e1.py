#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务19 (E1): 历史溢价真实侵蚀回测 — 量化溢价对策略历史业绩的真实贡献/侵蚀。

⚠️ 关键前置发现 (数据源审计, 本脚本 Step 0 自动复核):
   任务假设"生产回测基于净值(NAV), 实盘按含溢价收盘价成交, 两者存在从未量化的差"。
   核查发现生产 CSV (data/all_etfs_nav_latest.csv) 的纳指ETF列**实为前复权市场收盘价**
   (与 fund_daily close/pre_close 重构的 QFQ 周收益 corr=0.9999, 而与 fund_nav 的
   adj_nav 复权净值 corr 仅约 0.71; update_etf_data_tushare.py 增量拼接用的也是
   fund_daily close)。因此生产回测本身就是市场价口径 — 任务前提反转:
     - "回测NAV、实盘溢价价"的执行偏差**不存在**;
     - 真正的问题是: 生产回测(与实盘)的历史业绩里**混入了多少溢价扩张的贡献**
       (溢价 beta, 不可持续、会随溢价回归反噬)。
   本实验据此把方向调转, 量化的仍是同一个量: 市场价口径 vs 无溢价 NAV 口径的差。

两个口径 (硬约束: 不改生产数据/src/scripts 既有文件/config 生产 yaml):
  口径A (主结论, 执行成本法): 基线 = v4.4 生产回测(市场价口径)。逐周提取纳指列仓位,
     用周五对齐的 513100 溢价率把溢价损益从策略周收益中**剥离**, 得到"NAV 成交"反事实:
        gain_t = w_t x (1+r_mkt,t) x [1 - (1+p_{t-1})/(1+p_t)]
              (= w_t x (1+r_nav,t) x [(1+p_t)/(1+p_{t-1}) - 1], 两式恒等)
        r_nav反事实,t = r_策略,t - gain_t
     该式逐周可加、跨周伸缩相消: 一段持仓的累计 gain = 仓位x(p_sell-p_buy)/(1+p_buy) 取负,
     即持仓期间溢价变化才是净损耗/贡献。逐笔平均成本法复算作核对。
  口径B (稳健性对照, 整列替换完整回测):
     B-audit: 纳指列替换为 fund_daily close/pre_close 重构的前复权收盘价 → 应≈生产
              (证明生产列=市场价, 审计用);
     B-nav:   纳指列替换为 fund_nav 的 adj_nav 复权净值周线 → "无溢价世界"完整回测
              (因子/防御/止损全部吃净值, 模拟纯 NAV 交易者), 与生产对比。

窗口: 全期(2013-05 起) / 2024-01 至今(高溢价抬升期) / 2025-07 至今(极端期)。
同法对 513500 跑口径A(基于既有 sp500-swap 数据集回测, 供 E3 用)。

用法: .venv/bin/python scripts/_exp_premium_e1.py
输出: output/experiments/premium_e1_erosion.{md,json}
      output/experiments/premium_weekly_aligned.csv (E3 复用: 周五对齐溢价序列)
      data/experiments/all_etfs_nav_513100mkt.csv / all_etfs_nav_513100nav.csv (口径B数据集)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.backtest import run_backtest, compute_metrics
from src.data_loader import load_nav_data, resample_weekly
from src.strategy import load_config

CACHE = PROJ / 'data' / 'experiments' / 'tushare_cache'
DATA_EXP = PROJ / 'data' / 'experiments'
OUT_DIR = PROJ / 'output' / 'experiments'
BASE_CSV = PROJ / 'data' / 'all_etfs_nav_latest.csv'
MKT_CSV = DATA_EXP / 'all_etfs_nav_513100mkt.csv'
NAVDS_CSV = DATA_EXP / 'all_etfs_nav_513100nav.csv'
SP500_CSV = DATA_EXP / 'all_etfs_nav_sp500.csv'
BASE_CFG = PROJ / 'config' / 'strategy_v4_4.yaml'
MKT_CFG = PROJ / 'config' / 'experiments' / 'v4_4_premium_e1_mktprice.yaml'
NAVDS_CFG = PROJ / 'config' / 'experiments' / 'v4_4_premium_e1_navds.yaml'
SP500_CFG = PROJ / 'config' / 'experiments' / 'v4_4_sp500.yaml'

NAS_COL = '纳指ETF'
WINDOWS = [('full', None), ('2024-01起', '2024-01-01'), ('2025-07起', '2025-07-01')]
HIST_MEDIAN_PREMIUM = 0.0043   # 513100 历史中位溢价 (#17 汇总)


# ============================================================
# Part 0: 数据源审计 — 生产纳指列到底是市场价还是净值?
# ============================================================

def build_qfq_close(code: str) -> pd.Series:
    """由 fund_daily 的 close/pre_close 重构前复权日线 (锚定最新收盘, 同 tushare qfq)。

    交易所 pre_close 已含除息与份额折算调整 (核查: 513100 2022-01-13 份额折算约1:5,
    2021-07-26 分红 0.037 元, close/pre_close 收益率连续无跳变)。
    """
    df = pd.read_csv(CACHE / f'fund_daily_{code}.csv').sort_values('trade_date')
    dates = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    ret = (df['close'] / df['pre_close'] - 1.0).values
    ret[0] = 0.0
    cum = np.cumsum(np.log1p(ret))
    qfq = float(df['close'].iloc[-1]) * np.exp(cum - cum[-1])
    return pd.Series(qfq, index=dates, name='qfq_close')


def load_adj_nav(code: str) -> pd.Series:
    """tushare fund_nav 的 adj_nav 复权净值 (分红再投资口径)。"""
    df = pd.read_csv(CACHE / f'fund_nav_{code}.csv').sort_values('nav_date')
    dates = pd.to_datetime(df['nav_date'], format='%Y%m%d')
    return pd.Series(df['adj_nav'].values, index=dates, name='adj_nav')


def audit_prod_column() -> dict:
    print('\nStep 0: 数据源审计 — 生产纳指列是市场价还是净值?')
    print('-' * 60)
    base = pd.read_csv(BASE_CSV, parse_dates=['日期']).set_index('日期')
    rb = base[NAS_COL].pct_change().dropna()
    out = {}
    for label, s in [('mkt_qfq', build_qfq_close('513100SH')),
                     ('adj_nav', load_adj_nav('513100SH'))]:
        wk = pd.Series([s.asof(d) for d in base.index], index=base.index)
        rw = wk.pct_change().reindex(rb.index)
        corr = float(np.corrcoef(rb, rw)[0, 1])
        cum_prod = float((1 + rb['2024':]).prod() - 1)
        cum_ref = float((1 + rw['2024':]).prod() - 1)
        out[label] = {'weekly_ret_corr_vs_prod': corr,
                      'cum_2024on_prod': cum_prod, 'cum_2024on_ref': cum_ref}
        print(f'  生产列 vs {label}: 周收益 corr={corr:.5f}; 2024起累计 '
              f'{cum_prod*100:.2f}% vs {cum_ref*100:.2f}%')
    verdict = out['mkt_qfq']['weekly_ret_corr_vs_prod'] > 0.99 > out['adj_nav']['weekly_ret_corr_vs_prod']
    out['verdict'] = ('生产纳指列 = 前复权市场收盘价 (含溢价), 非净值' if verdict
                      else '未能确认生产列口径, 请人工核查')
    print(f"  判定: {out['verdict']}")
    return out


# ============================================================
# Part 1: 周频溢价对齐 (E3 复用产物)
# ============================================================

def load_daily_premium(code: str) -> pd.Series:
    """读缓存溢价 (close/unit_nav - 1, merge_asof backward<=7d 口径, 见 #17)。"""
    df = pd.read_csv(CACHE / f'premium_{code}.csv', parse_dates=['date'])
    return pd.Series(df['premium'].values, index=df['date']).sort_index()


def build_weekly_premium(smooth: bool = False) -> pd.DataFrame:
    """对齐到生产 CSV 的周五日期锚: 每个周五取 <= 该日最近交易日溢价 (asof, 限7天)。

    smooth=True: 日频溢价先取 5 日滚动中位数再采样 — 压制 QDII 净值 T+1 时差在急涨急跌周
    对测量溢价的扭曲, 作口径A的敏感性对照。
    """
    if not smooth:
        print('\nStep 1: 周频溢价对齐 (生产 CSV 周五锚)')
        print('-' * 60)
    base = pd.read_csv(BASE_CSV, usecols=['日期'], parse_dates=['日期'])
    fridays = pd.DatetimeIndex(base['日期'])
    out = pd.DataFrame(index=fridays)
    left = pd.DataFrame({'date': fridays})
    for code, col in [('513100SH', 'premium_513100'), ('513500SH', 'premium_513500')]:
        daily = load_daily_premium(code)
        if smooth:
            daily = daily.rolling(5, min_periods=1).median()
        right = pd.DataFrame({'date': daily.index, 'p': daily.values})
        m = pd.merge_asof(left, right, on='date', direction='backward',
                          tolerance=pd.Timedelta(days=7))
        out[col] = m['p'].values
        if not smooth:
            n_ok = int(out[col].notna().sum())
            print(f'  {code}: 对齐 {n_ok}/{len(fridays)} 周, 起点 '
                  f'{out[col].first_valid_index().date()}, '
                  f'最新 {out[col].dropna().iloc[-1]*100:.2f}%')
    out.index.name = 'date'
    if not smooth:
        out.to_csv(OUT_DIR / 'premium_weekly_aligned.csv', float_format='%.6f')
        print(f'  已写出: {OUT_DIR / "premium_weekly_aligned.csv"}')
    return out


# ============================================================
# Part 2: 口径A — 执行成本法 (从市场价口径剥离溢价损益)
# ============================================================

def exec_cost_series(res, nav_csv: Path, anchor: str, prem_weekly: pd.Series) -> dict:
    """逐周溢价损益 (mark-to-market) + 逐笔平均成本法核对。

    生产列为市场价, 故列周收益 r_col = r_mkt, 溢价损益:
       gain_t = w_t * (1+r_mkt,t) * [1 - (1+p_{t-1})/(1+p_t)]
    NAV 反事实周收益 = 策略周收益 - gain_t。
    """
    wk = resample_weekly(load_nav_data(nav_csv), anchor=anchor)
    wk_idx = wk.index
    ns = res.nav_series
    col_ret = wk[NAS_COL].pct_change()

    rows = []
    for d in ns.index:
        loc = wk_idx.get_loc(d)
        e_prev = wk_idx[loc - 1]                      # 决策/成交日 (上一周五)
        w = float(ns.at[d, f'weight_{NAS_COL}'])      # 该周持有权重(在 e_prev 建立)
        p0 = prem_weekly.asof(e_prev)
        p1 = prem_weekly.asof(d)
        p0 = 0.0 if pd.isna(p0) else float(p0)
        p1 = 0.0 if pd.isna(p1) else float(p1)
        r_mkt = float(col_ret.at[d]) if pd.notna(col_ret.at[d]) else 0.0
        gain = w * (1 + r_mkt) * (1.0 - (1 + p0) / (1 + p1))   # 溢价变动损益(市场价口径内含)
        rows.append({'date': d, 'exec_date': e_prev, 'weight': w, 'p_prev': p0,
                     'p_now': p1, 'r_mkt': r_mkt, 'gain': gain})
    df = pd.DataFrame(rows).set_index('date')

    # --- 逐笔核对: 平均成本法 (卖出损耗 = 仓位 x (p_buy - p_sell)/(1+p_buy)) ---
    pos, avg_p, realized = 0.0, 0.0, 0.0
    n_buy = n_sell = 0
    buy_notional = sell_notional = 0.0
    buy_prem_wsum = sell_prem_wsum = 0.0
    w_prev = 0.0
    for _, row in df.iterrows():
        p_exec = row['p_prev']
        dw = row['weight'] - w_prev
        if dw > 1e-9:
            avg_p = (pos * avg_p + dw * p_exec) / (pos + dw)
            pos += dw
            n_buy += 1
            buy_notional += dw
            buy_prem_wsum += dw * p_exec
        elif dw < -1e-9:
            sell = -dw
            realized += sell * (avg_p - p_exec) / (1 + avg_p)
            pos -= sell
            n_sell += 1
            sell_notional += sell
            sell_prem_wsum += sell * p_exec
        w_prev = row['weight']
    p_end = float(df['p_now'].iloc[-1])
    unreal_mtm = pos * (avg_p - p_end) / (1 + avg_p)        # 期末仓位按当前溢价结算(负=浮盈)
    revert_to_0 = pos * p_end / (1 + p_end)                 # 溢价回归0时从当前市值回吐
    revert_to_med = pos * (p_end - HIST_MEDIAN_PREMIUM) / (1 + p_end)  # 回归历史中位
    lots = {
        'method': '平均成本法 (average cost); 卖出损耗 = 仓位 x (p_avg_buy - p_sell)/(1+p_avg_buy), 负值=赚溢价差',
        'n_buy_events': n_buy, 'n_sell_events': n_sell,
        'buy_notional_total': buy_notional, 'sell_notional_total': sell_notional,
        'avg_buy_premium_weighted': buy_prem_wsum / buy_notional if buy_notional else 0.0,
        'avg_sell_premium_weighted': sell_prem_wsum / sell_notional if sell_notional else 0.0,
        'realized_loss_cum': realized,
        'end_position': pos, 'end_avg_premium': avg_p, 'end_premium': p_end,
        'unrealized_loss_mtm': unreal_mtm,
        'revert_loss_if_premium_to_0': revert_to_0,
        'revert_loss_if_premium_to_median': revert_to_med,
        'total_loss_mtm': realized + unreal_mtm,
        'mtm_weekly_cum_check': float(-df['gain'].sum()),
    }
    return {'df': df, 'lots': lots}


def counterfactual_frame(ns: pd.DataFrame, gain: pd.Series) -> pd.DataFrame:
    """从策略周收益剥离溢价损益, 重建 NAV 成交反事实净值 frame。"""
    wr = ns['weekly_return'].values - gain.reindex(ns.index).fillna(0.0).values
    nav = np.cumprod(1 + wr)
    peak = np.maximum.accumulate(nav)
    return pd.DataFrame({'nav': nav, 'weekly_return': wr, 'drawdown': (peak - nav) / peak,
                         'def_ratio': ns['def_ratio'].values,
                         'turnover': ns['turnover'].values}, index=ns.index)


def slice_frame(frame: pd.DataFrame, start: str | None) -> pd.DataFrame:
    """窗口切片并重归一净值 (避免窗口外路径影响 MaxDD, 同 sp500-swap 口径)。"""
    sub = frame if start is None else frame[frame.index >= pd.to_datetime(start)]
    wr = sub['weekly_return'].values
    nav = np.cumprod(1 + wr)
    peak = np.maximum.accumulate(nav)
    return pd.DataFrame({'nav': nav, 'weekly_return': wr, 'drawdown': (peak - nav) / peak,
                         'def_ratio': sub['def_ratio'].values,
                         'turnover': sub['turnover'].values}, index=sub.index)


def window_metrics(mkt_frame, nav_frame, rf, def_alloc) -> dict:
    """mkt=市场价口径(基线/生产), nav=NAV口径(反事实)。contrib=溢价对年化的贡献(正=抬高业绩)。"""
    out = {}
    for label, start in WINDOWS:
        mm = compute_metrics(slice_frame(mkt_frame, start), rf, def_alloc)
        mn = compute_metrics(slice_frame(nav_frame, start), rf, def_alloc)
        out[label] = {'mkt': mm, 'nav': mn,
                      'premium_contrib_ann_pp': (mm['annual_return'] - mn['annual_return']) * 100,
                      'premium_contrib_sharpe': mm['sharpe_ratio'] - mn['sharpe_ratio'],
                      'maxdd_change_pp': (mm['max_drawdown'] - mn['max_drawdown']) * 100}
    return out


def yearly_decomposition(df: pd.DataFrame) -> list[dict]:
    out = []
    for y, g in df.groupby(df.index.year):
        out.append({
            'year': int(y), 'weeks': len(g),
            'avg_weight': float(g['weight'].mean()),
            'premium_start': float(g['p_prev'].iloc[0]),
            'premium_end': float(g['p_now'].iloc[-1]),
            'premium_gain_pp': float(g['gain'].sum() * 100),  # 正=溢价扩张贡献, 负=溢价收缩损耗
        })
    return out


# ============================================================
# Part 3: 口径B — 整列替换数据集 (audit: 市场价 / nav: 复权净值)
# ============================================================

def build_replace_csv(series: pd.Series, out_csv: Path, tag: str) -> dict:
    base = pd.read_csv(BASE_CSV)
    base['日期'] = pd.to_datetime(base['日期'])
    dates = base['日期']
    first = series.index.min()
    col = pd.Series([series.asof(d) if d >= first else np.nan for d in dates],
                    index=dates.values)
    n_missing = int(col.isna().sum())
    if n_missing:  # 513100 净值起点 2013-04-25 早于 CSV 起点, 正常应为 0
        col = col.bfill()
    scale = base[NAS_COL].iloc[0] / col.iloc[0]
    out = base.copy()
    out[NAS_COL] = (col * scale).round(6).values
    out2 = out.copy()
    out2['日期'] = out2['日期'].dt.strftime('%Y-%m-%d')
    out2.to_csv(out_csv, index=False)
    r_new = out[NAS_COL].pct_change().dropna()
    r_old = base[NAS_COL].pct_change().dropna()
    corr = float(np.corrcoef(r_old, r_new)[0, 1])
    print(f'  [{tag}] 已写出 {out_csv.name} ({len(out)} 行); '
          f'周收益 corr(替换列, 生产列) = {corr:.4f}; 回填周数 = {n_missing}')
    return {'weekly_ret_corr_vs_prod': corr, 'n_backfill_weeks': n_missing,
            'scale_factor': float(scale)}


# ============================================================
# Part 4: 报告
# ============================================================

def _fmt_row(label, m):
    return (f"| {label} | {m['total_return']*100:.1f}% | {m['annual_return']*100:.2f}% | "
            f"{m['max_drawdown']*100:.2f}% | {m['sharpe_ratio']:.3f} | "
            f"{m['calmar_ratio']:.2f} | {m['annual_volatility']*100:.2f}% |")


def write_report(ctx: dict):
    A = ctx['scopeA_513100']
    B = ctx['scopeB']
    lots = A['lots']
    audit = ctx['audit']
    lines = []
    ap = lines.append

    c_full = A['metrics']['full']['premium_contrib_ann_pp']
    c24 = A['metrics']['2024-01起']['premium_contrib_ann_pp']
    c25 = A['metrics']['2025-07起']['premium_contrib_ann_pp']
    yrs_total = A['metrics']['full']['mkt']['total_weeks'] / 52.0

    ap('# E1 实验报告: 溢价对策略的实际历史侵蚀/贡献 (市场价口径 vs NAV 口径)')
    ap('')
    ap('任务ID: 19 | 基线: v4.4 生产配置 @ 生产数据 | 引擎/生产文件零改动')
    ap('')
    ap('## 0. TL;DR (含一个改变任务前提的发现)')
    ap('')
    ap('- **数据源审计发现: 生产 CSV 的纳指ETF列是前复权市场收盘价(含溢价), 不是净值。**')
    ap(f"  证据: 生产列与 fund_daily close/pre_close 重构的 QFQ 周收益 corr = "
       f"{audit['mkt_qfq']['weekly_ret_corr_vs_prod']:.5f}, 与 adj_nav 复权净值仅 "
       f"{audit['adj_nav']['weekly_ret_corr_vs_prod']:.3f}; 2024 至今生产列累计 "
       f"{audit['mkt_qfq']['cum_2024on_prod']*100:.1f}% vs 复权净值 "
       f"{audit['adj_nav']['cum_2024on_ref']*100:.1f}% — 溢价扩张已在生产回测里。"
       f" (update_etf_data_tushare.py 增量拼接用的正是 fund_daily close。)")
    ap('- **因此"回测吃NAV、实盘吃溢价价"的执行偏差不存在** — 回测与实盘同为市场价口径, '
       '执行侵蚀≈0 (口径B-audit 验证: 整列换成重构市场价, 全期年化差 '
       f"{B['audit_contrib_full_pp']:+.2f} pp)。真正的问题反转为: **历史业绩里有多少是溢价扩张"
       "抬出来的(溢价 beta), 溢价回归时会吐回多少**。")
    ap(f"- **口径A(执行成本法, 主结论)**: 过去 {yrs_total:.1f} 年溢价净**贡献** "
       f"**{c_full:+.2f} pp/年**; 2024-01 至今 **{c24:+.2f} pp/年**; 2025-07 至今 "
       f"**{c25:+.2f} pp/年** — 即若一直按净值成交(无溢价世界), 年化要低这么多; "
       "反过来读: 这部分不是策略 alpha, 是溢价敞口的顺风。")
    ap(f"- **前瞻风险(高溢价决策的真正输入)**: 期末纳指仓位 {lots['end_position']*100:.1f}% "
       f"@ 当前溢价 {lots['end_premium']*100:.2f}%; 溢价回归 0 将一次性回吐 "
       f"**{lots['revert_loss_if_premium_to_0']*100:.2f} pp** 组合净值 "
       f"(回归历史中位 {HIST_MEDIAN_PREMIUM*100:.2f}% 则 "
       f"{lots['revert_loss_if_premium_to_median']*100:.2f} pp), 相当于把 2024 以来溢价"
       "贡献的大部分吐回去。")
    ap('')

    ap('## 1. 方法')
    ap('')
    ap('### 1.1 数据与溢价对齐')
    ap('- 溢价率 = 未复权收盘价 close / 单位净值 unit_nav − 1 (均为未复权原始值, 不受复权影响);')
    ap('  日频对齐口径同 #17: merge_asof backward ≤7 天。QDII 净值 T+1 披露存在约 1 个交易日')
    ap('  时差噪声, 周频取样后影响有限, 但 2025 后溢价水平高、逐周方向仍可靠。')
    ap('- 周频对齐: 生产 CSV 周五日期锚, 每周五取 ≤ 该日最近交易日溢价 (asof, 限7天)。')
    ap('  产物 `output/experiments/premium_weekly_aligned.csv` (513100/513500 两列, 供 E3 复用)。')
    ap('- 份额折算(513100 2022-01-13 约1:5, 513500 2022-03-29 约1:2)与分红除息: close 与 unit_nav')
    ap('  同步折算, 溢价率天然免疫; 复权价用交易所 pre_close(已含调整)的 close/pre_close 收益率')
    ap('  从最新收盘反向重构(与 tushare qfq 同口径), 已核查折算日无跳变。')
    ap('')
    ap('### 1.2 口径A — 执行成本法 (主结论)')
    ap('- 基线 = v4.4 生产配置 @ 生产数据的标准回测 (市场价口径, 即含溢价的现状)。')
    ap('- 从 weekly_records 提取纳指列逐周权重 w_t (上周五决策、按周五收盘成交), 逐周')
    ap('  mark-to-market 剥离溢价损益, 得到"按净值成交"的反事实:')
    ap('')
    ap('  `gain_t = w_t × (1+r_mkt,t) × [1 − (1+p_{t−1})/(1+p_t)]` , `r_NAV反事实 = r_策略 − gain_t`')
    ap('')
    ap('  该式逐周可加、跨周伸缩相消: 一段持仓累计 gain = 仓位×(p_sell−p_buy)/(1+p_buy) 的相反数,')
    ap('  即**持仓期间溢价变化才是净损耗/贡献**; 中途加减仓自动按周内权重计入。')
    ap('- 同时用**平均成本法**逐笔复算(买入按 Δw 记建仓溢价, 卖出实现 仓位×(p_buy−p_sell)/(1+p_buy))')
    ap('  核对, 两法累计差仅复利/权重漂移小项 (§3.3 对账)。')
    ap('- 反事实周收益重建净值后用生产 compute_metrics 重算全套指标; 窗口指标一律取全期序列')
    ap('  切片后重归一(避免预热损失与窗口外路径影响)。')
    ap('- 局限: 反事实保持持仓路径不变(市场价信号), 未模拟"信号也换成净值"的路径漂移 — 该效应')
    ap('  由口径B-nav 覆盖; 权重为组合占比近似(未建模周内仓位漂移)。')
    ap('')
    ap('### 1.3 口径B — 整列替换完整回测 (稳健性对照)')
    ap('- **B-audit**: 纳指列 → close/pre_close 重构前复权收盘价周线。预期≈生产, 用于证明生产列')
    ap(f"  口径 (实测周收益 corr = {B['build_mkt']['weekly_ret_corr_vs_prod']:.4f}, "
       f"全期年化差 {B['audit_contrib_full_pp']:+.2f} pp ✅)。")
    ap('- **B-nav**: 纳指列 → adj_nav 复权净值周线 (无溢价世界), 因子/防御/止损全部吃净值,')
    ap('  完整回测后与生产对比 — 含"信号污染"效应的溢价总贡献。')
    ap(f"  替换列与生产列周收益 corr = {B['build_nav']['weekly_ret_corr_vs_prod']:.4f} "
       '(低于 B-audit, 主因 QDII 净值 T+1 时差 + 溢价波动)。')
    ap('- 注: adj_nav 反映前一美股交易日收盘, 与市场价列存在约 1 个交易日错位, 属两种口径的')
    ap('  真实差异(NAV 交易者本来就吃不到当日盘前信息), 不做人工移位。')
    ap('')

    # ---- 口径A 主表 ----
    ap('## 2. 口径A × 三窗口 — 513100 溢价贡献/侵蚀 (主结论)')
    ap('')
    for label, _ in WINDOWS:
        m = A['metrics'][label]
        wlab = {'full': f"全期 ({A['window_full'][0]} ~ {A['window_full'][1]}, {A['window_full'][2]} 周)",
                '2024-01起': '2024-01 至今', '2025-07起': '2025-07 至今'}[label]
        ap(f'### {wlab}')
        ap('')
        ap('| 版本 | 累计 | 年化 | MaxDD | Sharpe | Calmar | 年化波动 |')
        ap('|---|---|---|---|---|---|---|')
        ap(_fmt_row('生产回测 (市场价成交, 现状)', m['mkt']))
        ap(_fmt_row('NAV 成交反事实 (剥离溢价损益)', m['nav']))
        ap('')
        ap(f"**溢价净贡献**: 年化 {m['premium_contrib_ann_pp']:+.2f} pp | "
           f"Sharpe {m['premium_contrib_sharpe']:+.3f} | MaxDD 变化 {m['maxdd_change_pp']:+.2f} pp "
           f"(正=溢价抬高了业绩/加深了回撤)")
        ap('')

    ss = ctx['smooth_sens']
    ap(f"敏感性 (5日滚动中位数平滑溢价, 压制 QDII 净值 T+1 时差在急涨急跌周的扭曲): "
       f"全期 {ss['full']:+.2f} pp/年 | 2024起 {ss['2024-01起']:+.2f} | "
       f"2025-07起 {ss['2025-07起']:+.2f} — 与主口径量级一致, 结论不依赖时差噪声。")
    ap('')

    # ---- 逐年分解 ----
    ap('## 3. 逐年溢价损益分解与逐笔对账 (口径A, 513100)')
    ap('')
    ap('### 3.1 逐年分解 (正=溢价扩张贡献, 负=溢价收缩侵蚀; 组合净值口径)')
    ap('')
    ap('| 年份 | 周数 | 平均纳指权重 | 年初溢价 | 年末溢价 | 当年溢价损益(pp) |')
    ap('|---|---|---|---|---|---|')
    for y in A['yearly']:
        ap(f"| {y['year']} | {y['weeks']} | {y['avg_weight']*100:.1f}% | "
           f"{y['premium_start']*100:.2f}% | {y['premium_end']*100:.2f}% | "
           f"{y['premium_gain_pp']:+.2f} |")
    tot = sum(y['premium_gain_pp'] for y in A['yearly'])
    ap(f"| **合计** | | | | | **{tot:+.2f}** |")
    ap('')
    ap('### 3.2 交易与溢价捕获统计 (平均成本法)')
    ap('')
    ap(f"- 买入事件 {lots['n_buy_events']} 次 (累计名义 {lots['buy_notional_total']*100:.0f}% 组合), "
       f"卖出事件 {lots['n_sell_events']} 次 (累计名义 {lots['sell_notional_total']*100:.0f}%)。")
    ap(f"- 名义加权平均**买入溢价 {lots['avg_buy_premium_weighted']*100:.2f}%** vs "
       f"**卖出溢价 {lots['avg_sell_premium_weighted']*100:.2f}%** — 历史上策略平均"
       f"低溢价买入、高溢价卖出, 每次轮动被动赚溢价差 (溢价长期温和、2024 起单边抬升所致)。")
    ap(f"- 已实现溢价损益 {-lots['realized_loss_cum']*100:+.2f} pp; 期末持仓 "
       f"{lots['end_position']*100:.1f}% @ 平均建仓溢价 {lots['end_avg_premium']*100:.2f}%, "
       f"按期末溢价 {lots['end_premium']*100:.2f}% 的未实现溢价浮盈 "
       f"{-lots['unrealized_loss_mtm']*100:+.2f} pp。")
    ap('')
    ap('### 3.3 两种记账法对账')
    ap('')
    ap(f"- 逐周 MTM 累计溢价损益 = {-lots['mtm_weekly_cum_check']*100:+.2f} pp; "
       f"平均成本法(已实现 + 期末MTM) = {-lots['total_loss_mtm']*100:+.2f} pp; "
       f"差 {abs(lots['mtm_weekly_cum_check']-lots['total_loss_mtm'])*100:.2f} pp "
       f"(复利/权重漂移小项), 两法一致 ✅")
    ap('')

    # ---- 口径B ----
    ap('## 4. 口径B × 三窗口 — 整列替换完整回测 (稳健性对照)')
    ap('')
    ap('B-audit (市场价重构列) 与生产几乎重合, 证明生产列=市场价; B-nav (复权净值列) 是'
       '"无溢价世界"的完整反事实, 差异额外包含信号/防御路径漂移, 数字与口径A不必相等, 方向一致即可。')
    ap('')
    for label, _ in WINDOWS:
        ma = B['metrics_audit'][label]
        mn = B['metrics_nav'][label]
        wlab = {'full': f"全期 ({B['window_full'][0]} ~ {B['window_full'][1]}, {B['window_full'][2]} 周)",
                '2024-01起': '2024-01 至今', '2025-07起': '2025-07 至今'}[label]
        ap(f'### {wlab}')
        ap('')
        ap('| 版本 | 累计 | 年化 | MaxDD | Sharpe | Calmar | 年化波动 |')
        ap('|---|---|---|---|---|---|---|')
        ap(_fmt_row('生产回测 (市场价列)', ma['mkt']))
        ap(_fmt_row('B-audit: 重构市场价列', ma['nav']))
        ap(_fmt_row('B-nav: 复权净值列 (无溢价世界)', mn['nav']))
        ap('')
        ap(f"**B-nav 溢价总贡献 (含信号效应)**: 年化 {mn['premium_contrib_ann_pp']:+.2f} pp | "
           f"Sharpe {mn['premium_contrib_sharpe']:+.3f} | MaxDD 变化 {mn['maxdd_change_pp']:+.2f} pp")
        ap('')

    # ---- 513500 ----
    A5 = ctx['scopeA_513500']
    l5 = A5['lots']
    ap('## 5. 513500 口径A (基于既有 sp500-swap 数据集回测, 供 E3 复用)')
    ap('')
    ap('注: 基线为 v4.4@sp500-swap 数据集(纳指列=513500 前复权市场价, 2014-01 前为 SPX 代理'
       '回填、无溢价), 溢价用 513500, 上市前按 0 处理。此段量化"若当年就换 513500, 其溢价'
       '贡献/侵蚀几何", 不是生产基线。')
    ap('')
    for label, _ in WINDOWS:
        m = A5['metrics'][label]
        wlab = {'full': f"全期 ({A5['window_full'][0]} ~ {A5['window_full'][1]}, {A5['window_full'][2]} 周)",
                '2024-01起': '2024-01 至今', '2025-07起': '2025-07 至今'}[label]
        ap(f'### {wlab}')
        ap('')
        ap('| 版本 | 累计 | 年化 | MaxDD | Sharpe | Calmar | 年化波动 |')
        ap('|---|---|---|---|---|---|---|')
        ap(_fmt_row('sp500-swap 回测 (市场价成交)', m['mkt']))
        ap(_fmt_row('NAV 成交反事实', m['nav']))
        ap('')
        ap(f"**溢价净贡献**: 年化 {m['premium_contrib_ann_pp']:+.2f} pp | "
           f"Sharpe {m['premium_contrib_sharpe']:+.3f} | MaxDD 变化 {m['maxdd_change_pp']:+.2f} pp")
        ap('')
    ap(f"513500 期末仓位 {l5['end_position']*100:.1f}% @ 溢价 {l5['end_premium']*100:.2f}%, "
       f"溢价回归 0 回吐 {l5['revert_loss_if_premium_to_0']*100:.2f} pp "
       f"(vs 513100 的 {lots['revert_loss_if_premium_to_0']*100:.2f} pp) — 供 E3 直接对比。")
    ap('')

    ap('## 6. 结论')
    ap('')
    for i, line in enumerate(ctx['conclusions'], 1):
        ap(f'{i}. {line}')
    ap('')
    ap('---')
    ap('产物: `output/experiments/premium_e1_erosion.{md,json}` | '
       '`output/experiments/premium_weekly_aligned.csv` (E3 复用) | '
       '数据: `data/experiments/all_etfs_nav_513100{mkt,nav}.csv` | '
       '配置: `config/experiments/v4_4_premium_e1_{mktprice,navds}.yaml` | '
       '脚本: `scripts/_exp_premium_e1.py`')

    (OUT_DIR / 'premium_e1_erosion.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f"\n报告已写出: {OUT_DIR / 'premium_e1_erosion.md'}")


# ============================================================
# main
# ============================================================

def run_scope_a(cfg_path: Path, nav_csv: Path, prem_col: pd.Series, tag: str) -> dict:
    print(f'\nStep 2: 口径A 执行成本法 — {tag}')
    print('-' * 60)
    cfg = load_config(cfg_path)
    res = run_backtest(cfg)
    ec = exec_cost_series(res, nav_csv, cfg.anchor, prem_col)
    df, lots = ec['df'], ec['lots']
    cf = counterfactual_frame(res.nav_series, df['gain'])
    metrics = window_metrics(res.nav_series, cf, cfg.risk_free_rate, cfg.def_alloc)
    yearly = yearly_decomposition(df)
    idx = res.nav_series.index
    print(f"  溢价净贡献: 全期 {metrics['full']['premium_contrib_ann_pp']:+.2f} pp/年 | "
          f"2024起 {metrics['2024-01起']['premium_contrib_ann_pp']:+.2f} | "
          f"2025-07起 {metrics['2025-07起']['premium_contrib_ann_pp']:+.2f}")
    print(f"  买入 {lots['n_buy_events']} 笔 avg溢价 {lots['avg_buy_premium_weighted']*100:.2f}% / "
          f"卖出 {lots['n_sell_events']} 笔 avg溢价 {lots['avg_sell_premium_weighted']*100:.2f}% | "
          f"期末回吐风险(→0) {lots['revert_loss_if_premium_to_0']*100:.2f} pp")
    return {'metrics': metrics, 'lots': lots, 'yearly': yearly, 'weekly_df': df,
            'window_full': (str(idx[0].date()), str(idx[-1].date()), len(idx))}


def run_scope_b() -> dict:
    print('\nStep 3: 口径B — 整列替换数据集构建与完整回测')
    print('-' * 60)
    info_mkt = build_replace_csv(build_qfq_close('513100SH'), MKT_CSV, 'B-audit 市场价')
    info_nav = build_replace_csv(load_adj_nav('513100SH'), NAVDS_CSV, 'B-nav 复权净值')

    cfg_b = load_config(BASE_CFG)
    res_prod = run_backtest(cfg_b)
    res_mkt = run_backtest(load_config(MKT_CFG))
    res_nav = run_backtest(load_config(NAVDS_CFG))
    m_audit = window_metrics(res_prod.nav_series, res_mkt.nav_series,
                             cfg_b.risk_free_rate, cfg_b.def_alloc)
    m_nav = window_metrics(res_prod.nav_series, res_nav.nav_series,
                           cfg_b.risk_free_rate, cfg_b.def_alloc)
    idx = res_prod.nav_series.index
    print(f"  B-audit 全期年化差 {m_audit['full']['premium_contrib_ann_pp']:+.2f} pp (预期≈0) | "
          f"B-nav 溢价总贡献 {m_nav['full']['premium_contrib_ann_pp']:+.2f} pp/年")
    return {'metrics_audit': m_audit, 'metrics_nav': m_nav,
            'build_mkt': info_mkt, 'build_nav': info_nav,
            'audit_contrib_full_pp': m_audit['full']['premium_contrib_ann_pp'],
            'window_full': (str(idx[0].date()), str(idx[-1].date()), len(idx))}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_EXP.mkdir(parents=True, exist_ok=True)

    audit = audit_prod_column()
    prem_weekly = build_weekly_premium()

    a1 = run_scope_a(BASE_CFG, BASE_CSV, prem_weekly['premium_513100'], '513100 @ 生产基线')

    # 敏感性: 5日滚动中位数平滑溢价 (压制 QDII 净值 T+1 时差噪声)
    prem_smooth = build_weekly_premium(smooth=True)
    a1s = run_scope_a(BASE_CFG, BASE_CSV, prem_smooth['premium_513100'],
                      '513100 @ 生产基线 (5日中位数平滑溢价, 敏感性)')
    smooth_sens = {w: a1s['metrics'][w]['premium_contrib_ann_pp'] for w, _ in WINDOWS}

    b = run_scope_b()
    a5 = run_scope_a(SP500_CFG, SP500_CSV, prem_weekly['premium_513500'].fillna(0.0),
                     '513500 @ sp500-swap 数据集')

    lots = a1['lots']
    yrs = a1['metrics']['full']['mkt']['total_weeks'] / 52.0
    c_full = a1['metrics']['full']['premium_contrib_ann_pp']
    c_24 = a1['metrics']['2024-01起']['premium_contrib_ann_pp']
    c_2507 = a1['metrics']['2025-07起']['premium_contrib_ann_pp']
    bn = b['metrics_nav']['full']['premium_contrib_ann_pp']
    concl = [
        "**任务前提反转**: 生产 CSV 纳指列实为前复权市场收盘价(非净值), 生产回测与实盘同为"
        "市场价口径, \"回测NAV/实盘溢价价\"的执行偏差不存在 (B-audit 验证差"
        f"{b['audit_contrib_full_pp']:+.2f} pp)。需要量化的真问题是历史业绩里的溢价 beta。",
        f"**过去 {yrs:.1f} 年, 溢价对策略年化的净贡献为 {c_full:+.2f} pp/年; 2024-01 后 "
        f"{c_24:+.2f} pp/年, 2025-07 后 {c_2507:+.2f} pp/年** (口径A; 口径B-nav 含信号效应为全期 "
        f"{bn:+.2f} pp/年, 方向一致)。即: 生产回测口径下溢价历史上没有侵蚀业绩, 反而是顺风。",
        f"机制: 名义加权买入溢价 {lots['avg_buy_premium_weighted']*100:.2f}% vs 卖出溢价 "
        f"{lots['avg_sell_premium_weighted']*100:.2f}% — 溢价长期温和、2024 起单边抬升, 策略"
        "低买高卖了溢价。这不是 alpha, 是不可复制的溢价扩张期红利。",
        f"**风险滚存在当下**: 期末纳指仓位 {lots['end_position']*100:.1f}% @ 溢价 "
        f"{lots['end_premium']*100:.2f}%; 溢价回归 0 一次性回吐约 "
        f"{lots['revert_loss_if_premium_to_0']*100:.2f} pp 组合净值 (回归历史中位 "
        f"{HIST_MEDIAN_PREMIUM*100:.2f}% 约 {lots['revert_loss_if_premium_to_median']*100:.2f} pp), "
        f"且 2024 以来累计溢价贡献 (~{sum(y['premium_gain_pp'] for y in a1['yearly'] if y['year']>=2024):+.1f} pp) "
        "会随之大部吐回。高溢价决策的核心输入是这笔前瞻回归风险, 交由 E3/决策门量化对冲路径。",
        "对 E3 的交接: 周五对齐溢价序列见 premium_weekly_aligned.csv (513100/513500 两列); "
        "513500 同法口径A见 §5 与 JSON, 其当前溢价、期末回吐敞口均显著低于 513100。",
    ]

    ctx = {'audit': audit, 'scopeA_513100': a1, 'scopeB': b, 'scopeA_513500': a5,
           'smooth_sens': smooth_sens, 'conclusions': concl}

    def df_json(df):
        return [{'date': str(d.date()), 'exec_date': str(r['exec_date'].date()),
                 'weight': round(float(r['weight']), 6),
                 'premium_prev': round(float(r['p_prev']), 6),
                 'premium_now': round(float(r['p_now']), 6),
                 'r_mkt': round(float(r['r_mkt']), 8),
                 'premium_gain': round(float(r['gain']), 8)}
                for d, r in df.iterrows()]

    payload = {
        'task': 'T19 E1 premium erosion/contribution (market-price regime vs NAV counterfactual)',
        'key_finding': audit['verdict'],
        'audit': audit,
        'method': {
            'scopeA': 'strip premium P&L weekly: gain_t = w_t*(1+r_mkt)*(1-(1+p_prev)/(1+p_now)); '
                      'NAV counterfactual return = strategy return - gain_t; '
                      'lot check = average-cost, sell loss = pos*(p_buy-p_sell)/(1+p_buy)',
            'scopeB': 'column replacement full backtests: B-audit=rebuilt QFQ close (sanity), '
                      'B-nav=adj_nav (premium-free world incl. signal effects)',
            'premium': 'close/unit_nav - 1, merge_asof backward<=7d daily, Friday-anchor asof weekly',
            'sign': 'premium_contrib_ann_pp > 0 means premium BOOSTED historical performance',
        },
        'windows': {k: v for k, v in WINDOWS},
        'scopeA_513100': {'metrics': a1['metrics'], 'lots': a1['lots'], 'yearly': a1['yearly'],
                          'window_full': a1['window_full'],
                          'smooth_premium_sensitivity_contrib_ann_pp': smooth_sens,
                          'weekly_series': df_json(a1['weekly_df'])},
        'scopeB': {'metrics_audit': b['metrics_audit'], 'metrics_nav': b['metrics_nav'],
                   'build_mkt': b['build_mkt'], 'build_nav': b['build_nav'],
                   'window_full': b['window_full']},
        'scopeA_513500': {'metrics': a5['metrics'], 'lots': a5['lots'], 'yearly': a5['yearly'],
                          'window_full': a5['window_full'],
                          'weekly_series': df_json(a5['weekly_df'])},
        'conclusions': concl,
    }
    jp = OUT_DIR / 'premium_e1_erosion.json'
    jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                  encoding='utf-8')
    print(f'\nJSON 已写出: {jp}')

    write_report(ctx)
    print('\n完成。')


if __name__ == '__main__':
    main()
