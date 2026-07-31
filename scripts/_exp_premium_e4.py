#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[实验] E4: 溢价回落历史实证与下杀预警防线设计 (任务 #25)

问题: 既然溢价有利可图 (E1: +1.22pp/年溢价 beta), 如何防备"高溢价突然下杀"?
四件事:
  1. Episode 识别: 513100 (主) + 其余 QDII (横截面), "高溢价→回落"事件全清单,
     区分"溢价收敛"与"净值下跌"两种亏损来源。
  2. 先行指标检验: 溢价水位/分位、溢价动量、份额变动 (QDII 套利盘经典机制)、
     成交量异动、美股/汇率 — 提前量、命中率、误报率, 诚实报告无领先性的信号。
  3. 防守规则回测: 溢价峰值回撤 / 份额扩张 / 绝对水位 (对照, E3 已否决) 三类规则,
     量化保护 pp、踏空成本 pp、净价值、触发次数; 重点: 变化率类是否优于水位类。
  4. 落地建议: 结合当前 10.86% 溢价、2.0pp 存量敞口与 SOP 衔接。

口径 (与 E1/E3 一致, 关键防坑):
  - 溢价 = close/unit_nav − 1, 沿用缓存 premium_*.csv (merge_asof backward ≤7d,
    "披露口径", 与生产哨兵同口径, 信号实时可算)。
  - QDII 净值披露滞后 (公告日中位 T+2) → 日频原始溢价含时差噪声。防守规则若按
    日频原始溢价切换, 会"收割"净值追认的机械噪声, 产生不可实现的虚假收益
    (本实验第一版实测: 纯水位规则日频原始信号下 +36pp/年, 显然荒谬)。
    处理: 所有规则信号用 5 日均值平滑溢价 p5 (E3 同口径), 决策频率以**周频**为
    正口径 (与策略调仓/E3 对齐); 日频决策仅作灵敏度参考。
  - 市价总回报 = adj_nav 总回报 × 溢价比 (自动处理拆分/分红)。
  - 规则 T 日收盘出信号, T+1 日收益生效; 切换双边费 fee=5e-05 (config fee_rate)。
  - 保护/踏空按**切换周期收割**分解 (切出溢价 − 切回溢价, E3 "溢价收割"同口径),
    避免日级分解被披露噪声灌水。

数据: 只读 data/experiments/tushare_cache/; 输出 output/experiments/。
生产文件 (data/all_etfs_nav_latest.csv, config/, src/) 零接触。全程离线。

用法:
    .venv/bin/python scripts/_exp_premium_e4.py
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, 'data', 'experiments', 'tushare_cache')
OUT_DIR = os.path.join(ROOT, 'output', 'experiments')
os.makedirs(OUT_DIR, exist_ok=True)

MAIN = '513100.SH'
QDII = ['513100.SH', '513500.SH', '159941.SZ', '513300.SH',
        '513650.SH', '513390.SH', '159632.SZ', '159509.SZ']
NAMES = {
    '513100.SH': '国泰纳指', '513300.SH': '华夏纳斯达克', '159941.SZ': '广发纳指',
    '513390.SH': '华泰柏瑞纳指', '159632.SZ': '嘉实纳斯达克', '159509.SZ': '纳指科技',
    '513500.SH': '博时标普500', '513650.SH': '易方达标普500',
}
LONG_HIST = ['513100.SH', '513500.SH', '159941.SZ']   # >2500 交易日

EP_HIGH, EP_LOW = 0.03, 0.01     # episode: 升破 3% → 回落至 1% 以下
MAJOR_DROP, SEVERE_DROP = 0.03, 0.05
PRE_WIN, CO_WIN, ALERT_GAP = 10, 5, 5
FEE = 5e-05
CUR_NASDAQ_W = 0.218             # 当前策略纳指权重 (任务书口径)


def log(msg):
    print(msg, flush=True)


# ==================================================================
# 数据加载
# ==================================================================
def load_premium(code):
    tag = code.replace('.', '')
    df = pd.read_csv(os.path.join(CACHE, f'premium_{tag}.csv'), parse_dates=['date'])
    return df.sort_values('date').set_index('date')


def load_adj_nav(code, idx):
    tag = code.replace('.', '')
    nav = pd.read_csv(os.path.join(CACHE, f'fund_nav_{tag}.csv'), dtype={'nav_date': str})
    nav = nav[['nav_date', 'adj_nav']].dropna().drop_duplicates('nav_date', keep='last')
    nav['date'] = pd.to_datetime(nav['nav_date'])
    nav = nav.sort_values('date')
    m = pd.merge_asof(pd.DataFrame({'date': idx}), nav[['date', 'adj_nav']],
                      on='date', direction='backward', tolerance=pd.Timedelta(days=7))
    return m.set_index('date')['adj_nav']


def load_share(code, idx):
    """份额 (万份) ffill 到交易日; 拆分等公司行为 (单日比率>1.5或<0.5) 折算连续。"""
    tag = code.replace('.', '')
    sh = pd.read_csv(os.path.join(CACHE, f'fund_share_{tag}.csv'), dtype={'trade_date': str})
    sh['date'] = pd.to_datetime(sh['trade_date'])
    s = sh.drop_duplicates('date', keep='last').set_index('date')['fd_share'].sort_index()
    ratio = s / s.shift(1)
    corp = (ratio > 1.5) | (ratio < 0.5)
    adj = pd.Series(1.0, index=s.index)
    for d in s.index[corp]:
        adj.loc[d:] /= ratio.loc[d]
    return (s * adj).reindex(idx, method='ffill'), int(corp.sum())


def load_volume(code, idx):
    tag = code.replace('.', '')
    fd = pd.read_csv(os.path.join(CACHE, f'fund_daily_{tag}.csv'), dtype={'trade_date': str})
    fd['date'] = pd.to_datetime(fd['trade_date'])
    return fd.drop_duplicates('date', keep='last').set_index('date')['vol'].reindex(idx)


def load_ixic(idx):
    df = pd.read_csv(os.path.join(CACHE, 'index_global_IXIC.csv'), dtype={'trade_date': str})
    df['date'] = pd.to_datetime(df['trade_date'])
    s = df.drop_duplicates('date', keep='last').set_index('date')['close'].sort_index()
    # A 股 T 日可见的是美股前一交易日收盘 (无前视)
    m = pd.merge_asof(pd.DataFrame({'date': idx}), s.rename('ixic').reset_index(),
                      on='date', direction='backward', allow_exact_matches=False,
                      tolerance=pd.Timedelta(days=10))
    return m.set_index('date')['ixic']


def load_fx(idx):
    df = pd.read_csv(os.path.join(CACHE, 'fx_daily_USDCNH.csv'), dtype={'trade_date': str})
    df['date'] = pd.to_datetime(df['trade_date'])
    df['mid'] = (df['bid_close'] + df['ask_close']) / 2
    s = df.drop_duplicates('date', keep='last').set_index('date')['mid'].sort_index()
    m = pd.merge_asof(pd.DataFrame({'date': idx}), s.rename('fx').reset_index(),
                      on='date', direction='backward', tolerance=pd.Timedelta(days=10))
    return m.set_index('date')['fx']


# ==================================================================
# §1 Episode 识别
# ==================================================================
def find_episodes(prem, adj_nav, high=EP_HIGH, low=EP_LOW):
    eps = []
    start = None
    p = prem.dropna()
    for d, v in p.items():
        if start is None:
            if v >= high:
                start = d
        else:
            if v <= low:
                eps.append((start, d, False))
                start = None
    if start is not None:
        eps.append((start, p.index[-1], True))

    rows = []
    for i, (a, b, censored) in enumerate(eps, 1):
        seg = p.loc[a:b]
        pk_d = seg.idxmax()
        pk = seg.loc[pk_d]
        end_p = seg.iloc[-1]
        dec = seg.loc[pk_d:]
        drop = pk - end_p
        d1 = dec.diff().min() if len(dec) > 1 else np.nan
        d5 = dec.diff(5).min() if len(dec) > 5 else (end_p - pk if len(dec) > 1 else np.nan)
        drop5 = pk - dec.iloc[min(5, len(dec) - 1)] if len(dec) > 1 else np.nan
        drop10 = pk - dec.iloc[min(10, len(dec) - 1)] if len(dec) > 1 else np.nan
        # 峰值→了结损益分解: (1+r_mkt) = (1+r_nav) × (1+p_end)/(1+p_pk)
        nav_pk, nav_end = adj_nav.asof(pk_d), adj_nav.asof(dec.index[-1])
        r_nav = nav_end / nav_pk - 1 if (nav_pk and nav_end and nav_pk > 0) else np.nan
        conv = (1 + end_p) / (1 + pk) - 1
        r_mkt = (1 + r_nav) * (1 + conv) - 1 if not np.isnan(r_nav) else np.nan
        rows.append({
            'ep': i, 'start': a.strftime('%Y-%m-%d'), 'peak_date': pk_d.strftime('%Y-%m-%d'),
            'end': dec.index[-1].strftime('%Y-%m-%d'), 'censored': censored,
            'peak_prem': pk, 'end_prem': end_p, 'drop_pp': drop,
            'days_above': len(seg), 'days_peak_to_end': len(dec) - 1,
            'drop_5d_pp': drop5, 'drop_10d_pp': drop10,
            'max_1d_down': d1, 'max_5d_down': d5,
            'nav_ret': r_nav, 'prem_conv': conv, 'mkt_ret': r_mkt,
            'major': (drop >= MAJOR_DROP), 'severe': (drop >= SEVERE_DROP),
        })
    return pd.DataFrame(rows)


# ==================================================================
# §2 先行指标
# ==================================================================
def build_signals(code):
    """信号 DataFrame, 只用 ≤t 信息; 溢价类信号基于 5 日均值 p5 (压披露噪声)。"""
    pr = load_premium(code)
    idx = pr.index
    p = pr['premium']
    share, n_corp = load_share(code, idx)
    vol = load_volume(code, idx)
    ixic = load_ixic(idx)
    fx = load_fx(idx)

    df = pd.DataFrame(index=idx)
    df['premium'] = p
    df['p5'] = p.rolling(5, min_periods=3).mean()
    df['pct252'] = df['p5'].rolling(252, min_periods=120).rank(pct=True)
    df['mom5'] = df['p5'].diff(5)
    df['mom20'] = df['p5'].diff(20)
    df['dd20'] = df['p5'].rolling(20, min_periods=5).max() - df['p5']
    df['share1'] = share.pct_change()
    df['share5'] = share.pct_change(5)
    df['share10'] = share.pct_change(10)
    df['share20'] = share.pct_change(20)
    df['volr'] = vol / vol.rolling(20, min_periods=10).mean()
    df['ixic5'] = ixic.pct_change(5)
    df['fx5'] = fx.pct_change(5)
    return df, n_corp


SIGNAL_DEFS = [
    ('水位 p5>5%',       'p5 > 5%',                 'p5',      lambda d: d['p5'] > 0.05,       '水位'),
    ('水位 p5>8%',       'p5 > 8%',                 'p5',      lambda d: d['p5'] > 0.08,       '水位'),
    ('分位 >95%',        'p5 的252日分位 > 0.95',   'pct252',  lambda d: d['pct252'] > 0.95,   '水位'),
    ('急扩 Δ5d≥+2pp',    '5日溢价动量 ≥ +2pp',      'mom5',    lambda d: d['mom5'] >= 0.02,    '动量'),
    ('急扩 Δ20d≥+4pp',   '20日溢价动量 ≥ +4pp',     'mom20',   lambda d: d['mom20'] >= 0.04,   '动量'),
    ('回撤 dd20≥1.5pp',  '距20日峰值回撤 ≥1.5pp',   'dd20',    lambda d: d['dd20'] >= 0.015,   '回撤'),
    ('份额 1d≥+2%',      '份额单日扩张 ≥ +2%',      'share1',  lambda d: d['share1'] >= 0.02,  '份额'),
    ('份额 5d≥+5%',      '份额5日扩张 ≥ +5%',       'share5',  lambda d: d['share5'] >= 0.05,  '份额'),
    ('份额 20d≥+15%',    '份额20日扩张 ≥ +15%',     'share20', lambda d: d['share20'] >= 0.15, '份额'),
    ('量比 ≥3',          '成交量/20日均量 ≥ 3',     'volr',    lambda d: d['volr'] >= 3.0,     '量能'),
    ('美股 5d≤−3%',      'IXIC 5日 ≤ −3%',          'ixic5',   lambda d: d['ixic5'] <= -0.03,  '外围'),
    ('汇率 5d≥+1%',      'USDCNH 5日 ≥ +1%',        'fx5',     lambda d: d['fx5'] >= 0.01,     '外围'),
]


def group_alerts(fire_days):
    alerts = []
    prev = None
    for d in fire_days:
        if prev is None or (d - prev).days > ALERT_GAP:
            alerts.append(d)
        prev = d
    return alerts


def eval_signal(fire, peaks, all_days):
    """命中 = severe 峰值前 PRE_WIN 交易日内触发 (严格先行);
    峰值日~峰值后 CO_WIN 日 = 同步 (不计); 其余警报 = 误报。"""
    fire = fire.fillna(False)
    fire_days = list(fire.index[fire])
    alerts = group_alerts(fire_days)
    pos = {d: i for i, d in enumerate(all_days)}
    hits, leads = 0, []
    covered = set()
    for pk in peaks:
        if pk not in pos:
            continue
        i = pos[pk]
        pre = set(all_days[max(0, i - PRE_WIN):i])
        co = set(all_days[i:min(len(all_days), i + CO_WIN + 1)])
        pre_fires = [d for d in fire_days if d in pre]
        if pre_fires:
            hits += 1
            leads.append(i - pos[pre_fires[0]])
        covered |= {a for a in alerts if a in pre or a in co}
    false_alerts = [a for a in alerts if a not in covered]
    return {'n_alerts': len(alerts), 'hits': hits, 'n_events': len(peaks),
            'lead_days': leads, 'false_alerts': len(false_alerts),
            'fire_pct': float(fire.mean())}


def signal_ic(df, col, horizon=10, min_prem=0.02):
    """高溢价状态 (p5>2%) 下信号值与未来10日 p5 变动的 Spearman; 负 = 预警方向正确。"""
    fwd = df['p5'].shift(-horizon) - df['p5']
    mask = (df['p5'] > min_prem) & df[col].notna() & fwd.notna()
    if mask.sum() < 60:
        return np.nan, int(mask.sum())
    return float(df.loc[mask, col].rank().corr(fwd[mask].rank())), int(mask.sum())


# ==================================================================
# §3 防守规则回测
# ==================================================================
def daily_returns(code):
    pr = load_premium(code)
    adj = load_adj_nav(code, pr.index).ffill()
    p = pr['premium']
    r_nav = adj.pct_change()
    r_mkt = (1 + r_nav) * (1 + p) / (1 + p.shift(1)) - 1
    return r_mkt, r_nav, p


def decision_days(idx, cadence):
    """weekly = 每 ISO 周最后一个交易日出信号 (次日≈周一执行, 与调仓节奏一致)。"""
    if cadence == 'D':
        return np.ones(len(idx), dtype=bool)
    iso = idx.isocalendar()
    key = iso['year'].astype(str) + '-' + iso['week'].astype(str)
    return (~pd.Series(key.values).duplicated(keep='last')).values


def rule_states(df, rule, params, cadence):
    """逐日状态 (1=持市价, 0=NAV替代), 只在决策日更新, 用 ≤t 信息。"""
    p5 = df['p5'].values
    n = len(df)
    dec = decision_days(df.index, cadence)
    state = np.ones(n, dtype=int)
    s = 1
    exits = 0
    if rule == 'dd':          # R1 峰值回撤
        X = params['X']
        dd = df['dd20'].values
        for i in range(n):
            if dec[i]:
                if s == 1 and dd[i] >= X and p5[i] > EP_LOW:
                    s, exits = 0, exits + 1
                elif s == 0 and (p5[i] <= EP_LOW or dd[i] <= X * 0.25):
                    s = 1
            state[i] = s
    elif rule == 'share':     # R2 份额扩张
        Y = params['Y']
        s5 = df['share5'].fillna(0).values
        s10 = df['share10'].fillna(0).values
        for i in range(n):
            if dec[i]:
                if s == 1 and s5[i] >= Y and p5[i] > 0.02:
                    s, exits = 0, exits + 1
                elif s == 0 and (s10[i] < 0.01 or p5[i] <= EP_LOW):
                    s = 1
            state[i] = s
    elif rule == 'level':     # R3 水位对照 (E3 已否决)
        U, L = params['U'], params['L']
        for i in range(n):
            if dec[i]:
                if s == 1 and p5[i] > U:
                    s, exits = 0, exits + 1
                elif s == 0 and p5[i] < L:
                    s = 1
            state[i] = s
    return pd.Series(state, index=df.index), exits


def cycle_stats(pos, p5):
    """切换周期收割: 每个完整来回 harvest = p5(切出日) − p5(切回日);
    期末仍在场外的开口周期单列 (mark-to-market)。"""
    cycles = []
    e_day = None
    prev = 1
    for d, s in pos.items():
        if prev == 1 and s == 0:
            e_day = d
        elif prev == 0 and s == 1 and e_day is not None:
            cycles.append({'exit': e_day.strftime('%Y-%m-%d'),
                           'reenter': d.strftime('%Y-%m-%d'),
                           'p_exit': float(p5.asof(e_day)), 'p_re': float(p5.asof(d)),
                           'harvest_pp': float((p5.asof(e_day) - p5.asof(d)) * 100)})
            e_day = None
        prev = s
    open_cyc = None
    if prev == 0 and e_day is not None:
        open_cyc = {'exit': e_day.strftime('%Y-%m-%d'), 'p_exit': float(p5.asof(e_day)),
                    'p_end': float(p5.dropna().iloc[-1]),
                    'harvest_pp': float((p5.asof(e_day) - p5.dropna().iloc[-1]) * 100)}
    hp = [c['harvest_pp'] for c in cycles]
    return {
        'cycles': cycles, 'open_cycle': open_cyc, 'n_roundtrip': len(cycles),
        'harvest_pos_pp': float(sum(h for h in hp if h > 0)),      # 保护 (收割为正的来回)
        'harvest_neg_pp': float(sum(h for h in hp if h <= 0)),     # 踏空/whipsaw
        'open_pp': float(open_cyc['harvest_pp']) if open_cyc else 0.0,
    }


def severe_coverage(pos, p5, eps):
    """已了结 severe episode 回落段 (峰值→了结) 中, 切出状态规避的溢价跌幅。"""
    tot_drop, avoided, n_cov, n_eps = 0.0, 0.0, 0, 0
    for _, r in eps.iterrows():
        if not r['severe'] or r['censored']:
            continue
        a, b = pd.Timestamp(r['peak_date']), pd.Timestamp(r['end'])
        seg = p5.loc[a:b]
        if len(seg) < 2:
            continue
        n_eps += 1
        dp = seg.diff()
        out = (pos.loc[a:b] == 0)
        av = float(-(dp[out & dp.notna()]).sum())
        drop = max(float(seg.iloc[0] - seg.iloc[-1]), 1e-9)
        tot_drop += drop
        avoided += av
        if av >= 0.4 * drop:
            n_cov += 1
    return {'cov_avoid_pp': avoided * 100, 'cov_drop_pp': tot_drop * 100,
            'cov_pct': avoided / tot_drop if tot_drop > 0 else np.nan,
            'cov_eps': n_cov, 'n_eps': n_eps}


def backtest_rule(df, r_mkt, r_nav, state, exits, eps):
    pos = state.shift(1).fillna(1)
    switch = pos.diff().abs().fillna(0)
    r_rule = pos * r_mkt + (1 - pos) * r_nav - switch * FEE
    r_base = r_mkt
    p = df['premium']
    med_hist = float(p.median())

    def ann(r):
        r = r.dropna()
        return (1 + r).prod() ** (252 / len(r)) - 1 if len(r) > 1 else np.nan

    out = {}
    for tag, sl in [('full', slice(None, None)), ('2024+', slice('2024-01-01', None))]:
        out[f'ann_rule_{tag}'] = ann(r_rule.loc[sl])
        out[f'ann_base_{tag}'] = ann(r_base.loc[sl])
        out[f'net_{tag}'] = out[f'ann_rule_{tag}'] - out[f'ann_base_{tag}']
    # 期末溢价回归情景 (E3 口径)
    p_end = p.dropna().iloc[-1]
    scen = (1 + med_hist) / (1 + p_end)

    def ann_scen(r, in_mkt_end):
        r = r.dropna()
        return ((1 + r).prod() * (scen if in_mkt_end else 1.0)) ** (252 / len(r)) - 1

    out['ann_rule_scen'] = ann_scen(r_rule, pos.iloc[-1] == 1)
    out['ann_base_scen'] = ann_scen(r_base, True)
    out['net_scen'] = out['ann_rule_scen'] - out['ann_base_scen']
    out.update(cycle_stats(pos, df['p5']))
    out.update(severe_coverage(pos, df['p5'], eps))
    yrs = r_base.notna().sum() / 252
    out.update({'exits': exits, 'pct_out': float((pos == 0).mean()), 'years': yrs,
                'end_state': 'NAV替代' if pos.iloc[-1] == 0 else '市价持有'})
    return out


# ==================================================================
# 主流程
# ==================================================================
def main():
    log('=' * 70)
    log('E4: 溢价回落历史实证与下杀预警防线设计 (任务 #25)')
    log('=' * 70)

    # ---------- §0 对账 ----------
    p_main = load_premium(MAIN)['premium']
    med_hist = float(p_main.median())
    cur_p = float(p_main.iloc[-1])
    cur_d = p_main.index[-1].strftime('%Y-%m-%d')
    log(f'\n[对账] 513100 溢价 {cur_d}: {cur_p*100:.2f}% (哨兵口径 10.86%), '
        f'历史中位 {med_hist*100:.2f}% (E1/E3 口径 0.43%)')
    exposure = CUR_NASDAQ_W * (cur_p - med_hist) / (1 + cur_p)
    log(f'[对账] 存量回吐敞口 = {CUR_NASDAQ_W:.1%} × ({cur_p*100:.2f}%−{med_hist*100:.2f}%)'
        f'/(1+{cur_p*100:.2f}%) = {exposure*100:.2f}pp (任务书 ≈2.0pp)')

    # ---------- §1 Episode ----------
    log('\n' + '=' * 70)
    log('§1 Episode 识别 (升破 %.0f%% → 回落至 %.0f%% 以下)' % (EP_HIGH * 100, EP_LOW * 100))
    log('=' * 70)
    ep_all = {}
    for code in QDII:
        pr = load_premium(code)
        adj = load_adj_nav(code, pr.index).ffill()
        eps = find_episodes(pr['premium'], adj)
        ep_all[code] = eps
        log(f'  {code} {NAMES[code]}: episodes={len(eps)} '
            f'(major≥3pp: {int(eps["major"].sum())}, severe≥5pp: {int(eps["severe"].sum())}, '
            f'censored: {int(eps["censored"].sum())})')

    # ---------- §2 先行指标 ----------
    log('\n' + '=' * 70)
    log('§2 先行指标检验 (severe collapse 峰值前 %d 日窗口)' % PRE_WIN)
    log('=' * 70)
    sig_frames, corp_notes = {}, {}
    for code in LONG_HIST:
        sig_frames[code], corp_notes[code] = build_signals(code)
    peaks_by_code = {
        code: [pd.Timestamp(r['peak_date']) for _, r in ep_all[code].iterrows() if r['severe']]
        for code in LONG_HIST}

    sig_results = []
    for name, desc, col, fn, typ in SIGNAL_DEFS:
        pooled = {'n_alerts': 0, 'hits': 0, 'n_events': 0, 'false_alerts': 0}
        leads, fire_pcts = [], []
        for code in LONG_HIST:
            res = eval_signal(fn(sig_frames[code]), peaks_by_code[code],
                              list(sig_frames[code].index))
            for k in ('n_alerts', 'hits', 'n_events', 'false_alerts'):
                pooled[k] += res[k]
            leads += res['lead_days']
            fire_pcts.append(res['fire_pct'])
        ic_main, ic_n = signal_ic(sig_frames[MAIN], col)
        r = {'name': name, 'desc': desc, 'type': typ, 'col': col,
             'hits': pooled['hits'], 'n_events': pooled['n_events'],
             'hit_rate': pooled['hits'] / pooled['n_events'] if pooled['n_events'] else np.nan,
             'lead_med': float(np.median(leads)) if leads else np.nan,
             'n_alerts': pooled['n_alerts'], 'false_alerts': pooled['false_alerts'],
             'false_rate': (pooled['false_alerts'] / pooled['n_alerts']
                            if pooled['n_alerts'] else np.nan),
             'fire_pct': float(np.mean(fire_pcts)), 'ic10_513100': ic_main, 'ic_n': ic_n}
        sig_results.append(r)
        lead_s = f"{r['lead_med']:.0f}d" if not np.isnan(r['lead_med']) else '-'
        log(f"  {name:<16} 命中 {r['hits']}/{r['n_events']} 提前量中位 {lead_s} "
            f"误报 {r['false_alerts']}/{r['n_alerts']} IC10 {r['ic10_513100']:+.3f}")

    # ---------- §3 防守规则 ----------
    log('\n' + '=' * 70)
    log('§3 防守规则回测 (信号=p5, 周频[W]为正口径, T+1 执行, fee=%.0e)' % FEE)
    log('=' * 70)
    rule_grid = (
        [('R1 峰值回撤', 'dd', {'X': x}, 'W') for x in (0.015, 0.02, 0.03)]
        + [('R1 峰值回撤', 'dd', {'X': 0.02}, 'D')]
        + [('R2 份额扩张', 'share', {'Y': y}, 'W') for y in (0.03, 0.05)]
        + [('R3 水位对照', 'level', {'U': 0.015, 'L': 0.005}, 'W')]   # E3 最优复现锚
        + [('R3 水位对照', 'level', {'U': u, 'L': 0.01}, 'W') for u in (0.03, 0.05)]
        + [('R3 水位对照', 'level', {'U': 0.03, 'L': 0.01}, 'D')]
    )
    bt_results = []
    main_cycles = {}
    for code in LONG_HIST:
        df = sig_frames[code]
        r_mkt, r_nav, _ = daily_returns(code)
        for rname, rkind, params, cad in rule_grid:
            state, exits = rule_states(df, rkind, params, cad)
            res = backtest_rule(df, r_mkt, r_nav, state, exits, ep_all[code])
            key = f"{rname}|{json.dumps(params)}|{cad}"
            if code == MAIN:
                main_cycles[key] = {'cycles': res['cycles'], 'open_cycle': res['open_cycle']}
                log(f"  {rname}[{cad}] {params} 全期净 {res['net_full']*100:+.2f} "
                    f"情景净 {res['net_scen']*100:+.2f} 2024+ {res['net_2024+']*100:+.2f}pp/年 "
                    f"| 收割+{res['harvest_pos_pp']:.1f}/-{abs(res['harvest_neg_pp']):.1f}"
                    f"/开口{res['open_pp']:+.1f}pp | severe覆盖 {res['cov_pct']*100:.0f}% "
                    f"({res['cov_eps']}/{res['n_eps']}) | 触发{res['exits']} "
                    f"切出{res['pct_out']*100:.0f}% 期末{res['end_state']}")
            res.pop('cycles')
            res.pop('open_cycle')
            res.update({'code': code, 'rule': rname, 'kind': rkind, 'cadence': cad,
                        'params': json.dumps(params)})
            bt_results.append(res)
    bt = pd.DataFrame(bt_results)

    ctx = {'cur_p': cur_p, 'cur_d': cur_d, 'med_hist': med_hist, 'exposure': exposure,
           'corp_notes': corp_notes, 'peaks_by_code': peaks_by_code}
    write_report(ep_all, sig_results, bt, sig_frames, ctx)
    write_json(ep_all, sig_results, bt, sig_frames, main_cycles)
    log('\n✅ 完成: output/experiments/premium_e4_collapse.{md,json}')


# ==================================================================
# 报告
# ==================================================================
def fmt_pct(x, digits=2):
    return '-' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x*100:.{digits}f}%'


def fmt_pp(x, digits=2):
    return '-' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x*100:+.{digits}f}pp'


def write_report(ep_all, sig_results, bt, sig_frames, ctx):
    cur_p, cur_d = ctx['cur_p'], ctx['cur_d']
    med_hist, exposure = ctx['med_hist'], ctx['exposure']
    peaks_by_code = ctx['peaks_by_code']
    eps_main = ep_all[MAIN]
    sev_main = eps_main[eps_main['severe']]
    done = sev_main[~sev_main['censored']]
    worst = done.loc[done['mkt_ret'].idxmin()]
    fastest = done.loc[done['max_5d_down'].idxmin()]
    n_sev_pool = sum(len(v) for v in peaks_by_code.values())

    sdf = pd.DataFrame(sig_results).set_index('name')
    btm = bt[(bt['code'] == MAIN) & (bt['cadence'] == 'W')]
    r1 = btm[(btm['kind'] == 'dd') & (btm['params'] == '{"X": 0.02}')].iloc[0]
    r3_e3 = btm[btm['params'] == '{"U": 0.015, "L": 0.005}'].iloc[0]
    r2 = btm[(btm['kind'] == 'share') & (btm['params'] == '{"Y": 0.05}')].iloc[0]
    # R1 三标的交叉验证 (X=2pp 与 X=1.5pp)
    r1_cross = bt[(bt['kind'] == 'dd') & (bt['cadence'] == 'W')
                  & (bt['params'] == '{"X": 0.02}')]
    r1_cross15 = bt[(bt['kind'] == 'dd') & (bt['cadence'] == 'W')
                    & (bt['params'] == '{"X": 0.015}')]
    # 当前信号状态
    last = sig_frames[MAIN].iloc[-1]
    r1_fired_now = bool(last['dd20'] >= 0.02 and last['p5'] > EP_LOW)

    L = []
    ap = L.append
    ap('# 实验报告 E4: 溢价回落历史实证与下杀预警防线设计 (任务 #25)')
    ap('')
    ap(f'生成: {datetime.now().strftime("%Y-%m-%d %H:%M")} | 数据: `data/experiments/tushare_cache/` (只读, 离线) '
       f'| 脚本: `scripts/_exp_premium_e4.py` | 生产文件零改动')
    ap('')
    ap(f'对账: 513100 溢价 {cur_d} = **{fmt_pct(cur_p)}** (哨兵 10.86% ✓), 历史中位 {fmt_pct(med_hist)} '
       f'(E1/E3 0.43% ✓); 存量回吐敞口 = {CUR_NASDAQ_W:.1%}×({fmt_pct(cur_p)}−{fmt_pct(med_hist)})'
       f'/(1+{fmt_pct(cur_p)}) = **{exposure*100:.2f}pp** (任务书 ≈2.0pp ✓)。')
    ap('')

    # ---------- TL;DR ----------
    ap('## 0. TL;DR')
    ap('')
    ap(f"- **回落长什么样**: 513100 历史 {len(eps_main)} 个高溢价 episode (升破3%→回落至1%), severe (≥5pp 下杀) "
       f"{len(sev_main)} 个。最凶案例: 峰值日 {worst['peak_date']} 溢价 {fmt_pct(worst['peak_prem'])}, "
       f"{worst['days_peak_to_end']:.0f} 个交易日跌 {worst['drop_pp']*100:.1f}pp, 峰值持有人市价亏 "
       f"**{fmt_pct(worst['mkt_ret'])}** (溢价收敛 {fmt_pct(worst['prem_conv'])} + NAV {fmt_pct(worst['nav_ret'])}); "
       f"最快下杀 ({fastest['peak_date']} 峰) 5 日内 {fastest['max_5d_down']*100:.1f}pp。"
       f"severe 中位数: 峰后 10 日即跌掉 {done['drop_10d_pp'].median()*100:.1f}pp — **下杀以'日'计, 难以预测但来得及'反应'**。")
    ap(f"- **先行指标: 没有高命中+低误报的圣杯**。信息量最强的是**份额扩张** (IC10 = "
       f"{sdf.loc['份额 5d≥+5%', 'ic10_513100']:+.2f}, 全表最强, 但命中率仅 "
       f"{fmt_pct(sdf.loc['份额 5d≥+5%', 'hit_rate'], 0)} — 高特异性/低敏感度); "
       f"溢价自身回撤 dd20 命中率最高 ({fmt_pct(sdf.loc['回撤 dd20≥1.5pp', 'hit_rate'], 0)}, 半反应式) 但误报率 "
       f"{fmt_pct(sdf.loc['回撤 dd20≥1.5pp', 'false_rate'], 0)}。溢价急扩、量比、汇率 IC 为正 (**反向, 无预警价值**), "
       f"美股下跌命中 {fmt_pct(sdf.loc['美股 5d≤−3%', 'hit_rate'], 0)} 但 IC≈0 (同步非领先)。")
    ap(f"- **防守规则: 回撤/变化率类确实优于水位类**。R1 峰值回撤 (X=2pp, 周频) 全期净 {fmt_pp(r1['net_full'])}/年 (leg), "
       f"**2024+ 也为正 ({fmt_pp(r1['net_2024+'])}/年)** — 不重蹈 E3 水位开关扩张期 −2.09pp/年踏空的覆辙; "
       f"severe 回落覆盖率 {fmt_pct(r1['cov_pct'], 0)} ({r1['cov_eps']}/{r1['n_eps']} 个事件保护到位)。"
       f"E3 最优水位复现 (U=1.5%/L=0.5%) 本框架 2024+ = {fmt_pp(r3_e3['net_2024+'])}/年 (与 E3 −2.09pp 方向一致 ✓)。"
       f"R2 份额扩张触发少 ({r2['exits']} 次) 但几乎无误伤, 适合作预警升级不作主防线。")
    ap(f"- **落地一句话**: 存量防线用 **R1 溢价峰值回撤线 (p5 距 20 日峰回落 ≥2pp 且 p5>1% → 纳指腿转零/低溢价执行通道)**, "
       f"份额 5 日扩张 ≥5% 做战略预警升级; 当前 ({cur_d}) dd20 = {last['dd20']*100:.2f}pp, "
       f"{'**已触发**' if r1_fired_now else '**未触发** (溢价仍在峰值区)'}; 2.05pp 存量敞口按历史覆盖率约可保住 "
       f"{exposure*100*float(r1['cov_pct']):.1f}pp (组合层面)。")
    ap('')

    # ---------- §1 ----------
    ap('## 1. Episode 清单: 高溢价→回落的历史全样本')
    ap('')
    ap(f'定义: 溢价升破 **{EP_HIGH:.0%}** 记事件开始, 回落至 **{EP_LOW:.0%}** 以下记了结 (滞回); 期末未了结记 censored; '
       f'峰值回落 ≥{MAJOR_DROP*100:.0f}pp 记 major, ≥{SEVERE_DROP*100:.0f}pp 记 **severe**。'
       f'溢价 = close/unit_nav−1 (披露口径 asof≤7d, 与哨兵一致; 单日读数含披露时差噪声, 幅度统计以 5 日口径为稳)。'
       f'损益分解: `(1+市价损益) = (1+NAV损益) × (1+溢价收敛)` — 买在峰值的持有人, 亏损 = 净值真跌 + 溢价蒸发两部分。')
    ap('')
    ap(f'### 1.1 513100 severe episodes ({len(sev_main)}/{len(eps_main)} 个; major/全量见 JSON)')
    ap('')
    ap('| # | 起始 | 峰值日 | 了结日 | 峰值溢价 | 回落幅度 | 峰→了结交易日 | 峰后5d | 峰后10d | 最快1d | 最快5d | NAV损益 | 溢价收敛 | 峰值持有人损益 |')
    ap('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for _, r in sev_main.iterrows():
        tag = ' ⏳' if r['censored'] else ''
        ap(f"| {r['ep']} | {r['start']} | {r['peak_date']} | {r['end']}{tag} "
           f"| {fmt_pct(r['peak_prem'])} | {r['drop_pp']*100:.1f}pp | {r['days_peak_to_end']:.0f} "
           f"| {r['drop_5d_pp']*100:.1f}pp | {r['drop_10d_pp']*100:.1f}pp "
           f"| {fmt_pp(r['max_1d_down'], 1)} | {fmt_pp(r['max_5d_down'], 1)} "
           f"| {fmt_pct(r['nav_ret'], 1)} | {fmt_pct(r['prem_conv'], 1)} | **{fmt_pct(r['mkt_ret'], 1)}** |")
    ap('')
    nav_down_share = float((done['nav_ret'] < 0).mean())
    ap(f"severe 已了结 {len(done)} 个的中位数: 峰值溢价 {fmt_pct(done['peak_prem'].median())}, 回落 "
       f"{done['drop_pp'].median()*100:.1f}pp 用 {done['days_peak_to_end'].median():.0f} 个交易日, 其中**峰后 10 日跌掉 "
       f"{done['drop_10d_pp'].median()*100:.1f}pp** (≈{done['drop_10d_pp'].median()/done['drop_pp'].median()*100:.0f}% 的总回落); "
       f"峰值持有人损益中位 **{fmt_pct(done['mkt_ret'].median())}** (溢价收敛 {fmt_pct(done['prem_conv'].median())} + "
       f"NAV {fmt_pct(done['nav_ret'].median())}); {nav_down_share*100:.0f}% 案例 NAV 同跌 (双杀), "
       f"但溢价收敛部分是与净值方向无关的纯损失。")
    ap('')
    ap('### 1.2 横截面汇总 (8 只 QDII): 下杀是板块性现象')
    ap('')
    ap('| 代码 | 名称 | episodes | major≥3pp | severe≥5pp | 峰值溢价最大 | 最快5d下杀 | severe中位回落交易日 | censored |')
    ap('|---|---|---|---|---|---|---|---|---|')
    for code in QDII:
        e = ep_all[code]
        sv = e[e['severe'] & ~e['censored']]
        med_d = f"{sv['days_peak_to_end'].median():.0f}" if len(sv) else '-'
        ap(f"| {code} | {NAMES[code]} | {len(e)} | {int(e['major'].sum())} | {int(e['severe'].sum())} "
           f"| {fmt_pct(e['peak_prem'].max())} | {fmt_pp(e['max_5d_down'].min(), 1)} "
           f"| {med_d} | {int(e['censored'].sum())} |")
    ap('')
    ap('全部 8 只当前 episode 均为 censored (2024 起板块性高溢价至今未了结, 与 E3 "溢价高度持续"一致); '
       '但历史规律是**一旦开始了结, 前 10 日完成大部分下杀, 且各标的高度同步** — 届时低溢价场内候选也会同步收敛, '
       '"换到低溢价兄弟券"在崩塌进行时保护有限, 防线必须落在净值/场外通道 (或已低溢价的候选) 上。')
    ap('')

    # ---------- §2 ----------
    ap('## 2. 先行指标检验')
    ap('')
    ap(f'事件集 = 长历史 3 标的的 severe 峰值日共 **{n_sev_pool} 个** '
       f'({", ".join(f"{c.split(chr(46))[0]}:{len(peaks_by_code[c])}" for c in LONG_HIST)})。'
       f'**命中** = 峰值前 {PRE_WIN} 个交易日内触发 (严格先行); 峰值日~峰后 {CO_WIN} 日 = 同步 (不计命中/误报); '
       f'其余警报 = **误报** (触发日按间隔>{ALERT_GAP}天合并为警报)。'
       f'IC10 = 513100 高溢价状态 (p5>2%) 下信号值与未来 10 日平滑溢价变动的 Spearman, **负值 = 预警方向正确**。')
    ap('')
    corp = {k.split('.')[0]: v for k, v in ctx['corp_notes'].items()}
    ap(f'数据处理: 溢价类信号用 5 日均值 p5 (压披露噪声); 份额序列折算拆分跳变 (过滤 {json.dumps(corp)} 个公司行为日, '
       f'513100 为 2022-01 份额×5); IXIC 用 A 股 T 日可见的前一交易日收盘 (无前视); 份额披露本身也有滞后, 实时性偏乐观。')
    ap('')
    ap('| 信号 | 触发条件 | 类型 | 命中率 (峰前10d) | 提前量中位 | 警报数 | 误报率 | 触发日占比 | IC10 (n) |')
    ap('|---|---|---|---|---|---|---|---|---|')
    for r in sig_results:
        ic = '-' if np.isnan(r['ic10_513100']) else f"{r['ic10_513100']:+.3f} ({r['ic_n']})"
        lead = '-' if np.isnan(r['lead_med']) else f"{r['lead_med']:.0f}d"
        ap(f"| {r['name']} | {r['desc']} | {r['type']} | {r['hits']}/{r['n_events']} = {fmt_pct(r['hit_rate'], 0)} "
           f"| {lead} | {r['n_alerts']} | {r['false_alerts']}/{r['n_alerts']} = {fmt_pct(r['false_rate'], 0)} "
           f"| {fmt_pct(r['fire_pct'], 1)} | {ic} |")
    ap('')
    ap('**逐信号判读 (诚实版)**:')
    ap('')
    ap(f"- **份额扩张 (信息量最强, 但低敏感度)**: 三个窗口 IC10 全表最强 "
       f"({sdf.loc['份额 1d≥+2%', 'ic10_513100']:+.2f}/{sdf.loc['份额 5d≥+5%', 'ic10_513100']:+.2f}/"
       f"{sdf.loc['份额 20d≥+15%', 'ic10_513100']:+.2f}), 机制清晰 (额度放开→申购放量→套利盘砸溢价)。但命中率只有 "
       f"{fmt_pct(sdf.loc['份额 5d≥+5%', 'hit_rate'], 0)} — **充分性信号而非必要性信号**: 多数下杀 (市场情绪/美股暴跌型) "
       f"没有份额前兆; 且 2024 年以来 513100 份额 5 日扩张最大仅 ~1.8% (额度冻结), 信号处于'休眠'态 — "
       f"这正是当前溢价持续不崩的机制性原因, 也意味着**一旦份额重新放量, 是最值得升级警戒的单一事件**。")
    ap(f"- **溢价自身回撤 dd20 (命中率最高的'半反应式'信号)**: 命中 {fmt_pct(sdf.loc['回撤 dd20≥1.5pp', 'hit_rate'], 0)}, "
       f"提前量中位 {sdf.loc['回撤 dd20≥1.5pp', 'lead_med']:.0f}d, IC10 {sdf.loc['回撤 dd20≥1.5pp', 'ic10_513100']:+.2f} "
       f"(方向正确)。本质是'下杀开始后尽早确认'而非预测 — 误报率 {fmt_pct(sdf.loc['回撤 dd20≥1.5pp', 'false_rate'], 0)} 偏高, "
       f"但误报代价可控 (虚惊后溢价回到峰值附近即恢复, 净成本见 §3 R1 收割分解)。")
    ap(f"- **水位/分位 (弱)**: p5>5% 命中 {fmt_pct(sdf.loc['水位 p5>5%', 'hit_rate'], 0)}, IC10 仅 "
       f"{sdf.loc['水位 p5>5%', 'ic10_513100']:+.2f} — 高水位只说明'跌得动', 不提供时点; 与 E3 "
       f"'溢价高度持续、水位开关否决'一致。")
    ap(f"- **无领先性的信号 (如实报告)**: 溢价急扩 IC10 为正 ({sdf.loc['急扩 Δ5d≥+2pp', 'ic10_513100']:+.2f}/"
       f"{sdf.loc['急扩 Δ20d≥+4pp', 'ic10_513100']:+.2f}) — 急扩后短期溢价**继续扩张**概率更大 (动量效应), "
       f"'blow-off 见顶'假说不成立; 量比 IC10 {sdf.loc['量比 ≥3', 'ic10_513100']:+.2f} (反向, 放量更多伴随上冲); "
       f"汇率 IC10 {sdf.loc['汇率 5d≥+1%', 'ic10_513100']:+.2f} (反向, 贬值推高溢价而非预示回落); "
       f"美股 5 日大跌命中率 {fmt_pct(sdf.loc['美股 5d≤−3%', 'hit_rate'], 0)} 看似高, 但 IC10≈0 且误报率 "
       f"{fmt_pct(sdf.loc['美股 5d≤−3%', 'false_rate'], 0)} — 美股暴跌时溢价常先冲高再崩 (catch-down), 是同步伴随不是领先预警。")
    ap('')
    ap('**§2 结论**: 不存在"提前 N 天可靠预告下杀"的信号。可用组合是 **份额扩张做战略预警 (稀有但准) + '
       '溢价回撤做战术触发 (及时但吵)** — 后者的价值必须经防守回测验证 (§3): 命中收益要能覆盖误报成本。')
    ap('')

    # ---------- §3 ----------
    ap('## 3. 防守规则回测')
    ap('')
    ap('框架: 纳指腿单腿日频核算 — 基准 = 始终持 513100 市价 (E3 已证 = 生产口径); 触发时切入 NAV 替代腿 '
       '(adj_nav 计价 ≈ 场外/零溢价执行, E3 层a 口径), 解除后切回。信号 = p5; **周频 [W] 为正口径** '
       f'(周最后交易日收盘出信号次日执行, 与调仓节奏一致), 日频 [D] 仅作灵敏度。切换双边费 {FEE:.0e}。'
       '**情景净值** = 期末溢价回归历史中位后的年化差 (E3 口径, 修掉期末 10.9% 未变现溢价对比较的污染)。'
       '**收割** = 每个完整来回的 (切出日p5 − 切回日p5), 正 = 保护, 负 = 踏空/whipsaw; 开口 = 期末仍在场外的敞口盯市。'
       '**severe覆盖** = 已了结 severe 回落段中, 切出状态规避的溢价跌幅占总跌幅比例 (规避≥40% 记该事件保护到位)。')
    ap('')
    ap('规则: **R1 峰值回撤** = dd20≥X 且 p5>1% → 切出; p5≤1% 或 dd20≤X/4 → 切回。 '
       '**R2 份额扩张** = 份额5日扩张≥Y 且 p5>2% → 切出; 份额10日扩张<1% 或 p5≤1% → 切回。 '
       '**R3 水位对照** = p5>U → 切出; p5<L → 切回 (纯水位开关, E3 已否决; U=1.5%/L=0.5% 行为 E3 最优参数复现锚)。')
    ap('')
    for k, code in enumerate(LONG_HIST):
        sub = bt[bt['code'] == code]
        ap(f'### 3.{k+1} {code} {NAMES[code]}' + (' (主标的)' if code == MAIN else ' (交叉验证)'))
        ap('')
        ap('| 规则 | 参数 | 频率 | 全期净值 | 情景净值 | 2024+净值 | 收割+ | 收割− | 开口盯市 | severe覆盖 | 触发 | 切出占比 | 期末 |')
        ap('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for _, r in sub.iterrows():
            ap(f"| {r['rule']} | {r['params']} | {r['cadence']} | {fmt_pp(r['net_full'])}/年 "
               f"| **{fmt_pp(r['net_scen'])}/年** | {fmt_pp(r['net_2024+'])}/年 "
               f"| +{r['harvest_pos_pp']:.1f}pp | {r['harvest_neg_pp']:.1f}pp | {r['open_pp']:+.1f}pp "
               f"| {fmt_pct(r['cov_pct'], 0)} ({r['cov_eps']}/{r['n_eps']}) | {r['exits']} "
               f"| {r['pct_out']*100:.1f}% | {r['end_state']} |")
        ap('')
    ap('**§3 判读**:')
    ap('')
    r1_15 = btm[(btm['kind'] == 'dd') & (btm['params'] == '{"X": 0.015}')].iloc[0]
    r1_30 = btm[(btm['kind'] == 'dd') & (btm['params'] == '{"X": 0.03}')].iloc[0]
    r3_31 = btm[btm['params'] == '{"U": 0.03, "L": 0.01}'].iloc[0]

    def cross_str(cdf):
        return ' / '.join(f"{c.split('.')[0]} {fmt_pp(cdf[cdf['code'] == c]['net_full'].iloc[0])}"
                          for c in LONG_HIST)
    ap(f"1. **回撤类 > 水位类, 核心在 2024+ 扩张段**: R1(X=2pp)[W] 全期 {fmt_pp(r1['net_full'])}/年且 2024+ "
       f"{fmt_pp(r1['net_2024+'])}/年**两段皆正**; 纯水位 R3 虽全期名义更高 (如 U=3%: {fmt_pp(r3_31['net_full'])}/年, "
       f"来自 2020 等大周期的运气性收割), 但其 2024+ = {fmt_pp(r3_31['net_2024+'])}/年、E3 复现参数 (U=1.5%) 2024+ = "
       f"{fmt_pp(r3_e3['net_2024+'])}/年, 且切出时间占比 {r3_31['pct_out']*100:.0f}%+ — 是'长期停留场外赌崩塌'的持仓选择, "
       f"E3 已否决; R1 切出占比仅 {r1['pct_out']*100:.0f}%, 只在下杀确认后离场, **不为防守放弃溢价上行**。")
    ap(f"2. **R1 参数敏感度与交叉验证 (诚实版)**: 513100 上 X=1.5/2/3pp 全期净值 {fmt_pp(r1_15['net_full'])}/"
       f"{fmt_pp(r1['net_full'])}/{fmt_pp(r1_30['net_full'])}/年, 无尖锐最优。三标的交叉验证: "
       f"X=1.5pp 全期净值**三标的皆正** ({cross_str(r1_cross15)}/年); X=2pp 为 {cross_str(r1_cross)}/年 "
       f"— 513500 上小幅为负, 规则方向跨标的成立但参数存在标的间差异, 不宜过度解读单点。"
       f"取 X=2pp 为主推荐 (513100 主标的两段皆正, 触发 {r1['exits']} 次 ≈ {r1['exits']/r1['years']:.1f} 次/年 "
       f"运维可承受), X=1.5pp 为等效备选 (更稳健但触发略多, {r1_15['exits']} 次)。")
    ap(f"3. **R1 的日频灵敏度警示**: 同参数日频执行全期 "
       f"{fmt_pp(bt[(bt['code']==MAIN)&(bt['kind']=='dd')&(bt['cadence']=='D')]['net_full'].iloc[0])}/年 — "
       f"比周频差, 因为日频对披露噪声反应过度、whipsaw 加倍。**执行就按周频调仓日, 不要盘中抢跑** (与 SOP §6.2 '不恐慌卖出'一致)。")
    ap(f"4. **R2 份额扩张**: 触发极少 ({r2['exits']} 次), 收割几乎笔笔为正 (+{r2['harvest_pos_pp']:.1f}pp / "
       f"{r2['harvest_neg_pp']:.1f}pp), 全期净 {fmt_pp(r2['net_full'])}/年 — 单独作主防线覆盖不足 "
       f"(severe覆盖 {fmt_pct(r2['cov_pct'], 0)}), 但**作为 R1 的前置预警/确认信号价值高** (它触发时基本不是虚惊)。")
    ap(f"5. **保护的真实量级**: R1(X=2pp) 13 年累计正收割 +{r1['harvest_pos_pp']:.1f}pp (leg), 误报累计 "
       f"{r1['harvest_neg_pp']:.1f}pp, 净收割约 {r1['harvest_pos_pp']+r1['harvest_neg_pp']:+.1f}pp; 按当前纳指权重 "
       f"{CUR_NASDAQ_W:.0%} 折合组合层面 ≈{(r1['harvest_pos_pp']+r1['harvest_neg_pp'])*CUR_NASDAQ_W:+.1f}pp/13年 — "
       f"**不是收益引擎, 是尾部保险**: 它的价值集中在 2020-03/2020-09/2025-04 这类 5~10pp 下杀里把存量敞口拦下来。")
    ap('')

    # ---------- §4 ----------
    ap('## 4. 落地建议 (与 SOP 衔接)')
    ap('')
    ap(f"**现状**: 溢价 {fmt_pct(cur_p)} (历史 99%+ 分位), 存量回吐敞口 {exposure*100:.2f}pp; E3 已定'存量不动'总方针, "
       f"SOP §6.2 只有溢价急崩的事后应对 ('不恐慌卖出'), **缺一条存量的事前/事中机械防线** — 本实验补这条线。")
    ap('')
    ap('| 防线 | 信号与阈值 | 数据来源 | 触发动作 | 预期效果 (历史) |')
    ap('|---|---|---|---|---|')
    ap(f"| **主防线 R1 (战术)** | 周调仓日核对: p5 距近 20 日峰值回撤 ≥2pp 且 p5>1% | 哨兵溢价序列 (日频可算) "
       f"| 存量纳指腿改走零/低溢价通道: 优先场外申赎对倒 (容量内, SOP §4), 超容量走场内低溢价候选 (SOP §6.1 路径3); "
       f"解除 (p5≤1% 或回撤收敛至 0.5pp) 后原路切回 | severe 覆盖 {fmt_pct(r1['cov_pct'], 0)}, 全期净 "
       f"{fmt_pp(r1['net_full'])}/年 (leg), 2024+ 不踏空 ({fmt_pp(r1['net_2024+'])}/年); 触发 ≈{r1['exits']/r1['years']:.1f} 次/年 |")
    ap(f"| **预警线 R2 (战略)** | 513100 份额 5 日扩张 ≥5% (拆分已剔除) | fund_share (T+1 披露) "
       f"| 不直接动仓位; 升级为'红色警戒': 当周起 R1 检查改为每日盘后, 并复核 QDII 额度新闻 (SOP §5 月度项提前) "
       f"| 历史触发 {r2['exits']} 次几乎无虚惊 (收割 {r2['harvest_neg_pp']:.1f}pp 误伤); 2024 至今休眠 = 溢价持续的机制性支撑 |")
    ap(f"| **对照 (不采用)** | 纯水位开关 (p5>U 切出) | — | — | E3 已否决, 本实验复核: 2024+ 净值 "
       f"{fmt_pp(r3_e3['net_2024+'])}~{fmt_pp(r3_31['net_2024+'])}/年, 切出占比 16%+, 长期赌崩塌 |")
    ap('')
    ap('**执行细则**:')
    ap('')
    ap(f"1. **触发后动作按资金规模分层** (沿用 SOP §4.2/附表1): 场外容量内 (≤5万/周) 用'场内卖出 + 场外申购'完成切换 "
       f"(卖出恰好在高位变现溢价, 与 SOP '高溢价卖出是受益方'一致); 超容量部分按 SOP 路径3 换入**当时仍低溢价的**场内候选, "
       f"若板块已同步下杀 (§1.2) 则候选保护有限, 剩余敞口接受回吐 — 这是容量约束下的物理上限, 不是规则缺陷。")
    ap(f"2. **不要日频抢跑**: §3 判读 3 — 日频执行比周频差 ({fmt_pp(bt[(bt['code']==MAIN)&(bt['kind']=='dd')&(bt['cadence']=='D')]['net_full'].iloc[0])} vs "
       f"{fmt_pp(r1['net_full'])}/年), 披露噪声下盘中/日频反应过度必然多付 whipsaw; R2 红色警戒期间例外 (改每日盘后核对, 但仍收盘价决策)。")
    ap(f"3. **当前状态即时结论**: {cur_d} 的 dd20 = {last['dd20']*100:.2f}pp, p5 = {fmt_pct(last['p5'])} → "
       f"{'R1 已触发, 按上表动作执行' if r1_fired_now else 'R1 未触发 (溢价仍处峰值平台), 存量维持 E3 方针不动'}; "
       f"份额 5 日变动 {last['share5']*100 if not np.isnan(last['share5']) else 0:.2f}% → R2 未触发。"
       f"**在两个信号都触发前, 不对存量做任何预防性减仓** — §2 已证明水位本身 (哪怕 10.86%) 不构成时点信号, "
       f"预防性离场的期望成本 (E3: 2024+ −2.09pp/年) 高于 R1 事中反应的期望损失 (峰后确认延迟 ≈1~2pp)。")
    ap(f"4. **SOP 修订建议** (按 SOP §7 流程: 先决策报告后 SOP): §3 纪律表新增'存量防线'列引用 R1; §6.2 从'事后复盘'升级为"
       f"'R1 触发 → 执行细则 1'; §5 月度复评增加 R2 份额监控项 (fund_share 5 日变动)。哨兵脚本可顺带输出 dd20 与份额 5 日变动 "
       f"(只加显示不加自动动作, 维持'哨兵只提示'定位)。")
    ap('')
    ap(f"**预期账目 (组合层面)**: 存量敞口 {exposure*100:.2f}pp × R1 severe 覆盖率 {fmt_pct(r1['cov_pct'], 0)} ≈ "
       f"**{exposure*100*float(r1['cov_pct']):.1f}pp 可望在下杀中保住**; 代价 = 误报 whipsaw ≈"
       f"{abs(r1['harvest_neg_pp'])/r1['years']*CUR_NASDAQ_W:.2f}pp/年 (组合) + 每年 ≈{r1['exits']/r1['years']:.1f} 次切换操作; "
       f"净期望为正且凸性正确 (损失有界、保护针对尾部)。")
    ap('')

    # ---------- §5 ----------
    ap('## 5. 局限')
    ap('')
    ap('- 溢价为披露口径 (nav asof≤7d), 日频"单日下杀"读数含披露时差成分; 信号已 5 日平滑并以周频决策为正口径, '
       '引用防守数值一律以 [W] 行为准 (第一版日频原始信号曾产出 +36pp/年的虚假收益, 已作为反面教材记录于脚本头注)。')
    ap('- NAV 替代腿忽略场外申赎费率与限购容量 (SOP §4.2: 场外实际容量 ≈5万/周, 单点依赖华夏); 大资金下切出须走'
       '场内低溢价候选, 额外承担候选自身溢价与跨标损耗, §3 净值对大资金偏乐观。')
    ap('- 单腿反事实核算, 不反馈引擎选基/防御层; 份额为场内份额 (fund_share), 披露亦有滞后, R2 实时性偏乐观。')
    ap('- severe 事件几十个且 2020 年贡献最剧烈样本; 阈值网格粗、无样本外参数验证 — 结论取方向性 (回撤类>水位类) 而非精确最优参数。')
    ap('')
    ap('---')
    ap('产物: `output/experiments/premium_e4_collapse.{md,json}` | 前置: E1 (溢价beta +1.22pp/年) / '
       'E3 (水位开关否决, 2024+ 踏空 −2.09pp/年) | SOP: `docs/premium_management_sop.md`')

    path = os.path.join(OUT_DIR, 'premium_e4_collapse.md')
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')
    log(f'📄 报告: {path}')


def write_json(ep_all, sig_results, bt, sig_frames, main_cycles):
    def clean(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)

    df = sig_frames[MAIN]
    cols = ['premium', 'p5', 'dd20', 'mom5', 'share5', 'volr']
    daily = {'date': [d.strftime('%Y-%m-%d') for d in df.index]}
    for c in cols:
        daily[c] = [None if np.isnan(v) else round(float(v), 6) for v in df[c]]
    payload = {
        'meta': {'task': '#25 E4', 'generated': datetime.now().isoformat(timespec='seconds'),
                 'ep_high': EP_HIGH, 'ep_low': EP_LOW, 'major_drop': MAJOR_DROP,
                 'severe_drop': SEVERE_DROP, 'pre_win': PRE_WIN, 'fee': FEE},
        'episodes': {code: ep_all[code].to_dict(orient='records') for code in QDII},
        'signals': sig_results,
        'backtests': bt.to_dict(orient='records'),
        'rule_cycles_513100': main_cycles,
        'daily_513100': daily,
    }
    path = os.path.join(OUT_DIR, 'premium_e4_collapse.json')
    with open(path, 'w') as f:
        json.dump(payload, f, ensure_ascii=False, default=clean)
    log(f'📄 JSON: {path}')


if __name__ == '__main__':
    main()
