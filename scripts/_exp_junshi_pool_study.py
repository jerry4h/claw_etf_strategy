#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""军工 ETF 进池可行性: E0 数据体检 + 分散性诊断 (三门禁, **本轮不跑 E2 回测**)。

背景: 用户提出军工是否是合适的标的扩容, 依据三个判断 —— (1) 军工有自己的逻辑
(国防预算/订单周期/事件驱动); (2) 相关性可能比较低; (3) 因此可能适合扩容。
本脚本用真实数据逐条检验, 并额外检验一个用户未提到的机制风险: 生产 score 是
`mom_w x mom6 - vol_w x tapered_vol14` (v4.6 两者均为 1.1), 军工波动远高于中证500,
vol 项会直接惩罚它 —— 若历史上几乎选不中, 那么 E2 只会得到 no-op(未测出结论),
跑回测是浪费。故在 E2 之前先设一道**可选中率**前置筛。

三道门禁 (先定后测, 写死为下方常量, 不因结果调整):
  G1 低相关   与中证500 全样本周收益相关 < 0.70 **且** 条件段(高波/下行)相关 < 0.80
  G2 可选中率 用生产 score 回放, 军工进 TOP2 的周占比 >= 10% (分母为军工有效周)
  G3 独立信息 军工周收益对现有 5 腿多元回归的 idiosyncratic share (1-R^2) >= 30%
判定: 三项全过 -> 建议进 E2; G1 或 G2 不过 -> NO-GO(不浪费 E2); 其余 -> 条件性。

双口径 (ETF 最早 2016-08 上市, 只有指数能覆盖策略 2013-05 起的全样本):
  主口径  中证军工指数 399967.SZ (ETF 基准 "中证军工指数x100%"), 2013-01 起
  对照    国证军工 399368.SZ (更长历史, 口径稳健性)
  可交易  512660.SH 国泰中证军工ETF (2016-08-08, 军工 ETF 中最早且流动性最好),
          对照 512810.SH 华宝中证军工ETF (2016-08-22)
指数口径回答"军工这类资产是否与现有池低相关", ETF 口径回答"这个结论在可交易
标的上是否还成立(跟踪误差/溢价/流动性)"。两者都算, 报告并列, 不取有利的一个。

周频对齐: 基准 data/all_etfs_nav_latest.csv **本身已是周频**(682 周, 周五收盘),
src.data_loader.resample_weekly 对其为 no-op(中位间隔 7 天直接返回)。因此军工日线
按该 682 个日期做 asof(ffill) 对齐 —— 与生产完全同口径, 并统计对齐陈旧度。

硬约束: src/ config/ tests/ 零改动; 禁前视(条件段用 expanding 分位并 shift(1));
       涉及量一律用 amount(千元) 不用 vol(手); 原始抓取写 data/experiments/raw_junshi_*
       (命中 .gitignore 的 data/experiments/raw_*.csv, 不入库)。

用法: .venv/bin/python scripts/_exp_junshi_pool_study.py [--fetch] [--render-only]
输出: output/experiments/exp_junshi_pool.{md,json} + junshi_pool.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.data_loader import load_nav_data, resample_weekly  # noqa: E402
from src.factors import calculate_momentum, calculate_volatility_tapered  # noqa: E402
from src.strategy import load_config  # noqa: E402

# ============================================================
# 门禁常量 (先定后测 —— 看到结果后不得修改)
# ============================================================
G1_CORR_MAX_FULL = 0.70      # 与中证500 全样本周收益相关上限
G1_CORR_MAX_COND = 0.80      # 条件段(高波/下行)相关上限
G2_TOP2_MIN_PCT = 10.0       # 军工进 TOP2 的周占比下限(%)
G3_IDIO_MIN_PCT = 30.0       # 对 5 腿回归的 idiosyncratic share 下限(%)
E0_IDX_ETF_CORR_MIN = 0.95   # 指数与 ETF 重叠期周收益相关下限(低于则 ETF 跟踪有问题)

COND_VOL_PCT = 0.75          # 高波段: 中证500 tapered_vol14 的 expanding 分位阈
COND_DOWN_WEEKS = 8          # 下行段: 中证500 滚动 N 周累计收益 < 0
ROLL_CORR_W = 26             # 滚动相关窗口(周), 与 crisis_correlation.window 一致
MIN_COND_WEEKS = 30          # 条件段样本下限, 不足则该段结论标为 insufficient
EVENT_TOP_PCT = 5.0          # 事件形态: 超额收益前 N% 周的贡献占比

# 标的
IDX_PRIMARY = '399967.SZ'    # 中证军工 (ETF 基准)
IDX_ALT = '399368.SZ'        # 国证军工 (更长历史对照)
ETF_PRIMARY = '512660.SH'    # 国泰中证军工ETF (最早上市 + 流动性最好)
ETF_ALT = '512810.SH'        # 华宝中证军工ETF
JS = '军工'                   # 军工列名(诊断内部使用)
CSI500 = '中证500ETF'
CONFIG_PATH = 'config/strategy_v4_6.yaml'

DATA_EXP = PROJ / 'data' / 'experiments'
OUT_DIR = PROJ / 'output' / 'experiments'
MD_PATH = OUT_DIR / 'exp_junshi_pool.md'
JSON_PATH = OUT_DIR / 'exp_junshi_pool.json'
PNG_PATH = OUT_DIR / 'junshi_pool.png'

_API = {'pro': None}


def _log(msg: str) -> None:
    print(msg, flush=True)


# ============================================================
# E0-a/b: 取数与缓存
# ============================================================
def _pro():
    if _API['pro'] is None:
        import tushare as ts
        tok = os.environ.get('TUSHARE_TOKEN', '')
        if not tok:
            env = PROJ / '.env'
            if env.exists():
                for line in env.read_text().splitlines():
                    if line.strip().startswith('TUSHARE_TOKEN'):
                        tok = line.split('=', 1)[1].strip().strip('"').strip("'")
                        break
        if not tok:
            raise RuntimeError('未在环境变量或 .env 中找到 TUSHARE_TOKEN')
        ts.set_token(tok)
        _API['pro'] = ts.pro_api()
        _API['ts'] = ts
    return _API['pro']


def _cache_csv(name: str, fn, force: bool = False) -> pd.DataFrame:
    """带缓存的取数: data/experiments/raw_junshi_{name}.csv"""
    path = DATA_EXP / f'raw_junshi_{name}.csv'
    if path.exists() and not force:
        return pd.read_csv(path, dtype={'trade_date': str})
    DATA_EXP.mkdir(parents=True, exist_ok=True)
    df = fn()
    df.to_csv(path, index=False)
    _log(f'  [fetch] {path.name}  rows={len(df)}')
    return df


def fetch_index(code: str, force: bool = False) -> pd.DataFrame:
    def _f():
        pro = _pro()
        for attempt in range(3):
            try:
                d = pro.index_daily(ts_code=code, start_date='20130101', end_date='20261231')
                if d is not None and len(d):
                    return d.sort_values('trade_date').reset_index(drop=True)
            except Exception as e:  # noqa: BLE001
                _log(f'  [retry {attempt+1}] index_daily {code}: {repr(e)[:90]}')
                time.sleep(2)
        raise RuntimeError(f'index_daily {code} 取数失败')
    return _cache_csv(f'idx_{code.split(".")[0]}', _f, force)


def fetch_etf(code: str, force: bool = False) -> pd.DataFrame:
    """ETF 前复权日线 (qfq, 与 sp500_swap 同口径)。amount 单位千元。"""
    def _f():
        _pro()
        ts = _API['ts']
        for attempt in range(3):
            try:
                b = ts.pro_bar(ts_code=code, asset='FD', adj='qfq',
                               start_date='20130101', end_date='20261231')
                if b is not None and len(b):
                    return b.sort_values('trade_date').reset_index(drop=True)
            except Exception as e:  # noqa: BLE001
                _log(f'  [retry {attempt+1}] pro_bar {code}: {repr(e)[:90]}')
                time.sleep(2)
        raise RuntimeError(f'pro_bar {code} 取数失败')
    return _cache_csv(f'etf_{code.split(".")[0]}', _f, force)


def fetch_candidates(force: bool = False) -> pd.DataFrame:
    """军工/国防 场内 ETF 候选表 + 近一年日均 amount(千元)。

    只保留真正的 ETF: 名称含 'ETF' 且排除分级(A/B)与 LOF —— 分级基金 2020 年底已
    全部转型/清盘, LOF 场内流动性不足, 都不是可用标的。
    """
    def _f():
        pro = _pro()
        fb = pro.fund_basic(market='E')
        m = fb[fb['name'].str.contains('军工|国防', na=False)].copy()
        m = m[m['name'].str.contains('ETF', na=False)]
        m = m[~m['name'].str.contains('分级|LOF', na=False)]
        rows = []
        for _, r in m.iterrows():
            code = r['ts_code']
            avg_amt, n_days = np.nan, 0
            try:
                d = pro.fund_daily(ts_code=code, start_date='20250829', end_date='20260828')
                if d is not None and len(d):
                    avg_amt = float(d['amount'].mean())
                    n_days = int(len(d))
            except Exception as e:  # noqa: BLE001
                _log(f'  [warn] fund_daily {code}: {repr(e)[:80]}')
            rows.append({'ts_code': code, 'name': r['name'], 'list_date': r['list_date'],
                         'benchmark': r.get('benchmark', ''),
                         'avg_amount_1y_kcny': avg_amt, 'n_days_1y': n_days})
            time.sleep(0.35)
        return pd.DataFrame(rows).sort_values('list_date').reset_index(drop=True)
    return _cache_csv('candidates', _f, force)


# ============================================================
# 周频对齐
# ============================================================
def _daily_series(df: pd.DataFrame, col: str = 'close') -> pd.Series:
    s = df.copy()
    s['dt'] = pd.to_datetime(s['trade_date'].astype(str), format='%Y%m%d')
    return s.set_index('dt')[col].astype(float).sort_index()


def align_weekly(daily: pd.Series, base_index: pd.DatetimeIndex) -> tuple[pd.Series, dict]:
    """按基准周频日期做 asof(ffill) 对齐, 并统计陈旧度。

    生产基准文件本身是周频(周五收盘), A 股指数/ETF 与之同交易日历, 正常应精确命中;
    遇节假日基准取周四时会 ffill 前一日。陈旧度 > 7 天说明该周军工无成交(或未上市)。
    """
    aligned = daily.reindex(base_index, method='ffill')
    # 陈旧度: 每个基准日期对应的实际数据日期与基准日期的间隔
    idx_pos = daily.index.searchsorted(base_index, side='right') - 1
    stale = []
    for i, p in enumerate(idx_pos):
        if p < 0:
            stale.append(np.nan)
        else:
            stale.append((base_index[i] - daily.index[p]).days)
    stale = pd.Series(stale, index=base_index)
    valid = aligned.notna()
    diag = {
        'n_base_weeks': int(len(base_index)),
        'n_valid_weeks': int(valid.sum()),
        'first_valid': str(aligned[valid].index[0].date()) if valid.any() else None,
        'coverage_pct': round(100.0 * valid.sum() / len(base_index), 2),
        'exact_hit_pct': round(100.0 * float((stale[valid] == 0).sum()) / max(1, int(valid.sum())), 2),
        'stale_max_days': int(stale[valid].max()) if valid.any() else None,
        'stale_gt7_weeks': int((stale[valid] > 7).sum()) if valid.any() else 0,
    }
    return aligned, diag


# ============================================================
# E0-c: 数据质量体检
# ============================================================
def e0_health(name: str, daily: pd.Series, amount: pd.Series | None = None) -> dict:
    """日频质量体检: 覆盖、停滞、整数倍跳变(拆分/份额变更痕迹)、amount 可得性。"""
    r = daily.pct_change()
    ratio = (daily / daily.shift(1)).dropna()
    # 精确整数倍/整数分之一跳变 —— 拆分或复权口径断裂的特征
    jumps = []
    for k in (2, 3, 4, 5, 10):
        for v, tag in ((k, f'x{k}'), (1.0 / k, f'/{k}')):
            hit = ratio[(ratio - v).abs() < 0.01 * v]
            for dt, rv in hit.items():
                jumps.append({'date': str(dt.date()), 'ratio': round(float(rv), 4), 'tag': tag})
    out = {
        'name': name,
        'n_days': int(len(daily)),
        'first': str(daily.index[0].date()),
        'last': str(daily.index[-1].date()),
        'nan_pct': round(100.0 * float(daily.isna().sum()) / max(1, len(daily)), 3),
        'flat_pct': round(100.0 * float((r.abs() < 1e-9).sum()) / max(1, len(r)), 2),
        'ratio_jumps': jumps[:10],
        'n_ratio_jumps': len(jumps),
        'ann_vol_pct': round(float(r.std(ddof=0) * np.sqrt(252) * 100), 2),
    }
    if amount is not None:
        a = amount.dropna()
        out['amount_zero_pct'] = round(100.0 * float((a <= 0).sum()) / max(1, len(a)), 2)
        out['amount_median_kcny'] = round(float(a.median()), 1)
        out['amount_last20_median_kcny'] = round(float(a.tail(20).median()), 1)
    return out


# ============================================================
# G1: 相关性诊断
# ============================================================
def _corr(a: pd.Series, b: pd.Series, mask: pd.Series | None = None) -> tuple[float | None, int]:
    df = pd.concat([a, b], axis=1).dropna()
    if mask is not None:
        df = df[mask.reindex(df.index).fillna(False)]
    if len(df) < 10:
        return None, int(len(df))
    return round(float(df.iloc[:, 0].corr(df.iloc[:, 1])), 4), int(len(df))


def build_conditions(wk_nav: pd.DataFrame, cfg) -> dict:
    """预注册条件段 (禁前视: expanding 分位 + shift(1))。

    高波段: 中证500 tapered_vol14 的 expanding 分位 >= COND_VOL_PCT
    下行段: 中证500 滚动 COND_DOWN_WEEKS 周累计收益 < 0
    两者都在**上周末**即可知, 故 shift(1) 后作为本周的段标记。
    """
    vol = calculate_volatility_tapered(wk_nav[[CSI500]], window=cfg.vol_taper_window,
                                       taper=cfg.vol_taper_len)[CSI500]
    pct = vol.expanding(min_periods=52).apply(lambda x: (x.iloc[-1] >= x).mean(), raw=False)
    hi_vol = (pct >= COND_VOL_PCT).shift(1).fillna(False)
    rets = wk_nav[CSI500].pct_change()
    cum = (1 + rets).rolling(COND_DOWN_WEEKS).apply(np.prod, raw=True) - 1
    down = (cum < 0).shift(1).fillna(False)
    return {'high_vol': hi_vol.astype(bool), 'downside': down.astype(bool),
            'csi500_vol': vol, 'csi500_vol_pct': pct}


def g1_correlation(wk_rets: pd.DataFrame, js_col: str, conds: dict) -> dict:
    """军工 vs 现有 5 腿的相关性: 全样本 / 分年 / 滚动 / 条件段。"""
    legs = [c for c in wk_rets.columns if c != js_col]
    js = wk_rets[js_col]
    full = {}
    for leg in legs:
        c, n = _corr(js, wk_rets[leg])
        full[leg] = {'corr': c, 'n': n}
    by_year = {}
    for y, g in wk_rets.groupby(wk_rets.index.year):
        if g[js_col].notna().sum() < 20:
            continue
        c, n = _corr(g[js_col], g[CSI500])
        by_year[int(y)] = {'corr_csi500': c, 'n': n}
    roll = pd.concat([js, wk_rets[CSI500]], axis=1).dropna()
    roll_corr = (roll.iloc[:, 0].rolling(ROLL_CORR_W).corr(roll.iloc[:, 1])
                 if len(roll) > ROLL_CORR_W else pd.Series(dtype=float))
    cond = {}
    for tag, mask in (('high_vol', conds['high_vol']), ('downside', conds['downside'])):
        c, n = _corr(js, wk_rets[CSI500], mask)
        cond[tag] = {'corr_csi500': c, 'n': n,
                     'insufficient': bool(n < MIN_COND_WEEKS)}
        # 同时给出同段内 5 腿全表, 便于看是不是所有 A 股腿一起抬升
        cond[tag]['all_legs'] = {leg: _corr(js, wk_rets[leg], mask)[0] for leg in legs}
    calm = ~(conds['high_vol'] | conds['downside'])
    c, n = _corr(js, wk_rets[CSI500], calm)
    cond['calm'] = {'corr_csi500': c, 'n': n, 'insufficient': bool(n < MIN_COND_WEEKS)}
    cond_vals = [cond[t]['corr_csi500'] for t in ('high_vol', 'downside')
                 if cond[t]['corr_csi500'] is not None and not cond[t]['insufficient']]
    return {
        'full': full,
        'corr_csi500_full': full.get(CSI500, {}).get('corr'),
        'by_year': by_year,
        'rolling': {'window': ROLL_CORR_W,
                    'median': round(float(roll_corr.median()), 4) if len(roll_corr.dropna()) else None,
                    'min': round(float(roll_corr.min()), 4) if len(roll_corr.dropna()) else None,
                    'max': round(float(roll_corr.max()), 4) if len(roll_corr.dropna()) else None,
                    'series': {str(k.date()): (None if pd.isna(v) else round(float(v), 4))
                               for k, v in roll_corr.items()}},
        'conditional': cond,
        'cond_max': (round(max(cond_vals), 4) if cond_vals else None),
    }


# ============================================================
# G2: 可选中率回放
# ============================================================
def g2_selectability(wk_nav: pd.DataFrame, js_col: str, cfg) -> dict:
    """用生产 score 公式回放, 统计军工能否进 TOP2。

    score = mom_w x mom(mom_window) - vol_w x tapered_vol(vol_taper_window, vol_taper_len)
    只做横截面排序(不含 score_margin 滞回/防御层) —— 判断的是"有没有资格被选中"这个
    上游问题; margin 只影响换手时机, 不改变资格。分母用**军工有效周**(score 非 NaN),
    这是诚实的可选中率; 同时给出全样本占比作参照。
    """
    off = [c for c in wk_nav.columns if c not in ('红利低波ETF', '国债ETF')]
    mom = calculate_momentum(wk_nav, window=cfg.mom_window)
    vol = calculate_volatility_tapered(wk_nav, window=cfg.vol_taper_window,
                                       taper=cfg.vol_taper_len)
    score = cfg.mom_w * mom - cfg.vol_w * vol
    sc_off = score[off]
    valid = sc_off[js_col].notna()
    # 新增口径: 4 条进攻腿(纳指/中证500/黄金/军工) 里取 TOP2
    picked, ranks = [], []
    for dt, row in sc_off.iterrows():
        r = row.dropna()
        if len(r) < 2 or js_col not in r.index:
            picked.append(False)
            ranks.append(np.nan)
            continue
        order = r.sort_values(ascending=False)
        picked.append(js_col in order.index[:cfg.top_n])
        ranks.append(float(list(order.index).index(js_col) + 1))
    picked = pd.Series(picked, index=sc_off.index)
    ranks = pd.Series(ranks, index=sc_off.index)
    n_valid = int(valid.sum())
    # 替换口径: 军工 vs 中证500 谁的 score 更高
    both = sc_off[[js_col, CSI500]].dropna()
    js_beats = int((both[js_col] > both[CSI500]).sum())
    by_year = {}
    for y, g in picked[valid].groupby(picked[valid].index.year):
        by_year[int(y)] = {'n_weeks': int(len(g)), 'n_picked': int(g.sum()),
                           'pct': round(100.0 * float(g.sum()) / max(1, len(g)), 1)}
    avg_vol = {c: round(float(vol[c].mean() * 100), 2) for c in off}
    # 可选中周的年份集中度 (沿用稀疏脉冲课题的教训: 集中在少数年份
    # 的"有效"实际上是单年实验, 不能当持续有效)
    pick_years = {int(y): int(v['n_picked']) for y, v in by_year.items() if v['n_picked'] > 0}
    n_pick_tot = max(1, sum(pick_years.values()))
    concentration = {
        'n_years_with_pick': len(pick_years),
        'n_years_total': len(by_year),
        'max_year_share_pct': (round(100.0 * max(pick_years.values()) / n_pick_tot, 1)
                               if pick_years else 0.0),
        'top3_years_share_pct': (round(100.0 * sum(sorted(pick_years.values())[-3:]) / n_pick_tot, 1)
                                 if pick_years else 0.0),
        'zero_pick_years': sorted(int(y) for y, v in by_year.items() if v['n_picked'] == 0),
        'year_pct_min': (round(min(v['pct'] for v in by_year.values()), 1) if by_year else None),
        'year_pct_max': (round(max(v['pct'] for v in by_year.values()), 1) if by_year else None),
    }
    return {
        'params': {'mom_w': cfg.mom_w, 'vol_w': cfg.vol_w, 'mom_window': cfg.mom_window,
                   'vol_taper_window': cfg.vol_taper_window, 'vol_taper_len': cfg.vol_taper_len,
                   'top_n': cfg.top_n},
        'offensive_legs': off,
        'n_valid_weeks': n_valid,
        'n_all_weeks': int(len(sc_off)),
        'n_picked': int(picked.sum()),
        'pct_of_valid': round(100.0 * float(picked.sum()) / max(1, n_valid), 2),
        'pct_of_all': round(100.0 * float(picked.sum()) / max(1, len(sc_off)), 2),
        'mean_rank': round(float(ranks.dropna().mean()), 3) if ranks.notna().any() else None,
        'rank_dist': {int(k): int(v) for k, v in ranks.dropna().value_counts().sort_index().items()},
        'by_year': by_year,
        'swap_js_beats_csi500': {'n': js_beats, 'n_both': int(len(both)),
                                 'pct': round(100.0 * js_beats / max(1, len(both)), 2)},
        'avg_tapered_vol_pct': avg_vol,
        'year_concentration': concentration,
        'picked_series': picked,
    }


# ============================================================
# G3: 独立信息 + 分散贡献
# ============================================================
def g3_independence(wk_rets: pd.DataFrame, js_col: str, cfg) -> dict:
    """军工周收益对现有 5 腿的多元回归 + 加入军工后的组合波动变化。"""
    legs = [c for c in wk_rets.columns if c != js_col]
    df = wk_rets[[js_col] + legs].dropna()
    y = df[js_col].values
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in legs])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
    # 组合波动(等权 / inv-vol 两种口径, 5 腿 vs 6 腿):
    # 同期对比 —— 限定在军工有效期内, 否则 5 腿用了更长样本, 两者不可比
    span = wk_rets[[js_col]].dropna().index
    sub_all = wk_rets.loc[span]
    vols = {}
    for mode in ('eq', 'invvol'):
        v5 = _port_vol_on(sub_all, legs, mode, cfg)
        v6 = _port_vol_on(sub_all, legs + [js_col], mode, cfg)
        vols[mode] = {'vol5_pct': round(v5, 3), 'vol6_pct': round(v6, 3),
                      'delta_pp': round(v6 - v5, 3)}
    return {
        'n_obs': int(len(df)),
        'r2': round(float(r2), 4),
        'idio_share_pct': round(100.0 * (1.0 - float(r2)), 2),
        'betas': {c: round(float(b), 4) for c, b in zip(legs, beta[1:])},
        'alpha_weekly_pct': round(float(beta[0]) * 100, 4),
        'resid_ann_vol_pct': round(float(resid.std(ddof=0) * np.sqrt(52) * 100), 2),
        'portfolio_vol': vols,
        'note_same_span': f'组合波动对比限定在军工有效期 {span[0].date()} ~ {span[-1].date()}',
    }


def _port_vol_on(sub: pd.DataFrame, cols: list[str], mode: str, cfg) -> float:
    s = sub[cols].dropna()
    if mode == 'eq':
        p = s.mean(axis=1)
    else:
        v = s.rolling(cfg.inv_vol_window).std(ddof=cfg.vol_ddof)
        w = (1.0 / v).replace([np.inf, -np.inf], np.nan)
        w = w.div(w.sum(axis=1), axis=0)
        p = (s * w.shift(1)).sum(axis=1, min_count=len(cols))
    return float(p.dropna().std(ddof=0) * np.sqrt(52) * 100)


# ============================================================
# 事件驱动形态诊断
# ============================================================
def event_shape(wk_rets: pd.DataFrame, js_col: str) -> dict:
    """军工"有自己的逻辑"是否表现为稀疏事件, 以及上下行 beta 是否对称。

    若超额收益高度集中在少数周, 则连续因子框架(每周排序)会稀释它 —— 该测的是
    事件触发式条件激活, 属另一个假说(份额课题的教训)。
    """
    df = wk_rets[[js_col, CSI500]].dropna()
    ex = df[js_col] - df[CSI500]
    n_top = max(1, int(np.ceil(len(ex) * EVENT_TOP_PCT / 100.0)))
    top_pos = ex.nlargest(n_top).sum()
    top_neg = ex.nsmallest(n_top).sum()
    total = ex.sum()
    up = df[df[CSI500] > 0]
    dn = df[df[CSI500] < 0]
    def _beta(g):
        if len(g) < 20:
            return None
        cov = np.cov(g[js_col].values, g[CSI500].values, ddof=0)
        return round(float(cov[0, 1] / cov[1, 1]), 3)
    return {
        'n_obs': int(len(ex)),
        'excess_ann_pct': round(float(((1 + ex).prod()) ** (52.0 / len(ex)) - 1) * 100, 2),
        'excess_weekly_std_pct': round(float(ex.std(ddof=0) * 100), 3),
        'excess_kurtosis': round(float(ex.kurtosis()), 2),
        'excess_skew': round(float(ex.skew()), 3),
        f'top{int(EVENT_TOP_PCT)}pct_weeks': n_top,
        'top_pos_share_of_total': (round(float(top_pos / total), 2) if abs(total) > 1e-9 else None),
        'top_pos_sum_pct': round(float(top_pos) * 100, 2),
        'top_neg_sum_pct': round(float(top_neg) * 100, 2),
        'beta_up': _beta(up),
        'beta_down': _beta(dn),
        'n_up': int(len(up)),
        'n_dn': int(len(dn)),
        'ann_vol_js_pct': round(float(df[js_col].std(ddof=0) * np.sqrt(52) * 100), 2),
        'ann_vol_csi500_pct': round(float(df[CSI500].std(ddof=0) * np.sqrt(52) * 100), 2),
        'maxdd_js_pct': round(_maxdd(df[js_col]) * 100, 2),
        'maxdd_csi500_pct': round(_maxdd(df[CSI500]) * 100, 2),
    }


def _maxdd(rets: pd.Series) -> float:
    nav = (1 + rets).cumprod()
    return float((1 - nav / nav.cummax()).max())


# ============================================================
# 裁决
# ============================================================
def recompute_verdict(res: dict) -> dict:
    """从结果重算三门禁 (纯函数, --render-only 可复算, 保证幂等)。"""
    g1 = res['g1_index']
    full = g1.get('corr_csi500_full')
    cond_max = g1.get('cond_max')
    g1_pass = (full is not None and full < G1_CORR_MAX_FULL
               and cond_max is not None and cond_max < G1_CORR_MAX_COND)
    g2 = res['g2_index']
    g2_pass = g2['pct_of_valid'] >= G2_TOP2_MIN_PCT
    g3 = res['g3_index']
    g3_pass = g3['idio_share_pct'] >= G3_IDIO_MIN_PCT
    gates = {
        'G1_low_corr': {'passed': bool(g1_pass),
                        'detail': (f"全样本 corr(中证500)={full} (<{G1_CORR_MAX_FULL}?), "
                                   f"条件段最大 corr={cond_max} (<{G1_CORR_MAX_COND}?)")},
        'G2_selectability': {'passed': bool(g2_pass),
                             'detail': (f"进 TOP2 占有效周 {g2['pct_of_valid']}% "
                                        f"(>={G2_TOP2_MIN_PCT}?), 平均排名 {g2['mean_rank']}")},
        'G3_independence': {'passed': bool(g3_pass),
                            'detail': (f"idio share {g3['idio_share_pct']}% "
                                       f"(>={G3_IDIO_MIN_PCT}?), R2={g3['r2']}")},
    }
    n_pass = sum(1 for v in gates.values() if v['passed'])
    if n_pass == 3:
        verdict = 'GO(建议进 E2 集成回测)'
        reason = '三门禁全过: 低相关成立、有足够可选中率、且带来独立信息'
    elif not g2_pass:
        verdict = 'NO-GO(不进 E2)'
        reason = ('可选中率门禁未过 —— 生产 score 的 vol 惩罚使军工极少进入 TOP2, '
                  '即使集成也是 no-op, 跑 E2 只会得到"未测出结论"')
    elif not g1_pass:
        verdict = 'NO-GO(不进 E2)'
        reason = ('低相关前提不成立 —— 军工与中证500 高度同向, 扩容等于放大 A 股集中度, '
                  '而非补充新的风险溢价')
    else:
        verdict = '条件性(G3 未过, 需重新定位假说)'
        reason = 'G1/G2 过但独立信息不足, 军工的"自有逻辑"未转化为组合层面的分散收益'
    res['gates'] = gates
    res['n_gates_passed'] = n_pass
    res['verdict'] = verdict
    res['verdict_reason'] = reason
    return res


# ============================================================
# 报告
# ============================================================
def render(res: dict) -> str:
    L: list[str] = []
    A = L.append
    A('# 军工 ETF 进池可行性: E0 数据体检 + 分散性诊断')
    A('')
    A(f"**裁决: {res['verdict']}**")
    A('')
    A(f"> {res['verdict_reason']}")
    A('')
    A(f"- 生成时间: {res['generated_at']}")
    A(f"- 数据截止: {res['data_end']}")
    A(f"- 门禁通过: **{res['n_gates_passed']}/3**")
    A(f"- 范围: E0 + 相关性/可选中率/独立信息诊断, **不跑 E2 回测**; src/ config/ tests/ 零改动")
    A('')
    A('## 0. 三门禁结论 (先定后测)')
    A('')
    A('| 门禁 | 判据 | 实测 | 结论 |')
    A('|---|---|---|---|')
    crit = {'G1_low_corr': f'全样本<{G1_CORR_MAX_FULL} 且 条件段<{G1_CORR_MAX_COND}',
            'G2_selectability': f'TOP2 占比>={G2_TOP2_MIN_PCT}%',
            'G3_independence': f'idio share>={G3_IDIO_MIN_PCT}%'}
    for k, v in res['gates'].items():
        A(f"| {k} | {crit[k]} | {v['detail']} | {'✓ PASS' if v['passed'] else '✗ FAIL'} |")
    A('')
    A('## 1. 标的选择 (E0-a)')
    A('')
    A('军工/国防场内 ETF 全量候选 (已剔除分级基金与 LOF):')
    A('')
    A('| 代码 | 名称 | 上市日 | 近一年日均成交额(万元) | 基准 |')
    A('|---|---|---|---|---|')
    for c in res['candidates']:
        amt = c['avg_amount_1y_kcny']
        amt_s = f"{amt/10:,.0f}" if amt and not pd.isna(amt) else 'n/a'
        A(f"| {c['ts_code']} | {c['name']} | {c['list_date']} | {amt_s} | {c['benchmark']} |")
    A('')
    A(f"选定: **{ETF_PRIMARY}**(上市最早且流动性最好) 为可交易主标的, {ETF_ALT} 为对照。")
    A(f"指数口径 **{IDX_PRIMARY} 中证军工**(ETF 基准), 对照 {IDX_ALT} 国证军工。")
    A('')
    A('> ETF 最早 2016-08 上市, 而策略样本自 2013-05 起。**只有指数能覆盖全样本**,')
    A('> 故主口径用指数做相关性/独立性诊断, ETF 口径用于验证结论在可交易标的上是否成立。')
    A('')
    A('## 2. 数据质量体检 (E0-b/c)')
    A('')
    A('| 序列 | 交易日 | 起 | 止 | NaN% | 停滞% | 整数倍跳变 | 年化波动% |')
    A('|---|---|---|---|---|---|---|---|')
    for h in res['e0_health']:
        A(f"| {h['name']} | {h['n_days']} | {h['first']} | {h['last']} | {h['nan_pct']} | "
          f"{h['flat_pct']} | {h['n_ratio_jumps']} | {h['ann_vol_pct']} |")
    A('')
    for h in res['e0_health']:
        if 'amount_median_kcny' in h:
            A(f"- {h['name']} amount: 中位 {h['amount_median_kcny']/10:,.0f} 万元, "
              f"近 20 日中位 {h['amount_last20_median_kcny']/10:,.0f} 万元, "
              f"零成交日占比 {h['amount_zero_pct']}%")
    A('')
    A('周频对齐诊断 (基准 682 周, 周五收盘; 生产 `resample_weekly` 对周频数据为 no-op):')
    A('')
    A('| 序列 | 有效周 | 覆盖% | 首个有效周 | 精确命中% | 最大陈旧(天) | 陈旧>7天周数 |')
    A('|---|---|---|---|---|---|---|')
    for k, d in res['align'].items():
        A(f"| {k} | {d['n_valid_weeks']} | {d['coverage_pct']} | {d['first_valid']} | "
          f"{d['exact_hit_pct']} | {d['stale_max_days']} | {d['stale_gt7_weeks']} |")
    A('')
    ie = res['idx_etf_consistency']
    A(f"指数 vs ETF 重叠期一致性: 周收益相关 **{ie['corr']}** (n={ie['n']}, "
      f"门禁 >={E0_IDX_ETF_CORR_MIN}) → {'✓ 一致' if ie['passed'] else '✗ 跟踪异常'}; "
      f"年化跟踪差 {ie['tracking_diff_ann_pct']}pp")
    A('')
    A('## 3. G1 相关性诊断 (核心: 你的"相关性比较低"是否成立)')
    A('')
    for tag, key in (('指数口径(2013 起, 长样本)', 'g1_index'), ('ETF 口径(2016-08 起, 可交易)', 'g1_etf')):
        g = res[key]
        A(f"### {tag}")
        A('')
        A('| 对手腿 | 全样本相关 | n |')
        A('|---|---|---|')
        for leg, v in g['full'].items():
            A(f"| {leg} | **{v['corr']}** | {v['n']} |")
        A('')
        c = g['conditional']
        A('条件段相关性 (对中证500; 段定义用 expanding 分位并 shift(1), 无前视):')
        A('')
        A('| 段 | corr(中证500) | 周数 | 样本充足 |')
        A('|---|---|---|---|')
        for seg, label in (('calm', '平静段'), ('high_vol', '高波段(vol expanding 分位≥75%)'),
                           ('downside', f'下行段(中证500 滚动{COND_DOWN_WEEKS}周<0)')):
            v = c[seg]
            A(f"| {label} | **{v['corr_csi500']}** | {v['n']} | "
              f"{'否' if v.get('insufficient') else '是'} |")
        A('')
        A(f"滚动 {g['rolling']['window']} 周相关: 中位 {g['rolling']['median']}, "
          f"区间 [{g['rolling']['min']}, {g['rolling']['max']}]")
        A('')
        if g['by_year']:
            A('分年相关(对中证500): ' + ', '.join(
                f"{y}:{v['corr_csi500']}" for y, v in sorted(g['by_year'].items())))
            A('')
    A('## 4. G2 可选中率回放 (机制风险: vol 惩罚)')
    A('')
    g2 = res['g2_index']
    p = g2['params']
    A(f"生产参数(读自 `{CONFIG_PATH}`): score = {p['mom_w']}×mom{p['mom_window']} "
      f"− {p['vol_w']}×tapered_vol{p['vol_taper_window']}(taper={p['vol_taper_len']}), "
      f"进攻腿 {g2['offensive_legs']} 取 TOP{p['top_n']}")
    A('')
    A('| 口径 | 有效周 | 进 TOP2 周数 | 占有效周% | 平均排名 |')
    A('|---|---|---|---|---|')
    for tag, key in (('指数(长样本)', 'g2_index'), ('ETF(可交易)', 'g2_etf')):
        g = res[key]
        A(f"| {tag} | {g['n_valid_weeks']} | {g['n_picked']} | **{g['pct_of_valid']}** | "
          f"{g['mean_rank']} |")
    A('')
    A('各进攻腿平均 tapered vol (年化%, 解释为何被惩罚):')
    A('')
    A('| 腿 | 平均 tapered vol% |')
    A('|---|---|')
    for k, v in sorted(g2['avg_tapered_vol_pct'].items(), key=lambda x: -x[1]):
        A(f"| {k} | {v} |")
    A('')
    sw = g2['swap_js_beats_csi500']
    A(f"替换口径参照: 军工 score > 中证500 score 的周占比 **{sw['pct']}%** "
      f"({sw['n']}/{sw['n_both']})")
    A('')
    yc = g2['year_concentration']
    A(f"可选中周的年份分布: 跨 {yc['n_years_with_pick']}/{yc['n_years_total']} 个年份, "
      f"单年最高占全部选中的 {yc['max_year_share_pct']}%, 最高三年合计 "
      f"{yc['top3_years_share_pct']}%; 分年占比跨度 {yc['year_pct_min']}%~{yc['year_pct_max']}%, "
      f"零选中年份 {yc['zero_pick_years'] or '无'}")
    A('')
    if g2['by_year']:
        A('分年进 TOP2 占比: ' + ', '.join(
            f"{y}:{v['pct']}%({v['n_picked']}/{v['n_weeks']})"
            for y, v in sorted(g2['by_year'].items())))
        A('')
    A('## 5. G3 独立信息与分散贡献')
    A('')
    A('| 口径 | n | R² | idio share% | 残差年化波动% | 周 alpha% |')
    A('|---|---|---|---|---|---|')
    for tag, key in (('指数', 'g3_index'), ('ETF', 'g3_etf')):
        g = res[key]
        A(f"| {tag} | {g['n_obs']} | {g['r2']} | **{g['idio_share_pct']}** | "
          f"{g['resid_ann_vol_pct']} | {g['alpha_weekly_pct']} |")
    A('')
    g3 = res['g3_index']
    A('对 5 腿回归系数(指数口径): ' + ', '.join(f"{k}={v}" for k, v in g3['betas'].items()))
    A('')
    A(f"组合波动变化({g3['note_same_span']}):")
    A('')
    A('| 权重口径 | 5 腿波动% | 加军工 6 腿波动% | Δ(pp) |')
    A('|---|---|---|---|')
    for mode, label in (('eq', '等权'), ('invvol', f'inv-vol{res["cfg_inv_vol_window"]}')):
        v = g3['portfolio_vol'][mode]
        A(f"| {label} | {v['vol5_pct']} | {v['vol6_pct']} | {v['delta_pp']:+.3f} |")
    A('')
    A('## 6. 事件驱动形态 (你的"军工有自己的逻辑")')
    A('')
    ev = res['event_index']
    A('| 指标 | 军工 | 中证500 |')
    A('|---|---|---|')
    A(f"| 年化波动% | {ev['ann_vol_js_pct']} | {ev['ann_vol_csi500_pct']} |")
    A(f"| 最大回撤% | {ev['maxdd_js_pct']} | {ev['maxdd_csi500_pct']} |")
    A('')
    A(f"- 相对中证500 超额: 年化 {ev['excess_ann_pct']}%, 周标准差 "
      f"{ev['excess_weekly_std_pct']}%, 峰度 {ev['excess_kurtosis']}, 偏度 {ev['excess_skew']}")
    A(f"- 超额最高 {EVENT_TOP_PCT}% 周({ev[f'top{int(EVENT_TOP_PCT)}pct_weeks']} 周)累计贡献 "
      f"{ev['top_pos_sum_pct']}pp; 最差同样周数 {ev['top_neg_sum_pct']}pp")
    A(f"- 上下行 beta 不对称: 中证500 涨时 beta={ev['beta_up']} (n={ev['n_up']}), "
      f"跌时 beta={ev['beta_down']} (n={ev['n_dn']})")
    A('')
    A('## 7. 结论: 逐条回答')
    A('')
    for q in res['answers']:
        A(f"**{q['q']}**")
        A('')
        A(q['a'])
        A('')
    A('## 8. 复现')
    A('')
    A('```bash')
    A('.venv/bin/python scripts/_exp_junshi_pool_study.py            # 全量(缓存命中则不重复取数)')
    A('.venv/bin/python scripts/_exp_junshi_pool_study.py --fetch    # 强制重新取数')
    A('.venv/bin/python scripts/_exp_junshi_pool_study.py --render-only  # 仅重渲染(幂等)')
    A('```')
    A('')
    A('原始数据缓存 `data/experiments/raw_junshi_*.csv` 命中 `.gitignore` 不入库, 由本脚本再生。')
    A('')
    return '\n'.join(L)


def plot(res: dict, wk: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ('Noto Sans CJK JP', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans'):
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams['font.sans-serif'] = [f]
            break
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    ax = axes[0][0]
    rc = res['g1_index']['rolling']['series']
    s = pd.Series({pd.Timestamp(k): v for k, v in rc.items()}).dropna()
    ax.plot(s.index, s.values, lw=1.2, color='#c0392b')
    ax.axhline(G1_CORR_MAX_FULL, ls='--', color='k', lw=1,
               label=f'G1 门禁 {G1_CORR_MAX_FULL}')
    ax.axhline(G1_CORR_MAX_COND, ls=':', color='gray', lw=1,
               label=f'条件段门禁 {G1_CORR_MAX_COND}')
    ax.set_title(f'滚动 {ROLL_CORR_W} 周相关: 军工指数 vs 中证500ETF')
    ax.set_ylabel('corr')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    full = res['g1_index']['full']
    ks = list(full.keys())
    vs = [full[k]['corr'] for k in ks]
    cols = ['#c0392b' if v is not None and v >= G1_CORR_MAX_FULL else '#27ae60' for v in vs]
    ax.barh(ks, vs, color=cols)
    ax.axvline(G1_CORR_MAX_FULL, ls='--', color='k', lw=1)
    ax.set_title('军工指数 vs 现有 5 腿: 全样本周收益相关')
    for i, v in enumerate(vs):
        if v is not None:
            ax.text(v, i, f' {v:.2f}', va='center', fontsize=9)
    ax.grid(alpha=0.3, axis='x')

    ax = axes[1][0]
    g2 = res['g2_index']
    legs = list(g2['avg_tapered_vol_pct'].keys())
    vals = [g2['avg_tapered_vol_pct'][k] for k in legs]
    order = np.argsort(vals)[::-1]
    ax.bar([legs[i] for i in order], [vals[i] for i in order],
           color=['#c0392b' if legs[i] == JS else '#7f8c8d' for i in order])
    ax.set_title('平均 tapered vol (年化%): 军工被 vol 项惩罚的幅度')
    ax.set_ylabel('vol %')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1][1]
    by = g2['by_year']
    if by:
        ys = sorted(by.keys())
        ax.bar([str(y) for y in ys], [by[y]['pct'] for y in ys], color='#2980b9')
        ax.axhline(G2_TOP2_MIN_PCT, ls='--', color='k', lw=1,
                   label=f'G2 门禁 {G2_TOP2_MIN_PCT}%')
        ax.legend(fontsize=8)
    ax.set_title('军工进 TOP2 的周占比 (分年, 指数口径)')
    ax.set_ylabel('%')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3, axis='y')

    fig.suptitle(f"军工 ETF 进池诊断 — {res['verdict']} ({res['n_gates_passed']}/3 门禁)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PNG_PATH, dpi=110)
    plt.close(fig)
    _log(f'  [plot] {PNG_PATH}')


# ============================================================
# main
# ============================================================
def build_answers(res: dict) -> list[dict]:
    g1 = res['g1_index']
    g1e = res['g1_etf']
    g2 = res['g2_index']
    g2e = res['g2_etf']
    g3 = res['g3_index']
    ev = res['event_index']
    c_full = g1['corr_csi500_full']
    c_hi = g1['conditional']['high_vol']['corr_csi500']
    c_dn = g1['conditional']['downside']['corr_csi500']
    c_calm = g1['conditional']['calm']['corr_csi500']
    a1 = (f"**不成立**。军工指数与中证500ETF 全样本周收益相关 **{c_full}**, "
          f"远高于门禁 {G1_CORR_MAX_FULL}。" if c_full is not None and c_full >= G1_CORR_MAX_FULL
          else f"**成立**。全样本相关 {c_full} < {G1_CORR_MAX_FULL}。")
    a1 += (f" 分段看: 平静段 {c_calm}、高波段 {c_hi}、下行段 {c_dn} —— "
           f"{'相关性在最需要分散的时候并未下降' if (c_hi or 0) >= (c_calm or 0) else '高波段相关性有所下降'}。"
           f" 相对纳指({g1['full'].get('纳指ETF', {}).get('corr')})/黄金"
           f"({g1['full'].get('黄金ETF', {}).get('corr')})/国债"
           f"({g1['full'].get('国债ETF', {}).get('corr')})确实低, 但池中中证500 已在承担 A 股 beta, "
           f"新增军工主要是放大 A 股集中度。ETF 口径复核: {g1e['corr_csi500_full']}。")
    a2 = (f"军工年化波动 {ev['ann_vol_js_pct']}% vs 中证500 {ev['ann_vol_csi500_pct']}%, "
          f"平均 tapered vol {g2['avg_tapered_vol_pct'].get(JS)}% 为各进攻腿最高。"
          f"生产 score 的 vol 项系数与 mom 同为 {g2['params']['vol_w']}, 结果军工进 TOP2 "
          f"占 **{g2['pct_of_valid']}%** 的有效周(平均排名 {g2['mean_rank']}/"
          f"{len(g2['offensive_legs'])}), ETF 口径 {g2e['pct_of_valid']}%。")
    yc = g2['year_concentration']
    if g2['pct_of_valid'] >= G2_TOP2_MIN_PCT:
        a2 += (f"这**过了** G2 门禁 —— 需要如实标注: 本课题预先担心的‘vol 惩罚使军工选不中、"
               f"E2 退化为 no-op’这个先验**被数据推翻了**。原因是军工的 mom6 在主题行情"
               f"里足以补偿 vol 惩罚。")
        spread = ((yc['year_pct_max'] or 0) - (yc['year_pct_min'] or 0))
        concentrated = (yc['top3_years_share_pct'] > 50.0
                        or yc['n_years_with_pick'] < 0.6 * max(1, yc['n_years_total']))
        if concentrated:
            a2 += (f"但选中周**高度集中**: 仅跨 {yc['n_years_with_pick']}/{yc['n_years_total']} 个年份, "
                   f"最高三年就占了 {yc['top3_years_share_pct']}%。")
        else:
            a2 += (f"选中周在时间上**并不集中**(跨 {yc['n_years_with_pick']}/{yc['n_years_total']} 个年份, "
                   f"单年最高仅占全部选中的 {yc['max_year_share_pct']}%, 最高三年合计 "
                   f"{yc['top3_years_share_pct']}%), 这一点优于预期。")
        a2 += (f"真正的问题在**年度间起伏极大**: 分年占比从 {yc['year_pct_min']}% 到 "
               f"{yc['year_pct_max']}%(跨度 {spread:.1f}pp), 零选中年份 {yc['zero_pick_years'] or '无'}。"
               f"这是**主题轮动型暴露**的特征 —— 若未来真要跑 E2, 必须把分年/分期一致性"
               f"当硬门禁, 否则全样本 ΔSharpe 只是几个主题年的副产品。")
    else:
        a2 += ('这低于 G2 门禁, 说明即使把军工放进池子, 它也几乎不会被选中 —— '
               'E2 回测会得到"未测出结论"而非"有效/无效"。')
    b = g3['betas']
    a3 = (f"军工周收益对 5 腿回归 R²={g3['r2']}, idiosyncratic share "
          f"**{g3['idio_share_pct']}%**(门禁 {G3_IDIO_MIN_PCT}%, "
          f"{'擦边通过' if g3['idio_share_pct'] < G3_IDIO_MIN_PCT + 5 else '通过'}), "
          f"残差年化波动 {g3['resid_ann_vol_pct']}%。"
          f"但回归系数揭示了军工的**本质**: 中证500 beta={b.get(CSI500)}(>1 的放大器)、"
          f"红利低波 beta={b.get('红利低波ETF')}(负号)、纳指 beta={b.get('纳指ETF')}。"
          f"即军工 ≈ 中证500 加杠杆 − 低波红利 —— 它是一个**风格倾斜(高 beta/成长对低波/价值)**, "
          f"不是一个新的资产类别。形态上: 峰度 {ev['excess_kurtosis']}, 最高 {EVENT_TOP_PCT}% 周"
          f"贡献 {ev['top_pos_sum_pct']}pp 而最差同样周数 {ev['top_neg_sum_pct']}pp(双向都极端, "
          f"不是单向 alpha 脉冲), 上行 beta {ev['beta_up']} / 下行 beta {ev['beta_down']}。"
          + ('下行 beta 不低于上行 beta —— 军工在中证500 下跌时跟跌不打折。'
             if (ev['beta_down'] or 0) >= (ev['beta_up'] or 0) else
             '下行 beta 低于上行 beta, 形态上偏有利。'))
    eq = g3['portfolio_vol']['eq']['delta_pp']
    iv = g3['portfolio_vol']['invvol']['delta_pp']
    a4 = (f"价格上的总账: 军工年化波动 {ev['ann_vol_js_pct']}% vs 中证500 "
          f"{ev['ann_vol_csi500_pct']}%、最大回撤 {ev['maxdd_js_pct']}% vs "
          f"{ev['maxdd_csi500_pct']}%, 而相对中证500 的年化超额是 "
          f"**{ev['excess_ann_pct']}%** —— 全样本下它是用更高波动与更深回撤换了一个"
          f"{'负' if (ev['excess_ann_pct'] or 0) < 0 else '正'}超额。"
          f"组合层面: 等权加入军工使年化波动 {eq:+.3f}pp, inv-vol 口径 {iv:+.3f}pp。"
          f"结合三门禁 {res['n_gates_passed']}/3, 裁决为 **{res['verdict']}**。{res['verdict_reason']}。")
    if res['n_gates_passed'] < 3:
        a4 += ('  \n注: 这是"在当前 score 与池子结构下不合适", 不等于"军工资产本身没有价值"。'
               '若要继续这个方向, 应换假说 —— 例如把军工作为事件触发式的条件激活腿'
               '(而非常设第 6 腿), 并先在 E1 上验证触发信号, 而不是直接放进横截面排序。')
    return [
        {'q': 'Q1 军工与现有池的相关性到底低不低?', 'a': a1},
        {'q': 'Q2 军工放进池子后能被选中吗(机制风险)?', 'a': a2},
        {'q': 'Q3 "军工有自己的逻辑"是否转化为组合层面的独立收益?', 'a': a3},
        {'q': 'Q4 那么军工适合作为标的扩容吗?', 'a': a4},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fetch', action='store_true', help='强制重新取数')
    ap.add_argument('--render-only', action='store_true', help='仅从 json 重渲染 md')
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.render_only:
        res = json.loads(JSON_PATH.read_text())
        res = recompute_verdict(res)
        res['answers'] = build_answers(res)
        MD_PATH.write_text(render(res))
        JSON_PATH.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str))
        _log(f'[render-only] {MD_PATH}')
        return

    cfg = load_config(str(PROJ / CONFIG_PATH))
    _log('[1/6] E0-a 标的选择')
    cand = fetch_candidates(args.fetch)
    _log('[2/6] E0-b 取数')
    idx_p = fetch_index(IDX_PRIMARY, args.fetch)
    idx_a = fetch_index(IDX_ALT, args.fetch)
    etf_p = fetch_etf(ETF_PRIMARY, args.fetch)
    etf_a = fetch_etf(ETF_ALT, args.fetch)

    base = load_nav_data(PROJ / 'data' / 'all_etfs_nav_latest.csv')
    base_wk = resample_weekly(base)
    bidx = base_wk.index

    series = {
        f'{IDX_PRIMARY} 中证军工': _daily_series(idx_p),
        f'{IDX_ALT} 国证军工': _daily_series(idx_a),
        f'{ETF_PRIMARY} 国泰军工ETF(qfq)': _daily_series(etf_p),
        f'{ETF_ALT} 华宝军工ETF(qfq)': _daily_series(etf_a),
    }
    amounts = {
        f'{ETF_PRIMARY} 国泰军工ETF(qfq)': _daily_series(etf_p, 'amount'),
        f'{ETF_ALT} 华宝军工ETF(qfq)': _daily_series(etf_a, 'amount'),
    }
    _log('[3/6] E0-c 质量体检 + 周频对齐')
    health = [e0_health(k, v, amounts.get(k)) for k, v in series.items()]
    align, wk_cols = {}, {}
    for k, v in series.items():
        aligned, diag = align_weekly(v, bidx)
        align[k] = diag
        wk_cols[k] = aligned

    idx_wk = wk_cols[f'{IDX_PRIMARY} 中证军工']
    etf_wk = wk_cols[f'{ETF_PRIMARY} 国泰军工ETF(qfq)']
    ov = pd.concat([idx_wk.pct_change(), etf_wk.pct_change()], axis=1).dropna()
    ic = round(float(ov.iloc[:, 0].corr(ov.iloc[:, 1])), 4) if len(ov) > 10 else None
    td = (float((1 + ov.iloc[:, 1]).prod() ** (52 / len(ov)) - 1)
          - float((1 + ov.iloc[:, 0]).prod() ** (52 / len(ov)) - 1)) * 100 if len(ov) > 10 else None

    nav_idx = base_wk.copy()
    nav_idx[JS] = idx_wk
    nav_etf = base_wk.copy()
    nav_etf[JS] = etf_wk
    rets_idx = nav_idx.pct_change()
    rets_etf = nav_etf.pct_change()

    _log('[4/6] G1 相关性')
    conds = build_conditions(base_wk, cfg)
    g1_idx = g1_correlation(rets_idx, JS, conds)
    g1_etf = g1_correlation(rets_etf, JS, conds)
    _log('[5/6] G2 可选中率 + G3 独立信息')
    g2_idx = g2_selectability(nav_idx, JS, cfg)
    g2_etf = g2_selectability(nav_etf, JS, cfg)
    g3_idx = g3_independence(rets_idx, JS, cfg)
    g3_etf = g3_independence(rets_etf, JS, cfg)
    ev_idx = event_shape(rets_idx, JS)
    _log('[6/6] 裁决 + 报告')

    for g in (g2_idx, g2_etf):
        g.pop('picked_series', None)
    res = {
        'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_end': str(bidx[-1].date()),
        'config': CONFIG_PATH,
        'cfg_inv_vol_window': cfg.inv_vol_window,
        'gate_constants': {'G1_CORR_MAX_FULL': G1_CORR_MAX_FULL,
                           'G1_CORR_MAX_COND': G1_CORR_MAX_COND,
                           'G2_TOP2_MIN_PCT': G2_TOP2_MIN_PCT,
                           'G3_IDIO_MIN_PCT': G3_IDIO_MIN_PCT,
                           'E0_IDX_ETF_CORR_MIN': E0_IDX_ETF_CORR_MIN,
                           'COND_VOL_PCT': COND_VOL_PCT,
                           'COND_DOWN_WEEKS': COND_DOWN_WEEKS,
                           'ROLL_CORR_W': ROLL_CORR_W},
        'candidates': cand.to_dict('records'),
        'e0_health': health,
        'align': align,
        'idx_etf_consistency': {'corr': ic, 'n': int(len(ov)),
                                'tracking_diff_ann_pct': (round(td, 3) if td is not None else None),
                                'passed': bool(ic is not None and ic >= E0_IDX_ETF_CORR_MIN)},
        'g1_index': g1_idx, 'g1_etf': g1_etf,
        'g2_index': g2_idx, 'g2_etf': g2_etf,
        'g3_index': g3_idx, 'g3_etf': g3_etf,
        'event_index': ev_idx,
    }
    res = recompute_verdict(res)
    res['answers'] = build_answers(res)
    JSON_PATH.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    MD_PATH.write_text(render(res))
    plot(res, wk_cols)
    _log(f'\n裁决: {res["verdict"]}  ({res["n_gates_passed"]}/3)')
    for k, v in res['gates'].items():
        _log(f"  {'✓' if v['passed'] else '✗'} {k}: {v['detail']}")
    _log(f'\n报告: {MD_PATH}\n数据: {JSON_PATH}')


if __name__ == '__main__':
    main()
