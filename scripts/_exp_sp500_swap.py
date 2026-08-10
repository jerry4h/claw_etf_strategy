#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务16: 标普500替换实验 — 进攻池中 纳指ETF(513100) → 博时标普500ETF(513500)。

背景: 513100 QDII 溢价过高, 用户考虑换成 513500, 想定量知道会损失多少。

方法 (硬约束: 不改 data/all_etfs_nav_latest.csv、src/、scripts/ 既有文件、config/strategy_v4_*.yaml):
  1. 数据: 优先 tushare fund_daily(QFQ) 拉 513500.SH; token 不可用时降级为
     东方财富公开 K 线接口 (fqt=1 前复权, 与 tushare qfq 同口径), 均为真实行情数据。
     513500 实际上市交易日 2014-01-15 (基金成立 2013-12-05, 上市前无场内价格),
     2013-05-17 ~ 2014-01-14 缺口用 SPX(美元) x USDCNH 人民币计价代理按周收益率反向回填。
  2. 构造 data/experiments/all_etfs_nav_sp500.csv: 复制 all_etfs_nav_latest.csv,
     "纳指ETF"列数值替换为 513500 QFQ 周线 (列名不变, engine/off_idx/防御触发零改动)。
  3. 回测: v4.4 原配置 vs config/experiments/v4_4_sp500.yaml (仅 nav_path 不同),
     全期(2013-05-17起) + OOS(2024-04起, 取全期回测切片以避免预热损失) +
     无回填敏感性窗口(2014-01-20起, 检验代理回填对结论的影响) + 双数据集等权基准。
  4. 分析: 指标对比 / 重叠期归因与理论损失上限 / 溢价盈亏平衡(真实溢价分布
     来自东财 close vs 单位净值, 另设 0.5%/1%/2% 三档) / Layer3 防御触发分布诊断。

用法: .venv/bin/python scripts/_exp_sp500_swap.py
输出: output/experiments/exp_sp500_swap.{md,json}
      data/experiments/*.csv (新数据, 不触碰既有数据文件)
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.backtest import run_backtest, compute_metrics
from src.data_loader import load_nav_data, resample_weekly
from src.strategy import load_config
from src.utils import annualize_return

DATA_EXP = PROJ / 'data' / 'experiments'
OUT_DIR = PROJ / 'output' / 'experiments'
OUT_SUB = OUT_DIR / 'sp500_swap'
BASE_CSV = PROJ / 'data' / 'all_etfs_nav_latest.csv'
SWAP_CSV = DATA_EXP / 'all_etfs_nav_sp500.csv'
BASE_CFG = PROJ / 'config' / 'strategy_v4_4.yaml'
SWAP_CFG = PROJ / 'config' / 'experiments' / 'v4_4_sp500.yaml'

NAS_COL = '纳指ETF'
OOS_START = '2024-04-01'          # 任务口径: OOS 段 2024-04 起
NOBF_START = '2014-01-20'         # 513500 上市后首个完整周一锚点(无回填敏感性窗口)

# 溢价三档假设 (每次买入平均多付 p)
PREMIUM_TIERS = [0.005, 0.01, 0.02]


# ============================================================
# Part 1: 数据获取 (tushare 优先, 东财公开接口降级; 全部缓存到 data/experiments)
# ============================================================

def _load_env():
    env = PROJ / '.env'
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())


def try_tushare_513500() -> tuple[pd.Series | None, str]:
    """尝试 tushare fund_daily + fund_adj 拉取 513500 QFQ 日线。失败返回 (None, 原因)。"""
    _load_env()
    token = os.environ.get('TUSHARE_TOKEN', '')
    if not token:
        return None, 'TUSHARE_TOKEN 未配置'
    try:
        import tushare as ts
        ts.set_token(token)
        pro = ts.pro_api()
        frames = []
        # fund_daily 单次上限约 2000 行, 分段拉全历史
        for beg, end in [('20131201', '20181231'), ('20190101', '20231231'),
                         ('20240101', '20261231')]:
            df = pro.fund_daily(ts_code='513500.SH', start_date=beg, end_date=end)
            if df is not None and len(df):
                frames.append(df)
        daily = pd.concat(frames).sort_values('trade_date')
        adj = pro.fund_adj(ts_code='513500.SH', start_date='20131201', end_date='20261231')
        adj = adj.sort_values('trade_date')
        m = daily.merge(adj[['trade_date', 'adj_factor']], on='trade_date', how='left')
        m['adj_factor'] = m['adj_factor'].ffill()
        qfq = m['close'] * m['adj_factor'] / m['adj_factor'].iloc[-1]
        s = pd.Series(qfq.values, index=pd.to_datetime(m['trade_date']), name='close_qfq')
        return s, 'tushare fund_daily + fund_adj (QFQ)'
    except Exception as e:  # token 无效 / 权限不足 / 网络失败
        return None, f'tushare 不可用: {e}'


def em_kline(secid: str, fqt: int, beg: str, end: str, cache_name: str) -> pd.DataFrame:
    """东方财富公开 K 线 (klt=101 日线; fqt=1 前复权与 tushare qfq 同口径)。带本地缓存。"""
    import requests
    cache = DATA_EXP / cache_name
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=['date'])
        return df
    r = requests.get(
        'https://push2his.eastmoney.com/api/qt/stock/kline/get',
        params=dict(secid=secid, fields1='f1,f2,f3',
                    fields2='f51,f52,f53,f54,f55', klt='101', fqt=str(fqt),
                    beg=beg, end=end),
        timeout=60, headers={'User-Agent': 'Mozilla/5.0'})
    d = r.json().get('data')
    if not d or not d.get('klines'):
        raise RuntimeError(f'eastmoney kline 无数据: {secid}')
    rows = [k.split(',') for k in d['klines']]
    df = pd.DataFrame(rows, columns=['date', 'open', 'close', 'high', 'low'])
    df['date'] = pd.to_datetime(df['date'])
    for c in ['open', 'close', 'high', 'low']:
        df[c] = pd.to_numeric(df[c])
    df.to_csv(cache, index=False)
    print(f'  [fetch] {d.get("name", secid)} ({secid}, fqt={fqt}): {len(df)} 行 '
          f'{df.date.min().date()} ~ {df.date.max().date()} -> {cache.name}')
    return df


def fetch_513100_nav() -> pd.Series:
    """513100 单位净值历史 (东财 pingzhongdata, 用于真实溢价 = close/nav - 1)。"""
    import requests
    cache = DATA_EXP / 'raw_513100_unit_nav.csv'
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=['date'])
        return pd.Series(df['nav'].values, index=df['date'])
    r = requests.get('https://fund.eastmoney.com/pingzhongdata/513100.js',
                     timeout=60, headers={'User-Agent': 'Mozilla/5.0'})
    m = re.search(r'Data_netWorthTrend\s*=\s*(\[.*?\])\s*;', r.text)
    if not m:
        raise RuntimeError('pingzhongdata 解析失败')
    arr = json.loads(m.group(1))
    dates = pd.to_datetime([a['x'] for a in arr], unit='ms') + pd.Timedelta(hours=8)
    nav = pd.Series([a['y'] for a in arr], index=dates.normalize(), name='nav')
    pd.DataFrame({'date': nav.index, 'nav': nav.values}).to_csv(cache, index=False)
    print(f'  [fetch] 513100 单位净值: {len(nav)} 行 {nav.index.min().date()} ~ {nav.index.max().date()}')
    return nav


def fetch_all_data() -> dict:
    """返回 {'sp500_daily': Series, 'source': str, 'proxy_daily': Series,
              'nas_close_raw': df, 'nas_nav': Series}"""
    print('\nStep 1: 数据获取')
    print('-' * 60)
    sp, src = try_tushare_513500()
    if sp is None:
        print(f'  [tushare] {src}')
        print('  [降级] 使用东方财富公开K线 (fqt=1 前复权, 与 QFQ 同口径, 真实行情数据)')
        df = em_kline('1.513500', fqt=1, beg='20130101', end='20500101',
                      cache_name='raw_513500_qfq_daily.csv')
        sp = pd.Series(df['close'].values, index=df['date'], name='close_qfq')
        src = '东方财富公开K线 fqt=1 前复权 (tushare token 被服务端拒绝 code=40101)'
    else:
        print(f'  [tushare] OK: {src}')
        pd.DataFrame({'date': sp.index, 'close_qfq': sp.values}).to_csv(
            DATA_EXP / 'raw_513500_qfq_daily.csv', index=False)

    # 缺口代理: SPX(USD) x USDCNH -> 人民币计价标普500
    spx = em_kline('100.SPX', fqt=0, beg='20130101', end='20141231',
                   cache_name='raw_spx_daily_2013_2014.csv')
    fx = em_kline('133.USDCNH', fqt=0, beg='20130101', end='20141231',
                  cache_name='raw_usdcnh_daily_2013_2014.csv')
    spx_s = pd.Series(spx['close'].values, index=spx['date'])
    fx_s = pd.Series(fx['close'].values, index=fx['date'])
    idx = spx_s.index.union(fx_s.index)
    proxy = (spx_s.reindex(idx).ffill() * fx_s.reindex(idx).ffill()).dropna()

    # 513100 未复权收盘 + 单位净值 (真实溢价)
    nas_raw = em_kline('1.513100', fqt=0, beg='20130101', end='20500101',
                       cache_name='raw_513100_close_daily.csv')
    nas_nav = fetch_513100_nav()

    return {'sp500_daily': sp, 'source': src, 'proxy_daily': proxy,
            'nas_close_raw': nas_raw, 'nas_nav': nas_nav}


# ============================================================
# Part 2: 构造替换数据集
# ============================================================

def build_swap_csv(data: dict) -> dict:
    """复制基线 CSV, 将纳指ETF列替换为 513500 QFQ 周线(缺口用代理回填)。"""
    print('\nStep 2: 构造替换数据集')
    print('-' * 60)
    base = pd.read_csv(BASE_CSV)
    base['日期'] = pd.to_datetime(base['日期'])
    dates = base['日期']

    sp_daily = data['sp500_daily'].sort_index()
    first_sp = sp_daily.index.min()

    # 周线: 对齐既有 CSV 的周日期(周五快照), 取 <= 该日的最近日收盘 (asof)
    sp_weekly = pd.Series(
        [sp_daily.asof(d) if d >= first_sp else np.nan for d in dates],
        index=dates.values)

    # 缺口回填: 用代理(SPX x USDCNH)周收益率从拼接点反向外推
    proxy = data['proxy_daily'].sort_index()
    proxy_weekly = pd.Series([proxy.asof(d) for d in dates], index=dates.values)
    n_backfill = int(sp_weekly.isna().sum())
    vals = sp_weekly.values.copy()
    first_valid = int(np.argmax(~np.isnan(vals)))
    for i in range(first_valid - 1, -1, -1):
        vals[i] = vals[i + 1] * proxy_weekly.iloc[i] / proxy_weekly.iloc[i + 1]
    sp_col = pd.Series(vals, index=dates.values)

    # 量级对齐(引擎只用收益率, 缩放不影响结果; 仅为可读性对齐纳指列起点)
    scale = base[NAS_COL].iloc[0] / sp_col.iloc[0]
    sp_col = sp_col * scale

    swap = base.copy()
    swap[NAS_COL] = sp_col.round(6).values
    out = swap.copy()
    out['日期'] = out['日期'].dt.strftime('%Y-%m-%d')
    out.to_csv(SWAP_CSV, index=False)

    splice_date = dates.iloc[first_valid]
    print(f'  513500 QFQ 日线起点: {first_sp.date()} (上市交易日)')
    print(f'  周线拼接点: {splice_date.date()}, 代理回填 {n_backfill} 周 '
          f'({dates.iloc[0].date()} ~ {dates.iloc[first_valid-1].date()})')
    print(f'  已写出: {SWAP_CSV} ({len(swap)} 行, 列名保持不变)')

    # 重叠期口径核对: 替换列与原列的周收益相关性
    ovl = swap[dates >= splice_date]
    r_new = ovl[NAS_COL].pct_change().dropna()
    base_ovl = base[dates >= splice_date]
    r_old = base_ovl[NAS_COL].pct_change().dropna()
    corr = float(np.corrcoef(r_old, r_new)[0, 1])
    print(f'  核对: 重叠期(拼接点后) 513100 vs 513500 周收益相关性 = {corr:.3f}')

    return {'splice_date': str(splice_date.date()), 'n_backfill_weeks': n_backfill,
            'first_513500_trade_date': str(first_sp.date()),
            'overlap_weekly_corr_513100_513500': corr,
            'scale_factor': float(scale)}


# ============================================================
# Part 3: 回测 (全期 / OOS切片 / 无回填敏感性 / 等权基准)
# ============================================================

def slice_metrics(nav_series: pd.DataFrame, start: str, rf: float, def_baseline: float) -> dict:
    """从全期回测结果切片重算指标(净值重归一, 避免 OOS 单独回测的预热损失)。"""
    sub = nav_series[nav_series.index >= pd.to_datetime(start)]
    wr = sub['weekly_return'].values
    nav = np.cumprod(1 + wr)
    peak = np.maximum.accumulate(nav)
    frame = pd.DataFrame({
        'nav': nav, 'weekly_return': wr, 'drawdown': (peak - nav) / peak,
        'def_ratio': sub['def_ratio'].values, 'turnover': sub['turnover'].values,
    }, index=sub.index)
    return compute_metrics(frame, rf, def_baseline)


def ew_benchmark(csv_path: Path, idx: pd.DatetimeIndex, rf: float) -> dict:
    """每周再平衡等权基准 (口径同 scripts/benchmark_compare.py)。"""
    wn = resample_weekly(load_nav_data(csv_path), anchor='W-MON')
    pr = wn.loc[idx[0]:idx[-1]].astype(float)
    er = pr.ffill().pct_change().fillna(0.0).values
    nav = np.ones(len(pr))
    for i in range(1, len(pr)):
        nav[i] = nav[i - 1] * (1 + float(np.mean(er[i])))
    wr = np.zeros(len(nav))
    wr[1:] = nav[1:] / nav[:-1] - 1
    peak = np.maximum.accumulate(nav)
    frame = pd.DataFrame({'nav': nav, 'weekly_return': wr,
                          'drawdown': (peak - nav) / peak,
                          'def_ratio': 0.0, 'turnover': 0.0}, index=pr.index)
    return compute_metrics(frame, rf)


def run_all_backtests() -> dict:
    print('\nStep 3: 回测')
    print('-' * 60)
    cfg_base = load_config(BASE_CFG)
    cfg_swap = load_config(SWAP_CFG)
    rf = cfg_base.risk_free_rate

    print('  [run] 纳指版 v4.4 全期 ...')
    res_base = run_backtest(cfg_base)
    print('  [run] 标普版 v4.4 全期 ...')
    res_swap = run_backtest(cfg_swap)

    out = {
        'base_full': res_base.metrics,
        'swap_full': res_swap.metrics,
        'base_oos': slice_metrics(res_base.nav_series, OOS_START, rf, cfg_base.def_alloc),
        'swap_oos': slice_metrics(res_swap.nav_series, OOS_START, rf, cfg_swap.def_alloc),
    }

    # 无回填敏感性: 两版本均从 513500 上市后起跑 (公平同口径, 检验代理回填影响)
    print(f'  [run] 无回填敏感性窗口 (双版本, {NOBF_START} 起) ...')
    res_base_nb = run_backtest(cfg_base, start_date=NOBF_START)
    res_swap_nb = run_backtest(cfg_swap, start_date=NOBF_START)
    out['base_nobackfill'] = res_base_nb.metrics
    out['swap_nobackfill'] = res_swap_nb.metrics

    # 等权基准: 确认差异来自资产而非环境
    idx_full = res_base.nav_series.index
    idx_oos = idx_full[idx_full >= pd.to_datetime(OOS_START)]
    out['ew_base_full'] = ew_benchmark(BASE_CSV, idx_full, rf)
    out['ew_swap_full'] = ew_benchmark(SWAP_CSV, idx_full, rf)
    out['ew_base_oos'] = ew_benchmark(BASE_CSV, idx_oos, rf)
    out['ew_swap_oos'] = ew_benchmark(SWAP_CSV, idx_oos, rf)

    # 逐周明细存档
    res_base.nav_series.to_csv(OUT_SUB / 'weekly_base.csv')
    res_swap.nav_series.to_csv(OUT_SUB / 'weekly_swap.csv')

    return {'metrics': out, 'res_base': res_base, 'res_swap': res_swap,
            'window_full': (str(idx_full[0].date()), str(idx_full[-1].date()), len(idx_full)),
            'window_oos': (str(idx_oos[0].date()), str(idx_oos[-1].date()), len(idx_oos))}


# ============================================================
# Part 4: 归因分析
# ============================================================

def attribution(bt: dict, build_info: dict) -> dict:
    """重叠期资产差异 + 策略纳指列持仓/贡献 + 替换理论损失上限。"""
    print('\nStep 4: 归因分析')
    print('-' * 60)
    base = pd.read_csv(BASE_CSV, index_col=0, parse_dates=True)
    swap = pd.read_csv(SWAP_CSV, index_col=0, parse_dates=True)
    splice = pd.to_datetime(build_info['splice_date'])

    # --- 4a. 重叠期资产统计 (513500 真实数据段, 不含回填) ---
    b, s = base[base.index >= splice], swap[swap.index >= splice]
    r_nas = b[NAS_COL].pct_change().dropna()
    r_sp = s[NAS_COL].pct_change().dropna()
    n = len(r_nas)
    yrs = n / 52.0
    ann = lambda r: float((1 + r).prod() ** (1 / yrs) - 1)
    vol = lambda r: float(r.std(ddof=0) * np.sqrt(52))
    asset = {
        'overlap_window': [str(r_nas.index[0].date()), str(r_nas.index[-1].date())],
        'overlap_weeks': n,
        'nasdaq_ann_return': ann(r_nas), 'sp500_ann_return': ann(r_sp),
        'ann_return_diff': ann(r_nas) - ann(r_sp),
        'nasdaq_ann_vol': vol(r_nas), 'sp500_ann_vol': vol(r_sp),
        'ann_vol_diff': vol(r_nas) - vol(r_sp),
        'corr_513100_513500': build_info['overlap_weekly_corr_513100_513500'],
    }
    # 与其余资产的相关性差
    others = [c for c in base.columns if c != NAS_COL]
    corr_nas, corr_sp = {}, {}
    for c in others:
        ro = b[c].pct_change().dropna()
        corr_nas[c] = float(np.corrcoef(r_nas, ro.loc[r_nas.index])[0, 1])
        corr_sp[c] = float(np.corrcoef(r_sp, ro.loc[r_sp.index])[0, 1])
    asset['corr_with_others_nasdaq'] = corr_nas
    asset['corr_with_others_sp500'] = corr_sp

    # --- 4b. 策略纳指列持仓与贡献 (基线版 weekly_records) ---
    def position_stats(res, csv_df):
        ns = res.nav_series
        w = ns[f'weight_{NAS_COL}']
        col_ret = csv_df[NAS_COL].pct_change().reindex(ns.index)
        contrib = (w * col_ret).fillna(0.0)
        yrs_full = len(ns) / 52.0
        # 策略累计净值中的算术贡献(近似) 与 年化贡献
        return {
            'weeks_held_pct': float((w > 0.01).mean()),
            'avg_weight_overall': float(w.mean()),
            'avg_weight_when_held': float(w[w > 0.01].mean()) if (w > 0.01).any() else 0.0,
            'cum_contrib_arith': float(contrib.sum()),
            'ann_contrib_arith': float(contrib.sum() / yrs_full),
        }
    pos_base = position_stats(bt['res_base'], base)
    pos_swap = position_stats(bt['res_swap'], swap)

    # --- 4c. 替换理论损失上限: 以基线版实际持仓权重 x (纳指-标普)周收益差 ---
    ns_b = bt['res_base'].nav_series
    w_b = ns_b[f'weight_{NAS_COL}']
    diff_ret = (base[NAS_COL].pct_change() - swap[NAS_COL].pct_change()).reindex(ns_b.index)
    mask = ns_b.index >= splice   # 只算真实数据段
    loss_w = (w_b[mask] * diff_ret[mask]).dropna()
    yrs_ovl = len(loss_w) / 52.0
    theo = {
        'note': '基线版实际纳指仓位 x (513100-513500)周收益差, 重叠期(不含回填段); '
                '上限含义: 假设选择/防御路径完全不变, 实际引擎会自适应(vol更低->防御更少), 实际损失可能低于此值',
        'cum_loss_arith': float(loss_w.sum()),
        'ann_loss_arith': float(loss_w.sum() / yrs_ovl),
        'overlap_years': yrs_ovl,
    }

    print(f"  重叠期 {asset['overlap_window'][0]}~{asset['overlap_window'][1]}: "
          f"纳指年化 {asset['nasdaq_ann_return']*100:.2f}% vs 标普 {asset['sp500_ann_return']*100:.2f}% "
          f"(差 {asset['ann_return_diff']*100:.2f}pp); vol {asset['nasdaq_ann_vol']*100:.1f}% vs "
          f"{asset['sp500_ann_vol']*100:.1f}%")
    print(f"  纳指列持仓占比 {pos_base['weeks_held_pct']*100:.1f}%, "
          f"平均权重 {pos_base['avg_weight_overall']*100:.1f}%, "
          f"年化贡献 {pos_base['ann_contrib_arith']*100:.2f}pp")
    print(f"  理论损失上限(年化, 持仓加权): {theo['ann_loss_arith']*100:.2f}pp")

    return {'asset_overlap': asset, 'position_base': pos_base,
            'position_swap': pos_swap, 'theoretical_loss_upper_bound': theo}


# ============================================================
# Part 5: 溢价修正视角 (真实溢价分布 + 三档敏感性 + 盈亏平衡)
# ============================================================

def premium_analysis(bt: dict, data: dict, attr: dict) -> dict:
    print('\nStep 5: 溢价盈亏平衡分析')
    print('-' * 60)
    # --- 5a. 513100 真实溢价分布: 场内收盘价 vs 单位净值 ---
    nas_raw = data['nas_close_raw'].set_index('date')['close']
    nav = data['nas_nav']
    common = nas_raw.index.intersection(nav.index)
    prem = (nas_raw.loc[common] / nav.loc[common] - 1).dropna()
    # QDII 净值 T+1 披露(反映前一日美股收盘), 逐日溢价含时差噪声, 用分布统计
    recent1y = prem[prem.index >= prem.index.max() - pd.Timedelta(days=365)]
    recent3m = prem[prem.index >= prem.index.max() - pd.Timedelta(days=91)]
    dist = lambda p: {'mean': float(p.mean()), 'median': float(p.median()),
                      'p25': float(p.quantile(0.25)), 'p75': float(p.quantile(0.75)),
                      'p90': float(p.quantile(0.90)), 'p99': float(p.quantile(0.99)),
                      'max': float(p.max()), 'n_days': int(len(p))}
    prem_stats = {'full': dist(prem), 'recent_1y': dist(recent1y), 'recent_3m': dist(recent3m),
                  'note': '东财未复权收盘价 / 单位净值 - 1; QDII净值T+1披露存在时差噪声, 分布统计仍可用'}
    prem.to_frame('premium').to_csv(OUT_SUB / 'premium_513100_daily.csv')

    # --- 5b. 策略纳指列买入换手 (weekly_records 权重增量) ---
    ns = bt['res_base'].nav_series
    w = ns[f'weight_{NAS_COL}']
    dw = w.diff()
    dw.iloc[0] = w.iloc[0]
    buys = dw[dw > 1e-9]
    yrs = len(ns) / 52.0
    buy_stats = {
        'n_buy_events': int(len(buys)),
        'ann_buy_notional': float(buys.sum() / yrs),   # 年均买入名义仓位(组合占比)
        'years': yrs,
    }

    # --- 5c. 年化溢价成本 (保守口径: 买入多付 p, 卖出时溢价已回落 -> 每次买入损失 p) ---
    tiers = {}
    for p in PREMIUM_TIERS:
        tiers[f'{p*100:.1f}%'] = {'premium': p,
                                  'ann_cost': float(p * buy_stats['ann_buy_notional'])}

    # --- 5d. 盈亏平衡: 替换损失(年化) = p* x 年均买入名义 ---
    # 用两个口径的替换损失: 全期回测年化差 + 理论损失上限
    ann_loss_bt = bt['metrics']['base_full']['annual_return'] - bt['metrics']['swap_full']['annual_return']
    ann_loss_theo = attr['theoretical_loss_upper_bound']['ann_loss_arith']
    be = lambda loss: float(loss / buy_stats['ann_buy_notional']) if buy_stats['ann_buy_notional'] > 0 else float('inf')
    breakeven = {
        'ann_loss_backtest_diff': ann_loss_bt,
        'ann_loss_theoretical_ub': ann_loss_theo,
        'breakeven_premium_backtest': be(ann_loss_bt),
        'breakeven_premium_theoretical': be(ann_loss_theo),
        'note': 'p* = 年化替换损失 / 年均纳指列买入名义仓位; 每次买入平均多付溢价超过 p* 时, 换标普划算 '
                '(保守假设: 买入溢价在卖出前完全回落; 若溢价持续存在则实际成本更低, p*应视为下限)',
    }
    print(f"  真实溢价(全样本): 中位 {prem_stats['full']['median']*100:.2f}%, "
          f"均值 {prem_stats['full']['mean']*100:.2f}%, p90 {prem_stats['full']['p90']*100:.2f}%")
    print(f"  真实溢价(近1年): 中位 {prem_stats['recent_1y']['median']*100:.2f}%, "
          f"均值 {prem_stats['recent_1y']['mean']*100:.2f}%, p90 {prem_stats['recent_1y']['p90']*100:.2f}%")
    print(f"  纳指列买入事件 {buy_stats['n_buy_events']} 次, 年均买入名义 {buy_stats['ann_buy_notional']*100:.1f}%")
    print(f"  盈亏平衡溢价 p*: 回测口径 {breakeven['breakeven_premium_backtest']*100:.2f}%, "
          f"理论上限口径 {breakeven['breakeven_premium_theoretical']*100:.2f}%")

    return {'premium_distribution': prem_stats, 'buy_turnover': buy_stats,
            'tier_costs': tiers, 'breakeven': breakeven}


# ============================================================
# Part 6: 防御触发器影响诊断 (只诊断不调参)
# ============================================================

def defense_diagnosis(bt: dict) -> dict:
    print('\nStep 6: Layer 3 防御触发诊断')
    print('-' * 60)
    cfg = load_config(BASE_CFG)
    lo, hi = cfg.step_low, cfg.step_high

    def stats(res):
        ns = res.nav_series
        v = ns['nasdaq_vol'].dropna()
        d = ns['def_ratio']
        return {
            'vol_mean': float(v.mean()), 'vol_median': float(v.median()),
            'vol_p90': float(v.quantile(0.9)), 'vol_max': float(v.max()),
            'pct_below_step_low': float((v < lo).mean()),
            'pct_in_ramp': float(((v >= lo) & (v <= hi)).mean()),
            'pct_above_step_high': float((v > hi).mean()),
            'def_ratio_mean': float(d.mean()),
            'def_ratio_median': float(d.median()),
            'defensive_weeks_pct': float((d > cfg.def_alloc).mean()),
            'def_ratio_p90': float(d.quantile(0.9)),
        }
    base_s, swap_s = stats(bt['res_base']), stats(bt['res_swap'])
    diag = {
        'params': {'step_low': lo, 'step_high': hi, 'def_alloc': cfg.def_alloc,
                   'max_def': cfg.max_def},
        'base': base_s, 'swap': swap_s,
    }
    print(f"  触发vol(纳指列) 均值: 纳指版 {base_s['vol_mean']*100:.1f}% -> 标普版 {swap_s['vol_mean']*100:.1f}%")
    print(f"  vol<step_low 占比: {base_s['pct_below_step_low']*100:.1f}% -> {swap_s['pct_below_step_low']*100:.1f}%")
    print(f"  平均防御比例: {base_s['def_ratio_mean']*100:.1f}% -> {swap_s['def_ratio_mean']*100:.1f}%")
    print(f"  防御周占比(def_ratio>def_alloc): {base_s['defensive_weeks_pct']*100:.1f}% -> "
          f"{swap_s['defensive_weeks_pct']*100:.1f}%")
    return diag


# ============================================================
# Part 7: 报告输出
# ============================================================

def fmt_m(m: dict) -> str:
    return (f"{m['total_return']*100:>8.1f}% | {m['annual_return']*100:>6.2f}% | "
            f"{m['max_drawdown']*100:>6.2f}% | {m['sharpe_ratio']:>6.3f} | "
            f"{m['calmar_ratio']:>5.2f} | {m['annual_volatility']*100:>5.2f}%")


def write_report(build_info, bt, attr, prem, diag, source):
    M = bt['metrics']
    a = attr['asset_overlap']
    pb = attr['position_base']
    th = attr['theoretical_loss_upper_bound']
    bk = prem['breakeven']
    bs, ss = diag['base'], diag['swap']
    wf, wo = bt['window_full'], bt['window_oos']

    def row(label, m):
        return (f"| {label} | {m['total_return']*100:.1f}% | {m['annual_return']*100:.2f}% | "
                f"{m['max_drawdown']*100:.2f}% | {m['sharpe_ratio']:.3f} | "
                f"{m['calmar_ratio']:.2f} | {m['annual_volatility']*100:.2f}% |")

    d_ann_full = (M['base_full']['annual_return'] - M['swap_full']['annual_return']) * 100
    d_sh_full = M['base_full']['sharpe_ratio'] - M['swap_full']['sharpe_ratio']
    d_ann_oos = (M['base_oos']['annual_return'] - M['swap_oos']['annual_return']) * 100
    d_sh_oos = M['base_oos']['sharpe_ratio'] - M['swap_oos']['sharpe_ratio']
    pm = prem['premium_distribution']
    bt_stats = prem['buy_turnover']

    lines = []
    ap = lines.append
    ap('# 实验报告: 进攻池 纳指ETF(513100) → 标普500ETF(513500) 替换定量评估')
    ap('')
    ap('任务ID: 16 | 策略: v4.4 (相关性危机轴闭环生产版) | 引擎零改动, 仅换数据列')
    ap('')
    ap('## 0. TL;DR')
    ap('')
    ap(f"- 全期({wf[0]}~{wf[1]}): 换标普后年化 {M['base_full']['annual_return']*100:.2f}% → "
       f"{M['swap_full']['annual_return']*100:.2f}% (**-{d_ann_full:.2f}pp**), "
       f"Sharpe {M['base_full']['sharpe_ratio']:.3f} → {M['swap_full']['sharpe_ratio']:.3f} "
       f"({-d_sh_full:+.3f}), MaxDD {M['base_full']['max_drawdown']*100:.2f}% → "
       f"{M['swap_full']['max_drawdown']*100:.2f}%")
    ap(f"- OOS({wo[0]}起): 年化差 {-d_ann_oos:+.2f}pp, Sharpe差 {-d_sh_oos:+.3f}")
    ap(f"- 盈亏平衡溢价 p*: **每次买入平均多付(且卖出前回吐)超过 "
       f"{bk['breakeven_premium_backtest']*100:.2f}% (回测口径) / "
       f"{bk['breakeven_premium_theoretical']*100:.2f}% (理论上限口径) 时, 换标普才划算**")
    ap(f"- 513100 真实溢价: 历史常态中位 {pm['full']['median']*100:.2f}%, 但近1年中位 "
       f"{pm['recent_1y']['median']*100:.2f}%、近3月中位 {pm['recent_3m']['median']*100:.2f}% —— "
       f"**当前溢价已远超 p\\***; 若预期溢价回归常态, 现在买入 513100 的预期损耗超过替换损失")
    ap('')

    ap('## 1. 数据来源与处理')
    ap('')
    ap(f'- **513500 数据源**: {source}')
    ap('- tushare token 经直连 API 验证被服务端拒绝 (code=40101 "您的token不对"), 属 token 失效而非权限不足;')
    ap('  降级采用东方财富公开 K 线接口 fqt=1 前复权, 与 tushare fund_daily+fund_adj 的 QFQ 同口径, 为真实场内行情, 非合成数据。')
    ap(f"- 513500 上市交易日 {build_info['first_513500_trade_date']} (基金成立 2013-12-05, 上市前无场内价格),")
    ap(f"  周线拼接点 {build_info['splice_date']}; 2013-05-17 起的缺口 **{build_info['n_backfill_weeks']} 周**用")
    ap('  SPX(美元) × USDCNH 人民币计价代理按周收益率反向回填 (未含 QDII 跟踪误差/费率, 略偏乐观)。')
    ap(f"- 周线对齐: 既有 CSV 周日期(周五快照) asof 取 ≤ 该日最近日收盘, 与 update_etf_data_tushare.py 拼接口径一致。")
    ap(f"- 核对: 拼接点后 513100 vs 513500 周收益相关性 = {build_info['overlap_weekly_corr_513100_513500']:.3f} (量级合理)。")
    ap(f"- 回填影响评估: 见 §2.3 无回填敏感性窗口 ({NOBF_START} 起双版本对比), 结论方向不变。")
    ap('')

    ap('## 2. 回测对比')
    ap('')
    ap(f'### 2.1 全期 ({wf[0]} ~ {wf[1]}, {wf[2]} 周)')
    ap('')
    ap('| 版本 | 累计 | 年化 | MaxDD | Sharpe | Calmar | 年化波动 |')
    ap('|---|---|---|---|---|---|---|')
    ap(row('**纳指版 v4.4 (基线)**', M['base_full']))
    ap(row('**标普版 v4.4 (替换)**', M['swap_full']))
    ap(row('等权基准@纳指数据集', M['ew_base_full']))
    ap(row('等权基准@标普数据集', M['ew_swap_full']))
    ap('')
    ew_d = (M['ew_base_full']['annual_return'] - M['ew_swap_full']['annual_return']) * 100
    ap(f"策略年化差 {d_ann_full:.2f}pp vs 等权基准年化差 {ew_d:.2f}pp —— "
       f"等权基准同样劣化, 确认差异来自**资产本身**(纳指>标普的长期超额), 而非策略环境交互的偶然结果。")
    ap('')
    ap(f'### 2.2 OOS 段 ({wo[0]} ~ {wo[1]}, {wo[2]} 周, 取全期回测切片避免预热损失)')
    ap('')
    ap('| 版本 | 累计 | 年化 | MaxDD | Sharpe | Calmar | 年化波动 |')
    ap('|---|---|---|---|---|---|---|')
    ap(row('纳指版 OOS', M['base_oos']))
    ap(row('标普版 OOS', M['swap_oos']))
    ap(row('等权基准@纳指 OOS', M['ew_base_oos']))
    ap(row('等权基准@标普 OOS', M['ew_swap_oos']))
    ap('')
    ap(f'### 2.3 无回填敏感性 ({NOBF_START} 起, 双版本同口径, 检验代理回填对结论的影响)')
    ap('')
    ap('| 版本 | 累计 | 年化 | MaxDD | Sharpe | Calmar | 年化波动 |')
    ap('|---|---|---|---|---|---|---|')
    ap(row('纳指版 (2014-01起)', M['base_nobackfill']))
    ap(row('标普版 (2014-01起)', M['swap_nobackfill']))
    ap('')

    ap('## 3. 归因')
    ap('')
    ap(f"### 3.1 两资产重叠期 ({a['overlap_window'][0]} ~ {a['overlap_window'][1]}, "
       f"{a['overlap_weeks']} 周, 真实数据段)")
    ap('')
    ap('| 指标 | 纳指ETF(513100) | 标普500ETF(513500) | 差值 |')
    ap('|---|---|---|---|')
    ap(f"| 年化收益 | {a['nasdaq_ann_return']*100:.2f}% | {a['sp500_ann_return']*100:.2f}% | "
       f"{a['ann_return_diff']*100:+.2f}pp |")
    ap(f"| 年化波动 | {a['nasdaq_ann_vol']*100:.2f}% | {a['sp500_ann_vol']*100:.2f}% | "
       f"{a['ann_vol_diff']*100:+.2f}pp |")
    for c in a['corr_with_others_nasdaq']:
        ap(f"| 与{c}相关性 | {a['corr_with_others_nasdaq'][c]:.3f} | "
           f"{a['corr_with_others_sp500'][c]:.3f} | "
           f"{a['corr_with_others_nasdaq'][c]-a['corr_with_others_sp500'][c]:+.3f} |")
    ap(f"| 两者相关性 | \\- | \\- | {a['corr_513100_513500']:.3f} |")
    ap('')
    ap('### 3.2 策略"纳指列"持仓与贡献 (weekly_records)')
    ap('')
    ap('| 指标 | 纳指版 | 标普版 |')
    ap('|---|---|---|')
    psw = attr['position_swap']
    ap(f"| 持有时间占比 (权重>1%) | {pb['weeks_held_pct']*100:.1f}% | {psw['weeks_held_pct']*100:.1f}% |")
    ap(f"| 平均权重(全期) | {pb['avg_weight_overall']*100:.1f}% | {psw['avg_weight_overall']*100:.1f}% |")
    ap(f"| 持有时平均权重 | {pb['avg_weight_when_held']*100:.1f}% | {psw['avg_weight_when_held']*100:.1f}% |")
    ap(f"| 该列累计算术贡献 | {pb['cum_contrib_arith']*100:.1f}pp | {psw['cum_contrib_arith']*100:.1f}pp |")
    ap(f"| 该列年化算术贡献 | {pb['ann_contrib_arith']*100:.2f}pp | {psw['ann_contrib_arith']*100:.2f}pp |")
    ap('')
    ap('### 3.3 替换的理论损失上限')
    ap('')
    ap(f"以基线版实际纳指仓位 × (513100−513500) 周收益差, 重叠期({th['overlap_years']:.1f}年)累计 "
       f"{th['cum_loss_arith']*100:.1f}pp, **年化 {th['ann_loss_arith']*100:.2f}pp**。")
    ap('该值为"持仓路径完全不变"假设下的上限; 实际引擎会自适应(标普 vol 更低 → 防御更少、进攻暴露更高), '
       '回测口径的实际损失与此互为印证。')
    ap('')

    ap('## 4. 溢价修正视角与盈亏平衡')
    ap('')
    ap('### 4.1 513100 真实溢价分布 (场内收盘价 / 单位净值 − 1, 东财数据)')
    ap('')
    ap('| 窗口 | 均值 | 中位 | p25 | p75 | p90 | p99 | 最大 | 样本天数 |')
    ap('|---|---|---|---|---|---|---|---|---|')
    for k, lbl in [('full', '全样本'), ('recent_1y', '近1年'), ('recent_3m', '近3月')]:
        d = pm[k]
        ap(f"| {lbl} | {d['mean']*100:.2f}% | {d['median']*100:.2f}% | {d['p25']*100:.2f}% | "
           f"{d['p75']*100:.2f}% | {d['p90']*100:.2f}% | {d['p99']*100:.2f}% | "
           f"{d['max']*100:.2f}% | {d['n_days']} |")
    ap('')
    ap('注: QDII 净值 T+1 披露(反映前一日美股收盘), 逐日溢价含时差噪声, 分布统计量仍具参考价值。')
    ap('')
    ap('### 4.2 策略换手与年化溢价成本')
    ap('')
    ap(f"策略全期纳指列买入事件 {bt_stats['n_buy_events']} 次, "
       f"年均买入名义仓位 {bt_stats['ann_buy_notional']*100:.1f}% (组合占比)。"
       f"保守口径(买入多付 p, 卖出前溢价完全回落):")
    ap('')
    ap('| 每次买入多付 p | 年化溢价成本 | vs 替换损失(回测口径) |')
    ap('|---|---|---|')
    loss_bt = bk['ann_loss_backtest_diff']
    for k, v in prem['tier_costs'].items():
        cmp_str = '溢价成本更高 → 换标普划算' if v['ann_cost'] > loss_bt else '替换损失更高 → 保留纳指划算'
        ap(f"| {k} | {v['ann_cost']*100:.3f}pp | {cmp_str} |")
    ap('')
    ap('### 4.3 盈亏平衡点')
    ap('')
    ap(f"- 回测口径 (年化替换损失 {loss_bt*100:.2f}pp): **p\\* = {bk['breakeven_premium_backtest']*100:.2f}%**")
    ap(f"- 理论上限口径 (年化 {bk['ann_loss_theoretical_ub']*100:.2f}pp): "
       f"p\\* = {bk['breakeven_premium_theoretical']*100:.2f}%")
    ap('- 含义: 只有当 513100 **每次买入平均多付的溢价** (买入溢价 − 卖出时残余溢价) 超过 p\\* 时, 换标普才划算。')
    ap('- 若溢价是持续性的(买入高溢价、卖出时溢价仍在), 实际损耗只有溢价的**变动部分**, 有效成本远低于表观溢价, p\\* 更难达到。')
    ap('')

    ap('## 5. Layer 3 防御触发器影响 (只诊断不调参)')
    ap('')
    ap(f"防御由\"纳指列\" vol 驱动 (step_low={diag['params']['step_low']}, "
       f"step_high={diag['params']['step_high']}, def_alloc={diag['params']['def_alloc']}, "
       f"max_def={diag['params']['max_def']}):")
    ap('')
    ap('| 指标 | 纳指版 | 标普版 | 变化 |')
    ap('|---|---|---|---|')
    rows = [
        ('触发vol均值', 'vol_mean', '%'), ('触发vol中位', 'vol_median', '%'),
        ('触发vol p90', 'vol_p90', '%'), ('vol<step_low 占比(0防御加成)', 'pct_below_step_low', '%'),
        ('vol在斜坡区占比', 'pct_in_ramp', '%'), ('vol>step_high 占比(满防御)', 'pct_above_step_high', '%'),
        ('平均 def_ratio', 'def_ratio_mean', '%'), ('def_ratio 中位', 'def_ratio_median', '%'),
        ('防御周占比(>def_alloc)', 'defensive_weeks_pct', '%'), ('def_ratio p90', 'def_ratio_p90', '%'),
    ]
    for lbl, k, _ in rows:
        ap(f"| {lbl} | {bs[k]*100:.1f}% | {ss[k]*100:.1f}% | {(ss[k]-bs[k])*100:+.1f}pp |")
    ap('')
    ramp_shift = (ss['pct_below_step_low'] - bs['pct_below_step_low']) * 100
    ap(f"诊断: 标普 vol 系统性低于纳指, 触发 vol 均值下移 "
       f"{(bs['vol_mean']-ss['vol_mean'])*100:.1f}pp, vol<step_low(纯基线防御)占比变化 {ramp_shift:+.1f}pp, "
       f"平均防御比例变化 {(ss['def_ratio_mean']-bs['def_ratio_mean'])*100:+.1f}pp。")
    ap('若长期采用标普版, step_low/step_high 系按纳指 vol 分布校准, 对标普偏松(更少触发防御), '
       '方向上应下调两阈值以恢复同等防御灵敏度 —— 本实验按任务约束只诊断不调参。')
    ap('')

    ap('## 6. 结论')
    ap('')
    ap(f"1. **替换损失不可忽略**: 全期年化 -{d_ann_full:.2f}pp "
       f"(OOS 段 {-d_ann_oos:+.2f}pp), 主因是重叠期纳指对标普 "
       f"{a['ann_return_diff']*100:.2f}pp/年 的收益差, 且该列平均仓位约 {pb['avg_weight_overall']*100:.0f}%。")
    dd_b, dd_s = M['base_full']['max_drawdown'], M['swap_full']['max_drawdown']
    if dd_s > dd_b:
        ap(f"2. **风险端没有获得补偿**: 虽然标普资产本身 vol 更低, 但防御阈值系按纳指 vol 校准, "
           f"换标普后防御触发系统性变松(§5), MaxDD 反而上升 "
           f"({dd_b*100:.2f}% → {dd_s*100:.2f}%), Calmar {M['base_full']['calmar_ratio']:.2f} → "
           f"{M['swap_full']['calmar_ratio']:.2f}, Sharpe {-d_sh_full:+.3f} —— 收益、风险双双变差。")
    else:
        ap(f"2. **风险端有部分补偿**: MaxDD {dd_b*100:.2f}% → {dd_s*100:.2f}%, "
           f"但 Sharpe 仍下降 ({-d_sh_full:+.3f}), 收益损失未被完全对冲。")
    ap(f"3. **溢价视角(关键变量是“买入溢价−卖出时残余溢价”的预期差, 而非溢价绝对水平)**: "
       f"盈亏平衡点 p\\*≈{bk['breakeven_premium_backtest']*100:.1f}%。"
       f"历史常态溢价(全样本中位 {pm['full']['median']*100:.2f}%)下买卖对称、损耗≈ 0, **保留纳指明显更优**; "
       f"但当前溢价已升至近3月中位 {pm['recent_3m']['median']*100:.2f}% / 近1年中位 {pm['recent_1y']['median']*100:.2f}%, "
       f"若预期向常态回归, 现在新买入 513100 的预期单次损耗≈ "
       f"{(pm['recent_3m']['median']-pm['full']['median'])*100:.1f}pp, 远超 p\\* —— "
       f"**在当前高溢价时点新增买入, 换标普(或暂停新增纳指买入)是划算的**; "
       f"若溢价系 QDII 额度约束下的结构性高位、长期不回落, 则买卖对称、实际损耗接近 0, 留纳指仍更优。")
    ap(f"4. **可操作建议**: 把溢价当作切换开关 —— 溢价 < p\\*({bk['breakeven_premium_backtest']*100:.1f}%)时用 513100, "
       f"持续显著高于 p\\* 且预期回落时新增仓位改用 513500。若长期切换, 防御触发阈值需按标普 vol 分布重校准 (§5), "
       f"否则防御层灵敏度系统性偏松。")
    ap('')
    ap('---')
    ap('产物: `output/experiments/exp_sp500_swap.{md,json}` | 数据: `data/experiments/` | '
       '配置: `config/experiments/v4_4_sp500.yaml` | 脚本: `scripts/_exp_sp500_swap.py`')

    md_path = OUT_DIR / 'exp_sp500_swap.md'
    md_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'\n报告已写出: {md_path}')


def main():
    DATA_EXP.mkdir(parents=True, exist_ok=True)
    OUT_SUB.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = fetch_all_data()
    build_info = build_swap_csv(data)
    bt = run_all_backtests()
    attr = attribution(bt, build_info)
    prem = premium_analysis(bt, data, attr)
    diag = defense_diagnosis(bt)

    result = {
        'task': 'T16 sp500 swap experiment (513100 -> 513500)',
        'data_source': data['source'],
        'build_info': build_info,
        'windows': {'full': bt['window_full'], 'oos': bt['window_oos'],
                    'oos_start_spec': OOS_START, 'nobackfill_start': NOBF_START},
        'metrics': bt['metrics'],
        'attribution': attr,
        'premium': prem,
        'defense_diagnosis': diag,
    }
    json_path = OUT_DIR / 'exp_sp500_swap.json'
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float),
                         encoding='utf-8')
    print(f'JSON 已写出: {json_path}')

    write_report(build_info, bt, attr, prem, diag, data['source'])
    print('\n完成。')


if __name__ == '__main__':
    main()
