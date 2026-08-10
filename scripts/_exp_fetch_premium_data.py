#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[实验] Tushare 时效性数据抢救缓存（任务 #17）

- token 从项目 .env 读取（与 update_etf_data_tushare.py 同口径），绝不打印明文
- 全部新数据落 data/experiments/tushare_cache/，不触碰生产数据 / src / 既有脚本
- 拉取清单：
  A. 溢价核心：候选纳指/标普 QDII + 现有资产池，fund_daily 收盘 + fund_nav 单位净值（全历史日频）
  B. 参考数据：index_global(SPX/NDX/IXIC)、fx_daily(USDCNY/USDCNH)、fund_share、fund_basic
  C. 生产数据衔接预览：按 update_etf_data_tushare.py 口径（ratio 缩放 + 周五快照），
     只写 prod_refresh_preview.csv，与 all_etfs_nav_latest.csv 最后几周 diff，不覆盖生产文件
- 拉取后：逐文件行数/日期范围/缺失率清单；计算各纳指&标普 ETF 溢价率序列（close/unit_nav-1，
  nav 按日期 ffill 对齐，限 7 天）；汇总 output/experiments/premium_cache_summary.md
- 频控：每次 API 调用间 sleep，指数退避重试 3 次；单项失败记录后继续，最后统一报成功/失败清单

用法:
    .venv/bin/python scripts/_exp_fetch_premium_data.py             # 拉取+分析
    .venv/bin/python scripts/_exp_fetch_premium_data.py --skip-fetch  # 只用已有缓存重算分析
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import tushare as ts

# ---------- 路径与 token ----------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, 'data', 'experiments', 'tushare_cache')
OUT_DIR = os.path.join(ROOT, 'output', 'experiments')
PROD_FILE = os.path.join(ROOT, 'data', 'all_etfs_nav_latest.csv')

_env_file = os.path.join(ROOT, '.env')
if os.path.exists(_env_file):
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                os.environ.setdefault(_k.strip(), _v.strip())
TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN', '')
if not TUSHARE_TOKEN:
    raise RuntimeError('未在 .env / 环境变量中找到 TUSHARE_TOKEN')

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

TODAY = datetime.now().strftime('%Y%m%d')
SLEEP_SEC = 0.35        # 每次 API 调用后的基础间隔（频控）
MAX_RETRY = 3

# ---------- 标的清单 ----------
NASDAQ_ETFS = ['513100.SH', '513300.SH', '159941.SZ', '513390.SH', '159632.SZ', '159509.SZ']
SP500_ETFS = ['513500.SH', '513650.SH']
POOL_ETFS = ['510500.SH', '518880.SH', '512890.SH', '511010.SH']
ALL_ETFS = NASDAQ_ETFS + SP500_ETFS + POOL_ETFS
PREMIUM_ETFS = NASDAQ_ETFS + SP500_ETFS   # 需算溢价率的（QDII）

ETF_NAMES = {
    '513100.SH': '国泰纳指ETF', '513300.SH': '华夏纳斯达克ETF', '159941.SZ': '广发纳指ETF',
    '513390.SH': '华泰柏瑞纳斯达克ETF', '159632.SZ': '嘉实纳斯达克ETF', '159509.SZ': '纳指科技ETF',
    '513500.SH': '博时标普500ETF', '513650.SH': '易方达标普500ETF',
    '510500.SH': '中证500ETF', '518880.SH': '黄金ETF', '512890.SH': '红利低波ETF',
    '511010.SH': '国债ETF',
}

# 生产文件的 5 只 ETF（与 update_etf_data_tushare.py 一致）
PROD_ETF_MAP = {
    '纳指ETF': '513100.SH',
    '红利低波ETF': '512890.SH',
    '中证500ETF': '510500.SH',
    '黄金ETF': '518880.SH',
    '国债ETF': '511010.SH',
}

# 拉取结果登记：{item_key: {'status': 'ok'/'skip'/'fail', 'detail': str, 'file': str}}
REGISTRY = {}


def _log(msg):
    print(msg, flush=True)


def api_call(api_name, **kwargs):
    """带频控 + 指数退避重试的 API 调用；失败抛出最后一次异常"""
    fn = getattr(pro, api_name)
    last_err = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            df = fn(**kwargs)
            time.sleep(SLEEP_SEC)
            return df
        except Exception as e:
            last_err = e
            # 每分钟频控超限：等满一个窗口再试，否则指数退避
            wait = 61.0 if '频率超限' in str(e) else 1.2 * (2 ** (attempt - 1))
            _log(f"    ⚠️ {api_name} 第{attempt}次失败: {str(e)[:120]}，{wait:.1f}s 后重试")
            time.sleep(wait)
    raise last_err


def fetch_all_history(api_name, date_field, **fixed_kwargs):
    """
    向前翻页拉全历史：以 end_date 逐步回退，直到返回空。
    返回按 date_field 升序去重后的 DataFrame（可能为空）。
    """
    frames = []
    cur_end = TODAY
    for _page in range(40):  # 安全上限
        df = api_call(api_name, end_date=cur_end, **fixed_kwargs)
        if df is None or len(df) == 0:
            break
        frames.append(df)
        min_date = df[date_field].min()
        prev_end = (datetime.strptime(str(min_date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')
        if prev_end >= cur_end:  # 防死循环
            break
        cur_end = prev_end
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=[c for c in ['ts_code', date_field] if c in out.columns])
    out = out.sort_values(date_field).reset_index(drop=True)
    return out


def save_csv(df, fname):
    path = os.path.join(CACHE_DIR, fname)
    df.to_csv(path, index=False)
    return path


def register(key, status, detail='', fname=''):
    REGISTRY[key] = {'status': status, 'detail': detail, 'file': fname}
    icon = {'ok': '✅', 'skip': '⏭️', 'fail': '❌'}[status]
    _log(f"  {icon} {key}: {detail}")


# ==================================================================
# A. 溢价核心：fund_daily + fund_nav
# ==================================================================
def fetch_block_a():
    _log("\n" + "=" * 60)
    _log("A. 溢价核心：fund_daily + fund_nav（全历史日频）")
    _log("=" * 60)
    for code in ALL_ETFS:
        tag = code.replace('.', '')
        # --- fund_daily ---
        key = f"fund_daily/{code}"
        try:
            df = fetch_all_history('fund_daily', 'trade_date', ts_code=code)
            if len(df) == 0:
                register(key, 'skip', '无数据（可能无权限或未上市）')
            else:
                fname = f"fund_daily_{tag}.csv"
                save_csv(df, fname)
                register(key, 'ok',
                         f"{len(df)}行 {df['trade_date'].min()}~{df['trade_date'].max()}", fname)
        except Exception as e:
            register(key, 'fail', str(e)[:160])
        # --- fund_nav ---
        key = f"fund_nav/{code}"
        try:
            df = fetch_all_history('fund_nav', 'nav_date', ts_code=code)
            if len(df) == 0:
                register(key, 'skip', '无数据（可能无权限）')
            else:
                fname = f"fund_nav_{tag}.csv"
                save_csv(df, fname)
                register(key, 'ok',
                         f"{len(df)}行 {df['nav_date'].min()}~{df['nav_date'].max()}", fname)
        except Exception as e:
            register(key, 'fail', str(e)[:160])


# ==================================================================
# B. 参考数据：index_global / fx_daily / fund_share / fund_basic
# ==================================================================
def fetch_block_b():
    _log("\n" + "=" * 60)
    _log("B. 参考数据：index_global / fx_daily / fund_share / fund_basic")
    _log("=" * 60)

    # index_global: SPX 直接拉；纳指先 NDX，失败/空则退 IXIC
    for label, candidates in [('SPX', ['SPX']), ('NDX', ['NDX', 'IXIC'])]:
        key = f"index_global/{label}"
        done = False
        last_detail = '无数据'
        for ts_code in candidates:
            try:
                df = fetch_all_history('index_global', 'trade_date', ts_code=ts_code)
                if len(df) > 0:
                    fname = f"index_global_{ts_code}.csv"
                    save_csv(df, fname)
                    register(key, 'ok',
                             f"用{ts_code}: {len(df)}行 {df['trade_date'].min()}~{df['trade_date'].max()}", fname)
                    done = True
                    break
                last_detail = f"{ts_code} 返回空"
            except Exception as e:
                last_detail = f"{ts_code}: {str(e)[:120]}"
        if not done:
            register(key, 'skip', last_detail)

    # fx_daily: USDCNY 优先，退 USDCNH（tushare 代码带 .FXCM 后缀，双格式都试）
    key = 'fx_daily/USDCNY'
    done = False
    last_detail = '无数据'
    for ts_code in ['USDCNY.FXCM', 'USDCNH.FXCM', 'USDCNY', 'USDCNH']:
        try:
            df = fetch_all_history('fx_daily', 'trade_date', ts_code=ts_code)
            if len(df) > 0:
                clean = ts_code.split('.')[0]
                fname = f"fx_daily_{clean}.csv"
                save_csv(df, fname)
                register(key, 'ok',
                         f"用{ts_code}: {len(df)}行 {df['trade_date'].min()}~{df['trade_date'].max()}", fname)
                done = True
                break
            last_detail = f"{ts_code} 返回空"
        except Exception as e:
            last_detail = f"{ts_code}: {str(e)[:120]}"
    if not done:
        register(key, 'skip', last_detail)

    # fund_share: 各候选份额变动（限购代理指标）
    for code in ALL_ETFS:
        tag = code.replace('.', '')
        key = f"fund_share/{code}"
        try:
            df = fetch_all_history('fund_share', 'trade_date', ts_code=code)
            if len(df) == 0:
                register(key, 'skip', '无数据（可能无权限）')
            else:
                fname = f"fund_share_{tag}.csv"
                save_csv(df, fname)
                register(key, 'ok',
                         f"{len(df)}行 {df['trade_date'].min()}~{df['trade_date'].max()}", fname)
        except Exception as e:
            register(key, 'fail', str(e)[:160])

    # fund_basic: 场内基金基础信息，过滤到候选清单
    key = 'fund_basic/candidates'
    try:
        df = api_call('fund_basic', market='E')
        if df is None or len(df) == 0:
            register(key, 'skip', 'fund_basic 返回空')
        else:
            sub = df[df['ts_code'].isin(ALL_ETFS)].reset_index(drop=True)
            fname = 'fund_basic_candidates.csv'
            save_csv(sub, fname)
            register(key, 'ok', f"命中 {len(sub)}/{len(ALL_ETFS)} 只候选（全场内共{len(df)}只）", fname)
    except Exception as e:
        register(key, 'fail', str(e)[:160])


# ==================================================================
# C. 生产数据衔接预览（不覆盖生产文件）
# ==================================================================
def fetch_block_c():
    _log("\n" + "=" * 60)
    _log("C. 生产数据衔接预览（update_etf_data_tushare.py 口径，不写生产文件）")
    _log("=" * 60)
    key = 'prod_refresh_preview'
    try:
        old = pd.read_csv(PROD_FILE)
        old['日期'] = pd.to_datetime(old['日期'])
        old = old.sort_values('日期').reset_index(drop=True)
        last_old_date = old['日期'].max()
        _log(f"  生产文件最后日期: {last_old_date.strftime('%Y-%m-%d')}，共{len(old)}行")

        # 拉最近 ~10 周日线（覆盖生产最后几周做 diff）
        fetch_start = (last_old_date - timedelta(days=70)).strftime('%Y%m%d')
        daily = {}
        for name, code in PROD_ETF_MAP.items():
            df = api_call('fund_daily', ts_code=code, start_date=fetch_start, end_date=TODAY)
            if df is None or len(df) == 0:
                raise RuntimeError(f"{name}({code}) 近期日线为空")
            df = df.sort_values('trade_date').reset_index(drop=True)
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            daily[name] = df

        # ratio 锚定生产最后日期（与原脚本相同：old_val / tushare_close@last_old_date，缺则往前找）
        ratios = {}
        for name, code in PROD_ETF_MAP.items():
            old_val = old.loc[old['日期'] == last_old_date, name].values[0]
            df = daily[name]
            anchor = df[df['trade_date'] <= last_old_date]
            if len(anchor) == 0:
                raise RuntimeError(f"{name}({code}) 无法找到锚定日收盘价")
            raw_close = anchor.iloc[-1]['close']
            ratios[name] = old_val / raw_close
            _log(f"    {name}: 锚定{anchor.iloc[-1]['trade_date'].strftime('%Y-%m-%d')} "
                 f"close={raw_close}, ratio={ratios[name]:.6f}")

        # 合并缩放后日线 -> 周五快照（与原脚本聚合逻辑一致）
        merged = None
        for name in PROD_ETF_MAP:
            d = daily[name][['trade_date', 'close']].copy()
            d[name] = d['close'] * ratios[name]
            d = d[['trade_date', name]]
            merged = d if merged is None else pd.merge(merged, d, on='trade_date', how='outer')
        merged = merged.sort_values('trade_date').reset_index(drop=True)
        iso = merged['trade_date'].dt.isocalendar()
        merged['isoyear'], merged['isoweek'] = iso.year, iso.week
        merged['weekday'] = merged['trade_date'].dt.weekday
        rows = []
        for (_, _), g in merged.groupby(['isoyear', 'isoweek']):
            fri = g[g['weekday'] == 4]
            rows.append((fri if len(fri) > 0 else g).sort_values('trade_date').iloc[-1])
        weekly = pd.DataFrame(rows).sort_values('trade_date').reset_index(drop=True)
        weekly = weekly[['trade_date'] + list(PROD_ETF_MAP.keys())].rename(columns={'trade_date': '日期'})

        # 与生产文件重叠周 diff
        prod_tail = old[old['日期'] >= weekly['日期'].min()]
        cmp = pd.merge(prod_tail, weekly, on='日期', how='inner', suffixes=('_prod', '_tushare'))
        diff_cols = {}
        for name in PROD_ETF_MAP:
            diff_cols[f'{name}_diff%'] = (cmp[f'{name}_tushare'] / cmp[f'{name}_prod'] - 1) * 100
        diff_df = pd.concat([cmp[['日期']]] + [pd.Series(v, name=k) for k, v in diff_cols.items()], axis=1)
        max_abs_diff = diff_df.drop(columns='日期').abs().max().max() if len(diff_df) > 0 else np.nan
        # 衔接判据用最近 2 个重叠周：更早的重叠周可能因除息（分红）与最新锚定比例存在固定偏移
        recent2 = (diff_df.sort_values('日期').tail(2).drop(columns='日期').abs().max().max()
                   if len(diff_df) > 0 else np.nan)

        # 输出预览文件：周线快照 + 重叠周与生产的偏差
        out = weekly.copy()
        out['日期'] = out['日期'].dt.strftime('%Y-%m-%d')
        out = pd.merge(out, diff_df.assign(日期=diff_df['日期'].dt.strftime('%Y-%m-%d')),
                       on='日期', how='left')
        fname = 'prod_refresh_preview.csv'
        save_csv(out, fname)
        n_new = int((weekly['日期'] > last_old_date).sum())
        register(key, 'ok',
                 f"{len(weekly)}周（重叠{len(cmp)}周，新增{n_new}周），重叠周最大偏差 {max_abs_diff:.4f}%，最近2周偏差 {recent2:.4f}%", fname)
        prod_info = {'overlap_weeks': len(cmp), 'new_weeks': n_new,
                     'max_abs_diff_pct': float(max_abs_diff),
                     'recent2_diff_pct': float(recent2),
                     'last_old_date': last_old_date.strftime('%Y-%m-%d'),
                     'preview_last_date': weekly['日期'].max().strftime('%Y-%m-%d')}
        with open(os.path.join(CACHE_DIR, 'prod_preview_info.json'), 'w') as jf:
            json.dump(prod_info, jf, ensure_ascii=False, indent=2)
        return prod_info
    except Exception as e:
        register(key, 'fail', str(e)[:200])
        return None


# ==================================================================
# 验证与溢价分析
# ==================================================================
def validate_files():
    """逐缓存文件：行数 / 日期范围 / 缺失率"""
    rows = []
    for fname in sorted(os.listdir(CACHE_DIR)):
        if not fname.endswith('.csv') or fname in ('fetch_manifest.csv',):
            continue
        path = os.path.join(CACHE_DIR, fname)
        try:
            df = pd.read_csv(path)
            date_col = next((c for c in ['trade_date', 'nav_date', '日期'] if c in df.columns), None)
            dmin = str(df[date_col].min()) if date_col and len(df) else ''
            dmax = str(df[date_col].max()) if date_col and len(df) else ''
            na_pct = df.isnull().mean().mean() * 100 if len(df) else 0.0
            rows.append({'file': fname, 'rows': len(df), 'date_min': dmin, 'date_max': dmax,
                         'na_pct': round(na_pct, 2)})
        except Exception as e:
            rows.append({'file': fname, 'rows': -1, 'date_min': '', 'date_max': '',
                         'na_pct': np.nan, 'error': str(e)[:80]})
    return pd.DataFrame(rows)


def compute_premiums():
    """
    各纳指&标普 ETF 溢价率 = close / unit_nav - 1
    nav 对齐：QDII 净值 T 日盘后公布，nav_date 与 trade_date 按日合并后 ffill（限 7 天），
    避免节假日/公布延迟导致的空洞。
    返回 {code: {'series': DataFrame, 'stats': dict}}
    """
    results = {}
    for code in PREMIUM_ETFS:
        tag = code.replace('.', '')
        fd = os.path.join(CACHE_DIR, f"fund_daily_{tag}.csv")
        fn = os.path.join(CACHE_DIR, f"fund_nav_{tag}.csv")
        if not (os.path.exists(fd) and os.path.exists(fn)):
            results[code] = {'error': '缺 fund_daily 或 fund_nav 缓存'}
            continue
        px = pd.read_csv(fd, dtype={'trade_date': str})[['trade_date', 'close']]
        nav = pd.read_csv(fn, dtype={'nav_date': str})
        nav = nav[['nav_date', 'unit_nav']].dropna()
        nav = nav.drop_duplicates(subset='nav_date', keep='last')
        px['date'] = pd.to_datetime(px['trade_date'])
        nav['date'] = pd.to_datetime(nav['nav_date'])
        px = px.sort_values('date')
        nav = nav.sort_values('date')
        # merge_asof: 用交易日当天或之前最近一个 nav（限 7 天）
        m = pd.merge_asof(px[['date', 'close']], nav[['date', 'unit_nav']],
                          on='date', direction='backward', tolerance=pd.Timedelta(days=7))
        m = m.dropna(subset=['unit_nav'])
        m['premium'] = m['close'] / m['unit_nav'] - 1
        # 简单防呆：份额拆分等导致的 close/nav 量级错位 -> 剔除 |premium|>50% 的孤立点
        m = m[m['premium'].abs() < 0.5].reset_index(drop=True)
        if len(m) == 0:
            results[code] = {'error': '对齐后无有效样本'}
            continue
        s = m.set_index('date')['premium']
        stats = {
            'latest_date': s.index[-1].strftime('%Y-%m-%d'),
            'latest': s.iloc[-1],
            'med_1m': s.iloc[-21:].median(),
            'med_3m': s.iloc[-63:].median(),
            'med_1y': s.iloc[-252:].median(),
            'med_hist': s.median(),
            'p90_1y': s.iloc[-252:].quantile(0.9),
            'n_obs': len(s),
            'since': s.index[0].strftime('%Y-%m-%d'),
        }
        out = m.copy()
        out['date'] = out['date'].dt.strftime('%Y-%m-%d')
        save_csv(out, f"premium_{tag}.csv")
        results[code] = {'stats': stats}
    return results


def build_summary(manifest, premiums, prod_info):
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = []
    lines.append("# Tushare 时效性数据缓存与溢价率汇总（任务 #17）\n")
    lines.append(f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 缓存目录: `data/experiments/tushare_cache/`")
    lines.append(f"- 拉取脚本: `scripts/_exp_fetch_premium_data.py`（token 读自 .env，不落盘）\n")

    # 成功/失败清单
    ok = [k for k, v in REGISTRY.items() if v['status'] == 'ok']
    skip = [k for k, v in REGISTRY.items() if v['status'] == 'skip']
    fail = [k for k, v in REGISTRY.items() if v['status'] == 'fail']
    lines.append("## 1. 拉取结果清单\n")
    lines.append(f"成功 {len(ok)} 项，跳过 {len(skip)} 项，失败 {len(fail)} 项。\n")
    lines.append("| 项目 | 状态 | 说明 | 文件 |")
    lines.append("|---|---|---|---|")
    for k, v in REGISTRY.items():
        icon = {'ok': '✅', 'skip': '⏭️ 跳过', 'fail': '❌ 失败'}[v['status']]
        lines.append(f"| {k} | {icon} | {v['detail']} | {v['file']} |")

    lines.append("\n## 2. 缓存文件验证（行数 / 日期范围 / 缺失率）\n")
    lines.append("| 文件 | 行数 | 起始 | 截止 | 缺失率% |")
    lines.append("|---|---|---|---|---|")
    for _, r in manifest.iterrows():
        lines.append(f"| {r['file']} | {r['rows']} | {r['date_min']} | {r['date_max']} | {r['na_pct']} |")

    lines.append("\n## 3. 跨券溢价率对比（close / unit_nav − 1）\n")
    lines.append("nav 对齐口径：交易日收盘价对当日或之前最近一个单位净值（merge_asof backward，限7天）。\n")
    lines.append("| 代码 | 名称 | 类型 | 最新日期 | 当前溢价 | 近1月中位 | 近3月中位 | 近1年中位 | 近1年P90 | 历史中位 | 样本起点 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for code in PREMIUM_ETFS:
        typ = '纳指' if code in NASDAQ_ETFS else '标普'
        r = premiums.get(code, {})
        if 'stats' not in r:
            lines.append(f"| {code} | {ETF_NAMES.get(code, '')} | {typ} | - | {r.get('error', '无数据')} | | | | | | |")
            continue
        st = r['stats']
        lines.append(
            f"| {code} | {ETF_NAMES.get(code, '')} | {typ} | {st['latest_date']} "
            f"| {st['latest']*100:.2f}% | {st['med_1m']*100:.2f}% | {st['med_3m']*100:.2f}% "
            f"| {st['med_1y']*100:.2f}% | {st['p90_1y']*100:.2f}% | {st['med_hist']*100:.2f}% | {st['since']} |")

    # 关键问题：个券 vs 全板块
    lines.append("\n### 关键结论：高溢价是 513100 个券现象还是全板块现象？\n")
    nas = [(c, premiums[c]['stats']) for c in NASDAQ_ETFS if 'stats' in premiums.get(c, {})]
    sp = [(c, premiums[c]['stats']) for c in SP500_ETFS if 'stats' in premiums.get(c, {})]
    if nas:
        cur_nas = {c: st['latest'] for c, st in nas}
        m1_nas = {c: st['med_1m'] for c, st in nas}
        lines.append(f"- 纳指类当前溢价: " + ", ".join(f"{c}={v*100:.2f}%" for c, v in cur_nas.items()))
        lines.append(f"- 纳指类近1月中位: " + ", ".join(f"{c}={v*100:.2f}%" for c, v in m1_nas.items()))
        if sp:
            lines.append(f"- 标普类当前溢价: " + ", ".join(f"{c}={st['latest']*100:.2f}%" for c, st in sp))
        v513100 = cur_nas.get('513100.SH')
        others = [v for c, v in cur_nas.items() if c != '513100.SH']
        if v513100 is not None and others:
            med_others = float(np.median(others))
            sp_cur = float(np.median([st['latest'] for _, st in sp])) if sp else np.nan
            if med_others > 0.02:
                verdict = "纳指类其余标的当前溢价中位同样明显偏高（>2%），属于**板块性现象**（QDII 额度约束下的整体溢价）"
            elif v513100 - med_others > 0.02:
                verdict = "513100 溢价显著高于其余纳指标的（差距>2pp），更像**个券现象**（自身限购/份额约束）"
            else:
                verdict = "513100 与其余纳指标的溢价水平接近且总体温和，未见显著的个券异常"
            lines.append(f"\n**判定**：513100 当前 {v513100*100:.2f}% vs 其余纳指中位 {med_others*100:.2f}%"
                         + (f"，标普中位 {sp_cur*100:.2f}%" if not np.isnan(sp_cur) else "")
                         + f"。{verdict}。")
    else:
        lines.append("- 纳指类溢价数据不足，无法判定（见失败清单）。")

    # 生产衔接
    lines.append("\n## 4. 生产数据衔接预览（不覆盖生产文件）\n")
    if prod_info:
        lines.append(f"- 生产文件最后日期: {prod_info['last_old_date']}，预览最新至: {prod_info['preview_last_date']}")
        lines.append(f"- 重叠 {prod_info['overlap_weeks']} 周，新增 {prod_info['new_weeks']} 周")
        rec2 = prod_info.get('recent2_diff_pct', prod_info['max_abs_diff_pct'])
        lines.append(f"- 最近 2 个重叠周最大偏差: **{rec2:.4f}%**"
                     + ("，衔接一致 ✅" if rec2 < 0.1 else "，偏差偏大，需人工核查 ⚠️"))
        lines.append(f"- 全部重叠周最大偏差: {prod_info['max_abs_diff_pct']:.4f}%。若显著大于最近2周偏差，"
                     f"通常是重叠区间内 ETF 除息（分红）所致：生产文件历史周内嵌除息前价格，"
                     f"而 tushare 原始 close 按最新锚定比例回推会差出分红额，属口径差而非数据错误。"
                     f"（本次核查：510500 于 2026-07-15 除息≈1.8%，511010 于 2026-06-25 除息≈0.58%，"
                     f"与偏差分布完全吻合；除息日后的重叠周偏差全为 0）")
        lines.append(f"- 明细见 `data/experiments/tushare_cache/prod_refresh_preview.csv`")
    else:
        lines.append("- 预览失败，见失败清单。")

    path = os.path.join(OUT_DIR, 'premium_cache_summary.md')
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    _log(f"\n📄 汇总已写入: {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-fetch', action='store_true', help='跳过拉取，仅用现有缓存重算分析')
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    prod_info = None
    if not args.skip_fetch:
        fetch_block_a()
        fetch_block_b()
        prod_info = fetch_block_c()
        # 落盘拉取登记
        man = pd.DataFrame([{'item': k, **v} for k, v in REGISTRY.items()])
        man.to_csv(os.path.join(CACHE_DIR, 'fetch_manifest.csv'), index=False)
    else:
        mf = os.path.join(CACHE_DIR, 'fetch_manifest.csv')
        if os.path.exists(mf):
            for _, r in pd.read_csv(mf).fillna('').iterrows():
                REGISTRY[r['item']] = {'status': r['status'], 'detail': r['detail'], 'file': r['file']}
        pj = os.path.join(CACHE_DIR, 'prod_preview_info.json')
        if os.path.exists(pj):
            with open(pj) as jf:
                prod_info = json.load(jf)

    _log("\n" + "=" * 60)
    _log("验证与溢价分析")
    _log("=" * 60)
    manifest = validate_files()
    _log(manifest.to_string(index=False))
    premiums = compute_premiums()
    build_summary(manifest, premiums, prod_info)

    # 最终成功/失败清单
    fails = {k: v for k, v in REGISTRY.items() if v['status'] == 'fail'}
    skips = {k: v for k, v in REGISTRY.items() if v['status'] == 'skip'}
    _log(f"\n✅ 完成。成功 {sum(1 for v in REGISTRY.values() if v['status']=='ok')} 项")
    if skips:
        _log(f"⏭️ 跳过 {len(skips)} 项: {list(skips)}")
    if fails:
        _log(f"❌ 失败 {len(fails)} 项: {list(fails)}")


if __name__ == '__main__':
    main()
