#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主力 ETF 份额追踪 — 数据采集脚本

功能：
- 从 config/national_team_etfs.yaml 读取标的清单
- 逐只调用 tushare fund_share 拉取历史份额数据
- 增量模式：已有 CSV 则只追加新增数据
- 数据质量校验：份额>0 / 日期单调 / 单日跳变告警

用法：
    .venv/bin/python scripts/fetch_national_team_share.py
    .venv/bin/python scripts/fetch_national_team_share.py --priority-only
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

# ========== 项目路径 ==========
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

# ========== 配置 ==========
CONFIG_PATH = PROJECT / 'config' / 'national_team_etfs.yaml'
DATA_DIR = PROJECT / 'data' / 'national_team' / 'fund_share'
SLEEP_SEC = 0.35
MAX_RETRY = 3
TODAY = datetime.now().strftime('%Y%m%d')


# ========== Token / API ==========
def _load_env():
    env_file = PROJECT / '.env'
    if env_file.exists():
        for line in open(env_file):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())


def get_api():
    _load_env()
    import tushare as ts
    token = os.environ.get('TUSHARE_TOKEN', '')
    if not token:
        raise RuntimeError('TUSHARE_TOKEN not set')
    return ts.pro_api(token)


# ========== API 调用 (频控+重试) ==========
def api_call(pro, api_name, **kwargs):
    """带频控 + 指数退避重试的 API 调用"""
    fn = getattr(pro, api_name)
    last_err = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            df = fn(**kwargs)
            time.sleep(SLEEP_SEC)
            return df
        except Exception as e:
            last_err = e
            wait = 61.0 if '频率超限' in str(e) else 1.2 * (2 ** (attempt - 1))
            print(f"    ⚠️ {api_name} attempt {attempt} failed: {str(e)[:100]}, retry in {wait:.1f}s", flush=True)
            time.sleep(wait)
    raise last_err


def fetch_all_history(pro, ts_code, start_date='20130101'):
    """翻页拉取全历史 fund_share 数据"""
    frames = []
    cur_end = TODAY
    for _ in range(60):  # 安全上限
        df = api_call(pro, 'fund_share', ts_code=ts_code, start_date=start_date, end_date=cur_end)
        if df is None or len(df) == 0:
            break
        frames.append(df)
        min_date = df['trade_date'].min()
        prev_end = (datetime.strptime(str(min_date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')
        if prev_end >= cur_end:
            break
        cur_end = prev_end
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=['ts_code', 'trade_date'])
    out = out.sort_values('trade_date').reset_index(drop=True)
    return out


# ========== 增量拉取 ==========
def fetch_one(pro, ts_code, name=''):
    """拉取单只 ETF，增量追加模式"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fname = ts_code.replace('.', '_') + '.csv'
    fpath = DATA_DIR / fname

    start_date = '20130101'
    if fpath.exists():
        existing = pd.read_csv(fpath, dtype={'trade_date': str})
        if len(existing) > 0:
            last = existing['trade_date'].max()
            # 从次日开始拉
            start_date = (datetime.strptime(last, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
            if start_date > TODAY:
                return 'skip', f"已是最新 ({last})", fpath
    else:
        existing = pd.DataFrame()

    df = fetch_all_history(pro, ts_code, start_date=start_date)
    if df is None or len(df) == 0:
        if len(existing) > 0:
            return 'skip', f"无新增 (last={existing['trade_date'].max()})", fpath
        return 'fail', "无数据返回", fpath

    # 合并并去重
    if len(existing) > 0:
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['ts_code', 'trade_date'])
        combined = combined.sort_values('trade_date').reset_index(drop=True)
    else:
        combined = df

    combined.to_csv(fpath, index=False)
    new_rows = len(combined) - len(existing)
    return 'ok', f"+{new_rows} rows (total {len(combined)}, {combined['trade_date'].min()}~{combined['trade_date'].max()})", fpath


# ========== 数据质量校验 ==========
def validate_one(fpath, ts_code):
    """单只 ETF 数据质量校验，返回 (warnings: list, stats: dict)"""
    warnings = []
    if not fpath.exists():
        return ['文件不存在'], {}

    df = pd.read_csv(fpath, dtype={'trade_date': str})
    if len(df) == 0:
        return ['空文件'], {}

    stats = {
        'ts_code': ts_code,
        'rows': len(df),
        'start': df['trade_date'].min(),
        'end': df['trade_date'].max(),
    }

    # 份额 > 0
    if 'fd_share' in df.columns:
        neg = (df['fd_share'] <= 0).sum()
        if neg > 0:
            warnings.append(f"份额<=0: {neg} 行")

    # 日期单调不重复
    dates = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    dups = df['trade_date'].duplicated().sum()
    if dups > 0:
        warnings.append(f"日期重复: {dups} 行")
    if not dates.is_monotonic_increasing:
        warnings.append("日期非单调递增")

    # 单日跳变 > 50%
    if 'fd_share' in df.columns and len(df) > 1:
        pct = df['fd_share'].pct_change().abs()
        jumps = pct[pct > 0.5]
        if len(jumps) > 0:
            stats['jump_50pct_days'] = len(jumps)
            warnings.append(f"单日跳变>50%: {len(jumps)} 天")

    # 缺失率：按日历日计算
    total_cal_days = (dates.iloc[-1] - dates.iloc[0]).days + 1
    # 按交易日估算（约 244 天/年）
    expected_trade_days = total_cal_days * 244 / 365
    miss_rate = max(0, 1 - len(df) / expected_trade_days) if expected_trade_days > 0 else 0
    stats['miss_rate'] = round(miss_rate, 3)

    # 低频区段：连续间隔 >7 天
    gaps = dates.diff().dt.days
    low_freq_mask = gaps > 7
    if low_freq_mask.sum() > 0:
        low_freq_start = dates[low_freq_mask].min().strftime('%Y%m%d')
        low_freq_end = dates[low_freq_mask].max().strftime('%Y%m%d')
        stats['low_freq_range'] = f"{low_freq_start}~{low_freq_end}"
        stats['low_freq_gaps'] = int(low_freq_mask.sum())

    return warnings, stats


def validate_all(etf_codes):
    """批量校验，返回覆盖率报告"""
    report = []
    for code in etf_codes:
        fname = code.replace('.', '_') + '.csv'
        fpath = DATA_DIR / fname
        warns, stats = validate_one(fpath, code)
        stats['warnings'] = warns
        report.append(stats)
    return report


# ========== 主逻辑 ==========
def main():
    parser = argparse.ArgumentParser(description='主力ETF份额追踪数据采集')
    parser.add_argument('--priority-only', action='store_true', help='仅拉取 ★ 优先标的')
    parser.add_argument('--validate-only', action='store_true', help='仅做数据质量校验（不拉取）')
    args = parser.parse_args()

    # 读取标的清单
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    all_etfs = cfg['etfs']
    priority_codes = cfg.get('priority_codes', [])

    if args.priority_only:
        targets = [e for e in all_etfs if e['ts_code'] in priority_codes]
        print(f"📋 优先标的模式: {len(targets)} 只")
    else:
        targets = all_etfs
        print(f"📋 全量模式: {len(targets)} 只")

    if args.validate_only:
        codes = [e['ts_code'] for e in targets]
        report = validate_all(codes)
        _print_validation_report(report)
        return 0

    # 拉取
    pro = get_api()
    results = {'ok': 0, 'skip': 0, 'fail': 0}
    for i, etf in enumerate(targets, 1):
        code = etf['ts_code']
        name = etf.get('name', '')
        pri = '★' if etf.get('priority') else ' '
        try:
            status, detail, fpath = fetch_one(pro, code, name)
            results[status] += 1
            icon = {'ok': '✅', 'skip': '⏭️', 'fail': '❌'}[status]
            print(f"  [{i}/{len(targets)}] {pri} {icon} {code} {name}: {detail}", flush=True)
        except Exception as e:
            results['fail'] += 1
            print(f"  [{i}/{len(targets)}] {pri} ❌ {code} {name}: EXCEPTION {e}", flush=True)

    print(f"\n📊 采集统计: ✅成功={results['ok']} ⏭️跳过={results['skip']} ❌失败={results['fail']}")

    # 校验
    codes = [e['ts_code'] for e in targets]
    report = validate_all(codes)
    _print_validation_report(report)

    return 1 if results['fail'] > 0 else 0


def _print_validation_report(report):
    """打印覆盖率报告"""
    print(f"\n{'='*70}")
    print("📈 数据质量与覆盖率报告")
    print(f"{'='*70}")
    print(f"{'ETF':<12} {'行数':>6} {'起始':>10} {'截止':>10} {'缺失率':>6} {'低频段':>14} {'告警'}")
    print("-" * 70)
    for s in report:
        if not s:
            continue
        code = s.get('ts_code', '?')
        rows = s.get('rows', 0)
        start = s.get('start', '-')
        end = s.get('end', '-')
        miss = f"{s.get('miss_rate', 0):.1%}"
        low_f = s.get('low_freq_range', '-')
        warns = '; '.join(s.get('warnings', [])) or '无'
        print(f"{code:<12} {rows:>6} {start:>10} {end:>10} {miss:>6} {low_f:>14} {warns}")
    print(f"{'='*70}")


if __name__ == '__main__':
    sys.exit(main())
