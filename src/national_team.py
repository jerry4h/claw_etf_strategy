# -*- coding: utf-8 -*-
"""
主力 ETF 仓位建模管线（任务 #76）

Layer A: 单 ETF 信号因子（份额增幅 / expanding 百分位 / peak ratio / 异常检测）
Layer B: 按跟踪指数聚合（同指数多 ETF 合并）
Layer C: 全市场聚合（板块暴露 / rotation heat / direction sign / 集中度）

产出：
  - output/national_team/position_model.json  全市场仓位快照
  - data/national_team/events.csv             疑似主力介入事件流
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

# ========== 常量 ==========
PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT / 'data' / 'national_team'
DEFAULT_CONFIG = PROJECT / 'config' / 'national_team_etfs.yaml'
OUTPUT_DIR = PROJECT / 'output' / 'national_team'

# 板块分类映射（用于 Layer C）
SECTOR_MAP = {
    '大盘核心': ['沪深300', '上证50'],
    '中盘': ['中证500'],
    '小盘': ['中证1000', '中证2000'],
    '科创': ['科创'],
    '创业板': ['创业板'],
    '宽基综合': ['中证A', '深证', 'MSCI', '上证综指'],
}

# 异常检测阈值
ANOMALY_ABS_THRESHOLD = 0.10  # g20 > 10% 即为绝对异常（罕见，924 级别事件）
ANOMALY_PCTILE = 0.95  # expanding 百分位阈值（相对异常）


# ========== P1.1 规模拆解 ==========

def compute_aum_decomposition(share_df: pd.DataFrame, nav_series: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    AUM_t = S_t × NAV_t / 10000 (亿元)
    净申赎贡献 = NAV_{t-1} × (S_t - S_{t-1}) / 10000
    价格贡献 = S_t × (NAV_t - NAV_{t-1}) / 10000

    若 nav_series 缺失，仅返回 share 基础列（不做 AUM 拆解）。
    """
    df = share_df[['trade_date', 'fd_share']].copy()
    df = df.sort_values('trade_date').reset_index(drop=True)

    if nav_series is not None and len(nav_series) > 0:
        # 对齐日期
        nav_aligned = nav_series.reindex(df['trade_date'])
        df['nav'] = nav_aligned.values
        df['aum'] = df['fd_share'] * df['nav'] / 10000  # 亿元
        df['nav_prev'] = df['nav'].shift(1)
        df['share_prev'] = df['fd_share'].shift(1)
        df['net_flow'] = df['nav_prev'] * (df['fd_share'] - df['share_prev']) / 10000
        df['price_effect'] = df['fd_share'] * (df['nav'] - df['nav_prev']) / 10000
    else:
        df['aum'] = np.nan
        df['net_flow'] = np.nan
        df['price_effect'] = np.nan

    return df


# ========== P1.2 Layer A: 单 ETF 信号因子 ==========

def compute_etf_signals(
    share_series: pd.Series,
    dates: pd.Series,
    index_return_series: Optional[pd.Series] = None,
    window_short: int = 5,
    window_mid: int = 20,
    window_long: int = 60,
) -> pd.DataFrame:
    """
    返回 DataFrame 含列：
    - share_growth_5d/20d/60d: 滚动窗口份额增幅 (S_t/S_{t-N} - 1)
    - hist_pctile: expanding 百分位排名（share_growth_20d 在自身历史中的排名，无前视）
    - peak_ratio: 当前份额 / 历史峰值（expanding max）
    - contra_trend_flag: 份额增（growth_5d > 0）AND 跟踪指数跌（index_return < 0）
    - is_anomaly: share_growth_20d 超绝对阈值 OR 超 expanding p95
    """
    s = share_series.values.astype(float)
    n = len(s)

    result = pd.DataFrame({'trade_date': dates.values})

    # 滚动增幅
    for name, w in [('share_growth_5d', window_short), ('share_growth_20d', window_mid), ('share_growth_60d', window_long)]:
        g = np.full(n, np.nan)
        for i in range(w, n):
            if s[i - w] > 0:
                g[i] = s[i] / s[i - w] - 1
        result[name] = g

    # expanding 百分位排名（share_growth_20d 在自身历史中的位置）
    g20 = result['share_growth_20d'].values
    pctile = np.full(n, np.nan)
    for i in range(window_long, n):
        history = g20[:i + 1]
        valid = history[~np.isnan(history)]
        if len(valid) >= 30:
            pctile[i] = (valid < g20[i]).sum() / len(valid)
    result['hist_pctile'] = pctile

    # peak_ratio: 当前 / expanding max
    expanding_max = np.maximum.accumulate(np.where(np.isnan(s), 0, s))
    expanding_max[expanding_max == 0] = np.nan
    result['peak_ratio'] = s / expanding_max

    # contra_trend_flag
    if index_return_series is not None and len(index_return_series) == n:
        idx_ret = index_return_series.values
        g5 = result['share_growth_5d'].values
        result['contra_trend_flag'] = (g5 > 0) & (idx_ret < 0)
    else:
        result['contra_trend_flag'] = False

    # is_anomaly: expanding p95 OR absolute threshold
    expanding_p95 = np.full(n, np.nan)
    for i in range(window_long, n):
        valid = g20[:i + 1]
        valid = valid[~np.isnan(valid)]
        if len(valid) >= 30:
            expanding_p95[i] = np.percentile(valid, ANOMALY_PCTILE * 100)
    result['expanding_p95'] = expanding_p95
    result['is_anomaly'] = (
        (result['share_growth_20d'] > ANOMALY_ABS_THRESHOLD) |
        ((~np.isnan(expanding_p95)) & (result['share_growth_20d'] > expanding_p95))
    )

    return result


# ========== P1.3 Layer B: 指数级聚合 ==========

def _classify_etf_index(benchmark: str) -> str:
    """从 benchmark 文本推断跟踪指数分类"""
    bm = benchmark or ''
    if '沪深300' in bm:
        return '沪深300'
    elif '上证50' in bm or '50成份' in bm:
        return '上证50'
    elif '中证500' in bm or '小盘500' in bm:
        return '中证500'
    elif '中证1000' in bm:
        return '中证1000'
    elif '中证2000' in bm:
        return '中证2000'
    elif '创业板' in bm:
        return '创业板'
    elif '科创' in bm:
        return '科创'
    elif '中证A' in bm:
        return '中证A'
    elif '深证' in bm:
        return '深证'
    elif 'MSCI' in bm:
        return 'MSCI'
    elif '上证综指' in bm:
        return '上证综指'
    return 'other'


def aggregate_by_index(etf_signals_dict: dict, etf_master: list) -> dict:
    """
    按跟踪指数分组：
    - 合并同指数多只 ETF 的份额为"指数总份额"
    - 输出：每个宽基指数的 share_growth + 综合信号

    Returns: {index_name: DataFrame with aggregated signals}
    """
    # 按指数分组
    index_groups = {}
    for etf in etf_master:
        code = etf['ts_code']
        idx = _classify_etf_index(etf.get('benchmark', ''))
        if idx not in index_groups:
            index_groups[idx] = []
        index_groups[idx].append(code)

    result = {}
    for idx_name, codes in index_groups.items():
        # 收集有数据的 ETF
        available = [c for c in codes if c in etf_signals_dict]
        if not available:
            continue

        # 以第一个有数据的 ETF 为时间轴基准
        base = etf_signals_dict[available[0]].copy()
        base = base.rename(columns={'share_growth_20d': 'agg_growth_20d'})

        if len(available) > 1:
            # 多只 ETF: 取 share_growth_20d 的等权均值
            g20_frames = []
            for c in available:
                sig = etf_signals_dict[c]
                g20_frames.append(sig.set_index('trade_date')['share_growth_20d'].rename(c))
            merged = pd.concat(g20_frames, axis=1)
            agg = merged.mean(axis=1, skipna=True)
            base = base.set_index('trade_date')
            base['agg_growth_20d'] = agg
            base = base.reset_index()

        base['index_name'] = idx_name
        base['n_etfs'] = len(available)
        result[idx_name] = base

    return result


# ========== P1.4 Layer C: 全市场聚合 ==========

def _sector_for_index(idx_name: str) -> str:
    """将指数名映射到板块"""
    for sector, keywords in SECTOR_MAP.items():
        for kw in keywords:
            if kw in idx_name:
                return sector
    return '宽基综合'


def compute_market_overview(index_signals: dict) -> dict:
    """
    - 板块暴露向量：大盘核心/中盘/小盘/科创/创业板
    - rotation_heat: 本周被增持最多的板块
    - direction_sign: Σ 净申赎方向
    - concentration_index (HHI)
    """
    # 取各指数最新一行的 agg_growth_20d
    latest = {}
    for idx_name, df in index_signals.items():
        if len(df) == 0:
            continue
        row = df.iloc[-1]
        g20 = row.get('agg_growth_20d', np.nan)
        if pd.isna(g20):
            continue
        latest[idx_name] = g20

    if not latest:
        return {'error': '无可用信号数据'}

    # 按板块聚合
    sector_growth = {}
    for idx_name, g in latest.items():
        sector = _sector_for_index(idx_name)
        if sector not in sector_growth:
            sector_growth[sector] = []
        sector_growth[sector].append(g)

    sector_avg = {s: float(np.mean(gs)) for s, gs in sector_growth.items()}

    # rotation_heat: 最大增幅板块
    if sector_avg:
        rotation_heat = max(sector_avg, key=sector_avg.get)
    else:
        rotation_heat = 'N/A'

    # direction_sign: 总体方向
    all_g = list(latest.values())
    total_g = sum(all_g)
    direction_sign = '入场' if total_g > 0 else '兑现' if total_g < 0 else '持平'

    # HHI 集中度
    if sector_avg and sum(abs(v) for v in sector_avg.values()) > 0:
        abs_vals = np.array([abs(v) for v in sector_avg.values()])
        shares = abs_vals / abs_vals.sum() if abs_vals.sum() > 0 else abs_vals
        hhi = float((shares ** 2).sum())
    else:
        hhi = 0.0

    # 最新日期
    max_date = ''
    for df in index_signals.values():
        if len(df) > 0:
            d = str(df.iloc[-1]['trade_date'])
            if d > max_date:
                max_date = d

    return {
        'date': max_date,
        'sector_exposure': sector_avg,
        'rotation_heat': rotation_heat,
        'direction_sign': direction_sign,
        'concentration_hhi': round(hhi, 4),
        'index_growth_20d': {k: round(v, 6) for k, v in latest.items()},
        'n_indices_active': len(latest),
    }


# ========== P1.5 主入口 ==========

def _load_share_data(data_dir: Path) -> dict:
    """加载所有份额 CSV，返回 {ts_code: DataFrame}"""
    share_dir = data_dir / 'fund_share'
    if not share_dir.exists():
        return {}
    result = {}
    for f in share_dir.glob('*.csv'):
        df = pd.read_csv(f, dtype={'trade_date': str})
        if 'ts_code' not in df.columns or 'fd_share' not in df.columns:
            continue
        df = df.sort_values('trade_date').reset_index(drop=True)
        code = df['ts_code'].iloc[0] if len(df) > 0 else f.stem.replace('_', '.')
        result[code] = df
    return result


def _load_nav_data(ts_code: str, data_dir: Path) -> Optional[pd.Series]:
    """尝试从 tushare_cache 加载 close 序列作为 NAV 代理"""
    cache_dir = PROJECT / 'data' / 'experiments' / 'tushare_cache'
    # 映射 ts_code → cache 文件名
    tag = ts_code.replace('.', '')
    fname = f'fund_daily_{tag}.csv'
    fpath = cache_dir / fname
    if fpath.exists():
        df = pd.read_csv(fpath, dtype={'trade_date': str})
        df = df.sort_values('trade_date').reset_index(drop=True)
        return pd.Series(df['close'].values, index=df['trade_date'].values, name='nav')
    return None


def build_position_model(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    config_path: str | Path = DEFAULT_CONFIG,
    verbose: bool = True,
) -> dict:
    """
    主函数：读数据 → Layer A → Layer B → Layer C → 输出 JSON + events CSV
    返回 Layer C 市场概览 dict。
    """
    data_dir = Path(data_dir)
    config_path = Path(config_path)

    # 加载配置
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    etf_master = cfg['etfs']

    # 加载份额数据
    share_data = _load_share_data(data_dir)
    if verbose:
        print(f"已加载 {len(share_data)} 只 ETF 份额数据")

    # Layer A: 逐 ETF 计算信号
    etf_signals = {}
    events_rows = []

    for etf in etf_master:
        code = etf['ts_code']
        if code not in share_data:
            continue
        df = share_data[code]
        if len(df) < 30:
            continue

        # NAV（可选）
        nav = _load_nav_data(code, data_dir)

        # AUM 拆解
        aum_df = compute_aum_decomposition(df, nav)

        # 信号计算
        signals = compute_etf_signals(
            share_series=df['fd_share'],
            dates=df['trade_date'],
            index_return_series=None,  # P2/P3 补充
        )
        etf_signals[code] = signals

        # 提取异常事件
        anomalies = signals[signals['is_anomaly']].copy()
        for _, row in anomalies.iterrows():
            events_rows.append({
                'date': row['trade_date'],
                'index': _classify_etf_index(etf.get('benchmark', '')),
                'etf': code,
                'trigger_type': 'is_anomaly',
                'share_growth_20d': round(float(row['share_growth_20d']), 4) if not pd.isna(row['share_growth_20d']) else None,
                'hist_pctile': round(float(row['hist_pctile']), 4) if not pd.isna(row['hist_pctile']) else None,
                'index_return': None,  # 待补充
            })

    if verbose:
        print(f"Layer A 完成: {len(etf_signals)} 只 ETF 产出信号, {len(events_rows)} 个异常事件")

    # Layer B: 按指数聚合
    index_signals = aggregate_by_index(etf_signals, etf_master)
    if verbose:
        print(f"Layer B 完成: {len(index_signals)} 个指数")

    # Layer C: 全市场聚合
    overview = compute_market_overview(index_signals)
    if verbose:
        print(f"Layer C 完成: direction={overview.get('direction_sign')}, heat={overview.get('rotation_heat')}")

    # 输出
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # position_model.json
    json_path = OUTPUT_DIR / 'position_model.json'
    output_json = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_dir': str(data_dir),
        'n_etfs_loaded': len(share_data),
        'n_etfs_signals': len(etf_signals),
        'market_overview': overview,
        'index_latest': {},
    }
    # 每个指数的最新快照
    for idx_name, df in index_signals.items():
        if len(df) == 0:
            continue
        row = df.iloc[-1]
        output_json['index_latest'][idx_name] = {
            'date': str(row['trade_date']),
            'growth_20d': round(float(row.get('agg_growth_20d', 0)), 6) if not pd.isna(row.get('agg_growth_20d')) else None,
            'peak_ratio': round(float(row.get('peak_ratio', 0)), 4) if not pd.isna(row.get('peak_ratio')) else None,
            'hist_pctile': round(float(row.get('hist_pctile', 0)), 4) if not pd.isna(row.get('hist_pctile')) else None,
            'n_etfs': int(row.get('n_etfs', 1)),
        }

    with open(json_path, 'w') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"✅ {json_path}")

    # events.csv
    events_path = data_dir / 'events.csv'
    if events_rows:
        events_df = pd.DataFrame(events_rows)
        events_df = events_df.sort_values('date').reset_index(drop=True)
        events_df.to_csv(events_path, index=False)
        if verbose:
            print(f"✅ {events_path} ({len(events_df)} 事件)")
    else:
        if verbose:
            print("⚠️ 无异常事件检出")

    return overview


# ========== CLI ==========

if __name__ == '__main__':
    overview = build_position_model()
    print("\n=== 全市场仓位快照 ===")
    print(json.dumps(overview, ensure_ascii=False, indent=2))
