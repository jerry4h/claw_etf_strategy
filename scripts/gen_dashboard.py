#!/usr/bin/env python3
"""生成看板数据 JSON — 供 GitHub Pages dashboard 使用。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.strategy import load_config
from src.backtest import run_backtest
from src.data_loader import ETFS, OFFENSIVE, DEFENSIVE


def main():
    # 使用 v4.5-pvd 配置
    cfg_path = PROJ / 'config' / 'strategy_v4_5_pvd.yaml'
    cfg = load_config(cfg_path)
    result = run_backtest(cfg)
    nav = result.nav_series
    m = result.metrics

    # === 元数据 ===
    data = {
        "meta": {
            "strategy": cfg.name,
            "version": cfg.version,
            "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "data_range": f"{nav.index[0].date()} ~ {nav.index[-1].date()}",
            "weeks": m['total_weeks'],
        },
        # === 核心指标 ===
        "metrics": {
            "total_return": round(m['total_return'] * 100, 2),
            "annual_return": round(m['annual_return'] * 100, 2),
            "max_drawdown": round(m['max_drawdown'] * 100, 2),
            "current_drawdown": round(nav['drawdown'].iloc[-1] * 100, 2),
            "sharpe": round(m['sharpe_ratio'], 3),
            "calmar": round(m['calmar_ratio'], 2),
            "annual_vol": round(m['annual_volatility'] * 100, 2),
            "win_rate": round(m['win_rate'] * 100, 1),
            "defensive_weeks": m['defensive_weeks'],
        },
    }

    # === 净值序列（降采样到 ~260 点，避免数据太大）===
    step = max(1, len(nav) // 260)
    nav_sample = nav.iloc[::step]
    data["nav"] = {
        "dates": [str(d.date()) for d in nav_sample.index],
        "values": [round(v, 4) for v in nav_sample['nav'].tolist()],
        "drawdowns": [round(v * 100, 2) for v in nav_sample['drawdown'].tolist()],
    }

    # === 防御比例序列 ===
    data["defense"] = {
        "dates": [str(d.date()) for d in nav_sample.index],
        "ratios": [round(v * 100, 1) for v in nav_sample['def_ratio'].tolist()],
    }

    # === 纳指波动率序列（如有） ===
    if 'nasdaq_vol' in nav_sample.columns:
        data["nasdaq_vol"] = {
            "dates": [str(d.date()) for d in nav_sample.index],
            "values": [round(v * 100, 1) for v in nav_sample['nasdaq_vol'].tolist()],
        }

    # === 当前持仓 ===
    latest = nav.iloc[-1]
    holdings = []
    for etf in ETFS:
        col = f'weight_{etf}'
        w = latest.get(col, 0)
        cat = "进攻" if etf in OFFENSIVE else "防御"
        holdings.append({"name": etf, "weight": round(w * 100, 1), "category": cat})
    data["holdings"] = holdings

    # === 年度收益 ===
    nav_y = nav.copy()
    nav_y['year'] = nav_y.index.year
    annual_returns = []
    for year, group in nav_y.groupby('year'):
        yr_ret = (1 + group['weekly_return']).prod() - 1
        avg_def = group['def_ratio'].mean()
        annual_returns.append({
            "year": int(year),
            "return": round(yr_ret * 100, 1),
            "avg_defense": round(avg_def * 100, 1),
        })
    data["annual_returns"] = annual_returns

    # === ETF 持仓统计（全周期） ===
    etf_stats = []
    for etf in ETFS:
        col = f'weight_{etf}'
        if col in nav.columns:
            avg_w = nav[col].mean()
            held_weeks = int((nav[col] > 0.001).sum())
            etf_stats.append({
                "name": etf,
                "avg_weight": round(avg_w * 100, 1),
                "held_weeks": held_weeks,
                "held_pct": round(held_weeks / len(nav) * 100, 1),
            })
    data["etf_stats"] = etf_stats

    # === 近 1 年 / 近 3 月 / YTD 表现 ===
    now = nav.index[-1]
    for label, offset in [("ytd", pd.DateOffset(months=0)), ("year1", pd.DateOffset(years=1)),
                           ("month3", pd.DateOffset(months=3))]:
        start = pd.Timestamp(f"{now.year}-01-01") if label == "ytd" else now - offset
        seg = nav[nav.index >= start]
        if len(seg) > 0:
            seg_ret = seg['nav'].iloc[-1] / seg['nav'].iloc[0] - 1
            seg_dd = seg['drawdown'].max()
            data[f"recent_{label}"] = {
                "start": str(seg.index[0].date()),
                "end": str(seg.index[-1].date()),
                "return": round(seg_ret * 100, 2),
                "max_drawdown": round(seg_dd * 100, 2),
            }

    # === 配置参数快照 ===
    data["params"] = {
        "mom_w": cfg.mom_w,
        "vol_w": cfg.vol_w,
        "top_n": cfg.top_n,
        "score_margin": cfg.score_margin,
        "rebalance_threshold": cfg.rebalance_threshold,
        "max_single_alloc": cfg.max_single_alloc,
        "def_alloc": cfg.def_alloc,
        "step_low": cfg.step_low,
        "step_high": cfg.step_high,
        "max_def": cfg.max_def,
        "inv_vol_window": cfg.inv_vol_window,
        "vol_taper_enabled": cfg.vol_taper_enabled,
        "vol_taper_window": cfg.vol_taper_window,
        "vol_taper_len": cfg.vol_taper_len,
        "pvd_enabled": cfg.pvd_enabled,
        "pvd_w": cfg.pvd_w,
        "crisis_corr_ewma_enabled": getattr(cfg, 'crisis_corr_ewma_enabled', False),
    }

    # === 写入 ===
    out_dir = PROJ / 'dashboard'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'data.json'
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"✅ 看板数据已生成: {out_path}")
    print(f"   净值点数: {len(data['nav']['dates'])}")
    print(f"   覆盖年份: {len(annual_returns)}")
    print(f"   数据大小: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == '__main__':
    main()