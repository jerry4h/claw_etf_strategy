#!/usr/bin/env python3
"""Dashboard generator — runs backtest → JSON + embedded single-file index.html

Usage:
    python scripts/gen_dashboard.py              # 正常生成 index.html
    python scripts/gen_dashboard.py --preview    # 生成后启动本地 HTTP 服务器预览
"""

import argparse, json, sys
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.strategy import load_config
from src.backtest import run_backtest
from src.data_loader import OFFENSIVE, DEFENSIVE, load_nav_data, resample_weekly

# 看板显示顺序：纳指 → 中证500 → 黄金 → 红利低波 → 国债（进攻在前，防御在后）
DISPLAY_ORDER = ['纳指ETF', '中证500ETF', '黄金ETF', '红利低波ETF', '国债ETF']

# 主力追踪优先标的（6 只）
NT_PRIORITY_CODES = ['510300.SH', '510050.SH', '510500.SH', '512100.SH', '159915.SZ', '588000.SH']
NT_PRIORITY_NAMES = {
    '510300.SH': '沪深300ETF', '510050.SH': '上证50ETF',
    '510500.SH': '中证500ETF', '512100.SH': '中证1000ETF',
    '159915.SZ': '创业板ETF', '588000.SH': '科创50ETF',
}

# 指数级汇总配置（指数名→旗舰 ETF 代码 + benchmark 关键词）
INDEX_AGG_CONFIG = [
    {'name': '沪深300', 'flagship': '510300.SH', 'keywords': ['沪深300', '000300'], 'sector': '大盘核心'},
    {'name': '上证50', 'flagship': '510050.SH', 'keywords': ['上证50', '000016'], 'exclude': ['科创'], 'sector': '大盘核心'},
    {'name': '中证500', 'flagship': '510500.SH', 'keywords': ['中证500', '000905'], 'sector': '中盘'},
    {'name': '中证1000', 'flagship': '512100.SH', 'keywords': ['中证1000', '000852'], 'sector': '小盘'},
    {'name': '创业板', 'flagship': '159915.SZ', 'keywords': ['创业板', '399006', '399673'], 'sector': '创业板'},
    {'name': '科创50', 'flagship': '588000.SH', 'keywords': ['科创板50', '000688'], 'sector': '科创'},
    # 策略持仓关注 ETF（direct_codes 全市场同类基金，规模=Σ各只份额×各自价格）
    {'name': '纳指100', 'flagship': '513100.SH',
     'direct_codes': ['513100.SH', '513300.SH', '513390.SH', '159632.SZ', '159941.SZ',
                      '159696.SZ', '159513.SZ', '159501.SZ', '513110.SH', '159659.SZ',
                      '159660.SZ', '513870.SH'], 'sector': '海外科技'},
    {'name': '黄金', 'flagship': '518880.SH',
     'direct_codes': ['518880.SH', '159934.SZ', '159937.SZ', '518800.SH', '518660.SH',
                      '518850.SH', '159812.SZ'], 'sector': '商品'},
    {'name': '国债', 'flagship': '511010.SH',
     'direct_codes': ['511010.SH', '511260.SH', '511020.SH', '511290.SH', '511310.SH',
                      '159926.SZ'], 'sector': '利率'},
    {'name': '红利低波', 'flagship': '512890.SH',
     'direct_codes': ['512890.SH', '515100.SH', '515300.SH', '510890.SH', '515480.SH',
                      '159547.SZ', '159525.SZ', '560520.SH', '159549.SZ', '560720.SH',
                      '560730.SH'], 'sector': '红利防御'},
]


def _build_national_team_data():
    """
    构建主力追踪看板数据。若数据目录不存在/为空则返回降级标记。
    返回 dict 含 'available' 标记 + 数据字段，或 {'available': False, 'reason': ...}。
    """
    data_dir = PROJ / 'data' / 'national_team'
    share_dir = data_dir / 'fund_share'

    # 检测数据可用性
    if not share_dir.exists() or not any(share_dir.glob('*.csv')):
        return {'available': False, 'reason': '主力追踪数据未就绪，请先运行 scripts/fetch_national_team_share.py'}

    try:
        from src.national_team import build_position_model
        overview = build_position_model(data_dir=data_dir, verbose=False)
    except Exception as e:
        return {'available': False, 'reason': f'仓位建模异常: {e}'}

    if 'error' in overview:
        return {'available': False, 'reason': overview['error']}

    # 指数级汇总：按指数分组求和份额，乘以旗舰净值得规模(亿元)
    import yaml as _yaml
    config_path = PROJ / 'config' / 'national_team_etfs.yaml'
    with open(config_path) as _f:
        _cfg = _yaml.safe_load(_f)
    all_etfs = _cfg['etfs']

    price_cache_dir = data_dir / 'fund_daily_cache'
    share_trends = {}  # key=index_name

    for idx_cfg in INDEX_AGG_CONFIG:
        idx_name = idx_cfg['name']
        flagship = idx_cfg['flagship']
        keywords = idx_cfg.get('keywords', [])

        # 找属于该指数的所有 ETF（direct_codes 优先，否则按 benchmark 关键词匹配）
        member_codes = []
        exclude_kws = idx_cfg.get('exclude', [])
        if idx_cfg.get('direct_codes'):
            member_codes = list(idx_cfg['direct_codes'])
        else:
            for etf in all_etfs:
                bm = etf.get('benchmark', '') + ' ' + etf.get('name', '')
                if any(kw in bm for kw in keywords):
                    if exclude_kws and any(ek in bm for ek in exclude_kws):
                        continue
                    member_codes.append(etf['ts_code'])

        # 读取各 ETF 份额与价格，逐只计算规模后求和
        # 规模(亿元) = Σ(每只成员 份额_norm × 价格) / 10000
        # share_norm = 份额 × adj_latest / adj (把份额归一到最新单位)
        # 有价格文件的成员(持仓标的): 用各自前复权价精确计算
        # 无价格文件的成员(宽基指数): 用旗舰前复权价近似(成员价格≈指数价)
        # 折算日(如纳指 2022-01 1拆5): 份额_norm × qfq价 规模连续
        # 关键: adj 用 bfill——份额折算日份额已切换新单位(×N), adj 也应取折算后值
        # (ffill 取旧值会双重×N)。qfq 价用 ffill(折算日无价格行时用前值)
        share_frames = []
        member_prices = {}  # code -> 前复权价 Series（仅价格文件存在的成员）
        for code in member_codes:
            fname = code.replace('.', '_') + '.csv'
            fpath = share_dir / fname
            if not fpath.exists():
                continue
            sdf = pd.read_csv(fpath, dtype={'trade_date': str})
            sdf = sdf[sdf['trade_date'] >= '20180101'][['trade_date', 'fd_share']]
            if len(sdf) == 0:
                continue
            sdf = sdf.rename(columns={'fd_share': code})
            share_frames.append(sdf.set_index('trade_date'))

            # 成员的价格+复权因子（用于规模 + 净值曲线）
            pfile = price_cache_dir / fname if price_cache_dir.exists() else None
            if pfile and pfile.exists():
                pdf = pd.read_csv(pfile, dtype={'trade_date': str})
                pdf = pdf[pdf['trade_date'] >= '20180101'].sort_values('trade_date')
                pdf = pdf.set_index('trade_date')
                if 'close_qfq' in pdf.columns:
                    pser = pdf['close_qfq'].rename(code)
                else:
                    pser = pdf['close'].rename(code)
                member_prices[code] = pser

        if not share_frames:
            continue

        # 合并份额（逐日对齐）
        share_merged = pd.concat(share_frames, axis=1).sort_index()

        # 份额归一: share_norm = 份额 × adj_latest / adj (每只成员独立, adj 用 bfill)
        share_norm_series = []
        for code in share_merged.columns:
            pfile2 = price_cache_dir / (code.replace('.', '_') + '.csv')
            if pfile2.exists():
                pdf2 = pd.read_csv(pfile2, dtype={'trade_date': str})
                pdf2 = pdf2[pdf2['trade_date'] >= '20180101'].sort_values('trade_date')
                if 'adj_factor' in pdf2.columns and pdf2['adj_factor'].notna().any():
                    adj_s = pdf2.set_index('trade_date')['adj_factor']
                    la = adj_s.iloc[-1]
                    adj_aligned = adj_s.reindex(share_merged.index).bfill().ffill()
                    norm = share_merged[code] * la / adj_aligned
                    share_norm_series.append(norm.rename(code))
                else:
                    share_norm_series.append(share_merged[code])
            else:
                # 无价格文件(无 adj): 份额不归一（未发生折算或视为 adj=1）
                share_norm_series.append(share_merged[code])
        share_norm = pd.concat(share_norm_series, axis=1).sort_index()

        # 每只成员的定价: 有价格用各自 qfq, 无价格用旗舰 qfq (近似)
        # 旗舰价: 先取前复权, 无则原始 close
        flagship_price = None
        pfname = flagship.replace('.', '_') + '.csv'
        pfpath = price_cache_dir / pfname if price_cache_dir.exists() else None
        if pfpath and pfpath.exists():
            pdf = pd.read_csv(pfpath, dtype={'trade_date': str})
            pdf = pdf[pdf['trade_date'] >= '20180101'].sort_values('trade_date')
            pdf = pdf.set_index('trade_date')
            flagship_price = pdf['close_qfq'] if 'close_qfq' in pdf.columns else pdf['close']
            flagship_price = flagship_price.rename('__flagship__')

        # 构造逐日价格矩阵: 每个成员用自己的价, 缺的用旗舰价
        price_cols = {}
        for code in share_norm.columns:
            if code in member_prices:
                price_cols[code] = member_prices[code]
            elif flagship_price is not None:
                price_cols[code] = flagship_price.rename(code)
        if price_cols:
            price_merged = pd.concat(price_cols.values(), axis=1).sort_index()
            # qfq 价: 折算日无价格行用前值(ffill)
            price_merged = price_merged.ffill()
            # 逐只计算规模: share_norm × 各自价格, 再按日期求和
            aum_df = share_norm * price_merged.reindex(share_norm.index)
            aum_series = aum_df.sum(axis=1, min_count=1) / 10000  # 亿元
        else:
            aum_series = pd.Series(dtype=float)

        # 净值归一曲线: 用旗舰 ETF 前复权 close_qfq（消除分红/折算跳空）
        nav_series = flagship_price.rename(None) if flagship_price is not None else None

        # 周频降采样：每 5 个交易日取一个点
        aum_valid = aum_series.dropna()
        if len(aum_valid) == 0:
            continue
        aum_weekly = aum_valid.iloc[::5]
        if aum_valid.index[-1] not in aum_weekly.index:
            aum_weekly = pd.concat([aum_weekly, aum_valid.iloc[[-1]]])

        # 净值归一化曲线
        price_dates = []
        price_values = []
        if nav_series is not None and len(nav_series) > 0:
            nav_filt = nav_series[nav_series.index >= '20180101'].dropna()
            if len(nav_filt) > 0:
                nav_weekly = nav_filt.iloc[::5]
                if nav_filt.index[-1] not in nav_weekly.index:
                    nav_weekly = pd.concat([nav_weekly, nav_filt.iloc[[-1]]])
                base_p = nav_weekly.iloc[0]
                if base_p > 0:
                    price_dates = nav_weekly.index.tolist()
                    price_values = (nav_weekly / base_p).round(4).tolist()

        share_trends[idx_name] = {
            'name': idx_name,
            'dates': aum_weekly.index.tolist(),
            'aum': aum_weekly.round(1).tolist(),  # 亿元
            'price_dates': price_dates,
            'price_values': price_values,
            'n_etfs': len(share_frames),
            'current_aum': round(float(aum_valid.iloc[-1]), 1),
        }

    # 读取最近事件
    events_path = data_dir / 'events.csv'
    recent_events = []
    if events_path.exists():
        ev_df = pd.read_csv(events_path, dtype={'date': str})
        ev_df = ev_df.sort_values('date', ascending=False).head(20)
        for _, row in ev_df.iterrows():
            recent_events.append({
                'date': row.get('date', ''),
                'etf': row.get('etf', ''),
                'index': row.get('index', ''),
                'trigger_type': row.get('trigger_type', ''),
                'share_growth_20d': round(float(row['share_growth_20d']) * 100, 2) if pd.notna(row.get('share_growth_20d')) else None,
                'hist_pctile': round(float(row['hist_pctile']) * 100, 1) if pd.notna(row.get('hist_pctile')) else None,
            })

    # 板块概览（加入规模数据）
    # 建立 sector → 规模汇总映射
    sector_aum_map = {}
    for idx_cfg in INDEX_AGG_CONFIG:
        s = idx_cfg['sector']
        idx_name = idx_cfg['name']
        if idx_name in share_trends:
            aum = share_trends[idx_name].get('current_aum', 0)
            sector_aum_map[s] = sector_aum_map.get(s, 0) + aum

    sector_exposure = overview.get('sector_exposure', {})
    sector_cards = []
    for sector, growth in sorted(sector_exposure.items(), key=lambda x: abs(x[1]), reverse=True):
        card = {
            'sector': sector,
            'growth_20d': round(growth * 100, 1),
        }
        if sector in sector_aum_map and sector_aum_map[sector] > 0:
            card['current_aum'] = round(sector_aum_map[sector], 1)
        sector_cards.append(card)

    return {
        'available': True,
        'date': overview.get('date', ''),
        'direction_sign': overview.get('direction_sign', ''),
        'rotation_heat': overview.get('rotation_heat', ''),
        'concentration_hhi': overview.get('concentration_hhi', 0),
        'n_indices_active': overview.get('n_indices_active', 0),
        'sector_cards': sector_cards,
        'share_trends': share_trends,
        'recent_events': recent_events,
    }


def _build_data(cfg):
    result = run_backtest(cfg)
    nav = result.nav_series
    m = result.metrics

    data = {
        "meta": {
            "strategy": cfg.name, "version": cfg.version,
            "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M") + " (UTC+8)",
            "data_range": f"{nav.index[0].date()} ~ {nav.index[-1].date()}",
            "data_as_of": str(nav.index[-1].date()),
            "data_source": "Tushare", "weeks": m["total_weeks"],
        },
        "metrics": {
            "total_return": round(m["total_return"] * 100, 2),
            "annual_return": round(m["annual_return"] * 100, 2),
            "max_drawdown": round(m["max_drawdown"] * 100, 2),
            "current_drawdown": round(nav["drawdown"].iloc[-1] * 100, 2),
            "sharpe": round(m["sharpe_ratio"], 3),
            "calmar": round(m["calmar_ratio"], 2),
            "annual_vol": round(m["annual_volatility"] * 100, 2),
            "win_rate": round(m["win_rate"] * 100, 1),
            "defensive_weeks": m["defensive_weeks"],
        },
    }

    step = max(1, len(nav) // 260)
    nav_s = nav.iloc[::step]

    data["nav"] = {"dates": [str(d.date()) for d in nav_s.index],
                   "values": [round(v, 4) for v in nav_s["nav"].tolist()]}
    data["defense"] = {"dates": [str(d.date()) for d in nav_s.index],
                       "ratios": [round(v * 100, 1) for v in nav_s["def_ratio"].tolist()]}
    data["drawdown"] = {"dates": [str(d.date()) for d in nav_s.index],
                        "ratios": [round(v * 100, 2) for v in nav_s["drawdown"].tolist()]}

    latest = nav.iloc[-1]
    data["holdings"] = [
        {"name": e, "weight": round(latest.get(f"weight_{e}", 0) * 100, 1),
         "category": "进攻" if e in OFFENSIVE else "防御"}
        for e in DISPLAY_ORDER
    ]

    nav_c = nav.copy()
    nav_c["year"] = nav_c.index.year
    data["annual_returns"] = []
    for yr, grp in nav_c.groupby("year"):
        ret = (1 + grp["weekly_return"]).prod() - 1
        avg_d = grp["def_ratio"].mean()
        data["annual_returns"].append({
            "year": int(yr), "return": round(ret * 100, 1),
            "avg_defense": round(avg_d * 100, 1),
        })

    data["etf_stats"] = []
    for e in DISPLAY_ORDER:
        col = f"weight_{e}"
        if col in nav.columns:
            aw = nav[col].mean()
            hw = int((nav[col] > 0.001).sum())
            data["etf_stats"].append({
                "name": e, "avg_weight": round(aw * 100, 1),
                "held_weeks": hw, "held_pct": round(hw / len(nav) * 100, 1),
            })

    now = nav.index[-1]
    for label, start_fn in [
        ("ytd", lambda: pd.Timestamp(f"{now.year}-01-01")),
        ("year1", lambda: now - pd.DateOffset(years=1)),
        ("month3", lambda: now - pd.DateOffset(months=3)),
    ]:
        start = start_fn()
        seg = nav[nav.index >= start]
        if len(seg) > 0:
            # YTD 锚定到上一年末收盘（窗口前最后一个净值点），与"年度收益"图
            # 的当年口径一致——标准 YTD 从上年末算起，含跨年首周收益。其它滚动
            # 窗口沿用窗口内首点为基准。
            if label == "ytd":
                prior = nav[nav.index < start]
                base_nav = prior["nav"].iloc[-1] if len(prior) > 0 else seg["nav"].iloc[0]
            else:
                base_nav = seg["nav"].iloc[0]
            sr = seg["nav"].iloc[-1] / base_nav - 1
            sd = seg["drawdown"].max()
            data[f"recent_{label}"] = {
                "start": str(seg.index[0].date()),
                "end": str(seg.index[-1].date()),
                "return": round(sr * 100, 2),
                "max_drawdown": round(sd * 100, 2),
            }

    data["params"] = {
        "mom_w": cfg.mom_w, "vol_w": cfg.vol_w, "top_n": cfg.top_n,
        "def_alloc": cfg.def_alloc, "step_low": cfg.step_low,
        "step_high": cfg.step_high, "max_def": cfg.max_def,
        "max_single_alloc": cfg.max_single_alloc,
        "score_margin": cfg.score_margin,
        "rebalance_threshold": cfg.rebalance_threshold,
        "vol_taper_enabled": cfg.vol_taper_enabled,
        "vol_taper_window": cfg.vol_taper_window,
        "vol_taper_len": cfg.vol_taper_len,
        "inv_vol_window": cfg.inv_vol_window,
        "pvd_enabled": cfg.pvd_enabled, "pvd_w": cfg.pvd_w,
    }

    # 主力追踪数据
    data["national_team"] = _build_national_team_data()

    return data


def _generate_weekly_report(nt_data: dict):
    """生成 output/national_team/weekly_report.md"""
    out_dir = PROJ / 'output' / 'national_team'
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / 'weekly_report.md'

    today = datetime.now().strftime('%Y-%m-%d')

    if not nt_data.get('available'):
        report_path.write_text(
            f"# 主力追踪周报 — {today}\n\n> 数据未就绪: {nt_data.get('reason', '未知原因')}\n",
            encoding='utf-8')
        return report_path

    lines = [f"# 主力追踪周报 — {today}\n"]
    lines.append(f"**数据截至**: {nt_data.get('date', 'N/A')}\n")
    lines.append(f"**总体方向**: {nt_data.get('direction_sign', 'N/A')} | "
                 f"**轮动热点**: {nt_data.get('rotation_heat', 'N/A')} | "
                 f"**集中度 HHI**: {nt_data.get('concentration_hhi', 0):.4f}\n")

    # Top 5 板块变动
    lines.append("\n## 本周主要变动 Top 5（按 20d 增幅排序）\n")
    lines.append("| 板块 | 20d 增幅 | 方向 |")
    lines.append("|------|---------|------|")
    sector_cards = nt_data.get('sector_cards', [])
    for card in sector_cards[:5]:
        g = card['growth_20d']
        direction = '↑ 入场' if g > 0 else '↓ 兑现' if g < 0 else '— 持平'
        lines.append(f"| {card['sector']} | {g:+.1f}% | {direction} |")

    # Rotation 判定
    lines.append("\n## 板块 Rotation 判定\n")
    lines.append(f"- **Direction Sign**: {nt_data.get('direction_sign', 'N/A')}")
    lines.append(f"- **Rotation Heat（最大增幅板块）**: {nt_data.get('rotation_heat', 'N/A')}")
    lines.append(f"- **活跃指数数**: {nt_data.get('n_indices_active', 0)}")
    if sector_cards:
        lines.append("\n全板块概览:\n")
        for card in sector_cards:
            g = card['growth_20d']
            emoji = '🔴' if g > 5 else '🟡' if g > 0 else '🔵'
            lines.append(f"  {emoji} {card['sector']}: {g:+.1f}%")

    # 异常事件
    lines.append("\n## 本周异常事件\n")
    events = nt_data.get('recent_events', [])
    if events:
        lines.append("| 日期 | ETF | 板块 | 份额增幅 | 百分位 |")
        lines.append("|------|-----|------|---------|---------|--------|")
        # 取最近 7 天的事件(近似本周)
        if events:
            latest_date = events[0].get('date', '')
            # 简化：取最新 10 条作为"本周"
            for ev in events[:10]:
                g = f"{ev['share_growth_20d']:+.2f}%" if ev.get('share_growth_20d') is not None else '—'
                p = f"{ev['hist_pctile']:.1f}%" if ev.get('hist_pctile') is not None else '—'
                lines.append(f"| {ev['date']} | {ev['etf']} | {ev['index']} | {g} | {p} |")
    else:
        lines.append("*本周无异常事件*\n")

    # 与上周对比（基于当前数据快照，简要说明）
    lines.append("\n## 与上周对比\n")
    lines.append(f"- 总体方向: **{nt_data.get('direction_sign', 'N/A')}**")
    lines.append(f"- 轮动热点: **{nt_data.get('rotation_heat', 'N/A')}**")
    if sector_cards:
        positive = [c for c in sector_cards if c['growth_20d'] > 0]
        negative = [c for c in sector_cards if c['growth_20d'] < 0]
        lines.append(f"- 净入场板块数: {len(positive)} / 净兑现板块数: {len(negative)}")

    lines.append(f"\n---\n*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    report_path.write_text('\n'.join(lines), encoding='utf-8')
    return report_path


TPL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>虾池ETF轮动策略 · 实时看板</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
:root {--bg:#0b1121;--card:#151e34;--card-hover:#1c2844;--text:#e8edf5;--muted:#8892a8;--accent:#60a5fa;--green:#34d399;--red:#f87171;--orange:#fb923c;--border:#1e2a45;}
* {margin:0;padding:0;box-sizing:border-box;}
html {overflow-x:hidden;}
body {font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Noto Sans SC',sans-serif;background:var(--bg);color:var(--text);padding:24px;min-height:100vh;overflow-x:hidden;}
.container {max-width:1280px;margin:0 auto;width:100%;}
.header {margin-bottom:28px;}
.header h1 {font-size:1.5rem;font-weight:700;background:linear-gradient(135deg,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.header .sub {color:var(--muted);font-size:0.85rem;margin-top:4px;}
.header .sub em {color:var(--text);font-style:normal;}
.header .asof {margin-top:8px;font-size:0.95rem;font-weight:600;color:var(--accent);}
.cards {display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-bottom:20px;}
.card {background:var(--card);border-radius:10px;padding:14px 16px;border:1px solid var(--border);transition:border-color .2s,background .2s;}
.card:hover {border-color:var(--accent);background:var(--card-hover);}
.card .l {font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;}
.card .v {font-size:1.35rem;font-weight:700;margin-top:3px;font-variant-numeric:tabular-nums;}
.card .s {font-size:0.7rem;color:var(--muted);margin-top:1px;}
.card .v.big {font-size:1.7rem;}
.g {color:var(--green)} .r {color:var(--red)} .o {color:var(--orange)} .a {color:var(--accent)}
.panel {background:var(--card);border-radius:10px;padding:18px;border:1px solid var(--border);max-width:100%;overflow:hidden;}
.panel h2 {font-size:0.8rem;color:var(--muted);margin-bottom:14px;text-transform:uppercase;letter-spacing:.5px;}
.panel h2 .tgl {float:right;background:transparent;border:1px solid var(--border);color:var(--muted);font-size:0.68rem;padding:2px 10px;border-radius:6px;cursor:pointer;letter-spacing:0;text-transform:none;transition:color .2s,border-color .2s;}
.panel h2 .tgl:hover {color:var(--accent);border-color:var(--accent);}
.chart-err {padding:48px 12px;text-align:center;color:var(--muted);font-size:0.85rem;border:1px dashed var(--border);border-radius:8px;}
.grid-2 {display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin-bottom:16px;}
.grid-2-1 {display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin-bottom:16px;}
@media(max-width:600px){body{padding:14px;}.rc{grid-template-columns:1fr;}.panel{padding:12px;overflow-x:auto;}}
.table-wrap {width:100%;overflow-x:auto;-webkit-overflow-scrolling:touch;}
.ht {width:100%;border-collapse:collapse;font-size:0.82rem;}
.ht th {text-align:left;color:var(--muted);padding:6px 8px;font-weight:500;border-bottom:1px solid var(--border);white-space:nowrap;}
.ht td {padding:6px 8px;border-bottom:1px solid var(--border);word-break:break-all;}
.ht tr:last-child td {border:none;}
.hbar {background:var(--border);border-radius:4px;height:16px;overflow:hidden;min-width:60px;}
.hfill {height:100%;border-radius:4px;}
.st {width:100%;border-collapse:collapse;font-size:0.8rem;}
.st th {text-align:left;color:var(--muted);padding:5px 8px;font-weight:500;border-bottom:1px solid var(--border);white-space:nowrap;}
.st td {padding:5px 8px;border-bottom:1px solid var(--border);word-break:break-all;}
.st tr:hover td {background:rgba(96,165,250,0.04);}
.rc {display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;}
.rc-item {background:rgba(11,17,33,0.6);border-radius:8px;padding:12px;text-align:center;}
.rc-item .lbl {font-size:0.75rem;color:var(--muted);}
.rc-item .val {font-size:1.15rem;font-weight:700;margin-top:4px;}
.rc-item .sub {font-size:0.65rem;color:var(--muted);margin-top:3px;}
.px {display:grid;grid-template-columns:repeat(auto-fill,minmax(100px,1fr));gap:8px;}
.px-item {background:rgba(11,17,33,0.6);border-radius:6px;padding:6px 8px;text-align:center;}
.px-item .k {font-size:0.65rem;color:var(--muted);}
.px-item .v {font-size:0.82rem;font-weight:600;margin-top:2px;color:var(--accent);}
.chart-wrap {position:relative;width:100%;min-height:0;}
.chart-wrap canvas {display:block;width:100% !important;height:100% !important;}
.chart-stack {display:flex;flex-direction:column;width:100%;min-height:480px;max-width:100%;}
.chart-stack > .chart-wrap:nth-child(1) {flex:5;min-height:180px;}
.chart-stack > .chart-wrap:nth-child(2) {flex:3;min-height:120px;}
.chart-stack > .chart-wrap:nth-child(3) {flex:3;min-height:120px;}
.nt-section {margin-top:24px;border-top:1px solid var(--border);padding-top:24px;}
.nt-section h2.section-title {font-size:1.1rem;font-weight:700;color:var(--accent);margin-bottom:16px;}
.nt-cards {display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:16px;}
.nt-grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;}
.nt-card {background:var(--card);border-radius:8px;padding:12px;border:1px solid var(--border);text-align:center;}
.nt-card .sector {font-size:0.72rem;color:var(--muted);margin-bottom:4px;}
.nt-card .growth {font-size:1.2rem;font-weight:700;}
.nt-card .meta {font-size:0.65rem;color:var(--muted);margin-top:3px;}
.nt-degraded {padding:32px;text-align:center;color:var(--orange);font-size:0.9rem;background:rgba(251,146,60,0.05);border:1px dashed var(--orange);border-radius:10px;margin-top:16px;}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <h1>🦐 虾池ETF轮动策略 · 实时看板</h1>
  <div class="sub" id="header-sub">加载中...</div>
  <div class="asof" id="header-asof"></div>
</div>

<div class="cards" id="metric-cards"></div>

<div class="grid-2-1">
  <div class="panel">
    <h2>📈 净值 · 回撤 · 防御 <button class="tgl" id="scale-toggle" type="button">对数坐标</button></h2>
    <div class="chart-stack">
      <div class="chart-wrap"><canvas id="navChart"></canvas></div>
      <div class="chart-wrap"><canvas id="ddChart"></canvas></div>
      <div class="chart-wrap"><canvas id="defChart"></canvas></div>
    </div>
  </div>
  <div class="panel">
    <h2>🎯 当前持仓</h2>
    <div id="holdings-content"></div>
    <div id="offdef-summary" style="margin-top:12px;font-size:0.85rem;display:flex;gap:16px;"></div>
  </div>
</div>

<div class="grid-2">
  <div class="panel">
    <h2>📊 年度收益</h2>
    <div class="chart-wrap"><canvas id="annualChart"></canvas></div>
  </div>
  <div class="panel">
    <h2>⏱️ 近期表现</h2>
    <div class="rc" id="recent-cards"></div>
  </div>
</div>

<div class="grid-2">
  <div class="panel">
    <h2>📋 ETF 全周期统计</h2>
    <div class="table-wrap"><table class="st"><thead><tr><th>ETF</th><th>平均权重</th><th>持有周</th><th>持有率</th></tr></thead>
    <tbody id="etf-stats-body"></tbody></table></div>
  </div>
  <div class="panel">
    <h2>⚙️ 策略参数</h2>
    <div class="px" id="params-content"></div>
  </div>
</div>

<!-- 主力追踪板块 -->
<div class="nt-section" id="nt-section">
  <h2 class="section-title">🏛️ 主力追踪</h2>
  <div id="nt-content"></div>
</div>

</div>

<script id="main-script">
const DATA = __DATA__;

(function() {
  const d = DATA;
  const {meta,metrics:m,nav:nd,defense:dd,drawdown:dwd,
         holdings,annual_returns:ann,etf_stats:etfS,
         recent_ytd:rytd,recent_year1:r1y,recent_month3:r3m,params,national_team:nt} = d;

  // Header
  const endDate = (meta.data_range || '').split(' ~ ')[1] || '';
  const wd = endDate ? '日一二三四五六'[new Date(endDate+'T00:00:00').getDay()] : '';
  document.getElementById('header-sub').innerHTML =
    `${meta.strategy} · <em>v${meta.version}</em>${meta.version_note?`（${meta.version_note}）`:''}<br>` +
    `数据源: ${meta.data_source||'Tushare'} · 周频调仓 · 区间 ${meta.data_range} · 生成: ${meta.generated_at}`;
  document.getElementById('header-asof').textContent =
    endDate ? `📅 数据截至 ${endDate}（周${wd}收盘）` : '';

  // Metric cards
  const fnum = (v,dig) => Number.isFinite(v) ? v.toFixed(dig) : '—';
  const fpct = v => Number.isFinite(v) ? v+'%' : '—';
  const cards = [
    {l:'夏普比率',  v:fnum(m.sharpe,3),        c:'a', big:1},
    {l:'年化收益',  v:fpct(m.annual_return),    c:m.annual_return>10?'g':'o', big:1},
    {l:'最大回撤',  v:fpct(m.max_drawdown),     c:'r', big:1},
    {l:'当前回撤',  v:fpct(m.current_drawdown), c:m.current_drawdown<2?'g':(m.current_drawdown<4?'o':'r')},
    {l:'卡尔马',    v:fnum(m.calmar,2),         c:'a'},
    {l:'年化波动',  v:fpct(m.annual_vol),       c:'o'},
    {l:'周胜率',    v:fpct(m.win_rate),         c:'g'},
    {l:'防御周数',  v:m.defensive_weeks,        s:'/'+meta.weeks+'周', c:'a'},
  ];
  document.getElementById('metric-cards').innerHTML =
    cards.map(c=>`<div class="card"><div class="l">${c.l}</div><div class="v ${c.c}${c.big?' big':''}">${c.v}</div>${c.s?`<div class="s">${c.s}</div>`:''}</div>`).join('');

  // Chart presets
  const TIP_ANIM = {duration:0};
  const CHART_OPT = {
    responsive:true, maintainAspectRatio:false, resizeDelay:200,
    animation:{duration:0}, transitions:{active:{animation:{duration:0}}},
    plugins:{legend:{display:false}, tooltip:{animation:TIP_ANIM}},
    scales:{x:{ticks:{maxTicksLimit:8,color:'#8892a8',font:{size:9}}, grid:{color:'rgba(30,42,69,0.5)'}},
            y:{ticks:{color:'#8892a8',font:{size:9}}, grid:{color:'rgba(30,42,69,0.5)'}}},
    interaction:{mode:'nearest',axis:'x',intersect:false},
  };
  const vline = {
    id:'vline', afterDatasetsDraw(ch) {
      const act = ch.tooltip?.getActiveElements?.() || [];
      if (!act.length) return;
      const x = act[0].element.x, ctx = ch.ctx, ca = ch.chartArea;
      ctx.save(); ctx.strokeStyle='rgba(136,146,168,0.4)'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
      ctx.beginPath(); ctx.moveTo(x, ca.top); ctx.lineTo(x, ca.bottom); ctx.stroke(); ctx.restore();
    }
  };
  const clean = (dates, vals) => {
    const D=[], V=[]; (dates||[]).forEach((dt,i)=>{const v=(vals||[])[i]; if(v!==null&&v!==undefined&&isFinite(v)){D.push(dt);V.push(v);}}); return {dates:D, vals:V};
  };
  const chartMsg = (id, msg) => { const el=document.getElementById(id); if(el) el.parentElement.innerHTML=`<div class="chart-err">${msg}</div>`; };
  const canChart = typeof Chart !== 'undefined';
  const CDN_ERR = '⚠️ 图表库加载失败，请检查网络后刷新';

  // Crosshair sync
  const stkCharts = [];
  const syncCX = (src) => {
    if (!src || !stkCharts.length) return;
    const act = src.getActiveElements();
    if (!act.length) return;
    const idx = act[0].index;
    stkCharts.forEach(c => {
      if (c === src) return;
      c.tooltip.setActiveElements([{datasetIndex:0, index:idx}], {x:0, y:0});
      c.draw();
    });
  };
  const ON_HOVER = (e, el, chart) => syncCX(chart);

  // 1. NAV
  const ns = clean(nd?.dates, nd?.values);
  if (canChart && ns.dates.length) {
    const nc = new Chart(document.getElementById('navChart'), {
      type:'line', data:{labels:ns.dates, datasets:[{label:'净值', data:ns.vals, borderColor:'#60a5fa', backgroundColor:'rgba(96,165,250,0.08)', fill:true, tension:0.1, pointRadius:0, borderWidth:1.5}]},
      options:{...CHART_OPT,
        scales:{...CHART_OPT.scales, y:{...CHART_OPT.scales.y, type:'logarithmic', ticks:{...CHART_OPT.scales.y.ticks, callback:v=>Number(v).toFixed(1)+'x'}}},
        onHover:ON_HOVER,
      },
      plugins:[vline]
    });
    stkCharts.push(nc);
  } else if (!canChart) chartMsg('navChart', CDN_ERR);

  // 2. Drawdown
  const ds = clean(dwd?.dates, dwd?.ratios);
  if (canChart && ds.dates.length) {
    const dc = new Chart(document.getElementById('ddChart'), {
      type:'line', data:{labels:ds.dates, datasets:[{label:'回撤', data:ds.vals, borderColor:'#f87171', backgroundColor:'rgba(248,113,113,0.15)', fill:true, tension:0.1, pointRadius:0, borderWidth:1}]},
      options:{...CHART_OPT,
        plugins:{...CHART_OPT.plugins, tooltip:{...CHART_OPT.plugins.tooltip, callbacks:{label:c=>'回撤: '+c.parsed.y.toFixed(2)+'%'}}},
        scales:{...CHART_OPT.scales, y:{...CHART_OPT.scales.y, min:0, reverse:true, ticks:{...CHART_OPT.scales.y.ticks, callback:v=>v+'%'}}},
        onHover:ON_HOVER,
      },
      plugins:[vline]
    });
    stkCharts.push(dc);
  } else if (!canChart) chartMsg('ddChart', CDN_ERR);

  // 3. Defense
  const dds = clean(dd?.dates, dd?.ratios);
  if (canChart && dds.dates.length) {
    const dfc = new Chart(document.getElementById('defChart'), {
      type:'line', data:{labels:dds.dates, datasets:[{label:'防御比', data:dds.vals, borderColor:'#34d399', backgroundColor:'rgba(52,211,153,0.12)', fill:true, tension:0.1, pointRadius:0, borderWidth:1}]},
      options:{...CHART_OPT,
        plugins:{...CHART_OPT.plugins, tooltip:{...CHART_OPT.plugins.tooltip, callbacks:{label:c=>'防御比: '+c.parsed.y.toFixed(1)+'%'}}},
        scales:{...CHART_OPT.scales, y:{...CHART_OPT.scales.y, min:0, max:100, ticks:{...CHART_OPT.scales.y.ticks, callback:v=>v+'%'}}},
        onHover:ON_HOVER,
      },
      plugins:[vline]
    });
    stkCharts.push(dfc);
  } else if (!canChart) chartMsg('defChart', CDN_ERR);

  // Scale toggle
  const st = document.getElementById('scale-toggle');
  let logScale = true;
  const nc = Chart.getChart('navChart');
  if (nc) {
    const apply = () => { nc.options.scales.y.type = logScale ? 'logarithmic' : 'linear'; st.textContent = logScale ? '对数坐标' : '线性坐标'; nc.update(); };
    apply(); st.onclick = () => { logScale = !logScale; apply(); };
  } else st.style.display = 'none';

  // Holdings
  const catClr = {'进攻':'#f59e0b','防御':'#60a5fa'};
  let htm = '<table class="ht"><thead><tr><th>ETF</th><th>仓位</th><th></th></tr></thead><tbody>';
  for (const h of holdings) {
    if (h.weight<0) continue;
    const clr=catClr[h.category];
    const zero = h.weight<0.05;
    htm += `<tr${zero?' style="color:var(--muted)"':''}><td><span style="color:${zero?'var(--muted)':clr}">${h.category==='进攻'?'⚔️':'🛡️'}</span> ${h.name}</td><td style="font-weight:600">${h.weight.toFixed(1)}%</td><td><div class="hbar">${zero?'':`<div class="hfill" style="width:${h.weight>5?h.weight:5}%;background:${clr}"></div>`}</div></td></tr>`;
  }
  htm += '</tbody></table>';
  document.getElementById('holdings-content').innerHTML = htm;
  const off = holdings.filter(h=>h.category==='进攻').reduce((s,h)=>s+h.weight,0);
  const def = holdings.filter(h=>h.category==='防御').reduce((s,h)=>s+h.weight,0);
  document.getElementById('offdef-summary').innerHTML =
    `<span style="color:#f59e0b">⚔️ 进攻 ${off.toFixed(1)}%</span><span style="color:#60a5fa">🛡️ 防御 ${def.toFixed(1)}%</span>`;

  // Annual
  const endD = endDate ? new Date(endDate+'T00:00:00') : null;
  const annArr = ann || [];
  const annLabels = annArr.map(a => (endD && a.year===endD.getFullYear() && endD.getMonth()<11) ? a.year+' YTD' : String(a.year));
  if (canChart && annArr.length) {
    new Chart(document.getElementById('annualChart'), {
      type:'bar', data:{labels:annLabels, datasets:[{label:'年收益', data:annArr.map(a=>a.return), backgroundColor:annArr.map(a=>a.return>=0?'rgba(52,211,153,0.7)':'rgba(248,113,113,0.7)'), borderColor:annArr.map(a=>a.return>=0?'#34d399':'#f87171'), borderWidth:1, borderRadius:3}]},
      options:{...CHART_OPT,
        plugins:{...CHART_OPT.plugins, legend:{display:false}, tooltip:{...CHART_OPT.plugins.tooltip, callbacks:{afterLabel:ctx=>'平均防御: '+annArr[ctx.dataIndex].avg_defense+'%'}}},
        scales:{...CHART_OPT.scales, x:{...CHART_OPT.scales.x, grid:{display:false}}, y:{...CHART_OPT.scales.y, ticks:{...CHART_OPT.scales.y.ticks, callback:v=>v+'%'}}},
      },
    });
  } else if (!canChart) chartMsg('annualChart', CDN_ERR);

  // Recent
  document.getElementById('recent-cards').innerHTML =
    [{l:'今年 (YTD)',d:rytd},{l:'近1年',d:r1y},{l:'近3月',d:r3m}].filter(r=>r.d).map(r=>
      `<div class="rc-item"><div class="lbl">${r.l}</div><div class="val ${r.d.return>0?'g':'r'}">${r.d.return>0?'+':''}${r.d.return}%</div><div class="sub">最大回撤 ${r.d.max_drawdown}% · ${r.d.start}</div></div>`
    ).join('');

  // ETF stats
  document.getElementById('etf-stats-body').innerHTML =
    etfS.map(e=>`<tr><td>${e.name}</td><td>${e.avg_weight}%</td><td>${e.held_weeks}</td><td>${e.held_pct}%</td></tr>`).join('');

  // Params
  const pl = {
    mom_w:'动量权重', vol_w:'波动权重', top_n:'选TOP-N',
    score_margin:'分数门槛', rebalance_threshold:'调仓阈值',
    max_single_alloc:'单标上限', def_alloc:'基准防御',
    step_low:'防御下限', step_high:'防御上限', max_def:'最大防御',
    inv_vol_window:'InvVol窗口', vol_taper_enabled:'Taper',
    vol_taper_window:'Taper窗口', vol_taper_len:'Taper降权',
    pvd_enabled:'PVD', pvd_w:'PVD权重'
  };
  document.getElementById('params-content').innerHTML =
    Object.entries(params).filter(([k])=>pl[k]).map(([k,v])=>`<div class="px-item"><div class="k">${pl[k]}</div><div class="v">${typeof v==='boolean'?(v?'✅':'❌'):v}</div></div>`).join('');

  // ===== 主力追踪板块 =====
  const ntEl = document.getElementById('nt-content');
  if (!nt || !nt.available) {
    const reason = (nt && nt.reason) ? nt.reason : '主力追踪数据未就绪，请先运行 scripts/fetch_national_team_share.py';
    ntEl.innerHTML = `<div class="nt-degraded">⚠️ ${reason}</div>`;
  } else {
    let ntHtml = '';

    // 板块增幅概览卡片（含规模）
    ntHtml += `<div style="margin-bottom:8px;font-size:0.8rem;color:var(--muted)">数据截至 ${nt.date} · 方向: <span style="color:${nt.direction_sign==='入场'?'var(--green)':'var(--red)'};font-weight:600">${nt.direction_sign}</span> · 轮动热点: <span style="color:var(--accent);font-weight:600">${nt.rotation_heat}</span></div>`;
    ntHtml += '<div class="nt-cards">';
    for (const sc of (nt.sector_cards||[])) {
      const clr = sc.growth_20d > 5 ? 'var(--red)' : sc.growth_20d > 0 ? 'var(--green)' : 'var(--accent)';
      const heat = sc.sector === nt.rotation_heat ? ' 🔥' : '';
      const aumStr = sc.current_aum ? `<div class="meta" style="margin-top:2px">规模 ${sc.current_aum>=10000?((sc.current_aum/10000).toFixed(1)+'万亿'):(sc.current_aum.toFixed(0)+'亿')}</div>` : '';
      ntHtml += `<div class="nt-card"><div class="sector">${sc.sector}${heat}</div><div class="growth" style="color:${clr}">${sc.growth_20d>0?'+':''}${sc.growth_20d}%</div><div class="meta">20d 增幅</div>${aumStr}</div>`;
    }
    ntHtml += '</div>';

    // 量价对比图容器（一行最多 2 列）
    ntHtml += '<div class="panel" style="margin-bottom:16px"><h2>📈 主战场量价对比（2018 年起 · 左轴=规模亿元 · 右轴=净值归一）</h2>';
    ntHtml += '<div class="nt-grid" id="ntGridCharts"></div></div>';

    // 事件表
    ntHtml += '<div class="panel"><h2>\u26a1 \u7591\u4f3c\u4e3b\u529b\u4ecb\u5165\u4e8b\u4ef6\uff08\u6700\u8fd1 20 \u6761\uff09</h2>';
    ntHtml += '<div class="table-wrap">';
    if ((nt.recent_events||[]).length) {
      ntHtml += '<table class="st"><thead><tr><th>日期</th><th>ETF</th><th>板块</th><th>份额增幅</th><th>百分位</th></tr></thead><tbody>';
      for (const ev of nt.recent_events) {
        const g = ev.share_growth_20d !== null ? (ev.share_growth_20d > 0 ? '+' : '') + ev.share_growth_20d + '%' : '—';
        const p = ev.hist_pctile !== null ? ev.hist_pctile + '%' : '—';
        const gc = ev.share_growth_20d > 0 ? 'g' : (ev.share_growth_20d < 0 ? 'r' : '');
        ntHtml += `<tr><td>${ev.date}</td><td>${ev.etf}</td><td>${ev.index}</td><td class="${gc}">${g}</td><td>${p}</td></tr>`;
      }
      ntHtml += '</tbody></table>';
    } else {
      ntHtml += '<div style="padding:16px;color:var(--muted);text-align:center">暂无事件</div>';
    }
    ntHtml += '</div>';  // close table-wrap
    ntHtml += '</div>';

    ntEl.innerHTML = ntHtml;

    // 绘制量价对比小图（2×3 grid，每个指数双 Y 轴）
    if (canChart && nt.share_trends && Object.keys(nt.share_trends).length) {
      const colors = ['#60a5fa','#f87171','#34d399','#fb923c','#a78bfa','#f472b6'];
      const trends = nt.share_trends;
      const codes = Object.keys(trends);
      const gridEl = document.getElementById('ntGridCharts');
      codes.forEach((code, i) => {
        const t = trends[code];
        const wrap = document.createElement('div');
        wrap.style.cssText = 'height:200px;background:var(--panel-bg);border-radius:8px;padding:6px';
        const cvs = document.createElement('canvas');
        wrap.appendChild(cvs);
        gridEl.appendChild(wrap);
        const datasets = [{
          label: t.name + ' 规模(亿)',
          data: t.aum,
          borderColor: colors[i % colors.length],
          backgroundColor: 'transparent',
          tension: 0.15, pointRadius: 0, borderWidth: 1.5,
          yAxisID: 'yAum',
        }];
        if (t.price_values && t.price_values.length) {
          datasets.push({
            label: t.name + ' 净值',
            data: t.price_values,
            borderColor: colors[i % colors.length] + '80',
            backgroundColor: 'transparent',
            borderDash: [4,3],
            tension: 0.15, pointRadius: 0, borderWidth: 1.2,
            yAxisID: 'yPrice',
          });
        }
        const subtitle = t.n_etfs ? ` (${t.n_etfs}只 ETF)` : '';
        new Chart(cvs, {
          type: 'line',
          data: { labels: t.dates, datasets: datasets },
          options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: true, labels: { color:'#8892a8', font:{size:9}, boxWidth:10 } }, title: { display:true, text: t.name + subtitle, color:'#e2e8f0', font:{size:11} }, tooltip: { mode:'index', intersect:false } },
            scales: {
              x: { ticks: { color:'#64748b', maxTicksLimit:6, callback: function(val,idx) { const d=this.getLabelForValue(val); return d?d.substring(0,4):''; } }, grid:{color:'#1e293b'} },
              yAum: { position:'left', type:'logarithmic', title:{display:true, text:'规模(亿元)', color:'#8892a8', font:{size:8}}, ticks: { color: colors[i%colors.length], font:{size:9}, callback: function(v){ return v>=10000?((v/10000).toFixed(0)+'万亿'):(v>=1000?((v/1000).toFixed(0)+'k'):v.toFixed(0)); } }, grid:{color:'#1e293b55'} },
              yPrice: { position:'right', type:'linear', title:{display:true, text:'净值(归一)', color:'#8892a880', font:{size:8}}, ticks: { color: (colors[i%colors.length]+'80'), font:{size:9} }, grid:{display:false} },
            },
            interaction: { mode:'index', intersect:false },
          }
        });
      });
    }
  }

})();
</script>
</body>
</html>"""


def _html_template(data_json):
    return TPL.replace("__DATA__", data_json)


def main():
    parser = argparse.ArgumentParser(description="生成虾池ETF轮动策略看板")
    parser.add_argument("--preview", action="store_true",
                        help="生成后启动本地 HTTP 服务器（http://localhost:8000）以便预览")
    args = parser.parse_args()

    cfg = load_config(PROJ / "config" / "strategy_v4_6.yaml")
    data = _build_data(cfg)
    data["meta"]["version_note"] = "生产版（Layer3.5 定向 boost 分级应用）"

    out_dir = PROJ / "dashboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    html = _html_template(json.dumps(data, ensure_ascii=False))
    (PROJ / "index.html").write_text(html, encoding="utf-8")

    print(f"✅ 看板已生成")
    print(f"   JSON: {out_dir / 'data.json'} ({Path(out_dir / 'data.json').stat().st_size / 1024:.1f} KB)")
    print(f"   HTML: {PROJ / 'index.html'} ({Path(PROJ / 'index.html').stat().st_size / 1024:.1f} KB)")
    print(f"   净值点数: {len(data['nav']['dates'])}  覆盖年份: {len(data['annual_returns'])}")

    # 主力追踪状态
    nt = data.get("national_team", {})
    if nt.get("available"):
        print(f"   主力追踪: ✅ {len(nt.get('share_trends', {}))} 只优先标的 | {len(nt.get('recent_events', []))} 条事件 | 方向={nt.get('direction_sign')}")
    else:
        print(f"   主力追踪: ⚠️ 降级 — {nt.get('reason', '未知')}")

    # 生成周报
    report_path = _generate_weekly_report(nt)
    print(f"   周报: {report_path}")

    # --preview 模式：启动本地 HTTP 服务器
    if args.preview:
        import http.server
        import os
        os.chdir(str(PROJ))
        port = 8000
        print(f"\n{'='*60}")
        print(f"🌐 本地预览服务器已启动")
        print(f"   请在浏览器中打开 http://localhost:{port} 检查效果。")
        print(f"   按 Ctrl+C 结束。")
        print(f"{'='*60}\n")
        handler = http.server.SimpleHTTPRequestHandler
        with http.server.HTTPServer(('', port), handler) as httpd:
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n🛑 预览服务器已停止。")


if __name__ == "__main__":
    main()
