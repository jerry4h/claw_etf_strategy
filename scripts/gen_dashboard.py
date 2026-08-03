#!/usr/bin/env python3
"""Dashboard generator — runs backtest → JSON + embedded single-file index.html"""

import json, sys
from pathlib import Path
import numpy as np, pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.strategy import load_config
from src.backtest import run_backtest
from src.data_loader import ETFS, OFFENSIVE, DEFENSIVE, load_nav_data, resample_weekly


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

    # Downsample
    step = max(1, len(nav) // 260)
    nav_s = nav.iloc[::step]

    data["nav"] = {"dates": [str(d.date()) for d in nav_s.index],
                   "values": [round(v, 4) for v in nav_s["nav"].tolist()]}
    data["defense"] = {"dates": [str(d.date()) for d in nav_s.index],
                       "ratios": [round(v * 100, 1) for v in nav_s["def_ratio"].tolist()]}
    data["drawdown"] = {"dates": [str(d.date()) for d in nav_s.index],
                        "ratios": [round(v * 100, 2) for v in nav_s["drawdown"].tolist()]}
    if "nasdaq_vol" in nav_s.columns:
        data["nasdaq_vol"] = {"dates": [str(d.date()) for d in nav_s.index],
                              "values": [round(v * 100, 1) for v in nav_s["nasdaq_vol"].tolist()]}

    # Equal-weight weekly rebalance baseline
    nav_df = load_nav_data(PROJ / cfg.nav_path)
    weekly_raw = resample_weekly(nav_df, anchor=cfg.anchor)
    if cfg.start_date:
        weekly_raw = weekly_raw[weekly_raw.index >= pd.to_datetime(cfg.start_date)]
    weekly_aligned = weekly_raw.loc[nav.index[0]:nav.index[-1]]
    first_nav = np.mean(weekly_aligned.iloc[0].values)
    eq_vals = [round(np.mean(weekly_aligned.loc[dt].values) / first_nav, 4) for dt in nav_s.index]
    data["eq_nav"] = {"dates": data["nav"]["dates"], "values": eq_vals}

    # Holdings
    latest = nav.iloc[-1]
    data["holdings"] = [
        {"name": e, "weight": round(latest.get(f"weight_{e}", 0) * 100, 1),
         "category": "进攻" if e in OFFENSIVE else "防御"}
        for e in ETFS
    ]

    # Annual returns
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

    # ETF stats
    data["etf_stats"] = []
    for e in ETFS:
        col = f"weight_{e}"
        if col in nav.columns:
            aw = nav[col].mean()
            hw = int((nav[col] > 0.001).sum())
            data["etf_stats"].append({
                "name": e, "avg_weight": round(aw * 100, 1),
                "held_weeks": hw, "held_pct": round(hw / len(nav) * 100, 1),
            })

    # Recent windows
    now = nav.index[-1]
    for label, start_fn in [
        ("ytd", lambda: pd.Timestamp(f"{now.year}-01-01")),
        ("year1", lambda: now - pd.DateOffset(years=1)),
        ("month3", lambda: now - pd.DateOffset(months=3)),
    ]:
        seg = nav[nav.index >= start_fn()]
        if len(seg) > 0:
            sr = seg["nav"].iloc[-1] / seg["nav"].iloc[0] - 1
            sd = seg["drawdown"].max()
            data[f"recent_{label}"] = {
                "start": str(seg.index[0].date()),
                "end": str(seg.index[-1].date()),
                "return": round(sr * 100, 2),
                "max_drawdown": round(sd * 100, 2),
            }

    # Params
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
    return data


def _html_template(data_json):
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>虾池ETF轮动策略 · 实时看板</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
:root {{--bg:#0b1121;--card:#151e34;--card-hover:#1c2844;--text:#e8edf5;--muted:#8892a8;--accent:#60a5fa;--green:#34d399;--red:#f87171;--orange:#fb923c;--border:#1e2a45;}}
* {{margin:0;padding:0;box-sizing:border-box;}}
body {{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Noto Sans SC',sans-serif;background:var(--bg);color:var(--text);padding:24px;min-height:100vh;}}
.container {{max-width:1280px;margin:0 auto;}}
.header {{margin-bottom:28px;}}
.header h1 {{font-size:1.5rem;font-weight:700;background:linear-gradient(135deg,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}}
.header .sub {{color:var(--muted);font-size:0.85rem;margin-top:4px;}}
.header .sub em {{color:var(--text);font-style:normal;}}
.header .asof {{margin-top:8px;font-size:0.95rem;font-weight:600;color:var(--accent);}}
.cards {{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-bottom:20px;}}
.card {{background:var(--card);border-radius:10px;padding:14px 16px;border:1px solid var(--border);transition:border-color .2s,background .2s;}}
.card:hover {{border-color:var(--accent);background:var(--card-hover);}}
.card .l {{font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;}}
.card .v {{font-size:1.35rem;font-weight:700;margin-top:3px;font-variant-numeric:tabular-nums;}}
.card .s {{font-size:0.7rem;color:var(--muted);margin-top:1px;}}
.card .v.big {{font-size:1.7rem;}}
.g {{color:var(--green)}}.r {{color:var(--red)}}.o {{color:var(--orange)}}.a {{color:var(--accent)}}
.panel {{background:var(--card);border-radius:10px;padding:18px;border:1px solid var(--border);}}
.panel h2 {{font-size:0.8rem;color:var(--muted);margin-bottom:14px;text-transform:uppercase;letter-spacing:.5px;}}
.panel h2 .tgl {{float:right;background:transparent;border:1px solid var(--border);color:var(--muted);font-size:0.68rem;padding:2px 10px;border-radius:6px;cursor:pointer;letter-spacing:0;text-transform:none;transition:color .2s,border-color .2s;}}
.panel h2 .tgl:hover {{color:var(--accent);border-color:var(--accent);}}
.chart-err {{padding:48px 12px;text-align:center;color:var(--muted);font-size:0.85rem;border:1px dashed var(--border);border-radius:8px;}}
.grid-2 {{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;}}
.grid-2-1 {{display:grid;grid-template-columns:2fr 1fr;gap:16px;margin-bottom:16px;}}
@media(max-width:900px){{.grid-2,.grid-2-1{{grid-template-columns:1fr;}}}}
@media(max-width:600px){{body{{padding:14px;}}.px{{grid-template-columns:repeat(2,1fr);}}.rc{{grid-template-columns:1fr;}}}}
.ht {{width:100%;border-collapse:collapse;font-size:0.82rem;}}
.ht th {{text-align:left;color:var(--muted);padding:6px 8px;font-weight:500;border-bottom:1px solid var(--border);}}
.ht td {{padding:6px 8px;border-bottom:1px solid var(--border);}}
.ht tr:last-child td {{border:none;}}
.hbar {{background:var(--border);border-radius:4px;height:16px;overflow:hidden;}}
.hfill {{height:100%;border-radius:4px;}}
.st {{width:100%;border-collapse:collapse;font-size:0.8rem;}}
.st th {{text-align:left;color:var(--muted);padding:5px 8px;font-weight:500;border-bottom:1px solid var(--border);}}
.st td {{padding:5px 8px;border-bottom:1px solid var(--border);}}
.st tr:hover td {{background:rgba(96,165,250,0.04);}}
.rc {{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}}
.rc-item {{background:rgba(11,17,33,0.6);border-radius:8px;padding:12px;text-align:center;}}
.rc-item .lbl {{font-size:0.75rem;color:var(--muted);}}
.rc-item .val {{font-size:1.15rem;font-weight:700;margin-top:4px;}}
.rc-item .sub {{font-size:0.65rem;color:var(--muted);margin-top:3px;}}
.px {{display:grid;grid-template-columns:repeat(auto-fill,minmax(100px,1fr));gap:8px;}}
.px-item {{background:rgba(11,17,33,0.6);border-radius:6px;padding:6px 8px;text-align:center;}}
.px-item .k {{font-size:0.65rem;color:var(--muted);}}
.px-item .v {{font-size:0.82rem;font-weight:600;margin-top:2px;color:var(--accent);}}
.chart-wrap {{position:relative;width:100%;min-height:100px;}}
.chart-wrap canvas {{display:block;width:100% !important;}}
.chart-stack > .chart-wrap:nth-child(1) canvas {{aspect-ratio:2.2/1;}}
.chart-stack > .chart-wrap:nth-child(2) {{min-height:60px;}}
.chart-stack > .chart-wrap:nth-child(3) {{min-height:50px;}}
.chart-stack > .chart-wrap:nth-child(3) canvas {{aspect-ratio:6/1;}}
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
    <table class="st"><thead><tr><th>ETF</th><th>平均权重</th><th>持有周</th><th>持有率</th></tr></thead>
    <tbody id="etf-stats-body"></tbody></table>
  </div>
  <div class="panel">
    <h2>⚙️ 策略参数</h2>
    <div class="px" id="params-content"></div>
  </div>
</div>

</div>

<script>
const DATA = {data_json};
(function() {{
  const d = DATA;
  const {{meta,metrics:m,nav:nd,defense:dd,drawdown:dwd,eq_nav:eqd,
         holdings,annual_returns:ann,etf_stats:etfS,
         recent_ytd:rytd,recent_year1:r1y,recent_month3:r3m,params}} = d;

  // ---- Header ----
  const endDate = (meta.data_range || '').split(' ~ ')[1] || '';
  const wd = endDate ? '日一二三四五六'[new Date(endDate+'T00:00:00').getDay()] : '';
  document.getElementById('header-sub').innerHTML =
    `${{meta.strategy}} · <em>v${{meta.version}}</em>${{meta.version_note?`（${{meta.version_note}}）`:''}}<br>` +
    `数据源: ${{meta.data_source||'Tushare'}} · 周频调仓 · 区间 ${{meta.data_range}} · 生成: ${{meta.generated_at}}`;
  document.getElementById('header-asof').textContent =
    endDate ? `📅 数据截至 ${{endDate}}（周${{wd}}收盘）` : '';

  // ---- Metric cards ----
  const fnum = (v,dig) => Number.isFinite(v) ? v.toFixed(dig) : '—';
  const fpct = v => Number.isFinite(v) ? v+'%' : '—';
  const cards = [
    {{l:'夏普比率',  v:fnum(m.sharpe,3),        c:'a', big:1}},
    {{l:'年化收益',  v:fpct(m.annual_return),    c:m.annual_return>10?'g':'o', big:1}},
    {{l:'最大回撤',  v:fpct(m.max_drawdown),     c:'r', big:1}},
    {{l:'当前回撤',  v:fpct(m.current_drawdown), c:m.current_drawdown<2?'g':(m.current_drawdown<4?'o':'r')}},
    {{l:'卡尔马',    v:fnum(m.calmar,2),         c:'a'}},
    {{l:'年化波动',  v:fpct(m.annual_vol),       c:'o'}},
    {{l:'周胜率',    v:fpct(m.win_rate),         c:'g'}},
    {{l:'防御周数',  v:m.defensive_weeks,        s:'/'+meta.weeks+'周', c:'a'}},
  ];
  document.getElementById('metric-cards').innerHTML =
    cards.map(c=>`<div class="card"><div class="l">${{c.l}}</div><div class="v ${{c.c}}${{c.big?' big':''}}">${{c.v}}</div>${{c.s?`<div class="s">${{c.s}}</div>`:''}}</div>`).join('');

  // ---- Chart presets ----
  const TIP_ANIM = {{duration:0}};
  const CHART_OPT = {{
    responsive:true, maintainAspectRatio:false, resizeDelay:200,
    animation:{{duration:0}}, transitions:{{active:{{animation:{{duration:0}}}}}},
    plugins:{{legend:{{display:false}}, tooltip:{{animation:TIP_ANIM}}}},
    scales:{{x:{{ticks:{{maxTicksLimit:8,color:'#8892a8',font:{{size:9}}}}, grid:{{color:'rgba(30,42,69,0.5)'}}}},
              y:{{ticks:{{color:'#8892a8',font:{{size:9}}}}, grid:{{color:'rgba(30,42,69,0.5)'}}}}}},
    interaction:{{mode:'nearest',axis:'x',intersect:false}},
  }};
  const vline = {{
    id:'vline', afterDatasetsDraw(ch) {{
      const act = ch.tooltip?.getActiveElements?.() || [];
      if (!act.length) return;
      const x = act[0].element.x, ctx = ch.ctx, ca = ch.chartArea;
      ctx.save(); ctx.strokeStyle='rgba(136,146,168,0.4)'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
      ctx.beginPath(); ctx.moveTo(x, ca.top); ctx.lineTo(x, ca.bottom); ctx.stroke(); ctx.restore();
    }}
  }};
  const clean = (dates, vals) => {{
    const D=[], V=[]; (dates||[]).forEach((dt,i)=>{{const v=(vals||[])[i]; if(v!==null&&v!==undefined&&isFinite(v)){{D.push(dt);V.push(v);}}}}); return {{dates:D, vals:V}};
  }};
  const chartMsg = (id, msg) => {{ const el=document.getElementById(id); if(el) el.parentElement.innerHTML=`<div class="chart-err">${{msg}}</div>`; }};
  const canChart = typeof Chart !== 'undefined';
  const CDN_ERR = '⚠️ 图表库加载失败，请检查网络后刷新';

  // ---- Crosshair sync across stacked charts ----
  const stkCharts = [];
  const syncCX = (src) => {{
    if (!src || !stkCharts.length) return;
    const act = src.getActiveElements();
    if (!act.length) return;
    const idx = act[0].index;
    stkCharts.forEach(c => {{
      if (c === src) return;
      c.tooltip.setActiveElements([{{datasetIndex:0, index:idx}}], {{x:0, y:0}});
      c.draw();
    }});
  }};
  const ON_HOVER = (e, el, chart) => syncCX(chart);

  // ---- 1. NAV chart (strategy + equal-weight baseline) ----
  const ns = clean(nd?.dates, nd?.values);
  const es = clean(eqd?.dates, eqd?.values);
  if (canChart && ns.dates.length) {{
    const nc = new Chart(document.getElementById('navChart'), {{
      type:'line', data:{{
        labels:ns.dates,
        datasets:[
          {{label:'策略', data:ns.vals, borderColor:'#60a5fa', backgroundColor:'rgba(96,165,250,0.08)', fill:true, tension:0.1, pointRadius:0, borderWidth:1.5}},
          {{label:'等权再均衡', data:es.vals.length===ns.vals.length?es.vals:[], borderColor:'rgba(136,146,168,0.4)', borderDash:[4,3], tension:0.1, pointRadius:0, borderWidth:1, fill:false}},
        ]
      }},
      options:{{
        ...CHART_OPT,
        plugins:{{...CHART_OPT.plugins,
          legend:{{display:true, position:'top', labels:{{color:'var(--text)', font:{{size:9}}, boxWidth:12, boxHeight:2, padding:8, usePointStyle:true, pointStyle:'dash'}}}},
          tooltip:{{...CHART_OPT.plugins.tooltip, callbacks:{{label:ctx=>ctx.dataset.label+': '+ctx.parsed.y.toFixed(3)+'x'}}}},
        }},
        scales:{{...CHART_OPT.scales, y:{{...CHART_OPT.scales.y, type:'logarithmic', ticks:{{...CHART_OPT.scales.y.ticks, callback:v=>Number(v).toFixed(1)+'x'}}}}}},
        onHover:ON_HOVER,
      }},
      plugins:[vline]
    }});
    stkCharts.push(nc);
  }} else if (!canChart) chartMsg('navChart', CDN_ERR);

  // ---- 2. Drawdown chart ----
  const ds = clean(dwd?.dates, dwd?.ratios);
  if (canChart && ds.dates.length) {{
    const dc = new Chart(document.getElementById('ddChart'), {{
      type:'line', data:{{labels:ds.dates, datasets:[{{label:'回撤', data:ds.vals, borderColor:'#f87171', backgroundColor:'rgba(248,113,113,0.15)', fill:true, tension:0.1, pointRadius:0, borderWidth:1}}]}},
      options:{{
        ...CHART_OPT,
        plugins:{{...CHART_OPT.plugins, tooltip:{{...CHART_OPT.plugins.tooltip, callbacks:{{label:c=>'回撤: '+c.parsed.y.toFixed(2)+'%'}}}}}},
        scales:{{...CHART_OPT.scales, y:{{...CHART_OPT.scales.y, min:0, reverse:true, ticks:{{...CHART_OPT.scales.y.ticks, callback:v=>v+'%'}}}}}},
        onHover:ON_HOVER,
      }},
      plugins:[vline]
    }});
    stkCharts.push(dc);
  }} else if (!canChart) chartMsg('ddChart', CDN_ERR);

  // ---- 3. Defense ratio chart ----
  const dds = clean(dd?.dates, dd?.ratios);
  if (canChart && dds.dates.length) {{
    const dfc = new Chart(document.getElementById('defChart'), {{
      type:'line', data:{{labels:dds.dates, datasets:[{{label:'防御比', data:dds.vals, borderColor:'#34d399', backgroundColor:'rgba(52,211,153,0.12)', fill:true, tension:0.1, pointRadius:0, borderWidth:1}}]}},
      options:{{
        ...CHART_OPT,
        plugins:{{...CHART_OPT.plugins, tooltip:{{...CHART_OPT.plugins.tooltip, callbacks:{{label:c=>'防御比: '+c.parsed.y.toFixed(1)+'%'}}}}}},
        scales:{{...CHART_OPT.scales, y:{{...CHART_OPT.scales.y, min:0, max:100, ticks:{{...CHART_OPT.scales.y.ticks, callback:v=>v+'%'}}}}}},
        onHover:ON_HOVER,
      }},
      plugins:[vline]
    }});
    stkCharts.push(dfc);
  }} else if (!canChart) chartMsg('defChart', CDN_ERR);

  // ---- Scale toggle ----
  const st = document.getElementById('scale-toggle');
  let logScale = true;
  const nc = Chart.getChart('navChart');
  if (nc) {{
    const apply = () => {{
      nc.options.scales.y.type = logScale ? 'logarithmic' : 'linear';
      st.textContent = logScale ? '对数坐标' : '线性坐标';
      nc.update();
    }};
    apply();
    st.onclick = () => {{ logScale = !logScale; apply(); }};
  }} else st.style.display = 'none';

  // ---- Holdings ----
  const catClr = {{'进攻':'#f59e0b','防御':'#60a5fa'}};
  let htm = '<table class="ht"><thead><tr><th>ETF</th><th>仓位</th><th></th></tr></thead><tbody>';
  for (const h of holdings) {{
    if (h.weight<0) continue;
    const clr=catClr[h.category];
    const zero = h.weight<0.05;
    htm += `<tr${{zero?' style="color:var(--muted)"':''}}><td><span style="color:${{zero?'var(--muted)':clr}}">${{h.category==='进攻'?'⚔️':'🛡️'}}</span> ${{h.name}}</td><td style="font-weight:600">${{h.weight.toFixed(1)}}%</td><td><div class="hbar">${{zero?'':`<div class="hfill" style="width:${{h.weight>5?h.weight:5}}%;background:${{clr}}"></div>`}}</div></td></tr>`;
  }}
  htm += '</tbody></table>';
  document.getElementById('holdings-content').innerHTML = htm;
  const off = holdings.filter(h=>h.category==='进攻').reduce((s,h)=>s+h.weight,0);
  const def = holdings.filter(h=>h.category==='防御').reduce((s,h)=>s+h.weight,0);
  document.getElementById('offdef-summary').innerHTML =
    `<span style="color:#f59e0b">⚔️ 进攻 ${{off.toFixed(1)}}%</span><span style="color:#60a5fa">🛡️ 防御 ${{def.toFixed(1)}}%</span>`;

  // ---- Annual returns ----
  const endD = endDate ? new Date(endDate+'T00:00:00') : null;
  const annArr = ann || [];
  const annLabels = annArr.map(a => (endD && a.year===endD.getFullYear() && endD.getMonth()<11) ? a.year+' YTD' : String(a.year));
  if (canChart && annArr.length) {{
    new Chart(document.getElementById('annualChart'), {{
      type:'bar', data:{{labels:annLabels, datasets:[{{label:'年收益', data:annArr.map(a=>a.return), backgroundColor:annArr.map(a=>a.return>=0?'rgba(52,211,153,0.7)':'rgba(248,113,113,0.7)'), borderColor:annArr.map(a=>a.return>=0?'#34d399':'#f87171'), borderWidth:1, borderRadius:3}}]}},
      options:{{
        ...CHART_OPT,
        plugins:{{...CHART_OPT.plugins, legend:{{display:false}}, tooltip:{{...CHART_OPT.plugins.tooltip, callbacks:{{afterLabel:ctx=>'平均防御: '+annArr[ctx.dataIndex].avg_defense+'%'}}}}}},
        scales:{{...CHART_OPT.scales, x:{{...CHART_OPT.scales.x, grid:{{display:false}}}}, y:{{...CHART_OPT.scales.y, ticks:{{...CHART_OPT.scales.y.ticks, callback:v=>v+'%'}}}}}},
      }},
    }});
  }} else if (!canChart) chartMsg('annualChart', CDN_ERR);

  // ---- Recent performance ----
  document.getElementById('recent-cards').innerHTML =
    [{{l:'今年 (YTD)',d:rytd}},{{l:'近1年',d:r1y}},{{l:'近3月',d:r3m}}].filter(r=>r.d).map(r=>
      `<div class="rc-item"><div class="lbl">${{r.l}}</div><div class="val ${{r.d.return>0?'g':'r'}}">${{r.d.return>0?'+':''}}${{r.d.return}}%</div><div class="sub">最大回撤 ${{r.d.max_drawdown}}% · ${{r.d.start}}</div></div>`
    ).join('');

  // ---- ETF stats ----
  document.getElementById('etf-stats-body').innerHTML =
    etfS.map(e=>`<tr><td>${{e.name}}</td><td>${{e.avg_weight}}%</td><td>${{e.held_weeks}}</td><td>${{e.held_pct}}%</td></tr>`).join('');

  // ---- Params ----
  const pl = {{
    mom_w:'动量权重', vol_w:'波动权重', top_n:'选TOP-N',
    score_margin:'分数门槛', rebalance_threshold:'调仓阈值',
    max_single_alloc:'单标上限', def_alloc:'基准防御',
    step_low:'防御下限', step_high:'防御上限', max_def:'最大防御',
    inv_vol_window:'InvVol窗口', vol_taper_enabled:'Taper',
    vol_taper_window:'Taper窗口', vol_taper_len:'Taper降权',
    pvd_enabled:'PVD', pvd_w:'PVD权重'
  }};
  document.getElementById('params-content').innerHTML =
    Object.entries(params).filter(([k])=>pl[k]).map(([k,v])=>`<div class="px-item"><div class="k">${{pl[k]}}</div><div class="v">${{typeof v==='boolean'?(v?'✅':'❌'):v}}</div></div>`).join('');

}})();
</script>
</body>
</html>"""


def main():
    cfg = load_config(PROJ / "config" / "strategy_v4_5_pvd.yaml")
    data = _build_data(cfg)
    data["meta"]["version_note"] = "候选版（PVD 量价因子增强）"

    out_dir = PROJ / "dashboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    html = _html_template(json.dumps(data, ensure_ascii=False))
    (PROJ / "index.html").write_text(html, encoding="utf-8")

    print(f"✅ 看板已生成")
    print(f"   JSON: {out_dir / 'data.json'} ({Path(out_dir / 'data.json').stat().st_size / 1024:.1f} KB)")
    print(f"   HTML: {PROJ / 'index.html'} ({Path(PROJ / 'index.html').stat().st_size / 1024:.1f} KB)")
    print(f"   净值点数: {len(data['nav']['dates'])}  覆盖年份: {len(data['annual_returns'])}")


if __name__ == "__main__":
    main()