#!/usr/bin/env python3
"""看板数据生成 —— 跑回测 → JSON + 内嵌式 index.html（无 fetch，单文件）"""
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


def _build_data(cfg):
    """Run backtest and extract all dashboard data."""
    result = run_backtest(cfg)
    nav = result.nav_series
    m = result.metrics

    data = {
        "meta": {
            "strategy": cfg.name,
            "version": cfg.version,
            "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "data_range": f"{nav.index[0].date()} ~ {nav.index[-1].date()}",
            "weeks": m["total_weeks"],
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

    # NAV (降采样 ~300 点)
    step = max(1, len(nav) // 260)
    nav_s = nav.iloc[::step]
    data["nav"] = {
        "dates": [str(d.date()) for d in nav_s.index],
        "values": [round(v, 4) for v in nav_s["nav"].tolist()],
    }
    data["defense"] = {
        "dates": [str(d.date()) for d in nav_s.index],
        "ratios": [round(v * 100, 1) for v in nav_s["def_ratio"].tolist()],
    }
    data["drawdown"] = {
        "dates": [str(d.date()) for d in nav_s.index],
        "ratios": [round(v * 100, 2) for v in nav_s["drawdown"].tolist()],
    }
    if "nasdaq_vol" in nav_s.columns:
        data["nasdaq_vol"] = {
            "dates": [str(d.date()) for d in nav_s.index],
            "values": [round(v * 100, 1) for v in nav_s["nasdaq_vol"].tolist()],
        }

    # Holdings (latest)
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
        "pvd_enabled": cfg.pvd_enabled,
        "pvd_w": cfg.pvd_w,
    }
    return data, nav


def _html_template(data_json: str) -> str:
    """Generate the full HTML with embedded data."""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>虾池ETF轮动策略 · 实时看板</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
:root {{ --bg:#0b1121; --card:#151e34; --card-hover:#1c2844; --text:#e8edf5; --muted:#8892a8; --accent:#60a5fa; --green:#34d399; --red:#f87171; --orange:#fb923c; --border:#1e2a45; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Noto Sans SC',sans-serif; background:var(--bg); color:var(--text); padding:24px; min-height:100vh; }}
.container {{ max-width:1280px; margin:0 auto; }}
.header {{ margin-bottom:28px; }}
.header h1 {{ font-size:1.5rem; font-weight:700; background:linear-gradient(135deg,#60a5fa,#a78bfa); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
.header .sub {{ color:var(--muted); font-size:0.85rem; margin-top:4px; }}
.header .sub em {{ color:var(--text); font-style:normal; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:12px; margin-bottom:20px; }}
.card {{ background:var(--card); border-radius:10px; padding:14px 16px; border:1px solid var(--border); transition:border-color .2s,background .2s; }}
.card:hover {{ border-color:var(--accent); background:var(--card-hover); }}
.card .l {{ font-size:0.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:.8px; }}
.card .v {{ font-size:1.35rem; font-weight:700; margin-top:3px; font-variant-numeric:tabular-nums; }}
.card .s {{ font-size:0.7rem; color:var(--muted); margin-top:1px; }}
.g {{ color:var(--green) }} .r {{ color:var(--red) }} .o {{ color:var(--orange) }} .a {{ color:var(--accent) }}
.panel {{ background:var(--card); border-radius:10px; padding:18px; border:1px solid var(--border); }}
.panel h2 {{ font-size:0.8rem; color:var(--muted); margin-bottom:14px; text-transform:uppercase; letter-spacing:.5px; }}
.grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px; }}
.grid-2-1 {{ display:grid; grid-template-columns:2fr 1fr; gap:16px; margin-bottom:16px; }}
@media(max-width:900px){{ .grid-2,.grid-2-1{{ grid-template-columns:1fr; }} }}
.ht {{ width:100%; border-collapse:collapse; font-size:0.82rem; }}
.ht th {{ text-align:left; color:var(--muted); padding:6px 8px; font-weight:500; border-bottom:1px solid var(--border); }}
.ht td {{ padding:6px 8px; border-bottom:1px solid var(--border); }}
.ht tr:last-child td {{ border:none; }}
.hbar {{ background:var(--border); border-radius:4px; height:16px; overflow:hidden; }}
.hfill {{ height:100%; border-radius:4px; }}
.st {{ width:100%; border-collapse:collapse; font-size:0.8rem; }}
.st th {{ text-align:left; color:var(--muted); padding:5px 8px; font-weight:500; border-bottom:1px solid var(--border); }}
.st td {{ padding:5px 8px; border-bottom:1px solid var(--border); }}
.st tr:hover td {{ background:rgba(96,165,250,0.04); }}
.rc {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }}
.rc-item {{ background:rgba(11,17,33,0.6); border-radius:8px; padding:12px; text-align:center; }}
.rc-item .lbl {{ font-size:0.75rem; color:var(--muted); }}
.rc-item .val {{ font-size:1.15rem; font-weight:700; margin-top:4px; }}
.rc-item .sub {{ font-size:0.65rem; color:var(--muted); margin-top:3px; }}
.px {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(100px,1fr)); gap:8px; }}
.px-item {{ background:rgba(11,17,33,0.6); border-radius:6px; padding:6px 8px; text-align:center; }}
.px-item .k {{ font-size:0.65rem; color:var(--muted); }}
.px-item .v {{ font-size:0.82rem; font-weight:600; margin-top:2px; color:var(--accent); }}
.chart-wrap {{ position:relative; width:100%; }}
.chart-wrap canvas {{ width:100% !important; height:auto !important; }}
</style>
</head>
<body>
<div class="container">
<div class="header">
  <h1>🦐 虾池ETF轮动策略 · 实时看板</h1>
  <div class="sub" id="header-sub">加载中...</div>
</div>
<div class="cards" id="metric-cards"></div>

<div class="grid-2-1">
  <div class="panel">
    <h2>📈 净值曲线</h2>
    <div class="chart-wrap"><canvas id="navChart"></canvas></div>
  </div>
  <div class="panel">
    <h2>🎯 当前持仓</h2>
    <div id="holdings-content"></div>
    <div id="offdef-summary" style="margin-top:12px;font-size:0.85rem;display:flex;gap:16px;"></div>
  </div>
</div>

<div class="grid-2">
  <div class="panel">
    <h2>🛡️ 防御比例</h2>
    <div class="chart-wrap"><canvas id="defenseChart"></canvas></div>
  </div>
  <div class="panel">
    <h2>📊 年度收益</h2>
    <div class="chart-wrap"><canvas id="annualChart"></canvas></div>
  </div>
</div>

<div class="grid-2">
  <div class="panel">
    <h2>⏱️ 近期表现</h2>
    <div class="rc" id="recent-cards"></div>
  </div>
  <div class="panel">
    <h2>📋 ETF 全周期统计</h2>
    <table class="st"><thead><tr><th>ETF</th><th>平均权重</th><th>持有周</th><th>持有率</th></tr></thead>
    <tbody id="etf-stats-body"></tbody></table>
  </div>
</div>

<div class="panel" style="margin-bottom:16px;">
  <h2>⚙️ 策略参数</h2>
  <div class="px" id="params-content"></div>
</div>
</div>

<script>
// ===== 数据内嵌 =====
const DATA = {data_json};

// ===== 渲染 =====
(function() {{
  const d = DATA;
  const {{meta,metrics:m,nav:nd,defense:dd,drawdown:dwd,holdings,annual_returns:ann,etf_stats:etfS,recent_ytd:rytd,recent_year1:r1y,recent_month3:r3m,params}} = d;

  document.getElementById('header-sub').innerHTML = `${{meta.strategy}} · <em>v${{meta.version}}</em> · ${{meta.data_range}} · 更新: ${{meta.generated_at}}`;

  const cards = [
    {{l:'夏普比率', v:m.sharpe.toFixed(3), c:'a'}},
    {{l:'年化收益', v:m.annual_return+'%', c:m.annual_return>10?'g':'o'}},
    {{l:'最大回撤', v:m.max_drawdown+'%', c:'r'}},
    {{l:'当前回撤', v:m.current_drawdown+'%', c:m.current_drawdown<2?'g':(m.current_drawdown<4?'o':'r')}},
    {{l:'卡尔马', v:m.calmar.toFixed(2), c:'a'}},
    {{l:'年化波动', v:m.annual_vol+'%', c:'o'}},
    {{l:'周胜率', v:m.win_rate+'%', c:'g'}},
    {{l:'防御周数', v:m.defensive_weeks, s:'/'+meta.weeks+'周', c:'a'}},
  ];
  document.getElementById('metric-cards').innerHTML = cards.map(c =>
    `<div class="card"><div class="l">${{c.l}}</div><div class="v ${{c.c}}">${{c.v}}</div>${{c.s?`<div class="s">${{c.s}}</div>`:''}}</div>`
  ).join('');

  new Chart(document.getElementById('navChart'), {{
    type:'line', data:{{ labels:nd.dates, datasets:[{{ label:'净值', data:nd.values, borderColor:'#60a5fa', backgroundColor:'rgba(96,165,250,0.08)', fill:true, tension:0.1, pointRadius:0, borderWidth:1.5 }}] }},
    options:{{ responsive:true, maintainAspectRatio:true, aspectRatio:2.2, plugins:{{ legend:{{display:false}}, tooltip:{{callbacks:{{label:c=>'NAV: '+c.parsed.y.toFixed(3)+'x'}}}} }}, scales:{{ x:{{ ticks:{{maxTicksLimit:8,color:'#8892a8',font:{{size:9}}}}, grid:{{color:'rgba(30,42,69,0.5)'}} }}, y:{{ ticks:{{color:'#8892a8',font:{{size:9}},callback:v=>v.toFixed(1)+'x'}}, grid:{{color:'rgba(30,42,69,0.5)'}} }} }}, interaction:{{mode:'nearest',axis:'x',intersect:false}} }}
  }});

  // Holdings
  const catClr = {{'进攻':'#f59e0b','防御':'#60a5fa'}};
  let htm = '<table class="ht"><thead><tr><th>ETF</th><th>仓位</th><th></th></tr></thead><tbody>';
  for (const h of holdings) {{
    if (h.weight<0.1) continue;
    const clr=catClr[h.category];
    htm += `<tr><td><span style="color:${{clr}}">${{h.category==='进攻'?'⚔️':'🛡️'}}</span> ${{h.name}}</td><td style="font-weight:600">${{h.weight.toFixed(1)}}%</td><td><div class="hbar"><div class="hfill" style="width:${{h.weight>5?h.weight:5}}%;background:${{clr}}"></div></div></td></tr>`;
  }}
  htm += '</tbody></table>';
  document.getElementById('holdings-content').innerHTML = htm;
  const off=holdings.filter(h=>h.category==='进攻').reduce((s,h)=>s+h.weight,0);
  const def=holdings.filter(h=>h.category==='防御').reduce((s,h)=>s+h.weight,0);
  document.getElementById('offdef-summary').innerHTML =
    `<span style="color:#f59e0b">⚔️ 进攻 ${{off.toFixed(1)}}%</span><span style="color:#60a5fa">🛡️ 防御 ${{def.toFixed(1)}}%</span>`;

  new Chart(document.getElementById('defenseChart'), {{
    type:'line', data:{{ labels:dd.dates, datasets:[{{ label:'防御比', data:dd.ratios, borderColor:'#60a5fa', backgroundColor:'rgba(96,165,250,0.12)', fill:true, tension:0.1, pointRadius:0, borderWidth:1.2 }}] }},
    options:{{ responsive:true, maintainAspectRatio:true, aspectRatio:2.5, plugins:{{legend:{{display:false}}}}, scales:{{ x:{{ ticks:{{maxTicksLimit:8,color:'#8892a8',font:{{size:9}}}}, grid:{{color:'rgba(30,42,69,0.5)'}} }}, y:{{ min:0, max:100, ticks:{{color:'#8892a8',font:{{size:9}},callback:v=>v+'%'}}, grid:{{color:'rgba(30,42,69,0.5)'}} }} }} }}
  }});

  new Chart(document.getElementById('annualChart'), {{
    type:'bar', data:{{ labels:ann.map(a=>a.year), datasets:[{{ label:'年收益', data:ann.map(a=>a.return), backgroundColor:ann.map(a=>a.return>=0?'rgba(52,211,153,0.7)':'rgba(248,113,113,0.7)'), borderColor:ann.map(a=>a.return>=0?'#34d399':'#f87171'), borderWidth:1, borderRadius:3 }}] }},
    options:{{ responsive:true, maintainAspectRatio:true, aspectRatio:2.5, plugins:{{ legend:{{display:false}}, tooltip:{{callbacks:{{afterLabel:ctx=>'平均防御: '+ann[ctx.dataIndex].avg_defense+'%'}}}} }}, scales:{{ x:{{ ticks:{{color:'#8892a8',font:{{size:9}}}}, grid:{{display:false}} }}, y:{{ ticks:{{color:'#8892a8',font:{{size:9}},callback:v=>v+'%'}}, grid:{{color:'rgba(30,42,69,0.5)'}} }} }} }}
  }});

  document.getElementById('recent-cards').innerHTML =
    [{{l:'今年 (YTD)',d:rytd}},{{l:'近1年',d:r1y}},{{l:'近3月',d:r3m}}].map(r =>
      `<div class="rc-item"><div class="lbl">${{r.l}}</div><div class="val ${{r.d.return>0?'g':'r'}}">${{r.d.return>0?'+':''}}${{r.d.return}}%</div><div class="sub">最大回撤 ${{r.d.max_drawdown}}% · ${{r.d.start}}</div></div>`
    ).join('');

  document.getElementById('etf-stats-body').innerHTML =
    etfS.map(e=>`<tr><td>${{e.name}}</td><td>${{e.avg_weight}}%</td><td>${{e.held_weeks}}</td><td>${{e.held_pct}}%</td></tr>`).join('');

  const pl = {{ mom_w:'动量权重',vol_w:'波动权重',top_n:'选TOP-N',score_margin:'分数门槛',rebalance_threshold:'调仓阈值',max_single_alloc:'单标上限',def_alloc:'基准防御',step_low:'防御下限',step_high:'防御上限',max_def:'最大防御',inv_vol_window:'InvVol窗口',vol_taper_enabled:'Taper',vol_taper_window:'Taper窗口',vol_taper_len:'Taper降权',pvd_enabled:'PVD',pvd_w:'PVD权重' }};
  document.getElementById('params-content').innerHTML =
    Object.entries(params).filter(([k])=>pl[k]).map(([k,v])=>`<div class="px-item"><div class="k">${{pl[k]}}</div><div class="v">${{typeof v==='boolean'?(v?'✅':'❌'):v}}</div></div>`).join('');
}})();
</script>
</body>
</html>"""


def main():
    cfg = load_config(PROJ / "config" / "strategy_v4_5_pvd.yaml")
    data, nav = _build_data(cfg)

    # 写 JSON (for other uses)
    out_dir = PROJ / "dashboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # 写 index.html (单文件，数据内嵌)
    html = _html_template(json.dumps(data, ensure_ascii=False))
    (PROJ / "index.html").write_text(html, encoding="utf-8")

    print(f"✅ 看板已生成")
    print(f"   JSON: {out_dir / 'data.json'} ({Path(out_dir / 'data.json').stat().st_size / 1024:.1f} KB)")
    print(f"   HTML: {PROJ / 'index.html'} ({Path(PROJ / 'index.html').stat().st_size / 1024:.1f} KB)")
    print(f"   净值点数: {len(data['nav']['dates'])}  覆盖年份: {len(data['annual_returns'])}")


if __name__ == "__main__":
    main()