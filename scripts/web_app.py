#!/usr/bin/env python3
"""
虾池ETF轮动 v3.0 — Web 调仓助手

功能:
  - 查看当前持仓、历史数据
  - 在线录入新一周价格
  - 自动计算下周一调仓方案

启动: python scripts/web_app.py
访问: http://<IP>:8766
"""

import argparse
import csv
import hashlib
import hmac
import math
import os
import sys
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template_string, abort

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.data_loader import ETFS, OFFENSIVE, DEFENSIVE
from src.utils import compute_sharpe, annualize_return
from src.factors import calculate_momentum, calculate_volatility
from src.strategy import load_config

# ── 加载策略参数 ──
cfg = load_config(PROJECT / 'config/strategy_v3_0_final.yaml')
MOM_W = cfg.mom_w
VOL_W = cfg.vol_w
TOP_N = cfg.top_n
INV_VOL_W = cfg.inv_vol_window
DEF_ALLOC = cfg.def_alloc
STEP_LOW = cfg.step_low
STEP_HIGH = cfg.step_high
MAX_DEF = cfg.max_def
MAX_SINGLE = cfg.max_single_alloc
REBAL_THRESH = cfg.rebalance_threshold
FEE = cfg.fee_rate
RISK_FREE = cfg.risk_free_rate

DEFAULT_CSV = cfg.nav_path
# If DEFAULT_CSV is relative, resolve against project root
if not Path(DEFAULT_CSV).is_absolute():
    DEFAULT_CSV = str(PROJECT / DEFAULT_CSV)

# ── 核心计算（与 rebalance_live.py 一致）──

def load_csv(csv_path=None):
    path = csv_path or DEFAULT_CSV
    df = pd.read_csv(path)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.set_index('日期').sort_index()
    for c in ETFS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df[ETFS].ffill()

def engine_factors(nav):
    m4 = calculate_momentum(nav, window=4)
    v20 = calculate_volatility(nav, window=20)
    prices = nav.values
    wr_df = pd.DataFrame(
        np.diff(prices, axis=0) / prices[:-1],
        index=nav.index[1:], columns=ETFS
    )
    return wr_df, m4, v20

def defense_ratio(v_nasdaq):
    if pd.isna(v_nasdaq):
        return DEF_ALLOC
    if v_nasdaq < STEP_LOW:
        return DEF_ALLOC
    if v_nasdaq > STEP_HIGH:
        return MAX_DEF
    return DEF_ALLOC + (v_nasdaq - STEP_LOW) / (STEP_HIGH - STEP_LOW) * (MAX_DEF - DEF_ALLOC)

def invvol_weights(selected, wr, i):
    iv = {}
    for e in selected:
        start = max(0, i - 1 - INV_VOL_W + 1)
        end = i
        s = wr[e].iloc[start:end].dropna()
        v = np.std(s.values, ddof=0) * math.sqrt(52) if len(s) >= 3 else 0.20
        iv[e] = 1.0 / max(v, 0.05)
    t = sum(iv.values())
    if t <= 0:
        return {e: 1.0 / max(len(selected), 1) for e in selected}
    return {e: w / t for e, w in iv.items()}

def compute_allocation(nav, i):
    wr, m4, v20 = engine_factors(nav)
    sc = {}
    for e in OFFENSIVE:
        mv = m4[e].iloc[i] if i < len(m4) else m4[e].iloc[-1]
        vv = v20[e].iloc[i] if i < len(v20) else v20[e].iloc[-1]
        if pd.notna(mv) and pd.notna(vv):
            sc[e] = MOM_W * mv - VOL_W * vv
    ranked = sorted(sc, key=lambda e: sc[e], reverse=True)
    sel = ranked[:TOP_N]
    if not sel:
        return None, None, None, None, None
    def_r = defense_ratio(v20['纳指ETF'].iloc[i])
    wts = invvol_weights(sel, wr, i)
    alloc = {e: def_r / len(DEFENSIVE) for e in DEFENSIVE}
    off_t = 1.0 - def_r
    for e, w in wts.items():
        alloc[e] = alloc.get(e, 0) + w * off_t
    for e in alloc:
        alloc[e] = min(alloc[e], MAX_SINGLE)
    tot = sum(alloc.values())
    if tot < 1.0:
        df_total = sum(alloc.get(e, 0) for e in DEFENSIVE)
        if df_total > 0:
            excess = 1.0 - tot
            for e in DEFENSIVE:
                alloc[e] += excess * alloc[e] / df_total
    return alloc, sc, wr, m4, v20

# ── 认证 ──

def check_auth(username, password):
    """Simple auth: use env vars HERMES_WEB_USER / HERMES_WEB_PASS"""
    expected_user = os.environ.get('HERMES_WEB_USER', 'admin')
    expected_pass = os.environ.get('HERMES_WEB_PASS', 'claw2026')
    return hmac.compare_digest(username, expected_user) and \
           hmac.compare_digest(password, expected_pass)

def auth_required(f):
    """Decorator for basic auth"""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return jsonify({'error': '认证失败'}), 401, {
                'WWW-Authenticate': 'Basic realm="虾池ETF轮动"'
            }
        return f(*args, **kwargs)
    return decorated

# ── Flask App ──

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # keep Chinese chars

# ── API 路由 ──

@app.route('/')
@auth_required
def index():
    return render_template_string(HTML, now=datetime.now().strftime('%Y-%m-%d %H:%M'))

@app.route('/api/data')
@auth_required
def get_data():
    df = load_csv()
    # Latest rows
    latest = df.tail(20).reset_index()
    data = []
    for _, row in latest.iterrows():
        r = {'日期': row['日期'].strftime('%Y-%m-%d')}
        for e in ETFS:
            r[e] = round(float(row[e]), 4)
        data.append(r)

    # Compute current allocation
    idx = len(df) - 1
    alloc, sc, wr, m4, v20 = compute_allocation(df, idx)

    current = {}
    if alloc:
        current = {e: round(alloc[e] * 100, 1) for e in alloc if e in ETFS}
        current['防御率'] = round(sum(alloc.get(e, 0) for e in DEFENSIVE) * 100, 1)
        current['基准日期'] = df.index[idx].strftime('%Y-%m-%d')

    # Scoring details
    scores = []
    if sc:
        ranked = sorted(sc, key=lambda e: sc[e], reverse=True)
        for e in ranked:
            v20_val = v20[e].iloc[idx] if idx < len(v20) else v20[e].iloc[-1]
            m4_val = m4[e].iloc[idx] if idx < len(m4) else m4[e].iloc[-1]
            scores.append({
                'etf': e, 'mom4': f'{m4_val*100:.2f}%', 'vol20': f'{v20_val*100:.1f}%',
                'score': round(sc[e], 4), 'selected': e in alloc
            })

    last_date = df.index[-1].strftime('%Y-%m-%d')

    return jsonify({
        'latest_rows': data,
        'current': current,
        'scores': scores,
        'tickers': ETFS,
        'offensive': OFFENSIVE,
        'defensive': DEFENSIVE,
        'params': {
            'mom_w': MOM_W, 'vol_w': VOL_W, 'top_n': TOP_N,
            'invvol': INV_VOL_W, 'step_low': STEP_LOW,
            'step_high': STEP_HIGH, 'def_alloc': DEF_ALLOC,
            'max_def': MAX_DEF, 'max_single': MAX_SINGLE,
            'thresh': REBAL_THRESH, 'fee': FEE,
        },
        'last_date': last_date,
        'weeks': len(nav_df),
    })

@app.route('/api/preview', methods=['POST'])
@auth_required
def preview_update():
    """Preview allocation with new week of data (without saving)"""
    req = request.get_json()
    if not req:
        return jsonify({'error': '需要 JSON 数据'}), 400
    missing = [e for e in ETFS if e not in req]
    if missing:
        return jsonify({'error': f'缺少列: {", ".join(missing)}'}), 400

    df = load_csv()
    new_date = req.get('日期', (date.today().isoformat()))
    new_row = {e: float(req[e]) for e in ETFS}

    # Append temporally
    new_idx = pd.to_datetime(new_date)
    df.loc[new_idx] = pd.Series(new_row, index=ETFS)
    df = df.sort_index()
    idx = len(df) - 1

    alloc, sc, wr, m4, v20 = compute_allocation(df, idx)
    if not alloc:
        return jsonify({'error': '数据不足，无法计算调仓'})

    # Also compute previous week's allocation to show changes
    prev_alloc = None
    if idx > 20:
        prev_alloc, _, _, _, _ = compute_allocation(df, idx - 1)

    current = {e: round(alloc[e] * 100, 1) for e in ETFS}
    current['防御率'] = round(sum(alloc.get(e, 0) for e in DEFENSIVE) * 100, 1)

    changes = {}
    if prev_alloc:
        for e in ETFS:
            pw = prev_alloc.get(e, 0) * 100
            cw = alloc.get(e, 0) * 100
            changes[e] = round(cw - pw, 1)
        changes['防御率'] = round(
            sum(alloc.get(e, 0) for e in DEFENSIVE) * 100 -
            sum(prev_alloc.get(e, 0) for e in DEFENSIVE) * 100, 1
        )

    max_chg = max(abs(changes[e]) if e in changes else 0 for e in ETFS) / 100 if changes else 0
    do_rebal = max_chg >= REBAL_THRESH

    scores = []
    if sc:
        ranked = sorted(sc, key=lambda e: sc[e], reverse=True)
        for e in ranked:
            scores.append({
                'etf': e,
                'mom4': f'{m4[e].iloc[idx]*100:.2f}%' if idx < len(m4) else f'{m4[e].iloc[-1]*100:.2f}%',
                'vol20': f'{v20[e].iloc[idx]*100:.1f}%' if idx < len(v20) else f'{v20[e].iloc[-1]*100:.1f}%',
                'score': round(sc[e], 4),
                'selected': e in alloc
            })

    return jsonify({
        'current': current,
        'changes': changes,
        'do_rebalance': bool(do_rebal),
        'max_change': round(max_chg * 100, 1),
        'scores': scores,
        'base_date': df.index[idx].strftime('%Y-%m-%d'),
    })

@app.route('/api/update', methods=['POST'])
@auth_required
def update_data():
    """Save new week of data to CSV"""
    req = request.get_json()
    if not req:
        return jsonify({'error': '需要 JSON 数据'}), 400
    missing = [e for e in ETFS if e not in req]
    if missing:
        return jsonify({'error': f'缺少列: {", ".join(missing)}'}), 400

    new_date = req.get('日期', date.today().isoformat())
    new_row = {e: float(req[e]) for e in ETFS}

    # Validate date is not already in CSV
    df = load_csv()
    if new_date in df.index.strftime('%Y-%m-%d'):
        return jsonify({'error': f'日期 {new_date} 已存在!'}), 409

    # Check date is after last date
    last_csv_date = df.index[-1].strftime('%Y-%m-%d')
    if new_date <= last_csv_date:
        return jsonify({'error': f'新日期 {new_date} 必须晚于最后日期 {last_csv_date}'}), 400

    # Append to CSV
    csv_path = Path(DEFAULT_CSV)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([new_date] + [new_row[e] for e in ETFS])

    # Verify by reloading
    df_new = load_csv()
    new_len = len(df_new)
    old_len = len(df)

    # Calculate new allocation
    idx = new_len - 1
    alloc, sc, wr, m4, v20 = compute_allocation(df_new, idx)

    result = {
        'status': 'ok',
        'message': f'已添加 {new_date} 数据',
        'total_weeks': new_len,
    }
    if alloc:
        current = {e: round(alloc[e] * 100, 1) for e in ETFS}
        current['防御率'] = round(sum(alloc.get(e, 0) for e in DEFENSIVE) * 100, 1)
        result['current'] = current

    return jsonify(result)


# ── HTML 模板 ──

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>虾池ETF轮动 — 调仓助手</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }
  .container { max-width: 1200px; margin: 0 auto; }
  h1 { font-size: 1.5em; color: #58a6ff; margin-bottom: 5px; }
  .subtitle { color: #8b949e; font-size: 0.85em; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
  @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .card h2 { font-size: 1em; color: #58a6ff; margin-bottom: 10px; border-bottom: 1px solid #30363d; padding-bottom: 6px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
  th { text-align: left; color: #8b949e; font-weight: 500; padding: 4px 8px; border-bottom: 1px solid #21262d; }
  td { padding: 4px 8px; border-bottom: 1px solid #21262d; }
  .num { text-align: right; font-family: 'SF Mono', monospace; }
  .pos { color: #3fb950; }
  .neg { color: #f85149; }
  .highlight { background: #1f2937; font-weight: 600; }
  .selected-row { color: #58a6ff; }
  .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; margin: 10px 0; }
  .form-group label { display: block; font-size: 0.75em; color: #8b949e; margin-bottom: 2px; }
  .form-group input { width: 100%; padding: 6px 8px; background: #0d1117; border: 1px solid #30363d; border-radius: 4px; color: #c9d1d9; font-size: 0.85em; }
  .form-group input:focus { border-color: #58a6ff; outline: none; }
  .btn { padding: 8px 20px; border: 1px solid; border-radius: 6px; cursor: pointer; font-size: 0.85em; font-weight: 500; }
  .btn-primary { background: #238636; border-color: #2ea043; color: #fff; }
  .btn-primary:hover { background: #2ea043; }
  .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-secondary { background: #21262d; border-color: #30363d; color: #c9d1d9; margin-left: 8px; }
  .status { margin: 10px 0; padding: 8px 12px; border-radius: 6px; font-size: 0.85em; }
  .status-ok { background: #1a3a1a; border: 1px solid #2ea043; color: #7ee787; }
  .status-err { background: #3a1a1a; border: 1px solid #f85149; color: #ff7b72; }
  .hidden { display: none; }
  .action-bar { display: flex; gap: 10px; align-items: center; margin-top: 12px; }
  .pct { font-weight: 600; }
  .tag { display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 0.75em; margin-left: 4px; }
  .tag-green { background: #1a3a1a; color: #7ee787; }
  .tag-red { background: #3a1a1a; color: #ff7b72; }
  .tag-yellow { background: #3a3a1a; color: #d29922; }
  .loading { text-align: center; color: #8b949e; padding: 40px; }
  .loading::after { content: '...'; animation: dots 1.5s infinite; }
  @keyframes dots { 0%,20% { content: '.'; } 40% { content: '..'; } 60%,100% { content: '...'; } }
  .param-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 6px; }
  .param-item { font-size: 0.8em; }
  .param-item span { color: #8b949e; }
  .param-item strong { color: #c9d1d9; margin-left: 4px; }
  #preview-result .grid { grid-template-columns: 1fr; }
</style>
</head>
<body>
<div class="container">
  <h1>🦐 虾池ETF轮动 v3.0</h1>
  <div class="subtitle">调仓助手 · <span id="now">{{ now }}</span></div>

  <div class="grid">
    <div class="card" id="current-card">
      <h2>📊 当前持仓</h2>
      <div id="current-status" class="loading">加载中</div>
    </div>

    <div class="card" id="params-card">
      <h2>⚙️ 策略参数</h2>
      <div id="params-status" class="loading">加载中</div>
    </div>
  </div>

  <div class="card" style="margin-bottom: 20px;">
    <h2>📝 录入新一周数据</h2>
    <div class="form-grid" id="input-form">
      <div class="form-group">
        <label>日期</label>
        <input type="date" id="new-date">
      </div>
      <div class="form-group" id="ticker-inputs"></div>
    </div>
    <div id="input-status"></div>
    <div class="action-bar">
      <button class="btn btn-primary" id="preview-btn" onclick="previewUpdate()">👁 预览调仓</button>
      <button class="btn btn-primary" id="save-btn" onclick="saveUpdate()">💾 保存并计算</button>
      <button class="btn btn-secondary" onclick="refreshAll()">⟳ 刷新</button>
    </div>
  </div>

  <div id="preview-result" class="hidden">
    <div class="grid" style="grid-template-columns: 1fr;">
      <div class="card">
        <h2>🔮 预览调仓</h2>
        <div id="preview-content"></div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>📋 最新数据</h2>
    <div id="data-status" class="loading">加载中</div>
  </div>
</div>

<script>
let currentData = null;
let tickers = [];

function getAuth() {
  // Browser will prompt for credentials automatically on 401
  return {};
}

async function api(path, method='GET', body=null) {
  const opts = { method, headers: { 'Accept': 'application/json' } };
  if (body) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const resp = await fetch(path, opts);
  if (resp.status === 401) {
    // Force re-auth by sending dummy credentials to trigger browser prompt
    window.location.reload();
    return null;
  }
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ error: resp.statusText }));
    throw new Error(err.error || '请求失败');
  }
  return await resp.json();
}

async function refreshAll() {
  try {
    document.getElementById('current-status').className = 'loading';
    document.getElementById('current-status').textContent = '加载中';
    document.getElementById('params-status').className = 'loading';
    document.getElementById('params-status').textContent = '加载中';
    document.getElementById('data-status').className = 'loading';
    document.getElementById('data-status').textContent = '加载中';
    document.getElementById('preview-result').classList.add('hidden');

    const data = await api('/api/data');
    currentData = data;
    tickers = data.tickers;

    renderCurrent(data);
    renderParams(data);
    renderLatestRows(data);
    buildForm(data);

  } catch (e) {
    document.getElementById('current-status').textContent = '❌ ' + e.message;
    document.getElementById('params-status').textContent = '❌ ' + e.message;
    document.getElementById('data-status').textContent = '❌ ' + e.message;
  }
}

function renderCurrent(data) {
  const c = data.current;
  const div = document.getElementById('current-status');
  if (!c || Object.keys(c).length === 0) {
    div.innerHTML = '<p style="color: #8b949e;">暂无数据</p>';
    return;
  }
  const def_rate = c['防御率'] || 0;
  let def_tag = '<span class="tag tag-green">低</span>';
  if (def_rate > 60) def_tag = '<span class="tag tag-yellow">中</span>';
  if (def_rate > 80) def_tag = '<span class="tag tag-red">高</span>';

  let html = `<p style="margin-bottom:8px;">基准: <strong>${c['基准日期']}</strong> | 防御率: <strong>${def_rate}%</strong> ${def_tag}</p>`;
  html += '<table><tr><th>ETF</th><th class="num">权重</th></tr>';
  const def_etfs = data.defensive;
  for (const e of data.tickers) {
    const w = c[e];
    if (w === undefined) continue;
    const cls = def_etfs.includes(e) ? 'def' : '';
    html += `<tr class="${cls}"><td>${e}</td><td class="num pct ${w > 0 ? 'pos' : ''}">${w.toFixed(1)}%</td></tr>`;
  }
  html += '</table>';
  html += `<p style="margin-top:8px;color:#8b949e;font-size:0.8em;">数据共 ${data.weeks} 周 · 截至 ${data.last_date}</p>`;
  div.innerHTML = html;
}

function renderParams(data) {
  const p = data.params;
  const div = document.getElementById('params-status');
  let html = '<div class="param-grid">';
  const labels = {
    mom_w: '动量权重', vol_w: '波动权重', top_n: '选股数',
    invvol: 'inv-vol窗口', step_low: '防御下限', step_high: '防御上限',
    def_alloc: '基准防御', max_def: '极限防御', max_single: '上限',
    thresh: '调仓阈值', fee: '费率',
  };
  for (const [k, v] of Object.entries(p)) {
    const label = labels[k] || k;
    const val = k === 'fee' ? `${(v*10000).toFixed(1)}‱` : v;
    html += `<div class="param-item"><span>${label}</span><strong>${val}</strong></div>`;
  }
  html += '</div>';
  div.innerHTML = html;
}

function renderLatestRows(data) {
  const div = document.getElementById('data-status');
  let html = '<table><thead><tr><th>日期</th>';
  for (const e of data.tickers) html += `<th class="num">${e}</th>`;
  html += '</tr></thead><tbody>';
  for (const row of data.latest_rows.slice().reverse()) {
    html += '<tr>';
    html += `<td>${row['日期']}</td>`;
    for (const e of data.tickers) {
      const v = row[e];
      html += `<td class="num">${v.toFixed(4)}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table>';
  div.innerHTML = html;
}

function buildForm(data) {
  const container = document.getElementById('ticker-inputs');
  const today = new Date().toISOString().split('T')[0];
  document.getElementById('new-date').value = today;
  let html = '';
  for (const e of data.tickers) {
    // Get last value for default
    const lastRow = data.latest_rows[data.latest_rows.length - 1];
    const defaultVal = lastRow ? lastRow[e] : '';
    html += `<div class="form-group"><label>${e}</label><input type="number" step="0.0001" id="inp-${e}" value="${defaultVal}" placeholder="价格"></div>`;
  }
  container.innerHTML = html;
}

async function previewUpdate() {
  const data = collectForm();
  if (!data) return;
  const statusDiv = document.getElementById('input-status');
  statusDiv.className = 'status';
  statusDiv.textContent = '计算中...';

  try {
    const result = await api('/api/preview', 'POST', data);
    const div = document.getElementById('preview-result');
    div.classList.remove('hidden');
    renderPreview(result);
    statusDiv.className = 'status status-ok';
    statusDiv.textContent = '✅ 预览成功';
  } catch (e) {
    statusDiv.className = 'status status-err';
    statusDiv.textContent = '❌ ' + e.message;
  }
}

function renderPreview(result) {
  const div = document.getElementById('preview-content');
  const c = result.current;
  const ch = result.changes;
  const doReb = result.do_rebalance;

  let html = `<p>基准: <strong>${result.base_date}</strong> | 调仓: <strong>${doReb ? '⚡ 需要调仓' : '— 不调仓'}</strong>`;
  if (result.max_change !== undefined) {
    html += ` | 最大变化: <strong>${result.max_change}%</strong>`;
  }
  html += '</p>';

  html += '<table><tr><th>ETF</th><th class="num">权重</th><th class="num">变化</th></tr>';
  for (const e of tickers) {
    const w = c[e];
    if (w === undefined) continue;
    const change = ch ? ch[e] : null;
    let changeStr = '';
    if (change !== null && change !== undefined) {
      const cls = change > 0.5 ? 'pos' : (change < -0.5 ? 'neg' : '');
      changeStr = `<td class="num ${cls}">${change > 0 ? '+' : ''}${change.toFixed(1)}pp</td>`;
    } else {
      changeStr = '<td class="num">—</td>';
    }
    html += `<tr><td>${e}</td><td class="num pct ${w > 0 ? 'pos' : ''}">${w.toFixed(1)}%</td>${changeStr}</tr>`;
  }
  html += '</table>';

  // Scores
  if (result.scores && result.scores.length > 0) {
    html += '<h3 style="margin-top:12px;font-size:0.9em;color:#8b949e;">Layer 1 Scoring</h3>';
    html += '<table><tr><th>ETF</th><th class="num">mom4</th><th class="num">vol20</th><th class="num">score</th></tr>';
    for (const s of result.scores) {
      const cls = s.selected ? 'selected-row' : '';
      html += `<tr class="${cls}"><td>${s.etf}</td><td class="num">${s.mom4}</td><td class="num">${s.vol20}</td><td class="num">${s.score.toFixed(4)}</td></tr>`;
    }
    html += '</table>';
  }

  div.innerHTML = html;
}

async function saveUpdate() {
  const data = collectForm();
  if (!data) return;
  const btn = document.getElementById('save-btn');
  btn.disabled = true;
  btn.textContent = '保存中...';
  const statusDiv = document.getElementById('input-status');

  try {
    const result = await api('/api/update', 'POST', data);
    statusDiv.className = 'status status-ok';
    statusDiv.textContent = '✅ ' + result.message;
    if (result.current) {
      let allocStr = Object.entries(result.current)
        .filter(([k]) => k !== '防御率')
        .map(([k,v]) => `${k}:${v}%`).join(' | ');
      statusDiv.textContent += ` | 调仓: ${allocStr}`;
    }
    // Refresh all data
    await refreshAll();
  } catch (e) {
    statusDiv.className = 'status status-err';
    statusDiv.textContent = '❌ ' + e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = '💾 保存并计算';
  }
}

function collectForm() {
  const date = document.getElementById('new-date').value;
  if (!date) { alert('请填写日期'); return null; }
  const data = { '日期': date };
  for (const e of tickers) {
    const inp = document.getElementById(`inp-${e}`);
    if (!inp || !inp.value) { alert(`请填写 ${e} 的价格`); return null; }
    data[e] = parseFloat(inp.value);
  }
  return data;
}

// Initial load
refreshAll();
</script>
</body>
</html>"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='虾池ETF轮动 Web 助手')
    parser.add_argument('--port', type=int, default=8766, help='端口号')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='监听地址')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    args = parser.parse_args()

    print(f'启动 Web 助手: http://{args.host}:{args.port}')
    print(f'默认用户名: admin (或设置 HERMES_WEB_USER 环境变量)')
    print(f'默认密码:   claw2026 (或设置 HERMES_WEB_PASS 环境变量)')
    app.run(host=args.host, port=args.port, debug=args.debug)