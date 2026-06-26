#!/usr/bin/env python3
"""
T46: Phase 5c — Regime-Conditional Softmax + D4-Filtered Softmax Sweep

Run full-period + walk-forward backtests for 8 configs:
  4 A1 (regime-conditional softmax, T sweep)
  4 A2b (D4 threshold sweep + softmax T=1.5)
Compare against baseline (D4+3State) and Phase 5a best (A5 T=2.0).
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import run_backtest
from src.strategy import load_config

OUTPUT_DIR = PROJECT_ROOT / 'output' / 't46_phase5c'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configs ──
A1_CONFIGS = [
    ('A1 T=0.5', 'config/phase5c/p5c_A1_regime_softmax_T5.yaml', 0.5, 'RISK_ON only'),
    ('A1 T=1.0', 'config/phase5c/p5c_A1_regime_softmax_T10.yaml', 1.0, 'RISK_ON only'),
    ('A1 T=1.5', 'config/phase5c/p5c_A1_regime_softmax_T15.yaml', 1.5, 'RISK_ON only'),
    ('A1 T=2.0', 'config/phase5c/p5c_A1_regime_softmax_T20.yaml', 2.0, 'RISK_ON only'),
]

A2B_CONFIGS = [
    ('A2b D4=-0.050', 'config/phase5c/p5c_A2b_d4_thresh_m0p05.yaml', -0.05),
    ('A2b D4=-0.025', 'config/phase5c/p5c_A2b_d4_thresh_m0p025.yaml', -0.025),
    ('A2b D4=0.000', 'config/phase5c/p5c_A2b_d4_thresh_0p0.yaml', 0.0),
    ('A2b D4=0.025', 'config/phase5c/p5c_A2b_d4_thresh_0p025.yaml', 0.025),
]

ALL_CONFIGS = A1_CONFIGS + A2B_CONFIGS
BASELINE = 'config/strategy_v2_3_cap040_D4_tuned_regime_3state.yaml'
P5A_BEST = 'config/phase5/p5a_D5_full_t20.yaml'

def fmt_pct(v): return f"{v*100:.2f}%"
def fmt(v, d=3): return f"{v:.{d}f}"

# ══════════════════════════════════════════════════════════
# PHASE 1: Full-Period Backtests
# ══════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 1: Full-Period Backtests (Phase 5c)")
print("=" * 70)
print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

results = {}

# Baseline
print(f"\n  Running: baseline (D4+3State, no softmax)")
cfg_b = load_config(PROJECT_ROOT / BASELINE)
r_b = run_backtest(cfg_b)
results['baseline'] = r_b
mb = r_b.metrics
print(f"    AnnRet={fmt_pct(mb['annual_return'])}  MaxDD={fmt_pct(mb['max_drawdown'])}"
      f"  Sharpe={fmt(mb['sharpe_ratio'])}")

# Phase 5a best (A5 T=2.0)
print(f"\n  Running: P5a best (A5 Full-Offensive Softmax T=2.0)")
cfg_p5a = load_config(PROJECT_ROOT / P5A_BEST)
r_p5a = run_backtest(cfg_p5a)
results['P5a A5 T=2.0'] = r_p5a
mpa = r_p5a.metrics
print(f"    AnnRet={fmt_pct(mpa['annual_return'])}  MaxDD={fmt_pct(mpa['max_drawdown'])}"
      f"  Sharpe={fmt(mpa['sharpe_ratio'])}")

# All 8 Phase 5c configs
for label, cpath, *extra in ALL_CONFIGS:
    print(f"\n  Running: {label}")
    cfg = load_config(PROJECT_ROOT / cpath)
    r = run_backtest(cfg)
    results[label] = r
    m = r.metrics
    print(f"    AnnRet={fmt_pct(m['annual_return'])}  MaxDD={fmt_pct(m['max_drawdown'])}"
          f"  Sharpe={fmt(m['sharpe_ratio'])}  Calmar={fmt(m['calmar_ratio'],2)}")

# ══════════════════════════════════════════════════════════
# PHASE 2: Walk-Forward Backtests (9 windows)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 2: Walk-Forward Backtests (9 windows, 3yr train / 1yr test)")
print("=" * 70)

from src.data_loader import load_nav_data, resample_weekly

def get_wf_windows(config_path):
    cfg = load_config(PROJECT_ROOT / config_path)
    nav_daily = load_nav_data(PROJECT_ROOT / cfg.nav_path)
    nav_weekly = resample_weekly(nav_daily, cfg.anchor)
    start_dt = pd.Timestamp(cfg.start_date) if cfg.start_date else nav_weekly.index[0]
    end_dt = pd.Timestamp(cfg.end_date) if cfg.end_date else nav_weekly.index[-1]
    dates = nav_weekly.index[(nav_weekly.index >= start_dt) & (nav_weekly.index <= end_dt)]
    windows = []
    for i in range(156 + 52, len(dates) + 1, 52):
        windows.append((dates[i - 52], dates[i - 1]))
    return windows

wf_windows = get_wf_windows(A1_CONFIGS[0][1])
print(f"  WF windows: {len(wf_windows)}")
for wi, (ts, te) in enumerate(wf_windows):
    print(f"    W{wi}: {str(ts.date())[:7]} → {str(te.date())[:7]}")

wf_results = {}

# WF for all 10 configs (baseline + p5a + 8 phase5c)
wf_configs = [('baseline', BASELINE), ('P5a A5 T=2.0', P5A_BEST)] + \
             [(l, c) for l, c, *e in ALL_CONFIGS]

for label, cpath in wf_configs:
    print(f"\n  Walk-Forward: {label}")
    cfg = load_config(PROJECT_ROOT / cpath)
    wf = []
    for w_idx, (ts, te) in enumerate(wf_windows):
        r = run_backtest(cfg, start_date=str(ts.date()), end_date=str(te.date()))
        m = r.metrics
        wf.append({
            'window': w_idx,
            'period': f"{str(ts.date())[:7]}→{str(te.date())[:7]}",
            'sharpe': float(m['sharpe_ratio']),
            'ann_ret': float(m['annual_return']),
            'max_dd': float(m['max_drawdown']),
            'ann_vol': float(m['annual_volatility']),
        })
        print(f"    W{w_idx}: {wf[-1]['period']}  Sharpe={fmt(wf[-1]['sharpe'])}"
              f"  Ret={fmt_pct(wf[-1]['ann_ret'])}  DD={fmt_pct(wf[-1]['max_dd'])}")
    wf_results[label] = wf

def wf_summary(wf):
    a = np.array([w['sharpe'] for w in wf])
    b = np.array([w['ann_ret'] for w in wf])
    c = np.array([w['max_dd'] for w in wf])
    return {
        'n': len(wf),
        'sharpe_mean': float(np.mean(a)), 'sharpe_std': float(np.std(a)),
        'sharpe_min': float(np.min(a)), 'sharpe_max': float(np.max(a)),
        'ret_mean': float(np.mean(b)), 'dd_max': float(np.max(c)),
    }

# ══════════════════════════════════════════════════════════
# PHASE 3: Summary Tables
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 3: Full-Period Summary")
print("=" * 70)
labels_fp = ['baseline', 'P5a A5 T=2.0'] + [c[0] for c in ALL_CONFIGS]
print(f"{'Config':<25} {'Ann Ret':>8} {'Max DD':>8} {'Sharpe':>8} {'Calmar':>7} {'Def Weeks':>10}")
print("-" * 70)
for rk in labels_fp:
    m = results[rk].metrics
    print(f"{rk:<25} {fmt_pct(m['annual_return']):>8} {fmt_pct(m['max_drawdown']):>8}"
          f" {fmt(m['sharpe_ratio']):>8} {fmt(m['calmar_ratio'],2):>7} {m['defensive_weeks']:>10}")

print("\n" + "=" * 70)
print("PHASE 3b: Walk-Forward Summary")
print("=" * 70)
wf_sums = {}
print(f"\n{'Config':<25} {'WF Sharpe Mean':>14} {'WF Sharpe Std':>13} {'WF Sharpe Min':>14} {'G3 (Min>=0.8)':>13} {'G4 (Std<0.6)':>13} {'G5 (MaxDD<=8.5%)':>16}")
print("-" * 110)
for label in labels_fp:
    s = wf_summary(wf_results[label])
    wf_sums[label] = s
    g3 = '✅' if s['sharpe_min'] >= 0.8 else '❌'
    g4 = '✅' if s['sharpe_std'] < 0.60 else '❌'
    g5 = '✅' if s['dd_max'] <= 0.085 else '❌'
    print(f"{label:<25} {s['sharpe_mean']:>14.4f} {s['sharpe_std']:>13.4f} {s['sharpe_min']:>14.4f} {g3:>13} {g4:>13} {g5:>16}")

# ══════════════════════════════════════════════════════════
# PHASE 4: Gate Analysis
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 4: Gate Analysis (G4 < 0.60, G3 >= 0.80, G5 <= 8.5%)")
print("=" * 70)

phase5c_labels = [c[0] for c in ALL_CONFIGS]
for label in phase5c_labels:
    s = wf_sums[label]
    m = results[label].metrics
    g3 = 'PASS' if s['sharpe_min'] >= 0.8 else f"FAIL ({s['sharpe_min']:.3f})"
    g4 = 'PASS' if s['sharpe_std'] < 0.60 else f"FAIL ({s['sharpe_std']:.3f})"
    g5 = 'PASS' if s['dd_max'] <= 0.085 else f"FAIL ({s['dd_max']*100:.2f}%)"
    ann_ret_ok = 'PASS' if m['annual_return'] >= 0.14 else f"FAIL ({m['annual_return']*100:.2f}%)"
    sharpe_ok = 'PASS' if m['sharpe_ratio'] >= 1.10 else f"FAIL ({m['sharpe_ratio']:.3f})"
    print(f"  {label}:\n"
          f"    G4={g4}  G3={g3}  G5={g5}  AnnRet={ann_ret_ok}  Sharpe={sharpe_ok}\n"
          f"    FP: Ret={m['annual_return']*100:.2f}%  DD={m['max_drawdown']*100:.2f}%  Sharpe={m['sharpe_ratio']:.3f}")

# ══════════════════════════════════════════════════════════
# PHASE 5: Composite Ranking
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 5: Composite Ranking (0.4×G4 + 0.25×G3 + 0.15×G5 + 0.10×Ret + 0.10×Sharpe)")
print("=" * 70)

def composite_score(label):
    s = wf_sums[label]
    m = results[label].metrics

    # G4: WF Sharpe Std (lower is better) — normalize: 1 - std/max_std
    max_std = max(wf_sums[l]['sharpe_std'] for l in phase5c_labels)
    g4_score = 1 - s['sharpe_std'] / max_std if max_std > 0 else 0

    # G3: WF Sharpe Min (higher is better) — normalize as (val - min) / (max - min)
    all_mins = [wf_sums[l]['sharpe_min'] for l in phase5c_labels]
    min_min, max_min = min(all_mins), max(all_mins)
    g3_score = (s['sharpe_min'] - min_min) / (max_min - min_min) if max_min > min_min else 0.5

    # G5: MaxDD (lower is better) — normalize: 1 - dd/max_dd
    max_dd = max(wf_sums[l]['dd_max'] for l in phase5c_labels)
    g5_score = 1 - s['dd_max'] / max_dd if max_dd > 0 else 0

    # Ret: Annual Return (higher is better)
    all_rets = [results[l].metrics['annual_return'] for l in phase5c_labels]
    min_ret, max_ret = min(all_rets), max(all_rets)
    ret_score = (m['annual_return'] - min_ret) / (max_ret - min_ret) if max_ret > min_ret else 0.5

    # Sharpe (higher is better)
    all_sharpes = [results[l].metrics['sharpe_ratio'] for l in phase5c_labels]
    min_sh, max_sh = min(all_sharpes), max(all_sharpes)
    sh_score = (m['sharpe_ratio'] - min_sh) / (max_sh - min_sh) if max_sh > min_sh else 0.5

    composite = 0.40 * g4_score + 0.25 * g3_score + 0.15 * g5_score + 0.10 * ret_score + 0.10 * sh_score
    return composite, {'g4': g4_score, 'g3': g3_score, 'g5': g5_score, 'ret': ret_score, 'sharpe': sh_score}

print(f"\n{'Config':<25} {'G4':>6} {'G3':>6} {'G5':>6} {'Ret':>6} {'Sh':>6} {'Composite':>10}")
print("-" * 70)
comp_scores = {}
for label in phase5c_labels:
    cs, parts = composite_score(label)
    comp_scores[label] = (cs, parts)
    print(f"{label:<25} {parts['g4']:>6.4f} {parts['g3']:>6.4f} {parts['g5']:>6.4f} {parts['ret']:>6.4f} {parts['sharpe']:>6.4f} {cs:>10.4f}")

best = max(comp_scores, key=lambda l: comp_scores[l][0])
print(f"\n  🏆 BEST: {best} (composite={comp_scores[best][0]:.4f})")

# Also include baseline and p5a-best rankings
for ref in ['baseline', 'P5a A5 T=2.0']:
    cs, parts = composite_score(ref) if ref not in comp_scores else (comp_scores[ref][0], comp_scores[ref][1])
    print(f"  Ref  {ref:<21} composite={cs:.4f}")

# ══════════════════════════════════════════════════════════
# PHASE 6: Charts
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 6: Charts")
print("=" * 70)

palette = ['#E53935','#FB8C00','#43A047','#1E88E5','#8E24AA','#00ACC1','#D81B60','#7CB342',
           '#757575','#5C6BC0']
colors_map = {l: palette[i] for i, l in enumerate(phase5c_labels)}
colors_map['baseline'] = '#757575'
colors_map['P5a A5 T=2.0'] = '#333333'
paths = []

# 1) Full-period NAV (A1 configs only)
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(results['baseline'].nav_series.index, results['baseline'].nav_series['nav'],
        color='#333', lw=2, label='Baseline (D4+3S)', alpha=0.8)
for label, _, T, _ in A1_CONFIGS:
    ax.plot(results[label].nav_series.index, results[label].nav_series['nav'],
            color=colors_map[label], lw=1, alpha=0.8, label=f'A1 T={T}')
ax.axhline(y=1, color='gray', ls='--', lw=0.5, alpha=0.5)
ax.set_title('NAV: A1 Regime-Conditional Softmax (T sweep)', fontsize=14)
ax.set_ylabel('NAV'); ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.3)
p = OUTPUT_DIR / 'chart_nav_A1_sweep.png'; fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
paths.append(str(p)); print(f"  {p.name}")

# 2) Full-period NAV (A2b configs only)
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(results['baseline'].nav_series.index, results['baseline'].nav_series['nav'],
        color='#333', lw=2, label='Baseline', alpha=0.8)
for label, _, thresh in A2B_CONFIGS:
    ax.plot(results[label].nav_series.index, results[label].nav_series['nav'],
            color=colors_map[label], lw=1, alpha=0.8, label=f'A2b D4={thresh}')
ax.axhline(y=1, color='gray', ls='--', lw=0.5, alpha=0.5)
ax.set_title('NAV: A2b D4-Tight Softmax (threshold sweep)', fontsize=14)
ax.set_ylabel('NAV'); ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.3)
p = OUTPUT_DIR / 'chart_nav_A2b_sweep.png'; fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
paths.append(str(p)); print(f"  {p.name}")

# 3) WF Sharpe per window (all 8 configs)
fig, ax = plt.subplots(figsize=(14, 5))
for label in phase5c_labels:
    wf = wf_results[label]
    ax.plot(range(len(wf)), [w['sharpe'] for w in wf], 'o-',
            color=colors_map[label], lw=1.2, markersize=5, label=label)
ax.axhline(y=0, color='gray', ls=':', lw=0.5)
ax.axhline(y=0.8, color='green', ls='--', lw=1, alpha=0.5, label='G3 (0.8)')
ref_wf = wf_results[phase5c_labels[0]]
ax.set_xticks(range(len(ref_wf)))
ax.set_xticklabels([w['period'] for w in ref_wf], rotation=45)
ax.set_title('Walk-Forward Sharpe per Window', fontsize=14)
ax.set_ylabel('Sharpe'); ax.legend(loc='lower left', fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
p = OUTPUT_DIR / 'chart_wf_sharpe_all.png'; fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
paths.append(str(p)); print(f"  {p.name}")

# 4) G4 (WF Sharpe Std) comparison
fig, ax = plt.subplots(figsize=(12, 5))
all_labels = ['baseline', 'P5a A5 T=2.0'] + phase5c_labels
wf_stds = [wf_sums[l]['sharpe_std'] for l in all_labels]
bar_colors = [colors_map.get(l, '#999') for l in all_labels]
bars = ax.bar(range(len(all_labels)), wf_stds, color=bar_colors, alpha=0.85)
ax.axhline(y=0.60, color='red', ls='--', lw=1.5, label='G4 target (0.60)')
for bar, val in zip(bars, wf_stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', fontsize=8, rotation=90)
ax.set_xticks(range(len(all_labels)))
ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
ax.set_title('WF Sharpe Std (G4: < 0.60)', fontsize=14)
ax.set_ylabel('Sharpe Std'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
p = OUTPUT_DIR / 'chart_g4_all.png'; fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
paths.append(str(p)); print(f"  {p.name}")

# 5) Composite scores bar chart
fig, ax = plt.subplots(figsize=(12, 5))
cs_labels = phase5c_labels
cs_values = [comp_scores[l][0] for l in cs_labels]
cs_colors = [colors_map[l] for l in cs_labels]
bars = ax.bar(range(len(cs_labels)), cs_values, color=cs_colors, alpha=0.85)
for bar, val in zip(bars, cs_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontsize=9)
ax.set_xticks(range(len(cs_labels)))
ax.set_xticklabels(cs_labels, rotation=25, ha='right', fontsize=8)
ax.set_title('Composite Score (0.4×G4 + 0.25×G3 + 0.15×G5 + 0.10×Ret + 0.10×Sharpe)', fontsize=13)
ax.set_ylabel('Score'); ax.grid(True, alpha=0.3, axis='y')
p = OUTPUT_DIR / 'chart_composite.png'; fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
paths.append(str(p)); print(f"  {p.name}")

# ══════════════════════════════════════════════════════════
# PHASE 7: JSON Output
# ══════════════════════════════════════════════════════════
output = {
    'meta': {
        'task': 'T46 Phase 5c',
        'approaches': 'A1 (Regime-Conditional Softmax) + A2b (D4-Filtered Softmax)',
        'configs': [
            {'label': l, 'config': c, 'params': extra} for l, c, *extra in ALL_CONFIGS
        ],
        'baseline': BASELINE,
        'p5a_best': P5A_BEST,
        'baseline_metrics': {
            'annual_return': float(mb['annual_return']),
            'max_drawdown': float(mb['max_drawdown']),
            'sharpe_ratio': float(mb['sharpe_ratio']),
        },
    },
    'full_period': {
        l: {
            'annual_return': float(results[l].metrics['annual_return']),
            'max_drawdown': float(results[l].metrics['max_drawdown']),
            'sharpe_ratio': float(results[l].metrics['sharpe_ratio']),
            'calmar_ratio': float(results[l].metrics['calmar_ratio']),
            'annual_volatility': float(results[l].metrics['annual_volatility']),
            'win_rate': float(results[l].metrics['win_rate']),
            'defensive_weeks': int(results[l].metrics['defensive_weeks']),
            'total_weeks': int(results[l].metrics['total_weeks']),
        }
        for l in all_labels
    },
    'walk_forward': {l: wf_results[l] for l in all_labels},
    'walk_forward_summary': wf_sums,
    'gates': {
        l: {
            'G5_maxdd_85': 'PASS' if wf_sums[l]['dd_max'] <= 0.085 else 'FAIL',
            'G4_wf_std_60': 'PASS' if wf_sums[l]['sharpe_std'] < 0.60 else 'FAIL',
            'G3_min_sharpe_08': 'PASS' if wf_sums[l]['sharpe_min'] >= 0.8 else 'FAIL',
            'ann_ret_14': 'PASS' if results[l].metrics['annual_return'] >= 0.14 else 'FAIL',
            'sharpe_11': 'PASS' if results[l].metrics['sharpe_ratio'] >= 1.10 else 'FAIL',
        }
        for l in all_labels
    },
    'composite_scores': {l: {'total': comp_scores[l][0], 'components': comp_scores[l][1]}
                         for l in phase5c_labels},
    'recommendation': {
        'best': best,
        'best_composite': comp_scores[best][0],
        'best_g4_std': wf_sums[best]['sharpe_std'],
        'best_g3_min': wf_sums[best]['sharpe_min'],
        'best_ann_ret': results[best].metrics['annual_return'],
        'best_sharpe': results[best].metrics['sharpe_ratio'],
    },
    'charts': [Path(p).name for p in paths],
}

json_path = OUTPUT_DIR / 't46_phase5c_results.json'
with open(json_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\n  JSON: {json_path}")

# ══════════════════════════════════════════════════════════
# PHASE 8: Markdown Report
# ══════════════════════════════════════════════════════════
md_path = OUTPUT_DIR / 'T46_PHASE5C_REPORT.md'
with open(md_path, 'w') as f:
    f.write("# T46: Phase 5c — Regime-Conditional Softmax + D4-Filtered Softmax\n\n")
    f.write(f"**Evaluator**: quant-coder | **Date**: {time.strftime('%Y-%m-%d')}\n\n")

    f.write("## Overview\n\n")
    f.write("Evaluating 8 configs across 2 approaches:\n\n")
    f.write("**A1 — Regime-Conditional Softmax** (4 configs): Softmax only in RISK_ON; "
            "hard top_n in CAUTIOUS/DEFENSIVE. Temperature sweep: T ∈ {0.5, 1.0, 1.5, 2.0}.\n\n")
    f.write("**A2b — D4-Filtered Softmax** (4 configs): D4 momentum filter removes weak ETFs "
            "before softmax distributes. D4 threshold sweep: {-0.05, -0.025, 0.0, 0.025}.\n\n")
    f.write(f"**Baseline**: `{BASELINE}` — D4 tuned + 3-State Regime, NO softmax.\n")
    f.write(f"**P5a Best**: `{P5A_BEST}` — Phase 5a A5 Full-Offensive Softmax T=2.0.\n\n")

    f.write("## 1. Full-Period Backtest\n\n")
    f.write("| Config | Ann Ret | Max DD | Sharpe | Calmar | Ann Vol | Win Rate | Def Weeks |\n")
    f.write("|--------|:------:|:------:|:------:|:------:|:-------:|:--------:|:---------:|\n")
    for l in all_labels:
        m = results[l].metrics
        f.write(f"| {l} | {fmt_pct(m['annual_return'])} | {fmt_pct(m['max_drawdown'])} | "
                f"{fmt(m['sharpe_ratio'])} | {fmt(m['calmar_ratio'],2)} | "
                f"{fmt_pct(m['annual_volatility'])} | {fmt_pct(m['win_rate'])} | {m['defensive_weeks']} |\n")

    f.write("\n## 2. Walk-Forward Summary\n\n")
    f.write("| Config | WF Sharpe Mean | WF Sharpe Std | WF Sharpe Min | WF Sharpe Max | G3 | G4 | G5 |\n")
    f.write("|--------|:--------------:|:-------------:|:-------------:|:-------------:|:--:|:--:|:--:|\n")
    for l in all_labels:
        s = wf_sums[l]
        g3 = '✅' if s['sharpe_min'] >= 0.8 else '❌'
        g4 = '✅' if s['sharpe_std'] < 0.60 else '❌'
        g5 = '✅' if s['dd_max'] <= 0.085 else '❌'
        f.write(f"| {l} | {s['sharpe_mean']:.4f} | {s['sharpe_std']:.4f} | {s['sharpe_min']:.4f} | "
                f"{s['sharpe_max']:.4f} | {g3} | {g4} | {g5} |\n")

    f.write("\n### Window Detail\n\n")
    n_win = len(wf_results[all_labels[0]])
    for w_idx in range(n_win):
        f.write(f"**W{w_idx}** ({wf_results[all_labels[0]][w_idx]['period']}): ")
        sharpe_vals = [f"{wf_results[l][w_idx]['sharpe']:.3f}" for l in phase5c_labels]
        f.write(", ".join(f"{l}={s}" for l, s in zip(phase5c_labels, sharpe_vals)))
        f.write("\n")

    f.write("\n## 3. Gate Analysis\n\n")
    f.write("### Full Gate Results\n\n")
    f.write("| Config | G4 (Std<0.60) | G3 (Min>=0.80) | G5 (DD<=8.5%) | AnnRet>=14% | Sharpe>=1.10 |\n")
    f.write("|--------|:-------------:|:--------------:|:-------------:|:----------:|:-----------:|\n")
    for l in all_labels:
        s = wf_sums[l]
        m = results[l].metrics
        g4 = '✅' if s['sharpe_std'] < 0.60 else '❌'
        g3 = '✅' if s['sharpe_min'] >= 0.8 else '❌'
        g5 = '✅' if s['dd_max'] <= 0.085 else '❌'
        ar = '✅' if m['annual_return'] >= 0.14 else '❌'
        sh = '✅' if m['sharpe_ratio'] >= 1.10 else '❌'
        f.write(f"| {l} | {g4} | {g3} | {g5} | {ar} | {sh} |\n")

    f.write("\n## 4. Composite Ranking\n\n")
    f.write("Formula: 0.4 × G4 + 0.25 × G3 + 0.15 × G5 + 0.10 × Ret + 0.10 × Sharpe\n\n")
    f.write("| Config | G4 | G3 | G5 | Ret | Sharpe | Composite |\n")
    f.write("|--------|:---:|:---:|:---:|:---:|:------:|:---------:|\n")
    for l in sorted(phase5c_labels, key=lambda l: comp_scores[l][0], reverse=True):
        p = comp_scores[l][1]
        f.write(f"| {l} | {p['g4']:.4f} | {p['g3']:.4f} | {p['g5']:.4f} | "
                f"{p['ret']:.4f} | {p['sharpe']:.4f} | {comp_scores[l][0]:.4f} |\n")

    f.write(f"\n## 🏆 Recommendation\n\n")
    f.write(f"**Best config: {best}** (composite = {comp_scores[best][0]:.4f})\n\n")
    s = wf_sums[best]
    m = results[best].metrics
    f.write(f"- **G4 (WF Sharpe Std)**: {s['sharpe_std']:.4f} "
            f"{'✅ PASS' if s['sharpe_std'] < 0.60 else '❌ FAIL — needs further optimization'}\n")
    f.write(f"- **G3 (WF Sharpe Min)**: {s['sharpe_min']:.4f} "
            f"{'✅ PASS' if s['sharpe_min'] >= 0.8 else '❌ FAIL — needs further optimization'}\n")
    f.write(f"- **G5 (MaxDD)**: {s['dd_max']*100:.2f}% "
            f"{'✅ PASS' if s['dd_max'] <= 0.085 else '❌ FAIL'}\n")
    f.write(f"- **AnnRet**: {m['annual_return']*100:.2f}% "
            f"{'✅ PASS' if m['annual_return'] >= 0.14 else '❌ FAIL'}\n")
    f.write(f"- **Sharpe**: {m['sharpe_ratio']:.3f} "
            f"{'✅ PASS' if m['sharpe_ratio'] >= 1.10 else '❌ FAIL'}\n")

    # Show runner-up
    runner_up = sorted(phase5c_labels, key=lambda l: comp_scores[l][0], reverse=True)[1]
    f.write(f"\n**Runner-up: {runner_up}** (composite = {comp_scores[runner_up][0]:.4f})\n\n")

    # Comparison with baseline
    b_s = wf_sums['baseline']
    b_m = results['baseline'].metrics
    f.write("### Baseline Comparison\n\n")
    f.write(f"| Metric | Baseline | Best ({best}) | Delta |\n")
    f.write(f"|--------|:--------:|:------------:|:-----:|\n")
    f.write(f"| AnnRet | {fmt_pct(b_m['annual_return'])} | {fmt_pct(m['annual_return'])} | "
            f"{(m['annual_return'] - b_m['annual_return'])*100:+.2f}pp |\n")
    f.write(f"| MaxDD | {fmt_pct(b_m['max_drawdown'])} | {fmt_pct(m['max_drawdown'])} | "
            f"{(m['max_drawdown'] - b_m['max_drawdown'])*100:+.2f}pp |\n")
    f.write(f"| Sharpe | {fmt(b_m['sharpe_ratio'])} | {fmt(m['sharpe_ratio'])} | "
            f"{m['sharpe_ratio'] - b_m['sharpe_ratio']:+.3f} |\n")
    f.write(f"| WF Std (G4) | {b_s['sharpe_std']:.4f} | {s['sharpe_std']:.4f} | "
            f"{s['sharpe_std'] - b_s['sharpe_std']:+.4f} |\n")
    f.write(f"| WF Min (G3) | {b_s['sharpe_min']:.4f} | {s['sharpe_min']:.4f} | "
            f"{s['sharpe_min'] - b_s['sharpe_min']:+.4f} |\n")

    f.write("\n### Comparison with Phase 5a Best\n\n")
    p5a_s = wf_sums['P5a A5 T=2.0']
    p5a_m = results['P5a A5 T=2.0'].metrics
    f.write(f"| Metric | P5a Best (A5 T=2.0) | Best ({best}) | Delta |\n")
    f.write(f"|--------|:-------------------:|:------------:|:-----:|\n")
    f.write(f"| AnnRet | {fmt_pct(p5a_m['annual_return'])} | {fmt_pct(m['annual_return'])} | "
            f"{(m['annual_return'] - p5a_m['annual_return'])*100:+.2f}pp |\n")
    f.write(f"| MaxDD | {fmt_pct(p5a_m['max_drawdown'])} | {fmt_pct(m['max_drawdown'])} | "
            f"{(m['max_drawdown'] - p5a_m['max_drawdown'])*100:+.2f}pp |\n")
    f.write(f"| Sharpe | {fmt(p5a_m['sharpe_ratio'])} | {fmt(m['sharpe_ratio'])} | "
            f"{m['sharpe_ratio'] - p5a_m['sharpe_ratio']:+.3f} |\n")
    f.write(f"| WF Std (G4) | {p5a_s['sharpe_std']:.4f} | {s['sharpe_std']:.4f} | "
            f"{s['sharpe_std'] - p5a_s['sharpe_std']:+.4f} |\n")
    f.write(f"| WF Min (G3) | {p5a_s['sharpe_min']:.4f} | {s['sharpe_min']:.4f} | "
            f"{s['sharpe_min'] - p5a_s['sharpe_min']:+.4f} |\n")

    f.write("\n## 5. Next Steps\n\n")
    g4_pass = s['sharpe_std'] < 0.60
    g3_pass = s['sharpe_min'] >= 0.8
    if g4_pass and g3_pass:
        f.write("**Recommendation: PROMOTE to Phase 5d production tuning.** Both G4 and G3 pass. "
                f"Fine-tune the winning approach and prepare for production deployment.\n")
    elif g4_pass:
        f.write("**Recommendation: Phase 5d narrow tuning on G3.** G4 passes but G3 needs improvement. "
                "Focus on improving worst-window Sharpe in future iterations.\n")
    else:
        f.write("**Recommendation: Phase 5d explore additional hybrid approaches.** "
                "Neither G3 nor G4 fully passes. Consider Approach 3 (Temperature-Ramp) or "
                "combined A1 + A2b.\n")

print(f"\n  Report: {md_path}")
print(f"\n{'=' * 70}")
print(f"Evaluation complete! Output in: {OUTPUT_DIR}")
print(f"{'=' * 70}")