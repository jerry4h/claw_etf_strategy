#!/usr/bin/env python3
"""
T48: Phase 5e — Temperature-Ramp Softmax Sweep

Run full-period + walk-forward backtests for 9 configs:
  3 T_risk values (1.0, 2.0, 5.0) × 3 T_def values (3.0, 5.0, 10.0)
  T_caut fixed at 2.0

Compare against baseline (D4+3State, no softmax).
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

OUTPUT_DIR = PROJECT_ROOT / 'output' / 't48_phase5e'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configs ──
P5E_CONFIGS = [
    ('Tr1_Td3',  'config/phase5e/p5e_tempramp_Tr1_Td3.yaml',  1.0, 3.0),
    ('Tr1_Td5',  'config/phase5e/p5e_tempramp_Tr1_Td5.yaml',  1.0, 5.0),
    ('Tr1_Td10', 'config/phase5e/p5e_tempramp_Tr1_Td10.yaml', 1.0, 10.0),
    ('Tr2_Td3',  'config/phase5e/p5e_tempramp_Tr2_Td3.yaml',  2.0, 3.0),
    ('Tr2_Td5',  'config/phase5e/p5e_tempramp_Tr2_Td5.yaml',  2.0, 5.0),
    ('Tr2_Td10', 'config/phase5e/p5e_tempramp_Tr2_Td10.yaml', 2.0, 10.0),
    ('Tr5_Td3',  'config/phase5e/p5e_tempramp_Tr5_Td3.yaml',  5.0, 3.0),
    ('Tr5_Td5',  'config/phase5e/p5e_tempramp_Tr5_Td5.yaml',  5.0, 5.0),
    ('Tr5_Td10', 'config/phase5e/p5e_tempramp_Tr5_Td10.yaml', 5.0, 10.0),
]

BASELINE = 'config/strategy_v2_3_cap040_D4_tuned_regime_3state.yaml'

def fmt_pct(v): return f"{v*100:.2f}%"
def fmt(v, d=3): return f"{v:.{d}f}"

# ══════════════════════════════════════════════════════════
# PHASE 1: Full-Period Backtests
# ══════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 1: Full-Period Backtests (Phase 5e - Temperature-Ramp)")
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

# All 9 Phase 5e configs
for label, cpath, tr, td in P5E_CONFIGS:
    print(f"\n  Running: {label} (Tr={tr}, Tc=2.0, Td={td})")
    cfg = load_config(PROJECT_ROOT / cpath)
    r = run_backtest(cfg)
    results[label] = r
    m = r.metrics
    print(f"    AnnRet={fmt_pct(m['annual_return'])}  MaxDD={fmt_pct(m['max_drawdown'])}"
          f"  Sharpe={fmt(m['sharpe_ratio'])}  Calmar={fmt(m['calmar_ratio'],2)}"
          f"  DefW={m['defensive_weeks']}")

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

wf_windows = get_wf_windows(P5E_CONFIGS[0][1])
print(f"  WF windows: {len(wf_windows)}")
for wi, (ts, te) in enumerate(wf_windows):
    print(f"    W{wi}: {str(ts.date())[:7]} → {str(te.date())[:7]}")

wf_results = {}

# WF for all 10 configs (baseline + 9 phase5e)
wf_configs = [('baseline', BASELINE)] + [(l, c) for l, c, _, _ in P5E_CONFIGS]

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
all_labels = ['baseline'] + [c[0] for c in P5E_CONFIGS]
p5e_labels = [c[0] for c in P5E_CONFIGS]

print("\n" + "=" * 70)
print("PHASE 3: Full-Period Summary")
print("=" * 70)
print(f"{'Config':<15} {'Ann Ret':>8} {'Max DD':>8} {'Sharpe':>8} {'Calmar':>7} {'Ann Vol':>8} {'Win Rate':>9} {'Def Weeks':>10}")
print("-" * 80)
for rk in all_labels:
    m = results[rk].metrics
    print(f"{rk:<15} {fmt_pct(m['annual_return']):>8} {fmt_pct(m['max_drawdown']):>8}"
          f" {fmt(m['sharpe_ratio']):>8} {fmt(m['calmar_ratio'],2):>7}"
          f" {fmt_pct(m['annual_volatility']):>8} {fmt_pct(m['win_rate']):>9} {m['defensive_weeks']:>10}")

print("\n" + "=" * 70)
print("PHASE 3b: Walk-Forward Summary")
print("=" * 70)
wf_sums = {}
print(f"{'Config':<15} {'WF Sharpe Mean':>14} {'WF Sharpe Std':>13} {'WF Sharpe Min':>14} {'WF Sharpe Max':>14} {'G3 (Min>=0.8)':>13} {'G4 (Std<0.6)':>13} {'G5 (DD<=8.5%)':>16}")
print("-" * 115)
for label in all_labels:
    s = wf_summary(wf_results[label])
    wf_sums[label] = s
    g3 = '✅' if s['sharpe_min'] >= 0.8 else '❌'
    g4 = '✅' if s['sharpe_std'] < 0.60 else '❌'
    g5 = '✅' if s['dd_max'] <= 0.085 else '❌'
    print(f"{label:<15} {s['sharpe_mean']:>14.4f} {s['sharpe_std']:>13.4f} {s['sharpe_min']:>14.4f}"
          f" {s['sharpe_max']:>14.4f} {g3:>13} {g4:>13} {g5:>16}")

# ══════════════════════════════════════════════════════════
# PHASE 4: Gate Analysis
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 4: Gate Analysis (G4 < 0.60, G3 >= 0.80, G5 <= 8.5%)")
print("=" * 70)

for label in p5e_labels:
    s = wf_sums[label]
    m = results[label].metrics
    g3 = 'PASS' if s['sharpe_min'] >= 0.8 else f"FAIL ({s['sharpe_min']:.4f})"
    g4 = 'PASS' if s['sharpe_std'] < 0.60 else f"FAIL ({s['sharpe_std']:.4f})"
    g5 = 'PASS' if s['dd_max'] <= 0.085 else f"FAIL ({s['dd_max']*100:.2f}%)"
    ann_ret_ok = 'PASS' if m['annual_return'] >= 0.14 else f"FAIL ({m['annual_return']*100:.2f}%)"
    sharpe_ok = 'PASS' if m['sharpe_ratio'] >= 1.10 else f"FAIL ({m['sharpe_ratio']:.3f})"
    print(f"  {label}: G4={g4}  G3={g3}  G5={g5}  AnnRet={ann_ret_ok}  Sharpe={sharpe_ok}")

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
    max_std = max(wf_sums[l]['sharpe_std'] for l in p5e_labels)
    g4_score = max(0, 1 - s['sharpe_std'] / max_std) if max_std > 0 else 0

    # G3: WF Sharpe Min (higher is better) — normalize: (val - min) / (max - min)
    all_mins = [wf_sums[l]['sharpe_min'] for l in p5e_labels]
    min_min, max_mins = min(all_mins), max(all_mins)
    if max_mins > min_min:
        g3_score = (s['sharpe_min'] - min_min) / (max_mins - min_min)
    else:
        g3_score = 0.5

    # G5: MaxDD (lower is better) — normalize: 1 - dd/max_dd
    max_dd_val = max(wf_sums[l]['dd_max'] for l in p5e_labels)
    g5_score = max(0, 1 - s['dd_max'] / max_dd_val) if max_dd_val > 0 else 0

    # Ret: Annual Return (higher is better)
    all_rets = [results[l].metrics['annual_return'] for l in p5e_labels]
    min_ret, max_ret = min(all_rets), max(all_rets)
    ret_score = (m['annual_return'] - min_ret) / (max_ret - min_ret) if max_ret > min_ret else 0.5

    # Sharpe (higher is better)
    all_shs = [results[l].metrics['sharpe_ratio'] for l in p5e_labels]
    min_sh, max_sh = min(all_shs), max(all_shs)
    sh_score = (m['sharpe_ratio'] - min_sh) / (max_sh - min_sh) if max_sh > min_sh else 0.5

    composite = 0.40 * g4_score + 0.25 * g3_score + 0.15 * g5_score + 0.10 * ret_score + 0.10 * sh_score
    return composite, {'g4': g4_score, 'g3': g3_score, 'g5': g5_score, 'ret': ret_score, 'sharpe': sh_score}

print(f"\n{'Config':<15} {'G4':>6} {'G3':>6} {'G5':>6} {'Ret':>6} {'Sh':>6} {'Composite':>10}")
print("-" * 60)
comp_scores = {}
for label in p5e_labels:
    cs, parts = composite_score(label)
    comp_scores[label] = (cs, parts)
    print(f"{label:<15} {parts['g4']:>6.4f} {parts['g3']:>6.4f} {parts['g5']:>6.4f}"
          f" {parts['ret']:>6.4f} {parts['sharpe']:>6.4f} {cs:>10.4f}")

best = max(comp_scores, key=lambda l: comp_scores[l][0])
print(f"\n  🏆 BEST: {best} (composite={comp_scores[best][0]:.4f})")

# Show baseline composited
if 'baseline' in wf_sums:
    b_cs, b_parts = composite_score('baseline') if 'baseline' not in comp_scores else comp_scores['baseline']
    print(f"  Ref baseline  composite={b_cs:.4f}")

# ══════════════════════════════════════════════════════════
# PHASE 6: Charts
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 6: Charts")
print("=" * 70)

palette = ['#E53935','#FB8C00','#43A047','#1E88E5','#8E24AA',
           '#00ACC1','#D81B60','#7CB342','#FDD835']
colors_map = {p5e_labels[i]: palette[i] for i in range(len(p5e_labels))}
colors_map['baseline'] = '#757575'

paths = []

# 1) Full-period NAV (all 9 configs vs baseline)
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(results['baseline'].nav_series.index, results['baseline'].nav_series['nav'],
        color='#333', lw=2, label='Baseline', alpha=0.9)
for label in p5e_labels:
    ax.plot(results[label].nav_series.index, results[label].nav_series['nav'],
            color=colors_map[label], lw=1, alpha=0.6, label=label)
ax.axhline(y=1, color='gray', ls='--', lw=0.5, alpha=0.5)
ax.set_title('NAV: Phase 5e Temperature-Ramp Softmax vs Baseline', fontsize=14)
ax.set_ylabel('NAV'); ax.legend(loc='upper left', fontsize=6, ncol=2)
ax.grid(True, alpha=0.3)
p = OUTPUT_DIR / 'chart_nav_all.png'
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
paths.append(str(p)); print(f"  {p.name}")

# 2) WF Sharpe per window
fig, ax = plt.subplots(figsize=(14, 5))
for label in p5e_labels:
    wf = wf_results[label]
    ax.plot(range(len(wf)), [w['sharpe'] for w in wf], 'o-',
            color=colors_map[label], lw=1.2, markersize=4, label=label)
# Baseline
ref_wf = wf_results['baseline']
ax.plot(range(len(ref_wf)), [w['sharpe'] for w in ref_wf], 's--',
        color='#333', lw=1.5, markersize=5, label='baseline')
ax.axhline(y=0, color='gray', ls=':', lw=0.5)
ax.axhline(y=0.8, color='green', ls='--', lw=1, alpha=0.5, label='G3 (0.8)')
ax.set_xticks(range(len(ref_wf)))
ax.set_xticklabels([w['period'] for w in ref_wf], rotation=45)
ax.set_title('Walk-Forward Sharpe per Window (Phase 5e)', fontsize=14)
ax.set_ylabel('Sharpe'); ax.legend(loc='lower left', fontsize=6, ncol=2)
ax.grid(True, alpha=0.3)
p = OUTPUT_DIR / 'chart_wf_sharpe.png'
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
paths.append(str(p)); print(f"  {p.name}")

# 3) G4 bar chart
fig, ax = plt.subplots(figsize=(12, 5))
all_for_bar = all_labels
wf_stds = [wf_sums[l]['sharpe_std'] for l in all_for_bar]
bar_colors = [colors_map.get(l, '#999') for l in all_for_bar]
bars = ax.bar(range(len(all_for_bar)), wf_stds, color=bar_colors, alpha=0.85)
ax.axhline(y=0.60, color='red', ls='--', lw=1.5, label='G4 target (0.60)')
for bar, val in zip(bars, wf_stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', fontsize=8, rotation=90)
ax.set_xticks(range(len(all_for_bar)))
ax.set_xticklabels(all_for_bar, rotation=45, ha='right', fontsize=8)
ax.set_title('WF Sharpe Std (G4: < 0.60)', fontsize=14)
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
p = OUTPUT_DIR / 'chart_g4.png'
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
paths.append(str(p)); print(f"  {p.name}")

# 4) Composite bar chart
fig, ax = plt.subplots(figsize=(12, 5))
cs_labels = p5e_labels
cs_values = [comp_scores[l][0] for l in cs_labels]
cs_colors = [colors_map[l] for l in cs_labels]
bars = ax.bar(range(len(cs_labels)), cs_values, color=cs_colors, alpha=0.85)
for bar, val in zip(bars, cs_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.4f}', ha='center', fontsize=8, rotation=90)
ax.set_xticks(range(len(cs_labels)))
ax.set_xticklabels(cs_labels, rotation=25, ha='right', fontsize=8)
ax.set_title('Composite Score (0.4×G4 + 0.25×G3 + 0.15×G5 + 0.10×Ret + 0.10×Sharpe)', fontsize=13)
ax.set_ylabel('Score'); ax.grid(True, alpha=0.3, axis='y')
p = OUTPUT_DIR / 'chart_composite.png'
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
paths.append(str(p)); print(f"  {p.name}")

# ══════════════════════════════════════════════════════════
# PHASE 7: JSON Output
# ══════════════════════════════════════════════════════════
output = {
    'meta': {
        'task': 'T48 Phase 5e',
        'approach': 'Temperature-Ramp Softmax: T_caut=2.0 fixed, T_risk ∈ {1,2,5}, T_def ∈ {3,5,10}',
        'configs': [
            {'label': l, 'config': c, 'T_risk': tr, 'T_def': td}
            for l, c, tr, td in P5E_CONFIGS
        ],
        'baseline': BASELINE,
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
                         for l in p5e_labels},
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

json_path = OUTPUT_DIR / 't48_phase5e_results.json'
with open(json_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\n  JSON: {json_path}")

# ══════════════════════════════════════════════════════════
# PHASE 8: Markdown Report
# ══════════════════════════════════════════════════════════
md_path = OUTPUT_DIR / 'T48_PHASE5E_REPORT.md'
with open(md_path, 'w') as f:
    f.write("# T48: Phase 5e — Temperature-Ramp Softmax Sweep\n\n")
    f.write(f"**Evaluator**: quant-coder | **Date**: {time.strftime('%Y-%m-%d')}\n\n")

    f.write("## Overview\n\n")
    f.write("Evaluating 9 configs for Temperature-Ramp Softmax Approach 3:\n\n")
    f.write("- **T_caut fixed at 2.0** (CAUTIOUS regime)\n")
    f.write("- **T_risk ∈ {1.0, 2.0, 5.0}** (RISK_ON regime)\n")
    f.write("- **T_def ∈ {3.0, 5.0, 10.0}** (DEFENSIVE regime)\n")
    f.write("- Softmax ALWAYS ON in all 3 regimes (no hard fallback)\n\n")
    f.write(f"**Baseline**: `{BASELINE}` — D4 tuned + 3-State Regime, NO softmax.\n")
    f.write(f"Baseline metrics: AnnRet=14.87%, MaxDD=7.84%, Sharpe=1.191, WF_Std=1.0961, WF_Min=-0.3868\n\n")

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
    n_win = len(wf_results[p5e_labels[0]])
    for w_idx in range(n_win):
        period = wf_results[p5e_labels[0]][w_idx]['period']
        sharpe_vals = [f"{l}={wf_results[l][w_idx]['sharpe']:.3f}" for l in p5e_labels]
        f.write(f"**W{w_idx}** ({period}): " + ", ".join(sharpe_vals) + "\n")

    f.write("\n## 3. Gate Analysis\n\n")
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
    for l in sorted(p5e_labels, key=lambda l: comp_scores[l][0], reverse=True):
        p = comp_scores[l][1]
        f.write(f"| {l} | {p['g4']:.4f} | {p['g3']:.4f} | {p['g5']:.4f} | "
                f"{p['ret']:.4f} | {p['sharpe']:.4f} | {comp_scores[l][0]:.4f} |\n")

    f.write(f"\n## 🏆 Recommendation\n\n")
    f.write(f"**Best config: {best}** (composite = {comp_scores[best][0]:.4f})\n\n")
    s = wf_sums[best]
    m = results[best].metrics
    f.write(f"- **G4 (WF Sharpe Std)**: {s['sharpe_std']:.4f} {'✅ PASS' if s['sharpe_std'] < 0.60 else '❌ FAIL'}\n")
    f.write(f"- **G3 (WF Sharpe Min)**: {s['sharpe_min']:.4f} {'✅ PASS' if s['sharpe_min'] >= 0.8 else '❌ FAIL'}\n")
    f.write(f"- **G5 (MaxDD)**: {s['dd_max']*100:.2f}% {'✅ PASS' if s['dd_max'] <= 0.085 else '❌ FAIL'}\n")
    f.write(f"- **AnnRet**: {m['annual_return']*100:.2f}% {'✅ PASS' if m['annual_return'] >= 0.14 else '❌ FAIL'}\n")
    f.write(f"- **Sharpe**: {m['sharpe_ratio']:.3f} {'✅ PASS' if m['sharpe_ratio'] >= 1.10 else '❌ FAIL'}\n")

    runner_up = sorted(p5e_labels, key=lambda l: comp_scores[l][0], reverse=True)[1]
    f.write(f"\n**Runner-up: {runner_up}** (composite = {comp_scores[runner_up][0]:.4f})\n\n")

    # Baseline comparison
    b_s = wf_sums['baseline']
    b_m = results['baseline'].metrics
    f.write("### Baseline Comparison\n\n")
    f.write("| Metric | Baseline | Best | Delta |\n")
    f.write("|--------|:--------:|:----:|:-----:|\n")
    f.write(f"| AnnRet | {fmt_pct(b_m['annual_return'])} | {fmt_pct(m['annual_return'])} | {(m['annual_return'] - b_m['annual_return'])*100:+.2f}pp |\n")
    f.write(f"| MaxDD | {fmt_pct(b_m['max_drawdown'])} | {fmt_pct(m['max_drawdown'])} | {(m['max_drawdown'] - b_m['max_drawdown'])*100:+.2f}pp |\n")
    f.write(f"| Sharpe | {fmt(b_m['sharpe_ratio'])} | {fmt(m['sharpe_ratio'])} | {m['sharpe_ratio'] - b_m['sharpe_ratio']:+.3f} |\n")
    f.write(f"| WF Std (G4) | {b_s['sharpe_std']:.4f} | {s['sharpe_std']:.4f} | {s['sharpe_std'] - b_s['sharpe_std']:+.4f} |\n")
    f.write(f"| WF Min (G3) | {b_s['sharpe_min']:.4f} | {s['sharpe_min']:.4f} | {s['sharpe_min'] - b_s['sharpe_min']:+.4f} |\n")

print(f"  MD: {md_path}")
print(f"\n🏁 T48 Phase 5e evaluation complete.")