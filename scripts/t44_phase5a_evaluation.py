#!/usr/bin/env python3
"""
T44: Phase 5a — Sweep Direction A (Full-Offensive Softmax, config only)

Run full-period + walk-forward backtests for 4 Direction A ablation configs.
Configs differ ONLY in softmax temperature: 0.3, 0.5, 1.0, 2.0
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

OUTPUT_DIR = PROJECT_ROOT / 'output' / 't44_phase5a'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    ('A2  T=0.3', 'config/phase5/p5a_D5_full_t03.yaml', 0.3),
    ('A3  T=0.5', 'config/phase5/p5a_D5_full_t05.yaml', 0.5),
    ('A4  T=1.0', 'config/phase5/p5a_D5_full_t10.yaml', 1.0),
    ('A5  T=2.0', 'config/phase5/p5a_D5_full_t20.yaml', 2.0),
]

BASELINE_3S = 'config/strategy_v2_3_cap040_D4_tuned_regime_3state.yaml'

def fmt_pct(v): return f"{v*100:.2f}%"
def fmt(v, d=3): return f"{v:.{d}f}"

# ───────────────────────────────────────────────────────────
# PHASE 1: Full-Period Backtests
# ───────────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 1: Full-Period Backtests (Direction A Softmax Sweep)")
print("=" * 70)
print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

results = {}

# Baseline first
print(f"\n  Running: baseline (3-State Regime, no softmax)")
cfg_b = load_config(PROJECT_ROOT / BASELINE_3S)
r_b = run_backtest(cfg_b)
results['baseline'] = r_b
mb = r_b.metrics
print(f"    AnnRet={fmt_pct(mb['annual_return'])}  MaxDD={fmt_pct(mb['max_drawdown'])}"
      f"  Sharpe={fmt(mb['sharpe_ratio'])}  Calmar={fmt(mb['calmar_ratio'],2)}")

for label, cpath, T in CONFIGS:
    print(f"\n  Running: {label}")
    cfg = load_config(PROJECT_ROOT / cpath)
    r = run_backtest(cfg)
    results[label] = r
    m = r.metrics
    print(f"    AnnRet={fmt_pct(m['annual_return'])}  MaxDD={fmt_pct(m['max_drawdown'])}"
          f"  Sharpe={fmt(m['sharpe_ratio'])}  Calmar={fmt(m['calmar_ratio'],2)}"
          f"  AnnVol={fmt_pct(m['annual_volatility'])}")

# ───────────────────────────────────────────────────────────
# PHASE 2: Walk-Forward Backtests
# ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 2: Walk-Forward Backtests (9 windows, 3yr train / 1yr test)")
print("=" * 70)

from src.data_loader import load_nav_data, resample_weekly

def walk_forward(config_path):
    cfg = load_config(PROJECT_ROOT / config_path)
    nav_daily = load_nav_data(PROJECT_ROOT / cfg.nav_path)
    nav_weekly = resample_weekly(nav_daily, cfg.anchor)
    start_dt = pd.Timestamp(cfg.start_date) if cfg.start_date else nav_weekly.index[0]
    end_dt = pd.Timestamp(cfg.end_date) if cfg.end_date else nav_weekly.index[-1]
    dates = nav_weekly.index[(nav_weekly.index >= start_dt) & (nav_weekly.index <= end_dt)]
    windows = []
    for i in range(156 + 52, len(dates) + 1, 52):
        windows.append((dates[i - 52], dates[i - 1]))
    return cfg, windows

# Pre-compute the windows once (shared across configs since same start/end)
_, wf_windows = walk_forward(CONFIGS[0][1])
print(f"  WF windows: {len(wf_windows)}")
for wi, (ts, te) in enumerate(wf_windows):
    print(f"    W{wi}: {str(ts.date())[:7]} → {str(te.date())[:7]}")

wf_results = {}

for label, cpath, T in CONFIGS:
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
    d = np.array([w['ann_vol'] for w in wf])
    return {
        'n': len(wf),
        'sharpe_mean': float(np.mean(a)), 'sharpe_std': float(np.std(a)),
        'sharpe_min': float(np.min(a)), 'sharpe_max': float(np.max(a)),
        'ret_mean': float(np.mean(b)), 'dd_max': float(np.max(c)),
        'vol_mean': float(np.mean(d)),
    }

# ───────────────────────────────────────────────────────────
# PHASE 3: Comparison Tables
# ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 3: Full-Period Summary")
print("=" * 70)
print(f"{'Config':<25} {'Ann Ret':>8} {'Max DD':>8} {'Sharpe':>8} {'Calmar':>7} {'Ann Vol':>8}")
print("-" * 70)
for rk in ['baseline'] + [c[0] for c in CONFIGS]:
    m = results[rk].metrics
    delta = ""
    if rk != 'baseline':
        ds = m['sharpe_ratio'] - mb['sharpe_ratio']
        delta = f" (ΔSharpe={ds:+.3f})"
    print(f"{rk:<25} {fmt_pct(m['annual_return']):>8} {fmt_pct(m['max_drawdown']):>8}"
          f" {fmt(m['sharpe_ratio']):>8} {fmt(m['calmar_ratio'],2):>7} {fmt_pct(m['annual_volatility']):>8}")

print("\n" + "=" * 70)
print("PHASE 3b: Walk-Forward Summary")
print("=" * 70)

wf_sums = {}
print(f"\n{'Config':<25} {'WF Sharpe Mean':>14} {'WF Sharpe Std':>13} {'WF Sharpe Min':>14} {'G3 (Min>=0.8)':>13} {'G4 (Std<0.6)':>13} {'G5 (MaxDD<=8.5%)':>16}")
print("-" * 110)
for label in [c[0] for c in CONFIGS]:
    s = wf_summary(wf_results[label])
    wf_sums[label] = s
    g3 = '✅' if s['sharpe_min'] >= 0.8 else '❌'
    g4 = '✅' if s['sharpe_std'] < 0.60 else '❌'
    g5 = '✅' if s['dd_max'] <= 0.085 else '❌'
    print(f"{label:<25} {s['sharpe_mean']:>14.4f} {s['sharpe_std']:>13.4f} {s['sharpe_min']:>14.4f} {g3:>13} {g4:>13} {g5:>16}")

# ───────────────────────────────────────────────────────────
# PHASE 4: Charts
# ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 4: Charts")
print("=" * 70)

colors = {
    'A2  T=0.3': '#E53935',   # red (near-hard)
    'A3  T=0.5': '#FB8C00',   # orange (moderate)
    'A4  T=1.0': '#43A047',   # green (standard)
    'A5  T=2.0': '#1E88E5',   # blue (uniform-ish)
    'baseline': '#757575',     # grey
}
paths = []

# 1) NAV chart
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(results['baseline'].nav_series.index, results['baseline'].nav_series['nav'],
        color='#757575', lw=1.2, ls='--', label='Baseline (3S, no D5)', alpha=0.7)
for label in [c[0] for c in CONFIGS]:
    ax.plot(results[label].nav_series.index, results[label].nav_series['nav'],
            color=colors[label], lw=1, label=label)
ax.axhline(y=1, color='gray', ls='--', lw=0.5, alpha=0.5)
ax.set_title('NAV: Direction A Softmax Sweep (T=0.3→2.0)', fontsize=14)
ax.set_ylabel('NAV'); ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.3)
p = OUTPUT_DIR / 'chart_nav_sweep.png'
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig); paths.append(str(p))
print(f"  Chart: {p.name}")

# 2) Drawdown chart
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
for i, (label, _, T) in enumerate(CONFIGS):
    df = results[label].nav_series
    axes[i].fill_between(df.index, 0, df['drawdown'] * 100, color=colors[label], alpha=0.3)
    axes[i].plot(df.index, df['drawdown'] * 100, color=colors[label], lw=0.8)
    axes[i].set_ylabel(f'{label} DD (%)'); axes[i].invert_yaxis(); axes[i].grid(True, alpha=0.3)
axes[0].set_title('Drawdown Comparison: Direction A Softmax Sweep', fontsize=14)
p = OUTPUT_DIR / 'chart_dd_sweep.png'
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig); paths.append(str(p))
print(f"  Chart: {p.name}")

# 3) WF Sharpe per window
fig, ax = plt.subplots(figsize=(14, 5))
for label in [c[0] for c in CONFIGS]:
    wf = wf_results[label]
    ax.plot(range(len(wf)), [w['sharpe'] for w in wf], 'o-',
            color=colors[label], lw=1.5, markersize=6, label=label)
ax.axhline(y=0, color='gray', ls=':', lw=0.5)
ax.axhline(y=0.8, color='green', ls='--', lw=1, alpha=0.5, label='G3 target (0.8)')
ref_wf = wf_results[CONFIGS[0][0]]
ax.set_xticks(range(len(ref_wf)))
ax.set_xticklabels([w['period'] for w in ref_wf], rotation=45)
ax.set_title('Walk-Forward Sharpe per Window', fontsize=14)
ax.set_ylabel('Sharpe'); ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
p = OUTPUT_DIR / 'chart_wf_sharpe.png'
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig); paths.append(str(p))
print(f"  Chart: {p.name}")

# 4) Full-Period bar chart
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
labels = [c[0] for c in CONFIGS]
c_list = [colors[l] for l in labels]
ann_rets = [results[l].metrics['annual_return'] * 100 for l in labels]
maxdds = [results[l].metrics['max_drawdown'] * 100 for l in labels]
sharpes = [results[l].metrics['sharpe_ratio'] for l in labels]
axes[0].bar(labels, ann_rets, color=c_list, alpha=0.85)
axes[0].axhline(y=mb['annual_return'] * 100, color='#757575', ls='--', lw=1, label='Baseline')
axes[0].set_title('Annual Return (%)'); axes[0].legend(fontsize=7)
axes[0].tick_params(axis='x', rotation=15)
axes[1].bar(labels, maxdds, color=c_list, alpha=0.85)
axes[1].axhline(y=mb['max_drawdown'] * 100, color='#757575', ls='--', lw=1)
axes[1].set_title('Max DD (%)')
axes[1].tick_params(axis='x', rotation=15)
axes[2].bar(labels, sharpes, color=c_list, alpha=0.85)
axes[2].axhline(y=mb['sharpe_ratio'], color='#757575', ls='--', lw=1)
axes[2].set_title('Sharpe Ratio')
axes[2].tick_params(axis='x', rotation=15)
fig.suptitle('Full-Period Metrics: Direction A Softmax Sweep', fontsize=14)
fig.tight_layout()
p = OUTPUT_DIR / 'chart_full_period_bars.png'
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig); paths.append(str(p))
print(f"  Chart: {p.name}")

# 5) WF Std (G4) comparison
fig, ax = plt.subplots(figsize=(9, 5))
wf_stds = [wf_sums[l]['sharpe_std'] for l in labels]
bars = ax.bar(labels, wf_stds, color=c_list, alpha=0.85)
ax.axhline(y=0.60, color='red', ls='--', lw=1.5, label='G4 target (0.60)')
for bar, val in zip(bars, wf_stds):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontsize=10)
ax.set_title('WF Sharpe Std (G4: < 0.60)', fontsize=14)
ax.set_ylabel('Sharpe Std'); ax.legend()
ax.grid(True, alpha=0.3, axis='y')
p = OUTPUT_DIR / 'chart_g4_comparison.png'
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig); paths.append(str(p))
print(f"  Chart: {p.name}")

# 6) Temperature vs Sharpe relationship
fig, ax = plt.subplots(figsize=(9, 5))
Ts = [0.3, 0.5, 1.0, 2.0]
ax.plot(Ts, [results[l].metrics['sharpe_ratio'] for l in labels], 'o-',
        color='#333', lw=2, markersize=12, label='Full-Period Sharpe')
ax.plot(Ts, [wf_sums[l]['sharpe_mean'] for l in labels], 's--',
        color='#666', lw=2, markersize=12, label='WF Sharpe Mean')
ax.plot(Ts, [wf_sums[l]['sharpe_std'] for l in labels], '^:',
        color='#D32F2F', lw=2, markersize=12, label='WF Sharpe Std (G4)')
ax.set_xlabel('Temperature T'); ax.set_ylabel('Sharpe')
ax.set_title('Temperature vs Sharpe Metrics', fontsize=14)
ax.set_xscale('log'); ax.legend()
ax.grid(True, alpha=0.3)
p = OUTPUT_DIR / 'chart_temp_vs_sharpe.png'
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig); paths.append(str(p))
print(f"  Chart: {p.name}")

# ───────────────────────────────────────────────────────────
# PHASE 5: Recommendation
# ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 5: Gate Analysis & Recommendation")
print("=" * 70)

# Find best for G4 (primary: min WF Std)
best_g4 = min(labels, key=lambda l: wf_sums[l]['sharpe_std'])
best_g4_std = wf_sums[best_g4]['sharpe_std']

# Find best for G3 (secondary: max WF Min Sharpe)
best_g3 = max(labels, key=lambda l: wf_sums[l]['sharpe_min'])
best_g3_min = wf_sums[best_g3]['sharpe_min']

# G5 check
for label in labels:
    s = wf_sums[label]
    m = results[label].metrics
    g3 = 'PASS' if s['sharpe_min'] >= 0.8 else f"FAIL ({s['sharpe_min']:.3f})"
    g4 = 'PASS' if s['sharpe_std'] < 0.60 else f"FAIL ({s['sharpe_std']:.3f})"
    g5 = 'PASS' if s['dd_max'] <= 0.085 else f"FAIL ({s['dd_max']*100:.2f}%)"
    print(f"  {label}: G3={g3}  G4={g4}  G5={g5}  |  FP Sharpe={m['sharpe_ratio']:.3f}  FP MaxDD={m['max_drawdown']*100:.2f}%")

print(f"\n  ─── G4 Optimization (primary) ───")
print(f"  Best WF Std: {best_g4} → {best_g4_std:.4f}")

# Calculate improvement over baseline
base_std = 1.0961  # from T42 report
for label in labels:
    impr = (base_std - wf_sums[label]['sharpe_std']) / base_std * 100
    print(f"    {label}: WF Std={wf_sums[label]['sharpe_std']:.4f} (Δ from baseline {base_std:.4f}: {impr:+.1f}%)")

print(f"\n  ─── G3 Optimization (secondary) ───")
print(f"  Best WF Min Sharpe: {best_g3} → {best_g3_min:.4f}")
for label in labels:
    print(f"    {label}: WF Min Sharpe={wf_sums[label]['sharpe_min']:.4f}")

# Recommendation
print(f"\n  ═══ RECOMMENDATION ═══")
# Use composite: G4 (weight 0.6) + G3 (weight 0.4) normalized
scores = {}
for label in labels:
    # G4: lower is better → normalize as 1 - (val/max)
    max_std = max(wf_sums[l]['sharpe_std'] for l in labels)
    g4_score = 1 - wf_sums[label]['sharpe_std'] / max_std if max_std > 0 else 0
    # G3: higher is better → normalize as val/max
    max_min = max(wf_sums[l]['sharpe_min'] for l in labels)
    g3_score = wf_sums[label]['sharpe_min'] / (max_min + 0.5)  # +0.5 to avoid negative dominance
    # Composite
    scores[label] = 0.6 * g4_score + 0.4 * g3_score
    print(f"    {label}: composite={scores[label]:.4f} (G4_score={g4_score:.4f} G3_score={g3_score:.4f})")

best = max(scores, key=scores.get)
print(f"\n  🏆 RECOMMENDED CONFIG: {best}")

# ───────────────────────────────────────────────────────────
# PHASE 6: JSON Output
# ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 6: Output Files")
print("=" * 70)

output = {
    'meta': {
        'task': 'T44 Phase 5a',
        'direction': 'A (Full-Offensive Softmax)',
        'configs': [{'label': l, 'config': c, 'temperature': T} for l, c, T in CONFIGS],
        'baseline': BASELINE_3S,
        'baseline_metrics': {
            'annual_return': float(mb['annual_return']),
            'max_drawdown': float(mb['max_drawdown']),
            'sharpe_ratio': float(mb['sharpe_ratio']),
            'calmar_ratio': float(mb['calmar_ratio']),
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
        for l in labels
    },
    'walk_forward': {l: wf_results[l] for l in labels},
    'walk_forward_summary': wf_sums,
    'gates': {
        l: {
            'G5_maxdd_85': 'PASS' if wf_sums[l]['dd_max'] <= 0.085 else 'FAIL',
            'G4_wf_std_60': 'PASS' if wf_sums[l]['sharpe_std'] < 0.60 else 'FAIL',
            'G3_min_sharpe_08': 'PASS' if wf_sums[l]['sharpe_min'] >= 0.8 else 'FAIL',
        }
        for l in labels
    },
    'recommendation': {
        'best_g4': best_g4,
        'best_g4_std': best_g4_std,
        'best_g3': best_g3,
        'best_g3_min': best_g3_min,
        'best_composite': best,
        'composite_scores': scores,
    },
    'charts': [Path(p).name for p in paths],
}

json_path = OUTPUT_DIR / 't44_phase5a_results.json'
with open(json_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"  JSON: {json_path}")

# ───────────────────────────────────────────────────────────
# PHASE 7: Markdown Report
# ───────────────────────────────────────────────────────────
md_path = OUTPUT_DIR / 'T44_PHASE5A_REPORT.md'
with open(md_path, 'w') as f:
    f.write("# T44: Phase 5a — Direction A Softmax Sweep Report\n\n")
    f.write(f"**Evaluator**: quant-coder | **Date**: {time.strftime('%Y-%m-%d')}\n\n")
    f.write("## Overview\n\n")
    f.write("Sweeping 4 softmax temperature values for Direction A (Full-Offensive):\n\n")
    f.write("| Config | Temperature T | File |\n")
    f.write("|--------|:-------------:|------|\n")
    for l, c, T in CONFIGS:
        f.write(f"| {l} | {T} | `{c}` |\n")
    f.write(f"\n**Baseline**: `{BASELINE_3S}` — D4 tuned + 3-State Regime, NO softmax allocation.\n")
    f.write(f"**Baseline metrics**: AnnRet={fmt_pct(mb['annual_return'])}, "
            f"MaxDD={fmt_pct(mb['max_drawdown'])}, Sharpe={fmt(mb['sharpe_ratio'])}.\n\n")

    f.write("## 1. Full-Period Backtest\n\n")
    f.write("| Config | Ann Ret | Max DD | Sharpe | Calmar | Ann Vol | Win Rate | Def Weeks |\n")
    f.write("|--------|:------:|:------:|:------:|:------:|:-------:|:--------:|:---------:|\n")
    # Baseline row
    f.write(f"| Baseline | {fmt_pct(mb['annual_return'])} | {fmt_pct(mb['max_drawdown'])} | "
            f"{fmt(mb['sharpe_ratio'])} | {fmt(mb['calmar_ratio'],2)} | "
            f"{fmt_pct(mb['annual_volatility'])} | {fmt_pct(mb['win_rate'])} | {mb['defensive_weeks']} |\n")
    for l in labels:
        m = results[l].metrics
        ds = m['sharpe_ratio'] - mb['sharpe_ratio']
        f.write(f"| {l} | {fmt_pct(m['annual_return'])} | {fmt_pct(m['max_drawdown'])} | "
                f"{fmt(m['sharpe_ratio'])} ({ds:+.3f}) | {fmt(m['calmar_ratio'],2)} | "
                f"{fmt_pct(m['annual_volatility'])} | {fmt_pct(m['win_rate'])} | {m['defensive_weeks']} |\n")

    f.write("\n## 2. Walk-Forward Evaluation\n\n")
    f.write("9 windows, 3-year train / 1-year test (same as T42 methodology).\n\n")
    f.write("### Summary Statistics\n\n")
    f.write("| Config | WF Sharpe Mean | WF Sharpe Std | WF Sharpe Min | WF Sharpe Max | WF Ret Mean | WF MaxDD | G3 | G4 | G5 |\n")
    f.write("|--------|:--------------:|:-------------:|:-------------:|:-------------:|:-----------:|:--------:|:--:|:--:|:--:|\n")
    for l in labels:
        s = wf_sums[l]
        g3 = '✅' if s['sharpe_min'] >= 0.8 else '❌'
        g4 = '✅' if s['sharpe_std'] < 0.60 else '❌'
        g5 = '✅' if s['dd_max'] <= 0.085 else '❌'
        f.write(f"| {l} | {s['sharpe_mean']:.4f} | {s['sharpe_std']:.4f} | {s['sharpe_min']:.4f} | "
                f"{s['sharpe_max']:.4f} | {fmt_pct(s['ret_mean'])} | {fmt_pct(s['dd_max'])} | {g3} | {g4} | {g5} |\n")

    f.write("\n### Window Detail\n\n")
    # Headers: period + one column per config for sharpe
    n_win = len(wf_results[labels[0]])
    f.write("| W | Period |")
    for l in labels:
        f.write(f" {l} Sharpe | {l} Ret | {l} DD |")
    f.write("\n")
    f.write("|---:|--------|" + ":--------:|:------:|:----:|" * len(labels) + "\n")
    for w_idx in range(n_win):
        f.write(f"| {w_idx} | {wf_results[labels[0]][w_idx]['period']} |")
        for l in labels:
            w = wf_results[l][w_idx]
            f.write(f" {w['sharpe']:.3f} | {fmt_pct(w['ann_ret'])} | {fmt_pct(w['max_dd'])} |")
        f.write("\n")

    f.write("\n## 3. Gate Analysis\n\n")
    f.write("### G4: WF Sharpe Std < 0.60 (PRIMARY)\n\n")
    baseline_std = 1.0961
    f.write("| Config | WF Sharpe Std | Δ from Baseline | G4? |\n")
    f.write("|--------|:-------------:|:----------------:|:---:|\n")
    for l in sorted(labels, key=lambda l: wf_sums[l]['sharpe_std']):
        s = wf_sums[l]
        impr = (baseline_std - s['sharpe_std']) / baseline_std * 100
        g4 = '✅' if s['sharpe_std'] < 0.60 else '❌'
        f.write(f"| {l} | {s['sharpe_std']:.4f} | {impr:+.1f}% | {g4} |\n")

    f.write("\n### G3: WF Min Sharpe >= 0.80 (SECONDARY)\n\n")
    f.write("| Config | WF Sharpe Min | G3? |\n")
    f.write("|--------|:-------------:|:---:|\n")
    for l in sorted(labels, key=lambda l: wf_sums[l]['sharpe_min'], reverse=True):
        s = wf_sums[l]
        g3 = '✅' if s['sharpe_min'] >= 0.8 else '❌'
        f.write(f"| {l} | {s['sharpe_min']:.4f} | {g3} |\n")

    f.write("\n### G5: Max DD <= 8.5%\n\n")
    f.write("| Config | Full-Period MaxDD | WF MaxDD | G5? |\n")
    f.write("|--------|:-------------------:|:--------:|:---:|\n")
    for l in labels:
        f.write(f"| {l} | {fmt_pct(results[l].metrics['max_drawdown'])} | "
                f"{fmt_pct(wf_sums[l]['dd_max'])} | "
                f"{'✅' if wf_sums[l]['dd_max'] <= 0.085 else '❌'} |\n")

    f.write("\n## 4. Recommendation\n\n")
    # Best for G4
    f.write(f"### 🎯 Primary (G4: Minimize WF Std)\n\n")
    f.write(f"**{best_g4}** achieves lowest WF Sharpe Std = **{best_g4_std:.4f}**\n\n")
    f.write(f"### Secondary (G3: Maximize WF Min Sharpe)\n\n")
    f.write(f"**{best_g3}** achieves highest WF Min Sharpe = **{best_g3_min:.4f}**\n\n")
    f.write("### Composite Score (0.6 × G4 + 0.4 × G3)\n\n")
    f.write("| Config | G4 Score | G3 Score | Composite |\n")
    f.write("|--------|:--------:|:--------:|:---------:|\n")
    for l in sorted(labels, key=lambda l: scores[l], reverse=True):
        max_std = max(wf_sums[ll]['sharpe_std'] for ll in labels)
        max_min = max(wf_sums[ll]['sharpe_min'] for ll in labels)
        g4s = 1 - wf_sums[l]['sharpe_std'] / max_std if max_std > 0 else 0
        g3s = wf_sums[l]['sharpe_min'] / (max_min + 0.5)
        f.write(f"| {l} | {g4s:.4f} | {g3s:.4f} | {scores[l]:.4f} |\n")

    f.write(f"\n## 🏆 FINAL RECOMMENDATION: **{best}**\n\n")
    f.write(f"This config provides the best balance of G4 improvement (WF Sharpe Std) "
            f"and G3 (WF Min Sharpe).\n")
    f.write(f"- **G4 status**: {'✅ PASS' if wf_sums[best]['sharpe_std'] < 0.60 else '❌ FAIL — needs further optimization'}\n")
    f.write(f"- **G3 status**: {'✅ PASS' if wf_sums[best]['sharpe_min'] >= 0.8 else '❌ FAIL — needs further optimization'}\n")
    f.write(f"- **G5 status**: {'✅ PASS' if wf_sums[best]['dd_max'] <= 0.085 else '❌ FAIL'}\n\n")

    rec_idx = [i for i, (l, _, _) in enumerate(CONFIGS) if l == best][0]
    f.write(f"### Best Config Details\n\n")
    f.write(f"- **Config file**: `{CONFIGS[rec_idx][1]}`\n")
    f.write(f"- **Temperature**: T = {CONFIGS[rec_idx][2]}\n")
    f.write(f"- **Full-Period**: Sharpe = {results[best].metrics['sharpe_ratio']:.3f}, "
            f"MaxDD = {fmt_pct(results[best].metrics['max_drawdown'])}, "
            f"AnnRet = {fmt_pct(results[best].metrics['annual_return'])}\n")
    f.write(f"- **Walk-Forward**: Mean Sharpe = {wf_sums[best]['sharpe_mean']:.4f}, "
            f"Std = {wf_sums[best]['sharpe_std']:.4f}, Min = {wf_sums[best]['sharpe_min']:.4f}\n\n")

    f.write("## 5. Charts\n\n")
    for p in paths:
        f.write(f"- {Path(p).name}\n")

print(f"  Report: {md_path}")

print(f"\n{'='*70}")
print(f"Done at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*70}")
