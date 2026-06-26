"""Generate final comprehensive evaluation report with all charts and analysis."""
import json, sys, os
sys.path.insert(0, '/home/ubuntu/claw_etf_strategy')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Load data
with open('/home/ubuntu/claw_etf_strategy/output/all_ablation_results.json') as f:
    data = json.load(f)

OUT = '/home/ubuntu/claw_etf_strategy/output/evaluation_v2_6_comprehensive.md'
CHART_DIR = '/home/ubuntu/claw_etf_strategy/output/charts'
os.makedirs(CHART_DIR, exist_ok=True)

labels = ["v2.6 all-on", "v2.3 baseline", "ablation A (Fix#1)", "ablation B (Fix#2)", "ablation C (Fix#3)", "v2.5 P1 all-on"]

# ============================================================
# Chart 1: Core Metrics Bar Chart
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics_data = {}
short_labels = ["v2.6", "v2.3 BL", "Abl-A\n(Fix#1)", "Abl-B\n(Fix#2)", "Abl-C\n(Fix#3)", "v2.5 P1"]
for label in labels:
    m = data[label]['metrics']
    metrics_data[label] = {
        'annual_return': m['annual_return'] * 100,
        'max_drawdown': m['max_drawdown'] * 100,
        'sharpe_ratio': m['sharpe_ratio'],
        'calmar_ratio': m['calmar_ratio'],
    }

# Subplot 1: Annual Return
ax = axes[0, 0]
values = [metrics_data[l]['annual_return'] for l in labels]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#FF9800', '#FF9800', '#F44336']
bars = ax.bar(short_labels, values, color=colors, edgecolor='white', linewidth=0.5)
ax.axhline(y=13.5, color='green', linestyle='--', alpha=0.7, label='Gate: 13.5%')
ax.axhline(y=14.11, color='#4CAF50', linestyle=':', alpha=0.5, linewidth=2, label='v2.3 BL')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, f'{val:.2f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('Annual Return (%)', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, loc='lower right')
ax.grid(axis='y', alpha=0.3)

# Subplot 2: Max Drawdown
ax = axes[0, 1]
values = [metrics_data[l]['max_drawdown'] for l in labels]
bars = ax.bar(short_labels, values, color=colors, edgecolor='white', linewidth=0.5)
ax.axhline(y=10.0, color='red', linestyle='--', alpha=0.7, label='Gate: 10%')
ax.axhline(y=7.42, color='#4CAF50', linestyle=':', alpha=0.5, linewidth=2, label='v2.3 BL')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, f'{val:.2f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('Max Drawdown (%)', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, loc='lower right')
ax.grid(axis='y', alpha=0.3)

# Subplot 3: Sharpe Ratio
ax = axes[1, 0]
values = [metrics_data[l]['sharpe_ratio'] for l in labels]
bars = ax.bar(short_labels, values, color=colors, edgecolor='white', linewidth=0.5)
ax.axhline(y=1.05, color='green', linestyle='--', alpha=0.7, label='Gate: 1.05')
ax.axhline(y=1.102, color='#4CAF50', linestyle=':', alpha=0.5, linewidth=2, label='v2.3 BL')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('Sharpe Ratio', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, loc='lower right')
ax.grid(axis='y', alpha=0.3)

# Subplot 4: Calmar Ratio
ax = axes[1, 1]
values = [metrics_data[l]['calmar_ratio'] for l in labels]
bars = ax.bar(short_labels, values, color=colors, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, f'{val:.2f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('Calmar Ratio', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

fig.suptitle('v2.6 Comprehensive Evaluation — Core Metrics Comparison', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
chart1_path = os.path.join(CHART_DIR, 'core_metrics_comparison.png')
fig.savefig(chart1_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Chart 1 saved: {chart1_path}")

# ============================================================
# Chart 2: Ablation Delta Chart
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

base_label = "ablation A (Fix#1)"
base_m = data[base_label]['metrics']

fix_labels = ['Fix #1\n(parallel guard)', 'Fix #2\n(cap 0.30→0.40)', 'Fix #3\n(tighter CRISIS)']
fix_configs = ['v2.5 P1 all-on', 'ablation B (Fix#2)', 'ablation C (Fix#3)']
# For Fix#1: compare Ablation A vs v2.5 P1 (they should be same)

deltas_ret = []
deltas_dd = []
deltas_sharpe = []

# Fix #1: Ablation A (guard fixed) - v2.5 P1 (buggy) = 0 since they're identical
m_a = data["ablation A (Fix#1)"]['metrics']
m_p1 = data["v2.5 P1 all-on"]['metrics']
deltas_ret.append((m_a['annual_return'] - m_p1['annual_return']) * 100)
deltas_dd.append((m_a['max_drawdown'] - m_p1['max_drawdown']) * 100)
deltas_sharpe.append(m_a['sharpe_ratio'] - m_p1['sharpe_ratio'])

# Fix #2: Ablation B - Ablation A
m_b = data["ablation B (Fix#2)"]['metrics']
deltas_ret.append((m_b['annual_return'] - m_a['annual_return']) * 100)
deltas_dd.append((m_b['max_drawdown'] - m_a['max_drawdown']) * 100)
deltas_sharpe.append(m_b['sharpe_ratio'] - m_a['sharpe_ratio'])

# Fix #3: Ablation C - Ablation A
m_c = data["ablation C (Fix#3)"]['metrics']
deltas_ret.append((m_c['annual_return'] - m_a['annual_return']) * 100)
deltas_dd.append((m_c['max_drawdown'] - m_a['max_drawdown']) * 100)
deltas_sharpe.append(m_c['sharpe_ratio'] - m_a['sharpe_ratio'])

x = np.arange(len(fix_labels))
width = 0.25

bars1 = ax.bar(x - width, deltas_ret, width, label='Δ Annual Return (pp)', color='#2196F3')
bars2 = ax.bar(x, deltas_dd, width, label='Δ Max DD (pp)', color='#F44336')
bars3 = ax.bar(x + width, deltas_sharpe, width, label='Δ Sharpe', color='#4CAF50')

for bar, val in zip(bars1, deltas_ret):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.05 if val >= 0 else -0.25),
            f'{val:+.2f}', ha='center', va='bottom' if val >=0 else 'top', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, deltas_dd):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.05 if val >= 0 else -0.25),
            f'{val:+.2f}', ha='center', va='bottom' if val >=0 else 'top', fontsize=9, fontweight='bold')
for bar, val in zip(bars3, deltas_sharpe):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.005 if val >= 0 else -0.02),
            f'{val:+.3f}', ha='center', va='bottom' if val >=0 else 'top', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(fix_labels, fontsize=10)
ax.set_title('Ablation Study — Marginal Contribution of Each Fix', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.grid(axis='y', alpha=0.3)

chart2_path = os.path.join(CHART_DIR, 'ablation_delta.png')
fig.savefig(chart2_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Chart 2 saved: {chart2_path}")

# ============================================================
# Chart 3: Year-by-Year Returns v2.6 vs v2.3
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))

years = ['2013','2014','2015','2016','2017','2018','2019','2020','2021','2022','2023','2024','2025','2026']
v26_y = data["v2.6 all-on"]["yearly"]
v23_y = data["v2.3 baseline"]["yearly"]

v26_returns = [v26_y[y]['return'] * 100 for y in years]
v23_returns = [v23_y[y]['return'] * 100 for y in years]
deltas = [v26_returns[i] - v23_returns[i] for i in range(len(years))]

x = np.arange(len(years))
width = 0.35

bars1 = ax.bar(x - width/2, v26_returns, width, label='v2.6', color='#2196F3', edgecolor='white')
bars2 = ax.bar(x + width/2, v23_returns, width, label='v2.3 Baseline', color='#4CAF50', edgecolor='white', alpha=0.7)

# Add delta labels
for i, (d, yr) in enumerate(zip(deltas, years)):
    color = '#F44336' if d < -0.5 else '#4CAF50' if d > 0.5 else '#9E9E9E'
    ax.annotate(f'{d:+.1f}pp', xy=(i, max(v26_returns[i], v23_returns[i]) + 1.5),
                ha='center', fontsize=8, color=color, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(years, fontsize=10)
ax.set_title('Year-by-Year Returns: v2.6 vs v2.3 Baseline', fontsize=13, fontweight='bold')
ax.set_ylabel('Annual Return (%)')
ax.legend(fontsize=10)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.grid(axis='y', alpha=0.3)

chart3_path = os.path.join(CHART_DIR, 'year_by_year_comparison.png')
fig.savefig(chart3_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Chart 3 saved: {chart3_path}")

# ============================================================
# Chart 4: Market State Distribution
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

states = ['BULL', 'NORMAL', 'CORRECTION', 'CRISIS']
state_colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']

x = np.arange(len(labels))
width = 0.18

for i, state in enumerate(states):
    values = []
    for label in labels:
        ms = data[label]['market_state_distribution']
        total = sum(ms.values())
        values.append(ms.get(f'MarketState.{state}', 0) / total * 100)
    bars = ax.bar(x + i * width, values, width, label=state, color=state_colors[i], edgecolor='white')

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(short_labels, fontsize=9)
ax.set_title('Market State Distribution (% of Weeks)', fontsize=13, fontweight='bold')
ax.set_ylabel('% of Weeks')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

chart4_path = os.path.join(CHART_DIR, 'market_state_distribution.png')
fig.savefig(chart4_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Chart 4 saved: {chart4_path}")

# ============================================================
# Generate final Markdown report
# ============================================================

v26m = data["v2.6 all-on"]["metrics"]
v23m = data["v2.3 baseline"]["metrics"]
v26_ev = data["v2.6 all-on"]["max_dd_event"]
v23_ev = data["v2.3 baseline"]["max_dd_event"]

report = f"""# v2.6 Comprehensive Evaluation Report

**Date:** 2026-06-16 | **Tester:** quant-tester | **Task:** T17

---

## Executive Summary

v2.6 addresses 3 root causes from T13 evaluation:
1. Parallel stop loss guard (backtest.py line 226)
2. Weight cap increase: 0.30 → 0.40
3. Tighter CRISIS thresholds (ms_crisis_mom: -0.15, ms_high_vol_pct: 0.80)

**Decision Gate Result:** ⚠️ **MIXED** — Annual Return 13.58% ✅ / Max DD 10.24% ❌ / Sharpe 1.055 ✅

**Key Finding:** Fix #2 (cap 0.40) accounts for **all** the improvement. Fix #1 is cosmetic. Fix #3 adds no value.

---

## 1. Core Metrics Comparison

| Config | Annual Return | Max DD | Sharpe | Calmar | Vol | Win% | Def Wk% |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **v2.6 all-on** | **13.58%** | **10.24%** | **1.055** | 1.33 | 10.20% | 59.5% | 36.2% |
| v2.3 baseline | 14.11% | 7.42% | 1.102 | 1.90 | 10.19% | 60.1% | 32.5% |
| Ablation A (Fix#1 only) | 12.63% | 11.35% | 1.010 | 1.11 | 9.79% | 60.2% | 36.1% |
| **Ablation B (Fix#2 only)** | **13.67%** | **10.24%** | **1.063** | 1.33 | 10.20% | 59.8% | 35.7% |
| Ablation C (Fix#3 only) | 12.54% | 11.35% | 1.001 | 1.10 | 9.80% | 60.1% | 37.0% |
| v2.5 P1 all-on (buggy) | 12.63% | 11.35% | 1.010 | 1.11 | 9.79% | 60.2% | 36.1% |

![Core Metrics Comparison](charts/core_metrics_comparison.png)

---

## 2. Ablation Study — Contribution Per Fix

![Ablation Delta](charts/ablation_delta.png)

### Fix #1: Parallel Stop Loss Guard (code-level)
- Ablation A vs v2.5 P1: Δ = 0.00pp return / 0.00pp DD / +0.000 sharpe
- **Verdict: NO measurable impact.** The guard is purely defensive — prevents an edge case that wasn't occurring.

### Fix #2: Weight Cap 0.30 → 0.40 ⭐
- Ablation B vs Ablation A: Δ = **+1.04pp return** / -1.11pp DD / +0.053 sharpe
- **Verdict: DOMINANT improvement. This is the only fix that matters.**

### Fix #3: Tighter CRISIS Thresholds
- Ablation C vs Ablation A: Δ = -0.09pp return / 0.00pp DD / -0.009 sharpe
- **Verdict: No value added. Slightly reduces returns.**

### Interaction: Fix #2 + Fix #3 (v2.6 vs Ablation B)
- v2.6 vs Ablation B: Δ = -0.10pp return / 0.00pp DD / -0.008 sharpe
- Fix #3 on top of Fix #2 **reduces returns** — the tighter thresholds are counterproductive.

---

## 3. Year-by-Year Comparison: v2.6 vs v2.3 Baseline

![Year-by-Year Comparison](charts/year_by_year_comparison.png)

| Year | v2.6 | v2.3 | Δ | v2.6 Def% | v2.3 Def% | CRISIS% | Note |
|------|:---:|:---:|:---:|:---:|:---:|:---:|------|
| 2013 | -1.1% | -1.1% | +0.0pp | 25% | 25% | 0% | |
| 2014 | +23.9% | +23.9% | +0.0pp | 25% | 25% | 6% | |
| 2015 | +14.3% | +14.3% | -0.0pp | 42% | 41% | 52% | ⚡ Crisis |
| 2016 | +7.3% | +8.9% | **-1.6pp** | 35% | 27% | 0% | ❌ v2.6 DD peak |
| 2017 | +12.5% | +12.5% | +0.0pp | 25% | 25% | 0% | |
| 2018 | -5.0% | -2.2% | **-2.8pp** | 51% | 27% | 25% | ⚡ Over-defensive |
| 2019 | +23.3% | +23.5% | -0.2pp | 39% | 35% | 12% | 📈 Bull |
| 2020 | +20.1% | +20.1% | +0.0pp | 76% | 76% | 35% | 📈 Bull |
| 2021 | +7.7% | +7.7% | +0.0pp | 33% | 33% | 4% | |
| 2022 | -3.4% | -3.7% | +0.3pp | 44% | 40% | 2% | ⚡ Crisis |
| 2023 | +22.4% | +21.9% | +0.5pp | 26% | 25% | 0% | 📈 Better recovery |
| 2024 | +26.1% | +27.1% | -1.0pp | 42% | 41% | 19% | |
| 2025 | +21.4% | +22.8% | **-1.4pp** | 51% | 51% | 24% | ⚡ Over-defensive |
| 2026 | +5.4% | +5.7% | -0.3pp | 34% | 25% | 0% | |

### Key Patterns

- **Bull years (2019/2020/2023):** v2.6 roughly ties v2.3 — stateful stop loss does not impede bull market returns
- **Crisis years (2018/2025):** v2.6 is **worse** — defense escalates too aggressively, locking in losses
- **2016 DD Peak:** v2.6 max DD occurs in early 2016, a non-crisis year, with 85% defense — the stateful system over-reacted

---

## 4. Max Drawdown Event Analysis

### v2.6 (Max DD: {v26_ev['dd']*100:.2f}%)
- **Date:** {v26_ev['date']}
- **NAV at trough:** {v26_ev['nav']:.4f} (from peak {v26_ev['peak']:.4f})
- **Defense ratio:** {v26_ev['def_ratio']*100:.0f}%
- **Market state:** {v26_ev['market_state']}
- **Context:** Early 2016 — post-2015 China stock crash recovery stalls. The stateful system escalated defense to 85% during NORMAL market, yet the DD was deeper.

### v2.3 Baseline (Max DD: {v23_ev['dd']*100:.2f}%)
- **Date:** {v23_ev['date']}
- **Market state:** {v23_ev['market_state']} (no stateful system)
- **Context:** 2022 bear market bottom — reasonable DD for a major drawdown year.

### Analysis
- v2.6 DD is **1.38× deeper** than v2.3 (+2.82pp)
- The stateful system's DD peak is in a **non-crisis** year (2016) — unexpected and concerning
- Defense at the DD trough was 85% (extreme) yet didn't prevent the drawdown
- **Root cause hypothesis:** The defense escalation ramps up too fast during moderate corrections, then stays elevated preventing recovery participation

---

## 5. Market State Distribution

![Market State Distribution](charts/market_state_distribution.png)

| Config | BULL | NORMAL | CORRECTION | CRISIS |
|--------|:---:|:---:|:---:|:---:|
| v2.6 all-on | 253w (39%) | 197w (30%) | 107w (16%) | **92w (14%)** |
| v2.3 baseline | — | 649w | — | — |
| Ablation A (old thresholds) | 247w (38%) | 196w (30%) | 78w (12%) | **128w (20%)** |
| Ablation B (old thresholds) | 253w (39%) | 197w (30%) | 68w (10%) | **131w (20%)** |
| Ablation C (new thresholds) | 248w (38%) | 194w (30%) | 115w (18%) | **92w (14%)** |
| v2.5 P1 all-on | 247w (38%) | 196w (30%) | 78w (12%) | **128w (20%)** |

### Fix #3 Effect
- Tighter thresholds reduce CRISIS classification from **20% → 14%** (-36 weeks)
- But this reduction shifts weeks to CORRECTION state, not BULL/NORMAL
- The reduced CRISIS detection may miss early warning signals in 2018/2025

---

## 6. Decision Gate

| Condition | Threshold | v2.6 Actual | Result |
|-----------|:---:|:---:|:---:|
| Annual Return ≥ 13.5% | 13.50% | 13.58% | ✅ PASS |
| Max DD ≤ 10% | 10.00% | 10.24% | ❌ FAIL (by 0.24pp) |
| Sharpe ≥ 1.05 | 1.050 | 1.055 | ✅ PASS |

**Decision: ⚠️ MIXED — PM decides next steps.**

- Annual return passes but is still -0.53pp below v2.3 baseline
- Max DD fails the 10% gate — stateful stop loss systematically increases drawdown
- Sharpe passes but is below v2.3 baseline (1.102)

---

## 7. Bonus Test: v2.3 + Cap 0.40

Tested hypothesis: simple v2.3 baseline + cap 0.40 (no stateful stop loss).

**Result: 14.11% / 7.42% / 1.102 — IDENTICAL to v2.3 baseline.**

The cap 0.40 is not binding without stateful stop loss. Fix #2's value comes exclusively from its interaction with the stateful system.

---

## 8. Key Findings

1. **Fix #2 (cap 0.40) is the ONLY effective fix** — all improvement comes from this one change
2. **Fix #1 (parallel guard) is cosmetic** — no measurable impact on performance
3. **Fix #3 (tighter CRISIS thresholds) is counterproductive** — reduces return without improving DD
4. **Stateful stop loss INCREASES drawdown** — v2.6 (+2.82pp) and all variants have worse DD than v2.3
5. **The defense escalation mechanism is too aggressive** — locks in losses during moderate corrections
6. **Ablation B (Fix #2 only: 13.67%/10.24%/1.063) is slightly BETTER than v2.6** — Fix #3 should be removed

---

## 9. Recommendations

### Immediate (v2.6 → v2.6b)
1. **Remove Fix #3** — revert to old CRISIS thresholds (ms_crisis_mom: -0.12, ms_high_vol_pct: 0.67)
2. **Keep Fix #2** — cap 0.40 stays
3. This gives Ablation B config: **13.67% / 10.24% / 1.063**

### Short-term (v2.7 investigation)
4. Investigate why defense escalates to 85% during 2016 NORMAL market
5. Consider defense ramp smoothing: gradual escalation instead of step-function jumps
6. Test lower max_def in CORRECTION state (currently escalates to 95%)

### Medium-term (v2.8 redesign)
7. Test v2.3 + cap 0.40 + **lighter** stateful parameters (less aggressive defense)
8. Consider removing CORRECTION state entirely — let CRISIS handle real drawdowns
9. Benchmark against simple stop loss cap 0.08 with cap 0.40 — may be simpler and better

---

## 10. Ablation Config Files

| Config | File |
|--------|------|
| Ablation A (Fix#1) | `config/strategy_v2_6_ablation_a.yaml` |
| Ablation B (Fix#2) | `config/strategy_v2_6_ablation_b.yaml` |
| Ablation C (Fix#3) | `config/strategy_v2_6_ablation_c.yaml` |
| v2.3 + cap 0.40 | `config/strategy_v2_3_cap040.yaml` |

---

*Generated by quant-tester on 2026-06-16. All data reproducible with `config/*.yaml` files.
Raw JSON: `output/all_ablation_results.json`*
"""

with open(OUT, 'w') as f:
    f.write(report)

print(f"\nReport saved to: {OUT}")
print(f"Charts saved to: {CHART_DIR}/")
print("DONE")
