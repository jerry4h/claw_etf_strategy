"""Comprehensive v2.6 evaluation report: ablation + year-by-year + decision gate + drawdown analysis."""
import json, sys
from pathlib import Path

# Load all results
with open('/home/ubuntu/claw_etf_strategy/output/all_ablation_results.json') as f:
    data = json.load(f)

labels = ["v2.6 all-on", "v2.3 baseline", "ablation A (Fix#1)", "ablation B (Fix#2)", "ablation C (Fix#3)", "v2.5 P1 all-on"]

# ============================================================
# 1. Summary Metrics Table
# ============================================================
print("=" * 70)
print("V2.6 COMPREHENSIVE EVALUATION REPORT")
print("=" * 70)
print()

print("## 1. Core Metrics Comparison")
print()
print(f"{'Config':<28} {'Ann.Ret':>8} {'Max DD':>8} {'Sharpe':>8} {'Calmar':>8} {'Vol':>8} {'Win%':>7} {'DefWk%':>7}")
print("-" * 90)
for label in labels:
    m = data[label]['metrics']
    def_pct = m['defensive_weeks'] / m['total_weeks'] * 100
    print(f"{label:<28} {m['annual_return']*100:>7.2f}% {m['max_drawdown']*100:>7.2f}% "
          f"{m['sharpe_ratio']:>8.3f} {m['calmar_ratio']:>8.2f} {m['annual_volatility']*100:>7.2f}% "
          f"{m['win_rate']*100:>6.1f}% {def_pct:>6.1f}%")
print()

# ============================================================
# 2. Ablation Delta Analysis
# ============================================================
print("## 2. Ablation Study — Contribution of Each Fix")
print()

baseline = data["v2.3 baseline"]["metrics"]
v26 = data["v2.6 all-on"]["metrics"]
abl_a = data["ablation A (Fix#1)"]["metrics"]
abl_b = data["ablation B (Fix#2)"]["metrics"]
abl_c = data["ablation C (Fix#3)"]["metrics"]

# Fix #1 delta: Ablation A - v2.5 P1 (isolates code fix)
p1 = data["v2.5 P1 all-on"]["metrics"]
print("### Fix #1: Parallel Stop Loss Guard (code-level)")
print(f"  Ablation A vs v2.5 P1: ann_ret Δ={abl_a['annual_return']-p1['annual_return']:+.4f} "
      f"({(abl_a['annual_return']-p1['annual_return'])*100:+.2f}pp), "
      f"max_dd Δ={abl_a['max_drawdown']-p1['max_drawdown']:+.4f} "
      f"({(abl_a['max_drawdown']-p1['max_drawdown'])*100:+.2f}pp), "
      f"sharpe Δ={abl_a['sharpe_ratio']-p1['sharpe_ratio']:+.3f}")
print(f"  VERDICT: Fix #1 has NO measurable impact — guard is purely defensive (prevents edge case)")
print()

# Fix #2 delta: Ablation B - Ablation A (isolates cap 0.30→0.40)
print("### Fix #2: Weight Cap 0.30 → 0.40")
print(f"  Ablation B vs Ablation A: ann_ret Δ={abl_b['annual_return']-abl_a['annual_return']:+.4f} "
      f"({(abl_b['annual_return']-abl_a['annual_return'])*100:+.2f}pp), "
      f"max_dd Δ={abl_b['max_drawdown']-abl_a['max_drawdown']:+.4f} "
      f"({(abl_b['max_drawdown']-abl_a['max_drawdown'])*100:+.2f}pp), "
      f"sharpe Δ={abl_b['sharpe_ratio']-abl_a['sharpe_ratio']:+.3f}")
print(f"  VERDICT: Fix #2 is the DOMINANT improvement — +1.04pp annual return, DD unchanged")
print()

# Fix #3 delta: Ablation C - Ablation A (isolates crisis thresholds)
print("### Fix #3: Tighter CRISIS Thresholds")
print(f"  Ablation C vs Ablation A: ann_ret Δ={abl_c['annual_return']-abl_a['annual_return']:+.4f} "
      f"({(abl_c['annual_return']-abl_a['annual_return'])*100:+.2f}pp), "
      f"max_dd Δ={abl_c['max_drawdown']-abl_a['max_drawdown']:+.4f} "
      f"({(abl_c['max_drawdown']-abl_a['max_drawdown'])*100:+.2f}pp), "
      f"sharpe Δ={abl_c['sharpe_ratio']-abl_a['sharpe_ratio']:+.3f}")
print(f"  VERDICT: Fix #3 alone slightly HARMS return (-0.09pp) with same DD — tighter thresholds ")
print(f"  reduce CRISIS classification, potentially missing early warning signals")
print()

# Interaction: v2.6 - Ablation B (Fix #3's marginal benefit when Fix #2 is active)
print("### Interaction Effect: Fix #2 + Fix #3 (v2.6 vs Ablation B)")
print(f"  v2.6 vs Ablation B: ann_ret Δ={v26['annual_return']-abl_b['annual_return']:+.4f} "
      f"({(v26['annual_return']-abl_b['annual_return'])*100:+.2f}pp), "
      f"max_dd Δ={v26['max_drawdown']-abl_b['max_drawdown']:+.4f} "
      f"({(v26['max_drawdown']-abl_b['max_drawdown'])*100:+.2f}pp), "
      f"sharpe Δ={v26['sharpe_ratio']-abl_b['sharpe_ratio']:+.3f}")
print(f"  Fix #3 on top of Fix #2: -0.09pp return, DD unchanged — slightly reduces returns")
print()

# ============================================================
# 3. Year-by-Year Comparison: v2.6 vs v2.3 Baseline
# ============================================================
print("## 3. Year-by-Year Comparison: v2.6 vs v2.3 Baseline")
print()
print(f"{'Year':<6} {'v2.6 Ret':>9} {'v2.3 Ret':>9} {'Δ':>8} {'v2.6 Def%':>9} {'v2.3 Def%':>9} {'v2.6 CRISIS%':>12}")
print("-" * 70)

years = sorted([int(y) for y in data["v2.6 all-on"]["yearly"].keys()])
v26_y = data["v2.6 all-on"]["yearly"]
v23_y = data["v2.3 baseline"]["yearly"]

total_crisis_v26 = 0
total_weeks_v26 = 0

for yr in years:
    ystr = str(yr)
    v26r = v26_y[ystr]['return']
    v23r = v23_y[ystr]['return']
    delta = v26r - v23r
    v26d = v26_y[ystr]['avg_def']
    v23d = v23_y[ystr]['avg_def']
    
    # CRISIS weeks
    ms = v26_y[ystr].get('market_states', {})
    crisis_w = ms.get('MarketState.CRISIS', 0)
    total_w = sum(ms.values())
    crisis_pct = crisis_w / total_w * 100 if total_w else 0
    total_crisis_v26 += crisis_w
    total_weeks_v26 += total_w
    
    # Highlight key years
    marker = ""
    if yr in [2015, 2018, 2022, 2025]:
        marker = " ⚡"  # Crisis years
    if yr in [2019, 2020, 2023]:
        marker = " 📈"  # Bull years
    
    print(f"{yr:<6} {v26r*100:>+8.1f}% {v23r*100:>+8.1f}% {delta*100:>+7.1f}pp "
          f"{v26d*100:>8.1f}% {v23d*100:>8.1f}% {crisis_pct:>11.1f}%{marker}")

print()
print(f"v2.6 Total CRISIS weeks: {total_crisis_v26}/{total_weeks_v26} ({total_crisis_v26/total_weeks_v26*100:.1f}%)")

# Crisis/Bull year comparison
print()
print("### Bull Year Recovery Analysis")
print(f"{'Year':<6} {'v2.6':>9} {'v2.3':>9} {'Δ':>8} {'Assessment':>30}")
print("-" * 65)
bull_years = [2019, 2020, 2023]
for yr in bull_years:
    ystr = str(yr)
    v26r = v26_y[ystr]['return']
    v23r = v23_y[ystr]['return']
    delta = v26r - v23r
    if delta > 0.003:
        assess = "v2.6 better — improves recovery"
    elif delta > -0.003:
        assess = "roughly tied"
    else:
        assess = "v2.6 worse — recovery degraded"
    print(f"{yr:<6} {v26r*100:>+8.1f}% {v23r*100:>+8.1f}% {delta*100:>+7.1f}pp {assess:>30}")

print()
print("### Crisis Year Performance")
print(f"{'Year':<6} {'v2.6':>9} {'v2.3':>9} {'Δ':>8} {'v2.6 Def%':>9} {'Assessment':>30}")
print("-" * 65)
crisis_years = [2015, 2018, 2022, 2025]
for yr in crisis_years:
    ystr = str(yr)
    v26r = v26_y[ystr]['return']
    v23r = v23_y[ystr]['return']
    delta = v26r - v23r
    v26d = v26_y[ystr]['avg_def']
    if delta > 0.005:
        assess = "v2.6 better — defense effective"
    elif delta < -0.005:
        assess = "v2.6 worse — over-defensive"
    else:
        assess = "roughly tied"
    print(f"{yr:<6} {v26r*100:>+8.1f}% {v23r*100:>+8.1f}% {delta*100:>+7.1f}pp {v26d*100:>8.1f}% {assess:>30}")

print()

# ============================================================
# 4. Max Drawdown Event Analysis
# ============================================================
print("## 4. Max Drawdown Event Analysis")
print()

for label in ["v2.6 all-on", "v2.3 baseline"]:
    ev = data[label]['max_dd_event']
    m = data[label]['metrics']
    print(f"### {label}")
    print(f"  Max DD: {ev['dd']*100:.2f}% on {ev['date']}")
    print(f"  NAV at trough: {ev['nav']:.4f} | Peak: {ev['peak']:.4f}")
    print(f"  Def ratio at trough: {ev['def_ratio']*100:.0f}%")
    print(f"  Market state at trough: {ev['market_state']}")
    print(f"  Calmar: {m['calmar_ratio']:.2f}")
    print()

# Compare DD events
v26_ev = data["v2.6 all-on"]["max_dd_event"]
v23_ev = data["v2.3 baseline"]["max_dd_event"]
print("### Drawdown Comparison")
print(f"  v2.6 max DD = {v26_ev['dd']*100:.2f}% on {v26_ev['date']}")
print(f"  v2.3 max DD = {v23_ev['dd']*100:.2f}% on {v23_ev['date']}")
print(f"  v2.6 DD is {v26_ev['dd']/v23_ev['dd']:.2f}x deeper — {v26_ev['dd']*100-v23_ev['dd']*100:+.2f}pp worse")
print()

# ============================================================
# 5. Market State Distribution
# ============================================================
print("## 5. Market State Distribution")
print()
print(f"{'Config':<28} {'BULL':>7} {'NORMAL':>8} {'CORRECTION':>12} {'CRISIS':>8} {'CRISIS%':>9}")
print("-" * 80)

for label in labels:
    ms = data[label]['market_state_distribution']
    total = sum(ms.values())
    bull = ms.get('MarketState.BULL', 0)
    normal = ms.get('MarketState.NORMAL', 0)
    corr = ms.get('MarketState.CORRECTION', 0)
    crisis = ms.get('MarketState.CRISIS', 0)
    crisis_pct = crisis / total * 100
    print(f"{label:<28} {bull:>6}w {normal:>7}w {corr:>11}w {crisis:>7}w {crisis_pct:>8.1f}%")

print()

# Compare crisis % change
v26_crisis = data["v2.6 all-on"]["market_state_distribution"].get('MarketState.CRISIS', 0)
v26_total = sum(data["v2.6 all-on"]["market_state_distribution"].values())
abl_a_crisis = data["ablation A (Fix#1)"]["market_state_distribution"].get('MarketState.CRISIS', 0)
abl_a_total = sum(data["ablation A (Fix#1)"]["market_state_distribution"].values())

print(f"### CRISIS Classification Change")
print(f"  Ablation A (old thresholds): {abl_a_crisis}/{abl_a_total} = {abl_a_crisis/abl_a_total*100:.1f}%")
print(f"  v2.6 (new thresholds):       {v26_crisis}/{v26_total} = {v26_crisis/v26_total*100:.1f}%")
print(f"  Reduction: {abl_a_crisis - v26_crisis} weeks ({-(v26_crisis/abl_a_total - abl_a_crisis/abl_a_total)*100:.1f}pp)")
print()

# ============================================================
# 6. Decision Gate
# ============================================================
print("=" * 70)
print("## 6. DECISION GATE")
print("=" * 70)
print()

r = v26['annual_return']
d = v26['max_drawdown']
s = v26['sharpe_ratio']

print(f"v2.6 Metrics: Annual Return={r*100:.2f}%, Max DD={d*100:.2f}%, Sharpe={s:.3f}")
print()

# Gate conditions
print("| Condition | Required | Actual | Pass? |")
print("|-----------|----------|--------|-------|")

g1 = r >= 0.135
print(f"| Annual return >= 13.5% | 13.50% | {r*100:.2f}% | {'✅' if g1 else '❌'} |")

g2 = d <= 0.10
print(f"| Max DD <= 10% | 10.00% | {d*100:.2f}% | {'✅' if g2 else '❌'} |")

g3 = s >= 1.05
print(f"| Sharpe >= 1.05 | 1.050 | {s:.3f} | {'✅' if g3 else '❌'} |")
print()

all_pass = g1 and g2 and g3
fail = r < 0.13 or d > 0.11

if all_pass:
    verdict = "✅ v2.6 IS an improvement over v2.5 P1 — RECOMMEND for deployment"
elif fail:
    verdict = "❌ Fixes insufficient — need deeper rethink (v2.7)"
else:
    verdict = "⚠️ MIXED RESULT — PM decides next steps"

print(f"DECISION: {verdict}")
print()

# Detailed assessment
print("### Detailed Assessment")
print()
print(f"1. Annual Return: {r*100:.2f}% — passes 13.5% gate, +1.04pp over v2.5 P1")
print(f"   - But still -0.53pp BELOW v2.3 baseline ({baseline['annual_return']*100:.2f}%)")
print(f"2. Max Drawdown: {d*100:.2f}% — FAILS 10% gate (by {d*100-10:.2f}pp)")
print(f"   - +2.82pp WORSE than v2.3 baseline ({baseline['max_drawdown']*100:.2f}%)")
print(f"   - This is the MAIN concern — stateful stop loss increases DD")
print(f"3. Sharpe: {s:.3f} — passes 1.05 gate, but below v2.3 ({baseline['sharpe_ratio']:.3f})")
print()

# ============================================================
# 7. Key Insights & Recommendations
# ============================================================
print("=" * 70)
print("## 7. KEY INSIGHTS & RECOMMENDATIONS")
print("=" * 70)
print()

print("### Findings")
print()
print("1. Fix #2 (cap 0.40) is the ONLY effective fix")
print("   - Alone delivers 13.67%/10.24%/1.063 — virtually identical to v2.6")
print("   - Fix #1 (parallel guard) is cosmetic — no measurable impact")
print("   - Fix #3 (tighter CRISIS thresholds) adds NO value, slightly reduces return")
print()
print("2. The stateful stop loss system INCREASES drawdown risk")
print("   - v2.6 (10.24%) and v2.5 P1 (11.35%) both have higher DD than v2.3 (7.42%)")
print("   - Ablation B without CRISIS tightening: 10.24% DD")
print("   - The additional defense in crisis periods seems to 'lock in' losses before recovery")
print()
print("3. Year-by-year: v2.6 loses in most years vs v2.3")
print("   - 2015: +14.3% vs +14.3% (tied)")
print(f"   - 2018: {v26_y['2018']['return']*100:+.1f}% vs {v23_y['2018']['return']*100:+.1f}% (WORSE — over-defensive at {v26_y['2018']['avg_def']*100:.0f}% def)")
print(f"   - 2019: {v26_y['2019']['return']*100:+.1f}% vs {v23_y['2019']['return']*100:+.1f}% ({'better' if v26_y['2019']['return'] > v23_y['2019']['return'] else 'worse'})")
print()
print("4. Max DD timing issue")
print(f"   - v2.6 max DD: {v26_ev['date']} (early 2016) — position sizing / early strategy")
print(f"   - v2.3 max DD: {v23_ev['date']} — 2022 bear market bottom")
print(f"   - The v2.6 DD peak in 2016 is troubling — not a crisis year, yet deeper DD")
print()

print("### Recommendations")
print()
print("1. SHORT TERM: Strip Fix #3 (tighter CRISIS thresholds)")
print("   - Use Ablation B config (Fix #2 only): 13.67% return, same 10.24% DD")
print("   - Fix #3 adds complexity with no benefit — remove it")
print()
print("2. MEDIUM TERM: Investigate why stateful stop loss increases DD")
print("   - Root cause may be in the defense escalation mechanism")
print("   - Hypothesis: stepping defense too aggressively during moderate corrections")
print("   - Consider: smoother defense ramps, lower max_def in non-crisis states")
print()
print("3. LONG TERM: Consider reverting to simple stop loss + cap 0.40 only")
print("   - v2.3 baseline + simple cap 0.40 might beat all stateful variants")
print("   - Worth testing: v2.3 config with max_single_alloc=0.40")
print()
print("4. DECISION: PM should decide — v2.6 is mixed, Ablation B (Fix #2 only) is slightly better")
print("   Both fail the 10% DD gate but improve over v2.5 P1 in return")
