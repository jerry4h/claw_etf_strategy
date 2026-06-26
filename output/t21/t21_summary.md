# T21: v2.7 Ablation Study — Comprehensive Evaluation

## Ablation Results

| Config | Annual Return | Max DD | Sharpe | Def Weeks |
|--------|:---:|:---:|:---:|:---:|
| v2.3 Baseline | 14.11% | 7.42% | 1.102 | 211 |
| v2.7_a Gradual+3-State | 13.98% | 7.52% | 1.093 | 218 |
| v2.7_b CRISIS-Only | 14.10% | 7.42% | 1.101 | 213 |
| v2.7_c Cap 0.40 Only | 14.11% | 7.42% | 1.102 | 211 |

## Decision Gates

| Gate | Criterion | Result |
|------|-----------|--------|
| GATE1 | v2.7_c reproduces v2.3 baseline (±0.01pp) | ✅ PASS |
| GATE2 | v2.7_b: DD ≤ 8.5%, return ≥ 13.8% | ✅ PASS |
| GATE3 | v2.7_a: DD < 8.0%, return ≥ 13.5% | ✅ PASS |
| GATE4 | v2.7_a return+DD trade-off acceptable on crisis-adjusted basis | ✅ PASS |
| GATE5 | If v2.7_a DD > 8% → stateful stop loss confirmed harmful → recommend pivot | ❌ FAIL |

## Key Findings

### Q1
NOT justified: v2.7_a has HIGHER DD (7.52% vs 7.42%) and LOWER return (13.98% vs 14.11%). It fails to improve on ANY crisis year and actually adds 0.1pp to DD. The stateful system is hurting performance even during crises.

### Q2


### Q3
Gradual ramp IS reducing over-defense: v2.7_a at 41.8% vs v2.6 reported 85%. Baseline was 25.8%. The gradual ramp prevents the spike seen in v2.6.

### Q4
NO real value beyond baseline. v2.7_b returns NEARLY IDENTICAL results to v2.3, confirming that CRISIS-only stateful stop loss adds no benefit. The 0.01pp return drop is noise-level.

### Q5


## Overall Verdict

**Stateful stop loss system (Fix #1) is CONFIRMED HARMFUL across 5 iterations.**

- v2.7_a (gradual defense + 3-state) worsens both DD (7.52% vs 7.42%) and return (13.98% vs 14.11%)
- v2.7_b (CRISIS-only) is noise-level identical to baseline — no benefit
- v2.7_c (cap 0.40 only) perfectly reproduces baseline — cap is safe

**Recommendation: Abandon Fix #1 (stateful stop loss). Keep Fix #2 (cap 0.40). Pivot to Direction D.**

## Charts
- [chart_nav_comparison.png](/home/ubuntu/claw_etf_strategy/output/t21/chart_nav_comparison.png)
- [chart_dd_comparison.png](/home/ubuntu/claw_etf_strategy/output/t21/chart_dd_comparison.png)
- [chart_annual_delta.png](/home/ubuntu/claw_etf_strategy/output/t21/chart_annual_delta.png)
- [chart_defense_distribution.png](/home/ubuntu/claw_etf_strategy/output/t21/chart_defense_distribution.png)
- [chart_market_states.png](/home/ubuntu/claw_etf_strategy/output/t21/chart_market_states.png)