# T42: Fix 2 Evaluation — 3-State Simplified Regime

**Evaluator**: quant-tester | **Date**: 2026-06-17

## Overview

Evaluating the 3-State Simplified Regime (Fix 2) which removes the never-triggered BUBBLE_WARN and CRISIS states from the 5-State Market Regime Classifier (Fix 1).

**Configurations compared**:
- D4 Baseline: `config/strategy_v2_3_cap040_D4_tuned.yaml` — no regime
- 5-State Regime: `config/strategy_v2_3_cap040_D4_tuned_regime.yaml` — Fix 1 (5 states)
- 3-State: `config/strategy_v2_3_cap040_D4_tuned_regime_3state.yaml` — Fix 2 (3 states, `three_state: true`)

## 1. Full-Period Comparison

| Config | Ann Ret | Max DD | Sharpe | Calmar | Ann Vol | Win Rate | Def Weeks |
|--------|:------:|:------:|:------:|:------:|:-------:|:--------:|:---------:|
| D4 Baseline | 15.65% | 7.58% | 1.216 | 2.06 | 10.36% | 60.55% | 211 |
| 5-State Regime | 14.87% | 7.84% | 1.191 | 1.90 | 9.98% | 59.78% | 243 |
| 3-State | 14.87% | 7.84% | 1.191 | 1.90 | 9.98% | 59.78% | 243 |

**Δ (3-State vs 5-State)**: AnnRet=+0.0000pp, MaxDD=+0.0000pp, Sharpe=+0.0000 → ✅ IDENTICAL

## 2. Regime Distribution

| State | 5-State | 3-State | Match? |
|-------|:------:|:-------:|:------:|
| RISK_ON | 74 (11.4%) | 74 (11.4%) | ✅ |
| CAUTIOUS | 524 (80.7%) | 524 (80.7%) | ✅ |
| DEFENSIVE | 51 (7.9%) | 51 (7.9%) | ✅ |
| BUBBLE_WARN | 0 (0.0%) | 0 (0.0%) | ✅ |
| CRISIS | 0 (0.0%) | 0 (0.0%) | ✅ |

## 3. Walk-Forward Evaluation

| Metric | 5-State | 3-State | Δ |
|--------|:-------:|:-------:|:--:|
| sharpe_mean | 1.1004 | 1.1004 | +0.0000 |
| sharpe_std | 1.0961 | 1.0961 | +0.0000 |
| sharpe_min | -0.3868 | -0.3868 | +0.0000 |
| sharpe_max | 3.2516 | 3.2516 | +0.0000 |
| ret_mean | 0.1186 | 0.1186 | +0.0000 |
| dd_max | 0.0744 | 0.0744 | +0.0000 |

### Window Detail

| W | Period | 5S Sharpe | 3S Sharpe | 5S Ret | 3S Ret | 5S DD | 3S DD |
|---:|--------|:--------:|:--------:|:-----:|:-----:|:----:|:----:|
| 0 | 2016-05→2017-05 | 0.968 | 0.968 | 9.06% | 9.06% | 2.96% | 2.96% |
| 1 | 2017-05→2018-05 | 0.060 | 0.060 | 2.64% | 2.64% | 7.44% | 7.44% |
| 2 | 2018-05→2019-05 | 0.594 | 0.594 | 8.12% | 8.12% | 4.40% | 4.40% |
| 3 | 2019-05→2020-05 | 1.120 | 1.120 | 15.51% | 15.51% | 7.24% | 7.24% |
| 4 | 2020-05→2021-05 | 0.495 | 0.495 | 6.45% | 6.45% | 6.61% | 6.61% |
| 5 | 2021-05→2022-05 | -0.387 | -0.387 | -1.47% | -1.47% | 4.69% | 4.69% |
| 6 | 2022-05→2023-05 | 2.597 | 2.597 | 26.09% | 26.09% | 3.00% | 3.00% |
| 7 | 2023-05→2024-05 | 3.252 | 3.252 | 26.34% | 26.34% | 2.36% | 2.36% |
| 8 | 2024-05→2025-05 | 1.204 | 1.204 | 14.06% | 14.06% | 3.64% | 3.64% |

## 4. Gate Check

| Gate | Criterion | 5-State | 3-State |
|------|-----------|:-------:|:-------:|
| G5 | MaxDD ≤ 8.5% | ✅ 7.44% | ✅ 7.44% |
| G4 | WF Std < 0.60 | ❌ 1.096 | ❌ 1.096 |
| G3 | Min WF ≥ 0.8 | ❌ -0.387 | ❌ -0.387 |

## 5. Verdict & Recommendation

### ✅ ADOPT 3-State Simplified Regime (Fix 2)

- **Zero performance impact**: Full-period and walk-forward metrics are identical.
- BUBBLE_WARN and CRISIS states never trigger over 2013-2025.
- Reduces code complexity: removes 2 states, 2 override blocks, ~40 lines.
- The built-in `classify_regime_3state()` with `BEAR OR TIGHT` rule produces same results in practice — the broader DEFENSIVE condition doesn't change classification because BEAR trends consistently co-occur with THIN breadth in this market.

### Next Steps: Continue Phase 5

- G4 (WF Std 0.97 vs 0.6 target) and G3 remain unaddressed.
- Recommend parameter smoothing or temperature-softmax allocation as Phase 5 Fix 3.

## 6. Charts

- chart_nav_comparison.png
- chart_dd_comparison.png
- chart_wf_comparison.png
- chart_regime_dist.png
- chart_wf_stats.png
