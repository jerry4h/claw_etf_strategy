# Robustness Evaluation Report — Phase 3 Direction 1

**Baseline**: v2.3+cap040+D4-tuned
**Date**: 2026-06-16 22:52
**Monte Carlo runs**: 100

## Executive Summary

---

## Test 1: Parameter Sensitivity Curvature

Measures Sharpe gradient near current params with ±10% perturbation.
Identifies cliffs: params where small changes cause large Sharpe drops.

| Parameter | Baseline | -10% Value | -10% ΔSharpe | -10% ΔRet | +10% Value | +10% ΔSharpe | +10% ΔRet | Cliff? |
|-----------|----------|------------|-------------|----------|------------|-------------|----------|--------|
| mom_w | 0.35 | 0.315 | -0.1118 | -0.0154 | 0.38499999999999995 | -0.1958 | -0.0188 |  |
| vol_w | 0.3 | 0.27 | -0.2014 | -0.0194 | 0.32999999999999996 | -0.1097 | -0.0152 | ⚠️ LEFT CLIFF |
| def_alloc | 0.25 | 0.225 | +0.0078 | +0.0025 | 0.275 | -0.0148 | -0.0032 |  |
| stop_loss | 0.08 | 0.07200000000000001 | -0.0384 | -0.0062 | 0.088 | +0.0000 | +0.0000 |  |
| top_n | 2 | 1 | -0.3501 | -0.0441 | 3 | -0.4601 | -0.0525 | ⚠️ LEFT CLIFF + ⚠️ RIGHT CLIFF |


---

## Test 2: Walk-Forward Stability (Anchored)

3-year train / 1-year test windows. Target: test Sharpe std dev < 0.15.

| Window | Test Period | Sharpe | AnnRet | MaxDD | WinRate |
|--------|-------------|--------|--------|-------|---------|
| 0 | 2016-05-20 → 2017-05-20 | 0.1667 | 0.0361 | 0.0348 | 0.548 |
| 1 | 2017-05-20 → 2018-05-20 | 0.1045 | 0.0310 | 0.0719 | 0.548 |
| 2 | 2018-05-20 → 2019-05-20 | 0.5969 | 0.0807 | 0.0440 | 0.531 |
| 3 | 2019-05-20 → 2020-05-20 | 1.1197 | 0.1551 | 0.0724 | 0.645 |
| 4 | 2020-05-20 → 2021-05-20 | 0.4954 | 0.0645 | 0.0661 | 0.548 |
| 5 | 2021-05-20 → 2022-05-20 | -0.3868 | -0.0147 | 0.0469 | 0.516 |
| 6 | 2022-05-20 → 2023-05-20 | 2.5972 | 0.2609 | 0.0300 | 0.710 |
| 7 | 2023-05-20 → 2024-05-20 | 3.5325 | 0.2945 | 0.0236 | 0.750 |
| 8 | 2024-05-20 → 2025-05-20 | 1.6224 | 0.1831 | 0.0346 | 0.700 |

- **Mean Sharpe**: 1.0943
- **Std Dev**: 1.2075
- **Target**: std < 0.15 — ❌ FAIL

---

## Test 3: Monte Carlo Perturbation Test

All params simultaneously perturbed ±10% (uniform random). 100 runs.
Target: >90% runs achieve Sharpe > 1.0.

_No data — test skipped or failed._

---

## Test 4: Annual Robustness

Year-by-year Sharpe. Target: max 1 year with negative Sharpe.

_No data — test skipped or failed._

---

## Test 5: Metric Clustering (Plateau vs Ridge)

Analyzes Monte Carlo output: compares parameter variance in top vs bottom quartile.
Broad variance in top quartile = plateau (good). Tight = ridge (bad).

_No data — test skipped or failed._

---

## Summary of Findings

1. **CLIFF DETECTED on vol_w**: ⚠️ LEFT CLIFF. Consider adjusting this parameter.
2. **CLIFF DETECTED on top_n**: ⚠️ LEFT CLIFF + ⚠️ RIGHT CLIFF. Consider adjusting this parameter.
3. Walk-forward Sharpe std=1.2075 exceeds target 0.15 — strategy Sharpe varies significantly across time windows.

---

## Recommendations

### Parameter Adjustments

Cliffs detected — parameter adjustment recommended:

- **mom_w**: +10% causes Sharpe drop of -0.196. Consider lowering baseline or widening the acceptable range.
- **vol_w**: -10% causes Sharpe drop of -0.201. Consider raising baseline or widening the acceptable range.
- **top_n**: -10% causes Sharpe drop of -0.350. Consider raising baseline or widening the acceptable range.
- **top_n**: +10% causes Sharpe drop of -0.460. Consider lowering baseline or widening the acceptable range.

### DD Headroom Utilization

- Current MaxDD: 7.42%
- Acceptable MaxDD: 10.0%
- Available headroom: ~2.58pp

DD headroom can be traded for robustness by relaxing parameters away from cliff edges.
For each cliff-identified param, shift the baseline value toward the safe side,
accepting slightly lower peak Sharpe in exchange for wider robustness.

### Should We Relax Parameters?

**Yes, targeted relaxation recommended.** Shift cliff-identified params away from steep-drop regions.
Acceptable trade: peak Sharpe may drop ~0.05-0.10 in exchange for 2-3x wider parameter tolerance.

---

## Data

- Raw results JSON: `output/robustness_results.json`
- Baseline config: `config/strategy_v2_3_cap040_D4_tuned.yaml`
