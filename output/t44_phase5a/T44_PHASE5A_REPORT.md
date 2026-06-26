# T44: Phase 5a — Direction A Softmax Sweep Report

**Evaluator**: quant-coder | **Date**: 2026-06-17

## Overview

Sweeping 4 softmax temperature values for Direction A (Full-Offensive):

| Config | Temperature T | File |
|--------|:-------------:|------|
| A2  T=0.3 | 0.3 | `config/phase5/p5a_D5_full_t03.yaml` |
| A3  T=0.5 | 0.5 | `config/phase5/p5a_D5_full_t05.yaml` |
| A4  T=1.0 | 1.0 | `config/phase5/p5a_D5_full_t10.yaml` |
| A5  T=2.0 | 2.0 | `config/phase5/p5a_D5_full_t20.yaml` |

**Baseline**: `config/strategy_v2_3_cap040_D4_tuned_regime_3state.yaml` — D4 tuned + 3-State Regime, NO softmax allocation.
**Baseline metrics**: AnnRet=14.87%, MaxDD=7.84%, Sharpe=1.191.

## 1. Full-Period Backtest

| Config | Ann Ret | Max DD | Sharpe | Calmar | Ann Vol | Win Rate | Def Weeks |
|--------|:------:|:------:|:------:|:------:|:-------:|:--------:|:---------:|
| Baseline | 14.87% | 7.84% | 1.191 | 1.90 | 9.98% | 59.78% | 243 |
| A2  T=0.3 | 10.33% | 18.73% | 0.750 (-0.441) | 0.55 | 10.52% | 57.94% | 314 |
| A3  T=0.5 | 10.30% | 18.75% | 0.744 (-0.447) | 0.55 | 10.59% | 57.94% | 314 |
| A4  T=1.0 | 10.29% | 18.77% | 0.741 (-0.450) | 0.55 | 10.61% | 57.78% | 314 |
| A5  T=2.0 | 10.28% | 18.77% | 0.740 (-0.451) | 0.55 | 10.62% | 57.78% | 314 |

## 2. Walk-Forward Evaluation

9 windows, 3-year train / 1-year test (same as T42 methodology).

### Summary Statistics

| Config | WF Sharpe Mean | WF Sharpe Std | WF Sharpe Min | WF Sharpe Max | WF Ret Mean | WF MaxDD | G3 | G4 | G5 |
|--------|:--------------:|:-------------:|:-------------:|:-------------:|:-----------:|:--------:|:--:|:--:|:--:|
| A2  T=0.3 | 0.8953 | 0.9132 | -0.6744 | 2.3499 | 9.93% | 7.28% | ❌ | ❌ | ✅ |
| A3  T=0.5 | 0.8779 | 0.9048 | -0.6986 | 2.3208 | 9.83% | 7.26% | ❌ | ❌ | ✅ |
| A4  T=1.0 | 0.8667 | 0.8991 | -0.7166 | 2.2989 | 9.77% | 7.25% | ❌ | ❌ | ✅ |
| A5  T=2.0 | 0.8611 | 0.8963 | -0.7255 | 2.2880 | 9.74% | 7.24% | ❌ | ❌ | ✅ |

### Window Detail

| W | Period | A2  T=0.3 Sharpe | A2  T=0.3 Ret | A2  T=0.3 DD | A3  T=0.5 Sharpe | A3  T=0.5 Ret | A3  T=0.5 DD | A4  T=1.0 Sharpe | A4  T=1.0 Ret | A4  T=1.0 DD | A5  T=2.0 Sharpe | A5  T=2.0 Ret | A5  T=2.0 DD |
|---:|--------|:--------:|:------:|:----:|:--------:|:------:|:----:|:--------:|:------:|:----:|:--------:|:------:|:----:|
| 0 | 2016-05→2017-05 | 1.546 | 12.43% | 2.93% | 1.523 | 12.20% | 2.94% | 1.504 | 12.04% | 2.95% | 1.495 | 11.95% | 2.96% |
| 1 | 2017-05→2018-05 | 0.085 | 2.90% | 7.28% | 0.088 | 2.92% | 7.26% | 0.090 | 2.95% | 7.25% | 0.091 | 2.96% | 7.24% |
| 2 | 2018-05→2019-05 | 0.796 | 10.92% | 3.57% | 0.806 | 11.28% | 3.56% | 0.813 | 11.54% | 3.56% | 0.817 | 11.67% | 3.55% |
| 3 | 2019-05→2020-05 | 0.976 | 14.73% | 7.02% | 0.978 | 14.76% | 7.02% | 0.979 | 14.78% | 7.01% | 0.980 | 14.80% | 7.01% |
| 4 | 2020-05→2021-05 | -0.019 | 2.07% | 6.46% | -0.017 | 2.09% | 6.44% | -0.015 | 2.10% | 6.42% | -0.015 | 2.10% | 6.42% |
| 5 | 2021-05→2022-05 | -0.674 | -3.98% | 5.71% | -0.699 | -4.25% | 5.86% | -0.717 | -4.45% | 5.99% | -0.725 | -4.55% | 6.05% |
| 6 | 2022-05→2023-05 | 1.863 | 17.76% | 2.82% | 1.831 | 17.62% | 2.85% | 1.806 | 17.52% | 2.86% | 1.794 | 17.46% | 2.87% |
| 7 | 2023-05→2024-05 | 2.350 | 19.31% | 1.89% | 2.321 | 19.19% | 1.94% | 2.299 | 19.11% | 1.97% | 2.288 | 19.07% | 1.99% |
| 8 | 2024-05→2025-05 | 1.135 | 13.22% | 3.61% | 1.071 | 12.63% | 3.62% | 1.041 | 12.36% | 3.63% | 1.026 | 12.22% | 3.63% |

## 3. Gate Analysis

### G4: WF Sharpe Std < 0.60 (PRIMARY)

| Config | WF Sharpe Std | Δ from Baseline | G4? |
|--------|:-------------:|:----------------:|:---:|
| A5  T=2.0 | 0.8963 | +18.2% | ❌ |
| A4  T=1.0 | 0.8991 | +18.0% | ❌ |
| A3  T=0.5 | 0.9048 | +17.4% | ❌ |
| A2  T=0.3 | 0.9132 | +16.7% | ❌ |

### G3: WF Min Sharpe >= 0.80 (SECONDARY)

| Config | WF Sharpe Min | G3? |
|--------|:-------------:|:---:|
| A2  T=0.3 | -0.6744 | ❌ |
| A3  T=0.5 | -0.6986 | ❌ |
| A4  T=1.0 | -0.7166 | ❌ |
| A5  T=2.0 | -0.7255 | ❌ |

### G5: Max DD <= 8.5%

| Config | Full-Period MaxDD | WF MaxDD | G5? |
|--------|:-------------------:|:--------:|:---:|
| A2  T=0.3 | 18.73% | 7.28% | ✅ |
| A3  T=0.5 | 18.75% | 7.26% | ✅ |
| A4  T=1.0 | 18.77% | 7.25% | ✅ |
| A5  T=2.0 | 18.77% | 7.24% | ✅ |

## 4. Recommendation

### 🎯 Primary (G4: Minimize WF Std)

**A5  T=2.0** achieves lowest WF Sharpe Std = **0.8963**

### Secondary (G3: Maximize WF Min Sharpe)

**A2  T=0.3** achieves highest WF Min Sharpe = **-0.6744**

### Composite Score (0.6 × G4 + 0.4 × G3)

Normalized with min-max scaling across the 4 configs:
- G4: score = (max_std − wf_std) / (max_std − min_std), range [0,1]
- G3: score = (wf_min − min_min) / (max_min − min_min), range [0,1]

| Config | G4 Score | G3 Score | Composite |
|--------|:--------:|:--------:|:---------:|
| A5  T=2.0 | 1.0000 | 0.0000 | 0.6000 |
| A4  T=1.0 | 0.8343 | 0.1742 | 0.5703 |
| A3  T=0.5 | 0.4970 | 0.5264 | 0.5088 |
| A2  T=0.3 | 0.0000 | 1.0000 | 0.4000 |

**Note**: All 4 configs are within 0.017 of each other on G4 and 0.051 on G3 — temperature has minimal effect.

## 🏆 FINAL RECOMMENDATION: **A5  T=2.0** (best G4) with **A2  T=0.3** as G3 fallback

### Primary (G4 — Minimize WF Std): **A5 (T=2.0)**
- **WF Sharpe Std = 0.8963** (best of 4, +18.2% from baseline 1.0961)
- G4 target 0.60: ❌ FAIL (still 0.296 above target)

### Secondary (G3 — Maximize WF Min Sharpe): **A2 (T=0.3)**
- **WF Min Sharpe = -0.6744** (best of 4, but WORSE than baseline -0.387)
- G3 target 0.80: ❌ FAIL (1.474 below target)

### G5 (MaxDD ≤ 8.5%): ✅ PASS at all temperatures

### Critical Finding
Direction A (Full-Offensive Softmax) **improves G4 stability (+18%) but severely degrades G3 and full-period performance**:
- Full-period Sharpe drops from 1.191 → 0.74 (−38%)
- Full-period MaxDD jumps from 7.84% → 18.77% (+139%)
- Full-period AnnRet drops from 14.87% → 10.28% (−31%)
- G3 WF Min Sharpe worsens from -0.387 → -0.674

**Direction A is NOT viable as a standalone fix.** It trades too much return for stability. Recommend combining Direction A with Direction B (dynamic weighting) and/or Direction C (constituent signals) in Phase 5b/5c to recover returns while keeping the stability gain.

## 5. Charts

- chart_nav_sweep.png
- chart_dd_sweep.png
- chart_wf_sharpe.png
- chart_full_period_bars.png
- chart_g4_comparison.png
- chart_temp_vs_sharpe.png
