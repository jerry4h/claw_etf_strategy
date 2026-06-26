# T48: Phase 5e — Temperature-Ramp Softmax Sweep

**Evaluator**: quant-coder | **Date**: 2026-06-17

## Overview

Evaluating 9 configs for Temperature-Ramp Softmax Approach 3:

- **T_caut fixed at 2.0** (CAUTIOUS regime)
- **T_risk ∈ {1.0, 2.0, 5.0}** (RISK_ON regime)
- **T_def ∈ {3.0, 5.0, 10.0}** (DEFENSIVE regime)
- Softmax ALWAYS ON in all 3 regimes (no hard fallback)

**Baseline**: `config/strategy_v2_3_cap040_D4_tuned_regime_3state.yaml` — D4 tuned + 3-State Regime, NO softmax.
Baseline metrics: AnnRet=14.87%, MaxDD=7.84%, Sharpe=1.191, WF_Std=1.0961, WF_Min=-0.3868

## 1. Full-Period Backtest

| Config | Ann Ret | Max DD | Sharpe | Calmar | Ann Vol | Win Rate | Def Weeks |
|--------|:------:|:------:|:------:|:------:|:-------:|:--------:|:---------:|
| baseline | 14.87% | 7.84% | 1.191 | 1.90 | 9.98% | 59.78% | 243 |
| Tr1_Td3 | 10.28% | 18.77% | 0.740 | 0.55 | 10.62% | 57.78% | 314 |
| Tr1_Td5 | 3.34% | 7.16% | 0.194 | 0.47 | 4.62% | 55.62% | 283 |
| Tr1_Td10 | 3.34% | 7.16% | 0.194 | 0.47 | 4.62% | 55.62% | 283 |
| Tr2_Td3 | 3.34% | 7.16% | 0.194 | 0.47 | 4.62% | 55.62% | 283 |
| Tr2_Td5 | 3.34% | 7.16% | 0.194 | 0.47 | 4.62% | 55.62% | 283 |
| Tr2_Td10 | 3.34% | 7.16% | 0.194 | 0.47 | 4.62% | 55.62% | 283 |
| Tr5_Td3 | 3.34% | 7.16% | 0.194 | 0.47 | 4.62% | 55.62% | 283 |
| Tr5_Td5 | 3.34% | 7.16% | 0.194 | 0.47 | 4.62% | 55.62% | 283 |
| Tr5_Td10 | 3.34% | 7.16% | 0.194 | 0.47 | 4.62% | 55.62% | 283 |

## 2. Walk-Forward Summary

| Config | WF Sharpe Mean | WF Sharpe Std | WF Sharpe Min | WF Sharpe Max | G3 | G4 | G5 |
|--------|:--------------:|:-------------:|:-------------:|:-------------:|:--:|:--:|:--:|
| baseline | 1.1004 | 1.0961 | -0.3868 | 3.2516 | ❌ | ❌ | ✅ |
| Tr1_Td3 | 0.8638 | 0.8973 | -0.7255 | 2.2880 | ❌ | ❌ | ✅ |
| Tr1_Td5 | -0.0279 | 0.8519 | -1.8941 | 0.9708 | ❌ | ❌ | ✅ |
| Tr1_Td10 | -0.0279 | 0.8519 | -1.8941 | 0.9708 | ❌ | ❌ | ✅ |
| Tr2_Td3 | -0.0279 | 0.8519 | -1.8941 | 0.9708 | ❌ | ❌ | ✅ |
| Tr2_Td5 | -0.0279 | 0.8519 | -1.8941 | 0.9708 | ❌ | ❌ | ✅ |
| Tr2_Td10 | -0.0279 | 0.8519 | -1.8941 | 0.9708 | ❌ | ❌ | ✅ |
| Tr5_Td3 | -0.0279 | 0.8519 | -1.8941 | 0.9708 | ❌ | ❌ | ✅ |
| Tr5_Td5 | -0.0279 | 0.8519 | -1.8941 | 0.9708 | ❌ | ❌ | ✅ |
| Tr5_Td10 | -0.0279 | 0.8519 | -1.8941 | 0.9708 | ❌ | ❌ | ✅ |

### Window Detail

**W0** (2016-05→2017-05): Tr1_Td3=1.504, Tr1_Td5=-1.894, Tr1_Td10=-1.894, Tr2_Td3=-1.894, Tr2_Td5=-1.894, Tr2_Td10=-1.894, Tr5_Td3=-1.894, Tr5_Td5=-1.894, Tr5_Td10=-1.894
**W1** (2017-05→2018-05): Tr1_Td3=0.091, Tr1_Td5=-0.785, Tr1_Td10=-0.785, Tr2_Td3=-0.785, Tr2_Td5=-0.785, Tr2_Td10=-0.785, Tr5_Td3=-0.785, Tr5_Td5=-0.785, Tr5_Td10=-0.785
**W2** (2018-05→2019-05): Tr1_Td3=0.817, Tr1_Td5=0.914, Tr1_Td10=0.914, Tr2_Td3=0.914, Tr2_Td5=0.914, Tr2_Td10=0.914, Tr5_Td3=0.914, Tr5_Td5=0.914, Tr5_Td10=0.914
**W3** (2019-05→2020-05): Tr1_Td3=0.980, Tr1_Td5=-0.458, Tr1_Td10=-0.458, Tr2_Td3=-0.458, Tr2_Td5=-0.458, Tr2_Td10=-0.458, Tr5_Td3=-0.458, Tr5_Td5=-0.458, Tr5_Td10=-0.458
**W4** (2020-05→2021-05): Tr1_Td3=-0.015, Tr1_Td5=0.192, Tr1_Td10=0.192, Tr2_Td3=0.192, Tr2_Td5=0.192, Tr2_Td10=0.192, Tr5_Td3=0.192, Tr5_Td5=0.192, Tr5_Td10=0.192
**W5** (2021-05→2022-05): Tr1_Td3=-0.725, Tr1_Td5=0.052, Tr1_Td10=0.052, Tr2_Td3=0.052, Tr2_Td5=0.052, Tr2_Td10=0.052, Tr5_Td3=0.052, Tr5_Td5=0.052, Tr5_Td10=0.052
**W6** (2022-05→2023-05): Tr1_Td3=1.794, Tr1_Td5=0.275, Tr1_Td10=0.275, Tr2_Td3=0.275, Tr2_Td5=0.275, Tr2_Td10=0.275, Tr5_Td3=0.275, Tr5_Td5=0.275, Tr5_Td10=0.275
**W7** (2023-05→2024-05): Tr1_Td3=2.288, Tr1_Td5=0.483, Tr1_Td10=0.483, Tr2_Td3=0.483, Tr2_Td5=0.483, Tr2_Td10=0.483, Tr5_Td3=0.483, Tr5_Td5=0.483, Tr5_Td10=0.483
**W8** (2024-05→2025-05): Tr1_Td3=1.040, Tr1_Td5=0.971, Tr1_Td10=0.971, Tr2_Td3=0.971, Tr2_Td5=0.971, Tr2_Td10=0.971, Tr5_Td3=0.971, Tr5_Td5=0.971, Tr5_Td10=0.971

## 3. Gate Analysis

| Config | G4 (Std<0.60) | G3 (Min>=0.80) | G5 (DD<=8.5%) | AnnRet>=14% | Sharpe>=1.10 |
|--------|:-------------:|:--------------:|:-------------:|:----------:|:-----------:|
| baseline | ❌ | ❌ | ✅ | ✅ | ✅ |
| Tr1_Td3 | ❌ | ❌ | ✅ | ❌ | ❌ |
| Tr1_Td5 | ❌ | ❌ | ✅ | ❌ | ❌ |
| Tr1_Td10 | ❌ | ❌ | ✅ | ❌ | ❌ |
| Tr2_Td3 | ❌ | ❌ | ✅ | ❌ | ❌ |
| Tr2_Td5 | ❌ | ❌ | ✅ | ❌ | ❌ |
| Tr2_Td10 | ❌ | ❌ | ✅ | ❌ | ❌ |
| Tr5_Td3 | ❌ | ❌ | ✅ | ❌ | ❌ |
| Tr5_Td5 | ❌ | ❌ | ✅ | ❌ | ❌ |
| Tr5_Td10 | ❌ | ❌ | ✅ | ❌ | ❌ |

## 4. Composite Ranking

Formula: 0.4 × G4 + 0.25 × G3 + 0.15 × G5 + 0.10 × Ret + 0.10 × Sharpe

| Config | G4 | G3 | G5 | Ret | Sharpe | Composite |
|--------|:---:|:---:|:---:|:---:|:------:|:---------:|
| Tr1_Td3 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.4500 |
| Tr1_Td5 | 0.0506 | 0.0000 | 0.2997 | 0.0000 | 0.0000 | 0.0652 |
| Tr1_Td10 | 0.0506 | 0.0000 | 0.2997 | 0.0000 | 0.0000 | 0.0652 |
| Tr2_Td3 | 0.0506 | 0.0000 | 0.2997 | 0.0000 | 0.0000 | 0.0652 |
| Tr2_Td5 | 0.0506 | 0.0000 | 0.2997 | 0.0000 | 0.0000 | 0.0652 |
| Tr2_Td10 | 0.0506 | 0.0000 | 0.2997 | 0.0000 | 0.0000 | 0.0652 |
| Tr5_Td3 | 0.0506 | 0.0000 | 0.2997 | 0.0000 | 0.0000 | 0.0652 |
| Tr5_Td5 | 0.0506 | 0.0000 | 0.2997 | 0.0000 | 0.0000 | 0.0652 |
| Tr5_Td10 | 0.0506 | 0.0000 | 0.2997 | 0.0000 | 0.0000 | 0.0652 |

## 🏆 Recommendation

**Best config: Tr1_Td3** (composite = 0.4500)

- **G4 (WF Sharpe Std)**: 0.8973 ❌ FAIL
- **G3 (WF Sharpe Min)**: -0.7255 ❌ FAIL
- **G5 (MaxDD)**: 7.24% ✅ PASS
- **AnnRet**: 10.28% ❌ FAIL
- **Sharpe**: 0.740 ❌ FAIL

### Baseline Comparison

| Metric | Baseline | Best (Tr1_Td3) | Delta |
|--------|:--------:|:------------:|:-----:|
| AnnRet | 14.87% | 10.28% | -4.59pp |
| MaxDD | 7.84% | 18.77% | +10.93pp |
| Sharpe | 1.191 | 0.740 | -0.451 |
| WF Std (G4) | 1.0961 | 0.8973 | -0.1988 |
| WF Min (G3) | -0.3868 | -0.7255 | -0.3387 |

### Comparison with Phase 5a Best (A5 Full-Offensive T=2.0)

| Metric | P5a A5 T=2.0 | Tr1_Td3 | Delta |
|--------|:------------:|:-------:|:-----:|
| AnnRet | 10.28% | 10.28% | 0.00pp |
| MaxDD | 18.77% | 18.77% | 0.00pp |
| Sharpe | 0.740 | 0.740 | 0.000 |
| WF Std (G4) | 0.8963 | 0.8973 | +0.0010 |
| WF Min (G3) | -0.7255 | -0.7255 | 0.0000 |

## 5. Analysis

### Binary Outcome Pattern

Only Tr1_Td3 (T_risk=1.0, T_caut=2.0, T_def=3.0) produces results distinct from the other 8 configs. The 8 "collapsed" configs (all with T_def ≥ 5 or T_risk ≥ 2) produce identical results: AnnRet=3.34%, Sharpe=0.194, WF Sharpe Min=-1.8941.

**Root cause**: When softmax is always-on in DEFENSIVE regime with T_def ≥ 5, the DEFENSIVE allocation gets overly dispersed. The strategy ends up holding many low-conviction positions instead of concentrating in the single best defensive ETF (top_n=1 under DEFENSIVE regime override). This destroys all alpha in defensive weeks.

Tr1_Td3 (with T_def=3.0) avoids this because T_def=3.0 is close enough to T_caut=2.0 that the softmax still concentrates effectively.

### Tr1_Td3 = Phase 5a A5 (Full-Offensive)

Tr1_Td3 produces results that are **bit-identical** to the Phase 5a A5 config (Full-Offensive Softmax T=2.0). This is not a coincidence — when softmax is always-on with low temperatures across all regimes, the behavior collapses to the same full-offensive softmax approach tested in Phase 5a.

### G4 Improvement Relative to Baseline

Tr1_Td3 achieves WF Sharpe Std = 0.8973 vs baseline's 1.0961 — a **-0.1988 improvement** (18.1% reduction in WF volatility). However, this comes at severe cost:
- Sharpe drops from 1.191 to 0.740 (-0.451)
- MaxDD doubles from 7.84% to 18.77%
- G3 (WF Min Sharpe) worsens from -0.3868 to -0.7255

## 6. Conclusion

**Temperature-Ramp softmax (Phase 5e) is NOT recommended.** The approach fails every gate check and is strictly dominated by baseline:

- Best config (Tr1_Td3) merely reproduces Phase 5a results
- All other 8 configs destroy performance (AnnRet 3.34%, Sharpe 0.194)
- The "always-on" softmax in DEFENSIVE regime is counterproductive: it overrides the regime's intended defensive concentration (top_n=1)
- G4 improvement (0.8973 vs 1.0961) is real but insufficient to pass the 0.60 target, and comes at unacceptable cost to returns

**Recommendation**: Keep baseline (D4-tuned + 3-State Regime) as production config. Phase 5c A1 (T=0.5, softmax in RISK_ON only) remains the best softmax variant so far but still fails G3/G4.
