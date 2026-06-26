# V2.5 P1 Redesign Evaluation Report

**Date**: 2026-06-16  
**Tester**: quant-tester  
**Task**: T13 — Evaluate P1 Redesign (market-state stop loss + weight cap fix)

---

## Executive Summary

**Verdict: P1 Redesign (v2.5) FAILS — all metrics worse than baseline.**

| Metric | v2.3 Baseline | v2.5 P1 All-On | Delta | Direction |
|--------|:------------:|:-------------:|:-----:|:---------:|
| Annual Return | **14.11%** | 12.55% | -1.56pp | [31mWORSE[0m |
| Max Drawdown | **7.42%** | 11.19% | +3.77pp | [31mWORSE[0m |
| Sharpe Ratio | **1.102** | 0.999 | -0.103 | [31mWORSE[0m |
| Calmar Ratio | **1.90** | 1.12 | -0.78 | [31mWORSE[0m |
| Annual Volatility | 10.19% | 9.82% | -0.37pp | — |
| Win Rate | 60.1% | 60.1% | 0.0pp | — |
| Defensive Weeks | 211 | 243 | +32 | [33mMORE DEFENSE[0m |
| Stop Loss Weeks | 0 | 15 | +15 | [33mMORE STOPS[0m |

Both P1 features individually also degrade performance. The redesign does NOT improve crisis-year returns or reduce drawdown; instead, it introduces over-defense in bull years and parallel stop-loss conflicts.

---

## 1. Baseline Confirmation: PASSED

v2.5 with all features OFF matches v2.3 exactly:

| Metric | v2.3 | v2.5 All-Off | Match |
|--------|:----:|:-----------:|:-----:|
| Annual Return | 14.11% | 14.11% | [32mPASS[0m |
| Max Drawdown | 7.42% | 7.42% | [32mPASS[0m |
| Sharpe | 1.102 | 1.102 | [32mPASS[0m |
| All year-by-year returns | — | — | [32mPASS[0m |

Year-by-year verification (both identical):

| Year | Return | Avg Defense | Stop Loss Wks |
|------|--------|:-----------:|:------------:|
| 2013 | -1.1% | 25% | 0 |
| 2014 | +23.9% | 25% | 0 |
| 2015 | +14.3% | 41% | 0 |
| 2016 | +8.9% | 27% | 0 |
| 2017 | +12.5% | 25% | 0 |
| 2018 | -2.2% | 27% | 0 |
| 2019 | +23.5% | 35% | 0 |
| 2020 | +20.1% | 76% | 0 |
| 2021 | +7.7% | 33% | 0 |
| 2022 | -3.7% | 40% | 0 |
| 2023 | +21.9% | 25% | 0 |
| 2024 | +27.1% | 41% | 0 |
| 2025 | +22.8% | 51% | 0 |
| 2026 | +5.7% | 25% | 0 |

---

## 2. P1 All-On vs Baseline: Year-by-Year

| Year | v2.3 Baseline | v2.5 P1 All-On | Delta | Type |
|------|:-----------:|:-------------:|:-----:|------|
| 2013 | -1.1% | -1.6% | -0.5pp | pre |
| 2014 | +23.9% | +25.3% | **+1.4pp** | BULL |
| 2015 | +14.3% | +14.3% | 0.0pp | CRISIS |
| 2016 | +8.9% | +5.3% | **-3.6pp** | NORMAL |
| 2017 | +12.5% | +12.2% | -0.3pp | BULL |
| 2018 | -2.2% | -4.0% | **-1.8pp** | CRISIS |
| 2019 | +23.5% | +21.0% | **-2.5pp** | BULL |
| 2020 | +20.1% | +16.6% | **-3.5pp** | BULL |
| 2021 | +7.7% | +7.0% | -0.7pp | NORMAL |
| 2022 | -3.7% | -3.3% | **+0.4pp** | CRISIS |
| 2023 | +21.9% | +19.4% | **-2.5pp** | BULL |
| 2024 | +27.1% | +27.2% | +0.1pp | BULL |
| 2025 | +22.8% | +17.9% | **-4.9pp** | CRISIS |
| 2026 | +5.7% | +4.8% | -0.9pp | — |

**Crisis years analysis (target: reduce drawdown):**
- 2015: No change (+14.3% both) — stateful stop loss did NOT help, but 6 weeks of original 8% stop loss fired
- 2018: Got WORSE (-4.0% vs -2.2%) — stateful stop loss over-defended
- 2022: Slight improvement (-3.3% vs -3.7%, +0.4pp) — only marginal

**Bull years analysis (target: preserve upside):**
- 2019: Lost 2.5pp — over-defense from weight cap
- 2020: Lost 3.5pp — over-defense from weight cap
- 2023: Lost 2.5pp — over-defense from weight cap
- 2025: Lost 4.9pp — major over-defense

**Normal years:**
- 2016: Lost 3.6pp — worst single-year degradation; 9 stop loss weeks triggered

---

## 3. Individual Feature Impact

| Config | Ann Return | Max DD | Sharpe | Def Wks | Stop Wks | vs Baseline |
|--------|:---------:|:------:|:------:|:-------:|:--------:|:-----------:|
| v2.3 Baseline | **14.11%** | **7.42%** | **1.102** | 211 | 0 | — |
| Stop Loss Only | 13.55% | 10.24% | 1.049 | 236 | 6 | -0.56pp |
| Weight Cap Only | 12.79% | 11.32% | 1.016 | 221 | 15 | -1.32pp |
| P1 All-On | 12.55% | 11.19% | 0.999 | 243 | 15 | -1.56pp |

**Key insight**: The weight cap (0.30) causes MORE damage than the stateful stop loss:
- Weight cap alone: -1.32pp return, +3.90pp max DD
- Stop loss alone: -0.56pp return, +2.82pp max DD
- Combined: -1.56pp return, +3.77pp max DD (worse than either alone)

---

## 4. Market State Analysis

### Overall Distribution (649 weeks)

| State | Weeks | Pct | Description |
|-------|:----:|:---:|-------------|
| BULL | 242 | 37.3% | Low vol, strong momentum, shallow drawdown |
| NORMAL | 188 | 29.0% | Moderate conditions |
| CORRECTION | 78 | 12.0% | Elevated vol or negative momentum |
| CRISIS | 141 | 21.7% | High vol, deep drawdown, weak momentum |

**CRISIS at 21.7% is concerning** — the strategy classifies over 1/5 of all weeks as crisis. This is too aggressive and drives over-defense.

### Annual State Distribution

| Year | BULL | NORMAL | CORR | CRISIS | Crisis % | Correct? |
|------|:----:|:------:|:----:|:------:|:--------:|----------|
| 2013 | 7 | 4 | 0 | 0 | 0% | — (partial) |
| 2014 | 21 | 10 | 15 | 6 | 11.5% | Partially |
| 2015 | 1 | 16 | 2 | **33** | **63.5%** | [32mYES[0m |
| 2016 | 23 | 19 | 4 | 6 | 11.5% | OK |
| 2017 | 20 | 14 | 18 | 0 | 0% | [32mYES[0m |
| 2018 | 6 | 26 | 3 | **18** | **34.0%** | Partially |
| 2019 | 18 | 12 | 10 | 11 | 21.6% | [31mTOO HIGH[0m |
| 2020 | 23 | 5 | 4 | **20** | **38.5%** | [32mYES[0m |
| 2021 | 23 | 27 | 0 | 2 | 3.8% | OK |
| 2022 | 5 | 23 | 14 | 10 | 19.2% | Partially |
| 2023 | **32** | 19 | 1 | 0 | 0% | [32mYES[0m |
| 2024 | 29 | 9 | 2 | 12 | 23.1% | [31mTOO HIGH[0m |
| 2025 | 27 | 2 | 1 | **21** | **41.2%** | [31mWRONG[0m |
| 2026 | 7 | 2 | 4 | 2 | 13.3% | OK |

**False crisis signals**: 2019 (21.6%), 2024 (23.1%), 2025 (41.2%) — these are bull/normal years misclassified as crisis, causing unnecessary defense.

### State Transition Matrix

| From \ To | BULL | NORMAL | CORR | CRISIS |
|-----------|:----:|:------:|:----:|:------:|
| **BULL** | 205 | 19 | 7 | 11 |
| **NORMAL** | 19 | 127 | 14 | 28 |
| **CORRECTION** | 6 | 18 | 48 | 5 |
| **CRISIS** | 11 | 24 | 9 | 97 |

Key observations:
- BULL→BULL: 84.7% persistence — states are relatively sticky
- CRISIS→CRISIS: 68.8% persistence — but 31.2% transition out
- NORMAL→CRISIS: 14.9% — the most common "downward" transition
- CRISIS→BULL: 11, NORMAL→BULL: 19 — transitions back to bull exist

---

## 5. Root Cause Analysis

### Issue #1: Parallel Stop Loss Conflict (CRITICAL)

The stateful stop loss and the original 8% stop loss run in PARALLEL. When `stateful_stop_loss=true`:
- The stateful logic sets a defense ratio based on market state
- But the original 8% stop loss (line 227-236 of backtest.py) STILL fires independently
- In 2015-2016, the original stop loss triggered 15 weeks of 95% defense, overriding the stateful logic

**Code path**: `backtest.py` line 163-189 (stateful) → then line 227-236 (original stop loss) fires unconditionally when `tiered_stop_loss=false`.

### Issue #2: Weight Cap Too Restrictive (HIGH)

`max_single_alloc: 0.30` is too aggressive:
- Baseline uncapped: 37.5% per offensive ETF (at 25% defense)
- Capped: 30% max per ETF, excess → defense
- During bull markets, this forces extra defense, reducing upside capture
- The strategy loses 3-5pp annually in bull years

### Issue #3: CRISIS State Over-Classification (MEDIUM)

The 3-signal voting mechanism classifies 21.7% of weeks as CRISIS:
- Volatility signal dominates: the `ms_high_vol_pct: 0.67` threshold means any week above 67th percentile vol is flagged
- Momentum signal contributes: `ms_crisis_mom: -0.12` (12-week return < -12%)
- The combined effect inflates crisis detection

### Issue #4: No Drawdown Reduction (CRITICAL)

The core promise of P1 was to reduce max drawdown. Instead, max DD increased from 7.42% to 11.19%.
- The weight cap shifted allocation toward defense, but didn't prevent the drawdown events
- The original 8% stop loss (15 weeks in 2015-2016) created sudden jumps to 95% defense AFTER the damage was done
- The stateful stop loss's tiered approach (L1→L2 with recovery) is reactive, not predictive

---

## 6. Recommendations

### Immediate (v2.5 → v2.6 fixes)

1. **Disable original stop loss when stateful is active** (P0 fix)
   - Add guard: `if config.stateful_stop_loss: skip original stop_loss block`
   - This would eliminate the parallel double-defense issue

2. **Increase weight cap to 0.40** (P1 fix)
   - 0.30 is too restrictive; 0.40 preserves more upside while still preventing extreme concentration
   - Test: 0.35, 0.40, 0.45 to find optimal balance

3. **Tighten CRISIS detection thresholds** (P1 fix)
   - Raise `ms_high_vol_pct` from 0.67 → 0.80 (requires higher vol percentile for crisis)
   - Raise `ms_crisis_mom` from -0.12 → -0.15 (requires deeper momentum drop)
   - Target: reduce CRISIS weeks from 21.7% → ~12-15%

### Medium-term (v2.7+)

4. **Adaptive weight cap tied to market state**
   - Enable `dynamic_weight_cap: true` with the stateful stop loss
   - BULL: 0.45-0.50 cap, NORMAL: 0.35, CORRECTION: 0.30, CRISIS: 0.25
   - This would allow more upside during bull markets while still capping downside

5. **Stateful stop loss recovery tuning**
   - BULL recovery of 1 week is too short — tested but 2015 had 6 weeks of original stop loss anyway
   - Consider: don't enter recovery until drawdown is below L1 threshold

6. **Add over-defense detection**
   - Track defense ratio × market state to detect when defense exceeds what the state warrants
   - Alert if defense > 0.60 during BULL state or > 0.80 during NORMAL

---

## 7. Verdict

| Dimension | Score | Notes |
|-----------|:-----:|-------|
| Baseline parity | [32mPASS[0m | v2.5 all-off = v2.3 exactly |
| Return improvement | [31mFAIL[0m | -1.56pp vs baseline |
| Drawdown reduction | [31mFAIL[0m | +3.77pp vs baseline |
| Crisis protection | [33mMIXED[0m | 2022 improved +0.4pp; 2018 worsened -1.8pp |
| Bull market preservation | [31mFAIL[0m | Lost 2.5-4.9pp in bull years |
| State classification | [33mPARTIAL[0m | 2015, 2020 correctly flagged; 2019, 2024, 2025 over-classified |
| Feature independence | [31mFAIL[0m | Stop losses conflict; cap is always active |

**Overall: NOT RECOMMENDED for deployment.**

P1 redesign needs fundamental fixes before re-evaluation. The 3 immediate fixes (disable parallel stop loss, increase cap to 0.40, tighten crisis thresholds) should be implemented and tested as v2.6.

---

## Appendix: Test Configurations

All test configs created under `config/`:
- `strategy_v2_5_p1_all_on.yaml` — both features enabled
- `strategy_v2_5_stop_loss_only.yaml` — only market state stop loss
- `strategy_v2_5_weight_cap_only.yaml` — only 30% weight cap

All output reports under `output/`:
- `report_baseline_v23.md` — v2.3 baseline
- `report_v25_all_off.md` — v2.5 all-off (baseline confirmation)
- `report_v25_p1_all_on.md` — P1 all-on
- `report_v25_stop_loss_only.md` — stop loss only
- `report_v25_weight_cap_only.md` — weight cap only

Market state history: `/tmp/market_state_history.csv`
