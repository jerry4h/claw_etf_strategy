# Strategy iteration directions

These issues are recorded for future strategy evolution, not to be fixed in the current round.

## S-P0: Walk-Forward failure pattern

**Finding (2026-07-22)**: WF actually achieves 77.8% (7/9) win rate, not 55.6% as previously reported.
The 2 failed windows are both during strong bull markets (2013-2015 Nasdaq ann=20%, 2023-2025 Nasdaq ann=33.5%).
The defense mechanism holds back performance in strong trends — this is BY DESIGN, not a bug.
The strategy sacrifices upside for downside protection, and both failed windows still had better Sharpe and lower MaxDD than EW.

**Iteration direction**: If higher WF win rate is desired, consider:
- Bull-market relaxation: reduce defense ratio during confirmed bull regimes (12w momentum > threshold)
- Regime-conditional defense: lower step_low/step_high during bull markets
- Accept the trade-off: current 77.8% WF win rate with 6.38% MaxDD is excellent risk-adjusted

## S-P0: Train/Test split result

**Finding (2026-07-22)**: Train (2013-2020) Sharpe=1.605, Test (2020-2026) Sharpe=1.400.
12.8% Sharpe degradation is mild. Annual return actually IMPROVED in test (16.07% vs 14.98%).
Both periods comfortably beat equal-weight. **No overfitting detected.**

## S-P1: FX exposure analysis

**Finding (2026-07-22)**: FX hedge cost bug fixed. With 2% annual hedge cost deducted:
- Sharpe drops from 1.514 to 1.464 (-3.3%)
- Annual return drops from 15.98% to 15.49% (-0.49pp)
- MaxDD increases from 7.33% to 7.50% (+0.17pp)

Even with 3% annual hedge cost, Sharpe remains 1.439. FX exposure is NOT a critical risk.

**Iteration direction**: For institutional deployment, consider:
- Decompose Nasdaq ETF returns into: S&P 500 return + USD/CNY change + QDII premium/discount
- Evaluate whether hedging 50% of FX exposure improves risk-adjusted returns
- Monitor QDII premium as a timing signal (high premium -> reduce Nasdaq allocation)

## S-P1: Crisis correlation convergence

**Status**: Not yet analyzed. The 5-ETF universe (Nasdaq, CSI500, Gold, Hongli, Bond) has low normal-period correlation, but crisis periods may see convergence.

**Iteration direction**:
- Calculate rolling 26-week correlation matrix
- Identify periods where correlation > 0.7 between any 2 offensive ETFs
- Consider risk-parity variant of inv-vol weighting that accounts for correlation
- Stress test: what happens if all offensive ETFs drop simultaneously?

## S-P2: Momentum window noise

**Status**: mom_window=4 with score_margin=0.02 provides adequate noise suppression.

**Iteration direction**: If whipsawing becomes an issue:
- Multi-period momentum composite: 0.5*mom4 + 0.3*mom8 + 0.2*mom12
- Adaptive momentum: use trend quality (D1) to weight short vs long momentum
- Hysteresis: require score difference > score_margin for BOTH entry AND exit

## S-P2: Interest rate sensitivity

**Status**: Not yet analyzed. Bond ETF (511010) in defense layer is sensitive to interest rate changes.

**Iteration direction**:
- Analyze defense layer performance during rate hike cycles (2022-2023)
- Consider adding duration-matched bond ETF or TIPS equivalent
- Test whether defense layer effectiveness degrades in rising rate environments

## S-P3: DefAlloc constants

**Status**: hl_ratio = clip(0.80 - 2.67*vol_hongli, 0, 0.80) has hardcoded constants derived from T=0.30.

**Iteration direction**: Extract to config as:
- hl_ratio_base: 0.80 (= 1 - T/0.375 approximately)
- hl_ratio_slope: 2.67 (= 0.80 / 0.30)
- This allows T to be parameterized without code changes
