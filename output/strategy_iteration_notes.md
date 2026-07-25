# Strategy iteration directions

These issues are recorded for future strategy evolution, not to be fixed in the current round.

## S-P0: Walk-Forward failure pattern

**Verified (2026-07-25)**: WF 胜率为 **55.6% (5/9)**（`compute_benchmark_relative_win_rate` 可复现）。
此前文档中的 77.8% (7/9) 来自交互式分析会话，无可复现代码支撑，经 git 溯源已确认不可复现。

4 个失败窗口（W0/W2/W5/W7）共性：
1. 纳指 vol 低于 step_low(15%)，防御层无降波优势（<step_low 占比 35-77%）
2. 评分公式 `score = mom - 1.1*vol` 的黄金偏好：黄金因低波（10-12%）长期占据 top-2，策略错过中证500 上涨窗口（W0: 中证500+20.5% vs 黄金0%，W5: 中证500+12.1% vs 黄金-2.8%）

5 个胜利窗口共性：等权基准波动率高（策略防御层降波效果显著），Sharpe 优势来自风控。

**Iteration direction**: If higher WF win rate is desired, consider:
- 降低 vol 惩罚系数（vol_w 从 1.10 降至 ~0.80），减轻黄金偏好
- 给中证500 加最小评分加成（hysteresis），避免等动量时被低波资产挤出
- Bull-market relaxation: 纳指 vol < step_low 时降低基准防御比
- Accept the trade-off: current 55.6% WF with 6.97% MaxDD — Sharpe 1.61 和 DSR 1.0 说明风险调整收益仍优秀

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

**Status**: mom_window=6 (adopted in v3.0 final) with score_margin=0.02 provides adequate noise suppression.

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

---

## Updated 2026-07-22: Round 2 Analysis Results

### S-P1: ETF Universe validation [RESOLVED]

Compared current universe (with 中证500ETF) vs synthetic old universe (沪深300 proxy):
- Sharpe: 1.522 vs 1.476 (+0.046)
- Annual return: 15.97% vs 15.49% (+0.48pp)
- MaxDD: 6.38% vs 6.50% (-0.12pp)

**Conclusion**: Universe change is validated. Improvement is modest but consistent.

### S-P1: Crisis correlation convergence [ANALYZED]

Rolling 26-week correlation between offensive ETF pairs:
- Convergence (corr > 0.6) occurs in only 28/675 weeks (4.1%)
- Crisis windows show elevated correlations (NASDAQ-ZZ500 peaks at 0.64-0.65)
- **Strategy is highly vulnerable during convergence**: Sharpe drops from 1.610 to -0.374

**Iteration direction**: Consider risk-parity variant or correlation-adjusted inv-vol weighting.

### S-P2: Momentum window noise [ANALYZED]

- Baseline (mom_window=4): 79 switches over 664 weeks (1 per 8.4 weeks)
- score_margin=0.02 prevents 35 switches (30.7% noise reduction)
- mom_window=6 is actually the sweet spot: Sharpe 1.555 (vs 1.522 for window=4)
- mom_window=8 reduces switches by only 3 more but hurts Sharpe to 1.387

**Iteration direction**: ✅ Adopted — mom_window=6 is now the production config (Sharpe 1.555 vs 1.522 for window=4).

### S-P2: Interest rate sensitivity [ANALYZED]

During 2022-2023 Fed rate hikes:
- Strategy: +10.70% total, Sharpe 0.361
- Equal-weight: +7.34% total, Sharpe 0.186
- Strategy beat EW by +3.37% with lower MaxDD (4.84% vs 7.06%)

Surprise: 国债ETF had +5.44% returns during rate hikes (PBoC easing > Fed tightening).
Defense layer (avg 38.1% allocation) meaningfully protected portfolio.

**Conclusion**: No interest rate sensitivity issue detected for this strategy.
