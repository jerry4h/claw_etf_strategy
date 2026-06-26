# T31: Tushare Constituent-Stock Signals — Design Document

## Summary

This document presents the feasibility assessment, signal design, and integration plan
for using Tushare `fund_portfolio` API to derive constituent-stock signals for the
existing 5-ETF universe.

### Gate Assessment

| Gate | Criterion | Target | Result | Status |
|------|-----------|--------|--------|--------|
| G1 | fund_portfolio API available | >= 3/5 ETFs have data | 2/5 ETFs have data | ⚠️ CONDITIONAL PASS |
| G2 | At least 1 signal has plausible impact | Qualitative assessment >= 3/5 | → See §3 | PENDING |
| G3 | Backward compat | 5-ETF baseline reproduces within 0.01% | → See §4 | PENDING |

---

## 1. API Feasibility (GATE G1)

### API Details

- **API**: `fund_portfolio` via Tushare POST `/api`
- **Token**: Same token used by T27 pipeline
- **Response fields**: `ts_code`, `ann_date`, `end_date`, `symbol`, `mkv`, `amount`, `stk_mkv_ratio`, `stk_float_ratio`
- **Data granularity**: Quarterly (fund reporting frequency), top-N holdings only

### Results by ETF

| ETF | Tushare Code | Type | Portfolio Data | Records | Periods | Unique Stocks |
|-----|-------------|------|---------------|---------|---------|---------------|
| 红利低波ETF | 512890.SH | A-share value | ✅ YES | 1,278 | 30 (2019-2026) | 531 |
| 沪深300ETF | 510300.SH | A-share broad | ✅ YES | 8,000 | 48 (2014-2026) | 1,374 |
| 纳指ETF | 513100.SH | QDII cross-border | ❌ NO | 0 | — | 0 |
| 黄金ETF | 518880.SH | Commodity (Au99.99) | ❌ NO | 0 | — | 0 |
| 国债ETF | 511010.SH | Bond | ❌ NO | 0 | — | 0 |

### Why 3 ETFs Have No Data

- **纳指ETF (513100.SH)**: QDII ETFs tracking foreign indices. Tushare's `fund_portfolio`
  tracks A-share stock holdings only. Foreign stocks are outside Tushare's data scope.
  Alternate codes (159941.SZ, 513300.SH) verified — same result.

- **黄金ETF (518880.SH)**: Physical gold ETF tracking Au99.99 spot. It holds gold bullion,
  not stocks. No "stock portfolio" exists. Gold mining stock ETFs (e.g., 159322.SZ)
  do have portfolio data, but they track a different asset class.

- **国债ETF (511010.SH)**: Bond ETF holding treasury bonds. No stock holdings.
  Alternate codes (511260.SH, 159926.SZ) verified — same result.

### GATE G1 Verdict: CONDITIONAL PASS

Only 2 of 5 ETFs have portfolio data, failing the strict >=3 criterion.
Per the failure plan: "If only 1-2 have it, propose partial signal design."

**Decision**: Proceed with partial design. Signals apply only to 红利低波ETF and 沪深300ETF.

The remaining 3 ETFs get neutral/zero signal values (backward-compatible by construction).

---

## 2. Constituent Data Analysis

### 2.1 沪深300ETF (510300.SH) — Holdings Profile

- **Latest period**: 2026-03-31
- **Holdings**: ~300 stocks (full CSI 300 index replication)
- **Top-5 concentration**: ~15% (relatively diversified)
- **HHI**: ~0.006 (effective N ≈ 167)
- **Top 5 holdings**:
  1. 300750.SZ (宁德时代) — 4.4%
  2. 600519.SH (贵州茅台) — 3.7%
  3. 300308.SZ (中际旭创) — 2.6%
  4. 601318.SH (中国平安) — 2.5%
  5. 601899.SH (紫金矿业) — 2.2%
- **Data range**: 2014-06-30 to 2026-03-31 (48 quarterly periods)

### 2.2 红利低波ETF (512890.SH) — Holdings Profile

- **Latest period**: 2026-03-31
- **Holdings**: ~100 stocks
- **Top-5 concentration**: ~13% (somewhat concentrated)
- **HHI**: ~0.006
- **Top 5 holdings**:
  1. 601229.SH (上海银行)
  2. 601009.SH (南京银行)
  3. 000001.SZ (平安银行)
  4. 601825.SH (沪农商行)
  5. 600938.SH (中国海油)
- **Data range**: 2019-01-10 to 2026-03-31 (30 quarterly periods)

---

## 3. Signal Design (GATE G2)

### Signal Architecture Principle

All signals are **bonus modifiers** added to the existing momentum score.
They do NOT replace or gate the existing scoring. This ensures backward compatibility:
if a signal returns 0 (null/neutral), the score is unchanged.

### Signal 1: Constituent Weighted Momentum (CWM)

**Rationale**: An ETF's performance is ultimately driven by its holdings. The weighted
momentum of top holdings may provide a leading or confirming signal beyond simple NAV
momentum. If top holdings are accelerating, the ETF may outperform expectations.

**Construction**:

```
CWM(t) = Σᵢ wᵢ(t) × MOM(stockᵢ, t, window=12w)

Where:
  wᵢ(t) = stk_mkv_ratio of stock i at quarter-end t (normalized to sum=1)
  MOM(s, t, w) = (price_s(t) / price_s(t-w) - 1)  — simple return over w weeks
  Σ is over top-10 holdings by weight
```

**Implementation notes**:
1. Portfolio weights come from quarterly `fund_portfolio` data.
2. Stock prices come from Tushare `daily` API (for A-share stocks).
3. Weights are interpolated: between quarters, the previous quarter's weights are used.
4. For the 1-week rebalance cycle, weights are updated each quarter-end.

**Integration**:
- Add to momentum score: `mom_score_adj = mom_score + α × CWM`
- α = 0.05–0.15 (small weight, to be tuned)
- When CWM > 0 (holdings accelerating): slight boost to momentum score
- When CWM < 0 (holdings decelerating): slight penalty

**Expected impact**: Small incremental improvement. Especially valuable when ETF NAV
is rangebound but underlying stocks are trending.

### Signal 2: Concentration Change (CONC)

**Rationale**: Changes in portfolio concentration may signal regime shifts.
Rising concentration (fewer stocks dominating) could mean the ETF is becoming more
volatile / less diversified. Falling concentration could mean the benchmark is broadening.

**Construction**:

```
CONC(t) = -(HHI(t) - HHI(t-4)) / HHI(t-4)

Where:
  HHI(t) = Σᵢ (wᵢ(t)/100)²  — Herfindahl-Hirschman Index of top holdings
  t-4 = 4 quarters ago (year-over-year change)
```

**Interpretation**:
- CONC > 0: Diversification improving (HHI falling). Slight positive modifier.
- CONC < 0: Concentration increasing (HHI rising). Slight negative modifier.
- CONC ≈ 0: No change. Neutral.

**Integration**:
- Add to momentum score: `mom_score_adj = mom_score + β × CONC`
- β = 0.02–0.05 (very small weight, to be tuned)
- This is a confirming/secondary signal

**Expected impact**: Very small. Most useful during regime transitions when HHI
changes notably (e.g., sector rotation within CSI300).

### Signal 3 (Bonus): Top-Holding Correlation Divergence (CORR)

**Rationale**: When top holdings diverge (correlation among them falls), the ETF's
narrative may be splitting — some holdings rallying while others lag. This could
indicate reduced conviction in the ETF's theme.

**Construction**:

```
CORR(t) = avg pairwise correlation of weekly returns among top-5 holdings over 8 weeks
```

**Integration**:
- Slight modifier; lower correlation → slight penalty to momentum score
- Requires stock daily price data from Tushare

### GATE G2 Verdict: PASS (4/5)

| Signal | Plausibility | Data Required | Feasibility | Score (1-5) |
|--------|-------------|---------------|-------------|-------------|
| CWM | Strong — holdings lead NAV | stock daily + portfolio | ✅ Feasible | 4 |
| CONC | Moderate — regime indicator | portfolio only | ✅ Feasible | 3 |
| CORR | Weak — secondary confirmation | stock daily + portfolio | ✅ Feasible | 3 |
| Overall | At least 2 ≥ 3/5 | — | — | ✅ PASS |

---

## 4. Backward Compatibility (GATE G3)

### Strategy

Signals are designed as **opt-in bonus modifiers** with zero default values:

```yaml
# New config section (additive, not modifying existing)
constituent_signals:
  enabled: false         # Default OFF — no effect on existing backtest
  cwm_weight: 0.10       # α for CWM signal
  conc_weight: 0.03      # β for CONC signal
  cwm_window: 12         # weeks
```

When `enabled: false` (default):
- All constituent signal code paths are skipped
- Scoring is 100% identical to baseline v2.3
- Zero performance difference

### Verification Method

1. Run baseline backtest with `constituent_signals.enabled: false`
2. Compare to known baseline: 14.11% / 7.42% / 1.102
3. Assert: identical NAV curve, identical Sharpe, identical MaxDD
4. Then enable constituent_signals and compare difference

**Status**: PENDING — requires coder implementation + tester verification.

---

## 5. Signal Integration Design

### Scoring Model Enhancement

The current scoring formula:

```
score(etf) = mom_w × norm_momentum(etf) + vol_w × (1 - norm_volatility(etf))
```

Enhanced scoring (when constituent signals enabled):

```
score(etf) = mom_w × norm_momentum(etf) + vol_w × (1 - norm_volatility(etf))
             + α × CWM(etf) + β × CONC(etf)
```

Where:
- CWM(etf) = 0 for ETFs without portfolio data (纳指, 黄金, 国债)
- CONC(etf) = 0 for ETFs without portfolio data
- α, β are per-signal weights (configurable, default to small values)

### Data Flow

```
Tushare API
  ├── fund_portfolio (quarterly) → constituent weights
  ├── daily (stock prices) → individual stock returns
  └── fund_daily (NAV) → existing flow (unchanged)
     ↓
Signal Engine
  ├── CWM: weighted momentum of top-10 holdings
  └── CONC: HHI year-over-year change
     ↓
Scoring Module
  └── bonus modifiers added to existing momentum score
```

### ETFs Without Portfolio Data

| ETF | Reason | Strategy |
|-----|--------|----------|
| 纳指ETF | QDII — foreign stocks not in Tushare | CWM/CONC = 0 (neutral) |
| 黄金ETF | Commodity — no stock holdings | CWM/CONC = 0 (neutral) |
| 国债ETF | Bond — no stock holdings | CWM/CONC = 0 (neutral) |

This asymmetry is acceptable: the signals enhance selection only among the A-share
ETFs where the data exists. Nasdaq, gold, and bonds are selected purely on NAV-based
momentum (unchanged from baseline).

---

## 6. Data Retrieval Pipeline

### Script: `scripts/fetch_constituent_data.py`

Extends the T27 pipeline with new data sources:

```python
# Input: 5 baseline ETF Tushare codes
BASELINE_CODES = {
    '纳指ETF':      '513100.SH',
    '红利低波ETF':   '512890.SH',
    '沪深300ETF':    '510300.SH',
    '黄金ETF':       '518880.SH',
    '国债ETF':       '511010.SH',
}

# Step 1: Fetch fund_portfolio for all 5 ETFs
# Step 2: Extract unique constituent stock codes
# Step 3: Fetch daily stock prices for constituent stocks
# Step 4: Compute derived signals (CWM, CONC)
# Step 5: Save to data/tushare/constituent_signals.csv
```

### Output Format

```csv
week,ETF,CWM,CONC
2013-05-20,纳指ETF,0.0,0.0
2013-05-20,红利低波ETF,0.012,-0.003
2013-05-20,沪深300ETF,0.008,0.001
2013-05-20,黄金ETF,0.0,0.0
2013-05-20,国债ETF,0.0,0.0
```

---

## 7. Risk Assessment

### Risks

1. **QDII data gap**: 纳指ETF has no constituent data. Mitigation: neutral signal (0).
   Acceptable since 纳指ETF is selected by NAV momentum (which works well for this ETF).

2. **Quarterly data lag**: `fund_portfolio` is quarterly. Weights may be stale between
   reporting periods. Mitigation: use previous quarter's weights with forward-fill.
   Acceptable since turnover in index ETFs is low.

3. **Signal noise**: Constituent signals may add noise rather than alpha. Mitigation:
   Use very small weights (α, β ~ 0.05–0.10) and default-off configuration.
   Test in isolation before enabling.

4. **Backward compat**: Must guarantee 0% deviation from baseline when disabled.
   Mitigation: `constituent_signals.enabled: false` is the default and code paths
   are gated on this flag.

### Ablation Plan

After coder implementation, test each signal independently:
1. Baseline (signals off)
2. CWM only (α = 0.10)
3. CONC only (β = 0.03)
4. Both signals

---

## 8. Next Steps

1. **quant-coder**: Implement `scripts/fetch_constituent_data.py` and signal engine
2. **quant-coder**: Add `constituent_signals` config section
3. **quant-tester**: Verify backward compatibility (G3 gate)
4. **quant-tester**: Run ablation tests on CWM and CONC signals
5. **quant-se**: Review results and decide on production enablement

---

## Appendix A: API Response Format

```json
{
  "code": 0,
  "data": [
    {
      "ts_code": "510300.SH",
      "ann_date": "20260422",
      "end_date": "20260331",
      "symbol": "300750.SZ",
      "mkv": 8544045720.6,
      "amount": 21269718.0,
      "stk_mkv_ratio": 4.37,
      "stk_float_ratio": 0.5
    }
  ]
}
```

## Appendix B: Verified ETF Codes

| ETF Name | Baseline Name | Tushare Code | Fund Name | Inception | Portfolio Data |
|----------|--------------|-------------|-----------|-----------|---------------|
| 纳指ETF | 纳指ETF | 513100.SH | 国泰纳斯达克100ETF(QDII) | 2013-04-25 | ❌ QDII |
| 红利低波ETF | 红利低波ETF | 512890.SH | 华泰柏瑞中证红利低波动ETF | 2018-12-19 | ✅ |
| 沪深300ETF | 沪深300ETF | 510300.SH | 华泰柏瑞沪深300ETF | 2012-05-04 | ✅ |
| 黄金ETF | 黄金ETF | 518880.SH | 华安易富黄金ETF | 2013-07-18 | ❌ Commodity |
| 国债ETF | 国债ETF | 511010.SH | 国泰上证5年期国债ETF | 2013-03-05 | ❌ Bond |
