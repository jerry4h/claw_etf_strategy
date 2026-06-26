# T34: Robustness-First Redesign — Regime-Aware Strategy Framework

## Summary

This document presents a root-cause analysis of why the current momentum+volatility
scoring framework has inherent regime dependency (WF Sharpe std=1.208), and proposes
three redesign directions with at least two having full signal→integration designs.
All directions use **Tushare external data** — independent of the 5-ETF momentum
scores — to build a regime-aware strategy that can distinguish "normal bull market"
from "bubble final stage" and proactively adjust risk exposure.

### Gate Assessment

| Gate | Criterion | Target | Result | Status |
|------|-----------|--------|--------|--------|
| G1 | ≥1 direction has feasible data source | Tushare API returns ≥3 sequences | 9/10 APIs verified ✅ | PASS |
| G2 | Design completeness | Signal→integration chain complete for ≥2 directions | A, B, C all complete ✅ | PASS |
| G3 | Backward compatibility | Design does not break 5-ETF baseline | Opt-in flags, zero defaults ✅ | PASS |

### Recommendation Priority

| Priority | Direction | Expected Impact | Implementation Cost | Verdict |
|----------|-----------|----------------|---------------------|---------|
| **P0** | Direction A: Regime Classifier | HIGH — structural fix | MEDIUM (data pipeline + 5-state FSM) | DO FIRST |
| **P1** | Direction B: Adaptive Parameters | HIGH — multiplicative with A | LOW (config table + interpolation) | DO SECOND |
| **P2** | Direction C: New Tushare Factors | MEDIUM — incremental | LOW (3 additive signals) | DO THIRD |

---

## 1. Root Cause: Why Current Framework Has Inherent Regime Dependency

### 1.1 The Core Flaw

The current scoring formula:

```
score(etf) = mom_w × norm_momentum(etf) + vol_w × (1 - norm_volatility(etf))
```

Both `norm_momentum` and `norm_volatility` are **ranked within the 5-ETF universe**.
This means the scoring is a **relative ranking system**, not an absolute assessment.

In a raging bull market where ALL 5 ETFs have positive momentum:
- The top-2 ETFs get high scores → strategy allocates heavily → good returns
- But the system can't tell if this is a "healthy bull" or "bubble top"
- When the bubble pops, **all 5 ETFs crash together** → defense layer kicks in too late
- Recovery from drawdown takes months (the empirical 7.58% max DD for D4-tuned)

In a bear/choppy market where ALL 5 ETFs have negative or flat momentum:
- Scores are compressed → selection is essentially random
- Strategy struggles to differentiate → Sharpe drops to 0.1-0.6 (pre-2022 walk-forward)
- The defense layer provides some cushion but doesn't help selection

### 1.2 Empirical Evidence from Walk-Forward

The 9-window walk-forward decomposition (robustness_results.json) reveals:

```
Window 0 (2016-17): Sharpe 0.167  — poor
Window 1 (2017-18): Sharpe 0.105  — poor
Window 2 (2018-19): Sharpe 0.597  — mediocre
Window 3 (2019-20): Sharpe 1.120  — decent (bull market)
Window 4 (2020-21): Sharpe 0.495  — poor
Window 5 (2021-22): Sharpe -0.387 — NEGATIVE
Window 6 (2022-23): Sharpe 2.597  — excellent (strong bull)
Window 7 (2023-24): Sharpe 3.533  — excellent (strong bull)
Window 8 (2024-25): Sharpe 1.622  — good

std_dev = 1.208, mean = 1.094
```

The 5-10x Sharpe swing between 2016-21 (0.1-1.1) and 2022-24 (1.6-3.5) is the
structural problem. This is NOT parameter overfitting — it's a **regime identification
gap**. The strategy performed well only when the macro regime favored momentum
strategies (post-2022 Chinese equity rally), and poorly when it didn't.

### 1.3 Parameter Cliff Confirmation

Sensitivity analysis confirms the framework is brittle:

| Parameter | ΔSharpe (-10%) | ΔSharpe (+10%) |
|-----------|---------------|---------------|
| vol_w | **-0.201** | -0.110 |
| top_n=1 | **-0.350** | — |
| top_n=3 | — | **-0.460** |
| mom_w | -0.112 | -0.196 |

The DOUBLE CLIFF on top_n (both -1 and +1 are catastrophic) shows the scoring
function has a narrow sweet spot that's regime-dependent.

### 1.4 The Fundamental Solution

The framework needs an **external regime signal** — something that looks at the
broader market environment (CSI300 index trend, macro liquidity, market breadth)
and modulates the strategy's behavior BEFORE losses accumulate.

This is different from the failed D1 (dynamic weighting) because D1 only looked
at ETF-internal trend quality — it didn't know whether the market as a whole was
in a bull, bear, or bubble phase.

---

## 2. Direction A: Tushare-Based Market Regime Classifier (P0)

### 2.1 Concept

Build a **5-state market regime classifier** using external Tushare data that is
entirely independent of the 5-ETF momentum scores. The classifier output is a
discrete regime label that modulates strategy parameters.

### 2.2 Data Sources (Tushare API)

| Data Source | Tushare API | Fields | Frequency | Range Available |
|-------------|-------------|--------|-----------|-----------------|
| CSI300 index | `index_daily` | close, vol, amount | Daily | 2013-2026 (3253 records) ✅ |
| SH Composite | `index_daily` | close | Daily | 2013-2026 (3253 records) ✅ |
| CSI500 index | `index_daily` | close | Daily | 2013-2026 (3253 records) ✅ |
| Money supply | `cn_m` | m0, m1, m2 | Monthly | 1978-2026 (580 records) ✅ |
| CPI | `cn_cpi` | cpi | Monthly | 1951-2026 (509 records) ✅ |
| Market breadth | `stk_limit` | up_limit, down_limit | Daily | 7631 records ✅ |
| SW industries | `index_classify` | index_code, name | Static | 31 L1 industries ✅ |

All APIs verified working. Token reused from T27/T31 pipelines.

**Note**: SW industry index daily data (`801010.SI` etc.) returned 0 records in
this Tushare mirror. The classifier uses broad market indices (CSI300/500/SH)
instead, which are fully available. For sector rotation signals (Direction C),
we propose alternative approaches.

### 2.3 Signal Construction: 5-State Regime Classifier

The classifier combines 4 sub-signals into a 5-state regime label.

#### Sub-Signal 1: Trend Regime (TREND)

```
TREND(t) = classification of CSI300 26-week MA slope and position

Compute:
  MA_26 = SMA(CSI300.close, 26 weeks)
  MA_4  = SMA(CSI300.close, 4 weeks)
  slope = (MA_4 - MA_26) / MA_26  — normalized slope

States:
  BULL    = slope > 0.02 AND close > MA_26    → strong uptrend
  WEAK    = slope > 0 AND close > MA_26       → mild uptrend
  CHOPPY  = abs(slope) <= 0.01                → sideways
  BEAR    = slope < 0 AND close < MA_26       → downtrend
```

**Data needed**: CSI300 daily close → resampled to weekly. Available since 2013.
**Rationale**: The 26-week MA (≈ half-year) captures the medium-term market
direction. 4-week vs 26-week slope captures acceleration/deceleration.

#### Sub-Signal 2: Liquidity Environment (LIQUID)

```
LIQUID(t) = classification of M1/M2 monetary conditions

Compute:
  M1_YoY  = (M1(t) / M1(t-12) - 1)  — M1 year-over-year growth
  M2_YoY  = (M2(t) / M2(t-12) - 1)  — M2 year-over-year growth
  M1M2_gap = M1_YoY - M2_YoY         — "剪刀差" (liquidity preference)

States:
  LOOSE    = M1_YoY > 0.05 AND M1M2_gap > -0.02  → loose monetary
  NEUTRAL  = abs(M1_YoY) <= 0.05                   → neutral
  TIGHT    = M1_YoY < -0.02                         → tight monetary
```

**Data needed**: Monthly M1, M2 from `cn_m` API. Available 1978-2026.
**Rationale**: M1 growth is a leading indicator for equity markets. The M1-M2
gap (剪刀差) reflects whether money is flowing into demand deposits (risk-on) or
time deposits (risk-off). This is a well-established signal in Chinese macro
analysis.

#### Sub-Signal 3: Market Breadth (BREADTH)

```
BREADTH(t) = classification based on limit-up/down ratio

Compute:
  UpPct(t)   = avg(up_limit) over past 4 weeks   — from stk_limit
  DownPct(t) = avg(down_limit) over past 4 weeks  — from stk_limit
  ratio      = UpPct / max(DownPct, 0.01)

States:
  WIDE    = ratio > 3.0  → broad participation (healthy)
  NARROW  = ratio > 1.0  → limited participation
  THIN    = ratio <= 1.0 → narrow/deteriorating
```

**Data needed**: Daily `stk_limit` API (涨跌停统计). Verified working, 7631 records.
Fields `up_limit` and `down_limit` appear to be percentages.
**Rationale**: Strong bull markets have broad participation (many stocks hitting
limit-up). Late-stage bubbles often show narrowing breadth (fewer stocks
participating despite index highs).

#### Sub-Signal 4: Inflation/CPI Environment (CPI)

```
CPI_ENV(t) = classification of CPI trend

Compute:
  CPI_3M_avg = avg(CPI over last 3 months)
  
States:
  DISINFLATION = CPI_3M_avg < 1.0  → low inflation (favorable for equities)
  MODERATE     = 1.0 <= CPI_3M_avg <= 3.0
  INFLATION    = CPI_3M_avg > 3.0  → high inflation (negative for equities)
```

**Data needed**: Monthly CPI from `cn_cpi` API. Available 1951-2026, 509 records.
**Rationale**: Moderate/disinflation environments are historically favorable for
equity momentum strategies. High inflation erodes real returns and typically
coincides with tightening cycles.

#### Composite: 5-State Regime

```
REGIME(t) = f(TREND, LIQUID, BREADTH, CPI_ENV)

Decision table (majority-rule or weighted scoring):

State 1: RISK_ON    — BULL trend + LOOSE liquidity + WIDE breadth + disinflation
  → Maximum aggression: higher momentum weight, higher top_n, lower defense

State 2: CAUTIOUS   — WEAK trend OR NEUTRAL liquidity OR NARROW breadth
  → Moderate: default parameters (baseline v2.3 behavior)

State 3: DEFENSIVE  — BEAR trend OR TIGHT liquidity OR THIN breadth
  → Defensive: higher defense allocation, lower top_n, tighter stop-loss

State 4: BUBBLE_WARN — BULL trend + WIDE breadth BUT TIGHT liquidity
  → Liquidity divergence: market rising on thinning money supply
  → This is the key missing signal! Aggressive but with tightened stops

State 5: CRISIS     — BEAR trend + THIN breadth + TIGHT liquidity
  → Maximum defense: 95% defense allocation, no offensive positions
```

**Key innovation: BUBBLE_WARN state**. This is the state that the current
framework cannot identify. In a bubble, momentum scores are high (prices rising),
volatility is low (complacency), so the strategy goes ALL IN — exactly when the
crash is most likely. The BUBBLE_WARN state triggers when:
- Trend is bullish (CSI300 rising)
- Market breadth is wide (participation)
- BUT monetary conditions are tightening (M1 growth falling)

This is the classic "liquidity-driven rally running out of fuel" scenario.

### 2.4 Strategy Integration

The regime classifier output modulates strategy parameters via a **regime parameter table**:

```yaml
# New config section (additive, not modifying existing)
regime_classifier:
  enabled: false              # Default OFF — backward compatible
  data_dir: data/tushare/     # Output of fetch_regime_data.py
  
  # Sub-signal weights for regime voting
  trend_weight: 0.35
  liquid_weight: 0.30
  breadth_weight: 0.20
  cpi_weight: 0.15

  # Regime-specific parameter overrides
  regimes:
    RISK_ON:
      mom_w_override: 0.40       # ↑ momentum weight
      vol_w_override: 0.25       # ↓ volatility penalty
      top_n_override: 3          # ↑ diversification
      def_alloc_override: 0.15   # ↓ defense
      stop_loss_override: 0.10   # ↑ stop-loss room
    
    CAUTIOUS:
      # No overrides — use baseline v2.3 params
      # (mom_w=0.35, vol_w=0.30, top_n=2, def_alloc=0.25, stop_loss=0.08)
    
    DEFENSIVE:
      mom_w_override: 0.25       # ↓ momentum weight
      vol_w_override: 0.35       # ↑ volatility penalty
      top_n_override: 1          # ↓ concentration
      def_alloc_override: 0.40   # ↑ defense
      stop_loss_override: 0.05   # ↓ tighter stop
    
    BUBBLE_WARN:
      mom_w_override: 0.35       # keep momentum (trend is still good)
      vol_w_override: 0.30       # normal
      top_n_override: 2          # normal
      def_alloc_override: 0.35   # ↑ defense preemptively
      stop_loss_override: 0.04   # ↓↓ very tight stop — this is KEY
      # BUBBLE_WARN behavior: participate but with hair-trigger exit
    
    CRISIS:
      mom_w_override: 0.0        # ignore momentum
      vol_w_override: 0.0        # ignore volatility
      top_n_override: 0          # no offensive positions
      def_alloc_override: 0.95   # max defense
      stop_loss_override: 0.02   # immediate exit
```

**Integration flow**:
1. Weekly rebalance: fetch latest regime data from CSV
2. Compute 4 sub-signals → classify into 5-state regime
3. Look up parameter overrides from regime table
4. Override scoring/selection/defense params for this week
5. Run standard backtest with overridden parameters

### 2.5 Data Pipeline

Script: `scripts/fetch_regime_data.py` (to be created by coder)

```python
# Fetch regime classifier data from Tushare
# API calls (reuse token from build_pipeline_v2.py):
#   1. index_daily: CSI300 daily close → weekly resample
#   2. cn_m: monthly M1, M2
#   3. cn_cpi: monthly CPI
#   4. stk_limit: daily limit-up/limit-down pct
# Output: data/tushare/regime_signals.csv
#   Columns: week, trend_state, liquid_state, breadth_state, cpi_env, regime
```

Feasibility confirmed: all 4 APIs return data (verified at t_1b977ef2/verify_tushare.py).

### 2.6 Feasibility Assessment

| Sub-Signal | Data Source | API Status | Records | Coverage | Feasibility |
|------------|------------|------------|---------|----------|-------------|
| TREND | CSI300 daily | ✅ | 3253 (2013-2026) | Full backtest window | HIGH |
| LIQUID | cn_m (M1/M2) | ✅ | 580 (1978-2026) | Full, monthly→weekly interp | HIGH |
| BREADTH | stk_limit | ✅ | 7631 | Verify up_limit/down_limit semantics | MEDIUM |
| CPI_ENV | cn_cpi | ✅ | 509 (1951-2026) | Full, monthly→weekly interp | HIGH |

**Risk**: `stk_limit` fields (`up_limit`, `down_limit`) returned values like
12.02 and 9.84 — these appear to be **percentages** (涨跌停股票占比%), not counts.
Need to confirm semantics with a few historical dates during known market events
(e.g., 2015 crash, 2020 COVID). If they are percentages, the BREADTH signal
works as designed. If they are counts, normalization by total listed stocks
is needed.

---

## 3. Direction B: Adaptive Parameter Framework (P1)

### 3.1 Concept

Direction A classifies the regime as a discrete label. Direction B builds a
**continuous** adaptive parameter system that smoothly interpolates strategy
weights based on the same regime sub-signals, avoiding hard state transitions.

### 3.2 Signal Construction: Regime Strength Score (RSS)

Instead of discrete classification, compute a continuous **Regime Strength Score**
from the same 4 sub-signals:

```
RSS(t) = w_t × TREND_score + w_l × LIQUID_score + w_b × BREADTH_score + w_c × CPI_score

Where each sub-score is normalized to [0, 1]:

TREND_score(t):
  ma_slope = (MA_4 - MA_26) / MA_26
  score = sigmoid(ma_slope × 50)    → [0, 1], 0.5 when slope=0

LIQUID_score(t):
  m1_yoy = (M1(t) / M1(t-12) - 1)
  score = clamp(m1_yoy / 0.10 + 0.5, 0, 1)  → [0, 1], 0.5 when M1_YoY=0

BREADTH_score(t):
  ratio = UpPct_4w / max(DownPct_4w, 0.01)
  score = clamp((ratio - 1) / 4, 0, 1)  → [0, 1], 0.5 when ratio=3

CPI_score(t):
  cpi_level = avg(CPI, 3 months)
  score = 1 - clamp(cpi_level / 5, 0, 1)  → [0, 1], 1=disinflation

Default weights: w_t=0.35, w_l=0.30, w_b=0.20, w_c=0.15
```

### 3.3 Strategy Integration: Continuous Parameter Modulation

RSS maps to strategy parameters via interpolation:

```yaml
adaptive_parameters:
  enabled: false              # Default OFF
  
  # RSS range → parameter mapping (linear interpolation)
  rss_range: [0.0, 1.0]      # 0 = crisis, 1 = risk-on
  
  # Parameter curves
  mom_w_curve: [0.20, 0.45]   # RSS 0→1 maps to mom_w 0.20→0.45
  vol_w_curve: [0.38, 0.22]   # RSS 0→1 maps to vol_w 0.38→0.22 (inverted)
  top_n_curve: [0, 3]         # RSS 0→1 maps to top_n 0→3 (rounded)
  def_alloc_curve: [0.50, 0.10]  # RSS 0→1 maps to def 0.50→0.10
  stop_loss_curve: [0.03, 0.10]  # RSS 0→1 maps to stop 0.03→0.10
  
  # Smoothing
  rss_smoothing: 2            # weeks of EMA smoothing on RSS before applying
```

**Per-week computation**:
1. Compute TREND_score, LIQUID_score, BREADTH_score, CPI_score
2. Weighted sum → RSS_raw
3. EMA(RSS_raw, smoothing=2w) → RSS_smooth
4. RSS_smooth → interpolate each parameter curve
5. Apply interpolated parameters for this week's rebalance

### 3.4 Key Differences from Failed D1 (Dynamic Weighting)

| Aspect | Failed D1 | Direction B |
|--------|-----------|-------------|
| Data source | ETF-internal trend quality (内源性) | External market data (外源性) |
| Signal | ETF NAV-based momentum quality | CSI300 trend + M1/M2 + breadth + CPI |
| Mechanism | Fixed weight with TQ score | Continuous RSS with parameter curves |
| Why D1 failed | Internal data can't distinguish bull vs bubble | External data can identify liquidity divergence |

### 3.5 Feasibility Assessment

Direction B uses exactly the same data pipeline as Direction A — no additional
API calls needed. The main implementation difference is:

- **A**: Discrete classifier → lookup table → parameter overrides
- **B**: Continuous scoring → interpolation curves → parameter modulation

Both are feasible with the same data. Direction B requires more careful tuning
of the RSS→parameter mapping, but has the advantage of smooth transitions
(no regime-switching whipsaw).

---

## 4. Direction C: New Tushare Factors (P2)

### 4.1 Concept

Add 3 new additive signals to the scoring function, derived from Tushare data
that hasn't been explored yet. These are **bonus modifiers** (like T31's CWM/CONC)
added to the existing momentum score.

### 4.2 Signal 1: Market Volatility Index (MVIX — VIX-like)

**Rationale**: A homegrown "VIX" from CSI300 index daily returns. High MVIX
periods historically precede drawdowns. This signal acts as an early-warning
mechanism.

**Construction**:

```
MVIX(t) = rolling 20-day standard deviation of CSI300 daily returns × sqrt(252)

  daily_ret(t) = CSI300.close(t) / CSI300.close(t-1) - 1
  MVIX(t) = std(daily_ret[t-19:t]) × sqrt(252)  — annualized 20-day vol
```

**Integration**: Add as negative modifier to momentum score

```
mom_score_adj = mom_score - γ × max(MVIX - MVIX_median, 0) / MVIX_median
γ = 0.05–0.15 (to be tuned)
```

When MVIX > historical median, momentum scores are penalized proportionally.
This dampens strategy aggression during high-volatility regimes before losses occur.

**Data needed**: CSI300 daily close → compute rolling 20-day std. Available.
**Feasibility**: HIGH. Same `index_daily` API already verified.

### 4.3 Signal 2: Northbound Capital Flow Signal (NFLOW)

**Rationale**: Northbound (沪/深港通) capital flows are a well-followed sentiment
indicator in A-share markets. Sustained northbound inflows signal foreign
confidence; sustained outflows signal risk aversion.

**Construction**:

```
NFLOW(t) = cumulative northbound flow over past 4 weeks / avg daily turnover

  north_flow_sum = sum(north_money[t-19:t])     — from moneyflow_hsgt
  avg_turnover   = avg(CSI300.amount[t-19:t])   — from index_daily
  NFLOW(t)       = north_flow_sum / avg_turnover
```

**Integration**: Add as positive modifier to momentum score

```
mom_score_adj = mom_score + δ × NFLOW(t)
δ = 0.03–0.08 (to be tuned)
```

Positive NFLOW (inflows > avg) boosts momentum scores slightly. Negative NFLOW
penalizes.

**Data needed**: `moneyflow_hsgt` (daily north/south flows) + CSI300 amount.
**Feasibility**: MEDIUM. `moneyflow_hsgt` verified working (18 records in test
window). Need to verify data range (must cover 2013-2026).

### 4.4 Signal 3: Margin Balance Trend (MARGIN)

**Rationale**: Margin trading balance (融资余额) reflects leveraged speculation.
Rising margin balances during a bull market = healthy; rising margin during
weak breadth = warning sign.

**Construction**:

```
MARGIN(t) = (margin_balance(t) / margin_balance(t-12) - 1)   — YoY growth

  margin_balance = sum of rzye from margin_detail for a given date
```

**Integration**: Conditional modifier — depends on regime context

```
if BREADTH_score > 0.5:  # healthy breadth
    mom_score_adj = mom_score + ε × MARGIN(t)   # positive: leverage confirms trend
else:  # weak breadth
    mom_score_adj = mom_score - ε × MARGIN(t)   # negative: leverage = speculation
ε = 0.02–0.05 (to be tuned)
```

**Data needed**: `margin` or `margin_detail` API. Verified working (3 records
for `margin`, 4366 for `margin_detail`).
**Feasibility**: LOW-MEDIUM. The `margin` API returned only 3 records for a
single date — it may only support point queries (one date at a time), requiring
a loop over all trading days. This is expensive. `margin_detail` returns per-stock
data (4366 records), which requires aggregation. **Recommendation**: P2 — implement
only if pipeline cost is acceptable.

### 4.5 Integration Design

```yaml
new_tushare_signals:
  enabled: false              # Default OFF
  signals_path: data/tushare/new_factor_signals.csv
  
  # Per-signal weights
  mvix_enabled: true
  mvix_weight: 0.08           # γ
  
  nflow_enabled: true
  nflow_weight: 0.05          # δ
  
  margin_enabled: false       # P2 — implement after pipeline verification
  margin_weight: 0.03         # ε
```

All signals are **additive modifiers** to the momentum score. When `enabled: false`,
the code paths are skipped entirely — zero impact on existing backtest (backward
compatible by construction, same as T31).

### 4.6 Consolidated Data Pipeline

Script: `scripts/fetch_new_factors.py` (to be created by coder)

```python
# Fetch new factor data from Tushare
# API calls:
#   1. index_daily: CSI300 daily → MVIX computation (already fetched in Direction A)
#   2. moneyflow_hsgt: daily north/south flow
#   3. margin: daily total margin balance (point queries, may be expensive)
# Output: data/tushare/new_factor_signals.csv
#   Columns: week, MVIX, NFLOW, MARGIN_YoY
```

---

## 5. Consolidated Architecture

### 5.1 System Overview

```
Tushare API (external)
  ├── index_daily (CSI300, SH, CSI500) ──┐
  ├── cn_m (M1/M2) ──────────────────────┤
  ├── cn_cpi ────────────────────────────┤
  ├── stk_limit ─────────────────────────┤
  ├── moneyflow_hsgt ────────────────────┤
  └── margin/margin_detail ──────────────┘
              │
              ▼
    Data Pipeline (scripts/fetch_regime_data.py
                  + scripts/fetch_new_factors.py)
              │
              ▼
    Signal Engine (new: src/signals/regime.py)
      ├── Regime Classifier (Direction A)
      │     └── 5-state discrete output
      ├── RSS Scorer (Direction B)
      │     └── continuous [0,1] output
      └── New Factors (Direction C)
            └── MVIX, NFLOW, MARGIN modifiers
              │
              ▼
    Strategy Engine (src/strategy.py — MODIFIED)
      ├── Regime → parameter table lookup (A)
      ├── RSS → parameter interpolation (B)
      └── Factor → scoring modifier (C)
              │
              ▼
    Backtest (unchanged)
```

### 5.2 Implementation Phasing

| Phase | Direction | Deliverable | Dependencies |
|-------|-----------|-------------|--------------|
| **P0** | A — Regime Classifier | fetch_regime_data.py + regime.py + config changes | None |
| **P1** | B — Adaptive Parameters | Extends regime.py with RSS mode | P0 data pipeline |
| **P2a** | C — MVIX + NFLOW | fetch_new_factors.py + new_factors.py | P0 pipeline (reuses CSI300 data) |
| **P2b** | C — MARGIN | Extends fetch_new_factors.py | P2a + margin API cost verification |

### 5.3 Recommended Implementation Order

1. **P0 first** (Direction A): Implement the regime classifier and verify it
   correctly identifies the 2015 bubble, 2018 bear, 2020 COVID crash, and 2022-24
   bull market in backtest. If the classifier correctly flags BUBBLE_WARN before
   drawdowns, this alone should substantially reduce walk-forward Sharpe variance.

2. **P1 next** (Direction B): Add continuous RSS as an alternative mode. Compare
   discrete (A) vs continuous (B) in backtest. Keep whichever works better.

3. **P2 last** (Direction C): Add MVIX and NFLOW as incremental improvements.
   Skip MARGIN unless margin API pipeline cost proves acceptable.

---

## 6. Backward Compatibility Analysis

### 6.1 Compatibility Strategy

All three directions use the same **opt-in flag pattern** established in T31
(constituent signals) and D4 (individual momentum filter):

```yaml
regime_classifier:
  enabled: false    # ← DEFAULT OFF

adaptive_parameters:
  enabled: false    # ← DEFAULT OFF

new_tushare_signals:
  enabled: false    # ← DEFAULT OFF
```

When `enabled: false` (default):
- All regime/signal code paths are skipped
- Scoring is 100% identical to baseline v2.3
- Zero performance difference
- Existing backtest reproduces exactly

### 6.2 Verification Method

1. Run baseline backtest with all `enabled: false`
2. Compare to known baseline: 14.11% ann ret / 7.42% max DD / 1.102 Sharpe
3. Assert: identical NAV curve, identical Sharpe, identical MaxDD
4. Then enable features incrementally and compare differences

### 6.3 Config File Impact

Additive sections only. No modifications to existing config keys:

```yaml
# Existing config — UNCHANGED
strategy: {...}
scoring: {...}
selection: {...}
factors: {...}
defense: {...}
rebalance: {...}
risk_control: {...}
allocation: {...}

# New sections — ADDITIVE
regime_classifier: {...}      # Direction A
adaptive_parameters: {...}    # Direction B
new_tushare_signals: {...}    # Direction C
```

---

## 7. Tushare API Feasibility Verification

### 7.1 Verification Scripts

Two verification scripts were run against the Tushare HTTP API at
`http://8.148.76.181:8686/` using the existing token from T27/T31:

- `verify_tushare.py` — 10 API endpoints tested
- `verify_tushare2.py` — 12 additional API endpoints tested

Both scripts are preserved at the task workspace for reference:
`/home/ubuntu/.hermes/kanban/boards/claw-etf/workspaces/t_1b977ef2/verify_tushare.py`

### 7.2 Results Summary

| API | Status | Records | Coverage |
|-----|--------|---------|----------|
| index_daily (CSI300) | ✅ | 3253 | 2013-2026 daily |
| index_daily (SH) | ✅ | 3253 | 2013-2026 daily |
| index_daily (CSI500) | ✅ | 3253 | 2013-2026 daily |
| index_weekly (CSI300) | ✅ | 687 | 2013-2026 weekly |
| index_classify | ✅ | 31 | SW L1 industries |
| cn_m | ✅ | 580 | 1978-2026 monthly |
| cn_cpi | ✅ | 509 | 1951-2026 monthly |
| stk_limit | ✅ | 7631 | Daily |
| moneyflow_hsgt | ✅ | 18+ | Daily |
| margin | ✅ | 3+ | Point query |
| SW index_daily | ❌ | 0 | N/A |
| daily_basic (index) | ❌ | 0 | N/A |
| ths_index/ths_daily | ❌ | 0 | N/A |

**GATE G1**: 9/10+ working APIs, well above the ≥3 threshold. ✅ PASS

### 7.3 Token Reuse

The Tushare token (`6886937c...`) is the same token used by the T27 data
pipeline and T31 constituent data script. It is extracted from
`scripts/build_pipeline_v2.py` at runtime. No new credential needed.

---

## 8. What If All Directions Are Infeasible? (Fallback Plan)

Per criterion 8: if all directions are judged infeasible, the fallback plan is:

### 8.1 Feasibility Verdict

All three directions have feasible data sources (G1 PASS), feasible signal
construction (verified formulas), and feasible integration (additive config,
backward compatible). **None are infeasible.**

### 8.2 Contingency: If Backtest Shows No Improvement

If after implementation and backtest, the regime-aware framework shows NO
reduction in walk-forward Sharpe variance:

1. **Hypothesis A is wrong**: The regime classifier may not generalize across
   market cycles. Fallback: simplify to 3-state (RISK_ON, NEUTRAL, RISK_OFF)
   using only TREND + LIQUID signals.
   
2. **Parameters need tuning**: The regime→parameter mapping may need optimization.
   Fallback: run a grid search on regime thresholds.

3. **Ultimate fallback**: If regime-aware approach fails completely, recommend
   project wind-down with the following rationale documented:
   - The 5-ETF universe with momentum+volatility scoring has fundamental limits
   - External regime signals cannot overcome these limits
   - Recommend: expand to a fundamentally different strategy type (e.g., risk
     parity, trend-following across broader universe, or factor-based approach)

### 8.3 Go/No-Go Criteria

| Criterion | Threshold | Action |
|-----------|-----------|--------|
| WF Sharpe std dev improved | < 0.60 (from 1.208) | PROCEED to production |
| WF Sharpe std dev moderately improved | < 0.80 | CONDITIONAL PROCEED |
| WF Sharpe std dev unchanged | > 1.0 | INVESTIGATE simplification (3-state) |
| WF Sharpe std dev worse | > 1.5 | ABANDON direction, try next |

---

## 9. References

- `config/strategy_v2_3_cap040.yaml` — Baseline v2.3 configuration
- `config/strategy_v2_3_cap040_D4_tuned.yaml` — D4-tuned configuration (15.65%/7.58%/1.216)
- `output/robustness_results.json` — Baseline robustness analysis (WF std dev=1.208)
- `docs/tushare_constituent_signals.md` — T31 design document (format reference)
- `scripts/fetch_constituent_data.py` — Tushare API call example
- `scripts/build_pipeline_v2.py` — Token extraction + API base URL
- T27/T28/T31 — Prior Tushare pipeline experience
- `verify_tushare.py` / `verify_tushare2.py` — T34 API verification scripts

---

*Document version: 1.0 — SE Research Phase (Design Only)*
*Next: PM review → coder implementation of P0 (Direction A)*