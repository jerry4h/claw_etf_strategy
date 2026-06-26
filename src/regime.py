"""
T35: Tushare-Based Market Regime Classifier (Direction A)

Implements 4 sub-signals and a 5-state regime classifier using
Tushare external data (CSI300, M1/M2, CPI, market breadth).

4 Sub-signals:
  - TREND: CSI300 26-week MA slope and position (BULL/WEAK/CHOPPY/BEAR)
  - LIQUID: M1/M2 monetary conditions (LOOSE/NEUTRAL/TIGHT)
  - BREADTH: Limit-up/down ratio (WIDE/NARROW/THIN)
  - CPI_ENV: Inflation environment (DISINFLATION/MODERATE/INFLATION)

5-State Regime Classifier:
  - RISK_ON
  - CAUTIOUS
  - DEFENSIVE
  - BUBBLE_WARN
  - CRISIS
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class TrendState(Enum):
    BULL = "BULL"
    WEAK = "WEAK"
    CHOPPY = "CHOPPY"
    BEAR = "BEAR"


class LiquidState(Enum):
    LOOSE = "LOOSE"
    NEUTRAL = "NEUTRAL"
    TIGHT = "TIGHT"


class BreadthState(Enum):
    WIDE = "WIDE"
    NARROW = "NARROW"
    THIN = "THIN"


class CPIState(Enum):
    DISINFLATION = "DISINFLATION"
    MODERATE = "MODERATE"
    INFLATION = "INFLATION"


class Regime(Enum):
    RISK_ON = "RISK_ON"
    CAUTIOUS = "CAUTIOUS"
    DEFENSIVE = "DEFENSIVE"
    BUBBLE_WARN = "BUBBLE_WARN"
    CRISIS = "CRISIS"


def load_regime_data(csv_path: str) -> pd.DataFrame:
    """Load regime signals CSV into a DataFrame with week index.

    Returns DataFrame with columns: csi300_close, m1_yoy, m2_yoy, m1m2_gap,
    up_pct_4w, down_pct_4w, cpi_3m_avg, indexed by week (datetime).
    """
    df = pd.read_csv(csv_path)
    df['week'] = pd.to_datetime(df['week'], format='%Y%m%d')
    df = df.set_index('week')
    # Convert numeric columns
    for col in ['csi300_close', 'sh_close', 'm1_yoy', 'm2_yoy', 'm1m2_gap',
                'up_pct_4w', 'down_pct_4w', 'cpi_3m_avg', 'breadth_score']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# ===========================================================================
# Sub-Signal 1: TREND (CSI300 26-week MA slope + position)
# ===========================================================================

def compute_trend_state(
    csi300_close: float,
    ma_26: float,
    ma_4: float
) -> TrendState:
    """
    Compute TREND state from CSI300 close and moving averages.

    MA_26 = SMA(close, 26w), MA_4 = SMA(close, 4w)
    slope = (MA_4 - MA_26) / MA_26

    BULL:   slope > 0.02 AND close > MA_26
    WEAK:   slope > 0 AND close > MA_26
    CHOPPY: abs(slope) <= 0.01
    BEAR:   slope < 0 AND close < MA_26
    """
    if pd.isna(csi300_close) or pd.isna(ma_26) or pd.isna(ma_4):
        return TrendState.CHOPPY
    if ma_26 == 0:
        return TrendState.CHOPPY

    slope = (ma_4 - ma_26) / ma_26

    if slope > 0.02 and csi300_close > ma_26:
        return TrendState.BULL
    elif slope > 0 and csi300_close > ma_26:
        return TrendState.WEAK
    elif abs(slope) <= 0.01:
        return TrendState.CHOPPY
    elif slope < 0 and csi300_close < ma_26:
        return TrendState.BEAR
    else:
        # Catch-all: weak bear / transition
        return TrendState.CHOPPY


# ===========================================================================
# Sub-Signal 2: LIQUID (M1/M2 monetary conditions)
# ===========================================================================

def compute_liquid_state(
    m1_yoy: float,
    m1m2_gap: float
) -> LiquidState:
    """
    Compute LIQUID state from M1 YoY and M1-M2 gap.

    M1_YoY = (M1(t) / M1(t-12) - 1)
    M1M2_gap = M1_YoY - M2_YoY

    LOOSE:   M1_YoY > 0.05 AND M1M2_gap > -0.02
    NEUTRAL: abs(M1_YoY) <= 0.05
    TIGHT:   M1_YoY < -0.02
    """
    if pd.isna(m1_yoy) or pd.isna(m1m2_gap):
        return LiquidState.NEUTRAL

    if m1_yoy > 0.05 and m1m2_gap > -0.02:
        return LiquidState.LOOSE
    elif abs(m1_yoy) <= 0.05:
        return LiquidState.NEUTRAL
    elif m1_yoy < -0.02:
        return LiquidState.TIGHT
    else:
        return LiquidState.NEUTRAL


# ===========================================================================
# Sub-Signal 3: BREADTH (CSI300/SH Composite relative performance proxy)
# ===========================================================================

def compute_breadth_state(
    up_pct_4w: float,
    down_pct_4w: float
) -> BreadthState:
    """
    Compute BREADTH state from limit-up/down percentage averages.

    This is the LEGACY path using stk_limit data — DEPRECATED when
    stk_limit data is unavailable (common on Tushare mirrors).

    ratio = UpPct_4w / max(DownPct_4w, 0.01)

    WIDE:   ratio > 3.0
    NARROW: ratio > 1.0
    THIN:   ratio <= 1.0
    """
    if pd.isna(up_pct_4w) or pd.isna(down_pct_4w):
        # Missing data: default to NARROW (neutral)
        return BreadthState.NARROW

    safe_down = max(down_pct_4w, 0.01)
    ratio = up_pct_4w / safe_down

    if ratio > 3.0:
        return BreadthState.WIDE
    elif ratio > 1.0:
        return BreadthState.NARROW
    else:
        return BreadthState.THIN


def compute_breadth_from_proxy(breadth_score: float) -> BreadthState:
    """
    Compute BREADTH state from the CSI300/SH Composite proxy.

    breadth_score = z-score of CSI300/SH Composite 4-week avg ratio
                    over a 52-week rolling window.

    When CSI300 >> SH Composite (large caps dominating), the ratio is high
    → positive z-score = NARROW/THIN breadth.

    When CSI300 ≈ SH Composite (broad participation), the ratio is lower
    → negative z-score = WIDE breadth.

    Note: The z-score is inverted — a HIGH CSI300/SH ratio means narrow
    breadth (only large caps rising), while a LOW ratio means broad
    participation (small/mid caps keeping up).

    Thresholds (from T36 Phase 4 Fix 1):
    - WIDE:   breadth_score > +1.0 (broad participation — ratio is
              LOW relative to history → -z → inverted to +z)
    - THIN:   breadth_score < -1.0 (narrow leadership — ratio is
              HIGH relative to history → +z → inverted to -z)
    - NARROW: in between

    ACTUAL computation: we invert the z-score in fetch_regime_data.py,
    so here a positive breadth_score = broad, negative = narrow.
    """
    if pd.isna(breadth_score):
        return BreadthState.NARROW

    if breadth_score > 1.0:
        return BreadthState.WIDE
    elif breadth_score < -1.0:
        return BreadthState.THIN
    else:
        return BreadthState.NARROW


# ===========================================================================
# Sub-Signal 4: CPI_ENV (inflation environment)
# ===========================================================================

def compute_cpi_state(cpi_3m_avg: float) -> CPIState:
    """
    Compute CPI_ENV state from 3-month average CPI YoY.

    DISINFLATION: CPI_3M_avg < 1.0
    MODERATE:     1.0 <= CPI_3M_avg <= 3.0
    INFLATION:    CPI_3M_avg > 3.0
    """
    if pd.isna(cpi_3m_avg):
        return CPIState.MODERATE

    if cpi_3m_avg < 1.0:
        return CPIState.DISINFLATION
    elif cpi_3m_avg <= 3.0:
        return CPIState.MODERATE
    else:
        return CPIState.INFLATION


# ===========================================================================
# Composite: 5-State Regime Classifier
# ===========================================================================

def classify_regime(
    trend: TrendState,
    liquid: LiquidState,
    breadth: BreadthState,
    cpi_env: CPIState
) -> Regime:
    """
    Classify market regime from 4 sub-signal states.

    Decision table (v2 — full BREADTH integration with CSI300/SH proxy):
    RISK_ON:     (BULL|WEAK) trend + LOOSE liquid + NOT-THIN breadth
    CAUTIOUS:    default catch-all
    DEFENSIVE:   (BEAR AND THIN) OR TIGHT
    BUBBLE_WARN: (BULL|WEAK) trend + WIDE breadth + TIGHT liquid
                 (key signal: liquidity divergence — rising market
                  with broad participation but tightening money supply)
    CRISIS:      BEAR trend + THIN breadth + TIGHT liquid

    Priority: CRISIS > BUBBLE_WARN > DEFENSIVE > RISK_ON > CAUTIOUS
    """
    # CRISIS: BEAR trend + THIN breadth + TIGHT liquid
    if (trend == TrendState.BEAR and
        breadth == BreadthState.THIN and
        liquid == LiquidState.TIGHT):
        return Regime.CRISIS

    # BUBBLE_WARN: (BULL|WEAK) trend + WIDE breadth + TIGHT liquid
    # Liquidity divergence: market rising broadly but money supply tightening
    if (trend in (TrendState.BULL, TrendState.WEAK) and
        breadth == BreadthState.WIDE and
        liquid == LiquidState.TIGHT):
        return Regime.BUBBLE_WARN

    # DEFENSIVE: (BEAR AND THIN) OR TIGHT
    if ((trend == TrendState.BEAR and breadth == BreadthState.THIN) or
            liquid == LiquidState.TIGHT):
        return Regime.DEFENSIVE

    # RISK_ON: (BULL|WEAK) trend + LOOSE liquid + NOT-THIN breadth
    if (trend in (TrendState.BULL, TrendState.WEAK) and
        liquid == LiquidState.LOOSE and
        breadth != BreadthState.THIN):
        return Regime.RISK_ON

    # CAUTIOUS: everything else
    return Regime.CAUTIOUS


# ===========================================================================
# Fix 2: 3-State Simplified Regime Classifier
# ===========================================================================

def classify_regime_3state(
    trend: TrendState,
    liquid: LiquidState,
    breadth: BreadthState,
    cpi_env: CPIState
) -> Regime:
    """
    Fix 2: 3-state simplified regime classifier (CONSERVATIVE).

    Preserves Fix 1 (5-state) classification boundaries but collapses
    empty BUBBLE_WARN and CRISIS states into CAUTIOUS/DEFENSIVE.
    This produces identical classification to Fix 1 while simplifying
    the state-space.

    Decision table (3-state):
    RISK_ON:     (BULL|WEAK) trend + LOOSE liquid + NOT-THIN breadth
    DEFENSIVE:   (BEAR AND THIN) OR TIGHT  (same as 5-state DEFENSIVE rule)
    CAUTIOUS:    everything else (absorbs BUBBLE_WARN and CRISIS)

    Priority: DEFENSIVE > RISK_ON > CAUTIOUS
    """
    # DEFENSIVE: (BEAR AND THIN) OR TIGHT (same as original 5-state)
    if ((trend == TrendState.BEAR and breadth == BreadthState.THIN) or
            liquid == LiquidState.TIGHT):
        return Regime.DEFENSIVE

    # RISK_ON: (BULL|WEAK) trend + LOOSE liquid + NOT-THIN breadth
    if (trend in (TrendState.BULL, TrendState.WEAK) and
        liquid == LiquidState.LOOSE and
        breadth != BreadthState.THIN):
        return Regime.RISK_ON

    # CAUTIOUS: everything else (including old BUBBLE_WARN and CRISIS triggers)
    return Regime.CAUTIOUS


# ===========================================================================
# Convenience: classify from a single row of the regime CSV
# ===========================================================================

def classify_from_row_3state(row: pd.Series) -> Regime:
    """
    Classify regime from a single row using the 3-state classifier.

    Args:
        row: Pandas Series with keys:
            csi300_close, ma_26, ma_4, m1_yoy, m1m2_gap,
            up_pct_4w, down_pct_4w, cpi_3m_avg, breadth_score

    Returns:
        Regime enum value (only RISK_ON/CAUTIOUS/DEFENSIVE)
    """
    trend = compute_trend_state(
        row['csi300_close'], row['ma_26'], row['ma_4']
    )
    liquid = compute_liquid_state(
        row.get('m1_yoy', np.nan), row.get('m1m2_gap', np.nan)
    )
    breadth_score = row.get('breadth_score', np.nan)
    if not pd.isna(breadth_score):
        breadth = compute_breadth_from_proxy(breadth_score)
    else:
        breadth = compute_breadth_state(
            row.get('up_pct_4w', np.nan), row.get('down_pct_4w', np.nan)
        )
    cpi_env = compute_cpi_state(row.get('cpi_3m_avg', np.nan))

    return classify_regime_3state(trend, liquid, breadth, cpi_env)


def build_regime_lookup_3state(
    regime_df: pd.DataFrame
) -> dict:
    """
    Build a week_str -> Regime lookup using the 3-state classifier.

    Args:
        regime_df: DataFrame from load_regime_data(), indexed by week

    Returns:
        dict: week_str (YYYYMMDD) -> {'regime': Regime, 'trend': TrendState, ...}
    """
    df = regime_df.copy()
    df['ma_26'] = df['csi300_close'].rolling(26, min_periods=12).mean()
    df['ma_4'] = df['csi300_close'].rolling(4, min_periods=2).mean()

    lookup = {}
    for idx, row in df.iterrows():
        week_str = idx.strftime('%Y%m%d')
        trend = compute_trend_state(row['csi300_close'], row['ma_26'], row['ma_4'])
        liquid = compute_liquid_state(
            row.get('m1_yoy', np.nan), row.get('m1m2_gap', np.nan)
        )
        breadth_score = row.get('breadth_score', np.nan)
        if not pd.isna(breadth_score):
            breadth = compute_breadth_from_proxy(breadth_score)
        else:
            breadth = compute_breadth_state(
                row.get('up_pct_4w', np.nan), row.get('down_pct_4w', np.nan)
            )
        cpi_env = compute_cpi_state(row.get('cpi_3m_avg', np.nan))
        regime = classify_regime_3state(trend, liquid, breadth, cpi_env)

        lookup[week_str] = {
            'regime': regime,
            'trend': trend,
            'liquid': liquid,
            'breadth': breadth,
            'cpi_env': cpi_env,
        }

    return lookup


# ===========================================================================
# Fix 2: 3-State Regime Parameter Overrides
# ===========================================================================

DEFAULT_REGIME_OVERRIDES_3STATE = {
    Regime.RISK_ON: {
        'mom_w_override': 0.40,
        'vol_w_override': 0.25,
        'top_n_override': 3,
        'def_alloc_override': 0.15,
        'stop_loss_override': 0.10,
    },
    Regime.CAUTIOUS: {
        # No overrides — use baseline params
    },
    Regime.DEFENSIVE: {
        'mom_w_override': 0.25,
        'vol_w_override': 0.35,
        'top_n_override': 1,
        'def_alloc_override': 0.40,
        'stop_loss_override': 0.05,
    },
}


def get_regime_overrides_3state(regime: Regime, custom_overrides: dict | None = None) -> dict:
    """
    Get parameter overrides for a given regime using 3-state defaults.

    Args:
        regime: The classified regime (3-state)
        custom_overrides: Optional custom overrides dict (from config)

    Returns:
        dict of parameter overrides
    """
    source = custom_overrides if custom_overrides else DEFAULT_REGIME_OVERRIDES_3STATE
    return dict(source.get(regime, {}))


# ===========================================================================
# Convenience: classify from a single row of the regime CSV (5-state)
# ===========================================================================

def classify_from_row(row: pd.Series) -> Regime:
    """
    Classify regime from a single row of regime_signals.csv.

    The row must already have MA_26 and MA_4 pre-computed
    (or we compute them externally and attach to the row).

    Args:
        row: Pandas Series with keys:
            csi300_close, ma_26, ma_4, m1_yoy, m1m2_gap,
            up_pct_4w, down_pct_4w, cpi_3m_avg, breadth_score

    Returns:
        Regime enum value
    """
    trend = compute_trend_state(
        row['csi300_close'], row['ma_26'], row['ma_4']
    )
    liquid = compute_liquid_state(
        row.get('m1_yoy', np.nan), row.get('m1m2_gap', np.nan)
    )
    # Try breadth_score proxy first (CSI300/SH ratio), fall back to stk_limit
    breadth_score = row.get('breadth_score', np.nan)
    if not pd.isna(breadth_score):
        breadth = compute_breadth_from_proxy(breadth_score)
    else:
        breadth = compute_breadth_state(
            row.get('up_pct_4w', np.nan), row.get('down_pct_4w', np.nan)
        )
    cpi_env = compute_cpi_state(row.get('cpi_3m_avg', np.nan))

    return classify_regime(trend, liquid, breadth, cpi_env)


# ===========================================================================
# Build regime lookup table for backtest
# ===========================================================================

def build_regime_lookup(
    regime_df: pd.DataFrame
) -> dict:
    """
    Build a week_str -> Regime lookup for fast backtest access.

    Computes MA_26 and MA_4 on the CSI300 close series, then
    classifies regime for each week.

    Args:
        regime_df: DataFrame from load_regime_data(), indexed by week

    Returns:
        dict: week_str (YYYYMMDD) -> {'regime': Regime, 'trend': TrendState, ...}
    """
    # Compute moving averages on the full series
    df = regime_df.copy()
    df['ma_26'] = df['csi300_close'].rolling(26, min_periods=12).mean()
    df['ma_4'] = df['csi300_close'].rolling(4, min_periods=2).mean()

    lookup = {}
    for idx, row in df.iterrows():
        week_str = idx.strftime('%Y%m%d')
        trend = compute_trend_state(row['csi300_close'], row['ma_26'], row['ma_4'])
        liquid = compute_liquid_state(
            row.get('m1_yoy', np.nan), row.get('m1m2_gap', np.nan)
        )
        breadth_score = row.get('breadth_score', np.nan)
        if not pd.isna(breadth_score):
            breadth = compute_breadth_from_proxy(breadth_score)
        else:
            breadth = compute_breadth_state(
                row.get('up_pct_4w', np.nan), row.get('down_pct_4w', np.nan)
            )
        cpi_env = compute_cpi_state(row.get('cpi_3m_avg', np.nan))
        regime = classify_regime(trend, liquid, breadth, cpi_env)

        lookup[week_str] = {
            'regime': regime,
            'trend': trend,
            'liquid': liquid,
            'breadth': breadth,
            'cpi_env': cpi_env,
        }

    return lookup


# ===========================================================================
# Regime → Parameter Overrides
# ===========================================================================

# Default regime parameter overrides (matches design doc Section 2.4)
DEFAULT_REGIME_OVERRIDES = {
    Regime.RISK_ON: {
        'mom_w_override': 0.40,
        'vol_w_override': 0.25,
        'top_n_override': 3,
        'def_alloc_override': 0.15,
        'stop_loss_override': 0.10,
    },
    Regime.CAUTIOUS: {
        # No overrides — use baseline params
    },
    Regime.DEFENSIVE: {
        'mom_w_override': 0.25,
        'vol_w_override': 0.35,
        'top_n_override': 1,
        'def_alloc_override': 0.40,
        'stop_loss_override': 0.05,
    },
    Regime.BUBBLE_WARN: {
        'mom_w_override': 0.35,
        'vol_w_override': 0.30,
        'top_n_override': 2,
        'def_alloc_override': 0.35,
        'stop_loss_override': 0.04,  # KEY: very tight stop in bubble
    },
    Regime.CRISIS: {
        'mom_w_override': 0.0,
        'vol_w_override': 0.0,
        'top_n_override': 0,
        'def_alloc_override': 0.95,
        'stop_loss_override': 0.02,
    },
}


def get_regime_overrides(regime: Regime, custom_overrides: dict | None = None) -> dict:
    """
    Get parameter overrides for a given regime.

    Args:
        regime: The classified regime
        custom_overrides: Optional custom overrides dict (from config)

    Returns:
        dict of parameter overrides (may be empty for CAUTIOUS)
    """
    source = custom_overrides if custom_overrides else DEFAULT_REGIME_OVERRIDES
    return dict(source.get(regime, {}))
