"""T32: Constituent-Stock Signals (CWM + CONC) -- DISABLED in v3.0 final.

Loads quarterly constituent-level signals (cross-sectional momentum and
concentration) and applies bonus modifiers to ETF scoring:
  score_bonus = cwm_weight * cwm_signal + conc_weight * conc_signal

Disabled because the signal data requires tushare API access and the
bonus did not materially improve out-of-sample performance.
"""

import csv
from pathlib import Path


def load_constituent_signals(signals_path: str | Path) -> dict:
    """Load constituent signals from CSV.

    Returns:
        {end_date_str: {etf_name: {'cwm': float, 'conc': float}}}
    """
    raw = {}
    path = Path(signals_path)
    if not path.exists():
        return raw

    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row['end_date'].strip()
            etf = row['etf_name'].strip()
            raw.setdefault(d, {})[etf] = {
                'cwm': float(row.get('cwm', 0) or 0),
                'conc': float(row.get('conc', 0) or 0),
            }
    return raw


def build_constituent_lookup(
    raw: dict,
    week_index,
) -> dict:
    """Build forward-filled lookup: week_date_str -> {etf_name -> signals}.

    Args:
        raw: Raw signals from load_constituent_signals()
        week_index: DatetimeIndex of backtest weeks

    Returns:
        {week_date_str: {etf_name: {'cwm': float, 'conc': float}}}
    """
    lookup = {}
    if not raw:
        return lookup

    sorted_sig_dates = sorted(raw.keys())
    for week_dt in week_index:
        week_str = week_dt.strftime('%Y%m%d')
        best_date = None
        for sd in sorted_sig_dates:
            if sd <= week_str:
                best_date = sd
            else:
                break
        if best_date:
            lookup[week_str] = raw.get(best_date, {})
    return lookup


def apply_constituent_bonus(
    scores_vec,
    off_idx: list[int],
    etf_names: list[str],
    date_str: str,
    lookup: dict,
    cwm_weight: float,
    conc_weight: float,
) -> None:
    """Apply constituent signal bonus to scores_vec in place.

    Args:
        scores_vec: Score array (modified in place)
        off_idx: Offensive ETF indices
        etf_names: ETF name list
        date_str: Current date string (YYYYMMDD)
        lookup: Constituent signal lookup
        cwm_weight: Alpha weight for CWM signal
        conc_weight: Beta weight for CONC signal
    """
    sigs = lookup.get(date_str, {})
    if not sigs:
        return

    import numpy as np
    for j in off_idx:
        etf = etf_names[j]
        if etf in sigs and not np.isnan(scores_vec[j]):
            cwm_val = sigs[etf].get('cwm', 0.0)
            conc_val = sigs[etf].get('conc', 0.0)
            scores_vec[j] += cwm_weight * cwm_val + conc_weight * conc_val
