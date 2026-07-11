"""Market regime enum and stubs (regime features disabled in v3.0 final).

This module exists to satisfy imports from backtest.py and strategy.py
while all regime-based features are disabled in the final config.
"""
from __future__ import annotations

from enum import Enum


class Regime(str, Enum):
    RISK_ON = "RISK_ON"
    CAUTIOUS = "CAUTIOUS"
    DEFENSIVE = "DEFENSIVE"
    BUBBLE_WARN = "BUBBLE_WARN"
    CRISIS = "CRISIS"


def load_regime_data(*args, **kwargs) -> None:
    """Regime data loading is disabled in v3.0 final."""
    return None


def build_regime_lookup(*args, **kwargs) -> None:
    return None


def build_regime_lookup_3state(*args, **kwargs) -> None:
    return None


def get_regime_overrides(*args, **kwargs) -> dict:
    return {}


def get_regime_overrides_3state(*args, **kwargs) -> dict:
    return {}