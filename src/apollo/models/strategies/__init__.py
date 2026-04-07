"""
Strategy Registry
==================
All strategies registered here. Use compute_all() for batch execution.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from apollo.models.strategies.base import BaseStrategy
from apollo.models.strategies.trend import TrendStrategy
from apollo.models.strategies.mean_reversion import MeanReversionStrategy
from apollo.models.strategies.squeeze import SqueezeStrategy
from apollo.models.strategies.smart_money import SmartMoneyStrategy
from apollo.models.strategies.basis_arb import BasisArbStrategy
from apollo.models.strategies.breakout import BreakoutStrategy
from apollo.models.strategies.funding_momentum import FundingMomentumStrategy
from apollo.models.strategies.oi_divergence import OIDivergenceStrategy
from apollo.models.strategies.liquidation_cascade import LiquidationCascadeStrategy
from apollo.models.strategies.volume_profile import VolumeProfileStrategy

logger = logging.getLogger("models.strategies")

REGISTRY: dict[str, type[BaseStrategy]] = {
    "trend": TrendStrategy,
    "mean_reversion": MeanReversionStrategy,
    "squeeze": SqueezeStrategy,
    "smart_money": SmartMoneyStrategy,
    "basis_arb": BasisArbStrategy,
    "breakout": BreakoutStrategy,
    "funding_mom": FundingMomentumStrategy,
    "oi_divergence": OIDivergenceStrategy,
    "liq_cascade": LiquidationCascadeStrategy,
    "vol_profile": VolumeProfileStrategy,
}

ALL_STRATEGY_NAMES = list(REGISTRY.keys())


def get_all_strategies(htf_prefix: str = "htf_") -> dict[str, BaseStrategy]:
    """Instantiate all strategies with default params."""
    return {name: cls(htf_prefix=htf_prefix) for name, cls in REGISTRY.items()}


def compute_all(df: pd.DataFrame, htf_prefix: str = "htf_") -> pd.DataFrame:
    """Run all strategies and return DataFrame of signals."""
    strategies = get_all_strategies(htf_prefix)
    signals = {}
    for name, strat in strategies.items():
        signals[name] = strat.compute(df)
        active = (signals[name] != 0).sum()
        logger.info("  %s: %d active bars (%.1f%%)", name, active, active / len(df) * 100)
    return pd.DataFrame(signals, index=df.index)
