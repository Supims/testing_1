"""
Strategy Base + Registry
========================
Protocol for all strategies. Each strategy returns a signal in [-1, +1].
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger("models.strategies")


@dataclass
class StrategyParams:
    """Override in subclasses."""
    pass


class BaseStrategy(ABC):
    """Abstract base for all strategies. Output: continuous signal in [-1, +1]."""

    name: str = "base"

    def __init__(self, params: StrategyParams = None, htf_prefix: str = "htf_"):
        self.params = params or self._default_params()
        self.htf = htf_prefix

    @abstractmethod
    def _default_params(self) -> StrategyParams:
        ...

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute signal. Must return Series in [-1, +1]."""
        ...

    @abstractmethod
    def param_space(self) -> list[tuple[str, float, float]]:
        """Return [(name, low, high)] for hyperopt."""
        ...

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.params, k):
                setattr(self.params, k, v)

    def _check(self, df: pd.DataFrame, cols: list[str]) -> bool:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            logger.debug("%s: missing columns %s", self.name, missing)
            return False
        return True

    def _htf_col(self, df: pd.DataFrame, col: str) -> str:
        """Resolve HTF column with LTF fallback."""
        htf = f"{self.htf}{col}"
        if htf in df.columns:
            return htf
        return col

    def _clip(self, raw, df: pd.DataFrame) -> pd.Series:
        """Clip signal to [-1, +1] and convert to named Series."""
        clipped = np.clip(raw, -1.0, 1.0)
        return pd.Series(clipped, index=df.index, name=self.name).fillna(0.0)

    @staticmethod
    def _sigmoid(x, center=0.0, sharpness=5.0):
        """Safe sigmoid activation: clips exponent to prevent overflow."""
        z = np.clip(sharpness * (np.asarray(x, dtype=float) - np.asarray(center, dtype=float)), -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

