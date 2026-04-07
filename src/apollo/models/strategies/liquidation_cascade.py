"""
Liquidation Cascade -- Graduated cascade probability scoring
=============================================================
Detects fragile positioning (extreme OI/funding + taker stress + volume surge)
that precedes forced liquidation cascades. Unique to crypto derivatives.

Uses soft activation for each condition contributing [0,1] continuously,
combined as geometric mean. Still event-driven (~10-15% activation)
but produces smooth, proportional signals when stress builds.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from apollo.models.strategies.base import BaseStrategy, StrategyParams


@dataclass
class LiqCascadeParams(StrategyParams):
    funding_extreme: float = 1.5
    funding_sharpness: float = 2.5
    oi_extreme_threshold: float = 1.0
    oi_sharpness: float = 2.0
    taker_threshold: float = 1.5
    taker_sharpness: float = 2.0
    volume_surge_threshold: float = 1.5
    volume_sharpness: float = 2.0
    cascade_decay: int = 5


class LiquidationCascadeStrategy(BaseStrategy):
    name = "liq_cascade"

    def _default_params(self):
        return LiqCascadeParams()

    def param_space(self):
        return [
            ("funding_extreme", 1.0, 2.5),
            ("funding_sharpness", 1.5, 5.0),
            ("oi_extreme_threshold", 0.5, 2.0),
            ("oi_sharpness", 1.0, 4.0),
            ("taker_threshold", 1.0, 3.0),
            ("volume_surge_threshold", 1.0, 3.0),
            ("cascade_decay", 3, 10),
        ]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        required = ["oi_vel_5", "funding_zscore", "taker_delta", "volume"]
        if not self._check(df, required):
            return pd.Series(0.0, index=df.index, name=self.name)

        funding = df["funding_zscore"]
        oi_vel = df["oi_vel_5"]
        taker = df["taker_delta"]
        vol_avg = df["volume"].rolling(50, min_periods=5).mean().replace(0, 1)
        vol_ratio = df["volume"] / vol_avg

        # Normalize taker delta as z-score for comparable thresholding
        taker_std = taker.rolling(50, min_periods=5).std().replace(0, np.nan).ffill().fillna(1.0)
        taker_z = np.abs(taker) / taker_std

        # Soft scores [0, 1] for each cascade condition
        funding_score = self._sigmoid(np.abs(funding), center=p.funding_extreme, sharpness=p.funding_sharpness)
        oi_score = self._sigmoid(-oi_vel, center=p.oi_extreme_threshold, sharpness=p.oi_sharpness)
        taker_score = self._sigmoid(taker_z, center=p.taker_threshold, sharpness=p.taker_sharpness)
        volume_score = self._sigmoid(vol_ratio, center=p.volume_surge_threshold, sharpness=p.volume_sharpness)

        # Cascade probability: geometric mean (requires ALL conditions elevated)
        cascade_prob = (funding_score * oi_score * taker_score * volume_score) ** 0.25

        # Direction: fade the crowded side
        # Positive funding = longs crowded, cascade = bearish
        # Negative funding = shorts crowded, cascade = bullish
        direction = -np.tanh(funding / p.funding_extreme)

        # Intensity: proportional to funding extreme
        intensity = np.clip(np.abs(funding) / (p.funding_extreme * 1.5), 0.5, 1.0)

        raw = direction * cascade_prob * intensity

        # Exponential decay: signal persists briefly after trigger
        raw = raw.ewm(span=p.cascade_decay, min_periods=1).mean()

        return self._clip(raw, df)
