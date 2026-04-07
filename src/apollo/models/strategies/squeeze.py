"""Microstructure Squeeze -- Graduated stress scoring for crowded trade fading."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from apollo.models.strategies.base import BaseStrategy, StrategyParams


@dataclass
class SqueezeParams(StrategyParams):
    stress_threshold: float = 1.2
    stress_sharpness: float = 3.0
    oi_drop_threshold: float = -0.001
    oi_sharpness: float = 500.0
    taker_scale: float = 5000.0
    fragility_cap: float = 3.0


class SqueezeStrategy(BaseStrategy):
    name = "squeeze"

    def _default_params(self):
        return SqueezeParams()

    def param_space(self):
        return [
            ("stress_threshold", 0.8, 2.0),
            ("stress_sharpness", 1.5, 6.0),
            ("oi_drop_threshold", -0.005, 0.0),
            ("oi_sharpness", 200.0, 1000.0),
            ("taker_scale", 2000.0, 10000.0),
        ]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        required = ["oi_vel_5", "taker_delta_ema_20", "basis_zscore_100", "funding_zscore"]
        if not self._check(df, required):
            return pd.Series(0.0, index=df.index, name=self.name)

        fz = df["funding_zscore"]
        bz = df["basis_zscore_100"]

        # Soft stress scores [0, 1] via sigmoid
        funding_stress = self._sigmoid(np.abs(fz), center=p.stress_threshold, sharpness=p.stress_sharpness)
        basis_stress = self._sigmoid(np.abs(bz), center=p.stress_threshold, sharpness=p.stress_sharpness)

        # OI velocity: negative = liquidations happening
        oi_stress = self._sigmoid(-df["oi_vel_5"], center=-p.oi_drop_threshold, sharpness=p.oi_sharpness)

        # Composite stress: geometric mean (all must be elevated, not binary AND)
        stress_magnitude = (funding_stress * basis_stress * oi_stress) ** (1.0 / 3.0)

        # Direction from taker imbalance: continuous score
        taker_dir = np.tanh(df["taker_delta_ema_20"] / p.taker_scale)
        # Fade the crowded side: continuous direction, not ternary
        crowd_dir = -np.tanh(fz * 2)
        direction = taker_dir * crowd_dir

        # Fragility boost from futures/spot volume ratio
        fragility = 1.0
        if "vol_fut_spot_ratio_ema" in df.columns:
            fsr = df["vol_fut_spot_ratio_ema"]
            fsr_mean = fsr.rolling(100, min_periods=10).mean()
            fragility = np.clip(fsr / fsr_mean.replace(0, 1), 1.0, p.fragility_cap)

        raw = direction * stress_magnitude * fragility
        return self._clip(raw, df)
