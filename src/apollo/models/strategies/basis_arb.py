"""Basis Arbitrage -- Fade extreme futures premium/discount with soft activation."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from apollo.models.strategies.base import BaseStrategy, StrategyParams


@dataclass
class BasisArbParams(StrategyParams):
    activation_center: float = 1.5
    activation_sharpness: float = 3.0
    sensitivity: float = 1.5
    funding_boost: float = 0.3


class BasisArbStrategy(BaseStrategy):
    name = "basis_arb"

    def _default_params(self):
        return BasisArbParams()

    def param_space(self):
        return [
            ("activation_center", 1.0, 2.5),
            ("activation_sharpness", 1.5, 6.0),
            ("sensitivity", 1.0, 2.5),
            ("funding_boost", 0.1, 0.5),
        ]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        if "basis_zscore_100" not in df.columns:
            return pd.Series(0.0, index=df.index, name=self.name)

        bz = df["basis_zscore_100"]

        # Core: fade premium/discount (negative when overpriced)
        core = -np.tanh(bz / p.sensitivity)

        # Soft activation: ramps from 0 to 1 around activation_center
        activation = self._sigmoid(np.abs(bz), center=p.activation_center, sharpness=p.activation_sharpness)

        # Funding rate confirmation: boosts signal when funding agrees
        if "funding_zscore" in df.columns:
            fz = df["funding_zscore"]
            # If basis positive AND funding positive -> stronger short signal
            funding_agreement = np.tanh(bz * fz / 4)  # positive when same direction
            confirmation = 1.0 + funding_agreement * p.funding_boost
        else:
            confirmation = 1.0

        signal = core * activation * confirmation
        return self._clip(signal, df)
