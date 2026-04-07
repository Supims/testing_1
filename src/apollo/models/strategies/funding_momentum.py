"""Funding Rate Momentum -- Crowded trade fading with soft CVD confirmation."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from apollo.models.strategies.base import BaseStrategy, StrategyParams


@dataclass
class FundingMomParams(StrategyParams):
    zscore_entry: float = 1.0
    entry_sharpness: float = 3.0
    zscore_extreme: float = 2.5
    confirmation_window: int = 30


class FundingMomentumStrategy(BaseStrategy):
    name = "funding_mom"

    def _default_params(self):
        return FundingMomParams()

    def param_space(self):
        return [
            ("zscore_entry", 0.5, 2.0),
            ("entry_sharpness", 1.5, 6.0),
            ("zscore_extreme", 2.0, 3.5),
            ("confirmation_window", 15, 60),
        ]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        if not self._check(df, ["funding_zscore", "cvd_rolling_100"]):
            return pd.Series(0.0, index=df.index, name=self.name)

        fz = df["funding_zscore"]
        cvd_trend = df["cvd_rolling_100"].diff(p.confirmation_window)

        # Core signal: fade funding extremes (continuous)
        raw_signal = np.tanh(-fz / p.zscore_extreme)

        # Soft activation: ramps near zscore_entry
        activation = self._sigmoid(np.abs(fz), center=p.zscore_entry, sharpness=p.entry_sharpness)

        # Soft CVD confirmation: agreement boosts, disagreement dampens
        cvd_std = cvd_trend.rolling(50, min_periods=5).std().replace(0, np.nan).ffill().fillna(1.0)
        cvd_z = cvd_trend / cvd_std
        # Agreement: when raw_signal and cvd_z point same direction
        agreement = np.tanh(raw_signal * cvd_z * 2)
        confirmation = (agreement + 1) / 2  # [0, 1]: 0=disagree, 1=agree

        signal = raw_signal * activation * confirmation
        return self._clip(signal, df)
