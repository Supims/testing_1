"""Smart Money Divergence -- Continuous CVD vs price divergence scoring."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from apollo.models.strategies.base import BaseStrategy, StrategyParams


@dataclass
class SmartMoneyParams(StrategyParams):
    macro_lookback: int = 5
    cvd_lookback: int = 20
    cvd_smooth: int = 10
    zscore_window: int = 50
    intensity_cap: float = 2.0
    base_conviction: float = 0.8


class SmartMoneyStrategy(BaseStrategy):
    name = "smart_money"

    def _default_params(self):
        return SmartMoneyParams()

    def param_space(self):
        return [
            ("macro_lookback", 3, 10),
            ("cvd_lookback", 10, 40),
            ("cvd_smooth", 5, 20),
            ("zscore_window", 30, 100),
            ("intensity_cap", 1.5, 3.0),
            ("base_conviction", 0.5, 1.0),
        ]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        htf_close = self._htf_col(df, "close")

        if not self._check(df, ["cvd_rolling_100", "volume", htf_close]):
            return pd.Series(0.0, index=df.index, name=self.name)

        # Price trend: smoothed diff -> z-score for normalization
        macro_trend = df[htf_close].diff(p.macro_lookback).rolling(3).mean()
        macro_std = macro_trend.rolling(p.zscore_window, min_periods=10).std().replace(0, np.nan).ffill().fillna(1.0)
        macro_z = macro_trend / macro_std

        # CVD trend: smoothed diff -> z-score
        cvd_trend = df["cvd_rolling_100"].diff(p.cvd_lookback).rolling(p.cvd_smooth).mean()
        cvd_std = cvd_trend.rolling(p.zscore_window, min_periods=10).std().replace(0, np.nan).ffill().fillna(1.0)
        cvd_z = cvd_trend / cvd_std

        # Divergence strength: product of magnitudes, capped
        # When macro and CVD disagree, that IS the signal
        div_strength = np.minimum(np.abs(macro_z), np.abs(cvd_z))
        div_strength = np.tanh(div_strength / 2)  # smooth cap [0, ~1]

        # Direction: follow smart money (CVD) -- it leads price
        # Agreement indicator: +1 when aligned, -1 when diverging
        agreement = np.tanh(macro_z * cvd_z)

        # Signal: when they agree -> weak continuation in CVD direction
        # When they disagree -> strong reversal in CVD direction
        cvd_dir = np.tanh(cvd_z)
        signal = cvd_dir * (1 - agreement * 0.5) * div_strength

        # Volume intensity modifier
        vol_ratio = df["volume"] / df["volume"].rolling(50, min_periods=5).mean()
        vol_int = np.clip(vol_ratio, 0.5, p.intensity_cap)
        vol_int = np.tanh(vol_int / p.intensity_cap * 2)  # smooth cap

        raw = signal * vol_int * p.base_conviction
        return self._clip(raw, df)
