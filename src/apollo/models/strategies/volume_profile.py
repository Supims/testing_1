"""
Volume Profile -- Institutional accumulation/distribution detection
====================================================================
Detects institutional accumulation/distribution zones using VWAP band
reversion + volume skew + OBV/CMF confirmation.
All components produce continuous signals.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from apollo.models.strategies.base import BaseStrategy, StrategyParams


@dataclass
class VolumeProfileParams(StrategyParams):
    vwap_extreme: float = 1.5         # VWAP distance % for extreme
    vwap_sharpness: float = 3.0       # sigmoid sharpness for VWAP activation
    obv_ema_smooth: int = 20          # OBV EMA period
    obv_zscore_window: int = 50       # window for OBV z-score normalization
    cmf_threshold: float = 0.1       # CMF confirmation threshold
    taker_weight: float = 0.3        # Weight for taker ratio
    vwap_weight: float = 0.4         # Weight for VWAP reversion
    flow_weight: float = 0.3         # Weight for flow confirmation


class VolumeProfileStrategy(BaseStrategy):
    name = "vol_profile"

    def _default_params(self):
        return VolumeProfileParams()

    def param_space(self):
        return [
            ("vwap_extreme", 1.0, 3.0),
            ("vwap_sharpness", 1.5, 6.0),
            ("cmf_threshold", 0.05, 0.2),
            ("taker_weight", 0.1, 0.5),
            ("vwap_weight", 0.2, 0.6),
            ("flow_weight", 0.1, 0.5),
        ]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        required = ["vwap_dist", "cmf"]
        if not self._check(df, required):
            return pd.Series(0.0, index=df.index, name=self.name)

        # 1. VWAP reversion: fade extreme distances
        vwap_signal = -np.tanh(df["vwap_dist"] / 100 * 50)
        # Soft activation: sigmoid ramp around vwap_extreme (NOT binary)
        vwap_activation = self._sigmoid(
            np.abs(df["vwap_dist"]), center=p.vwap_extreme, sharpness=p.vwap_sharpness
        )

        # 2. Flow confirmation: CMF + taker ratio (already continuous)
        cmf_signal = np.tanh(df["cmf"] / 0.15)

        taker_signal = 0.0
        if "taker_ratio" in df.columns:
            taker_signal = np.tanh((df["taker_ratio"] - 0.5) * 4)

        # 3. OBV trend: continuous z-score normalization (NOT np.sign)
        obv_signal = 0.0
        if "obv_ema_20" in df.columns:
            obv_trend = df["obv_ema_20"].diff(10)
            obv_std = obv_trend.rolling(p.obv_zscore_window, min_periods=5).std()
            obv_std = obv_std.replace(0, np.nan).ffill().fillna(1.0)
            obv_signal = np.tanh(obv_trend / obv_std)

        # Composite
        flow = cmf_signal * (1 - p.taker_weight) + taker_signal * p.taker_weight
        composite = (
            vwap_signal * vwap_activation * p.vwap_weight +
            flow * p.flow_weight +
            obv_signal * (1 - p.vwap_weight - p.flow_weight)
        )

        return self._clip(composite, df)
