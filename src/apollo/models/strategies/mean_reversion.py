"""Mean Reversion -- Fade extremes from BB/VWAP/CCI/WillR with soft activation."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from apollo.models.strategies.base import BaseStrategy, StrategyParams


@dataclass
class MeanReversionParams(StrategyParams):
    activation_threshold: float = 0.35
    activation_sharpness: float = 8.0
    signal_amplification: float = 2.0
    vwap_sensitivity: float = 80.0
    bw_expansion_limit: float = 1.2
    bw_sharpness: float = 5.0
    bb_weight: float = 0.40
    vwap_weight: float = 0.30
    cci_weight: float = 0.15
    willr_weight: float = 0.15


class MeanReversionStrategy(BaseStrategy):
    name = "mean_reversion"

    def _default_params(self):
        return MeanReversionParams()

    def param_space(self):
        return [
            ("activation_threshold", 0.2, 0.6),
            ("activation_sharpness", 3.0, 15.0),
            ("signal_amplification", 1.0, 3.0),
            ("vwap_sensitivity", 40.0, 120.0),
            ("bw_expansion_limit", 1.05, 1.5),
            ("bw_sharpness", 2.0, 10.0),
        ]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        htf_bw = self._htf_col(df, "bb_bandwidth")

        if not self._check(df, ["bb_pct_b", "cci", "williams_r", "vwap_dist", "bb_bandwidth", htf_bw]):
            return pd.Series(0.0, index=df.index, name=self.name)

        # Soft macro safety: bandwidth expansion penalty [1.0 -> 0.0]
        expansion = df[htf_bw] / df[htf_bw].rolling(50, min_periods=5).mean()
        macro_safe = 1.0 - self._sigmoid(expansion, center=p.bw_expansion_limit, sharpness=p.bw_sharpness)

        # Component signals (all continuous)
        bb_fade = (0.5 - df["bb_pct_b"]) * 2  # [-1, +1]
        vwap_fade = -np.tanh(df["vwap_dist"] / 100 * p.vwap_sensitivity)
        cci_fade = -np.tanh(df["cci"] / 150)  # smooth, no clip
        willr_fade = -np.tanh((df["williams_r"] + 50) / 30)

        # Weighted composite
        raw = (bb_fade * p.bb_weight + vwap_fade * p.vwap_weight +
               cci_fade * p.cci_weight + willr_fade * p.willr_weight)

        # Soft activation: sigmoid ramp around threshold (no hard gate)
        activation = self._sigmoid(np.abs(raw), center=p.activation_threshold, sharpness=p.activation_sharpness)

        signal = np.tanh(raw * p.signal_amplification) * activation * macro_safe
        return self._clip(signal, df)
