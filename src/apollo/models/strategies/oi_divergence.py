"""OI Divergence -- Magnitude-proportional Open Interest vs Price analysis."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from apollo.models.strategies.base import BaseStrategy, StrategyParams


@dataclass
class OIDivergenceParams(StrategyParams):
    price_lookback: int = 20
    oi_lookback: int = 20
    zscore_window: int = 50
    vol_confirm_threshold: float = 1.0
    vol_sharpness: float = 3.0


class OIDivergenceStrategy(BaseStrategy):
    name = "oi_divergence"

    def _default_params(self):
        return OIDivergenceParams()

    def param_space(self):
        return [
            ("price_lookback", 10, 40),
            ("oi_lookback", 10, 40),
            ("zscore_window", 30, 100),
            ("vol_confirm_threshold", 0.5, 2.0),
            ("vol_sharpness", 1.5, 6.0),
        ]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        oi_col = "oi_vel_20" if "oi_vel_20" in df.columns else "oi_vel_5"
        if not self._check(df, ["close", oi_col, "volume"]):
            return pd.Series(0.0, index=df.index, name=self.name)

        # Price trend -> z-score
        price_trend = df["close"].pct_change(p.price_lookback)
        price_std = price_trend.rolling(p.zscore_window, min_periods=10).std().replace(0, np.nan).ffill().fillna(1.0)
        price_z = price_trend / price_std

        # OI trend -> z-score
        oi_trend = df[oi_col].rolling(p.oi_lookback, min_periods=3).mean()
        oi_std = oi_trend.rolling(p.zscore_window, min_periods=10).std().replace(0, np.nan).ffill().fillna(1.0)
        oi_z = oi_trend / oi_std

        # Agreement metric: +1 when aligned, -1 when diverging
        agreement = np.tanh(price_z * oi_z)

        # Divergence strength: how significant are both signals?
        strength = np.sqrt(np.clip(np.abs(price_z * oi_z), 0, None))
        strength = np.tanh(strength / 2)  # cap at ~1

        # Direction: follow OI conviction
        # - OI rising = new positions -> conviction in direction
        # - OI falling = closing positions -> fading
        oi_dir = np.tanh(oi_z)

        # When agreeing (continuation): follow OI direction with moderate strength
        # When diverging (reversal): follow OI direction with high strength
        signal = oi_dir * strength * (1 - agreement * 0.3)

        # Volume filter: soft gate around threshold
        vol_ratio = df["volume"] / df["volume"].rolling(50, min_periods=5).mean()
        vol_filter = self._sigmoid(vol_ratio, center=p.vol_confirm_threshold, sharpness=p.vol_sharpness)
        # Don't kill the signal, just scale it
        vol_mult = 0.3 + vol_filter * 0.7

        raw = signal * vol_mult
        return self._clip(raw, df)
