"""Leveraged Breakout -- Soft BB breakout + continuous OI/ADX confirmation."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from apollo.models.strategies.base import BaseStrategy, StrategyParams


@dataclass
class BreakoutParams(StrategyParams):
    bb_upper: float = 0.85
    bb_lower: float = 0.15
    bb_sharpness: float = 15.0
    oi_growth_min: float = 0.01
    oi_scale: float = 0.02
    adx_min: float = 20.0
    adx_sharpness: float = 0.2


class BreakoutStrategy(BaseStrategy):
    name = "breakout"

    def _default_params(self):
        return BreakoutParams()

    def param_space(self):
        return [
            ("bb_upper", 0.75, 0.95),
            ("bb_lower", 0.05, 0.25),
            ("bb_sharpness", 5.0, 25.0),
            ("oi_growth_min", 0.005, 0.03),
            ("oi_scale", 0.01, 0.05),
            ("adx_min", 15.0, 30.0),
            ("adx_sharpness", 0.1, 0.5),
        ]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        htf_adx = self._htf_col(df, "adx")
        oi_col = "oi_vel_20" if "oi_vel_20" in df.columns else "oi_vel_5"

        if not self._check(df, [oi_col, "bb_pct_b", htf_adx]):
            return pd.Series(0.0, index=df.index, name=self.name)

        # Soft BB breakout scores [0, 1]
        up_score = self._sigmoid(df["bb_pct_b"], center=p.bb_upper, sharpness=p.bb_sharpness)
        down_score = self._sigmoid(-df["bb_pct_b"] + p.bb_lower, center=0, sharpness=p.bb_sharpness)

        # Directional signal: up = long, down = short
        direction = up_score - down_score

        # OI confirmation: continuous ramp [0, 1], sharpness capped to prevent step-function
        oi_sharpness = min(1.0 / max(p.oi_scale, 0.001), 20.0)
        oi_conf = self._sigmoid(df[oi_col], center=p.oi_growth_min, sharpness=oi_sharpness)
        # Blend: half signal from direction, half boosted by OI
        oi_mult = 0.5 + oi_conf * 0.5

        # Futures leading indicator (if available)
        fut_leading = 1.0
        if "vol_fut_spot_ratio_ema" in df.columns:
            ratio = df["vol_fut_spot_ratio_ema"]
            ratio_avg = ratio.rolling(100, min_periods=10).mean()
            fut_leading = self._sigmoid(ratio, center=ratio_avg, sharpness=2.0) * 0.5 + 0.5

        # Macro trend strength: soft ADX gate
        macro = self._sigmoid(df[htf_adx], center=p.adx_min, sharpness=p.adx_sharpness)

        raw = direction * oi_mult * fut_leading * macro
        return self._clip(raw, df)
