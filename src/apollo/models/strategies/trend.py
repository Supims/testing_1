"""Trend Momentum -- MTF trend following with soft alignment filters."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from apollo.models.strategies.base import BaseStrategy, StrategyParams


@dataclass
class TrendParams(StrategyParams):
    adx_threshold: float = 20.0
    adx_scale: float = 25.0
    vwap_overextension: float = 1.5  # percentage
    alignment_penalty: float = 0.1
    volume_boost: float = 1.2
    volume_penalty: float = 0.5


class TrendStrategy(BaseStrategy):
    name = "trend"

    def _default_params(self):
        return TrendParams()

    def param_space(self):
        return [
            ("adx_threshold", 15.0, 30.0),
            ("adx_scale", 15.0, 40.0),
            ("vwap_overextension", 0.5, 3.0),
            ("alignment_penalty", 0.0, 0.3),
            ("volume_boost", 1.0, 1.5),
            ("volume_penalty", 0.3, 0.8),
        ]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        htf_macd = self._htf_col(df, "macd_hist")
        htf_adx = self._htf_col(df, "adx")

        if not self._check(df, ["macd_hist", "cmf", "vwap_dist", htf_macd, htf_adx]):
            return pd.Series(0.0, index=df.index, name=self.name)

        # Macro direction: continuous [-1, +1] via tanh of HTF MACD
        macro_vol = df[htf_macd].rolling(20).std().replace(0, np.nan).ffill().fillna(1.0)
        macro_dir = np.tanh(df[htf_macd] / macro_vol * 2)

        # Macro strength: soft ramp from 0 at adx_threshold to 1 at threshold+scale
        macro_str = self._sigmoid(df[htf_adx], center=p.adx_threshold, sharpness=0.15)

        # Micro signal: MACD histogram normalized by volatility
        vol_proxy = df["close"].rolling(20).std() / df["close"]
        vol_proxy = vol_proxy.replace(0, np.nan).ffill().fillna(0.001)
        micro = np.tanh(df["macd_hist"] / vol_proxy * 2)

        # Soft alignment: how much macro and micro agree [-1,1] -> [penalty, 1.0]
        agreement = np.tanh(macro_dir * micro * 3)
        align = (agreement + 1) / 2 * (1 - p.alignment_penalty) + p.alignment_penalty

        # Volume flow confirmation: soft agreement between micro and CMF
        cmf_norm = np.tanh(df["cmf"] * 5)
        flow_agreement = np.tanh(micro * cmf_norm * 3)
        vol_align = (flow_agreement + 1) / 2 * (p.volume_boost - p.volume_penalty) + p.volume_penalty

        # VWAP overextension: soft penalty instead of hard 0.2
        vwap_overext_long = self._sigmoid(df["vwap_dist"], center=p.vwap_overextension, sharpness=3.0)
        vwap_overext_short = self._sigmoid(-df["vwap_dist"], center=p.vwap_overextension, sharpness=3.0)
        vwap_pen = 1.0 - 0.8 * np.where(micro > 0, vwap_overext_long,
                                          np.where(micro < 0, vwap_overext_short, 0.0))

        raw = micro * macro_str * align * vol_align * vwap_pen
        return self._clip(raw, df)
