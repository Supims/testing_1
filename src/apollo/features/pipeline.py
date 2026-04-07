"""
Feature Pipeline
================
Orchestrates all feature computation on a raw OHLCV DataFrame.
Takes raw data -> returns a feature-enriched DataFrame.

Responsibilities:
  - Call all technical indicator functions
  - Handle NaN from warmup periods (explicit drop, not silent)
  - Validate output schema (all expected columns present)
  - Return metadata about the computation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from apollo.features import technical as ta

logger = logging.getLogger("features.pipeline")


@dataclass
class FeatureMetadata:
    """Info about the feature computation output."""
    total_columns: int = 0
    warmup_rows_dropped: int = 0
    nan_columns: list[str] = field(default_factory=list)
    input_rows: int = 0
    output_rows: int = 0


class FeaturePipeline:
    """
    Computes all features on a raw DataFrame.

    Input requirements:
      - Index: DatetimeIndex (UTC)
      - Required columns: open, high, low, close, volume
      - Optional columns: taker_buy_volume, funding_rate,
                          open_interest, premium_index, spot_close,
                          spot_volume, basis

    Output:
      - Same index, all original + computed columns
      - Warmup rows dropped (first N rows with NaN from indicators)
    """

    def __init__(self, warmup_bars: int = 60):
        self._warmup = warmup_bars

    def compute(self, df: pd.DataFrame,
                is_train: bool = False) -> tuple[pd.DataFrame, FeatureMetadata]:
        """
        Run the full feature pipeline.

        Args:
            df: Raw OHLCV+ DataFrame
            is_train: If True, generate forward-return targets (contain lookahead)

        Returns:
            (feature_df, metadata)
        """
        meta = FeatureMetadata(input_rows=len(df))

        # Validate required columns
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        result = df.copy()

        # -- Returns (base for HMM/GARCH) --
        result["log_return"] = ta.log_return(df["close"])

        # -- Trend / Momentum --
        result["rsi"] = ta.rsi(df["close"])
        result["rsi_norm"] = ta.rsi_normalized(df["close"])

        macd_df = ta.macd(df["close"])
        for col in macd_df.columns:
            result[col] = macd_df[col]

        adx_df = ta.adx_full(df["high"], df["low"], df["close"])
        for col in adx_df.columns:
            result[col] = adx_df[col]

        result["cci"] = ta.cci(df["high"], df["low"], df["close"])
        result["williams_r"] = ta.williams_r(df["high"], df["low"], df["close"])

        stoch_df = ta.stochastic(df["high"], df["low"], df["close"])
        for col in stoch_df.columns:
            result[col] = stoch_df[col]

        ichi_df = ta.ichimoku(df["high"], df["low"])
        for col in ichi_df.columns:
            result[col] = ichi_df[col]

        # -- Autocorrelation (critical for HMM) --
        result["autocorr_w20_l1"] = ta.autocorrelation(result["log_return"])

        # -- Volatility --
        bb_df = ta.bollinger_bands(df["close"])
        for col in bb_df.columns:
            result[col] = bb_df[col]

        result["atr"] = ta.atr(df["high"], df["low"], df["close"])
        result["natr"] = ta.natr(df["high"], df["low"], df["close"])
        result["gk_vol"] = ta.garman_klass_vol(
            df["high"], df["low"], df["open"], df["close"]
        )
        result["parkinson_vol"] = ta.parkinson_vol(df["high"], df["low"])
        result["realized_vol_20"] = ta.realized_vol(df["close"], period=20)
        result["realized_vol_100"] = ta.realized_vol(df["close"], period=100)

        # -- Volume --
        result["obv"] = ta.obv(df["close"], df["volume"])
        result["obv_ema_20"] = ta.obv_ema(df["close"], df["volume"])
        result["cmf"] = ta.cmf(df["high"], df["low"], df["close"], df["volume"])
        result["vwap_dist"] = ta.vwap_distance(
            df["close"], df["high"], df["low"], df["volume"]
        )

        # -- Microstructure (only if columns available) --
        if "taker_buy_volume" in df.columns:
            result["taker_delta"] = ta.taker_delta(df["taker_buy_volume"], df["volume"])
            result["taker_delta_ema_20"] = ta.taker_delta_ema(
                df["taker_buy_volume"], df["volume"]
            )
            result["taker_ratio"] = ta.taker_ratio(df["taker_buy_volume"], df["volume"])
            result["cvd_rolling_24"] = ta.cvd_rolling(
                df["taker_buy_volume"], df["volume"], window=24
            )
            result["cvd_rolling_100"] = ta.cvd_rolling(
                df["taker_buy_volume"], df["volume"], window=100
            )

        if "funding_rate" in df.columns:
            result["funding_zscore"] = ta.funding_zscore(df["funding_rate"])

        if "premium_index" in df.columns:
            result["basis_zscore"] = ta.basis_zscore(df["premium_index"])

        # Compute basis from spot BEFORE z-scoring it
        if "spot_close" in df.columns and "close" in df.columns:
            result["basis"] = (df["close"] - df["spot_close"]) / df["spot_close"]

        # basis_zscore_100: prefer computed basis, fallback to raw column
        if "basis" in result.columns:
            result["basis_zscore_100"] = ta.basis_zscore(result["basis"], window=100)
        elif "basis" in df.columns:
            result["basis_zscore_100"] = ta.basis_zscore(df["basis"], window=100)

        if "open_interest" in df.columns:
            oi_df = ta.oi_velocity(df["open_interest"])
            for col in oi_df.columns:
                result[col] = oi_df[col]
            result["vol_oi_ratio"] = ta.vol_oi_ratio(df["volume"], df["open_interest"])

        if "spot_volume" in df.columns:
            sfv = ta.spot_fut_vol_ratio(df["volume"], df["spot_volume"])
            for col in sfv.columns:
                result[col] = sfv[col]

        # -- Forward targets (training only, CONTAIN LOOKAHEAD) --
        if is_train:
            targets = ta.generate_targets(df["close"], horizons=[1, 5, 12])
            for col in targets.columns:
                result[col] = targets[col]

        # -- Handle NaN from warmup --
        warmup_rows = min(self._warmup, len(result) - 1)
        result = result.iloc[warmup_rows:]
        meta.warmup_rows_dropped = warmup_rows

        # Forward-fill remaining NaN then track still-NaN cols
        result = result.ffill()
        nan_cols = result.columns[result.isna().any()].tolist()
        meta.nan_columns = nan_cols

        # Fill remaining NaN with 0 (safer than crashing in ML)
        result = result.fillna(0.0)

        # Replace inf/-inf
        result = result.replace([np.inf, -np.inf], 0.0)

        meta.total_columns = len(result.columns)
        meta.output_rows = len(result)

        return result, meta
