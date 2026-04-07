"""
Multi-Timeframe Alignment
=========================
Resamples LTF data to HTF, computes features on the HTF, then projects
back onto the LTF index with a shift(1) protection to prevent lookahead.

The key guarantee:
  At any bar on the LTF, HTF features reflect ONLY completed HTF bars.
  The current (incomplete) HTF bar is never used.
"""

from __future__ import annotations

import logging

import pandas as pd

from apollo.features.pipeline import FeaturePipeline

logger = logging.getLogger("features.mtf")

# Resampling map: which columns to aggregate and how
OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}

# Optional columns and their aggregation
OPTIONAL_AGG = {
    "taker_buy_volume": "sum",
    "taker_sell_volume": "sum",
    "trades": "sum",
}


class MTFAligner:
    """
    Multi-timeframe feature alignment with anti-lookahead.

    Usage:
        aligner = MTFAligner(htf="4h")
        enriched = aligner.align(ltf_df)
        # enriched now has columns like "htf_rsi", "htf_macd", etc.
    """

    def __init__(self, htf: str = "4h", prefix: str = "htf_"):
        self._htf = htf
        self._prefix = prefix
        self._pipeline = FeaturePipeline(warmup_bars=20)

    def align(self, ltf_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute HTF features and project onto LTF index.

        Steps:
          1. Resample LTF -> HTF
          2. Compute features on HTF
          3. Shift HTF features by 1 bar (anti-lookahead)
          4. Merge onto LTF using merge_asof(direction='backward')

        Args:
            ltf_df: LTF DataFrame with DatetimeIndex and OHLCV columns

        Returns:
            LTF DataFrame with added HTF feature columns (prefixed)
        """
        if ltf_df.empty or len(ltf_df) < 50:
            logger.warning("Not enough LTF data for MTF alignment (%d rows)", len(ltf_df))
            return ltf_df

        # Step 1: Resample to HTF
        agg = {k: v for k, v in OHLCV_AGG.items() if k in ltf_df.columns}
        for col, func in OPTIONAL_AGG.items():
            if col in ltf_df.columns:
                agg[col] = func

        htf_df = ltf_df.resample(self._htf).agg(agg).dropna(subset=["close"])

        if len(htf_df) < 30:
            logger.warning("Not enough HTF bars for meaningful features (%d)", len(htf_df))
            return ltf_df

        # Step 2: Compute features on HTF
        htf_features, _ = self._pipeline.compute(htf_df)

        # Step 3: SHIFT by 1 to prevent lookahead
        # This ensures we only use COMPLETED HTF bars
        htf_features = htf_features.shift(1)
        htf_features = htf_features.dropna(how="all")

        # Step 4: Select only computed feature columns (not original OHLCV)
        original_cols = set(ltf_df.columns)
        feature_cols = [c for c in htf_features.columns if c not in original_cols]

        if not feature_cols:
            return ltf_df

        htf_subset = htf_features[feature_cols].copy()
        htf_subset.columns = [f"{self._prefix}{c}" for c in htf_subset.columns]

        # Step 5: Merge onto LTF using backward merge (anti-lookahead)
        result = pd.merge_asof(
            ltf_df,
            htf_subset,
            left_index=True,
            right_index=True,
            direction="backward",
        )

        # Forward-fill HTF values (they stay constant within each HTF bar)
        htf_cols = [c for c in result.columns if c.startswith(self._prefix)]
        result[htf_cols] = result[htf_cols].ffill()
        result[htf_cols] = result[htf_cols].fillna(0.0)

        logger.debug(
            "MTF aligned: %d LTF rows, %d HTF bars, %d feature columns added",
            len(result), len(htf_features), len(htf_cols),
        )

        return result
