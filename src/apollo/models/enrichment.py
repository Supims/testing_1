"""
Signal Enrichment -- Position context and meta-analytics
=========================================================
Enriches raw strategy signals with contextual metadata that helps
the AI brain make better decisions:

  1. Signal age: how long the current direction has persisted
  2. Signal acceleration: is conviction increasing or decreasing?
  3. Regime alignment: does the signal agree with the current regime's bias?
  4. Extremity score: is the signal at historical extremes?
  5. Consensus strength: how many strategies agree?

Output: a flat DataFrame of enrichment columns that can be appended to
the signal DataFrame for AI consumption.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from apollo.models.strategies import ALL_STRATEGY_NAMES

logger = logging.getLogger("models.enrichment")


class SignalEnrichment:
    """
    Adds positional context to raw strategy signals.
    No lookahead -- all computations use past/current data only.
    """

    def __init__(self, regime_bias: dict[str, float] = None):
        """
        Args:
            regime_bias: optional {regime_label: expected_direction}
                e.g. {"High Volatility (Trending)": 0.0,  # neutral
                       "High Volatility (Ranging)": 0.0,
                       ...}
        """
        self._regime_bias = regime_bias or {}

    def compute(
        self,
        signals_df: pd.DataFrame,
        hmm_states: pd.Series = None,
        label_map: dict[int, str] = None,
    ) -> pd.DataFrame:
        """
        Compute signal enrichment features.

        Args:
            signals_df: DataFrame of strategy signals
            hmm_states: Series of HMM state integers
            label_map: {state_id: label_string}

        Returns:
            DataFrame with enrichment columns
        """
        available = [s for s in ALL_STRATEGY_NAMES if s in signals_df.columns]
        result = pd.DataFrame(index=signals_df.index)

        for strat in available:
            sig = signals_df[strat]

            # 1. Signal Age: how many consecutive bars in the same direction
            direction = np.sign(sig)
            direction_changes = (direction != direction.shift(1)).astype(int)
            age_groups = direction_changes.cumsum()
            result[f"{strat}_age"] = age_groups.groupby(age_groups).cumcount() + 1
            # Normalize: tanh(age/24) -> [0, ~1] over 24 bars
            result[f"{strat}_age_norm"] = np.tanh(result[f"{strat}_age"] / 24)

            # 2. Signal Acceleration: is conviction rising or falling?
            abs_sig = np.abs(sig)
            fast_ema = abs_sig.ewm(span=6, min_periods=1).mean()
            slow_ema = abs_sig.ewm(span=24, min_periods=1).mean()
            result[f"{strat}_accel"] = np.tanh((fast_ema - slow_ema) * 10)

            # 3. Historical Extremity: percentile rank of current signal
            result[f"{strat}_extremity"] = sig.rolling(168, min_periods=20).apply(
                lambda x: (x.iloc[-1] > x[:-1]).mean() if len(x) > 1 else 0.5,
                raw=False
            ).fillna(0.5) * 2 - 1  # map [0,1] -> [-1,1]

            # 4. Signal Stability: recent stddev (low = stable, high = noisy)
            stability = 1.0 - np.tanh(sig.rolling(12, min_periods=3).std().fillna(0) * 5)
            result[f"{strat}_stability"] = stability

        # 5. Consensus features (across all strategies)
        signs = np.sign(signals_df[available])
        abs_sigs = np.abs(signals_df[available])

        # How many agree on long vs short
        n_long = (signs > 0).sum(axis=1)
        n_short = (signs < 0).sum(axis=1)
        result["consensus_ratio"] = (n_long - n_short) / len(available)
        result["consensus_strength"] = np.maximum(n_long, n_short) / len(available)

        # Weighted consensus: signals weighted by their absolute strength
        result["weighted_consensus"] = signals_df[available].sum(axis=1) / len(available)

        # 6. Regime alignment (if HMM available)
        if hmm_states is not None and label_map is not None:
            regime_labels = hmm_states.map(label_map).fillna("Unknown")
            # Encode regime as features
            for state_id, label in label_map.items():
                result[f"in_regime_{state_id}"] = (hmm_states == state_id).astype(float)

            # Is the consensus aligned with regime character?
            # Trending regimes: trend signal should dominate
            # Ranging regimes: MR signal should dominate
            if "trend" in available and "mean_reversion" in available:
                trend_sig = signals_df["trend"]
                mr_sig = signals_df["mean_reversion"]
                result["trend_mr_ratio"] = np.tanh((np.abs(trend_sig) - np.abs(mr_sig)) * 3)

        # Fill any NaN
        result = result.fillna(0.0).replace([np.inf, -np.inf], 0.0)

        logger.info(
            "Signal enrichment: %d strategies, %d bars, %d columns",
            len(available), len(result), len(result.columns)
        )
        return result
