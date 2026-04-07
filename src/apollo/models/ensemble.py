"""
Regime-Weighted Ensemble
=========================
Two modes:
  StaticEnsemble  -- Domain-knowledge priors per HMM regime (cold start)
  AdaptiveEnsemble -- Online learning from rolling PnL (after warmup)

Output: composite signal in [-1, +1] via tanh compression.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from apollo.models.strategies import ALL_STRATEGY_NAMES

logger = logging.getLogger("models.ensemble")


# =========================================================================
# Static Regime Ensemble (domain-knowledge priors)
# =========================================================================

class StaticEnsemble:
    """
    Hardcoded prior weights per HMM regime.
    Negative weights invert the strategy signal (e.g., MR = -0.2 in trending
    means we weakly counter the mean-reversion signal).
    """

    WEIGHT_MATRIX = {
        "High Volatility (Trending)": {
            "trend": 1.0, "mean_reversion": -0.2, "squeeze": 0.3,
            "smart_money": 0.8, "basis_arb": 0.15, "breakout": 1.0,
            "funding_mom": 0.3, "oi_divergence": 0.6,
            "liq_cascade": 0.7, "vol_profile": 0.3,
        },
        "Low Volatility (Trending)": {
            "trend": 0.8, "mean_reversion": 0.2, "squeeze": 0.15,
            "smart_money": 0.5, "basis_arb": 0.3, "breakout": 0.5,
            "funding_mom": 0.4, "oi_divergence": 0.5,
            "liq_cascade": 0.2, "vol_profile": 0.4,
        },
        "High Volatility (Ranging)": {
            "trend": -0.1, "mean_reversion": 1.0, "squeeze": 0.8,
            "smart_money": -0.5, "basis_arb": 1.0, "breakout": -0.8,
            "funding_mom": 0.8, "oi_divergence": -0.3,
            "liq_cascade": 0.9, "vol_profile": 0.6,
        },
        "Low Volatility (Quiet Range)": {
            "trend": 0.1, "mean_reversion": 0.5, "squeeze": 0.15,
            "smart_money": 0.2, "basis_arb": 0.8, "breakout": 0.1,
            "funding_mom": 0.6, "oi_divergence": 0.3,
            "liq_cascade": 0.1, "vol_profile": 0.5,
        },
    }

    EQUAL_WEIGHTS = {s: 1.0 / len(ALL_STRATEGY_NAMES) for s in ALL_STRATEGY_NAMES}

    def __init__(self, label_map: dict[int, str]):
        """
        Args:
            label_map: {state_id: 'Label Name'} from RegimeDetector.label_map
        """
        self._label_map = label_map

    def _weights_for_state(self, state_id: int) -> dict[str, float]:
        label = self._label_map.get(state_id, "")
        for prefix, weights in self.WEIGHT_MATRIX.items():
            if label.startswith(prefix):
                return weights
        return dict(self.EQUAL_WEIGHTS)

    def compute(
        self, signals_df: pd.DataFrame, hmm_probs_df: pd.DataFrame
    ) -> pd.Series:
        """
        Probability-weighted sum of strategy signals per regime.

        Args:
            signals_df: DataFrame with columns matching strategy names.
            hmm_probs_df: DataFrame with 'hmm_prob_state_0', etc.

        Returns:
            Series in [-1, 1] via tanh compression.
        """
        available = [s for s in ALL_STRATEGY_NAMES if s in signals_df.columns]
        ensemble = pd.Series(0.0, index=signals_df.index)
        total_w = pd.Series(0.0, index=signals_df.index)
        n_states = len(self._label_map)

        # Check if any HMM probability columns exist
        has_probs = any(
            f"hmm_prob_state_{sid}" in hmm_probs_df.columns
            for sid in range(n_states)
        )

        if has_probs:
            for sid in range(n_states):
                prob_col = f"hmm_prob_state_{sid}"
                if prob_col not in hmm_probs_df.columns:
                    continue

                prob = hmm_probs_df[prob_col]
                weights = self._weights_for_state(sid)

                state_sum = pd.Series(0.0, index=signals_df.index)
                for strat in available:
                    state_sum += signals_df[strat] * weights.get(strat, 0)

                max_w = sum(abs(weights.get(s, 0)) for s in available)
                ensemble += state_sum * prob
                total_w += max_w * prob
        else:
            # Fallback: equal-weighted average when HMM is not fitted
            logger.warning("No HMM probabilities -- using equal-weighted ensemble")
            weights = dict(self.EQUAL_WEIGHTS)
            for strat in available:
                ensemble += signals_df[strat] * weights.get(strat, 0)
            total_w = sum(abs(weights.get(s, 0)) for s in available)

        return pd.Series(
            np.tanh(ensemble / (total_w if isinstance(total_w, (int, float)) else total_w.replace(0, 1))),
            index=signals_df.index,
            name="ensemble_signal",
        )

    def get_weight_matrix(self, n_states: int) -> pd.DataFrame:
        """Returns DataFrame of prior weights for each (state, strategy)."""
        rows = [self._weights_for_state(sid) for sid in range(n_states)]
        return pd.DataFrame(rows, columns=ALL_STRATEGY_NAMES)


# =========================================================================
# Adaptive Regime Ensemble (online learning)
# =========================================================================

class AdaptiveEnsemble:
    """
    Online learning ensemble with cold-start mitigation.
    Tracks per-strategy per-regime rolling PnL (EWM).
    Transitions from static priors to data-driven weights via decay.
    Only profitable strategies get weight (ReLU activation).
    """

    def __init__(
        self,
        performance_window: int = 168,
        warmup_decay_rate: float = 0.002,
        static_prior: StaticEnsemble = None,
    ):
        self._perf_window = performance_window
        self._decay_rate = warmup_decay_rate
        self._prior = static_prior

    def compute(
        self,
        signals_df: pd.DataFrame,
        hmm_probs_df: pd.DataFrame,
        returns: pd.Series,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """
        Compute adaptive ensemble signal.

        Returns:
            (ensemble_signal, dynamic_weights_df)
        """
        available = [s for s in ALL_STRATEGY_NAMES if s in signals_df.columns]
        n_strats = len(available)
        n_states = len([c for c in hmm_probs_df.columns if "hmm_prob_state_" in c])

        # Shift by 1 to prevent look-ahead in PnL attribution
        shifted_sig = signals_df[available].shift(1)
        shifted_prob = hmm_probs_df.shift(1)

        has_prior = self._prior is not None
        prior_matrix = None
        if has_prior:
            prior_matrix = self._prior.get_weight_matrix(n_states)

        # Warmup factor: 1.0 (full prior) -> 0.0 (full adaptive)
        t = np.arange(len(signals_df), dtype=float)
        warmup = np.exp(-self._decay_rate * t)

        weights_dict: dict[str, pd.Series] = {}
        ensemble = pd.Series(0.0, index=signals_df.index)

        for sid in range(n_states):
            prob_col = f"hmm_prob_state_{sid}"
            if prob_col not in hmm_probs_df.columns:
                continue

            past_prob = shifted_prob[prob_col]

            # Per-strategy PnL in this regime
            perf = pd.DataFrame(index=signals_df.index)
            for strat in available:
                pnl = shifted_sig[strat] * returns * past_prob
                perf[strat] = pnl.ewm(
                    span=self._perf_window,
                    min_periods=self._perf_window // 10,
                ).mean()

            # Adaptive: ReLU + normalize
            adaptive_w = perf.clip(lower=0.0)
            adaptive_sum = adaptive_w.sum(axis=1).replace(0, 1.0)
            adaptive_norm = adaptive_w.div(adaptive_sum, axis=0)

            # Cold-start blending with static priors
            if has_prior and prior_matrix is not None:
                prior_row = prior_matrix.iloc[sid][available]
                prior_relu = prior_row.clip(lower=0)
                prior_total = prior_relu.sum()
                prior_norm = (
                    prior_relu / prior_total
                    if prior_total > 0
                    else pd.Series(1.0 / n_strats, index=available)
                )

                for strat in available:
                    adaptive_norm[strat] = (
                        warmup * prior_norm[strat]
                        + (1.0 - warmup) * adaptive_norm[strat].fillna(prior_norm[strat])
                    )
            else:
                adaptive_norm = adaptive_norm.fillna(1.0 / n_strats)

            for strat in available:
                weights_dict[f"w_s{sid}_{strat}"] = adaptive_norm[strat]

            # State contribution
            cur_prob = hmm_probs_df[prob_col]
            state_contrib = pd.Series(0.0, index=signals_df.index)
            for strat in available:
                state_contrib += signals_df[strat] * adaptive_norm[strat]
            ensemble += state_contrib * cur_prob

        weights_df = pd.DataFrame(weights_dict, index=signals_df.index)
        final = pd.Series(
            np.tanh(ensemble), index=signals_df.index, name="ensemble_signal"
        )

        return final, weights_df
