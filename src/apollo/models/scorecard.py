"""
Strategy Scorecard -- Historical Performance Tracking
======================================================
Evaluates each strategy's predictive accuracy per HMM regime by looking
at historical signal-vs-forward-return alignment.

Produces per-strategy, per-regime confidence scores that the AI brain
can use to weight predictions more intelligently.

Key metrics computed:
  - Rolling IC (Information Coefficient): rank correlation of signal vs return
  - Rolling Hit Rate: % of bars where signal direction matched return direction
  - Regime-Conditional IC: IC computed only during each HMM regime
  - Signal Persistence: how consistent the signal direction is over time
  - Cross-Strategy Agreement: how many strategies agree at each bar
  - Confidence Score: composite of all above, per strategy per bar
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from apollo.models.strategies import ALL_STRATEGY_NAMES

logger = logging.getLogger("models.scorecard")


@dataclass
class ScorecardConfig:
    """Configuration for the strategy scorecard."""
    ic_window: int = 168           # rolling IC window (1 week of hourly)
    hit_rate_window: int = 168     # rolling hit rate window
    persistence_window: int = 12   # bars for signal persistence check
    forward_horizon: int = 5       # forward return horizon (bars)
    min_periods: int = 30          # minimum periods for rolling stats
    decay_alpha: float = 0.02     # EWM decay for regime-conditional stats


class StrategyScorecard:
    """
    Computes historical performance metrics for each strategy.

    This module is NOT used during signal generation (no lookahead).
    It evaluates past performance to produce confidence metadata
    that the AI brain can consume at inference time.

    All forward returns are shifted by 1 bar to prevent lookahead:
    signal at bar t is compared to return from t+1 to t+1+horizon.
    """

    def __init__(self, config: ScorecardConfig = None):
        self.config = config or ScorecardConfig()

    def compute(
        self,
        signals_df: pd.DataFrame,
        returns: pd.Series,
        hmm_states: pd.Series = None,
        hmm_probs_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Compute the full scorecard.

        Args:
            signals_df: DataFrame with columns = strategy names, values = signals
            returns: Series of log returns (same index as signals_df)
            hmm_states: Series of integer HMM state labels (optional)
            hmm_probs_df: DataFrame of HMM state probabilities (optional)

        Returns:
            DataFrame with enriched metadata columns:
              - {strat}_ic: rolling IC per strategy
              - {strat}_hit_rate: rolling hit rate per strategy
              - {strat}_persistence: signal direction consistency
              - {strat}_regime_ic: IC conditioned on current regime
              - {strat}_confidence: composite confidence score
              - cross_agreement: fraction of strategies agreeing on direction
              - cross_conviction: average absolute signal strength
              - dominant_direction: sign of majority vote
        """
        cfg = self.config
        available = [s for s in ALL_STRATEGY_NAMES if s in signals_df.columns]

        # Forward returns: shift by 1 to prevent lookahead
        fwd_ret = returns.shift(-cfg.forward_horizon)

        result = pd.DataFrame(index=signals_df.index)

        # -----------------------------------------------------------
        # Per-Strategy Metrics
        # -----------------------------------------------------------
        for strat in available:
            sig = signals_df[strat]

            # 1. Rolling IC (Spearman rank correlation proxy via Pearson of ranks)
            sig_rank = sig.rolling(cfg.ic_window, min_periods=cfg.min_periods).apply(
                lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False
            )
            ret_rank = fwd_ret.rolling(cfg.ic_window, min_periods=cfg.min_periods).apply(
                lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False
            )
            # Approximate rolling IC via product-moment of ranks
            result[f"{strat}_ic"] = self._rolling_ic(sig, fwd_ret, cfg.ic_window, cfg.min_periods)

            # 2. Rolling Hit Rate: did signal direction match return direction?
            correct = (np.sign(sig) == np.sign(fwd_ret)).astype(float)
            # Exclude zero signals from hit rate (they're not predictions)
            active = (sig != 0).astype(float)
            active_correct = correct * active
            result[f"{strat}_hit_rate"] = (
                active_correct.rolling(cfg.hit_rate_window, min_periods=cfg.min_periods).sum() /
                active.rolling(cfg.hit_rate_window, min_periods=cfg.min_periods).sum().replace(0, np.nan)
            ).fillna(0.5)  # 0.5 = no information

            # 3. Signal Persistence: consistency of direction over recent window
            sign_changes = np.abs(np.sign(sig).diff()).fillna(0)
            persistence_score = 1.0 - sign_changes.rolling(
                cfg.persistence_window, min_periods=3
            ).mean()
            result[f"{strat}_persistence"] = persistence_score.fillna(0.5)

            # 4. Signal Magnitude Trend: is the strategy getting more or less convicted?
            abs_sig = np.abs(sig)
            mag_trend = abs_sig.ewm(span=cfg.persistence_window).mean() - \
                        abs_sig.ewm(span=cfg.persistence_window * 3).mean()
            result[f"{strat}_momentum"] = np.tanh(mag_trend * 10)

            # 5. Regime-Conditional IC (if HMM states available)
            if hmm_states is not None:
                regime_ic = self._regime_conditional_ic(
                    sig, fwd_ret, hmm_states, cfg.ic_window, cfg.min_periods, cfg.decay_alpha
                )
                result[f"{strat}_regime_ic"] = regime_ic
            else:
                result[f"{strat}_regime_ic"] = result[f"{strat}_ic"]

            # 6. Composite Confidence Score
            ic_score = np.clip(result[f"{strat}_ic"] * 5, -1, 1)  # amplify IC
            hr_score = (result[f"{strat}_hit_rate"] - 0.5) * 2   # center at 0
            regime_score = np.clip(result[f"{strat}_regime_ic"] * 5, -1, 1)
            pers_score = result[f"{strat}_persistence"]
            mom_score = result[f"{strat}_momentum"]

            # Weighted combination
            result[f"{strat}_confidence"] = np.tanh(
                ic_score * 0.3 + hr_score * 0.25 + regime_score * 0.25 +
                pers_score * 0.1 + mom_score * 0.1
            )

        # -----------------------------------------------------------
        # Cross-Strategy Agreement Metrics
        # -----------------------------------------------------------
        signs = np.sign(signals_df[available])
        abs_sigs = np.abs(signals_df[available])

        # Fraction of active strategies agreeing on direction
        n_long = (signs > 0).sum(axis=1)
        n_short = (signs < 0).sum(axis=1)
        n_active = (signs != 0).sum(axis=1).replace(0, 1)
        result["cross_agreement"] = np.maximum(n_long, n_short) / n_active

        # Average conviction (absolute signal strength)
        result["cross_conviction"] = abs_sigs.mean(axis=1)

        # Dominant direction: majority vote weighted by signal strength
        weighted_vote = signals_df[available].sum(axis=1)
        result["dominant_direction"] = np.tanh(weighted_vote)

        # Signal dispersion: high = strategies disagree, low = consensus
        result["signal_dispersion"] = signs.std(axis=1).fillna(0)

        # -----------------------------------------------------------
        # Cleanup
        # -----------------------------------------------------------
        result = result.fillna(0.0).replace([np.inf, -np.inf], 0.0)

        logger.info(
            "Scorecard computed: %d strategies, %d bars, %d columns",
            len(available), len(result), len(result.columns)
        )
        return result

    @staticmethod
    def _rolling_ic(signal: pd.Series, returns: pd.Series, window: int, min_periods: int) -> pd.Series:
        """Compute rolling Pearson correlation as IC proxy (fast)."""
        # Center both series
        sig_dm = signal - signal.rolling(window, min_periods=min_periods).mean()
        ret_dm = returns - returns.rolling(window, min_periods=min_periods).mean()

        # Rolling covariance / (std * std)
        cov = (sig_dm * ret_dm).rolling(window, min_periods=min_periods).mean()
        sig_std = sig_dm.rolling(window, min_periods=min_periods).std()
        ret_std = ret_dm.rolling(window, min_periods=min_periods).std()

        ic = cov / (sig_std * ret_std).replace(0, np.nan)
        return ic.fillna(0).clip(-1, 1)

    @staticmethod
    def _regime_conditional_ic(
        signal: pd.Series,
        returns: pd.Series,
        hmm_states: pd.Series,
        window: int,
        min_periods: int,
        decay_alpha: float,
    ) -> pd.Series:
        """
        Compute IC conditioned on the current HMM regime.
        Uses EWM within each regime block for smooth transitions.
        """
        unique_states = hmm_states.dropna().unique()
        regime_ic = pd.Series(0.0, index=signal.index)

        for state in unique_states:
            mask = (hmm_states == state)
            if mask.sum() < min_periods:
                continue

            # Extract regime-only data
            sig_regime = signal.where(mask, np.nan)
            ret_regime = returns.where(mask, np.nan)

            # Compute IC within this regime via EWM
            sig_dm = sig_regime - sig_regime.ewm(span=window, min_periods=min_periods // 3).mean()
            ret_dm = ret_regime - ret_regime.ewm(span=window, min_periods=min_periods // 3).mean()

            cov = (sig_dm * ret_dm).ewm(span=window, min_periods=min_periods // 3).mean()
            sig_std = sig_dm.ewm(span=window, min_periods=min_periods // 3).std()
            ret_std = ret_dm.ewm(span=window, min_periods=min_periods // 3).std()

            ic = cov / (sig_std * ret_std).replace(0, np.nan)
            ic = ic.clip(-1, 1).fillna(0)

            # Write regime IC where this regime is active
            regime_ic = regime_ic.where(~mask, ic)

        return regime_ic

    def summary(self, scorecard_df: pd.DataFrame) -> dict:
        """
        Extract summary statistics from the scorecard for the AI brain.
        Returns a dict of per-strategy performance summaries.
        """
        available = [
            c.replace("_confidence", "")
            for c in scorecard_df.columns if c.endswith("_confidence")
        ]

        summary = {}
        for strat in available:
            recent = scorecard_df.iloc[-24:]  # last 24 bars
            summary[strat] = {
                "avg_ic": float(scorecard_df[f"{strat}_ic"].iloc[-168:].mean()),
                "recent_ic": float(recent[f"{strat}_ic"].mean()),
                "avg_hit_rate": float(scorecard_df[f"{strat}_hit_rate"].iloc[-168:].mean()),
                "recent_hit_rate": float(recent[f"{strat}_hit_rate"].mean()),
                "regime_ic": float(recent[f"{strat}_regime_ic"].mean()),
                "persistence": float(recent[f"{strat}_persistence"].mean()),
                "momentum": float(recent[f"{strat}_momentum"].mean()),
                "confidence": float(recent[f"{strat}_confidence"].mean()),
            }

        # Cross-strategy
        recent = scorecard_df.iloc[-24:]
        summary["_cross"] = {
            "agreement": float(recent["cross_agreement"].mean()),
            "conviction": float(recent["cross_conviction"].mean()),
            "dominant_direction": float(recent["dominant_direction"].mean()),
            "dispersion": float(recent["signal_dispersion"].mean()),
        }

        return summary
