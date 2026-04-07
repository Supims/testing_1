"""Tests for Ensemble Engine."""

import numpy as np
import pandas as pd
import pytest


def _make_signals_and_probs(n=200):
    """Helper: fake strategy signals + HMM probs."""
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    np.random.seed(42)
    from apollo.models.strategies import ALL_STRATEGY_NAMES
    signals = pd.DataFrame(
        np.random.uniform(-0.5, 0.5, (n, len(ALL_STRATEGY_NAMES))),
        columns=ALL_STRATEGY_NAMES,
        index=idx,
    )
    probs = pd.DataFrame({
        "hmm_prob_state_0": np.random.dirichlet([2, 1, 1, 1], n)[:, 0],
        "hmm_prob_state_1": np.random.dirichlet([1, 2, 1, 1], n)[:, 1],
        "hmm_prob_state_2": np.random.dirichlet([1, 1, 2, 1], n)[:, 2],
        "hmm_prob_state_3": np.random.dirichlet([1, 1, 1, 2], n)[:, 3],
    }, index=idx)
    return signals, probs


class TestStaticEnsemble:
    def test_output_range(self):
        from apollo.models.ensemble import StaticEnsemble
        label_map = {
            0: "High Volatility (Trending)",
            1: "Low Volatility (Trending)",
            2: "High Volatility (Ranging)",
            3: "Low Volatility (Quiet Range)",
        }
        ens = StaticEnsemble(label_map)
        signals, probs = _make_signals_and_probs()
        result = ens.compute(signals, probs)

        assert result.min() >= -1.0
        assert result.max() <= 1.0
        assert not result.isna().any()

    def test_cold_start_uses_priors(self):
        from apollo.models.ensemble import StaticEnsemble
        label_map = {0: "High Volatility (Trending)", 1: "Low Volatility (Quiet Range)",
                     2: "High Volatility (Ranging)", 3: "Low Volatility (Trending)"}
        ens = StaticEnsemble(label_map)

        # Verify weight matrix exists for all regimes
        matrix = ens.get_weight_matrix(4)
        assert matrix.shape == (4, 10)

    def test_regime_switch_changes_signal(self):
        from apollo.models.ensemble import StaticEnsemble
        from apollo.models.strategies import ALL_STRATEGY_NAMES
        label_map = {0: "High Volatility (Trending)", 1: "Low Volatility (Quiet Range)",
                     2: "High Volatility (Ranging)", 3: "Low Volatility (Trending)"}
        ens = StaticEnsemble(label_map)

        n = 50
        idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
        signals = pd.DataFrame(0.5, index=idx, columns=ALL_STRATEGY_NAMES)

        # 100% in trending regime
        probs_trending = pd.DataFrame(0.0, index=idx, columns=[f"hmm_prob_state_{i}" for i in range(4)])
        probs_trending["hmm_prob_state_0"] = 1.0

        # 100% in ranging regime
        probs_ranging = probs_trending.copy()
        probs_ranging["hmm_prob_state_0"] = 0.0
        probs_ranging["hmm_prob_state_2"] = 1.0

        sig_trend = ens.compute(signals, probs_trending)
        sig_range = ens.compute(signals, probs_ranging)

        # Different weights -> different signals
        assert not np.allclose(sig_trend.values, sig_range.values)


class TestAdaptiveEnsemble:
    def test_output_range(self):
        from apollo.models.ensemble import AdaptiveEnsemble, StaticEnsemble
        label_map = {0: "High Volatility (Trending)", 1: "Low Volatility (Trending)",
                     2: "High Volatility (Ranging)", 3: "Low Volatility (Quiet Range)"}
        prior = StaticEnsemble(label_map)
        adaptive = AdaptiveEnsemble(static_prior=prior)

        signals, probs = _make_signals_and_probs()
        returns = pd.Series(np.random.normal(0, 0.01, len(signals)), index=signals.index)

        result, weights = adaptive.compute(signals, probs, returns)

        assert result.min() >= -1.0
        assert result.max() <= 1.0
        assert not weights.empty
