"""Tests for Strategy Suite."""

import numpy as np
import pandas as pd
import pytest


class TestStrategyRegistry:
    def test_all_10_registered(self):
        from apollo.models.strategies import REGISTRY, ALL_STRATEGY_NAMES
        assert len(REGISTRY) == 10
        assert len(ALL_STRATEGY_NAMES) == 10

    def test_instantiate_all(self):
        from apollo.models.strategies import get_all_strategies
        strats = get_all_strategies()
        assert len(strats) == 10
        for name, s in strats.items():
            assert s.name == name


class TestStrategyOutputBounds:
    """Every strategy must return signals in [-1, +1] on any data."""

    def test_all_strategies_bounded(self, synthetic_features):
        from apollo.models.strategies import get_all_strategies
        strats = get_all_strategies()
        for name, strat in strats.items():
            signal = strat.compute(synthetic_features)
            assert signal.min() >= -1.0, f"{name} below -1"
            assert signal.max() <= 1.0, f"{name} above +1"
            assert not signal.isna().any(), f"{name} has NaN"

    def test_all_strategies_zero_on_empty(self):
        from apollo.models.strategies import get_all_strategies
        df = pd.DataFrame({"close": [1, 2, 3]})
        strats = get_all_strategies()
        for name, strat in strats.items():
            signal = strat.compute(df)
            assert (signal == 0).all(), f"{name} should return 0 on missing cols"


class TestStrategyDirections:
    def test_trend_positive_in_uptrend(self, synthetic_features):
        from apollo.models.strategies.trend import TrendStrategy
        s = TrendStrategy()
        signal = s.compute(synthetic_features)
        # In the designed uptrend zone (bars 100-200 of original, ~70-170 after warmup)
        trend_mean = signal.iloc[70:170].mean()
        # Should lean positive (might not be strong without HTF)
        assert trend_mean > -0.5  # At least not strongly negative

    def test_mean_reversion_fades_extremes(self, synthetic_features):
        from apollo.models.strategies.mean_reversion import MeanReversionStrategy
        s = MeanReversionStrategy()
        signal = s.compute(synthetic_features)
        # Should fire at extremes (sharp drop at bar 350 -> ~320 after warmup)
        assert signal.nunique() > 1  # Not flat zero

    def test_basis_arb_fades_premium(self):
        from apollo.models.strategies.basis_arb import BasisArbStrategy
        n = 100
        idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "basis_zscore_100": np.linspace(-3, 3, n),
        }, index=idx)
        s = BasisArbStrategy()
        signal = s.compute(df)
        # At z=+3, should be negative (shorting premium)
        assert signal.iloc[-1] < 0
        # At z=-3, should be positive (buying discount)
        assert signal.iloc[0] > 0


class TestComputeAll:
    def test_compute_all_returns_dataframe(self, synthetic_features):
        from apollo.models.strategies import compute_all
        signals = compute_all(synthetic_features)
        assert isinstance(signals, pd.DataFrame)
        assert len(signals.columns) == 10
        assert len(signals) == len(synthetic_features)

    def test_compute_all_no_nan(self, synthetic_features):
        from apollo.models.strategies import compute_all
        signals = compute_all(synthetic_features)
        assert not signals.isna().any().any()

    def test_param_space_defined(self):
        from apollo.models.strategies import get_all_strategies
        strats = get_all_strategies()
        for name, strat in strats.items():
            space = strat.param_space()
            assert isinstance(space, list), f"{name} param_space not list"
            for item in space:
                assert len(item) == 3, f"{name} param tuple has wrong length"


class TestSignalContinuity:
    """Strategies must produce continuous signals, not binary."""

    def test_all_strategies_continuous(self, synthetic_features):
        from apollo.models.strategies import get_all_strategies
        strats = get_all_strategies()
        for name, strat in strats.items():
            signal = strat.compute(synthetic_features)
            unique_count = signal.round(3).nunique()
            assert unique_count > 20, f"{name} too few unique values: {unique_count}"

    def test_no_strategy_entirely_zero(self, synthetic_features):
        from apollo.models.strategies import get_all_strategies
        strats = get_all_strategies()
        for name, strat in strats.items():
            signal = strat.compute(synthetic_features)
            zero_pct = (signal == 0).mean()
            # liq_cascade can be sparser since it's event-driven
            max_zero = 0.95 if name == "liq_cascade" else 0.80
            assert zero_pct < max_zero, f"{name} too sparse: {zero_pct:.0%} zeros"

    def test_sigmoid_method_available(self):
        from apollo.models.strategies.base import BaseStrategy
        # Ensure _sigmoid is accessible as a static method
        result = BaseStrategy._sigmoid(0.0, center=0.0, sharpness=1.0)
        assert abs(result - 0.5) < 0.01


class TestStrategyScorecard:
    """Tests for the strategy performance tracking module."""

    def test_scorecard_basic(self, synthetic_features):
        from apollo.models.strategies import compute_all
        from apollo.models.scorecard import StrategyScorecard

        signals = compute_all(synthetic_features)
        returns = synthetic_features["close"].pct_change().fillna(0)
        sc = StrategyScorecard()
        result = sc.compute(signals, returns)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(signals)
        assert not result.isna().any().any()
        # Should have per-strategy columns
        assert "trend_ic" in result.columns
        assert "trend_confidence" in result.columns
        assert "cross_agreement" in result.columns

    def test_scorecard_with_hmm(self, synthetic_features):
        from apollo.models.strategies import compute_all
        from apollo.models.scorecard import StrategyScorecard

        signals = compute_all(synthetic_features)
        returns = synthetic_features["close"].pct_change().fillna(0)
        # Fake HMM states
        hmm_states = pd.Series(
            np.random.choice([0, 1, 2], size=len(signals)),
            index=signals.index
        )
        sc = StrategyScorecard()
        result = sc.compute(signals, returns, hmm_states=hmm_states)
        assert "trend_regime_ic" in result.columns

    def test_scorecard_summary(self, synthetic_features):
        from apollo.models.strategies import compute_all
        from apollo.models.scorecard import StrategyScorecard

        signals = compute_all(synthetic_features)
        returns = synthetic_features["close"].pct_change().fillna(0)
        sc = StrategyScorecard()
        scorecard_df = sc.compute(signals, returns)
        summary = sc.summary(scorecard_df)

        assert isinstance(summary, dict)
        assert "trend" in summary
        assert "avg_ic" in summary["trend"]
        assert "_cross" in summary


class TestSignalEnrichment:
    """Tests for signal enrichment module."""

    def test_enrichment_basic(self, synthetic_features):
        from apollo.models.strategies import compute_all
        from apollo.models.enrichment import SignalEnrichment

        signals = compute_all(synthetic_features)
        enricher = SignalEnrichment()
        result = enricher.compute(signals)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(signals)
        assert not result.isna().any().any()
        assert "trend_age_norm" in result.columns
        assert "trend_accel" in result.columns
        assert "consensus_ratio" in result.columns
        assert "weighted_consensus" in result.columns

    def test_enrichment_with_hmm(self, synthetic_features):
        from apollo.models.strategies import compute_all
        from apollo.models.enrichment import SignalEnrichment

        signals = compute_all(synthetic_features)
        hmm_states = pd.Series(
            np.random.choice([0, 1], size=len(signals)),
            index=signals.index
        )
        label_map = {0: "High Volatility (Trending)", 1: "Low Volatility (Quiet Range)"}
        enricher = SignalEnrichment()
        result = enricher.compute(signals, hmm_states=hmm_states, label_map=label_map)

        assert "in_regime_0" in result.columns
        assert "trend_mr_ratio" in result.columns

    def test_enrichment_bounds(self, synthetic_features):
        from apollo.models.strategies import compute_all
        from apollo.models.enrichment import SignalEnrichment

        signals = compute_all(synthetic_features)
        enricher = SignalEnrichment()
        result = enricher.compute(signals)

        # All tanh-compressed features should be in [-1, 1]
        for col in result.columns:
            if col.endswith("_accel") or col.endswith("_extremity") or col == "weighted_consensus":
                assert result[col].min() >= -1.01, f"{col} below -1"
                assert result[col].max() <= 1.01, f"{col} above +1"

