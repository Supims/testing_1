"""Tests for Risk Dashboard."""

import numpy as np
import pytest


def _make_paths(price=50000, n=500, horizon=24, seed=42):
    """Synthetic MC paths for testing."""
    np.random.seed(seed)
    returns = np.random.normal(0.001, 0.01, (n, horizon))
    paths = price * np.exp(np.cumsum(returns, axis=1))
    return paths


class TestRiskProfile:
    def test_profile_structure(self):
        from apollo.execution.risk import RiskDashboard
        rd = RiskDashboard()
        paths = _make_paths()
        profile = rd.compute_profile(paths, 50000.0, ensemble_signal=0.3)

        expected_keys = [
            "direction", "expected_return_pct", "var_5pct", "cvar_5pct",
            "sl_price", "tp_price", "payoff_ratio", "kelly_fraction_pct",
            "suggested_size_usd", "prob_profit_pct",
        ]
        for key in expected_keys:
            assert key in profile, f"Missing key: {key}"

    def test_long_sl_below_price(self):
        from apollo.execution.risk import RiskDashboard
        rd = RiskDashboard()
        paths = _make_paths()
        profile = rd.compute_profile(paths, 50000.0, ensemble_signal=0.5)
        assert profile["direction"] == "LONG"
        assert profile["sl_price"] < 50000.0
        assert profile["tp_price"] > 50000.0

    def test_short_sl_above_price(self):
        from apollo.execution.risk import RiskDashboard
        rd = RiskDashboard()
        paths = _make_paths()
        profile = rd.compute_profile(paths, 50000.0, ensemble_signal=-0.5)
        assert profile["direction"] == "SHORT"
        assert profile["sl_price"] > 50000.0
        assert profile["tp_price"] < 50000.0

    def test_payoff_positive(self):
        from apollo.execution.risk import RiskDashboard
        rd = RiskDashboard()
        paths = _make_paths()
        profile = rd.compute_profile(paths, 50000.0, ensemble_signal=0.3)
        assert profile["payoff_ratio"] > 0

    def test_format_profile(self):
        from apollo.execution.risk import RiskDashboard
        rd = RiskDashboard()
        paths = _make_paths()
        profile = rd.compute_profile(paths, 50000.0, ensemble_signal=0.3)
        text = RiskDashboard.format_profile(profile)
        assert "LONG" in text
        assert "VaR" in text


class TestDynamicSizing:
    def test_sizing_bounds(self):
        from apollo.execution.risk import RiskDashboard
        size = RiskDashboard.dynamic_size(
            kelly=0.02,
            regime_label="Low Volatility (Trending)",
            confidence="HIGH",
            current_drawdown_pct=0,
            capital=10000.0,
        )
        assert 0 < size <= 500  # max 5% of 10000

    def test_drawdown_cutoff(self):
        from apollo.execution.risk import RiskDashboard
        # At 15%+ drawdown --> size = 0
        size = RiskDashboard.dynamic_size(
            kelly=0.02,
            regime_label="Low Volatility (Trending)",
            confidence="HIGH",
            current_drawdown_pct=16.0,
            capital=10000.0,
        )
        assert size == 0.0

    def test_regime_reduces_hv_ranging(self):
        from apollo.execution.risk import RiskDashboard
        size_trend = RiskDashboard.dynamic_size(
            kelly=0.02, regime_label="Low Volatility (Trending)",
            confidence="HIGH", current_drawdown_pct=0, capital=10000.0,
        )
        size_range = RiskDashboard.dynamic_size(
            kelly=0.02, regime_label="High Volatility (Ranging)",
            confidence="HIGH", current_drawdown_pct=0, capital=10000.0,
        )
        assert size_range < size_trend

    def test_confidence_scaling(self):
        from apollo.execution.risk import RiskDashboard
        size_high = RiskDashboard.dynamic_size(
            kelly=0.02, regime_label="Low Volatility (Trending)",
            confidence="HIGH", current_drawdown_pct=0, capital=10000.0,
        )
        size_low = RiskDashboard.dynamic_size(
            kelly=0.02, regime_label="Low Volatility (Trending)",
            confidence="LOW", current_drawdown_pct=0, capital=10000.0,
        )
        assert size_low < size_high
