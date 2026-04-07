"""Tests for Monte Carlo Simulator."""

import numpy as np
import pandas as pd
import pytest


def _make_mc_data(n=500):
    """Returns (log_returns, regimes) for MC fitting."""
    np.random.seed(42)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    returns = pd.Series(np.random.normal(0, 0.005, n), index=idx, name="ret")
    regimes = pd.Series(np.random.choice([0, 1, 2, 3], n, p=[0.3, 0.2, 0.3, 0.2]), index=idx)
    return returns, regimes


class TestMonteCarloFit:
    def test_fit_populates_residuals(self):
        from apollo.models.monte_carlo import MonteCarloSimulator
        mc = MonteCarloSimulator()
        rets, regs = _make_mc_data()
        mc.fit(rets, regs)
        assert mc.is_fitted
        assert len(mc._global_resid) > 0

    def test_fit_too_little_data_raises(self):
        from apollo.models.monte_carlo import MonteCarloSimulator, MCConfig
        mc = MonteCarloSimulator(MCConfig(seed=42))
        rets = pd.Series(np.random.normal(0, 0.005, 30))
        regs = pd.Series(np.zeros(30))
        with pytest.raises(Exception):
            mc.fit(rets, regs)


class TestMonteCarloSimulate:
    def test_simulate_shape(self):
        from apollo.models.monte_carlo import MonteCarloSimulator, MCConfig
        mc = MonteCarloSimulator(MCConfig(n_scenarios=100, horizon=24, seed=42))
        rets, regs = _make_mc_data()
        mc.fit(rets, regs)
        paths = mc.simulate(50000.0, regime_id=0)
        assert paths.shape == (100, 24)

    def test_prices_positive(self):
        from apollo.models.monte_carlo import MonteCarloSimulator, MCConfig
        mc = MonteCarloSimulator(MCConfig(n_scenarios=200, seed=42))
        rets, regs = _make_mc_data()
        mc.fit(rets, regs)
        paths = mc.simulate(50000.0, regime_id=0)
        assert (paths > 0).all()

    def test_convergence(self):
        """Mean of many paths should be close to current price."""
        from apollo.models.monte_carlo import MonteCarloSimulator, MCConfig
        price = 50000.0
        mc = MonteCarloSimulator(MCConfig(n_scenarios=2000, horizon=10, seed=42))
        rets, regs = _make_mc_data(1000)
        mc.fit(rets, regs)
        paths = mc.simulate(price, regime_id=0)
        mean_final = paths[:, -1].mean()
        # Should be within 5% of current price
        assert abs(mean_final - price) / price < 0.05

    def test_simulate_before_fit_raises(self):
        from apollo.models.monte_carlo import MonteCarloSimulator
        mc = MonteCarloSimulator()
        with pytest.raises(Exception):
            mc.simulate(50000.0, 0)


class TestPathStatistics:
    def test_var_ordering(self):
        from apollo.models.monte_carlo import MonteCarloSimulator, MCConfig
        mc = MonteCarloSimulator(MCConfig(n_scenarios=500, seed=42))
        rets, regs = _make_mc_data()
        mc.fit(rets, regs)
        paths = mc.simulate(50000.0, 0)

        stats = MonteCarloSimulator.path_statistics(paths, 50000.0)
        assert stats["VaR_5pct"] <= stats["median_return"]
        assert stats["CVaR_5pct"] <= stats["VaR_5pct"]

    def test_prob_profit_range(self):
        from apollo.models.monte_carlo import MonteCarloSimulator, MCConfig
        mc = MonteCarloSimulator(MCConfig(n_scenarios=500, seed=42))
        rets, regs = _make_mc_data()
        mc.fit(rets, regs)
        paths = mc.simulate(50000.0, 0)

        stats = MonteCarloSimulator.path_statistics(paths, 50000.0)
        assert 0 <= stats["prob_profit"] <= 1


class TestMCSaveLoad:
    def test_roundtrip(self, tmp_path):
        from apollo.models.monte_carlo import MonteCarloSimulator, MCConfig
        mc = MonteCarloSimulator(MCConfig(n_scenarios=50, seed=42))
        rets, regs = _make_mc_data()
        mc.fit(rets, regs)

        path = tmp_path / "mc.joblib"
        mc.save(path)

        loaded = MonteCarloSimulator.load(path)
        assert loaded.is_fitted

        # Both should produce same statistics
        p1 = mc.simulate(50000.0, 0)
        p2 = loaded.simulate(50000.0, 0)
        assert p1.shape == p2.shape
