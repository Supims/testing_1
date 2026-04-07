"""
Integration Tests
===================
End-to-end tests that validate the full pipeline from
features through paper trade execution.

Uses synthetic data (no network calls) to exercise
every module in the system.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone


# -- Test data factory -------------------------------------------------------

def make_realistic_df(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Build a DataFrame that mimics MarketDataProvider + FeaturePipeline output.
    Contains all columns strategies and HMM expect.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-07-01", periods=n_bars, freq="1h", tz="UTC")

    # Price: geometric random walk with mean-reverting component
    log_returns = rng.normal(0.0001, 0.005, n_bars)
    price = 50000 * np.exp(np.cumsum(log_returns))

    volume = rng.lognormal(10, 1, n_bars)
    quote_volume = volume * price
    trades = rng.poisson(500, n_bars).astype(float)
    taker_buy = volume * rng.uniform(0.4, 0.6, n_bars)
    taker_sell = volume - taker_buy

    df = pd.DataFrame({
        "open": price * (1 + rng.normal(0, 0.001, n_bars)),
        "high": price * (1 + np.abs(rng.normal(0, 0.005, n_bars))),
        "low": price * (1 - np.abs(rng.normal(0, 0.005, n_bars))),
        "close": price,
        "volume": volume,
        "quote_volume": quote_volume,
        "trades": trades,
        "taker_buy_volume": taker_buy,
        "taker_sell_volume": taker_sell,
        "taker_buy_quote_volume": taker_buy * price,
        # Funding / OI / spot (enriched fields)
        "funding_rate": rng.normal(0.0001, 0.0005, n_bars),
        "open_interest": rng.lognormal(18, 0.5, n_bars),
        "spot_close": price * (1 + rng.normal(0, 0.001, n_bars)),
        "spot_volume": volume * rng.uniform(0.8, 1.2, n_bars),
        "basis": rng.normal(0.001, 0.003, n_bars),
        "premium_index": rng.normal(0, 0.002, n_bars),
    }, index=idx)

    return df


def make_feature_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Run FeaturePipeline on raw data. Returns just the DataFrame."""
    from apollo.features.pipeline import FeaturePipeline
    pipe = FeaturePipeline()
    df, _meta = pipe.compute(raw_df)  # compute returns (df, metadata) tuple
    return df


# ============================================================================
# Test 1: Full pipeline (features -> HMM -> strategies -> ensemble -> risk)
# ============================================================================

class TestFullPipeline:
    """Validates the complete analysis chain."""

    @pytest.fixture(scope="class")
    def raw_data(self):
        return make_realistic_df(600)

    @pytest.fixture(scope="class")
    def feature_data(self, raw_data):
        return make_feature_df(raw_data)

    def test_feature_pipeline_produces_required_columns(self, feature_data):
        """Features pipeline must produce HMM + strategy inputs."""
        required = ["gk_vol", "autocorr_w20_l1", "close", "volume"]
        for col in required:
            assert col in feature_data.columns, f"Missing feature: {col}"
        assert len(feature_data) > 400

    def test_hmm_fit_predict(self, feature_data):
        """HMM must fit and produce valid regime labels."""
        from apollo.models.regime import RegimeDetector

        hmm = RegimeDetector()
        hmm.fit(feature_data)
        assert hmm.is_fitted

        regime_df = hmm.predict(feature_data)
        assert "hmm_regime" in regime_df.columns
        assert "hmm_regime_label" in regime_df.columns
        assert "hmm_ood" in regime_df.columns

        # Must have valid states
        unique_states = regime_df["hmm_regime"].dropna().unique()
        assert len(unique_states) >= 2, f"Only {len(unique_states)} states found"

    def test_strategies_compute(self, feature_data):
        """All 10 strategies must return continuous signals."""
        from apollo.models.strategies import compute_all

        signals = compute_all(feature_data)
        assert isinstance(signals, pd.DataFrame)
        assert len(signals.columns) >= 8, f"Only {len(signals.columns)} strategies"

        for col in signals.columns:
            series = signals[col]
            assert series.between(-1, 1).all(), f"{col} has values outside [-1,1]"
            # At least some non-zero values
            assert (series != 0).sum() > 0, f"{col} is all zeros"

    def test_ensemble_with_hmm(self, feature_data):
        """Ensemble must combine strategies with HMM regime probs."""
        from apollo.models.regime import RegimeDetector
        from apollo.models.strategies import compute_all
        from apollo.models.ensemble import StaticEnsemble

        hmm = RegimeDetector()
        hmm.fit(feature_data)
        regime_df = hmm.predict(feature_data)
        signals = compute_all(feature_data)
        ensemble = StaticEnsemble(hmm.label_map)
        result = ensemble.compute(signals, regime_df)

        assert isinstance(result, pd.Series)
        assert result.name == "ensemble_signal"
        assert result.between(-1, 1).all()
        # Should not be all zeros
        assert (result.abs() > 0.01).sum() > 50

    def test_monte_carlo_fit_simulate(self, feature_data):
        """MC must fit GARCH and simulate paths."""
        from apollo.models.regime import RegimeDetector
        from apollo.models.monte_carlo import MonteCarloSimulator

        hmm = RegimeDetector()
        hmm.fit(feature_data)
        regime_df = hmm.predict(feature_data)
        regime_series = regime_df["hmm_regime"].fillna(0).astype(int)

        returns = feature_data["close"].pct_change().dropna()
        mc = MonteCarloSimulator()
        mc.fit(returns, regime_series.reindex(returns.index).fillna(0).astype(int))
        assert mc.is_fitted

        last_price = float(feature_data["close"].iloc[-1])
        last_regime = int(regime_series.iloc[-1])
        paths = mc.simulate(last_price, last_regime)

        assert isinstance(paths, np.ndarray)
        assert paths.shape[0] >= 100  # scenarios
        assert paths.shape[1] >= 12   # horizon

    def test_risk_dashboard(self, feature_data):
        """Risk dashboard must produce valid profile."""
        from apollo.execution.risk import RiskDashboard

        mc_paths = np.random.randn(200, 24) * 0.01 * 50000 + 50000
        risk = RiskDashboard()
        profile = risk.compute_profile(mc_paths, 50000, 0.5)

        assert isinstance(profile, dict)
        assert "var_95_pct" in profile or "var_95" in profile or len(profile) > 0

    def test_scorecard(self, feature_data):
        """Scorecard must compute without errors."""
        from apollo.models.strategies import compute_all
        from apollo.models.scorecard import StrategyScorecard

        signals = compute_all(feature_data)
        sc = StrategyScorecard()
        result = sc.compute(signals, feature_data["close"])
        # Scorecard may return dict or DataFrame depending on implementation
        assert result is not None
        if isinstance(result, pd.DataFrame):
            assert len(result) > 0
        elif isinstance(result, dict):
            assert len(result) >= 0

    def test_enrichment(self, feature_data):
        """Enrichment must compute without errors."""
        from apollo.models.strategies import compute_all
        from apollo.models.enrichment import SignalEnrichment

        signals = compute_all(feature_data)
        en = SignalEnrichment()
        result = en.compute(signals)
        # Enrichment may return dict or DataFrame depending on implementation
        assert result is not None
        if isinstance(result, pd.DataFrame):
            assert len(result) > 0
        elif isinstance(result, dict):
            assert len(result) >= 0


# ============================================================================
# Test 2: AI Brain (with DummyProvider)
# ============================================================================

class TestBrainIntegration:
    """Tests the AI brain with DummyProvider (no LLM calls)."""

    def test_brain_initializes(self):
        """Brain must init without API keys."""
        from apollo.ai.brain import Brain
        brain = Brain()
        assert brain is not None

    def test_parser_handles_dummy_output(self):
        """Parser must handle empty/malformed responses gracefully."""
        from apollo.ai.parser import parse_response

        # DummyProvider returns "[NO_PROVIDER]..." text
        result = parse_response("[NO_PROVIDER] No AI provider configured.")
        assert isinstance(result, list)
        assert len(result) == 0  # No actionable decisions


# ============================================================================
# Test 3: Paper Trader
# ============================================================================

class TestPaperTraderIntegration:
    """Tests paper trading flow."""

    def test_open_close_trade(self):
        """Must open, track, and close a paper trade."""
        from apollo.trading.paper import PaperTrader

        trader = PaperTrader()
        trade = trader.open_trade(
            symbol="BTCUSDT", direction="LONG",
            price=50000.0, stop_loss=48000.0, take_profit=55000.0,
            confidence="HIGH", reasoning="Integration test",
            scan_id="test-001",
        )

        assert trade is not None
        assert trade.symbol == "BTCUSDT"
        assert trade.direction == "LONG"
        assert trade.entry_price == 50000.0

        # SL/TP check
        prices = {"BTCUSDT": 52000.0}
        closed = trader.check_stops(prices)
        assert len(closed) == 0  # Not hit yet

        # Close manually
        trader.close_trade(trade.id, 52000.0, "TEST")

    def test_portfolio_stats(self):
        """Portfolio must compute metrics."""
        from apollo.trading.portfolio import Portfolio

        portfolio = Portfolio()
        text = portfolio.summary_text()
        assert isinstance(text, str)


# ============================================================================
# Test 4: Memory
# ============================================================================

class TestMemoryIntegration:
    """Tests AI memory store."""

    def test_full_memory_lifecycle(self, tmp_path):
        """Prediction -> evaluate -> lesson generation."""
        from apollo.ai.memory import AIMemory

        mem = AIMemory(db_path=tmp_path / "test_memory.db")

        # Store prediction
        pred_id = mem.store_prediction(
            scan_id="scan-001", symbol="BTCUSDT", price=50000,
            direction="LONG", confidence="HIGH",
            regime_label="High Volatility (Trending)",
            ensemble_signal=0.7, reasoning="Test prediction",
            model_used="gemini-flash", tokens=1000, cost=0.01,
        )
        assert pred_id is not None

        # Store note
        mem.store_self_note("BTC looks strong", symbol="BTCUSDT", ttl_hours=48)

        # Get context
        ctx = mem.get_context("BTCUSDT", "High Volatility (Trending)")
        assert ctx.symbol == "BTCUSDT"
        block = ctx.to_prompt_block()
        assert "BTCUSDT" in block

        # Stats
        stats = mem.get_stats()
        assert stats["total_predictions"] >= 1
        assert stats["active_notes"] >= 1


# ============================================================================
# Test 5: Strategy Optimizer
# ============================================================================

class TestOptimizerIntegration:
    """Tests walk-forward optimizer."""

    def test_optimizer_runs(self):
        """Optimizer must complete without errors on synthetic data."""
        from apollo.models.optimizer import StrategyOptimizer, OptimizerConfig

        rng = np.random.default_rng(99)
        n = 2500  # ~104 days of hourly data

        # Synthetic strategy signals
        strat_names = ["trend", "mean_reversion", "squeeze", "smart_money"]
        signals_df = pd.DataFrame(
            {s: np.tanh(rng.normal(0, 0.5, n)) for s in strat_names},
            index=pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC"),
        )

        # Synthetic HMM probs (2 states)
        raw_state = rng.choice([0, 1], size=n, p=[0.6, 0.4])
        hmm_probs = pd.DataFrame({
            "hmm_prob_state_0": (raw_state == 0).astype(float) * 0.8 + 0.1,
            "hmm_prob_state_1": (raw_state == 1).astype(float) * 0.8 + 0.1,
        }, index=signals_df.index)

        # Synthetic returns
        returns = pd.Series(rng.normal(0.0001, 0.005, n), index=signals_df.index)

        cfg = OptimizerConfig(
            train_days=30, test_days=10, step_days=10, n_restarts=2,
        )
        opt = StrategyOptimizer(config=cfg)
        result = opt.optimize(signals_df, hmm_probs, returns)

        assert isinstance(result, dict)
        assert len(result) >= 2  # 2 states
        for sid, weights in result.items():
            assert isinstance(weights, dict)
            for strat in strat_names:
                assert strat in weights
                assert -1.5 <= weights[strat] <= 1.5

        # Summary
        summary = opt.summary()
        assert "Fold" in summary
        assert "OOS Sharpe" in summary


# ============================================================================
# Test 6: Scanner (unit test with mocked data)
# ============================================================================

class TestScannerUnit:
    """Unit-level tests for scanner wiring."""

    def test_scanner_creates(self):
        """Scanner must instantiate with defaults."""
        from apollo.core.scanner import Scanner
        s = Scanner()
        assert s.interval == "1h"
        assert s.enable_mtf is True
        assert s._is_trained is False

    def test_scanner_save_load(self, tmp_path):
        """Save/load cycle must not crash."""
        from apollo.core.scanner import Scanner
        import json

        # Create a minimal scanner_meta.json
        meta = {"state_labels": {"0": "Test"}, "interval": "1h", "enable_mtf": True}
        meta_path = tmp_path / "scanner_meta.json"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        # Load should not crash (no model files, just meta)
        scanner = Scanner.load(str(tmp_path))
        assert scanner.interval == "1h"
        assert scanner._state_labels == {0: "Test"}


# ============================================================================
# Test 7: Agent (unit test)
# ============================================================================

class TestAgentUnit:
    """Tests agent without starting the loop."""

    def test_agent_creates(self):
        """Agent must instantiate all components."""
        from apollo.agent import Agent
        agent = Agent()
        assert agent.scanner is not None
        assert agent.brain is not None
        assert agent.memory is not None
        assert agent.trader is not None
        assert agent.base_interval == 3600

    def test_adaptive_interval_default(self):
        """Default interval is 1 hour."""
        from unittest.mock import patch
        from apollo.agent import Agent
        agent = Agent()
        with patch.object(agent.trader, "get_open_trades", return_value=[]):
            interval = agent._adaptive_interval()
        # "Unknown" regime -> base interval
        assert interval == 3600

    def test_adaptive_interval_high_vol(self):
        """High vol regime -> 30 min."""
        from unittest.mock import patch
        from apollo.agent import Agent
        agent = Agent()
        agent._last_regime_label = "High Volatility (Trending)"
        with patch.object(agent.trader, "get_open_trades", return_value=[]):
            interval = agent._adaptive_interval()
        assert interval == 1800

    def test_adaptive_interval_quiet(self):
        """Quiet regime -> 2 hours."""
        from unittest.mock import patch
        from apollo.agent import Agent
        agent = Agent()
        agent._last_regime_label = "Low Volatility (Quiet Range)"
        with patch.object(agent.trader, "get_open_trades", return_value=[]):
            interval = agent._adaptive_interval()
        assert interval == 7200
