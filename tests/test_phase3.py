"""
Tests for Phase 3 AI modules: parser, memory, budget, brain, paper trader, portfolio.
"""

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest
import numpy as np
import pandas as pd


# ============================================================================
# Parser Tests
# ============================================================================

class TestParser:
    """Tests for AI response parser."""

    def test_parse_standard_format(self):
        from apollo.ai.parser import parse_response

        text = """MARKET: Trending regime, BTC dominance rising.

DECISION: LONG BTCUSDT
CONFIDENCE: HIGH
REASONING: Strong trend alignment with 3 strategies agreeing. MC shows 62% profit probability.
SL: $82,000 | TP: $89,500
ALERT: NONE
SELF_NOTES: Watch funding rate flip
SELF_ERRORS: Previous SHORT in trending was a mistake"""

        decisions = parse_response(text)
        assert len(decisions) == 1
        d = decisions[0]
        assert d.symbol == "BTCUSDT"
        assert d.action == "LONG"
        assert d.confidence == "HIGH"
        assert d.sl_price == 82000.0
        assert d.tp_price == 89500.0
        assert d.self_notes == "Watch funding rate flip"
        assert d.self_errors != ""
        assert d.market_assessment != ""
        assert d.is_actionable

    def test_parse_skip_decision(self):
        from apollo.ai.parser import parse_response

        text = """DECISION: SKIP ETHUSDT
CONFIDENCE: LOW
REASONING: Signals conflicting. No clear edge."""

        decisions = parse_response(text)
        assert len(decisions) == 1
        assert not decisions[0].is_actionable
        assert decisions[0].action == "SKIP"

    def test_parse_close_decision(self):
        from apollo.ai.parser import parse_response

        text = """DECISION: CLOSE SOLUSDT
CONFIDENCE: HIGH
REASONING: Regime shifted to ranging. Original thesis invalidated."""

        decisions = parse_response(text)
        assert len(decisions) == 1
        assert decisions[0].is_close
        assert decisions[0].symbol == "SOLUSDT"

    def test_parse_multiple_decisions(self):
        from apollo.ai.parser import parse_response

        text = """DECISION: LONG BTCUSDT
CONFIDENCE: HIGH
REASONING: Strong trend.
SL: $80000 | TP: $90000
ALERT: NONE

DECISION: SHORT ETHUSDT
CONFIDENCE: MEDIUM
REASONING: Divergence from BTC.
SL: $3500 | TP: $3100
ALERT: NONE"""

        decisions = parse_response(text)
        assert len(decisions) == 2
        assert decisions[0].action == "LONG"
        assert decisions[1].action == "SHORT"

    def test_parse_alert(self):
        from apollo.ai.parser import parse_response

        text = """DECISION: SKIP BTCUSDT
CONFIDENCE: MEDIUM
REASONING: Waiting for pullback.
SL: $0 | TP: $0
ALERT: SET_LONG_ALERT $82500"""

        decisions = parse_response(text)
        assert len(decisions) == 1
        assert decisions[0].has_alert
        assert decisions[0].alert_type == "SET_LONG_ALERT"
        assert decisions[0].alert_price == 82500.0

    def test_parse_legacy_format(self):
        from apollo.ai.parser import parse_response

        text = """PAIR: BTCUSDT
ACTION: STRONG LONG
CONFIDENCE: HIGH
REASONING: Bullish momentum confirmed by ADX.
RISK: VaR at -4.2%"""

        decisions = parse_response(text)
        assert len(decisions) >= 1
        assert decisions[0].symbol == "BTCUSDT"
        assert decisions[0].action == "LONG"

    def test_parse_empty_response(self):
        from apollo.ai.parser import parse_response
        decisions = parse_response("")
        assert decisions == []

    def test_parse_chat_response(self):
        from apollo.ai.parser import parse_chat_response
        result = parse_chat_response("  Hello world!  ")
        assert result == "Hello world!"


# ============================================================================
# Memory Tests
# ============================================================================

class TestMemory:
    """Tests for AI Memory module."""

    @pytest.fixture
    def memory(self, tmp_path):
        """Create a memory instance with temp DB."""
        # Monkey-patch settings
        import apollo.config as cfg
        original = cfg.settings.models_dir
        cfg.settings.__dict__['_models_dir_override'] = tmp_path

        from apollo.ai.memory import AIMemory
        mem = AIMemory(db_path=tmp_path / "test_memory.db")
        yield mem

        # Cleanup
        if hasattr(cfg.settings, '_models_dir_override'):
            del cfg.settings.__dict__['_models_dir_override']

    def test_store_and_get_prediction(self, memory):
        pred_id = memory.store_prediction(
            scan_id="scan1", symbol="BTCUSDT", price=85000,
            direction="LONG", confidence="HIGH",
            regime_label="High Volatility (Trending)",
            ensemble_signal=0.45, reasoning="Strong trend",
            model_used="gemini-pro", tokens=1000, cost=0.01,
        )
        assert len(pred_id) == 12

    def test_store_and_get_self_note(self, memory):
        memory.store_self_note("Watch funding rate", symbol="BTCUSDT", ttl_hours=24)
        notes = memory.get_active_notes()
        assert len(notes) == 1
        assert "funding rate" in notes[0]

    def test_cleanup_expired_notes(self, memory):
        # Store a note with 0 TTL (immediately expired)
        from datetime import datetime, timezone, timedelta
        conn = memory._connect()
        now = datetime.now(timezone.utc)
        expired = (now - timedelta(hours=1)).isoformat()
        conn.execute("""
            INSERT INTO self_notes (created_at, expires_at, symbol, note_text, active)
            VALUES (?, ?, NULL, 'old note', 1)
        """, (expired, expired))
        conn.commit()
        conn.close()

        cleaned = memory.cleanup_expired_notes()
        assert cleaned >= 1

    def test_memory_context(self, memory):
        ctx = memory.get_context("BTCUSDT", "High Volatility (Trending)")
        assert ctx.symbol == "BTCUSDT"
        assert ctx.past_predictions_count == 0

    def test_store_scan(self, memory):
        memory.store_scan("s1", 20, 3, "BTC=trending", 500, "gemini-pro", 0.01)
        stats = memory.get_stats()
        assert stats["total_scans"] == 1

    def test_store_position_journal(self, memory):
        memory.store_position_journal(
            "trade1", "BTCUSDT", "LONG",
            "Strong trend alignment", "Expect +3% in 24h",
        )
        # Just verify no errors

    def test_format_stats(self, memory):
        text = memory.format_stats()
        assert "Memory:" in text


# ============================================================================
# Budget Tests
# ============================================================================

class TestBudget:
    """Tests for Token Budget."""

    @pytest.fixture
    def budget(self, tmp_path):
        from apollo.ai.budget import TokenBudget, BudgetConfig
        config = BudgetConfig(daily_limit_usd=5.0, weekly_limit_usd=25.0)
        return TokenBudget(config=config, usage_file=tmp_path / "usage.json")

    def test_cost_calculation(self, budget):
        cost = budget.calculate_cost("gemini-flash", 1_000_000, 0)
        assert abs(cost - 0.15) < 0.01

    def test_model_selection(self, budget):
        model = budget.select_model(task_tier=1)
        assert model in ("gemini-flash", "gpt-4o-mini", "claude-haiku")

    def test_budget_pressure(self, budget):
        assert budget.budget_pressure == 0.0

    def test_record_usage(self, budget):
        cost = budget.record_usage("gemini-flash", 1000, 500, task_tier=1)
        assert cost > 0
        assert budget.get_daily_spend() > 0

    def test_compress_prompt(self, budget):
        prompt = "Hello world\n" * 100
        compressed = budget.compress_prompt(prompt, target_reduction=0.30)
        assert len(compressed) <= len(prompt)

    def test_status(self, budget):
        text = budget.status()
        assert "Budget:" in text


# ============================================================================
# Paper Trader Tests
# ============================================================================

class TestPaperTrader:
    """Tests for Paper Trader."""

    @pytest.fixture
    def trader(self, tmp_path):
        from apollo.trading.paper import PaperTrader
        return PaperTrader(initial_capital=10000.0, db_path=tmp_path / "test_trades.db")

    def test_open_trade(self, trader):
        trade = trader.open_trade(
            "BTCUSDT", "LONG", 85000, size_usd=200,
            stop_loss=82000, take_profit=89000,
        )
        assert trade is not None
        assert trade.symbol == "BTCUSDT"
        assert trade.direction == "LONG"
        assert trade.status == "OPEN"

    def test_close_trade(self, trader):
        trade = trader.open_trade("ETHUSDT", "SHORT", 3200, size_usd=100)
        assert trade is not None
        closed = trader.close_trade(trade.id, 3100, "TP")
        assert closed is not None
        assert closed.pnl_usd > 0
        assert closed.status == "CLOSED"

    def test_duplicate_prevention(self, trader):
        trade1 = trader.open_trade("BTCUSDT", "LONG", 85000)
        assert trade1 is not None
        trade2 = trader.open_trade("BTCUSDT", "LONG", 85000)
        assert trade2 is None  # Blocked

    def test_stop_loss_trigger(self, trader):
        trader.open_trade(
            "BTCUSDT", "LONG", 85000, size_usd=200,
            stop_loss=82000, take_profit=90000,
        )
        closed = trader.check_stops({"BTCUSDT": 81000})
        assert len(closed) == 1
        assert closed[0].exit_reason == "SL"
        assert closed[0].pnl_usd < 0

    def test_take_profit_trigger(self, trader):
        trader.open_trade(
            "BTCUSDT", "SHORT", 85000, size_usd=200,
            stop_loss=88000, take_profit=80000,
        )
        closed = trader.check_stops({"BTCUSDT": 79000})
        assert len(closed) == 1
        assert closed[0].exit_reason == "TP"
        assert closed[0].pnl_usd > 0

    def test_alerts(self, trader):
        trader.set_alert("BTCUSDT", "SET_LONG_ALERT", 82000)
        alerts = trader.get_pending_alerts()
        assert len(alerts) == 1

        triggered = trader.check_alerts({"BTCUSDT": 81000})
        assert len(triggered) == 1

    def test_capital_accounting(self, trader):
        assert trader.get_capital() == 10000.0
        trade = trader.open_trade("BTCUSDT", "LONG", 100, size_usd=1000)
        trader.close_trade(trade.id, 110, "SIGNAL")  # +10%
        capital = trader.get_capital()
        assert capital > 10000.0

    def test_get_open_positions_as_dicts(self, trader):
        trader.open_trade("BTCUSDT", "LONG", 85000, size_usd=200)
        dicts = trader.get_open_positions_as_dicts()
        assert len(dicts) == 1
        assert dicts[0]["symbol"] == "BTCUSDT"

    def test_snapshot(self, trader):
        trader.open_trade("BTCUSDT", "LONG", 85000, size_usd=200)
        trader.snapshot()  # Should not raise


# ============================================================================
# Portfolio Tests
# ============================================================================

class TestPortfolio:
    """Tests for Portfolio analytics."""

    @pytest.fixture
    def portfolio(self, tmp_path):
        from apollo.trading.paper import PaperTrader
        from apollo.trading.portfolio import Portfolio

        # Create some trades
        trader = PaperTrader(initial_capital=10000.0, db_path=tmp_path / "test.db")
        t1 = trader.open_trade("BTCUSDT", "LONG", 100, size_usd=1000)
        trader.close_trade(t1.id, 105, "TP")  # +5%
        t2 = trader.open_trade("ETHUSDT", "SHORT", 100, size_usd=500)
        trader.close_trade(t2.id, 103, "SL")  # -3%

        return Portfolio(db_path=tmp_path / "test.db")

    def test_stats(self, portfolio):
        s = portfolio.stats()
        assert s["total_trades"] == 2
        assert s["wins"] == 1
        assert s["losses"] == 1

    def test_summary_text(self, portfolio):
        text = portfolio.summary_text()
        assert "Portfolio Summary" in text
        assert "Win Rate" in text
        assert "Sharpe" in text


# ============================================================================
# Alignment Tests
# ============================================================================

class TestAlignment:
    """Tests for pair alignment."""

    def test_correlation_matrix(self):
        from apollo.core.alignment import PairAlignment
        align = PairAlignment()
        returns = {
            "BTCUSDT": pd.Series(np.random.randn(200)),
            "ETHUSDT": pd.Series(np.random.randn(200)),
        }
        corr = align.correlation_matrix(returns)
        assert not corr.empty
        assert corr.shape == (2, 2)

    def test_cluster_pairs(self):
        from apollo.core.alignment import PairAlignment
        align = PairAlignment()
        # Perfectly correlated
        corr = pd.DataFrame(
            [[1.0, 0.9], [0.9, 1.0]],
            index=["A", "B"], columns=["A", "B"],
        )
        clusters = align.cluster_pairs(corr)
        assert len(clusters) >= 1

    def test_cross_validate_signal_no_contradiction(self):
        from apollo.core.alignment import PairAlignment
        align = PairAlignment()
        signals = {"BTCUSDT": 0.5, "ETHUSDT": 0.3}
        clusters = {0: ["BTCUSDT", "ETHUSDT"]}
        contradiction = align.cross_validate_signal("BTCUSDT", "LONG", signals, clusters)
        assert contradiction == 0.0  # Both bullish

    def test_cross_validate_signal_with_contradiction(self):
        from apollo.core.alignment import PairAlignment
        align = PairAlignment()
        signals = {"BTCUSDT": 0.5, "ETHUSDT": -0.4}
        clusters = {0: ["BTCUSDT", "ETHUSDT"]}
        contradiction = align.cross_validate_signal("BTCUSDT", "LONG", signals, clusters)
        assert contradiction > 0  # ETH disagrees

    def test_beta_to_btc(self):
        from apollo.core.alignment import PairAlignment
        align = PairAlignment()
        np.random.seed(42)
        btc = pd.Series(np.random.randn(100) * 0.02)
        eth = btc * 1.5 + np.random.randn(100) * 0.005
        beta = align.beta_to_btc(eth, btc)
        assert abs(beta - 1.5) < 0.5  # Approximately 1.5


# ============================================================================
# Sentiment Tests
# ============================================================================

class TestSentiment:
    """Basic tests for sentiment collector (mocked)."""

    def test_import(self):
        from apollo.ai.sentiment import SentimentCollector
        collector = SentimentCollector()
        assert collector is not None


# ============================================================================
# Prompts Tests
# ============================================================================

class TestPrompts:
    """Tests for prompt builder."""

    def test_full_system_prompt(self):
        from apollo.ai.prompts import build_system_prompt
        prompt = build_system_prompt(compact=False)
        assert "quantitative" in prompt.lower()
        assert "SCORECARD" in prompt
        assert "ENRICHMENT" in prompt
        assert "DECISION:" in prompt

    def test_compact_system_prompt(self):
        from apollo.ai.prompts import build_system_prompt
        prompt = build_system_prompt(compact=True)
        assert len(prompt) < 1500
        assert "DECISION:" in prompt

    def test_market_prompt(self):
        from apollo.ai.prompts import build_market_prompt
        results = [{
            "symbol": "BTCUSDT",
            "regime": {"label": "Trending", "is_ood": False},
            "signals": {"trend": 0.5, "mr": -0.2, "ensemble": 0.35},
            "probabilities": {"prob_up_1p5_24h": 0.62},
            "risk": {"var_5pct": -3.2, "prob_profit_pct": 58, "payoff_ratio": 1.5},
            "current_price": 85000,
        }]
        prompt = build_market_prompt(scan_results=results)
        assert "BTCUSDT" in prompt
        assert "Trending" in prompt
        assert "85,000" in prompt

    def test_chat_prompt(self):
        from apollo.ai.prompts import build_chat_prompt
        prompt = build_chat_prompt("How is BTC?", positions=[
            {"symbol": "BTCUSDT", "direction": "LONG", "unrealized_pnl_pct": 2.5},
        ])
        assert "BTC" in prompt
        assert "LONG" in prompt
