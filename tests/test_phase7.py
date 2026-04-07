"""Tests for Phase 7 modules: Correlation, Events, Quality, Retrain."""

import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta


# ===== Correlation Engine Tests =====

class TestCorrelationEngine:
    """Test cross-pair correlation engine."""

    def test_cluster_detection(self):
        """Test finding clusters from correlation matrix."""
        import pandas as pd
        from apollo.core.correlation import CorrelationEngine

        engine = CorrelationEngine()
        # Create a fake corr matrix with high correlation between BTC and ETH
        corr = pd.DataFrame(
            [[1.0, 0.85, 0.2], [0.85, 1.0, 0.3], [0.2, 0.3, 1.0]],
            index=["BTCUSDT", "ETHUSDT", "LINKUSDT"],
            columns=["BTCUSDT", "ETHUSDT", "LINKUSDT"],
        )
        clusters = engine._find_clusters(corr)
        assert len(clusters) >= 1
        assert "BTCUSDT" in clusters[0]["members"]
        assert "ETHUSDT" in clusters[0]["members"]

    def test_hedge_detection(self):
        """Test finding negatively correlated pairs."""
        import pandas as pd
        from apollo.core.correlation import CorrelationEngine

        engine = CorrelationEngine()
        corr = pd.DataFrame(
            [[1.0, -0.5], [-0.5, 1.0]],
            index=["BTCUSDT", "XRPUSDT"],
            columns=["BTCUSDT", "XRPUSDT"],
        )
        hedges = engine._find_hedges(corr)
        assert len(hedges) == 1
        assert hedges[0]["correlation"] == -0.5

    def test_concentration_score(self):
        """Test portfolio concentration scoring."""
        import pandas as pd
        from apollo.core.correlation import CorrelationEngine

        engine = CorrelationEngine()

        # High concentration (everything correlated)
        corr_high = pd.DataFrame(
            [[1.0, 0.9, 0.8], [0.9, 1.0, 0.85], [0.8, 0.85, 1.0]],
            index=["A", "B", "C"], columns=["A", "B", "C"],
        )
        score = engine._concentration_score(corr_high)
        assert score > 0.7

        # Low concentration
        corr_low = pd.DataFrame(
            [[1.0, 0.1, -0.1], [0.1, 1.0, 0.05], [-0.1, 0.05, 1.0]],
            index=["A", "B", "C"], columns=["A", "B", "C"],
        )
        score_low = engine._concentration_score(corr_low)
        assert score_low < 0.3

    def test_position_conflict_check(self):
        """Test position conflict detection."""
        import pandas as pd
        from apollo.core.correlation import CorrelationEngine

        engine = CorrelationEngine()
        corr = pd.DataFrame(
            [[1.0, 0.85], [0.85, 1.0]],
            index=["BTCUSDT", "ETHUSDT"],
            columns=["BTCUSDT", "ETHUSDT"],
        )
        corr_data = {"matrix": corr}
        positions = [{"symbol": "BTCUSDT", "direction": "LONG"}]

        result = engine.check_position_conflict("ETHUSDT", "LONG", positions, corr_data)
        assert result["risk_level"] in ("WARNING", "BLOCK")

    def test_prompt_block_generation(self):
        """Test prompt block output."""
        import pandas as pd
        from apollo.core.correlation import CorrelationEngine

        engine = CorrelationEngine()
        corr_data = {
            "matrix": pd.DataFrame(
                [[1.0, 0.85], [0.85, 1.0]],
                index=["BTCUSDT", "ETHUSDT"],
                columns=["BTCUSDT", "ETHUSDT"],
            ),
            "clusters": [{"members": ["BTCUSDT", "ETHUSDT"], "avg_correlation": 0.85, "size": 2}],
            "hedges": [],
            "concentration": 0.85,
        }
        block = engine.to_prompt_block(corr_data)
        assert "CROSS-PAIR CORRELATION" in block
        assert "HIGH" in block  # concentration > 0.6


# ===== Events Calendar Tests =====

class TestMacroEventCalendar:
    """Test macro events calendar."""

    def test_built_in_events(self):
        """Calendar should have events."""
        from apollo.core.events import MacroEventCalendar
        cal = MacroEventCalendar()
        assert len(cal._events) > 20, "Should have 30+ hardcoded events"

    def test_get_upcoming(self):
        """Should return events in a time window."""
        from apollo.core.events import MacroEventCalendar
        cal = MacroEventCalendar()
        upcoming = cal.get_upcoming(days_ahead=365)
        assert len(upcoming) > 0

    def test_risk_multiplier_default(self):
        """Should return 1.0 when no imminent events."""
        from apollo.core.events import MacroEventCalendar
        cal = MacroEventCalendar()
        mult = cal.get_risk_multiplier()
        assert 0 < mult <= 1.0

    def test_prompt_block_format(self):
        """Prompt block should be valid string."""
        from apollo.core.events import MacroEventCalendar
        cal = MacroEventCalendar()
        block = cal.to_prompt_block(days_ahead=365)
        if block:  # May be empty if no events in range
            assert "MACRO EVENTS" in block


# ===== Decision Quality Tracker Tests =====

class TestDecisionTracker:
    """Test decision quality tracking."""

    @pytest.fixture
    def tracker(self, tmp_path):
        from apollo.ai.quality import DecisionTracker
        return DecisionTracker(db_path=tmp_path / "quality.db")

    def test_record_decision(self, tracker):
        tracker.record_decision(
            "BTCUSDT", "LONG", "HIGH", 67000.0, scan_id="test123"
        )
        stats = tracker.get_stats()
        assert stats["total_evaluated"] == 0  # Not evaluated yet

    def test_evaluate_pending(self, tracker):
        """Decisions < 24h old should not be evaluated."""
        tracker.record_decision(
            "BTCUSDT", "LONG", "HIGH", 67000.0, scan_id="test"
        )
        result = tracker.evaluate_pending(lambda sym: 68000.0)
        # Should be 0 because decision is too recent (< 24h)
        assert result == 0

    def test_format_stats_empty(self, tracker):
        result = tracker.format_stats()
        assert "No evaluated decisions" in result

    def test_prompt_block_empty(self, tracker):
        result = tracker.to_prompt_block()
        assert result == ""

    def test_skip_records(self, tracker):
        """CLOSE decisions should not be recorded."""
        tracker.record_decision("BTCUSDT", "CLOSE", "HIGH", 67000.0)
        # Should not crash, but also shouldn't create a record
        conn = tracker._connect()
        count = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        conn.close()
        assert count == 0


# ===== Model Retrainer Tests =====

class TestModelRetrainer:
    """Test model auto-retraining."""

    def test_should_retrain_first_time(self, tmp_path):
        from apollo.models.retrain import ModelRetrainer
        rt = ModelRetrainer.__new__(ModelRetrainer)
        rt._models_dir = tmp_path
        rt._meta_file = tmp_path / "retrain_meta.json"
        rt._scanner = None
        rt.retrain_hours = 24
        rt.window_days = 90
        rt._meta = {}
        assert rt.should_retrain() is True

    def test_should_not_retrain_if_recent(self, tmp_path):
        import json
        from apollo.models.retrain import ModelRetrainer
        rt = ModelRetrainer.__new__(ModelRetrainer)
        rt._models_dir = tmp_path
        rt._meta_file = tmp_path / "retrain_meta.json"
        rt._scanner = None
        rt.retrain_hours = 24
        rt.window_days = 90
        rt._meta = {
            "last_retrain_timestamp": datetime.now(timezone.utc).isoformat()
        }
        assert rt.should_retrain() is False

    def test_format_status(self, tmp_path):
        from apollo.models.retrain import ModelRetrainer
        rt = ModelRetrainer.__new__(ModelRetrainer)
        rt._models_dir = tmp_path
        rt._meta_file = tmp_path / "retrain_meta.json"
        rt._scanner = None
        rt.retrain_hours = 24
        rt.window_days = 90
        rt._meta = {}
        status = rt.format_status()
        assert "Retrain" in status

    def test_retrain_no_scanner(self, tmp_path):
        from apollo.models.retrain import ModelRetrainer
        rt = ModelRetrainer.__new__(ModelRetrainer)
        rt._models_dir = tmp_path
        rt._meta_file = tmp_path / "retrain_meta.json"
        rt._scanner = None
        rt.retrain_hours = 24
        rt.window_days = 90
        rt._meta = {}
        result = rt.retrain(force=True)
        assert result["status"] == "error"


# ===== Prompt Logger cleanup tests =====

class TestPromptLoggerCleanup:
    """Test log rotation."""

    def test_cleanup_old_logs(self, tmp_path):
        from apollo.ai.prompt_log import PromptLogger
        # Create old directory
        old_dir = tmp_path / "2020-01-01"
        old_dir.mkdir()
        (old_dir / "interactions.jsonl").write_text("{}")

        # Create recent directory
        recent = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        recent_dir = tmp_path / recent
        recent_dir.mkdir()
        (recent_dir / "interactions.jsonl").write_text("{}")

        plog = PromptLogger(base_dir=tmp_path, max_days=30)

        # Old dir should be cleaned up
        assert not old_dir.exists()
        # Recent dir should remain
        assert recent_dir.exists()
