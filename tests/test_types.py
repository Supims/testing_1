"""
Tests for types -- Domain model validation, serialization, edge cases.
"""

from datetime import datetime, timezone
import pytest


class TestRegimeInfo:
    def test_trending_detection(self):
        from apollo.types import RegimeInfo
        r = RegimeInfo(label="High Volatility (Trending)", state_id=0, probabilities={0: 1.0})
        assert r.is_trending is True
        assert r.is_high_vol is True

    def test_quiet_detection(self):
        from apollo.types import RegimeInfo
        r = RegimeInfo(label="Low Volatility (Quiet Range)", state_id=3, probabilities={3: 0.9})
        assert r.is_trending is False
        assert r.is_high_vol is False

    def test_unknown_label_fallback(self):
        from apollo.types import RegimeInfo
        r = RegimeInfo(label="Something Invented", state_id=0, probabilities={})
        assert r.label == "Unknown"

    def test_ood_flag(self):
        from apollo.types import RegimeInfo
        r = RegimeInfo(
            label="High Volatility (Trending)", state_id=0,
            probabilities={}, is_ood=True, ood_score=5.0,
        )
        assert r.is_ood is True


class TestProbabilityProfile:
    def test_clamping(self):
        from apollo.types import ProbabilityProfile
        p = ProbabilityProfile(prob_up_1p5_12h=1.5, prob_dd_2p0_24h=-0.1)
        assert p.prob_up_1p5_12h == 1.0
        assert p.prob_dd_2p0_24h == 0.0

    def test_best_directional(self):
        from apollo.types import ProbabilityProfile
        p = ProbabilityProfile(prob_up_1p5_12h=0.3, prob_up_1p5_24h=0.6, prob_up_3p0_48h=0.4)
        assert p.best_directional == 0.6

    def test_directional_avg(self):
        from apollo.types import ProbabilityProfile
        p = ProbabilityProfile(prob_up_1p5_12h=0.3, prob_up_1p5_24h=0.6, prob_up_3p0_48h=0.3)
        assert p.directional_avg == pytest.approx(0.4)

    def test_worst_risk(self):
        from apollo.types import ProbabilityProfile
        p = ProbabilityProfile(prob_dd_1p0_24h=0.2, prob_dd_2p0_24h=0.35)
        assert p.worst_risk == 0.35

    def test_risk_avg(self):
        from apollo.types import ProbabilityProfile
        p = ProbabilityProfile(prob_dd_1p0_24h=0.2, prob_dd_2p0_24h=0.4)
        assert p.risk_avg == pytest.approx(0.3)


class TestRiskProfile:
    def test_tradeable_good(self, sample_risk):
        assert sample_risk.is_tradeable is True

    def test_tradeable_bad_payoff(self):
        from apollo.types import RiskProfile
        r = RiskProfile(payoff_ratio=0.5, kelly_fraction_pct=2.0, var_5pct=-3.0)
        assert r.is_tradeable is False

    def test_tradeable_extreme_var(self):
        from apollo.types import RiskProfile
        r = RiskProfile(payoff_ratio=2.0, kelly_fraction_pct=2.0, var_5pct=-20.0)
        assert r.is_tradeable is False

    def test_suggested_size_usd(self):
        from apollo.types import RiskProfile
        r = RiskProfile(suggested_size_usd=500.0)
        assert r.suggested_size_usd == 500.0


class TestStrategyOutput:
    def test_active_signals(self):
        from apollo.types import StrategyOutput
        s = StrategyOutput(signals={"A": 0.5, "B": 0.0, "C": -0.3, "D": 0.0001})
        active = s.active_signals
        assert "A" in active
        assert "C" in active
        assert "B" not in active

    def test_agreement_all_same(self):
        from apollo.types import StrategyOutput
        s = StrategyOutput(signals={"A": 0.5, "B": 0.3, "C": 0.8})
        assert s.agreement_score == 1.0

    def test_agreement_split(self):
        from apollo.types import StrategyOutput
        s = StrategyOutput(signals={"A": 0.5, "B": -0.3})
        assert s.agreement_score == 0.0


class TestPairAnalysis:
    def test_direction_long(self, sample_pair_analysis):
        from apollo.types import Direction
        assert sample_pair_analysis.direction == Direction.LONG

    def test_direction_short(self, sample_pair_analysis):
        from apollo.types import Direction
        sample_pair_analysis.opportunity_score = -0.5
        assert sample_pair_analysis.direction == Direction.SHORT

    def test_direction_neutral(self, sample_pair_analysis):
        from apollo.types import Direction
        sample_pair_analysis.opportunity_score = 0.05
        assert sample_pair_analysis.direction == Direction.NEUTRAL

    def test_confidence_mapping(self, sample_pair_analysis):
        from apollo.types import Confidence
        sample_pair_analysis.opportunity_score = 0.5
        assert sample_pair_analysis.confidence == Confidence.HIGH
        sample_pair_analysis.opportunity_score = 0.15
        assert sample_pair_analysis.confidence == Confidence.MEDIUM
        sample_pair_analysis.opportunity_score = 0.05
        assert sample_pair_analysis.confidence == Confidence.LOW

    def test_interval_field(self, sample_pair_analysis):
        assert sample_pair_analysis.interval == "1h"

    def test_summary_string(self, sample_pair_analysis):
        s = sample_pair_analysis.summary()
        assert "BTCUSDT" in s
        assert "Score:" in s
        assert "LONG" in s

    def test_dashboard_string(self, sample_pair_analysis):
        d = sample_pair_analysis.dashboard()
        assert "BTCUSDT" in d
        assert "DIRECTIONAL" in d
        assert "ENSEMBLE" in d
        assert "+" in d or "-" in d  # ASCII box chars

    def test_round_trip_serialization(self, sample_pair_analysis):
        from apollo.types import PairAnalysis
        d = sample_pair_analysis.to_dict()
        restored = PairAnalysis.from_dict(d)
        assert restored.symbol == sample_pair_analysis.symbol
        assert restored.price == sample_pair_analysis.price
        assert restored.regime.label == sample_pair_analysis.regime.label
        assert restored.opportunity_score == sample_pair_analysis.opportunity_score
        assert restored.interval == sample_pair_analysis.interval
        assert (
            restored.probabilities.prob_up_1p5_24h
            == sample_pair_analysis.probabilities.prob_up_1p5_24h
        )

    def test_timestamp_gets_utc(self):
        from apollo.types import (
            PairAnalysis, RegimeInfo, StrategyOutput,
            ProbabilityProfile, RiskProfile,
        )
        pa = PairAnalysis(
            symbol="TEST",
            price=100.0,
            timestamp=datetime(2025, 1, 1, 12, 0),
            regime=RegimeInfo(label="Unknown", state_id=0, probabilities={}),
            strategies=StrategyOutput(),
            probabilities=ProbabilityProfile(),
            risk=RiskProfile(),
        )
        assert pa.timestamp.tzinfo is not None


class TestTrade:
    def test_pnl_long(self):
        from apollo.types import Trade, Direction
        t = Trade(
            id="t1", symbol="BTCUSDT", direction=Direction.LONG,
            entry_price=50000, entry_time=datetime.now(timezone.utc),
            size_usd=1000,
        )
        usd, pct = t.compute_pnl(51000)
        assert pct == pytest.approx(2.0, abs=0.01)
        assert usd == pytest.approx(20.0, abs=0.1)

    def test_pnl_short(self):
        from apollo.types import Trade, Direction
        t = Trade(
            id="t2", symbol="ETHUSDT", direction=Direction.SHORT,
            entry_price=3000, entry_time=datetime.now(timezone.utc),
            size_usd=500,
        )
        usd, pct = t.compute_pnl(2900)
        assert pct == pytest.approx(3.33, abs=0.1)
        assert usd > 0

    def test_pnl_symmetry(self):
        from apollo.types import Trade, Direction
        entry = 1000.0
        exit_px = 1050.0
        size = 1000.0
        now = datetime.now(timezone.utc)

        long_trade = Trade(
            id="L", symbol="X", direction=Direction.LONG,
            entry_price=entry, entry_time=now, size_usd=size,
        )
        short_trade = Trade(
            id="S", symbol="X", direction=Direction.SHORT,
            entry_price=entry, entry_time=now, size_usd=size,
        )

        long_usd, _ = long_trade.compute_pnl(exit_px)
        short_usd, _ = short_trade.compute_pnl(exit_px)
        assert long_usd == pytest.approx(-short_usd, abs=0.01)
