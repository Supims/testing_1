"""
Domain Types
=============
Every data structure the system produces or consumes.
This is THE contract -- all modules speak these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


# =========================================================================
# Enums
# =========================================================================

class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Action(str, Enum):
    STRONG_LONG = "STRONG LONG"
    LONG = "LONG"
    SKIP = "SKIP"
    SHORT = "SHORT"
    STRONG_SHORT = "STRONG SHORT"


class RegimeLabel(str, Enum):
    HV_TRENDING = "High Volatility (Trending)"
    LV_TRENDING = "Low Volatility (Trending)"
    HV_RANGING = "High Volatility (Ranging)"
    LV_QUIET = "Low Volatility (Quiet Range)"
    UNKNOWN = "Unknown"


class TradeStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED_TP = "CLOSED_TP"
    CLOSED_SL = "CLOSED_SL"
    CLOSED_MANUAL = "CLOSED_MANUAL"
    CLOSED_TIMEOUT = "CLOSED_TIMEOUT"


class PredictionOutcome(str, Enum):
    PENDING = "PENDING"
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    PARTIAL = "PARTIAL"
    EXPIRED = "EXPIRED"


# =========================================================================
# Regime
# =========================================================================

@dataclass
class RegimeInfo:
    """HMM regime detection output."""
    label: str
    state_id: int
    probabilities: dict[int, float]
    is_ood: bool = False
    ood_score: float = 0.0

    def __post_init__(self):
        if self.label not in [e.value for e in RegimeLabel]:
            self.label = RegimeLabel.UNKNOWN.value

    @property
    def is_trending(self) -> bool:
        return "Trending" in self.label

    @property
    def is_high_vol(self) -> bool:
        return "High Volatility" in self.label

    @property
    def dominant_probability(self) -> float:
        return max(self.probabilities.values()) if self.probabilities else 0.0


# =========================================================================
# Probabilities
# =========================================================================

@dataclass
class ProbabilityProfile:
    """Calibrated XGBoost probability outputs (5 models)."""
    prob_up_1p5_12h: float = 0.0
    prob_up_1p5_24h: float = 0.0
    prob_up_3p0_48h: float = 0.0
    prob_dd_1p0_24h: float = 0.0
    prob_dd_2p0_24h: float = 0.0

    def __post_init__(self):
        for f in ['prob_up_1p5_12h', 'prob_up_1p5_24h', 'prob_up_3p0_48h',
                   'prob_dd_1p0_24h', 'prob_dd_2p0_24h']:
            val = getattr(self, f)
            if not (0.0 <= val <= 1.0):
                setattr(self, f, max(0.0, min(1.0, val)))

    @property
    def best_directional(self) -> float:
        return max(self.prob_up_1p5_12h, self.prob_up_1p5_24h, self.prob_up_3p0_48h)

    @property
    def directional_avg(self) -> float:
        """Average bullish probability."""
        return (self.prob_up_1p5_12h + self.prob_up_1p5_24h + self.prob_up_3p0_48h) / 3

    @property
    def worst_risk(self) -> float:
        return max(self.prob_dd_1p0_24h, self.prob_dd_2p0_24h)

    @property
    def risk_avg(self) -> float:
        """Average drawdown probability."""
        return (self.prob_dd_1p0_24h + self.prob_dd_2p0_24h) / 2


# =========================================================================
# Risk
# =========================================================================

@dataclass
class RiskProfile:
    """Monte Carlo risk assessment."""
    direction: str = "NEUTRAL"
    expected_return_pct: float = 0.0
    median_return_pct: float = 0.0
    return_std_pct: float = 0.0
    prob_profit_pct: float = 50.0
    var_5pct: float = 0.0
    cvar_5pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    sl_distance_pct: float = 0.0
    tp_distance_pct: float = 0.0
    payoff_ratio: float = 0.0
    kelly_fraction_pct: float = 0.0
    suggested_size_usd: float = 0.0

    @property
    def is_tradeable(self) -> bool:
        return (
            self.payoff_ratio >= 1.0
            and self.kelly_fraction_pct >= 0.5
            and self.var_5pct > -15.0
        )


# =========================================================================
# Strategies
# =========================================================================

@dataclass
class StrategyOutput:
    """Combined strategy signals from all active strategies."""
    signals: dict[str, float] = field(default_factory=dict)
    ensemble_signal: float = 0.0

    @property
    def active_signals(self) -> dict[str, float]:
        return {k: v for k, v in self.signals.items() if abs(v) > 1e-6}

    @property
    def agreement_score(self) -> float:
        """1.0 = all same direction, 0.0 = split."""
        active = list(self.active_signals.values())
        if len(active) < 2:
            return 0.0
        positive = sum(1 for v in active if v > 0)
        negative = sum(1 for v in active if v < 0)
        return abs(positive - negative) / len(active)


# =========================================================================
# Pair Analysis (main output)
# =========================================================================

@dataclass
class PairAnalysis:
    """Complete scan result for a single trading pair."""
    symbol: str
    price: float
    timestamp: datetime
    regime: RegimeInfo
    strategies: StrategyOutput
    probabilities: ProbabilityProfile
    risk: RiskProfile
    interval: str = "1h"
    opportunity_score: float = 0.0
    risk_score: float = 0.0
    sentiment: Optional[float] = None

    def __post_init__(self):
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

    @property
    def direction(self) -> Direction:
        if self.opportunity_score > 0.1:
            return Direction.LONG
        if self.opportunity_score < -0.1:
            return Direction.SHORT
        return Direction.NEUTRAL

    @property
    def confidence(self) -> Confidence:
        score = abs(self.opportunity_score)
        if score > 0.3:
            return Confidence.HIGH
        if score > 0.1:
            return Confidence.MEDIUM
        return Confidence.LOW

    def summary(self) -> str:
        ood = " [OOD!]" if self.regime.is_ood else ""
        return (
            f"{self.symbol:<12} ${self.price:>10,.2f} | "
            f"Score:{self.opportunity_score:+.4f} | "
            f"{self.direction.value}({self.confidence.value}) | "
            f"Regime:{self.regime.label}{ood} | "
            f"P(up24h):{self.probabilities.prob_up_1p5_24h:.1%} | "
            f"VaR5%:{self.risk.var_5pct:.1f}%"
        )

    def dashboard(self) -> str:
        """Full formatted dashboard for display (pure ASCII)."""
        p = self.probabilities
        r = self.risk
        s = self.strategies

        def bar(val: float, width: int = 10) -> str:
            filled = int(max(0, min(1, val)) * width)
            return "#" * filled + "." * (width - filled)

        ood_str = "YES" if self.regime.is_ood else "No"
        ts_str = self.timestamp.strftime("%Y-%m-%d %H:%M") if self.timestamp else "N/A"

        lines = [
            "+" + "-" * 60 + "+",
            f"| {self.symbol:<58} |",
            f"| ${self.price:,.2f} | {self.regime.label:<38} |",
            f"| {ts_str} UTC | OOD: {ood_str:<25} |",
            "+" + "-" * 60 + "+",
            f"| DIRECTIONAL:                                             |",
            f"|   P(+1.5% 12h): {p.prob_up_1p5_12h:5.1%}  {bar(p.prob_up_1p5_12h):<20} |",
            f"|   P(+1.5% 24h): {p.prob_up_1p5_24h:5.1%}  {bar(p.prob_up_1p5_24h):<20} |",
            f"|   P(+3.0% 48h): {p.prob_up_3p0_48h:5.1%}  {bar(p.prob_up_3p0_48h):<20} |",
            f"| RISK:                                                    |",
            f"|   P(DD>1.0% 24h): {p.prob_dd_1p0_24h:5.1%}  {bar(p.prob_dd_1p0_24h):<19} |",
            f"|   P(DD>2.0% 24h): {p.prob_dd_2p0_24h:5.1%}  {bar(p.prob_dd_2p0_24h):<19} |",
            "+" + "-" * 60 + "+",
            f"| MC: E[ret]={r.expected_return_pct:+.2f}% | "
            f"VaR5={r.var_5pct:+.2f}% | CVaR5={r.cvar_5pct:+.2f}%",
            f"| SL: ${r.sl_price:,.2f} ({r.sl_distance_pct:.1f}%) | "
            f"TP: ${r.tp_price:,.2f} ({r.tp_distance_pct:.1f}%)",
            f"| Payoff: {r.payoff_ratio:.2f}:1 | "
            f"Kelly: {r.kelly_fraction_pct:.2f}% | "
            f"Size: ${r.suggested_size_usd:,.0f}",
            "+" + "-" * 60 + "+",
            f"| ENSEMBLE: {s.ensemble_signal:+.3f} | "
            f"OppScore: {self.opportunity_score:+.3f}",
            "+" + "-" * 60 + "+",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        d["signal_summary"] = {
            "direction": self.direction.value,
            "confidence": self.confidence.value,
            "regime_trending": self.regime.is_trending,
            "regime_high_vol": self.regime.is_high_vol,
            "is_ood": self.regime.is_ood,
            "best_directional_prob": self.probabilities.best_directional,
            "worst_risk_prob": self.probabilities.worst_risk,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PairAnalysis:
        return cls(
            symbol=d["symbol"],
            price=d["price"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            regime=RegimeInfo(**d["regime"]),
            strategies=StrategyOutput(**d["strategies"]),
            probabilities=ProbabilityProfile(**d["probabilities"]),
            risk=RiskProfile(**d["risk"]),
            interval=d.get("interval", "1h"),
            opportunity_score=d.get("opportunity_score", 0.0),
            risk_score=d.get("risk_score", 0.0),
            sentiment=d.get("sentiment"),
        )


# =========================================================================
# Scan Result (batch output)
# =========================================================================

@dataclass
class ScanMeta:
    """Metadata about a scan run."""
    scan_id: str
    timestamp: datetime
    interval: str
    train_symbol: str
    analysis_window_start: str
    analysis_window_end: str
    pairs_scanned: int
    pairs_success: int
    pairs_errors: int
    errors: dict[str, str] = field(default_factory=dict)


@dataclass
class ScanResult:
    """Complete output of a batch scan."""
    meta: ScanMeta
    results: list[PairAnalysis]
    alignment: Optional[dict] = None

    @property
    def top_opportunities(self) -> list[PairAnalysis]:
        return sorted(self.results, key=lambda x: x.opportunity_score, reverse=True)


# =========================================================================
# Trading
# =========================================================================

@dataclass
class Trade:
    """A paper or live trade."""
    id: str
    symbol: str
    direction: Direction
    entry_price: float
    entry_time: datetime
    size_usd: float
    sl_price: float = 0.0
    tp_price: float = 0.0
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    status: TradeStatus = TradeStatus.OPEN
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    confidence: Confidence = Confidence.MEDIUM
    regime_at_entry: str = ""
    strategy_signals: dict[str, float] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.status == TradeStatus.OPEN

    def compute_pnl(self, current_price: float) -> tuple[float, float]:
        """Compute PnL for a given current price. Returns (usd, pct)."""
        if self.direction == Direction.LONG:
            pct = (current_price - self.entry_price) / self.entry_price * 100
        else:
            pct = (self.entry_price - current_price) / self.entry_price * 100
        usd = self.size_usd * pct / 100
        return usd, pct


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio state."""
    timestamp: datetime
    total_capital: float
    available_capital: float
    open_positions: int
    total_pnl_usd: float
    total_pnl_pct: float
    max_drawdown_pct: float
    win_rate: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    sharpe_ratio: Optional[float] = None


# =========================================================================
# AI Memory
# =========================================================================

@dataclass
class Prediction:
    """An AI prediction stored for tracking and self-correction."""
    id: str
    symbol: str
    action: Action
    confidence: Confidence
    reasoning: str
    scan_id: str
    model_used: str
    cost_usd: float
    timestamp: datetime
    regime_label: str
    price_at_prediction: float
    target_price: Optional[float] = None
    target_horizon_hours: int = 24
    outcome: PredictionOutcome = PredictionOutcome.PENDING
    actual_return_pct: Optional[float] = None
    evaluated_at: Optional[datetime] = None


@dataclass
class Lesson:
    """An AI-generated insight from past prediction failures."""
    id: str
    created_at: datetime
    lesson_text: str
    source_prediction_ids: list[str]
    pair_pattern: str
    regime_pattern: str
    severity: str = "CAUTION"
    times_applied: int = 0
    last_applied: Optional[datetime] = None
    is_active: bool = True


@dataclass
class MemoryContext:
    """Memory block injected into AI prompts for context."""
    pair_history: list[Prediction]
    relevant_lessons: list[Lesson]
    overall_hit_rate: float
    pair_hit_rate: Optional[float] = None
    regime_hit_rate: Optional[float] = None
    total_predictions: int = 0
    streak: int = 0


# =========================================================================
# AI Analysis Result
# =========================================================================

@dataclass
class AIAnalysisResult:
    """Output from brain.analyze_scan()."""
    ai_text: str
    model_used: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    predictions_stored: int = 0
    memory_context_used: bool = False
    errors: list[str] = field(default_factory=list)
