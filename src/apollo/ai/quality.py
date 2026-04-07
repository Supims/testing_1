"""
Decision Quality Tracker
===========================
Measures and scores the quality of AI decisions over time.

Tracks:
  - Decision hit rate (did the predicted direction work?)
  - Average time to target
  - Drawdown after entry
  - Signal quality per confidence level
  - Prompt variant performance (A/B testing)

Data source: prompt logs + paper trading results.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ai.quality")


class DecisionTracker:
    """
    Tracks AI decisions and evaluates their quality against actual outcomes.

    Every decision (LONG/SHORT/SKIP) is logged with the market price at
    decision time. After a configurable evaluation window, the actual
    outcome is compared to the prediction to score accuracy.

    Usage:
        tracker = DecisionTracker()
        tracker.record_decision("BTCUSDT", "LONG", "HIGH", 67000.0, scan_id)
        # ... later ...
        tracker.evaluate_pending(price_fetcher)
        stats = tracker.get_stats()
    """

    EVAL_WINDOWS = {
        "4h": 4,
        "12h": 12,
        "24h": 24,
    }

    # Target returns to measure accuracy
    TARGET_PCT = 1.5  # Consider a decision "correct" if price moved 1.5% in direction

    def __init__(self, db_path: Path = None):
        from apollo.config import settings
        self.db_path = db_path or (settings.models_dir / "decision_quality.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence TEXT DEFAULT '',
                entry_price REAL NOT NULL,
                scan_id TEXT DEFAULT '',
                regime TEXT DEFAULT '',
                ensemble_signal REAL DEFAULT 0,
                on_chain_ls_ratio REAL DEFAULT 0,

                -- Evaluation results (filled later)
                price_4h REAL,
                price_12h REAL,
                price_24h REAL,
                return_4h REAL,
                return_12h REAL,
                return_24h REAL,
                hit_4h INTEGER,
                hit_12h INTEGER,
                hit_24h INTEGER,
                evaluated_at TEXT,
                status TEXT DEFAULT 'PENDING'
            );
            CREATE INDEX IF NOT EXISTS idx_dec_status ON decisions(status);
            CREATE INDEX IF NOT EXISTS idx_dec_symbol ON decisions(symbol);
            CREATE INDEX IF NOT EXISTS idx_dec_ts ON decisions(timestamp);

            CREATE TABLE IF NOT EXISTS quality_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_decisions INTEGER DEFAULT 0,
                hit_rate_4h REAL DEFAULT 0,
                hit_rate_12h REAL DEFAULT 0,
                hit_rate_24h REAL DEFAULT 0,
                avg_return_4h REAL DEFAULT 0,
                avg_return_12h REAL DEFAULT 0,
                avg_return_24h REAL DEFAULT 0,
                high_conf_hit_rate REAL DEFAULT 0,
                medium_conf_hit_rate REAL DEFAULT 0,
                low_conf_hit_rate REAL DEFAULT 0,
                skip_accuracy REAL DEFAULT 0
            );
        """)
        conn.commit()
        conn.close()

    def record_decision(
        self,
        symbol: str,
        action: str,
        confidence: str,
        price: float,
        scan_id: str = "",
        regime: str = "",
        ensemble_signal: float = 0.0,
        on_chain_ls_ratio: float = 0.0,
    ):
        """Record a new AI decision for future evaluation."""
        if action == "CLOSE":
            return  # Don't track close decisions

        conn = self._connect()
        conn.execute("""
            INSERT INTO decisions (timestamp, symbol, action, confidence,
                                   entry_price, scan_id, regime,
                                   ensemble_signal, on_chain_ls_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            symbol, action, confidence, price,
            scan_id, regime, ensemble_signal, on_chain_ls_ratio,
        ))
        conn.commit()
        conn.close()
        logger.debug("Decision recorded: %s %s @ $%.2f", action, symbol, price)

    def evaluate_pending(self, price_fetcher) -> int:
        """
        Evaluate decisions that are old enough.

        Args:
            price_fetcher: callable(symbol) -> current_price

        Returns:
            Number of decisions evaluated.
        """
        conn = self._connect()
        pending = conn.execute("""
            SELECT * FROM decisions
            WHERE status = 'PENDING'
            AND timestamp < ?
        """, (
            (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(),
        )).fetchall()

        if not pending:
            conn.close()
            return 0

        evaluated = 0
        for row in pending:
            try:
                current_price = price_fetcher(row["symbol"])
                if current_price is None or current_price <= 0:
                    continue

                entry = row["entry_price"]
                action = row["action"]

                # Compute returns based on direction
                raw_return = (current_price - entry) / entry * 100
                if action == "SHORT":
                    raw_return = -raw_return
                elif action == "SKIP":
                    # For SKIPs, "hit" means the market didn't move much
                    raw_return = abs((current_price - entry) / entry * 100)

                # Determine hit/miss
                if action in ("LONG", "SHORT"):
                    hit = 1 if raw_return >= self.TARGET_PCT else 0
                else:  # SKIP
                    hit = 1 if raw_return < self.TARGET_PCT else 0

                conn.execute("""
                    UPDATE decisions SET
                        price_24h = ?, return_24h = ?, hit_24h = ?,
                        evaluated_at = ?, status = 'EVALUATED'
                    WHERE id = ?
                """, (
                    current_price, round(raw_return, 4), hit,
                    datetime.now(timezone.utc).isoformat(), row["id"],
                ))
                evaluated += 1

            except Exception as e:
                logger.debug("Failed to evaluate decision %s: %s", row["id"], e)

        conn.commit()
        conn.close()

        if evaluated > 0:
            logger.info("Evaluated %d decisions", evaluated)
            self._take_snapshot()

        return evaluated

    def _take_snapshot(self):
        """Take a quality snapshot for trending."""
        stats = self.get_stats()
        conn = self._connect()
        conn.execute("""
            INSERT INTO quality_snapshots (
                timestamp, total_decisions,
                hit_rate_24h, avg_return_24h,
                high_conf_hit_rate, medium_conf_hit_rate, low_conf_hit_rate,
                skip_accuracy
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            stats.get("total_evaluated", 0),
            stats.get("hit_rate_24h", 0),
            stats.get("avg_return_24h", 0),
            stats.get("high_conf_hit_rate", 0),
            stats.get("medium_conf_hit_rate", 0),
            stats.get("low_conf_hit_rate", 0),
            stats.get("skip_accuracy", 0),
        ))
        conn.commit()
        conn.close()

    def get_stats(self, last_n_days: int = 7) -> dict:
        """Get aggregated quality stats."""
        conn = self._connect()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=last_n_days)).isoformat()

        # Overall hit rate
        all_eval = conn.execute("""
            SELECT action, confidence, hit_24h, return_24h
            FROM decisions
            WHERE status = 'EVALUATED' AND timestamp > ?
        """, (cutoff,)).fetchall()
        conn.close()

        if not all_eval:
            return {
                "total_evaluated": 0, "hit_rate_24h": 0, "avg_return_24h": 0,
                "high_conf_hit_rate": 0, "medium_conf_hit_rate": 0,
                "low_conf_hit_rate": 0, "skip_accuracy": 0,
                "by_action": {},
            }

        total = len(all_eval)
        hits = sum(1 for r in all_eval if r["hit_24h"])
        avg_ret = sum(r["return_24h"] or 0 for r in all_eval) / max(total, 1)

        # Per confidence level
        conf_stats = {}
        for conf in ["HIGH", "MEDIUM", "LOW"]:
            subset = [r for r in all_eval if r["confidence"] == conf and r["action"] in ("LONG", "SHORT")]
            if subset:
                conf_stats[f"{conf.lower()}_conf_hit_rate"] = round(
                    sum(1 for r in subset if r["hit_24h"]) / len(subset), 3
                )
            else:
                conf_stats[f"{conf.lower()}_conf_hit_rate"] = 0

        # Skip accuracy
        skips = [r for r in all_eval if r["action"] == "SKIP"]
        skip_acc = round(sum(1 for r in skips if r["hit_24h"]) / max(len(skips), 1), 3)

        # Per action
        by_action = {}
        for action in ["LONG", "SHORT", "SKIP"]:
            subset = [r for r in all_eval if r["action"] == action]
            if subset:
                by_action[action] = {
                    "count": len(subset),
                    "hit_rate": round(sum(1 for r in subset if r["hit_24h"]) / len(subset), 3),
                    "avg_return": round(sum(r["return_24h"] or 0 for r in subset) / len(subset), 3),
                }

        return {
            "total_evaluated": total,
            "hit_rate_24h": round(hits / max(total, 1), 3),
            "avg_return_24h": round(avg_ret, 3),
            "high_conf_hit_rate": conf_stats.get("high_conf_hit_rate", 0),
            "medium_conf_hit_rate": conf_stats.get("medium_conf_hit_rate", 0),
            "low_conf_hit_rate": conf_stats.get("low_conf_hit_rate", 0),
            "skip_accuracy": skip_acc,
            "by_action": by_action,
        }

    def to_prompt_block(self) -> str:
        """Generate a prompt block about decision quality for AI self-awareness."""
        stats = self.get_stats()
        if stats["total_evaluated"] == 0:
            return ""

        lines = ["=== YOUR PAST PERFORMANCE ==="]
        lines.append(f"Decisions evaluated (7d): {stats['total_evaluated']}")
        lines.append(f"Overall hit rate: {stats['hit_rate_24h']:.0%}")
        lines.append(f"Avg return per decision: {stats['avg_return_24h']:+.2f}%")

        if stats["high_conf_hit_rate"]:
            lines.append(
                f"  HIGH conf: {stats['high_conf_hit_rate']:.0%} hit rate"
            )
        if stats["medium_conf_hit_rate"]:
            lines.append(
                f"  MEDIUM conf: {stats['medium_conf_hit_rate']:.0%} hit rate"
            )
        if stats["skip_accuracy"]:
            lines.append(f"  SKIP accuracy: {stats['skip_accuracy']:.0%}")

        # Feedback
        if stats["hit_rate_24h"] < 0.4:
            lines.append("⚠ Your hit rate is LOW. Be more selective. Increase SKIP frequency.")
        elif stats["hit_rate_24h"] > 0.65:
            lines.append("✓ Strong performance. Maintain discipline.")

        if stats.get("high_conf_hit_rate", 0) < stats.get("medium_conf_hit_rate", 0):
            lines.append(
                "⚠ HIGH confidence decisions perform WORSE than MEDIUM. "
                "Calibrate your confidence ratings."
            )

        return "\n".join(lines)

    def format_stats(self) -> str:
        """Human-readable stats string."""
        stats = self.get_stats()
        if stats["total_evaluated"] == 0:
            return "Decision Quality: No evaluated decisions yet"

        return (
            f"Decision Quality (7d): {stats['total_evaluated']} evaluated | "
            f"Hit rate: {stats['hit_rate_24h']:.0%} | "
            f"Avg return: {stats['avg_return_24h']:+.2f}% | "
            f"HIGH: {stats['high_conf_hit_rate']:.0%} | "
            f"SKIP: {stats['skip_accuracy']:.0%}"
        )
