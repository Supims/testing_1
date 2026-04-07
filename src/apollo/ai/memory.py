"""
AI Memory -- Persistent Knowledge Store
=========================================
SQLite-backed memory with self-notes, position journal,
and error pattern tracking.

Tables:
  predictions       -- every AI decision + outcome
  error_patterns    -- hit rates per (symbol, regime, direction)
  scan_history      -- per-scan metadata + cost
  lessons           -- auto-generated cautionary insights
  self_notes        -- AI's notes to its future self (TTL 48h)
  position_journal  -- reasoning behind each trade
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ai.memory")


@dataclass
class MemoryContext:
    """Rich context for a symbol + regime, injected into prompts."""
    symbol: str
    regime_label: str
    past_predictions_count: int = 0
    hit_rate: float = 0.0
    avg_return_when_long: float = 0.0
    avg_return_when_short: float = 0.0
    recent_predictions: list[dict] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    active_notes: list[str] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        if self.past_predictions_count == 0 and not self.cautions and not self.active_notes:
            return ""

        lines = [f"=== MEMORY: {self.symbol} in {self.regime_label} ==="]

        if self.past_predictions_count > 0:
            lines.append(
                f"Past: {self.past_predictions_count} predictions | "
                f"Hit rate: {self.hit_rate:.0%} | "
                f"LONG avg: {self.avg_return_when_long:+.2f}% | "
                f"SHORT avg: {self.avg_return_when_short:+.2f}%"
            )

        if self.recent_predictions:
            lines.append("Recent:")
            for p in self.recent_predictions[-5:]:
                ok = "+" if p.get("correct") else "-" if p.get("correct") is not None else "?"
                lines.append(
                    f"  [{ok}] {p['created_at'][:10]} {p['direction']}({p['confidence']}) "
                    f"@ ${p['price']:.2f} -> {p.get('return', '?')}"
                )

        if self.cautions:
            lines.append("CAUTIONS:")
            for c in self.cautions:
                lines.append(f"  * {c}")

        if self.active_notes:
            lines.append("YOUR NOTES:")
            for n in self.active_notes:
                lines.append(f"  > {n}")

        return "\n".join(lines)


class AIMemory:
    """Persistent memory for the AI brain."""

    def __init__(self, db_path: Path = None):
        from apollo.config import settings
        self.db_path = db_path or (settings.models_dir / "ai_memory.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                scan_id TEXT,
                created_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                price_at_prediction REAL NOT NULL,
                direction TEXT NOT NULL,
                confidence TEXT NOT NULL,
                regime_label TEXT,
                ensemble_signal REAL DEFAULT 0,
                ai_reasoning TEXT,
                model_used TEXT,
                tokens_used INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0,
                outcome_evaluated INTEGER DEFAULT 0,
                actual_return_24h REAL,
                prediction_correct INTEGER,
                outcome_ts TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_pred_symbol ON predictions(symbol);
            CREATE INDEX IF NOT EXISTS idx_pred_pending ON predictions(outcome_evaluated);
            CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions(created_at);

            CREATE TABLE IF NOT EXISTS error_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                regime_label TEXT NOT NULL,
                direction TEXT NOT NULL,
                total_predictions INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                avg_confidence_when_wrong REAL DEFAULT 0,
                last_updated TEXT,
                UNIQUE(symbol, regime_label, direction)
            );

            CREATE TABLE IF NOT EXISTS scan_history (
                scan_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                n_pairs_scanned INTEGER,
                n_opportunities INTEGER,
                market_regime_summary TEXT,
                tokens_used INTEGER DEFAULT 0,
                model_used TEXT,
                cost_usd REAL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                lesson_type TEXT NOT NULL,
                symbol TEXT,
                regime TEXT,
                lesson_text TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                n_supporting_samples INTEGER DEFAULT 0,
                active INTEGER DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_lessons_active ON lessons(active);

            CREATE TABLE IF NOT EXISTS self_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                symbol TEXT,
                note_text TEXT NOT NULL,
                active INTEGER DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_notes_active ON self_notes(active);

            CREATE TABLE IF NOT EXISTS position_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                reasoning TEXT NOT NULL,
                expected_outcome TEXT,
                actual_outcome TEXT,
                was_correct INTEGER,
                closed_at TEXT
            );
        """)
        conn.commit()
        conn.close()
        logger.info("Memory store initialized: %s", self.db_path)

    # -- Store operations ---------------------------------------------------

    def store_prediction(self, scan_id: str, symbol: str, price: float,
                         direction: str, confidence: str, regime_label: str,
                         ensemble_signal: float, reasoning: str,
                         model_used: str, tokens: int, cost: float) -> str:
        pred_id = str(uuid.uuid4())[:12]
        conn = self._connect()
        conn.execute("""
            INSERT INTO predictions
            (id, scan_id, created_at, symbol, price_at_prediction, direction,
             confidence, regime_label, ensemble_signal, ai_reasoning,
             model_used, tokens_used, cost_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pred_id, scan_id, datetime.now(timezone.utc).isoformat(),
            symbol, price, direction, confidence, regime_label,
            ensemble_signal, reasoning, model_used, tokens, cost,
        ))
        conn.commit()
        conn.close()
        return pred_id

    def store_scan(self, scan_id: str, n_pairs: int, n_opps: int,
                   regime_summary: str, tokens: int, model: str, cost: float):
        conn = self._connect()
        conn.execute("""
            INSERT OR REPLACE INTO scan_history
            (scan_id, created_at, n_pairs_scanned, n_opportunities,
             market_regime_summary, tokens_used, model_used, cost_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            scan_id, datetime.now(timezone.utc).isoformat(),
            n_pairs, n_opps, regime_summary, tokens, model, cost,
        ))
        conn.commit()
        conn.close()

    def store_self_note(self, note_text: str, symbol: str = None,
                        ttl_hours: int = 48):
        now = datetime.now(timezone.utc)
        expires = (now + timedelta(hours=ttl_hours)).isoformat()
        conn = self._connect()
        conn.execute("""
            INSERT INTO self_notes (created_at, expires_at, symbol, note_text)
            VALUES (?, ?, ?, ?)
        """, (now.isoformat(), expires, symbol, note_text))
        conn.commit()
        conn.close()
        logger.info("Self-note stored (expires in %dh): %s", ttl_hours, note_text[:80])

    def store_position_journal(self, trade_id: str, symbol: str,
                               direction: str, reasoning: str,
                               expected_outcome: str = ""):
        conn = self._connect()
        conn.execute("""
            INSERT INTO position_journal
            (trade_id, created_at, symbol, direction, reasoning, expected_outcome)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            trade_id, datetime.now(timezone.utc).isoformat(),
            symbol, direction, reasoning, expected_outcome,
        ))
        conn.commit()
        conn.close()

    def store_lesson(self, lesson_type: str, lesson_text: str,
                     symbol: str = None, regime: str = None,
                     confidence: float = 0.5, n_samples: int = 0):
        conn = self._connect()
        conn.execute("""
            INSERT INTO lessons (created_at, lesson_type, symbol, regime,
                                 lesson_text, confidence, n_supporting_samples)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            lesson_type, symbol, regime, lesson_text, confidence, n_samples,
        ))
        conn.commit()
        conn.close()

    # -- Query operations ---------------------------------------------------

    def get_context(self, symbol: str, regime_label: str) -> MemoryContext:
        conn = self._connect()
        ctx = MemoryContext(symbol=symbol, regime_label=regime_label)

        # Overall stats
        stats = conn.execute("""
            SELECT COUNT(*) as total, SUM(prediction_correct) as correct
            FROM predictions
            WHERE symbol = ? AND regime_label = ? AND outcome_evaluated = 1
        """, (symbol, regime_label)).fetchone()

        if stats and stats['total'] > 0:
            ctx.past_predictions_count = stats['total']
            ctx.hit_rate = (stats['correct'] or 0) / stats['total']

        # Avg returns by direction
        for direction in ("LONG", "SHORT"):
            avg = conn.execute("""
                SELECT AVG(actual_return_24h) as avg_ret
                FROM predictions
                WHERE symbol = ? AND direction = ? AND outcome_evaluated = 1
            """, (symbol, direction)).fetchone()
            if avg and avg['avg_ret'] is not None:
                if direction == "LONG":
                    ctx.avg_return_when_long = avg['avg_ret']
                else:
                    ctx.avg_return_when_short = avg['avg_ret']

        # Recent predictions
        recent = conn.execute("""
            SELECT created_at, direction, confidence, price_at_prediction,
                   actual_return_24h, prediction_correct
            FROM predictions WHERE symbol = ?
            ORDER BY created_at DESC LIMIT 10
        """, (symbol,)).fetchall()

        for r in recent:
            ctx.recent_predictions.append({
                "created_at": r['created_at'],
                "direction": r['direction'],
                "confidence": r['confidence'],
                "price": r['price_at_prediction'],
                "return": f"{r['actual_return_24h']:+.2f}%" if r['actual_return_24h'] is not None else "pending",
                "correct": bool(r['prediction_correct']) if r['prediction_correct'] is not None else None,
            })

        # Cautions
        cautions = conn.execute("""
            SELECT lesson_text FROM lessons
            WHERE active = 1
              AND (symbol = ? OR symbol IS NULL)
              AND (regime = ? OR regime IS NULL)
            ORDER BY confidence DESC, n_supporting_samples DESC
            LIMIT 5
        """, (symbol, regime_label)).fetchall()
        ctx.cautions = [c['lesson_text'] for c in cautions]

        # Active self-notes
        now = datetime.now(timezone.utc).isoformat()
        notes = conn.execute("""
            SELECT note_text FROM self_notes
            WHERE active = 1 AND expires_at > ?
              AND (symbol = ? OR symbol IS NULL)
            ORDER BY created_at DESC LIMIT 5
        """, (now, symbol)).fetchall()
        ctx.active_notes = [n['note_text'] for n in notes]

        conn.close()
        return ctx

    def get_multi_pair_context(self, pairs: list[tuple[str, str]]) -> str:
        blocks = []
        for symbol, regime in pairs:
            ctx = self.get_context(symbol, regime)
            block = ctx.to_prompt_block()
            if block:
                blocks.append(block)
        return "\n\n".join(blocks) if blocks else ""

    def get_active_notes(self) -> list[str]:
        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        notes = conn.execute("""
            SELECT note_text FROM self_notes
            WHERE active = 1 AND expires_at > ?
            ORDER BY created_at DESC
        """, (now,)).fetchall()
        conn.close()
        return [n['note_text'] for n in notes]

    # -- Evaluation ---------------------------------------------------------

    def evaluate_pending(self, price_fetcher=None, min_age_hours: int = 24) -> int:
        conn = self._connect()
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=min_age_hours)).isoformat()

        pending = conn.execute("""
            SELECT id, symbol, price_at_prediction, direction, confidence,
                   regime_label, created_at
            FROM predictions
            WHERE outcome_evaluated = 0 AND created_at < ?
            ORDER BY created_at
        """, (cutoff,)).fetchall()

        if not pending:
            conn.close()
            return 0

        evaluated = 0
        for row in pending:
            if price_fetcher is None:
                continue
            current_price = price_fetcher(row['symbol'])
            if current_price is None:
                continue

            entry_price = row['price_at_prediction']
            actual_return = (current_price - entry_price) / entry_price * 100
            direction = row['direction']

            if direction == "LONG":
                correct = actual_return > 0.5
            elif direction == "SHORT":
                correct = actual_return < -0.5
            else:
                correct = abs(actual_return) < 1.5

            conn.execute("""
                UPDATE predictions
                SET outcome_evaluated = 1, actual_return_24h = ?,
                    prediction_correct = ?, outcome_ts = ?
                WHERE id = ?
            """, (
                actual_return, int(correct),
                datetime.now(timezone.utc).isoformat(), row['id'],
            ))
            evaluated += 1

        conn.commit()
        conn.close()

        if evaluated > 0:
            self._refresh_error_patterns()
            self._generate_lessons()

        logger.info("Evaluated %d predictions", evaluated)
        return evaluated

    def _refresh_error_patterns(self):
        conn = self._connect()
        conn.execute("DELETE FROM error_patterns")
        rows = conn.execute("""
            SELECT symbol, regime_label, direction,
                   COUNT(*) as total, SUM(prediction_correct) as correct,
                   AVG(CASE WHEN prediction_correct = 0 THEN
                       CASE confidence WHEN 'HIGH' THEN 3 WHEN 'MEDIUM' THEN 2 ELSE 1 END
                   END) as avg_conf_wrong
            FROM predictions WHERE outcome_evaluated = 1
            GROUP BY symbol, regime_label, direction
        """).fetchall()

        for r in rows:
            conn.execute("""
                INSERT INTO error_patterns
                (symbol, regime_label, direction, total_predictions,
                 correct_predictions, avg_confidence_when_wrong, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                r['symbol'], r['regime_label'], r['direction'],
                r['total'], r['correct'] or 0, r['avg_conf_wrong'] or 0,
                datetime.now(timezone.utc).isoformat(),
            ))
        conn.commit()
        conn.close()

    def _generate_lessons(self):
        conn = self._connect()
        bad_combos = conn.execute("""
            SELECT symbol, regime_label, direction,
                   total_predictions, correct_predictions,
                   CAST(correct_predictions AS REAL) / total_predictions as hit_rate
            FROM error_patterns
            WHERE total_predictions >= 3
              AND CAST(correct_predictions AS REAL) / total_predictions < 0.40
        """).fetchall()

        for combo in bad_combos:
            lesson_text = (
                f"{combo['direction']} on {combo['symbol']} in "
                f"\"{combo['regime_label']}\" has {combo['hit_rate']:.0%} success "
                f"({combo['correct_predictions']}/{combo['total_predictions']}). "
                f"Exercise extreme caution or skip."
            )
            existing = conn.execute("""
                SELECT id FROM lessons
                WHERE symbol = ? AND regime = ? AND lesson_type = 'regime_caution' AND active = 1
            """, (combo['symbol'], combo['regime_label'])).fetchone()

            if existing:
                conn.execute("""
                    UPDATE lessons SET lesson_text = ?, confidence = ?,
                           n_supporting_samples = ?, created_at = ?
                    WHERE id = ?
                """, (
                    lesson_text, 1.0 - combo['hit_rate'],
                    combo['total_predictions'],
                    datetime.now(timezone.utc).isoformat(),
                    existing['id'],
                ))
            else:
                conn.execute("""
                    INSERT INTO lessons
                    (created_at, lesson_type, symbol, regime, lesson_text,
                     confidence, n_supporting_samples)
                    VALUES (?, 'regime_caution', ?, ?, ?, ?, ?)
                """, (
                    datetime.now(timezone.utc).isoformat(),
                    combo['symbol'], combo['regime_label'],
                    lesson_text, 1.0 - combo['hit_rate'],
                    combo['total_predictions'],
                ))

        conn.commit()
        conn.close()

    def cleanup_expired_notes(self) -> int:
        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        cursor = conn.execute(
            "UPDATE self_notes SET active = 0 WHERE active = 1 AND expires_at <= ?",
            (now,)
        )
        count = cursor.rowcount
        conn.commit()
        conn.close()
        if count > 0:
            logger.info("Cleaned up %d expired self-notes", count)
        return count

    # -- Stats --------------------------------------------------------------

    def get_stats(self) -> dict:
        conn = self._connect()
        total = conn.execute("SELECT COUNT(*) as n FROM predictions").fetchone()['n']
        evaluated = conn.execute(
            "SELECT COUNT(*) as n FROM predictions WHERE outcome_evaluated = 1"
        ).fetchone()['n']
        correct = conn.execute(
            "SELECT SUM(prediction_correct) as n FROM predictions WHERE outcome_evaluated = 1"
        ).fetchone()['n'] or 0
        scans = conn.execute("SELECT COUNT(*) as n FROM scan_history").fetchone()['n']
        total_cost = conn.execute(
            "SELECT SUM(cost_usd) as c FROM scan_history"
        ).fetchone()['c'] or 0
        lessons = conn.execute(
            "SELECT COUNT(*) as n FROM lessons WHERE active = 1"
        ).fetchone()['n']
        active_notes = conn.execute(
            "SELECT COUNT(*) as n FROM self_notes WHERE active = 1 AND expires_at > ?",
            (datetime.now(timezone.utc).isoformat(),)
        ).fetchone()['n']
        conn.close()

        return {
            "total_predictions": total,
            "evaluated_predictions": evaluated,
            "pending_predictions": total - evaluated,
            "correct_predictions": correct,
            "overall_hit_rate": correct / max(evaluated, 1),
            "total_scans": scans,
            "total_cost_usd": total_cost,
            "active_lessons": lessons,
            "active_notes": active_notes,
        }

    def format_stats(self) -> str:
        s = self.get_stats()
        return (
            f"Memory: {s['total_predictions']} predictions | "
            f"{s['evaluated_predictions']} evaluated | "
            f"Hit: {s['overall_hit_rate']:.1%} | "
            f"Notes: {s['active_notes']} | "
            f"Lessons: {s['active_lessons']} | "
            f"Cost: ${s['total_cost_usd']:.2f}"
        )
