"""
Paper Trader
==============
Virtual position management with dynamic SL/TP from Monte Carlo.
SQLite-backed with batch operations. Supports price alerts.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("trading.paper")


@dataclass
class Trade:
    """A single paper trade."""
    id: str
    symbol: str
    direction: str              # LONG / SHORT
    entry_price: float
    current_price: float
    size_usd: float
    stop_loss: float
    take_profit: float
    opened_at: str
    closed_at: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None   # SL / TP / SIGNAL / MANUAL
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    status: str = "OPEN"
    ai_confidence: str = ""
    ai_reasoning: str = ""
    scan_id: str = ""

    @property
    def unrealized_pnl_usd(self) -> float:
        if self.status == "CLOSED":
            return self.pnl_usd
        if self.direction == "LONG":
            return (self.current_price - self.entry_price) / self.entry_price * self.size_usd
        else:
            return (self.entry_price - self.current_price) / self.entry_price * self.size_usd

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.status == "CLOSED":
            return self.pnl_pct
        if self.direction == "LONG":
            return (self.current_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.current_price) / self.entry_price * 100


class PaperTrader:
    """Virtual trading engine with SL/TP monitoring and alerts."""

    def __init__(self, initial_capital: float = 10000.0, db_path: Path = None):
        from apollo.config import settings
        self.initial_capital = initial_capital
        self.db_path = db_path or (settings.models_dir / "paper_trades.db")
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
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL,
                size_usd REAL NOT NULL,
                stop_loss REAL DEFAULT 0,
                take_profit REAL DEFAULT 0,
                opened_at TEXT NOT NULL,
                closed_at TEXT,
                exit_price REAL,
                exit_reason TEXT,
                pnl_usd REAL DEFAULT 0,
                pnl_pct REAL DEFAULT 0,
                status TEXT DEFAULT 'OPEN',
                ai_confidence TEXT DEFAULT '',
                ai_reasoning TEXT DEFAULT '',
                scan_id TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);

            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                target_price REAL NOT NULL,
                created_at TEXT NOT NULL,
                triggered_at TEXT,
                status TEXT DEFAULT 'PENDING'
            );
            CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);

            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                capital REAL NOT NULL,
                open_positions INTEGER DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0,
                total_equity REAL NOT NULL,
                daily_pnl REAL DEFAULT 0
            );
        """)
        conn.commit()
        conn.close()

    # -- Open / Close -------------------------------------------------------

    def open_trade(self, symbol: str, direction: str, price: float,
                   size_usd: float = None, stop_loss: float = 0,
                   take_profit: float = 0, confidence: str = "",
                   reasoning: str = "", scan_id: str = "") -> Optional[Trade]:
        conn = self._connect()
        existing = conn.execute(
            "SELECT id FROM trades WHERE symbol = ? AND direction = ? AND status = 'OPEN'",
            (symbol, direction)
        ).fetchone()
        if existing:
            logger.warning("Already have OPEN %s on %s -- skipping", direction, symbol)
            conn.close()
            return None

        if size_usd is None:
            capital = self.get_capital()
            size_usd = round(capital * 0.02, 2)

        # Auto SL/TP from RiskDashboard if not provided
        if stop_loss <= 0:
            sl_pct = 0.03  # 3% default
            stop_loss = price * (1 - sl_pct) if direction == "LONG" else price * (1 + sl_pct)
        if take_profit <= 0:
            tp_pct = 0.045  # 4.5% default
            take_profit = price * (1 + tp_pct) if direction == "LONG" else price * (1 - tp_pct)

        trade_id = str(uuid.uuid4())[:12]
        now = datetime.now(timezone.utc).isoformat()

        trade = Trade(
            id=trade_id, symbol=symbol, direction=direction,
            entry_price=price, current_price=price,
            size_usd=size_usd, stop_loss=round(stop_loss, 6),
            take_profit=round(take_profit, 6), opened_at=now,
            ai_confidence=confidence, ai_reasoning=reasoning[:500],
            scan_id=scan_id,
        )

        conn.execute("""
            INSERT INTO trades (id, symbol, direction, entry_price, current_price,
                               size_usd, stop_loss, take_profit, opened_at,
                               status, ai_confidence, ai_reasoning, scan_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?)
        """, (
            trade.id, symbol, direction, price, price,
            size_usd, trade.stop_loss, trade.take_profit, now,
            confidence, reasoning[:500], scan_id,
        ))
        conn.commit()
        conn.close()

        logger.info(
            "TRADE OPENED: %s %s @ $%.2f | Size: $%.0f | SL: $%.2f | TP: $%.2f",
            direction, symbol, price, size_usd, trade.stop_loss, trade.take_profit,
        )
        return trade

    def close_trade(self, trade_id: str, exit_price: float,
                    reason: str = "SIGNAL") -> Optional[Trade]:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM trades WHERE id = ? AND status = 'OPEN'",
            (trade_id,)
        ).fetchone()
        if not row:
            conn.close()
            return None

        direction = row['direction']
        entry = row['entry_price']
        size = row['size_usd']

        if direction == "LONG":
            pnl_pct = (exit_price - entry) / entry * 100
        else:
            pnl_pct = (entry - exit_price) / entry * 100

        pnl_usd = pnl_pct / 100 * size
        now = datetime.now(timezone.utc).isoformat()

        conn.execute("""
            UPDATE trades SET status = 'CLOSED', exit_price = ?, exit_reason = ?,
                   pnl_usd = ?, pnl_pct = ?, closed_at = ?, current_price = ?
            WHERE id = ?
        """, (exit_price, reason, pnl_usd, pnl_pct, now, exit_price, trade_id))
        conn.commit()
        conn.close()

        logger.info(
            "TRADE CLOSED: %s %s @ $%.2f -> $%.2f | PnL: $%+.2f (%+.2f%%) | Reason: %s",
            direction, row['symbol'], entry, exit_price, pnl_usd, pnl_pct, reason,
        )
        return Trade(
            id=trade_id, symbol=row['symbol'], direction=direction,
            entry_price=entry, current_price=exit_price,
            size_usd=size, stop_loss=row['stop_loss'],
            take_profit=row['take_profit'], opened_at=row['opened_at'],
            closed_at=now, exit_price=exit_price, exit_reason=reason,
            pnl_usd=pnl_usd, pnl_pct=pnl_pct, status="CLOSED",
        )

    # -- Alerts -------------------------------------------------------------

    def set_alert(self, symbol: str, alert_type: str, price: float):
        conn = self._connect()
        conn.execute("""
            INSERT INTO alerts (symbol, alert_type, target_price, created_at)
            VALUES (?, ?, ?, ?)
        """, (symbol, alert_type, price, datetime.now(timezone.utc).isoformat()))
        conn.commit()
        conn.close()
        logger.info("Alert set: %s %s @ $%.2f", alert_type, symbol, price)

    def check_alerts(self, prices: dict[str, float]) -> list[dict]:
        conn = self._connect()
        pending = conn.execute(
            "SELECT * FROM alerts WHERE status = 'PENDING'"
        ).fetchall()

        triggered = []
        for alert in pending:
            symbol = alert['symbol']
            price = prices.get(symbol)
            if price is None:
                continue

            target = alert['target_price']
            atype = alert['alert_type']

            hit = False
            if atype == "SET_LONG_ALERT" and price <= target:
                hit = True
            elif atype == "SET_SHORT_ALERT" and price >= target:
                hit = True

            if hit:
                conn.execute(
                    "UPDATE alerts SET status = 'TRIGGERED', triggered_at = ? WHERE id = ?",
                    (datetime.now(timezone.utc).isoformat(), alert['id']),
                )
                triggered.append({
                    "symbol": symbol, "type": atype,
                    "target_price": target, "current_price": price,
                })
                logger.info("Alert TRIGGERED: %s %s target=$%.2f current=$%.2f",
                            atype, symbol, target, price)

        if triggered:
            conn.commit()
        conn.close()
        return triggered

    # -- SL/TP check --------------------------------------------------------

    def check_stops(self, prices: dict[str, float]) -> list[Trade]:
        conn = self._connect()
        open_trades = conn.execute(
            "SELECT * FROM trades WHERE status = 'OPEN'"
        ).fetchall()

        # Batch update prices
        for row in open_trades:
            price = prices.get(row['symbol'])
            if price is not None:
                conn.execute(
                    "UPDATE trades SET current_price = ? WHERE id = ?",
                    (price, row['id']),
                )
        conn.commit()
        conn.close()

        closed = []
        for row in open_trades:
            price = prices.get(row['symbol'])
            if price is None:
                continue

            triggered = None
            if row['direction'] == "LONG":
                if row['stop_loss'] > 0 and price <= row['stop_loss']:
                    triggered = "SL"
                elif row['take_profit'] > 0 and price >= row['take_profit']:
                    triggered = "TP"
            else:
                if row['stop_loss'] > 0 and price >= row['stop_loss']:
                    triggered = "SL"
                elif row['take_profit'] > 0 and price <= row['take_profit']:
                    triggered = "TP"

            if triggered:
                result = self.close_trade(row['id'], price, triggered)
                if result:
                    closed.append(result)

        return closed

    # -- Queries ------------------------------------------------------------

    def get_open_trades(self) -> list[Trade]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM trades WHERE status = 'OPEN' ORDER BY opened_at DESC"
        ).fetchall()
        conn.close()
        return [self._row_to_trade(r) for r in rows]

    def get_closed_trades(self, limit: int = 50) -> list[Trade]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM trades WHERE status = 'CLOSED' ORDER BY closed_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [self._row_to_trade(r) for r in rows]

    def get_capital(self) -> float:
        conn = self._connect()
        realized = conn.execute(
            "SELECT COALESCE(SUM(pnl_usd), 0) as total FROM trades WHERE status = 'CLOSED'"
        ).fetchone()['total']
        conn.close()
        return self.initial_capital + realized

    def get_open_positions_as_dicts(self) -> list[dict]:
        """For prompt injection."""
        trades = self.get_open_trades()
        return [
            {
                "symbol": t.symbol, "direction": t.direction,
                "entry_price": t.entry_price, "current_price": t.current_price,
                "size_usd": t.size_usd, "unrealized_pnl_pct": t.unrealized_pnl_pct,
                "stop_loss": t.stop_loss, "take_profit": t.take_profit,
            }
            for t in trades
        ]

    def get_pending_alerts(self) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM alerts WHERE status = 'PENDING' ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        return [
            {"symbol": r['symbol'], "type": r['alert_type'],
             "price": r['target_price'], "created_at": r['created_at']}
            for r in rows
        ]

    def _row_to_trade(self, row) -> Trade:
        return Trade(
            id=row['id'], symbol=row['symbol'], direction=row['direction'],
            entry_price=row['entry_price'],
            current_price=row['current_price'] or row['entry_price'],
            size_usd=row['size_usd'], stop_loss=row['stop_loss'],
            take_profit=row['take_profit'], opened_at=row['opened_at'],
            closed_at=row['closed_at'], exit_price=row['exit_price'],
            exit_reason=row['exit_reason'], pnl_usd=row['pnl_usd'],
            pnl_pct=row['pnl_pct'], status=row['status'],
            ai_confidence=row['ai_confidence'],
            ai_reasoning=row['ai_reasoning'], scan_id=row['scan_id'],
        )

    # -- Price fetching -----------------------------------------------------

    @staticmethod
    def fetch_prices(symbols: list[str]) -> dict[str, float]:
        """Fetch current prices from Binance Futures REST API."""
        prices = {}
        try:
            url = "https://fapi.binance.com/fapi/v1/ticker/price"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                symbol_set = set(symbols)
                for item in data:
                    if item['symbol'] in symbol_set:
                        prices[item['symbol']] = float(item['price'])
        except Exception as e:
            logger.warning("Price fetch failed: %s", e)
        return prices

    # -- Snapshot -----------------------------------------------------------

    def snapshot(self):
        capital = self.get_capital()
        open_trades = self.get_open_trades()
        unrealized = sum(t.unrealized_pnl_usd for t in open_trades)

        conn = self._connect()
        conn.execute("""
            INSERT INTO portfolio (timestamp, capital, open_positions,
                                   unrealized_pnl, total_equity, daily_pnl)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            capital, len(open_trades), unrealized, capital + unrealized, 0,
        ))
        conn.commit()
        conn.close()
