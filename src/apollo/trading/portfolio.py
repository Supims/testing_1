"""
Portfolio Tracking
===================
Equity curve, drawdown, Sharpe/Sortino, and win rate analysis.
Reads from PaperTrader's SQLite database.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("trading.portfolio")


class Portfolio:
    """Portfolio analytics from paper trading history."""

    def __init__(self, db_path: Path = None):
        from apollo.config import settings
        self.db_path = db_path or (settings.models_dir / "paper_trades.db")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def stats(self) -> dict:
        """Complete portfolio statistics."""
        conn = self._connect()

        # Closed trades
        rows = conn.execute(
            "SELECT * FROM trades WHERE status = 'CLOSED' ORDER BY closed_at"
        ).fetchall()
        conn.close()

        if not rows:
            return {
                "total_trades": 0, "win_rate": 0, "total_pnl_usd": 0,
                "total_pnl_pct": 0, "sharpe": 0, "sortino": 0,
                "max_drawdown_pct": 0, "profit_factor": 0,
                "avg_win_pct": 0, "avg_loss_pct": 0,
                "best_trade_pct": 0, "worst_trade_pct": 0,
                "avg_holding_hours": 0,
            }

        pnl_list = [float(r['pnl_pct']) for r in rows]
        pnl_usd = [float(r['pnl_usd']) for r in rows]
        pnl_arr = np.array(pnl_list)

        n_trades = len(rows)
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p < 0]
        n_wins = len(wins)
        n_losses = len(losses)

        # Sharpe (annualized, assuming ~6 trades/day for 1h scanning)
        if len(pnl_arr) > 1 and np.std(pnl_arr) > 0:
            sharpe = np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Sortino (downside deviation only)
        downside = pnl_arr[pnl_arr < 0]
        if len(downside) > 1 and np.std(downside) > 0:
            sortino = np.mean(pnl_arr) / np.std(downside) * np.sqrt(252)
        else:
            sortino = sharpe

        # Max drawdown from cumulative PnL
        cum_pnl = np.cumsum(pnl_arr)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = cum_pnl - running_max
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0

        # Profit factor
        gross_wins = sum(wins) if wins else 0
        gross_losses = abs(sum(losses)) if losses else 0
        profit_factor = gross_wins / max(gross_losses, 0.01)

        # Average holding period
        holding_hours = []
        for r in rows:
            if r['opened_at'] and r['closed_at']:
                try:
                    opened = datetime.fromisoformat(r['opened_at'])
                    closed = datetime.fromisoformat(r['closed_at'])
                    hours = (closed - opened).total_seconds() / 3600
                    holding_hours.append(hours)
                except (ValueError, TypeError):
                    pass

        return {
            "total_trades": n_trades,
            "wins": n_wins,
            "losses": n_losses,
            "win_rate": n_wins / max(n_trades, 1),
            "total_pnl_usd": sum(pnl_usd),
            "total_pnl_pct": sum(pnl_list),
            "sharpe": round(sharpe, 2),
            "sortino": round(sortino, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_win_pct": np.mean(wins) if wins else 0,
            "avg_loss_pct": np.mean(losses) if losses else 0,
            "best_trade_pct": max(pnl_list) if pnl_list else 0,
            "worst_trade_pct": min(pnl_list) if pnl_list else 0,
            "avg_holding_hours": np.mean(holding_hours) if holding_hours else 0,
        }

    def win_rate_by_regime(self) -> dict[str, float]:
        """Win rate broken down by HMM regime (from AI reasoning)."""
        # This would need regime data stored in trades -- for now returns empty
        return {}

    def summary_text(self) -> str:
        """Formatted summary for Telegram / console."""
        s = self.stats()
        if s["total_trades"] == 0:
            return "No trades yet."

        lines = [
            "--- Portfolio Summary ---",
            f"Trades: {s['total_trades']} ({s['wins']}W / {s['losses']}L)",
            f"Win Rate: {s['win_rate']:.1%}",
            f"Total PnL: ${s['total_pnl_usd']:+,.2f} ({s['total_pnl_pct']:+.2f}%)",
            f"Sharpe: {s['sharpe']:.2f} | Sortino: {s['sortino']:.2f}",
            f"Max DD: {s['max_drawdown_pct']:.2f}%",
            f"Profit Factor: {s['profit_factor']:.2f}",
            f"Avg Win: {s['avg_win_pct']:+.2f}% | Avg Loss: {s['avg_loss_pct']:+.2f}%",
            f"Best: {s['best_trade_pct']:+.2f}% | Worst: {s['worst_trade_pct']:+.2f}%",
            f"Avg Hold: {s['avg_holding_hours']:.1f}h",
        ]
        return "\n".join(lines)
