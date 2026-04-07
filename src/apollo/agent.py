"""
Autonomous Agent Loop
=======================
1-hour scanning cycle with adaptive intervals.

Steps per cycle:
1. Ensure models are trained (first run / load from disk)
2. Evaluate past predictions (self-correction)
3. Cleanup expired self-notes
4. Check SL/TP + alerts on open trades
5. Poll Telegram for commands
6. Run scanner (multi-pair)
7. AI analysis with full context
8. Execute paper trades from decisions
9. Send Telegram alerts
10. Portfolio snapshot

Features:
- Auto-training on first run (BTCUSDT last 90 days)
- Adaptive interval (30min HV, 1h normal, 2h quiet)
- Exponential backoff on errors
- Graceful shutdown via stop file or SIGINT
"""

from __future__ import annotations

import logging
import signal
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger("agent")


class Agent:
    """Autonomous scanning and trading agent."""

    # Adaptive interval thresholds
    INTERVAL_HIGH_VOL = 1800     # 30 min
    INTERVAL_TRENDING = 2700     # 45 min
    INTERVAL_NORMAL = 3600       # 60 min
    INTERVAL_QUIET = 7200        # 120 min
    PNL_URGENCY_PCT = 3.0        # If any position has > X% PnL, scan faster

    def __init__(self, base_interval_hours: float = 1.0):
        from apollo.config import settings
        from apollo.ai.brain import Brain
        from apollo.ai.memory import AIMemory
        from apollo.trading.paper import PaperTrader
        from apollo.trading.portfolio import Portfolio
        from apollo.alerts.telegram import TelegramAlerter
        from apollo.alerts.handler import TelegramHandler
        from apollo.core.scanner import Scanner

        self.base_interval = base_interval_hours * 3600
        self.settings = settings

        # Core components
        self.scanner = Scanner()
        self.brain = Brain()
        self.memory = AIMemory()
        self.trader = PaperTrader()
        self.portfolio = Portfolio()
        self.alerter = TelegramAlerter()
        self.handler = TelegramHandler()

        # State
        self._running = False
        self._cycle_count = 0
        self._consecutive_errors = 0
        self._stop_file = settings.project_root / ".stop_agent"
        self._last_regime_label = "Unknown"

        # Register Telegram commands
        self._register_commands()

    def _register_commands(self):
        self.handler.register_handler("/positions", self._cmd_positions)
        self.handler.register_handler("/portfolio", self._cmd_portfolio)
        self.handler.register_handler("/memory", self._cmd_memory)
        self.handler.register_handler("/status", self._cmd_status)
        self.handler.register_handler("/ask", self._cmd_ask)
        self.handler.register_handler("/prompts", self._cmd_prompts)

    def _cmd_positions(self, args: str) -> str:
        trades = self.trader.get_open_trades()
        if not trades:
            return "No open positions."
        lines = []
        for t in trades:
            lines.append(
                f"{t.direction} {t.symbol} @ ${t.entry_price:,.2f} "
                f"PnL: {t.unrealized_pnl_pct:+.2f}%"
            )
        return "\n".join(lines)

    def _cmd_portfolio(self, args: str) -> str:
        return self.portfolio.summary_text()

    def _cmd_memory(self, args: str) -> str:
        stats = self.memory.format_stats()
        notes = self.memory.get_active_notes()
        result = stats
        if notes:
            result += "\n\nActive notes:\n" + "\n".join(f"  > {n}" for n in notes[:5])
        return result

    def _cmd_status(self, args: str) -> str:
        budget = self.brain.budget.status()
        interval = self._adaptive_interval()
        return (
            f"Agent: cycle #{self._cycle_count} | "
            f"errors: {self._consecutive_errors}\n"
            f"Regime: {self._last_regime_label}\n"
            f"Next scan: {interval / 60:.0f}min\n"
            f"{budget}"
        )

    def _cmd_ask(self, args: str) -> str:
        if not args:
            return "Usage: /ask [your question]"
        positions = self.trader.get_open_positions_as_dicts()
        return self.brain.chat(args, positions)

    def _cmd_prompts(self, args: str) -> str:
        """Show prompt log stats and recent interactions."""
        stats = self.brain.prompt_log.format_stats()
        recent = self.brain.prompt_log.get_recent_interactions(3)
        lines = [stats]
        if recent:
            lines.append("\nRecent:")
            for r in recent:
                ts = r.get("timestamp", "?")[:16]
                model = r.get("model", "?")
                tokens = r.get("total_tokens", 0)
                cost = r.get("cost_usd", 0)
                task = r.get("task_type", "?")
                lines.append(f"  {ts} | {model} | {tokens} tok | ${cost:.4f} | {task}")
        return "\n".join(lines)

    # -- Startup training ---------------------------------------------------

    def _ensure_trained(self):
        """
        Load pre-trained models from disk, or train from scratch on first run.

        Model dir: models/scanner/
        If scanner_meta.json exists -> load
        Else -> fetch BTCUSDT 90d of 1h data, train HMM+MC+XGBoost, save
        """
        models_dir = self.settings.models_dir / "scanner"

        # Try loading existing models
        if (models_dir / "scanner_meta.json").exists():
            try:
                from apollo.core.scanner import Scanner
                self.scanner = Scanner.load(str(models_dir))
                logger.info("Loaded pre-trained models from %s", models_dir)
                return
            except Exception as e:
                logger.warning("Failed to load models: %s -- retraining", e)

        # First run: train from scratch on BTC + ETH for better regime generalization
        logger.info("First run -- training models on BTCUSDT + ETHUSDT (last 90 days)...")
        now = datetime.now(timezone.utc)
        end_str = now.strftime("%Y-%m-%d %H:%M:%S")
        start_str = (now - timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S")

        try:
            self.scanner.train(["BTCUSDT", "ETHUSDT"], start_str, end_str)
            self.scanner.save_models(str(models_dir))
            logger.info("Models trained and saved to %s", models_dir)
        except Exception as e:
            logger.error("Training failed: %s -- running without trained models", e)
            # Agent continues without trained models (scanner uses fallbacks)

    # -- Main loop ----------------------------------------------------------

    def run(self):
        """Start the autonomous agent loop."""
        self._running = True
        logger.info("Agent starting...")

        # Ensure models are trained before first cycle
        self._ensure_trained()

        # Handle graceful shutdown
        def _signal_handler(sig, frame):
            logger.info("Shutdown signal received")
            self._running = False
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        # Start Telegram handler
        self.handler.start()
        self.alerter.send_startup()

        try:
            while self._running:
                # Check stop file
                if self._stop_file.exists():
                    logger.info("Stop file detected -- shutting down")
                    self._stop_file.unlink()
                    break

                # Run cycle
                try:
                    self.run_once()
                    self._consecutive_errors = 0
                except Exception as e:
                    self._consecutive_errors += 1
                    logger.error("Cycle error (#%d): %s", self._consecutive_errors, e)
                    self.alerter.send_error(str(e)[:200])

                    if self._consecutive_errors >= 5:
                        logger.critical("Too many consecutive errors -- stopping")
                        break

                # Adaptive wait
                wait = self._adaptive_interval()
                backoff = min(2 ** self._consecutive_errors, 8) if self._consecutive_errors > 0 else 1
                actual_wait = wait * backoff

                logger.info(
                    "Next cycle in %.0f min (base=%.0f, backoff=x%d)",
                    actual_wait / 60, wait / 60, backoff,
                )

                # Interruptible sleep
                end_time = time.time() + actual_wait
                while time.time() < end_time and self._running:
                    # Check for forced scan request
                    if self.handler.force_scan_requested:
                        logger.info("Forced scan requested via Telegram")
                        break
                    time.sleep(5)

        finally:
            self.handler.stop()
            logger.info("Agent stopped after %d cycles", self._cycle_count)

    def run_once(self):
        """Execute a single scan cycle."""
        self._cycle_count += 1
        cycle_start = time.time()
        logger.info("=== Cycle #%d starting ===", self._cycle_count)

        # 1. Evaluate past predictions
        try:
            prices = self.trader.fetch_prices(
                [t.symbol for t in self.trader.get_open_trades()]
            )

            def price_fetcher(symbol):
                return prices.get(symbol)

            evaluated = self.memory.evaluate_pending(price_fetcher)
            if evaluated > 0:
                logger.info("Evaluated %d past predictions", evaluated)

            # Also evaluate decision quality
            if self.brain.quality:
                try:
                    self.brain.quality.evaluate_pending(price_fetcher)
                except Exception:
                    pass
        except Exception as e:
            logger.warning("Prediction evaluation failed: %s", e)

        # 2. Cleanup expired notes
        self.memory.cleanup_expired_notes()

        # 2b. Auto-retrain check
        try:
            from apollo.models.retrain import ModelRetrainer
            retrainer = ModelRetrainer(self.scanner)
            if retrainer.should_retrain():
                logger.info("Auto-retraining models...")
                result = retrainer.retrain()
                logger.info("Retrain result: %s", result.get("status"))
        except Exception as e:
            logger.debug("Auto-retrain skipped: %s", e)

        # 3. Check SL/TP + alerts
        try:
            symbols = [t.symbol for t in self.trader.get_open_trades()]
            if symbols:
                prices = self.trader.fetch_prices(symbols)
                closed = self.trader.check_stops(prices)
                for trade in closed:
                    self.alerter.send_trade_notification(
                        "CLOSE", trade.symbol, trade.direction,
                        trade.exit_price or trade.current_price,
                        trade.size_usd, pnl=trade.pnl_usd,
                    )

                # Check alerts
                all_alert_symbols = [a["symbol"] for a in self.trader.get_pending_alerts()]
                if all_alert_symbols:
                    alert_prices = self.trader.fetch_prices(all_alert_symbols)
                    triggered = self.trader.check_alerts(alert_prices)
                    for alert in triggered:
                        self.alerter.send_alert_triggered(alert)
        except Exception as e:
            logger.warning("SL/TP check failed: %s", e)

        # 4. Scanner
        scan_result = self.scanner.scan()
        results = scan_result.get("results", [])
        scan_id = scan_result.get("scan_id", "")

        # Track last regime for adaptive interval
        for r in results:
            regime = r.get("regime", {})
            if regime.get("label"):
                self._last_regime_label = regime["label"]
                break

        if not results:
            logger.warning("No scan results -- skipping AI analysis")
            return

        # Filter out errors
        valid_results = [r for r in results if "error" not in r]
        if not valid_results:
            logger.warning("All pairs errored -- skipping")
            return

        # 5. AI Analysis
        positions = self.trader.get_open_positions_as_dicts()
        alerts = self.trader.get_pending_alerts()

        decisions = self.brain.analyze_scan(
            scan_results=valid_results,
            scorecard_summary=scan_result.get("scorecard_summary"),
            enrichment_summary=scan_result.get("enrichment_summary"),
            positions=positions,
            alerts=alerts,
            scan_id=scan_id,
            correlation_prompt=scan_result.get("correlation_prompt", ""),
            events_prompt=scan_result.get("events_prompt", ""),
        )

        # 5b. Record decisions for quality tracking
        if self.brain.quality and decisions:
            for dec in decisions:
                price = 0.0
                for r in valid_results:
                    if r.get("symbol") == dec.symbol:
                        price = r.get("current_price", 0)
                        break
                if price > 0:
                    self.brain.quality.record_decision(
                        symbol=dec.symbol,
                        action=dec.action,
                        confidence=dec.confidence,
                        price=price,
                        scan_id=scan_id,
                    )

        # 6. Execute decisions
        actionable = [d for d in decisions if d.is_actionable]
        closes = [d for d in decisions if d.is_close]

        for dec in closes:
            open_trades = self.trader.get_open_trades()
            for trade in open_trades:
                if trade.symbol == dec.symbol:
                    # Get current price
                    price_dict = self.trader.fetch_prices([dec.symbol])
                    price = price_dict.get(dec.symbol, trade.current_price)
                    self.trader.close_trade(trade.id, price, "AI_SIGNAL")

        for dec in actionable:
            # Find risk profile for SL/TP
            risk = {}
            for r in valid_results:
                if r.get("symbol") == dec.symbol:
                    risk = r.get("risk", {})
                    break

            sl = dec.sl_price if dec.sl_price > 0 else risk.get("sl_price", 0)
            tp = dec.tp_price if dec.tp_price > 0 else risk.get("tp_price", 0)
            price = risk.get("current_price", 0)

            if price <= 0:
                price_dict = self.trader.fetch_prices([dec.symbol])
                price = price_dict.get(dec.symbol, 0)

            if price > 0:
                trade = self.trader.open_trade(
                    symbol=dec.symbol, direction=dec.action,
                    price=price, stop_loss=sl, take_profit=tp,
                    confidence=dec.confidence,
                    reasoning=dec.reasoning[:500], scan_id=scan_id,
                )
                if trade:
                    self.alerter.send_trade_notification(
                        "OPEN", trade.symbol, trade.direction,
                        trade.entry_price, trade.size_usd,
                        sl=trade.stop_loss, tp=trade.take_profit,
                    )
                    # Store in position journal
                    self.memory.store_position_journal(
                        trade.id, trade.symbol, trade.direction,
                        dec.reasoning,
                    )

            # Handle alerts
            if dec.has_alert:
                self.trader.set_alert(dec.symbol, dec.alert_type, dec.alert_price)

        # 7. Send Telegram notifications
        if decisions:
            self.alerter.send_decisions(decisions)

        # 8. Portfolio snapshot
        self.trader.snapshot()

        elapsed = time.time() - cycle_start
        logger.info(
            "=== Cycle #%d complete (%.1fs) | %d decisions, %d trades ===",
            self._cycle_count, elapsed, len(decisions), len(actionable),
        )

    # -- Adaptive interval --------------------------------------------------

    def _adaptive_interval(self) -> float:
        """
        Adjust scan interval based on market regime and position state.

        Returns seconds to next scan.

        Logic:
          - High Volatility regime -> scan every 30 min
          - Trending regime -> scan every 45 min
          - Quiet/Ranging regime -> scan every 2 hours
          - If any open position has > PNL_URGENCY_PCT unrealized PnL -> 30 min
          - Default: base_interval (1 hour)
        """
        # 1. Position urgency: if any trade is moving big, scan faster
        try:
            trades = self.trader.get_open_trades()
            for t in trades:
                if abs(getattr(t, "unrealized_pnl_pct", 0)) >= self.PNL_URGENCY_PCT:
                    logger.info(
                        "Position urgency: %s has %.1f%% PnL -> 30min interval",
                        t.symbol, t.unrealized_pnl_pct,
                    )
                    return self.INTERVAL_HIGH_VOL
        except Exception:
            pass

        # 2. Regime-based interval
        label = self._last_regime_label.lower()
        if "high volatility" in label:
            return self.INTERVAL_HIGH_VOL
        elif "trending" in label:
            return self.INTERVAL_TRENDING
        elif "quiet" in label or "low volatility" in label:
            return self.INTERVAL_QUIET

        # 3. Default
        return self.base_interval
