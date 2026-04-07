"""
Telegram Alerter
==================
Sends formatted alerts via Telegram Bot API.
Uses HTML parse_mode (no markdown crashes on underscores).
Raw HTTP -- no extra dependencies.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from datetime import datetime, timezone

logger = logging.getLogger("alerts.telegram")


class TelegramAlerter:
    """Sends formatted alerts via Telegram Bot API."""

    API_BASE = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, bot_token: str = None, chat_id: str = None):
        from apollo.config import settings
        self._token = bot_token or settings.telegram_bot_token
        self._chat_id = chat_id or settings.telegram_chat_id
        self._enabled = bool(self._token and self._chat_id)
        if self._enabled:
            logger.info("Telegram alerter enabled")
        else:
            logger.info("Telegram alerter disabled (no token/chat_id)")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def _send(self, text: str, parse_mode: str = "HTML") -> bool:
        if not self._enabled:
            return False

        url = self.API_BASE.format(token=self._token)
        payload = json.dumps({
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                if result.get("ok"):
                    return True
                logger.warning("Telegram API error: %s", result)
                return False
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            logger.error("Telegram HTTP %d: %s", e.code, body[:200])
            return False
        except Exception as e:
            logger.error("Telegram send failed: %s", e)
            return False

    # -- High-level alerts --------------------------------------------------

    def send_decisions(self, decisions: list, prices: dict = None) -> int:
        """Send AI decisions as Telegram alerts."""
        if not self._enabled:
            return 0

        sent = 0
        for dec in decisions:
            if dec.action == "SKIP":
                continue

            if dec.action in ("LONG", "SHORT"):
                emoji = "&#x1F7E2;" if dec.action == "LONG" else "&#x1F534;"
                price_str = ""
                if prices and dec.symbol in prices:
                    price_str = f"\nPrice: <code>${prices[dec.symbol]:,.2f}</code>"

                msg = (
                    f"{emoji} <b>{dec.symbol}</b> -- {dec.action}\n"
                    f"Confidence: <b>{dec.confidence}</b>{price_str}\n"
                    f"Reasoning: <i>{dec.reasoning[:400]}</i>"
                )
                if dec.sl_price > 0:
                    msg += f"\nSL: <code>${dec.sl_price:,.2f}</code> | TP: <code>${dec.tp_price:,.2f}</code>"

            elif dec.action == "CLOSE":
                msg = f"&#x26A0; <b>CLOSE {dec.symbol}</b>\n{dec.reasoning[:300]}"
            else:
                continue

            if self._send(msg):
                sent += 1

        return sent

    def send_trade_notification(self, action: str, symbol: str, direction: str,
                                price: float, size_usd: float,
                                sl: float = 0, tp: float = 0,
                                pnl: float = None) -> bool:
        if not self._enabled:
            return False

        if action == "OPEN":
            emoji = "&#x1F7E2;" if direction == "LONG" else "&#x1F534;"
            msg = (
                f"{emoji} <b>Trade OPENED</b>\n"
                f"Symbol: <code>{symbol}</code> | Dir: {direction}\n"
                f"Entry: <code>${price:,.2f}</code> | Size: <code>${size_usd:,.0f}</code>"
            )
            if sl > 0:
                msg += f"\nSL: <code>${sl:,.2f}</code> | TP: <code>${tp:,.2f}</code>"
        elif action == "CLOSE":
            pnl_emoji = "&#x1F4B0;" if (pnl or 0) > 0 else "&#x1F4B8;"
            msg = (
                f"{pnl_emoji} <b>Trade CLOSED</b>\n"
                f"Symbol: <code>{symbol}</code> | Dir: {direction}\n"
                f"Exit: <code>${price:,.2f}</code>\n"
                f"PnL: <code>{pnl:+,.2f}</code> USD"
            )
        else:
            return False

        return self._send(msg)

    def send_alert_triggered(self, alert: dict) -> bool:
        if not self._enabled:
            return False
        msg = (
            f"&#x1F514; <b>Alert Triggered</b>\n"
            f"Symbol: <code>{alert.get('symbol','?')}</code>\n"
            f"Type: {alert.get('type','?')}\n"
            f"Target: <code>${alert.get('target_price',0):,.2f}</code> | "
            f"Current: <code>${alert.get('current_price',0):,.2f}</code>"
        )
        return self._send(msg)

    def send_status(self, status_text: str) -> bool:
        if not self._enabled:
            return False
        return self._send(f"<pre>{status_text}</pre>")

    def send_error(self, error_msg: str) -> bool:
        if not self._enabled:
            return False
        return self._send(f"&#x26A0; <b>Error</b>\n<code>{error_msg[:500]}</code>")

    def send_startup(self) -> bool:
        if not self._enabled:
            return False
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return self._send(
            f"&#x1F680; <b>Agent Started</b>\n"
            f"Time: {now}\nReady for autonomous scanning."
        )

    def test_connection(self) -> bool:
        return self._send("&#x1F9EA; Telegram connection test -- OK!")
