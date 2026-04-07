"""
Telegram Command Handler
==========================
Polls for incoming Telegram messages and dispatches commands.
Runs in a separate thread within the Agent loop.

Commands:
  /positions  -- show open paper trades
  /portfolio  -- portfolio stats
  /memory     -- AI memory summary
  /status     -- agent + budget status
  /ask [text] -- chat with AI
  /scan       -- force immediate scan (sets flag)
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
import urllib.error
from typing import Callable, Optional

logger = logging.getLogger("alerts.handler")


class TelegramHandler:
    """Polls Telegram for commands and dispatches them."""

    API_BASE = "https://api.telegram.org/bot{token}"

    def __init__(self, bot_token: str = None, chat_id: str = None):
        from apollo.config import settings
        self._token = bot_token or settings.telegram_bot_token
        self._chat_id = chat_id or settings.telegram_chat_id
        self._enabled = bool(self._token and self._chat_id)
        self._last_update_id = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._force_scan = False

        # Command handlers (set externally)
        self._handlers: dict[str, Callable] = {}

    @property
    def force_scan_requested(self) -> bool:
        if self._force_scan:
            self._force_scan = False
            return True
        return False

    def register_handler(self, command: str, handler: Callable):
        """Register a command handler. Handler receives args string, returns response text."""
        self._handlers[command] = handler

    def start(self):
        """Start polling in a background thread."""
        if not self._enabled:
            logger.info("Telegram handler disabled")
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("Telegram handler started (polling)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _poll_loop(self):
        while self._running:
            try:
                updates = self._get_updates()
                for update in updates:
                    self._process_update(update)
            except Exception as e:
                logger.error("Telegram poll error: %s", e)
            time.sleep(3)  # Poll every 3 seconds

    def _get_updates(self) -> list[dict]:
        url = f"{self.API_BASE.format(token=self._token)}/getUpdates"
        params = f"?offset={self._last_update_id + 1}&timeout=5"
        try:
            req = urllib.request.Request(url + params)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                if data.get("ok"):
                    return data.get("result", [])
        except Exception:
            pass
        return []

    def _process_update(self, update: dict):
        update_id = update.get("update_id", 0)
        if update_id > self._last_update_id:
            self._last_update_id = update_id

        message = update.get("message", {})
        chat_id = str(message.get("chat", {}).get("id", ""))
        text = message.get("text", "").strip()

        # Only respond to our configured chat
        if chat_id != self._chat_id or not text:
            return

        logger.info("Telegram command: %s", text[:50])

        if text.startswith("/"):
            parts = text.split(maxsplit=1)
            command = parts[0].lower()
            # Strip @botname from commands (e.g. /positions@mybot -> /positions)
            if "@" in command:
                command = command.split("@")[0]
            args = parts[1] if len(parts) > 1 else ""

            if command == "/scan":
                self._force_scan = True
                self._reply("Scan requested. Will run on next cycle.")
                return

            handler = self._handlers.get(command)
            if handler:
                try:
                    response = handler(args)
                    self._reply(response)
                except Exception as e:
                    self._reply(f"Error: {e}")
            else:
                commands = "\n".join([
                    "/positions - Open trades",
                    "/portfolio - Stats",
                    "/memory - AI memory",
                    "/status - Agent status",
                    "/scan - Force scan",
                    "/prompts - Prompt log stats",
                    "/ask [q] - Chat with AI",
                ])
                self._reply(f"Commands:\n{commands}")
        else:
            # Free-form text -> treat as /ask
            handler = self._handlers.get("/ask")
            if handler:
                try:
                    response = handler(text)
                    self._reply(response)
                except Exception as e:
                    self._reply(f"Error: {e}")

    def _reply(self, text: str):
        if not self._enabled:
            return
        url = f"{self.API_BASE.format(token=self._token)}/sendMessage"
        # Truncate for Telegram limit
        if len(text) > 4000:
            text = text[:4000] + "..."
        payload = json.dumps({
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            logger.error("Reply failed: %s", e)
