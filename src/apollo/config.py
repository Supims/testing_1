"""
Configuration
=============
Single validated config. Loads from .env + environment variables.
Replaces scattered .env parsing across multiple old modules.

Usage:
    from apollo.config import settings
    print(settings.google_api_key)
    print(settings.scan_interval_hours)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("config")

# Project root = two levels up from this file (new/src/apollo/config.py -> new/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """
    Validated configuration loaded from .env + environment.
    Every field has a sensible default except API keys.
    """

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- AI Provider Keys --------------------------------------------------
    google_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # -- AI Behavior -------------------------------------------------------
    apollo_ai_provider: Literal["google", "openai", "anthropic", "auto"] = "auto"
    apollo_daily_budget: float = Field(default=50.0, ge=0.0)
    apollo_weekly_budget: float = Field(default=200.0, ge=0.0)
    apollo_default_tier: int = Field(default=2, ge=1, le=3)
    apollo_allow_tier3: bool = True
    apollo_max_pairs: int = Field(default=15, ge=1, le=50)

    # -- Telegram ----------------------------------------------------------
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # -- Agent -------------------------------------------------------------
    apollo_scan_interval_hours: float = Field(default=2.0, ge=0.1, le=24.0)
    apollo_adaptive_scan: bool = True
    apollo_alert_min_confidence: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"
    apollo_deep_analysis: bool = True

    # -- Nansen (optional) -------------------------------------------------
    nansen_api_key: str = ""

    # -- Paths (computed, not from env) ------------------------------------
    project_root: Path = _PROJECT_ROOT

    # ======================================================================
    # Computed properties
    # ======================================================================

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"

    @property
    def has_ai(self) -> bool:
        return bool(self.active_ai_provider)

    @property
    def has_telegram(self) -> bool:
        return bool(self.telegram_bot_token)

    @property
    def has_nansen(self) -> bool:
        return bool(self.nansen_api_key)

    @property
    def active_ai_provider(self) -> Optional[str]:
        """Auto-detect the best available AI provider."""
        if self.apollo_ai_provider != "auto":
            key_map = {
                "google": self.google_api_key,
                "openai": self.openai_api_key,
                "anthropic": self.anthropic_api_key,
            }
            if key_map.get(self.apollo_ai_provider):
                return self.apollo_ai_provider
            return None

        # Auto-detect: prefer Google (cheapest), then OpenAI, then Anthropic
        if self.google_api_key:
            return "google"
        if self.openai_api_key:
            return "openai"
        if self.anthropic_api_key:
            return "anthropic"
        return None

    @property
    def active_ai_key(self) -> Optional[str]:
        """Returns the API key for the active provider."""
        provider = self.active_ai_provider
        if provider == "google":
            return self.google_api_key
        if provider == "openai":
            return self.openai_api_key
        if provider == "anthropic":
            return self.anthropic_api_key
        return None

    # ======================================================================
    # Validators
    # ======================================================================

    @field_validator("google_api_key", "openai_api_key", "anthropic_api_key")
    @classmethod
    def strip_placeholder_keys(cls, v: str) -> str:
        """Remove placeholder values that are not real keys."""
        placeholders = {"your-google-api-key-here", "sk-your-openai-key-here",
                        "sk-ant-your-anthropic-key-here", "placeholder", ""}
        return "" if v.strip().lower() in placeholders else v.strip()

    @model_validator(mode="after")
    def validate_telegram_pair(self):
        """Telegram: auto-discover chat_id if bot_token is set but chat_id is missing."""
        if self.telegram_bot_token and not self.telegram_chat_id:
            # Try to auto-discover chat_id from recent updates
            discovered = self._discover_telegram_chat_id(self.telegram_bot_token)
            if discovered:
                self.telegram_chat_id = discovered
                logger.info("Telegram chat_id auto-discovered: %s", discovered)
            else:
                logger.warning(
                    "Telegram bot_token set but no chat_id found. "
                    "Send a message to your bot first, then restart."
                )
        return self

    @staticmethod
    def _discover_telegram_chat_id(token: str = None) -> str:
        """Try to discover chat_id by reading recent messages to the bot."""
        import json
        import urllib.request
        try:
            # Use the instance's token if none provided
            url = f"https://api.telegram.org/bot{token}/getUpdates?limit=10"
            req = urllib.request.Request(url, headers={"User-Agent": "apollo/2.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                if data.get("ok") and data.get("result"):
                    # Get the most recent message's chat_id
                    for update in reversed(data["result"]):
                        msg = update.get("message", {})
                        chat_id = msg.get("chat", {}).get("id")
                        if chat_id:
                            return str(chat_id)
        except Exception:
            pass
        return ""

    # ======================================================================
    # Utilities
    # ======================================================================

    def ensure_dirs(self):
        """Create runtime directories if they do not exist."""
        for d in (self.models_dir, self.data_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    def status(self) -> str:
        """Human-readable config status."""
        lines = [
            "=" * 50,
            "  Configuration",
            "=" * 50,
            f"  AI Provider   : {self.active_ai_provider or 'NONE'}",
            f"  AI Budget     : ${self.apollo_daily_budget}/day, ${self.apollo_weekly_budget}/week",
            f"  Default Tier  : {self.apollo_default_tier} (Tier 3: {'yes' if self.apollo_allow_tier3 else 'no'})",
            f"  Max Pairs     : {self.apollo_max_pairs}",
            f"  Scan Interval : {self.apollo_scan_interval_hours}h (adaptive: {'yes' if self.apollo_adaptive_scan else 'no'})",
            f"  Deep Analysis : {'yes' if self.apollo_deep_analysis else 'no'}",
            f"  Alert Level   : {self.apollo_alert_min_confidence}",
            f"  Telegram      : {'connected' if self.has_telegram else 'not configured'}",
            f"  Nansen        : {'connected' if self.has_nansen else 'not configured'}",
            f"  Project Root  : {self.project_root}",
            "=" * 50,
        ]
        return "\n".join(lines)


# -- Singleton --------------------------------------------------------------
# Import anywhere: `from apollo.config import settings`

def _load_settings() -> Settings:
    """Load settings, falling back gracefully if .env does not exist."""
    try:
        s = Settings()
        s.ensure_dirs()
        return s
    except Exception as e:
        logger.warning(f"Config load issue: {e} -- using defaults")
        s = Settings(
            _env_file=None,
            project_root=_PROJECT_ROOT,
        )
        s.ensure_dirs()
        return s


settings = _load_settings()
