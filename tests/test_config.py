"""
Tests for config -- Settings validation and behavior.
"""

import pytest


class TestSettings:
    """Validate config loading, defaults, auto-detection, and validation."""

    def test_loads_with_defaults(self):
        from apollo.config import Settings
        s = Settings(_env_file=None, google_api_key="test-key")
        assert s.apollo_daily_budget == 50.0
        assert s.apollo_default_tier == 2
        assert s.apollo_scan_interval_hours == 2.0

    def test_auto_detect_google(self):
        from apollo.config import Settings
        s = Settings(
            _env_file=None,
            google_api_key="gk-123",
            openai_api_key="sk-456",
            apollo_ai_provider="auto",
        )
        assert s.active_ai_provider == "google"
        assert s.active_ai_key == "gk-123"

    def test_auto_detect_fallback_openai(self):
        from apollo.config import Settings
        s = Settings(
            _env_file=None,
            google_api_key="",
            openai_api_key="sk-real",
            apollo_ai_provider="auto",
        )
        assert s.active_ai_provider == "openai"

    def test_auto_detect_none(self):
        from apollo.config import Settings
        s = Settings(
            _env_file=None,
            google_api_key="",
            openai_api_key="",
            anthropic_api_key="",
            apollo_ai_provider="auto",
        )
        assert s.active_ai_provider is None
        assert s.has_ai is False

    def test_placeholder_keys_stripped(self):
        from apollo.config import Settings
        s = Settings(_env_file=None, google_api_key="your-google-api-key-here")
        assert s.google_api_key == ""
        assert s.has_ai is False

    def test_telegram_requires_both(self):
        """With only bot_token, has_telegram is True but chat_id discovery will be attempted."""
        from apollo.config import Settings
        s = Settings(
            _env_file=None,
            telegram_bot_token="abc123",
            telegram_chat_id="",
        )
        # has_telegram should be True since bot_token is set
        assert s.has_telegram is True
        # bot_token should NOT be cleared anymore
        assert s.telegram_bot_token == "abc123"

    def test_telegram_complete(self):
        from apollo.config import Settings
        s = Settings(
            _env_file=None,
            telegram_bot_token="abc",
            telegram_chat_id="123",
        )
        assert s.has_telegram is True

    def test_forced_provider_with_key(self):
        from apollo.config import Settings
        s = Settings(
            _env_file=None,
            openai_api_key="sk-real",
            apollo_ai_provider="openai",
        )
        assert s.active_ai_provider == "openai"

    def test_forced_provider_without_key(self):
        from apollo.config import Settings
        s = Settings(
            _env_file=None,
            openai_api_key="",
            apollo_ai_provider="openai",
        )
        assert s.active_ai_provider is None

    def test_budget_zero_allowed(self):
        from apollo.config import Settings
        s = Settings(_env_file=None, apollo_daily_budget=0.0)
        assert s.apollo_daily_budget == 0.0

    def test_tier_bounds(self):
        from apollo.config import Settings
        with pytest.raises(Exception):
            Settings(_env_file=None, apollo_default_tier=5)

    def test_status_output(self):
        from apollo.config import Settings
        s = Settings(_env_file=None, google_api_key="test")
        status = s.status()
        assert "Configuration" in status
        assert "google" in status

    def test_ensure_dirs(self, tmp_path):
        from apollo.config import Settings
        s = Settings(_env_file=None, project_root=tmp_path)
        s.ensure_dirs()
        assert (tmp_path / "models").exists()
        assert (tmp_path / "data").exists()
        assert (tmp_path / "logs").exists()
