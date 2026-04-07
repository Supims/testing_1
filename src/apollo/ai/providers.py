"""
LLM Provider Abstraction
=========================
Clean multi-provider interface for Google, OpenAI, and Anthropic.
Each provider is its own class. Factory auto-detects from settings.

All providers return: (response_text, input_tokens, output_tokens)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Protocol, Tuple

logger = logging.getLogger("ai.providers")


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    def call(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> Tuple[str, int, int]:
        """Returns (response_text, input_tokens, output_tokens)."""
        ...


class GoogleProvider:
    """Google Gemini via OpenAI-compatible endpoint."""

    def __init__(self, api_key: str, timeout: float = 120.0):
        self._api_key = api_key
        self._timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            import httpx
            self._client = OpenAI(
                api_key=self._api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                timeout=httpx.Timeout(self._timeout, connect=15.0),
            )
        return self._client

    def call(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> Tuple[str, int, int]:
        client = self._get_client()
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = response.choices[0].message.content or ""
                usage = response.usage
                in_tok = usage.prompt_tokens if usage else _estimate_tokens(system + user)
                out_tok = usage.completion_tokens if usage else _estimate_tokens(text)
                return text, in_tok, out_tok
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    wait = 2 ** (attempt + 1)
                    logger.warning("Google rate limit, retrying in %ds...", wait)
                    time.sleep(wait)
                    continue
                raise


class OpenAIProvider:
    """OpenAI (or any OpenAI-compatible endpoint)."""

    def __init__(self, api_key: str, base_url: str = None, timeout: float = 120.0):
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            import httpx
            self._client = OpenAI(
                api_key=self._api_key, base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout, connect=15.0),
            )
        return self._client

    def call(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> Tuple[str, int, int]:
        client = self._get_client()
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = response.choices[0].message.content or ""
                usage = response.usage
                return text, usage.prompt_tokens, usage.completion_tokens
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    wait = 2 ** (attempt + 1)
                    logger.warning("OpenAI rate limit, retrying in %ds...", wait)
                    time.sleep(wait)
                    continue
                raise


class AnthropicProvider:
    """Anthropic Claude API."""

    def __init__(self, api_key: str, timeout: float = 120.0):
        self._api_key = api_key
        self._timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=self._api_key,
                timeout=self._timeout,
            )
        return self._client

    def call(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> Tuple[str, int, int]:
        client = self._get_client()
        for attempt in range(3):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                text = response.content[0].text
                return text, response.usage.input_tokens, response.usage.output_tokens
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    wait = 2 ** (attempt + 1)
                    logger.warning("Anthropic rate limit, retrying in %ds...", wait)
                    time.sleep(wait)
                    continue
                raise


class DummyProvider:
    """No-op provider when no API keys are configured."""

    def call(self, model: str, system: str, user: str,
             max_tokens: int = 4096, temperature: float = 0.3) -> Tuple[str, int, int]:
        logger.warning("No LLM provider configured -- returning empty response")
        return "[NO_PROVIDER] No AI provider configured.", 0, 0


def get_provider(provider_name: str = None) -> LLMProvider:
    """
    Factory -- returns the appropriate LLM provider.

    Auto-detects from settings if provider_name is None.
    """
    from apollo.config import settings

    name = provider_name or settings.active_ai_provider

    if name == "google" and settings.google_api_key:
        logger.info("Using Google Gemini provider")
        return GoogleProvider(settings.google_api_key)

    if name == "openai" and settings.openai_api_key:
        base_url = os.environ.get("OPENAI_BASE_URL")
        logger.info("Using OpenAI provider")
        return OpenAIProvider(settings.openai_api_key, base_url=base_url)

    if name == "anthropic" and settings.anthropic_api_key:
        logger.info("Using Anthropic provider")
        return AnthropicProvider(settings.anthropic_api_key)

    logger.warning("No AI provider detected -- using DummyProvider")
    return DummyProvider()


def _estimate_tokens(text: str) -> int:
    """Rough token estimate when the API doesn't report usage."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        return len(enc.encode(text))
    except (ImportError, KeyError):
        pass
    return int(len(text) / 3.8)
