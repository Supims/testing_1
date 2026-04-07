"""
Token Budget Manager
=====================
Cost-aware AI model routing with daily/weekly spend tracking.

Tiered strategy:
  TIER 1 (cheap/fast)  -- pre-filtering, quick checks
  TIER 2 (standard)    -- main analysis
  TIER 3 (premium)     -- deep reasoning

Model selection respects .env settings:
  MAIN_MODEL, BACKUP_MODEL, PREMIUM_MODEL
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ai.budget")


# -- Pricing ($ per 1M tokens, 2026-Q2) ------------------------------------

@dataclass
class ModelPricing:
    name: str
    provider: str
    input_per_1m: float
    output_per_1m: float
    context_window: int
    tier: int

MODEL_CATALOG = {
    "gpt-4o-mini": ModelPricing("gpt-4o-mini", "openai", 0.15, 0.60, 128_000, 1),
    "gpt-4o": ModelPricing("gpt-4o", "openai", 2.50, 10.00, 128_000, 2),
    "o1": ModelPricing("o1", "openai", 15.00, 60.00, 200_000, 3),
    "claude-haiku": ModelPricing("claude-3-5-haiku-latest", "anthropic", 0.80, 4.00, 200_000, 1),
    "claude-sonnet": ModelPricing("claude-sonnet-4-20250514", "anthropic", 3.00, 15.00, 200_000, 2),
    "claude-opus": ModelPricing("claude-opus-4-20250514", "anthropic", 15.00, 75.00, 200_000, 3),
    "gemini-flash": ModelPricing("gemini-2.5-flash", "google", 0.15, 0.60, 1_000_000, 1),
    "gemini-pro": ModelPricing("gemini-2.5-pro", "google", 1.25, 10.00, 1_000_000, 2),
}


@dataclass
class UsageRecord:
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    task_tier: int
    description: str = ""


@dataclass
class BudgetConfig:
    daily_limit_usd: float = 50.0
    weekly_limit_usd: float = 200.0
    preferred_provider: str = "google"
    allow_tier3: bool = True
    compression_threshold: float = 0.70
    emergency_model: str = "gemini-flash"


class TokenBudget:
    """Manages token usage, costs, and model routing."""

    def __init__(self, config: BudgetConfig = None, usage_file: Path = None):
        from apollo.config import settings
        self.config = config or BudgetConfig(
            daily_limit_usd=settings.apollo_daily_budget,
            weekly_limit_usd=settings.apollo_weekly_budget,
            preferred_provider=settings.active_ai_provider or "google",
            allow_tier3=settings.apollo_allow_tier3,
        )
        self._usage_file = usage_file or (settings.models_dir / "token_usage.json")
        self.usage_history: list[UsageRecord] = []
        self._load_history()

    def _load_history(self):
        if self._usage_file.exists():
            try:
                data = json.loads(self._usage_file.read_text(encoding="utf-8"))
                for entry in data.get("history", []):
                    self.usage_history.append(UsageRecord(**entry))
            except Exception as e:
                logger.warning("Failed to load usage history: %s", e)

    def _save_history(self):
        self._usage_file.parent.mkdir(parents=True, exist_ok=True)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        recent = [u for u in self.usage_history if u.timestamp >= cutoff]
        self.usage_history = recent
        data = {"history": [
            {
                "timestamp": u.timestamp, "model": u.model,
                "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                "cost_usd": u.cost_usd, "task_tier": u.task_tier,
                "description": u.description,
            }
            for u in recent
        ]}
        self._usage_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # -- Cost calculation ---------------------------------------------------

    @staticmethod
    def calculate_cost(model_key: str, input_tokens: int, output_tokens: int) -> float:
        pricing = MODEL_CATALOG.get(model_key)
        if not pricing:
            return 0.0
        input_cost = (input_tokens / 1_000_000) * pricing.input_per_1m
        output_cost = (output_tokens / 1_000_000) * pricing.output_per_1m
        return round(input_cost + output_cost, 6)

    def get_daily_spend(self, day: Optional[date] = None) -> float:
        target = (day or datetime.now(timezone.utc).date()).isoformat()
        return sum(u.cost_usd for u in self.usage_history if u.timestamp[:10] == target)

    def get_weekly_spend(self) -> float:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        return sum(u.cost_usd for u in self.usage_history if u.timestamp >= cutoff)

    @property
    def daily_remaining(self) -> float:
        return max(0, self.config.daily_limit_usd - self.get_daily_spend())

    @property
    def weekly_remaining(self) -> float:
        return max(0, self.config.weekly_limit_usd - self.get_weekly_spend())

    @property
    def budget_pressure(self) -> float:
        daily_pct = self.get_daily_spend() / max(self.config.daily_limit_usd, 0.01)
        weekly_pct = self.get_weekly_spend() / max(self.config.weekly_limit_usd, 0.01)
        return max(daily_pct, weekly_pct)

    @property
    def should_compress(self) -> bool:
        return self.budget_pressure >= self.config.compression_threshold

    # -- Model selection ----------------------------------------------------

    def select_model(self, task_tier: int = 2, est_input_tokens: int = 2000,
                     est_output_tokens: int = 1000) -> str:
        # Budget override
        if self.daily_remaining < 0.10 or self.weekly_remaining < 0.50:
            logger.warning(
                "Budget critically low (daily: $%.2f, weekly: $%.2f). Forcing %s",
                self.daily_remaining, self.weekly_remaining, self.config.emergency_model,
            )
            return self.config.emergency_model

        effective_tier = task_tier
        if self.budget_pressure > 0.85 and task_tier >= 3:
            effective_tier = 2
        elif self.budget_pressure > 0.95 and task_tier >= 2:
            effective_tier = 1

        if not self.config.allow_tier3 and effective_tier >= 3:
            effective_tier = 2

        total_tokens = est_input_tokens + est_output_tokens
        candidates = [
            (key, m) for key, m in MODEL_CATALOG.items()
            if m.tier <= effective_tier and m.context_window >= total_tokens
        ]
        if not candidates:
            return self.config.emergency_model

        # Prefer provider, then best tier, then cheapest
        provider_matches = [
            (k, m) for k, m in candidates
            if m.provider == self.config.preferred_provider
        ]
        pool = provider_matches if provider_matches else candidates

        best_tier = max(m.tier for _, m in pool)
        best_tier_models = [(k, m) for k, m in pool if m.tier == best_tier]
        best_tier_models.sort(
            key=lambda x: self.calculate_cost(x[0], est_input_tokens, est_output_tokens)
        )

        selected = best_tier_models[0][0]
        est_cost = self.calculate_cost(selected, est_input_tokens, est_output_tokens)
        logger.info("Model selected: %s (tier %d) -- est. $%.4f",
                     selected, MODEL_CATALOG[selected].tier, est_cost)
        return selected

    # -- Recording ----------------------------------------------------------

    def record_usage(self, model_key: str, input_tokens: int, output_tokens: int,
                     task_tier: int = 2, description: str = "") -> float:
        cost = self.calculate_cost(model_key, input_tokens, output_tokens)
        record = UsageRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=model_key, input_tokens=input_tokens,
            output_tokens=output_tokens, cost_usd=cost,
            task_tier=task_tier, description=description,
        )
        self.usage_history.append(record)
        self._save_history()
        logger.info("Recorded: %s | %d+%d tokens | $%.4f | Daily: $%.4f",
                     model_key, input_tokens, output_tokens, cost, self.get_daily_spend())
        return cost

    # -- Prompt compression -------------------------------------------------

    @staticmethod
    def compress_prompt(prompt: str, target_reduction: float = 0.40) -> str:
        original_len = len(prompt)
        target_len = int(original_len * (1 - target_reduction))

        lines = prompt.split("\n")
        compressed = []
        for line in lines:
            stripped = line.strip()
            if all(c in '=-_|+' for c in stripped) and len(stripped) > 3:
                continue
            compressed.append(line)

        result = "\n".join(compressed)
        while "\n\n\n" in result:
            result = result.replace("\n\n\n", "\n\n")

        # Remove enrichment/scorecard sections first (least critical)
        if len(result) > target_len:
            for section in ["SCORECARD:", "ENRICHMENT:", "SENTIMENT:"]:
                idx = result.find(section)
                if idx > 0:
                    end = result.find("\n\n", idx + 10)
                    if end > 0:
                        result = result[:idx] + result[end:]

        savings = (1 - len(result) / max(original_len, 1)) * 100
        logger.info("Prompt compressed: %d -> %d chars (%.0f%% saved)",
                     original_len, len(result), savings)
        return result

    # -- Status display -----------------------------------------------------

    def get_model_name(self, model_key: str) -> str:
        m = MODEL_CATALOG.get(model_key)
        return m.name if m else model_key

    def get_model_provider(self, model_key: str) -> str:
        m = MODEL_CATALOG.get(model_key)
        return m.provider if m else "openai"

    def status(self) -> str:
        daily = self.get_daily_spend()
        weekly = self.get_weekly_spend()
        pressure = self.budget_pressure
        bar_len = int(min(pressure, 1.0) * 20)
        bar = "#" * bar_len + "." * (20 - bar_len)

        return (
            f"Budget: ${daily:.4f}/${self.config.daily_limit_usd:.2f} daily "
            f"[{bar}] {pressure:.0%}\n"
            f"Weekly: ${weekly:.4f}/${self.config.weekly_limit_usd:.2f} | "
            f"Remaining: ${self.daily_remaining:.4f}"
        )
