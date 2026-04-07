"""
AI Brain -- Thin Orchestrator
===============================
Wires together providers, budget, prompts, memory, and parser.
Under 300 lines. All logic lives in sub-modules.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from apollo.ai.budget import TokenBudget
from apollo.ai.memory import AIMemory
from apollo.ai.parser import AIDecision, parse_response, parse_chat_response
from apollo.ai.prompt_log import PromptLogger
from apollo.ai.prompts import (
    build_system_prompt,
    build_market_prompt,
    build_chat_prompt,
)
from apollo.ai.providers import get_provider, LLMProvider
from apollo.ai.sentiment import SentimentCollector
from apollo.config import settings

logger = logging.getLogger("ai.brain")


class Brain:
    """
    AI orchestrator. Delegates everything to sub-modules.

    Usage:
        brain = Brain()
        decisions = brain.analyze_scan(scan_data, scorecard, enrichment, positions)
        response = brain.chat("How is BTCUSDT looking?")
    """

    def __init__(self, provider: LLMProvider = None):
        self.provider = provider or get_provider()
        self.budget = TokenBudget()
        self.memory = AIMemory()
        self.sentiment = SentimentCollector()
        self.prompt_log = PromptLogger()
        try:
            from apollo.ai.quality import DecisionTracker
            self.quality = DecisionTracker()
        except Exception:
            self.quality = None

    def analyze_scan(
        self,
        scan_results: list[dict],
        scorecard_summary: dict = None,
        enrichment_summary: dict = None,
        positions: list[dict] = None,
        alerts: list[dict] = None,
        symbols: list[str] = None,
        scan_id: str = None,
        correlation_prompt: str = "",
        events_prompt: str = "",
    ) -> list[AIDecision]:
        """
        Main analysis pipeline:
        1. Gather sentiment
        2. Build memory block
        3. Build prompt
        4. Select model (budget-aware)
        5. Call LLM
        6. Parse response
        7. Store predictions + self-notes
        """
        scan_id = scan_id or str(uuid.uuid4())[:12]
        symbols = symbols or [r.get("symbol", "") for r in scan_results]

        # 1. Sentiment
        try:
            sentiment = self.sentiment.get_market_summary(symbols)
        except Exception as e:
            logger.warning("Sentiment collection failed: %s", e)
            sentiment = {}

        # 2. Memory context
        memory_block = ""
        pairs_for_memory = []
        for result in scan_results:
            sym = result.get("symbol", "")
            regime = result.get("regime", {}).get("label", "")
            if sym and regime:
                pairs_for_memory.append((sym, regime))

        if pairs_for_memory:
            memory_block = self.memory.get_multi_pair_context(pairs_for_memory)

        # Add active self-notes that are NOT already shown in per-pair memory
        active_notes = self.memory.get_active_notes()
        if active_notes:
            # Deduplicate: only show notes not already in per-pair memory
            existing = set()
            if memory_block:
                for line in memory_block.splitlines():
                    stripped = line.strip().lstrip("> ").strip()
                    if stripped:
                        existing.add(stripped)
            unique_notes = [n for n in active_notes if n.strip() not in existing]
            if unique_notes:
                notes_block = "=== YOUR ACTIVE NOTES ===\n"
                notes_block += "\n".join(f"  > {n}" for n in unique_notes)
                memory_block = notes_block + "\n\n" + memory_block if memory_block else notes_block

        # 3. Build prompts
        compact = self.budget.should_compress
        system_prompt = build_system_prompt(compact=compact)

        # Collect extra context blocks
        extra = []
        if correlation_prompt:
            extra.append(correlation_prompt)
        if events_prompt:
            extra.append(events_prompt)
        # Add quality self-awareness
        if self.quality:
            try:
                quality_block = self.quality.to_prompt_block()
                if quality_block:
                    extra.append(quality_block)
            except Exception:
                pass

        user_prompt = build_market_prompt(
            scan_results=scan_results,
            scorecard_summary=scorecard_summary,
            enrichment_summary=enrichment_summary,
            sentiment=sentiment,
            positions=positions,
            alerts=alerts,
            memory_block=memory_block,
            extra_context=extra,
        )

        if compact:
            user_prompt = TokenBudget.compress_prompt(user_prompt)

        # 4. Select model
        est_input = len(system_prompt + user_prompt) // 4
        est_output = 500
        model_key = self.budget.select_model(
            task_tier=settings.apollo_default_tier, est_input_tokens=est_input, est_output_tokens=est_output,
        )
        model_name = self.budget.get_model_name(model_key)

        # 5. Call LLM (with retry on empty response)
        logger.info("Calling %s with %d char prompt...", model_name, len(user_prompt))
        text = ""
        in_tokens = 0
        out_tokens = 0
        for attempt in range(2):
            try:
                text, in_tokens, out_tokens = self.provider.call(
                    model=model_name,
                    system=system_prompt,
                    user=user_prompt,
                )
            except Exception as e:
                logger.error("LLM call failed: %s", e)
                return []

            if text and text.strip():
                break
            logger.warning("LLM returned empty response (attempt %d/2)", attempt + 1)

        if not text or not text.strip():
            logger.error("LLM returned empty response after 2 attempts")
            return []

        # Record usage
        cost = self.budget.record_usage(
            model_key, in_tokens, out_tokens,
            task_tier=2, description=f"scan_{scan_id}",
        )
        logger.info("LLM response: %d tokens | $%.4f", in_tokens + out_tokens, cost)

        # Log the full interaction
        self.prompt_log.log_interaction(
            scan_id=scan_id,
            model=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=text,
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            cost=cost,
            task_type="scan",
        )

        # 6. Parse
        decisions = parse_response(text)

        # 7. Store predictions + notes
        for dec in decisions:
            if dec.is_actionable:
                # Find price from scan results
                price = 0.0
                ensemble = 0.0
                regime_label = ""
                for r in scan_results:
                    if r.get("symbol") == dec.symbol:
                        price = r.get("current_price", 0)
                        ensemble = r.get("signals", {}).get("ensemble", 0)
                        regime_label = r.get("regime", {}).get("label", "")
                        break

                self.memory.store_prediction(
                    scan_id=scan_id, symbol=dec.symbol, price=price,
                    direction=dec.action, confidence=dec.confidence,
                    regime_label=regime_label, ensemble_signal=ensemble,
                    reasoning=dec.reasoning, model_used=model_name,
                    tokens=in_tokens + out_tokens, cost=cost / max(len(decisions), 1),
                )

            # Store self-notes
            if dec.self_notes:
                self.memory.store_self_note(dec.self_notes, symbol=dec.symbol)

            # Store self-errors as lessons
            if dec.self_errors:
                self.memory.store_lesson(
                    lesson_type="self_error",
                    lesson_text=dec.self_errors,
                    symbol=dec.symbol,
                )

        # Store scan
        n_actionable = sum(1 for d in decisions if d.is_actionable)
        regime_summary = "; ".join(
            f"{r.get('symbol','?')}={r.get('regime',{}).get('label','?')}"
            for r in scan_results[:5]
        )
        self.memory.store_scan(
            scan_id=scan_id, n_pairs=len(scan_results),
            n_opps=n_actionable, regime_summary=regime_summary,
            tokens=in_tokens + out_tokens, model=model_name, cost=cost,
        )

        logger.info(
            "Analysis complete: %d decisions (%d actionable) from %d pairs",
            len(decisions), n_actionable, len(scan_results),
        )
        return decisions

    def analyze_deep(
        self,
        scan_results: list[dict],
        scorecard_summary: dict = None,
        enrichment_summary: dict = None,
        positions: list[dict] = None,
    ) -> list[AIDecision]:
        """
        Two-step analysis:
        1. Cheap pre-filter (tier 1) to identify top opportunities
        2. Deep dive (tier 2+) on filtered pairs only
        """
        # Step 1: Quick pre-filter
        compact_system = build_system_prompt(compact=True)
        compact_user = build_market_prompt(scan_results=scan_results)

        if self.budget.should_compress:
            compact_user = TokenBudget.compress_prompt(compact_user, target_reduction=0.50)

        filter_model_key = self.budget.select_model(
            task_tier=1,
            est_input_tokens=len(compact_system + compact_user) // 4,
            est_output_tokens=200,
        )
        filter_model = self.budget.get_model_name(filter_model_key)

        filter_prompt = (
            compact_user + "\n\n"
            "List ONLY the top 3-5 symbols worth deep analysis. "
            "Format: SYMBOLS: SYM1, SYM2, SYM3"
        )

        try:
            text, in_t, out_t = self.provider.call(
                model=filter_model, system=compact_system,
                user=filter_prompt, max_tokens=200,
            )
            self.budget.record_usage(filter_model_key, in_t, out_t, task_tier=1,
                                     description="pre_filter")
        except Exception as e:
            logger.warning("Pre-filter failed: %s -- falling back to full scan", e)
            return self.analyze_scan(
                scan_results, scorecard_summary, enrichment_summary, positions,
            )

        # Extract symbols from response
        import re
        sym_match = re.search(r'SYMBOLS?:\s*([\w,\s]+)', text, re.IGNORECASE)
        if sym_match:
            found = [s.strip().upper() for s in sym_match.group(1).split(",")]
            found = [s for s in found if s.endswith("USDT")]
        else:
            found = [r.get("symbol", "") for r in scan_results[:5]]

        # Step 2: Deep dive on filtered pairs
        filtered = [r for r in scan_results if r.get("symbol") in found]
        if not filtered:
            filtered = scan_results[:5]

        logger.info("Deep analysis: %d pairs pre-filtered -> %d",
                     len(scan_results), len(filtered))

        return self.analyze_scan(
            filtered, scorecard_summary, enrichment_summary, positions,
        )

    def chat(self, user_message: str, positions: list[dict] = None) -> str:
        """
        Telegram chat mode -- memory + positions context only.
        No full market scan, cheap model.
        """
        memory_block = ""
        active_notes = self.memory.get_active_notes()
        if active_notes:
            memory_block = "Your active notes:\n" + "\n".join(f"  > {n}" for n in active_notes)

        stats = self.memory.format_stats()
        if stats:
            memory_block += f"\n\nMemory stats: {stats}"

        system = "You are a quant trading assistant. Answer based on your memory and current positions. Be concise."
        user = build_chat_prompt(user_message, memory_block, positions)

        model_key = self.budget.select_model(task_tier=1, est_input_tokens=500, est_output_tokens=300)
        model_name = self.budget.get_model_name(model_key)

        try:
            text, in_t, out_t = self.provider.call(
                model=model_name, system=system, user=user,
                max_tokens=500, temperature=0.5,
            )
            self.budget.record_usage(model_key, in_t, out_t, task_tier=1, description="chat")
            self.prompt_log.log_interaction(
                model=model_name,
                system_prompt=system,
                user_prompt=user,
                response=text,
                in_tokens=in_t,
                out_tokens=out_t,
                cost=0,
                task_type="chat",
            )
            return parse_chat_response(text)
        except Exception as e:
            logger.error("Chat failed: %s", e)
            return f"Error: {e}"
