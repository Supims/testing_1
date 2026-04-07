"""
AI Response Parser
===================
Parses structured AI responses into typed AIDecision objects.
Extracted from brain.py for single responsibility.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("ai.parser")


@dataclass
class AIDecision:
    """A single parsed AI trading decision."""
    symbol: str = ""
    action: str = "SKIP"           # LONG / SHORT / CLOSE / SKIP
    confidence: str = "LOW"        # HIGH / MEDIUM / LOW
    reasoning: str = ""
    self_notes: str = ""           # Notes for future self
    self_errors: str = ""          # Self-correction
    sl_price: float = 0.0
    tp_price: float = 0.0
    alert_type: str = "NONE"      # SET_LONG_ALERT / SET_SHORT_ALERT / NONE
    alert_price: float = 0.0
    market_assessment: str = ""

    @property
    def is_actionable(self) -> bool:
        return self.action in ("LONG", "SHORT") and self.symbol != ""

    @property
    def is_close(self) -> bool:
        return self.action == "CLOSE" and self.symbol != ""

    @property
    def has_alert(self) -> bool:
        return self.alert_type != "NONE" and self.alert_price > 0


def parse_response(text: str) -> list[AIDecision]:
    """
    Parse AI response text into a list of AIDecision objects.

    Handles the structured format:
        DECISION: [action] [symbol]
        CONFIDENCE: [level]
        REASONING: [text]
        SL: [price] | TP: [price]
        ALERT: [type] [price]
        SELF_NOTES: [text]
        SELF_ERRORS: [text]
    """
    decisions = []

    # Extract market assessment
    market_match = re.search(
        r'MARKET:\s*(.*?)(?=\nDECISION:|$)', text, re.DOTALL | re.IGNORECASE
    )
    market_assessment = market_match.group(1).strip() if market_match else ""

    # Split by DECISION: blocks
    decision_blocks = re.split(r'(?=DECISION:)', text, flags=re.IGNORECASE)

    for block in decision_blocks:
        block = block.strip()
        if not block.upper().startswith("DECISION:"):
            continue

        decision = AIDecision(market_assessment=market_assessment)

        # Parse DECISION: [action] [symbol]
        dec_match = re.search(
            r'DECISION:\s*(LONG|SHORT|CLOSE|SKIP)\s+(\w+)',
            block, re.IGNORECASE
        )
        if dec_match:
            decision.action = dec_match.group(1).upper()
            decision.symbol = dec_match.group(2).upper()
            # Ensure symbol ends with USDT
            if not decision.symbol.endswith("USDT"):
                decision.symbol += "USDT"

        # Parse CONFIDENCE
        conf_match = re.search(
            r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', block, re.IGNORECASE
        )
        if conf_match:
            decision.confidence = conf_match.group(1).upper()

        # Parse REASONING
        reason_match = re.search(
            r'REASONING:\s*(.*?)(?=\n(?:SL:|ALERT:|SELF_|DECISION:)|$)',
            block, re.DOTALL | re.IGNORECASE
        )
        if reason_match:
            decision.reasoning = reason_match.group(1).strip()

        # Parse SL / TP
        sl_tp_match = re.search(
            r'SL:\s*\$?([\d,.]+)\s*\|?\s*TP:\s*\$?([\d,.]+)',
            block, re.IGNORECASE
        )
        if sl_tp_match:
            try:
                decision.sl_price = float(sl_tp_match.group(1).replace(",", ""))
                decision.tp_price = float(sl_tp_match.group(2).replace(",", ""))
            except ValueError:
                pass

        # Parse ALERT
        alert_match = re.search(
            r'ALERT:\s*(SET_LONG_ALERT|SET_SHORT_ALERT|NONE)\s*\$?([\d,.]*)',
            block, re.IGNORECASE
        )
        if alert_match:
            decision.alert_type = alert_match.group(1).upper()
            price_str = alert_match.group(2).replace(",", "")
            if price_str:
                try:
                    decision.alert_price = float(price_str)
                except ValueError:
                    pass

        # Parse SELF_NOTES
        notes_match = re.search(
            r'SELF_NOTES:\s*(.*?)(?=\nSELF_ERRORS:|DECISION:|$)',
            block, re.DOTALL | re.IGNORECASE
        )
        if notes_match:
            decision.self_notes = notes_match.group(1).strip()

        # Parse SELF_ERRORS
        errors_match = re.search(
            r'SELF_ERRORS:\s*(.*?)(?=\nDECISION:|$)',
            block, re.DOTALL | re.IGNORECASE
        )
        if errors_match:
            decision.self_errors = errors_match.group(1).strip()

        decisions.append(decision)

    if not decisions:
        # Fallback: try old-style PAIR: format
        decisions = _parse_legacy_format(text, market_assessment)

    return decisions


def _parse_legacy_format(text: str, market_assessment: str) -> list[AIDecision]:
    """Fallback parser for less structured responses."""
    decisions = []
    pattern = re.compile(
        r'PAIR:\s*(\w+USDT)\s*\n'
        r'ACTION:\s*(.*?)\n'
        r'CONFIDENCE:\s*(.*?)\n'
        r'REASONING:\s*(.*?)(?=\nRISK:|PAIR:|$)',
        re.DOTALL | re.IGNORECASE
    )
    for match in pattern.finditer(text):
        symbol = match.group(1).strip().upper()
        action_raw = match.group(2).strip().upper()
        confidence = match.group(3).strip().upper()
        reasoning = match.group(4).strip()

        # Normalize action
        if "LONG" in action_raw:
            action = "LONG"
        elif "SHORT" in action_raw:
            action = "SHORT"
        elif "SKIP" in action_raw or "NEUTRAL" in action_raw:
            action = "SKIP"
        else:
            action = "SKIP"

        # Normalize confidence
        if "HIGH" in confidence:
            conf = "HIGH"
        elif "MED" in confidence:
            conf = "MEDIUM"
        else:
            conf = "LOW"

        decisions.append(AIDecision(
            symbol=symbol, action=action, confidence=conf,
            reasoning=reasoning, market_assessment=market_assessment,
        ))

    return decisions


def parse_chat_response(text: str) -> str:
    """For Telegram chat -- just return the cleaned text."""
    # Remove any accidental format markers
    cleaned = text.strip()
    # Truncate for Telegram (4096 char limit)
    if len(cleaned) > 3900:
        cleaned = cleaned[:3900] + "..."
    return cleaned
