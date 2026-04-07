"""
Macro Events Calendar
=======================
Tracks upcoming macro events that impact crypto markets.

Sources (FREE, no API keys):
  - Built-in calendar: known recurring dates (FOMC, CPI, NFP, etc.)
  - CoinGecko events: protocol launches, unlocks
  - Custom user events

Events are injected into the AI prompt so it can adjust risk
around high-impact periods (e.g., reduce position sizes before CPI).
"""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger("core.events")


# =============================================================================
# Built-in macro calendar (hardcoded known recurring events for 2026)
# These are approximate — real dates shift slightly each month/quarter.
# The agent can be extended to scrape exact dates from public calendars.
# =============================================================================

# Format: (month, day, name, impact, category)
# Impact: HIGH / MEDIUM / LOW
# Category: MACRO / CRYPTO / FED

RECURRING_EVENTS_2026 = [
    # FOMC meetings (2-day, decision on second day)
    (1, 29, "FOMC Rate Decision", "HIGH", "FED"),
    (3, 19, "FOMC Rate Decision", "HIGH", "FED"),
    (5, 7, "FOMC Rate Decision", "HIGH", "FED"),
    (6, 18, "FOMC Rate Decision", "HIGH", "FED"),
    (7, 30, "FOMC Rate Decision", "HIGH", "FED"),
    (9, 17, "FOMC Rate Decision", "HIGH", "FED"),
    (11, 5, "FOMC Rate Decision", "HIGH", "FED"),
    (12, 17, "FOMC Rate Decision", "HIGH", "FED"),
    # CPI Release (~13th of each month)
    (1, 14, "CPI Data Release", "HIGH", "MACRO"),
    (2, 12, "CPI Data Release", "HIGH", "MACRO"),
    (3, 12, "CPI Data Release", "HIGH", "MACRO"),
    (4, 10, "CPI Data Release", "HIGH", "MACRO"),
    (5, 13, "CPI Data Release", "HIGH", "MACRO"),
    (6, 11, "CPI Data Release", "HIGH", "MACRO"),
    (7, 15, "CPI Data Release", "HIGH", "MACRO"),
    (8, 13, "CPI Data Release", "HIGH", "MACRO"),
    (9, 10, "CPI Data Release", "HIGH", "MACRO"),
    (10, 14, "CPI Data Release", "HIGH", "MACRO"),
    (11, 12, "CPI Data Release", "HIGH", "MACRO"),
    (12, 10, "CPI Data Release", "HIGH", "MACRO"),
    # Non-Farm Payrolls (~first Friday of each month)
    (1, 9, "NFP Jobs Report", "HIGH", "MACRO"),
    (2, 6, "NFP Jobs Report", "HIGH", "MACRO"),
    (3, 6, "NFP Jobs Report", "HIGH", "MACRO"),
    (4, 3, "NFP Jobs Report", "HIGH", "MACRO"),
    (5, 1, "NFP Jobs Report", "HIGH", "MACRO"),
    (6, 5, "NFP Jobs Report", "HIGH", "MACRO"),
    (7, 3, "NFP Jobs Report", "HIGH", "MACRO"),
    (8, 7, "NFP Jobs Report", "HIGH", "MACRO"),
    (9, 4, "NFP Jobs Report", "HIGH", "MACRO"),
    (10, 2, "NFP Jobs Report", "HIGH", "MACRO"),
    (11, 6, "NFP Jobs Report", "HIGH", "MACRO"),
    (12, 4, "NFP Jobs Report", "HIGH", "MACRO"),
    # GDP releases (quarterly)
    (1, 30, "Q4 GDP Advance", "MEDIUM", "MACRO"),
    (4, 30, "Q1 GDP Advance", "MEDIUM", "MACRO"),
    (7, 30, "Q2 GDP Advance", "MEDIUM", "MACRO"),
    (10, 29, "Q3 GDP Advance", "MEDIUM", "MACRO"),
    # Crypto-specific
    (4, 15, "BTC Halving Anniversary", "MEDIUM", "CRYPTO"),
    (4, 15, "US Tax Deadline (sell pressure)", "MEDIUM", "MACRO"),
]


class MacroEventCalendar:
    """
    Tracks macro and crypto events that impact market volatility.

    Usage:
        cal = MacroEventCalendar()
        upcoming = cal.get_upcoming(days_ahead=3)
        prompt_block = cal.to_prompt_block()
    """

    def __init__(self):
        self._events = self._build_event_list()
        self._last_crypto_fetch: Optional[datetime] = None
        self._crypto_events: list[dict] = []

    def _build_event_list(self) -> list[dict]:
        """Build event list from hardcoded calendar."""
        events = []
        year = datetime.now(timezone.utc).year
        for month, day, name, impact, category in RECURRING_EVENTS_2026:
            try:
                dt = datetime(year, month, day, tzinfo=timezone.utc)
                events.append({
                    "date": dt,
                    "name": name,
                    "impact": impact,
                    "category": category,
                    "source": "calendar",
                })
            except ValueError:
                pass  # Invalid date
        return sorted(events, key=lambda e: e["date"])

    def get_upcoming(self, days_ahead: int = 3) -> list[dict]:
        """Get events in the next N days."""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=days_ahead)

        # Also include events in the last 24h (just happened)
        lookback = now - timedelta(hours=24)

        upcoming = []
        for ev in self._events:
            if lookback <= ev["date"] <= cutoff:
                hours_away = (ev["date"] - now).total_seconds() / 3600
                upcoming.append({
                    **ev,
                    "date_str": ev["date"].strftime("%Y-%m-%d"),
                    "hours_away": round(hours_away, 1),
                    "is_past": hours_away < 0,
                    "is_today": abs(hours_away) < 24,
                    "is_imminent": 0 < hours_away < 12,
                })

        # Add crypto events from CoinGecko
        crypto = self._fetch_crypto_events()
        for cev in crypto:
            if lookback <= cev["date"] <= cutoff:
                hours_away = (cev["date"] - now).total_seconds() / 3600
                upcoming.append({
                    **cev,
                    "hours_away": round(hours_away, 1),
                    "is_past": hours_away < 0,
                    "is_today": abs(hours_away) < 24,
                    "is_imminent": 0 < hours_away < 12,
                })

        return sorted(upcoming, key=lambda e: e.get("hours_away", 999))

    def _fetch_crypto_events(self) -> list[dict]:
        """Fetch upcoming crypto events from CoinGecko (cached 1h)."""
        now = datetime.now(timezone.utc)
        if (self._last_crypto_fetch and
                (now - self._last_crypto_fetch) < timedelta(hours=1)):
            return self._crypto_events

        try:
            url = "https://api.coingecko.com/api/v3/events"
            req = urllib.request.Request(url, headers={
                "User-Agent": "apollo-quant/2.0",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())

            events = []
            for ev in data.get("data", [])[:10]:
                try:
                    start = ev.get("start_date", "")
                    if start:
                        dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                        events.append({
                            "date": dt,
                            "date_str": dt.strftime("%Y-%m-%d"),
                            "name": ev.get("title", "Unknown")[:80],
                            "impact": "MEDIUM",
                            "category": "CRYPTO",
                            "source": "coingecko",
                        })
                except Exception:
                    pass

            self._crypto_events = events
            self._last_crypto_fetch = now
            return events

        except Exception as e:
            logger.debug("CoinGecko events fetch failed: %s", e)
            return self._crypto_events

    def get_risk_multiplier(self) -> float:
        """
        Get a risk multiplier based on upcoming events.

        Returns:
          1.0 = normal
          0.5 = reduce size by 50% (HIGH impact imminent)
          0.75 = reduce size by 25% (MEDIUM impact imminent)
        """
        upcoming = self.get_upcoming(days_ahead=1)

        for ev in upcoming:
            if ev.get("is_imminent") and ev.get("impact") == "HIGH":
                logger.info(
                    "HIGH impact event imminent: %s in %.1fh -> risk x0.5",
                    ev["name"], ev["hours_away"],
                )
                return 0.5
            if ev.get("is_today") and ev.get("impact") == "HIGH":
                return 0.75

        return 1.0

    def to_prompt_block(self, days_ahead: int = 3) -> str:
        """Generate prompt block for AI context."""
        upcoming = self.get_upcoming(days_ahead)
        if not upcoming:
            return ""

        lines = ["=== MACRO EVENTS ==="]
        risk_mult = self.get_risk_multiplier()

        if risk_mult < 1.0:
            lines.append(
                f"⚠ RISK ADJUSTMENT: x{risk_mult:.2f} "
                f"(HIGH impact event imminent)"
            )

        for ev in upcoming[:8]:
            status = ""
            if ev.get("is_past"):
                status = "[JUST HAPPENED]"
            elif ev.get("is_imminent"):
                status = "[IMMINENT ⚠]"
            elif ev.get("is_today"):
                status = "[TODAY]"

            lines.append(
                f"  {ev.get('date_str', '?')} | {ev.get('name', '?')} | "
                f"{ev.get('impact', '?')} | {ev.get('category', '?')} "
                f"{status}"
            )

        return "\n".join(lines)
