"""
Sentiment & Market Context Collector
======================================
Gathers free-tier market sentiment data for AI prompt injection.

Sources (no API keys required):
  - Fear & Greed Index (Alternative.me)
  - BTC Dominance (CoinGecko)
  - Funding rate heatmap (Binance)
  - Recent large liquidations (Binance)
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("ai.sentiment")

_TIMEOUT = 10


def _fetch_json(url: str) -> Optional[dict]:
    """Fetch JSON from a URL with timeout. Returns None on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "quant-scanner/1.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return None


class SentimentCollector:
    """Gathers free-tier sentiment data for AI context."""

    _btc_dom_cache: float = 0.0
    _btc_dom_cache_ts: float = 0.0

    def get_fear_greed_index(self) -> dict:
        """Alternative.me Fear & Greed Index (free, no API key)."""
        data = _fetch_json("https://api.alternative.me/fng/?limit=1")
        if data and data.get("data"):
            entry = data["data"][0]
            return {
                "value": int(entry.get("value", 50)),
                "label": entry.get("value_classification", "Neutral"),
                "timestamp": entry.get("timestamp", ""),
            }
        return {"value": 50, "label": "Neutral", "timestamp": ""}

    def get_btc_dominance(self) -> float:
        """BTC dominance from CoinGecko (free tier). Caches last valid value."""
        import time as _time
        # Return cache if less than 5 minutes old
        if _time.time() - self._btc_dom_cache_ts < 300 and self._btc_dom_cache > 0:
            return self._btc_dom_cache
        data = _fetch_json("https://api.coingecko.com/api/v3/global")
        if data and "data" in data:
            val = float(data["data"].get("market_cap_percentage", {}).get("btc", 0))
            if val > 0:
                SentimentCollector._btc_dom_cache = val
                SentimentCollector._btc_dom_cache_ts = _time.time()
            return val
        # Return cached value if API failed, or 0 if never fetched
        if self._btc_dom_cache > 0:
            logger.warning("CoinGecko unavailable, using cached BTC.D=%.1f%%", self._btc_dom_cache)
            return self._btc_dom_cache
        return 0.0

    def get_funding_heatmap(self, symbols: list[str] = None) -> dict:
        """Aggregate funding rates from Binance Futures."""
        data = _fetch_json("https://fapi.binance.com/fapi/v1/premiumIndex")
        if not data:
            return {"avg_funding": 0.0, "extreme_pairs": []}

        if symbols:
            symbol_set = set(symbols)
            data = [d for d in data if d.get("symbol") in symbol_set]

        rates = []
        extreme = []
        for item in data:
            rate = float(item.get("lastFundingRate", 0))
            rates.append(rate)
            if abs(rate) > 0.001:  # > 0.1% = extreme
                extreme.append({
                    "symbol": item.get("symbol", "?"),
                    "rate": rate,
                })

        avg = sum(rates) / max(len(rates), 1)
        extreme.sort(key=lambda x: abs(x["rate"]), reverse=True)
        return {
            "avg_funding": avg,
            "n_pairs": len(rates),
            "extreme_pairs": extreme[:5],
        }

    def get_liquidation_summary(self) -> list[dict]:
        """
        Recent liquidation events summary.
        Uses Binance forced liquidation orders endpoint.
        Returns top events by size.
        """
        # Binance forceOrders endpoint requires API key in some regions
        # Fallback: aggregate from OI changes as proxy
        data = _fetch_json("https://fapi.binance.com/fapi/v1/ticker/24hr")
        if not data:
            return []

        # Use large 24h volume delta as proxy for liquidation activity
        events = []
        for item in data:
            symbol = item.get("symbol", "")
            if not symbol.endswith("USDT"):
                continue
            price_change = float(item.get("priceChangePercent", 0))
            volume = float(item.get("quoteVolume", 0))
            # Large price moves with high volume suggest liquidation cascades
            if abs(price_change) > 5 and volume > 100_000_000:
                events.append({
                    "symbol": symbol,
                    "side": "LONG" if price_change < 0 else "SHORT",
                    "price_change_pct": price_change,
                    "volume_24h": volume,
                })

        events.sort(key=lambda x: abs(x["price_change_pct"]), reverse=True)
        return events[:5]

    def get_market_summary(self, symbols: list[str] = None) -> dict:
        """
        Combines all sentiment sources into a compact dict for prompt injection.
        """
        fg = self.get_fear_greed_index()
        btc_dom = self.get_btc_dominance()
        funding = self.get_funding_heatmap(symbols)
        liqs = self.get_liquidation_summary()

        summary = {
            "fear_greed": fg,
            "btc_dominance": btc_dom,
            "total_funding_bias": funding.get("avg_funding", 0),
            "extreme_funding": funding.get("extreme_pairs", []),
            "recent_large_liqs": liqs,
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "Sentiment: F&G=%d (%s) | BTC.D=%.1f%% | Funding=%.6f | Liqs=%d",
            fg.get("value", 0), fg.get("label", "?"),
            btc_dom, funding.get("avg_funding", 0), len(liqs),
        )

        return summary
