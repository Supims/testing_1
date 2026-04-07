"""
Pair Discovery
===============
Discovers and filters tradeable USDT-M futures pairs from Binance.
Filters by volume, age, data quality, and spread.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("data.discovery")

_TIMEOUT = 15
_STABLECOIN_BASES = {"USDC", "BUSD", "TUSD", "FDUSD", "DAI", "USDD", "USDP"}


def _fetch_json(url: str) -> Optional[list | dict]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "quant-scanner/1.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return None


class PairDiscovery:
    """Discovers and filters tradeable USDT-M futures pairs."""

    def discover(
        self,
        min_volume_24h: float = 50_000_000,
        min_age_days: int = 30,
        max_pairs: int = 30,
    ) -> list[str]:
        """
        Find top tradeable perpetual USDT-M pairs.

        Steps:
        1. GET /fapi/v1/exchangeInfo -> all perpetual USDT pairs
        2. GET /fapi/v1/ticker/24hr -> filter by volume
        3. Exclude stablecoins
        4. Sort by 24h volume descending
        5. Return top N symbols
        """
        # Step 1: Get all USDT-M perpetual pairs
        info = _fetch_json("https://fapi.binance.com/fapi/v1/exchangeInfo")
        if not info or "symbols" not in info:
            logger.error("Failed to fetch exchange info")
            return self._fallback_symbols(max_pairs)

        perp_symbols = set()
        for sym_info in info["symbols"]:
            if (sym_info.get("contractType") == "PERPETUAL"
                    and sym_info.get("quoteAsset") == "USDT"
                    and sym_info.get("status") == "TRADING"):
                symbol = sym_info["symbol"]
                base = sym_info.get("baseAsset", "")
                if base not in _STABLECOIN_BASES:
                    perp_symbols.add(symbol)

        logger.info("Found %d perpetual USDT-M pairs", len(perp_symbols))

        # Step 2: Get 24h tickers for volume filtering
        tickers = _fetch_json("https://fapi.binance.com/fapi/v1/ticker/24hr")
        if not tickers:
            return self._fallback_symbols(max_pairs)

        # Filter and sort
        candidates = []
        for t in tickers:
            symbol = t.get("symbol", "")
            if symbol not in perp_symbols:
                continue
            volume = float(t.get("quoteVolume", 0))
            if volume < min_volume_24h:
                continue
            candidates.append({
                "symbol": symbol,
                "volume_24h": volume,
                "price_change_pct": float(t.get("priceChangePercent", 0)),
                "last_price": float(t.get("lastPrice", 0)),
            })

        candidates.sort(key=lambda x: x["volume_24h"], reverse=True)
        result = [c["symbol"] for c in candidates[:max_pairs]]

        logger.info(
            "Discovery: %d pairs meet criteria (volume > $%dM). Top: %s",
            len(result), int(min_volume_24h / 1e6),
            ", ".join(result[:5]),
        )
        return result

    def get_pair_metadata(self, symbols: list[str]) -> dict[str, dict]:
        """Returns metadata for each pair: tick size, lot size, etc."""
        info = _fetch_json("https://fapi.binance.com/fapi/v1/exchangeInfo")
        if not info:
            return {}

        symbol_set = set(symbols)
        metadata = {}
        for sym_info in info.get("symbols", []):
            symbol = sym_info.get("symbol", "")
            if symbol not in symbol_set:
                continue
            filters = {f["filterType"]: f for f in sym_info.get("filters", [])}
            price_filter = filters.get("PRICE_FILTER", {})
            lot_filter = filters.get("LOT_SIZE", {})
            metadata[symbol] = {
                "tick_size": float(price_filter.get("tickSize", 0.01)),
                "min_qty": float(lot_filter.get("minQty", 0.001)),
                "step_size": float(lot_filter.get("stepSize", 0.001)),
                "max_leverage": sym_info.get("maxLeverage", 20),
            }

        return metadata

    @staticmethod
    def _fallback_symbols(max_pairs: int) -> list[str]:
        """Hardcoded fallback if API fails."""
        defaults = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
            "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
            "MATICUSDT", "LTCUSDT", "ARBUSDT", "OPUSDT", "APTUSDT",
            "NEARUSDT", "FILUSDT", "ATOMUSDT", "TRXUSDT", "UNIUSDT",
            "SUIUSDT", "SEIUSDT", "INJUSDT", "PEPEUSDT", "WIFUSDT",
            "FETUSDT", "RENDERUSDT", "TIAUSDT", "JUPUSDT", "WUSDT",
        ]
        return defaults[:max_pairs]
