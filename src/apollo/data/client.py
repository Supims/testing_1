"""
Binance REST Client
===================
Low-level HTTP client for Binance Futures/Spot REST API.
Handles session pooling, rate limiting, retries, and maintenance windows.

This module has NO business logic -- it only fetches and returns raw data.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from apollo.errors import DataError, DataUnavailableError

logger = logging.getLogger("data.client")

# Binance API base URLs
FUTURES_BASE = "https://fapi.binance.com"
SPOT_BASE = "https://api.binance.com"

# Rate limit: 2400 weight / minute for futures, we stay conservative
MAX_WEIGHT_PER_MIN = 1800
KLINE_LIMIT = 1500  # max klines per request


def _ts_ms(dt_str: str) -> int:
    """Convert datetime string to millisecond timestamp."""
    dt = pd.Timestamp(dt_str, tz="UTC")
    return int(dt.timestamp() * 1000)


def _build_session() -> requests.Session:
    """Build a session with connection pooling and automatic retries."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=5,
        pool_maxsize=10,
    )
    session.mount("https://", adapter)
    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "quant-engine/2.0",
    })
    return session


class BinanceClient:
    """
    Low-level Binance REST client with:
      - Connection pooling (single session reused across calls)
      - Automatic retry with exponential backoff on 429/5xx
      - Rate limit tracking (used weight from response headers)
      - Maintenance window detection (returns None instead of crashing)
      - Pagination for large kline requests
    """

    def __init__(self):
        self._session = _build_session()
        self._weight_used = 0
        self._weight_reset_time = 0.0

    def _get(self, base: str, path: str, params: dict,
             timeout: int = 15) -> Optional[list]:
        """
        Execute a GET request. Returns parsed JSON list or None on failure.
        Handles rate limiting and maintenance window detection.
        """
        url = f"{base}{path}"

        # Rate limit check
        now = time.time()
        if now > self._weight_reset_time:
            self._weight_used = 0
            self._weight_reset_time = now + 60

        if self._weight_used > MAX_WEIGHT_PER_MIN:
            wait = self._weight_reset_time - now
            logger.warning(f"Rate limit approached, waiting {wait:.1f}s")
            time.sleep(max(wait, 1.0))
            self._weight_used = 0

        try:
            resp = self._session.get(url, params=params, timeout=timeout)

            # Track weight from headers
            weight = int(resp.headers.get("X-MBX-USED-WEIGHT-1M", 0))
            if weight:
                self._weight_used = weight

            if resp.status_code == 200:
                return resp.json()

            # Maintenance window detection
            if resp.status_code in (451, 418):
                logger.warning("Binance maintenance detected (HTTP %d)", resp.status_code)
                return None

            # Rate limited
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                logger.warning("Rate limited, waiting %ds", retry_after)
                time.sleep(retry_after)
                return self._get(base, path, params, timeout)

            # "Invalid symbol" on spot is expected for futures-only tokens
            body = resp.text[:100]
            if resp.status_code == 400 and "Invalid symbol" in body:
                logger.debug("Symbol %s not on %s", params.get("symbol", "?"), path)
            else:
                logger.warning("Binance API %d for %s: %s", resp.status_code, path, body)
            return None

        except requests.exceptions.ConnectionError:
            logger.error("Connection error -- Binance may be in maintenance")
            return None
        except requests.exceptions.Timeout:
            logger.error("Request timed out after %ds", timeout)
            return None
        except Exception as e:
            logger.error("Unexpected error fetching %s: %s", path, e)
            return None

    # =====================================================================
    # Klines (OHLCV)
    # =====================================================================

    def get_futures_klines(
        self, symbol: str, interval: str,
        start: str, end: str,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch futures klines with automatic pagination.
        Returns DataFrame with columns: open, high, low, close, volume, open_time.
        Returns None if exchange is unavailable.
        """
        all_rows = []
        start_ms = _ts_ms(start)
        end_ms = _ts_ms(end)

        while start_ms < end_ms:
            data = self._get(FUTURES_BASE, "/fapi/v1/klines", {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": KLINE_LIMIT,
            })
            if data is None:
                if all_rows:
                    logger.warning("Partial kline data (got %d bars before failure)", len(all_rows))
                    break
                return None
            if not data:
                break

            all_rows.extend(data)
            last_close_time = data[-1][6]
            start_ms = last_close_time + 1

            if len(data) < KLINE_LIMIT:
                break

        if not all_rows:
            return None

        return self._parse_klines(all_rows)

    def get_spot_klines(
        self, symbol: str, interval: str,
        start: str, end: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch spot klines. Same pagination as futures."""
        all_rows = []
        start_ms = _ts_ms(start)
        end_ms = _ts_ms(end)

        while start_ms < end_ms:
            data = self._get(SPOT_BASE, "/api/v3/klines", {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": KLINE_LIMIT,
            })
            if data is None:
                if all_rows:
                    break
                return None
            if not data:
                break

            all_rows.extend(data)
            last_close_time = data[-1][6]
            start_ms = last_close_time + 1

            if len(data) < KLINE_LIMIT:
                break

        if not all_rows:
            return None

        return self._parse_klines(all_rows)

    @staticmethod
    def _parse_klines(raw: list) -> pd.DataFrame:
        """Parse Binance kline array into a clean DataFrame."""
        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_volume", "taker_buy_quote_volume", "ignore",
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume",
                     "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype(int)
        df = df.set_index("open_time").sort_index()
        df = df.drop(columns=["close_time", "ignore"], errors="ignore")
        # Remove duplicate timestamps
        df = df[~df.index.duplicated(keep="last")]
        return df

    # =====================================================================
    # Funding rate
    # =====================================================================

    def get_funding_rate(
        self, symbol: str, start: str, end: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch historical funding rate."""
        all_rows = []
        start_ms = _ts_ms(start)
        end_ms = _ts_ms(end)

        while start_ms < end_ms:
            data = self._get(FUTURES_BASE, "/fapi/v1/fundingRate", {
                "symbol": symbol,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000,
            })
            if data is None or not data:
                break
            all_rows.extend(data)
            start_ms = data[-1]["fundingTime"] + 1
            if len(data) < 1000:
                break

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows)
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        df = df.rename(columns={"fundingTime": "timestamp", "fundingRate": "funding_rate"})
        df = df.set_index("timestamp").sort_index()
        df = df[["funding_rate"]]
        return df

    # =====================================================================
    # Premium index (basis)
    # =====================================================================

    def get_premium_index_klines(
        self, symbol: str, interval: str,
        start: str, end: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch premium index klines (futures-spot basis)."""
        all_rows = []
        start_ms = _ts_ms(start)
        end_ms = _ts_ms(end)

        while start_ms < end_ms:
            data = self._get(FUTURES_BASE, "/fapi/v1/premiumIndexKlines", {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": KLINE_LIMIT,
            })
            if data is None or not data:
                break
            all_rows.extend(data)
            start_ms = data[-1][6] + 1
            if len(data) < KLINE_LIMIT:
                break

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows, columns=[
            "open_time", "open", "high", "low", "close",
            "_vol", "close_time", "_qv", "_t", "_tbv", "_tbqv", "_ig",
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["premium_index"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.set_index("open_time").sort_index()
        return df[["premium_index"]]

    # =====================================================================
    # Open interest
    # =====================================================================

    def get_open_interest_hist(
        self, symbol: str, interval: str,
        start: str, end: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch open interest history."""
        all_rows = []
        start_ms = _ts_ms(start)
        end_ms = _ts_ms(end)

        # OI hist endpoint only provides the latest 30 days
        one_month_ms = 30 * 24 * 3600 * 1000
        earliest = int(time.time() * 1000) - one_month_ms
        if start_ms < earliest:
            start_ms = earliest

        while start_ms < end_ms:
            data = self._get(FUTURES_BASE, "/futures/data/openInterestHist", {
                "symbol": symbol,
                "period": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 500,
            })
            if data is None or not data:
                break
            all_rows.extend(data)
            start_ms = data[-1]["timestamp"] + 1
            if len(data) < 500:
                break

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["open_interest"] = pd.to_numeric(
            df.get("sumOpenInterest", df.get("openInterest", 0)), errors="coerce"
        )
        df = df.set_index("timestamp").sort_index()
        return df[["open_interest"]]

    # =====================================================================
    # Ticker / current price
    # =====================================================================

    def get_ticker_price(self, symbol: str) -> Optional[float]:
        """Get current futures price for a symbol."""
        data = self._get(FUTURES_BASE, "/fapi/v1/ticker/price", {"symbol": symbol})
        if data and isinstance(data, dict):
            return float(data.get("price", 0))
        if data and isinstance(data, list) and data:
            return float(data[0].get("price", 0))
        return None

    def get_all_ticker_prices(self) -> dict[str, float]:
        """Get current prices for ALL futures symbols."""
        data = self._get(FUTURES_BASE, "/fapi/v1/ticker/price", {})
        if not data:
            return {}
        return {item["symbol"]: float(item["price"]) for item in data if "price" in item}
