"""
Market Data Provider
====================
Unified data access layer. Orchestrates:
  client (fetch) -> cache (store) -> impute (gaps) -> merge (combine) -> output

This is the ONLY module other code should use to get market data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from apollo.data.client import BinanceClient
from apollo.data.cache import ParquetCache
from apollo.errors import DataError

logger = logging.getLogger("data.provider")

# Interval -> pandas freq mapping
TF_MAPPING = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1h": "1h", "2h": "2h", "4h": "4h",
    "6h": "6h", "8h": "8h", "12h": "12h", "1d": "1D",
}


@dataclass
class DataStatus:
    """Metadata about the data quality returned."""
    is_fresh: bool = True
    is_stale: bool = False
    is_partial: bool = False
    rows: int = 0
    gaps_imputed: int = 0
    degraded_reason: str = ""


def ensure_temporal_integrity(df: pd.DataFrame, interval: str) -> tuple[pd.DataFrame, int]:
    """
    Guarantee ZERO holes in the time series.
    If Binance missed candles (maintenance), impute rows:
      - Price: forward-fill (flat candle)
      - Volume: zero (no activity during gap)

    Returns (imputed_df, n_gaps_filled).
    """
    if df.empty:
        return df, 0

    freq = TF_MAPPING.get(interval, interval)
    start = df.index[0]
    end = df.index[-1]

    perfect_index = pd.date_range(start=start, end=end, freq=freq)
    missing = perfect_index.difference(df.index)
    n_missing = len(missing)

    if n_missing == 0:
        return df, 0

    logger.warning("Temporal gaps: %d missing candles detected, imputing", n_missing)

    df = df.reindex(perfect_index)

    # Price columns: forward-fill (flat candle)
    price_cols = [c for c in ["open", "high", "low", "close", "spot_close"] if c in df.columns]
    for c in price_cols:
        df[c] = df[c].ffill()

    # For imputed bars, set open=high=low=close (flat candle)
    if "close" in df.columns:
        for c in ["open", "high", "low"]:
            if c in df.columns:
                df[c] = df[c].fillna(df["close"])

    # Volume columns: zero during gap
    vol_cols = [c for c in ["volume", "quote_volume", "trades",
                             "taker_buy_volume", "taker_buy_quote_volume",
                             "taker_sell_volume", "spot_volume"]
                if c in df.columns]
    for c in vol_cols:
        df[c] = df[c].fillna(0)

    # Other columns: forward-fill
    df = df.ffill().fillna(0)

    return df, n_missing


class MarketDataProvider:
    """
    Unified market data access for all downstream consumers.

    Features:
      - Incremental fetching: only downloads new bars since last cache
      - Stale fallback: returns cached data if exchange is down
      - Temporal integrity: imputes missing candles from maintenance windows
      - Multi-source merge: futures + spot + premium + funding + OI
      - Anti-lookahead: funding merged with merge_asof(direction='backward')
      - Float32 optimization: reduces memory footprint by ~50%
    """

    def __init__(self, client: BinanceClient = None, cache: ParquetCache = None):
        self._client = client or BinanceClient()
        self._cache = cache  # None = no caching

    @staticmethod
    def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Downcast float64 -> float32 for memory efficiency."""
        float_cols = df.select_dtypes(include=["float64"]).columns
        df[float_cols] = df[float_cols].astype("float32")
        return df

    def get_klines(
        self, symbol: str, interval: str,
        start: str, end: str,
        source: str = "futures",
    ) -> tuple[pd.DataFrame, DataStatus]:
        """
        Get OHLCV klines for a symbol.

        Args:
            symbol: e.g. "BTCUSDT"
            interval: e.g. "1h", "4h", "1d"
            start: start date string
            end: end date string
            source: "futures" or "spot"

        Returns:
            (DataFrame, DataStatus)
        """
        cache_type = f"{source}_klines"
        status = DataStatus()

        # Try incremental fetch
        fetch_start = start
        if self._cache:
            last_ts = self._cache.last_timestamp(symbol, interval, cache_type)
            if last_ts is not None:
                fetch_start = (last_ts + timedelta(milliseconds=1)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                logger.debug(
                    "Incremental fetch from %s (cache until %s)", fetch_start, last_ts
                )

        # Fetch from exchange
        if source == "futures":
            new_data = self._client.get_futures_klines(symbol, interval, fetch_start, end)
        else:
            new_data = self._client.get_spot_klines(symbol, interval, fetch_start, end)

        # Merge with cache
        if new_data is not None and not new_data.empty:
            if self._cache:
                df = self._cache.append(new_data, symbol, interval, cache_type)
            else:
                df = new_data
            status.is_fresh = True
            status.rows = len(df)
        else:
            # Exchange failed -- try stale cache
            if self._cache:
                df = self._cache.read(symbol, interval, cache_type)
                if df is not None and not df.empty:
                    status.is_fresh = False
                    status.is_stale = True
                    status.degraded_reason = "Exchange unavailable, using cached data"
                    status.rows = len(df)
                    logger.warning(
                        "Using stale cache for %s/%s (%d rows)", symbol, interval, len(df)
                    )
                else:
                    raise DataError(
                        f"No data for {symbol}/{interval}: exchange down and no cache"
                    )
            else:
                raise DataError(f"No data for {symbol}/{interval}: exchange unavailable")

        # Filter to requested window
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        df = df[(df.index >= start_ts) & (df.index <= end_ts)]

        # Temporal integrity: fill gaps from maintenance windows
        df, n_gaps = ensure_temporal_integrity(df, interval)
        status.gaps_imputed = n_gaps

        # Float32 optimization
        df = self._optimize_dtypes(df)

        status.rows = len(df)
        return df, status

    def get_enriched_dataset(
        self, symbol: str, interval: str,
        start: str, end: str,
    ) -> tuple[pd.DataFrame, DataStatus]:
        """
        Get a fully enriched dataset: futures OHLCV + spot + premium + funding + OI.
        All sources merged with anti-lookahead protection.

        Returns:
            (DataFrame with all columns, DataStatus)
        """
        status = DataStatus()

        # 1) Futures klines (required)
        df, kline_status = self.get_klines(symbol, interval, start, end, source="futures")
        if df.empty:
            raise DataError(f"No futures klines for {symbol}")
        status.is_stale = kline_status.is_stale
        status.gaps_imputed = kline_status.gaps_imputed

        # 2) Spot klines (optional -- for basis + volume ratio)
        # Skip spot for tokens that likely don't exist on spot market
        try:
            spot_df, _ = self.get_klines(symbol, interval, start, end, source="spot")
            if spot_df is not None and not spot_df.empty:
                df["spot_close"] = spot_df["close"].reindex(df.index, method="ffill")
                df["spot_volume"] = spot_df["volume"].reindex(df.index, method="ffill")
                # Compute true basis
                df["basis"] = (
                    (df["close"] - df["spot_close"])
                    / df["spot_close"].replace(0, np.nan)
                )
        except Exception as e:
            logger.debug("Spot data not available for %s: %s", symbol, type(e).__name__)

        # 3) Premium index (optional)
        try:
            premium = self._client.get_premium_index_klines(symbol, interval, start, end)
            if premium is not None and not premium.empty:
                premium = self._optimize_dtypes(premium)
                df = pd.merge_asof(
                    df, premium,
                    left_index=True, right_index=True,
                    direction="backward",
                )
                df["premium_index"] = df["premium_index"].ffill().fillna(0)
        except Exception as e:
            logger.debug("Premium index not available for %s: %s", symbol, e)

        # 4) Funding rate (optional -- backward merge to prevent lookahead)
        try:
            funding = self._client.get_funding_rate(symbol, start, end)
            if funding is not None and not funding.empty:
                funding = self._optimize_dtypes(funding)
                df = pd.merge_asof(
                    df, funding,
                    left_index=True, right_index=True,
                    direction="backward",
                )
                df["funding_rate"] = df["funding_rate"].fillna(0)
        except Exception as e:
            logger.debug("Funding rate not available for %s: %s", symbol, e)

        # 5) Open interest (optional)
        try:
            oi = self._client.get_open_interest_hist(symbol, interval, start, end)
            if oi is not None and not oi.empty:
                oi = self._optimize_dtypes(oi)
                df = pd.merge_asof(
                    df, oi,
                    left_index=True, right_index=True,
                    direction="backward",
                )
                df["open_interest"] = df["open_interest"].ffill().fillna(0)
        except Exception as e:
            logger.debug("OI data not available for %s: %s", symbol, e)

        # Compute taker sell volume
        if "taker_buy_volume" in df.columns and "volume" in df.columns:
            df["taker_sell_volume"] = df["volume"] - df["taker_buy_volume"]

        # Final dtype optimization
        df = self._optimize_dtypes(df)

        status.rows = len(df)
        if status.is_stale:
            status.degraded_reason = "Some data from cache (exchange was unavailable)"

        return df, status

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get latest futures price for a symbol."""
        return self._client.get_ticker_price(symbol)

    def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        """Get latest prices for multiple symbols in one API call."""
        all_prices = self._client.get_all_ticker_prices()
        return {s: all_prices[s] for s in symbols if s in all_prices}
