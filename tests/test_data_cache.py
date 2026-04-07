"""
Tests for data cache -- Parquet I/O, append, dedup, stale detection.
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path


class TestParquetCache:
    def _make_cache(self, tmp_path):
        from apollo.data.cache import ParquetCache
        return ParquetCache(tmp_path / "cache")

    def _sample_df(self, n=100, start="2025-01-01"):
        idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
        return pd.DataFrame({
            "close": np.random.uniform(40000, 60000, n),
            "volume": np.random.uniform(100, 1000, n),
        }, index=idx)

    def test_write_and_read(self, tmp_path):
        cache = self._make_cache(tmp_path)
        df = self._sample_df()
        cache.write(df, "BTCUSDT", "1h")
        result = cache.read("BTCUSDT", "1h")
        assert result is not None
        assert len(result) == 100

    def test_has(self, tmp_path):
        cache = self._make_cache(tmp_path)
        assert cache.has("BTCUSDT", "1h") is False
        cache.write(self._sample_df(), "BTCUSDT", "1h")
        assert cache.has("BTCUSDT", "1h") is True

    def test_read_missing_returns_none(self, tmp_path):
        cache = self._make_cache(tmp_path)
        assert cache.read("BTCUSDT", "1h") is None

    def test_append_dedup(self, tmp_path):
        cache = self._make_cache(tmp_path)
        df1 = self._sample_df(50, "2025-01-01")
        df2 = self._sample_df(50, "2025-01-02")  # overlapping + new

        cache.write(df1, "BTCUSDT", "1h")
        merged = cache.append(df2, "BTCUSDT", "1h")

        # Should be less than 100 if there's overlap, or 100 if no overlap
        assert len(merged) <= 100
        # No duplicate indices
        assert not merged.index.duplicated().any()

    def test_last_timestamp(self, tmp_path):
        cache = self._make_cache(tmp_path)
        df = self._sample_df(50)
        cache.write(df, "BTCUSDT", "1h")
        ts = cache.last_timestamp("BTCUSDT", "1h")
        assert ts is not None
        assert ts == df.index.max()

    def test_last_timestamp_empty(self, tmp_path):
        cache = self._make_cache(tmp_path)
        assert cache.last_timestamp("BTCUSDT", "1h") is None

    def test_clear_all(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.write(self._sample_df(), "BTCUSDT", "1h")
        cache.write(self._sample_df(), "ETHUSDT", "1h")
        deleted = cache.clear()
        assert deleted == 2

    def test_clear_symbol_filter(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.write(self._sample_df(), "BTCUSDT", "1h")
        cache.write(self._sample_df(), "ETHUSDT", "1h")
        deleted = cache.clear(symbol="BTCUSDT")
        assert deleted == 1
        assert cache.has("ETHUSDT", "1h")

    def test_stats(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.write(self._sample_df(), "BTCUSDT", "1h")
        s = cache.stats()
        assert s["files"] == 1
        assert s["total_size_mb"] > 0

    def test_empty_df_not_written(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.write(pd.DataFrame(), "BTCUSDT", "1h")
        assert cache.has("BTCUSDT", "1h") is False
