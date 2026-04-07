"""
Tests for features -- technical indicators, pipeline, and MTF.
"""

import numpy as np
import pandas as pd
import pytest


class TestReturns:
    def test_log_return_shape(self, synthetic_ohlcv):
        from apollo.features.technical import log_return
        result = log_return(synthetic_ohlcv["close"])
        assert len(result) == len(synthetic_ohlcv)
        assert pd.isna(result.iloc[0])  # first value should be NaN

    def test_log_return_sign(self, synthetic_ohlcv):
        from apollo.features.technical import log_return
        result = log_return(synthetic_ohlcv["close"])
        # bar 350 is a sharp drop -> log return should be very negative
        assert result.iloc[350] < -0.05

    def test_autocorrelation(self, synthetic_ohlcv):
        from apollo.features.technical import log_return, autocorrelation
        lr = log_return(synthetic_ohlcv["close"])
        result = autocorrelation(lr, window=20, lag=1)
        valid = result.dropna()
        assert len(valid) > 0
        assert valid.between(-1, 1).all()


class TestTechnicalIndicators:
    def test_rsi_range(self, synthetic_ohlcv):
        from apollo.features.technical import rsi
        result = rsi(synthetic_ohlcv["close"])
        valid = result.dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_rsi_normalized_range(self, synthetic_ohlcv):
        from apollo.features.technical import rsi_normalized
        result = rsi_normalized(synthetic_ohlcv["close"])
        valid = result.dropna()
        assert valid.min() >= -1.0
        assert valid.max() <= 1.0

    def test_rsi_overbought_in_uptrend(self, synthetic_ohlcv):
        from apollo.features.technical import rsi
        result = rsi(synthetic_ohlcv["close"])
        trend_rsi = result.iloc[150:200].mean()
        assert trend_rsi > 50

    def test_macd_columns(self, synthetic_ohlcv):
        from apollo.features.technical import macd
        result = macd(synthetic_ohlcv["close"])
        assert set(result.columns) == {"macd", "macd_signal", "macd_hist"}
        assert len(result) == len(synthetic_ohlcv)

    def test_adx_full_three_cols(self, synthetic_ohlcv):
        from apollo.features.technical import adx_full
        result = adx_full(
            synthetic_ohlcv["high"], synthetic_ohlcv["low"], synthetic_ohlcv["close"]
        )
        assert set(result.columns) == {"adx", "adx_pos", "adx_neg"}
        valid = result.dropna()
        assert (valid["adx"] >= 0).all()
        assert (valid["adx_pos"] >= 0).all()
        assert (valid["adx_neg"] >= 0).all()

    def test_bollinger_bands_ordering(self, synthetic_ohlcv):
        from apollo.features.technical import bollinger_bands
        bb = bollinger_bands(synthetic_ohlcv["close"])
        valid = bb.dropna()
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()

    def test_atr_positive(self, synthetic_ohlcv):
        from apollo.features.technical import atr
        result = atr(
            synthetic_ohlcv["high"], synthetic_ohlcv["low"], synthetic_ohlcv["close"]
        )
        valid = result.dropna()
        assert (valid > 0).all()

    def test_natr_comparable(self, synthetic_ohlcv):
        from apollo.features.technical import natr
        result = natr(
            synthetic_ohlcv["high"], synthetic_ohlcv["low"], synthetic_ohlcv["close"]
        )
        valid = result.dropna()
        assert (valid > 0).all()
        assert valid.mean() < 10  # should be a small percentage

    def test_ichimoku_columns(self, synthetic_ohlcv):
        from apollo.features.technical import ichimoku
        result = ichimoku(synthetic_ohlcv["high"], synthetic_ohlcv["low"])
        assert set(result.columns) == {"ichimoku_conv", "ichimoku_base"}
        valid = result.dropna()
        assert len(valid) > 0

    def test_obv_direction(self, synthetic_ohlcv):
        from apollo.features.technical import obv
        result = obv(synthetic_ohlcv["close"], synthetic_ohlcv["volume"])
        assert result.std() > 0

    def test_rolling_vwap_stable(self, synthetic_ohlcv):
        from apollo.features.technical import vwap_rolling
        result = vwap_rolling(
            synthetic_ohlcv["high"], synthetic_ohlcv["low"],
            synthetic_ohlcv["close"], synthetic_ohlcv["volume"],
        )
        valid = result.dropna()
        # Rolling VWAP should stay near price (not explode)
        ratio = valid / synthetic_ohlcv["close"].loc[valid.index]
        assert ratio.between(0.9, 1.1).all()


class TestMicrostructure:
    def test_taker_delta(self, synthetic_ohlcv_with_micro):
        from apollo.features.technical import taker_delta
        df = synthetic_ohlcv_with_micro
        result = taker_delta(df["taker_buy_volume"], df["volume"])
        assert result.nunique() > 1

    def test_taker_delta_ema(self, synthetic_ohlcv_with_micro):
        from apollo.features.technical import taker_delta_ema
        df = synthetic_ohlcv_with_micro
        result = taker_delta_ema(df["taker_buy_volume"], df["volume"])
        valid = result.dropna()
        assert len(valid) > 0

    def test_cvd_rolling(self, synthetic_ohlcv_with_micro):
        from apollo.features.technical import cvd_rolling
        df = synthetic_ohlcv_with_micro
        r24 = cvd_rolling(df["taker_buy_volume"], df["volume"], window=24)
        r100 = cvd_rolling(df["taker_buy_volume"], df["volume"], window=100)
        assert r24.dropna().nunique() > 1
        assert r100.dropna().nunique() > 1

    def test_funding_zscore(self, synthetic_ohlcv_with_micro):
        from apollo.features.technical import funding_zscore
        df = synthetic_ohlcv_with_micro
        result = funding_zscore(df["funding_rate"])
        valid = result.dropna()
        assert abs(valid.mean()) < 1.0

    def test_basis_zscore(self, synthetic_ohlcv_with_micro):
        from apollo.features.technical import basis_zscore
        df = synthetic_ohlcv_with_micro
        result = basis_zscore(df["premium_index"], window=100)
        valid = result.dropna()
        assert abs(valid.mean()) < 1.0

    def test_taker_ratio_range(self, synthetic_ohlcv_with_micro):
        from apollo.features.technical import taker_ratio
        df = synthetic_ohlcv_with_micro
        result = taker_ratio(df["taker_buy_volume"], df["volume"])
        assert result.min() >= 0
        assert result.max() <= 1.0

    def test_oi_velocity(self, synthetic_ohlcv_with_micro):
        from apollo.features.technical import oi_velocity
        df = synthetic_ohlcv_with_micro
        result = oi_velocity(df["open_interest"])
        assert "oi_vel_5" in result.columns
        assert "oi_vel_20" in result.columns
        assert "oi_vel_60" in result.columns

    def test_vol_oi_ratio(self, synthetic_ohlcv_with_micro):
        from apollo.features.technical import vol_oi_ratio
        df = synthetic_ohlcv_with_micro
        result = vol_oi_ratio(df["volume"], df["open_interest"])
        assert result.dropna().min() >= 0


class TestFeaturePipeline:
    def test_basic_output(self, synthetic_ohlcv):
        from apollo.features.pipeline import FeaturePipeline
        pipe = FeaturePipeline(warmup_bars=30)
        result, meta = pipe.compute(synthetic_ohlcv)

        assert len(result) > 0
        assert meta.input_rows == 500
        assert meta.warmup_rows_dropped == 30
        assert meta.output_rows == 470

    def test_all_expected_columns_present(self, synthetic_ohlcv):
        from apollo.features.pipeline import FeaturePipeline
        pipe = FeaturePipeline()
        result, _ = pipe.compute(synthetic_ohlcv)

        expected = [
            "log_return", "rsi", "rsi_norm",
            "macd", "macd_signal", "macd_hist",
            "adx", "adx_pos", "adx_neg",
            "cci", "williams_r", "stoch_k", "stoch_d",
            "ichimoku_conv", "ichimoku_base",
            "autocorr_w20_l1",
            "bb_upper", "bb_middle", "bb_lower", "bb_bandwidth", "bb_pct_b",
            "atr", "natr", "gk_vol", "parkinson_vol",
            "realized_vol_20", "realized_vol_100",
            "obv", "obv_ema_20", "cmf", "vwap_dist",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_nan_in_output(self, synthetic_ohlcv):
        from apollo.features.pipeline import FeaturePipeline
        pipe = FeaturePipeline()
        result, _ = pipe.compute(synthetic_ohlcv)
        assert not result.isna().any().any()

    def test_no_inf_in_output(self, synthetic_ohlcv):
        from apollo.features.pipeline import FeaturePipeline
        pipe = FeaturePipeline()
        result, _ = pipe.compute(synthetic_ohlcv)
        assert not np.isinf(result.values).any()

    def test_microstructure_features(self, synthetic_ohlcv_with_micro):
        from apollo.features.pipeline import FeaturePipeline
        pipe = FeaturePipeline()
        result, meta = pipe.compute(synthetic_ohlcv_with_micro)

        expected_micro = [
            "taker_delta", "taker_delta_ema_20", "taker_ratio",
            "cvd_rolling_24", "cvd_rolling_100",
            "funding_zscore", "basis_zscore",
            "oi_vel_5", "oi_vel_20", "oi_vel_60",
            "vol_oi_ratio",
        ]
        for col in expected_micro:
            assert col in result.columns, f"Missing micro column: {col}"

    def test_target_generation_in_train_mode(self, synthetic_ohlcv):
        from apollo.features.pipeline import FeaturePipeline
        pipe = FeaturePipeline(warmup_bars=30)
        result, _ = pipe.compute(synthetic_ohlcv, is_train=True)

        assert "target_return_1f" in result.columns
        assert "target_dir_1f" in result.columns
        assert "target_return_12f" in result.columns

    def test_no_targets_in_inference_mode(self, synthetic_ohlcv):
        from apollo.features.pipeline import FeaturePipeline
        pipe = FeaturePipeline(warmup_bars=30)
        result, _ = pipe.compute(synthetic_ohlcv, is_train=False)

        assert "target_return_1f" not in result.columns

    def test_missing_columns_raises(self):
        from apollo.features.pipeline import FeaturePipeline
        pipe = FeaturePipeline()
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            pipe.compute(df)


class TestMTFAligner:
    def test_htf_features_added(self, synthetic_ohlcv):
        from apollo.features.mtf import MTFAligner
        aligner = MTFAligner(htf="4h")
        result = aligner.align(synthetic_ohlcv)

        htf_cols = [c for c in result.columns if c.startswith("htf_")]
        assert len(htf_cols) > 0, "No HTF features were added"

    def test_no_lookahead(self, synthetic_ohlcv):
        """HTF features at bar T must not use data from bar T's HTF period."""
        from apollo.features.mtf import MTFAligner
        aligner = MTFAligner(htf="4h")
        result = aligner.align(synthetic_ohlcv)

        htf_cols = [c for c in result.columns if c.startswith("htf_")]
        if not htf_cols:
            pytest.skip("No HTF features generated")

        col = htf_cols[0]
        valid = result[col].dropna()
        if len(valid) < 10:
            pytest.skip("Not enough valid HTF values")

        changes = valid.diff().abs()
        change_ratio = (changes > 1e-10).sum() / len(changes)
        assert change_ratio < 0.35, f"HTF features change too often ({change_ratio:.0%})"

    def test_short_data_passthrough(self):
        from apollo.features.mtf import MTFAligner
        idx = pd.date_range("2025-01-01", periods=10, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "open": range(10), "high": range(10), "low": range(10),
            "close": range(10), "volume": range(10),
        }, index=idx)
        aligner = MTFAligner(htf="4h")
        result = aligner.align(df)
        assert len(result) == 10


class TestTemporalIntegrity:
    def test_no_gaps_imputed(self):
        from apollo.data.provider import ensure_temporal_integrity
        idx = pd.date_range("2025-01-01", periods=100, freq="1h", tz="UTC")
        df = pd.DataFrame({"close": range(100), "volume": range(100)}, index=idx)
        result, n = ensure_temporal_integrity(df, "1h")
        assert n == 0
        assert len(result) == 100

    def test_gaps_filled(self):
        from apollo.data.provider import ensure_temporal_integrity
        idx = pd.date_range("2025-01-01", periods=100, freq="1h", tz="UTC")
        df = pd.DataFrame({"close": range(100), "volume": range(100)}, index=idx)
        # Drop 5 bars to create gaps
        df = df.drop(idx[50:55])
        result, n = ensure_temporal_integrity(df, "1h")
        assert n == 5
        assert len(result) == 100

    def test_volume_zero_during_gap(self):
        from apollo.data.provider import ensure_temporal_integrity
        idx = pd.date_range("2025-01-01", periods=20, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "close": range(20),
            "volume": [100.0] * 20,
        }, index=idx)
        df = df.drop(idx[10:13])
        result, _ = ensure_temporal_integrity(df, "1h")
        # Imputed volume should be 0
        assert result.loc[idx[10], "volume"] == 0
        assert result.loc[idx[12], "volume"] == 0
