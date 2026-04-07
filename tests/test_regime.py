"""Tests for HMM Regime Detector."""

import numpy as np
import pandas as pd
import pytest


class TestRegimeDetector:
    def _make_regime_data(self, n=1000):
        """Create data with gk_vol and autocorr_w20_l1 features."""
        np.random.seed(42)
        idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
        return pd.DataFrame({
            "gk_vol": np.abs(np.random.normal(0.02, 0.01, n)),
            "autocorr_w20_l1": np.random.uniform(-0.5, 0.5, n),
        }, index=idx)

    def test_fit_and_predict_shape(self):
        from apollo.models.regime import RegimeDetector
        df = self._make_regime_data(500)
        det = RegimeDetector()
        det.fit(df)
        assert det.is_fitted

        result = det.predict(df)
        assert "hmm_regime" in result.columns
        assert "hmm_regime_label" in result.columns
        assert "hmm_ood" in result.columns
        assert "hmm_prob_state_0" in result.columns
        assert len(result) == len(df)

    def test_label_map_has_4_states(self):
        from apollo.models.regime import RegimeDetector
        df = self._make_regime_data(500)
        det = RegimeDetector()
        det.fit(df)

        labels = det.label_map
        assert len(labels) == 4
        # All labels should be known archetype names
        valid_names = {
            "High Volatility (Trending)",
            "Low Volatility (Trending)",
            "High Volatility (Ranging)",
            "Low Volatility (Quiet Range)",
        }
        for label in labels.values():
            assert label in valid_names, f"Unexpected label: {label}"

    def test_label_stability(self):
        """Fit 3 times on same data -- labels should be identical."""
        from apollo.models.regime import RegimeDetector
        df = self._make_regime_data(500)

        labels_runs = []
        for _ in range(3):
            det = RegimeDetector()
            det.fit(df)
            labels_runs.append(det.label_map)

        for i in range(1, len(labels_runs)):
            assert labels_runs[i] == labels_runs[0], "Labels changed across refits!"

    def test_ood_detection(self):
        from apollo.models.regime import RegimeDetector
        df = self._make_regime_data(500)
        det = RegimeDetector()
        det.fit(df)

        # Inject extreme outlier data
        outlier_df = df.copy()
        outlier_df["gk_vol"] = 1.0  # extreme
        outlier_df["autocorr_w20_l1"] = 5.0  # extreme

        result = det.predict(outlier_df)
        ood_pct = result["hmm_ood"].mean()
        assert ood_pct > 0.5, f"Expected high OOD rate, got {ood_pct:.1%}"

    def test_missing_features_raises(self):
        from apollo.models.regime import RegimeDetector, RegimeError
        det = RegimeDetector()
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(Exception):
            det.fit(df)

    def test_predict_before_fit_raises(self):
        from apollo.models.regime import RegimeDetector
        det = RegimeDetector()
        df = self._make_regime_data(100)
        with pytest.raises(Exception):
            det.predict(df)

    def test_save_load_roundtrip(self, tmp_path):
        from apollo.models.regime import RegimeDetector
        df = self._make_regime_data(500)
        det = RegimeDetector()
        det.fit(df)

        path = tmp_path / "hmm.joblib"
        det.save(path)

        loaded = RegimeDetector.load(path)
        assert loaded.is_fitted
        assert loaded.label_map == det.label_map

        # Predictions should match
        r1 = det.predict(df)["hmm_regime"].dropna()
        r2 = loaded.predict(df)["hmm_regime"].dropna()
        assert (r1 == r2).all()

    def test_too_little_data_raises(self):
        from apollo.models.regime import RegimeDetector
        df = self._make_regime_data(50)  # below 200 threshold
        det = RegimeDetector()
        with pytest.raises(Exception):
            det.fit(df)
