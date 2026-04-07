"""
HMM Regime Detector
====================
Fits a Gaussian Mixture HMM on orthogonal market features to detect latent
regimes (trending, ranging, volatile, quiet).

Key upgrade over old code:
  - Labels are matched to fixed archetypes via L2 distance on centroids,
    so they stay stable across refits (no state permutation bug).
  - Sticky transition matrix prevents regime flickering.
  - Out-of-Distribution detection via log-likelihood threshold.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM
from sklearn.preprocessing import StandardScaler

from apollo.errors import ModelError, RegimeError

logger = logging.getLogger("models.regime")

# Fixed archetypes -- labels are always assigned by nearest centroid match
# Dimensions: [volatility, trendiness] (both standardized)
ARCHETYPES = {
    "High Volatility (Trending)":   np.array([+1.0, +1.0]),
    "Low Volatility (Trending)":    np.array([-1.0, +1.0]),
    "High Volatility (Ranging)":    np.array([+1.0, -1.0]),
    "Low Volatility (Quiet Range)": np.array([-1.0, -1.0]),
}

# Features chosen for orthogonality: vol, trend, leverage, speculation
REGIME_FEATURES = [
    "gk_vol",              # Garman-Klass volatility
    "autocorr_w20_l1",     # Lag-1 autocorrelation (trendiness proxy)
]

OPTIONAL_FEATURES = [
    "funding_zscore",      # Leverage stress
    "vol_fut_spot_ratio",  # Speculation ratio
]


@dataclass
class RegimeConfig:
    """Configuration for the HMM regime detector."""
    n_states: int = 4
    n_mix: int = 2
    covariance_type: str = "diag"  # 'diag' avoids singularity issues
    sticky_prob: float = 0.95
    n_restarts: int = 10
    ood_percentile: float = 1.0
    smoothing_span: int = 5
    max_iter: int = 150


class RegimeDetector:
    """
    HMM-based market regime detector.

    Usage:
        detector = RegimeDetector()
        detector.fit(feature_df)
        regime_df = detector.predict(feature_df)
    """

    def __init__(self, config: RegimeConfig = None):
        self.config = config or RegimeConfig()
        self._model: Optional[GMMHMM] = None
        self._scaler = StandardScaler()
        self._feature_cols: list[str] = []
        self._label_map: dict[int, str] = {}
        self._ood_threshold: float = -np.inf
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def label_map(self) -> dict[int, str]:
        """State ID -> human-readable label."""
        return dict(self._label_map)

    # =====================================================================
    # Feature selection
    # =====================================================================

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and validate regime features."""
        # Required features
        missing = [f for f in REGIME_FEATURES if f not in df.columns]
        if missing:
            raise RegimeError(f"Missing required regime features: {missing}")

        # Use optional features if available
        cols = list(REGIME_FEATURES)
        for feat in OPTIONAL_FEATURES:
            if feat in df.columns:
                cols.append(feat)

        self._feature_cols = cols
        X = df[cols].copy()

        # Slight smoothing for HMM stability
        for col in cols:
            X[col] = X[col].ewm(span=self.config.smoothing_span, adjust=False).mean()

        return X

    # =====================================================================
    # Label assignment (stable across refits)
    # =====================================================================

    def _assign_labels(self, X_scaled: np.ndarray, states: np.ndarray) -> dict[int, str]:
        """
        Assign semantic labels by matching each state's centroid to the
        nearest archetype. Uses only [vol, trend] dimensions (first 2 cols).
        """
        n = self.config.n_states
        centroids = np.zeros((n, 2))

        for sid in range(n):
            mask = states == sid
            if mask.sum() > 0:
                # Use first 2 features (vol, trend) for matching
                centroids[sid] = X_scaled[mask, :2].mean(axis=0)

        # Greedy assignment: closest centroid gets the archetype
        archetype_names = list(ARCHETYPES.keys())
        archetype_vecs = np.array(list(ARCHETYPES.values()))
        used_archetypes: set[int] = set()
        labels: dict[int, str] = {}

        # Sort by distance to best archetype (greedy matching)
        assignments = []
        for sid in range(n):
            dists = np.linalg.norm(archetype_vecs - centroids[sid], axis=1)
            assignments.append((sid, dists))

        # Assign greedily: best match first
        for _ in range(n):
            best_sid, best_arch, best_dist = -1, -1, np.inf
            for sid, dists in assignments:
                if sid in [s for s, _ in labels.items()]:
                    continue
                for aidx in range(len(archetype_names)):
                    if aidx in used_archetypes:
                        continue
                    if dists[aidx] < best_dist:
                        best_sid, best_arch, best_dist = sid, aidx, dists[aidx]

            if best_sid >= 0:
                labels[best_sid] = archetype_names[best_arch]
                used_archetypes.add(best_arch)

        # Fallback for any unmatched states
        for sid in range(n):
            if sid not in labels:
                labels[sid] = "Unknown"

        return labels

    # =====================================================================
    # Fitting
    # =====================================================================

    def _build_sticky_transmat(self) -> np.ndarray:
        """Transition matrix with high diagonal to prevent flickering."""
        n = self.config.n_states
        off_diag = (1.0 - self.config.sticky_prob) / (n - 1)
        mat = np.full((n, n), off_diag)
        np.fill_diagonal(mat, self.config.sticky_prob)
        return mat

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the HMM with multiple random restarts, keeping the best model.

        Args:
            df: DataFrame with feature columns from the pipeline.
        """
        cfg = self.config
        X = self._select_features(df)
        X_clean = X.dropna()

        if len(X_clean) < 200:
            raise RegimeError(
                f"Need >= 200 clean rows for HMM, got {len(X_clean)}"
            )

        if len(X_clean) < len(X):
            logger.warning(
                "Dropped %d NaN rows before HMM fit", len(X) - len(X_clean)
            )

        X_scaled = self._scaler.fit_transform(X_clean)

        best_model = None
        best_score = -np.inf

        logger.info(
            "Fitting GMMHMM (%d states, %d restarts)...",
            cfg.n_states, cfg.n_restarts,
        )

        for i in range(cfg.n_restarts):
            model = GMMHMM(
                n_components=cfg.n_states,
                n_mix=cfg.n_mix,
                covariance_type=cfg.covariance_type,
                n_iter=cfg.max_iter,
                random_state=42 + i,
                init_params="smcw",
                params="stmcw",
            )
            model.transmat_ = self._build_sticky_transmat()

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*ill-defined.*")
                    warnings.filterwarnings("ignore", message=".*covariance.*")
                    model.fit(X_scaled)
                    score = model.score(X_scaled)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                logger.debug("HMM restart %d failed: %s", i, e)

        if best_model is None:
            raise RegimeError("All HMM fitting attempts failed")

        self._model = best_model
        logger.info("Best HMM log-likelihood: %.2f", best_score)

        # OOD threshold
        ll_per_frame = self._model.score_samples(X_scaled)[0]
        self._ood_threshold = float(
            np.percentile(ll_per_frame, cfg.ood_percentile)
        )

        # Stable label assignment
        states = self._model.predict(X_scaled)
        self._label_map = self._assign_labels(X_scaled, states)
        self._is_fitted = True

        for sid, label in sorted(self._label_map.items()):
            mask = states == sid
            n_bars = int(mask.sum())
            logger.info("  State %d: %s (%d bars)", sid, label, n_bars)

    # =====================================================================
    # Prediction
    # =====================================================================

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime probabilities and OOD flag for each bar.

        Returns DataFrame with columns:
          - hmm_regime (int)
          - hmm_regime_label (str)
          - hmm_ood (bool)
          - hmm_prob_state_0 ... hmm_prob_state_N (float)
        """
        if not self._is_fitted:
            raise RegimeError("Model not fitted. Call fit() first.")

        # Use the same feature columns that were used during training
        missing = [f for f in self._feature_cols if f not in df.columns]
        if missing:
            logger.debug("Missing regime features for prediction: %s -- filling with 0", missing)

        X = df.reindex(columns=self._feature_cols, fill_value=0.0).copy()
        # Apply same smoothing as training
        for col in self._feature_cols:
            X[col] = X[col].ewm(span=self.config.smoothing_span, adjust=False).mean()

        out = pd.DataFrame(index=X.index)

        valid_mask = X.notna().all(axis=1)
        X_valid = X[valid_mask]

        if X_valid.empty:
            logger.error("No valid rows for regime prediction")
            return out

        X_scaled = self._scaler.transform(X_valid)
        probs = self._model.predict_proba(X_scaled)
        states = np.argmax(probs, axis=1)
        ll_per_frame = self._model.score_samples(X_scaled)[0]
        is_ood = ll_per_frame < self._ood_threshold

        # Initialize output columns
        n = self.config.n_states
        out["hmm_regime"] = np.nan
        out["hmm_ood"] = False
        for i in range(n):
            out[f"hmm_prob_state_{i}"] = np.nan

        # Fill valid rows
        idx = X_valid.index
        out.loc[idx, "hmm_regime"] = pd.Series(
            states.astype(float), index=idx
        )
        out.loc[idx, "hmm_ood"] = pd.Series(is_ood, index=idx)
        for i in range(n):
            out.loc[idx, f"hmm_prob_state_{i}"] = pd.Series(
                probs[:, i], index=idx
            )

        out["hmm_regime_label"] = out["hmm_regime"].map(self._label_map)

        ood_pct = out["hmm_ood"].mean() * 100
        if ood_pct > 10:
            logger.debug("OOD in %.1f%% of data -- model may be stale", ood_pct)

        return out

    # =====================================================================
    # Persistence
    # =====================================================================

    def save(self, path: str | Path) -> None:
        """Save fitted model to disk."""
        if not self._is_fitted:
            raise RegimeError("Cannot save unfitted model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self._model,
            "scaler": self._scaler,
            "feature_cols": self._feature_cols,
            "label_map": self._label_map,
            "ood_threshold": self._ood_threshold,
            "config": self.config,
        }, path, compress=3)
        logger.info("Regime model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> RegimeDetector:
        """Load a fitted model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        data = joblib.load(path)
        instance = cls(config=data["config"])
        instance._model = data["model"]
        instance._scaler = data["scaler"]
        instance._feature_cols = data["feature_cols"]
        instance._label_map = data["label_map"]
        instance._ood_threshold = data["ood_threshold"]
        instance._is_fitted = True
        logger.info("Regime model loaded from %s", path)
        return instance
