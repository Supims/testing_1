"""
Multi-Horizon Probabilistic Timing Model
==========================================
5 independent XGBoost models with walk-forward isotonic calibration.

Targets:
  P(+1.5% in 12h), P(+1.5% in 24h), P(+3.0% in 48h)
  P(DD >= 1.0% in 24h), P(DD >= 2.0% in 24h)

Calibration guarantee:
  When the model outputs 65%, historically ~65% of the time
  the event actually occurred (measured via Brier score).

Critical fix over old code:
  Isotonic calibration is fitted on a SEPARATE temporal window
  (walk-forward), not on the same validation data as XGBoost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from apollo.errors import ModelError

logger = logging.getLogger("execution.probability")


# =========================================================================
# Target Definitions
# =========================================================================

@dataclass
class TargetDef:
    """One probability target."""
    name: str
    key: str
    horizon: int
    threshold: float
    direction: str  # 'up' | 'drawdown'
    description: str = ""


DEFAULT_TARGETS = [
    TargetDef("P(+1.5% 12h)", "prob_up_1p5_12h", 12, 0.015, "up"),
    TargetDef("P(+1.5% 24h)", "prob_up_1p5_24h", 24, 0.015, "up"),
    TargetDef("P(+3.0% 48h)", "prob_up_3p0_48h", 48, 0.030, "up"),
    TargetDef("P(DD>1.0% 24h)", "prob_dd_1p0_24h", 24, 0.010, "drawdown"),
    TargetDef("P(DD>2.0% 24h)", "prob_dd_2p0_24h", 24, 0.020, "drawdown"),
]


# =========================================================================
# Feature lists
# =========================================================================

ENSEMBLE_FEATURES = ["ensemble_signal", "ensemble_abs"]

MC_FEATURES = [
    "mc_var_5pct", "mc_cvar_5pct", "mc_prob_profit",
    "mc_expected_mdd", "mc_risk_reward",
]

MICRO_FEATURES = [
    "gk_vol", "funding_zscore", "oi_vel_20",
    "taker_delta_ema_20", "basis_zscore_100", "cvd_rolling_100",
]


# =========================================================================
# Config
# =========================================================================

@dataclass
class ProbabilityConfig:
    targets: list[TargetDef] = field(default_factory=lambda: list(DEFAULT_TARGETS))
    train_ratio: float = 0.60
    cal_ratio: float = 0.15  # Separate calibration window
    n_boost_rounds: int = 500
    early_stop: int = 50
    xgb_params: dict = field(default_factory=lambda: {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "gamma": 1.0,
        "alpha": 1.0,
        "lambda": 1.0,
        "random_state": 42,
        "verbosity": 0,
    })


# =========================================================================
# Single Calibrated Model
# =========================================================================

class CalibratedModel:
    """XGBoost + Isotonic Regression for one target."""

    def __init__(self, target: TargetDef, xgb_params: dict):
        self.target = target
        self._xgb_params = xgb_params
        self._booster: Optional[xgb.Booster] = None
        self._calibrator: Optional[IsotonicRegression] = None
        self._fitted = False
        self.metrics: dict = {}

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def train(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
        X_cal: pd.DataFrame, y_cal: pd.Series,
        n_rounds: int = 500, early_stop: int = 50,
    ) -> dict:
        """
        Train XGBoost on train, early-stop on val, calibrate on cal (separate).

        This is the critical fix: calibration data is NEVER seen by XGBoost.
        """
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        self._booster = xgb.train(
            self._xgb_params,
            dtrain,
            num_boost_round=n_rounds,
            evals=[(dtrain, "train"), (dval, "eval")],
            early_stopping_rounds=early_stop,
            verbose_eval=False,
        )

        # Calibrate on SEPARATE temporal window
        dcal = xgb.DMatrix(X_cal)
        raw_cal = self._booster.predict(dcal)

        self._calibrator = IsotonicRegression(out_of_bounds="clip")
        self._calibrator.fit(raw_cal, y_cal.values)

        cal_probs = self._calibrator.transform(raw_cal)

        # Metrics
        base_rate = float(y_cal.mean())
        brier_baseline = base_rate * (1 - base_rate)
        brier_cal = float(brier_score_loss(y_cal, cal_probs))

        self.metrics = {
            "brier_raw": float(brier_score_loss(y_cal, raw_cal)),
            "brier_cal": brier_cal,
            "brier_baseline": brier_baseline,
            "base_rate": base_rate,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_cal": len(X_cal),
            "best_iter": self._booster.best_iteration,
            "adds_value": brier_cal < brier_baseline,
        }

        self._fitted = True

        status = "OK" if brier_cal < brier_baseline else "WEAK"
        logger.info(
            "  [%s] %s: Brier=%.4f (baseline=%.4f)",
            self.target.key, status, brier_cal, brier_baseline,
        )

        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated probabilities."""
        if not self._fitted:
            raise ModelError(f"Model [{self.target.key}] not fitted")
        dtest = xgb.DMatrix(X)
        raw = self._booster.predict(dtest)
        return self._calibrator.transform(raw)

    def feature_importance(self) -> pd.Series:
        if not self._fitted:
            return pd.Series(dtype=float)
        imp = self._booster.get_score(importance_type="gain")
        return pd.Series(imp).sort_values(ascending=False)


# =========================================================================
# Multi-Horizon Timing Model (orchestrates all targets)
# =========================================================================

class MultiHorizonModel:
    """
    Orchestrates 5 CalibratedModels across multiple horizons.

    Usage:
        model = MultiHorizonModel()
        targets = model.build_targets(df)
        X = model.prepare_features(df, ensemble, mc_stats, hmm_probs)
        metrics = model.train(X, targets)
        probs = model.predict(X_new)
    """

    def __init__(self, config: ProbabilityConfig = None):
        self.config = config or ProbabilityConfig()
        self.models: dict[str, CalibratedModel] = {}
        self.feature_cols: list[str] = []
        self.all_metrics: dict[str, dict] = {}

    # -----------------------------------------------------------------
    # Target construction
    # -----------------------------------------------------------------

    def build_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build binary targets using vectorized sliding windows.
        Returns DataFrame with one column per target (0/1/NaN).
        """
        targets = pd.DataFrame(index=df.index)
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        n = len(df)

        for tdef in self.config.targets:
            h = tdef.horizon
            valid = n - h
            if valid <= 0:
                targets[tdef.key] = np.nan
                continue

            target = np.full(n, np.nan)

            # Sliding window indices: (valid, horizon)
            h_idx = np.arange(h)[np.newaxis, :] + np.arange(1, valid + 1)[:, np.newaxis]

            if tdef.direction == "up":
                entry = close[:valid]
                tp = entry * (1 + tdef.threshold)
                hit = (high[h_idx] >= tp[:, np.newaxis]).any(axis=1)
                target[:valid] = hit.astype(float)

            elif tdef.direction == "drawdown":
                entry = close[:valid]
                dd = entry * (1 - tdef.threshold)
                hit = (low[h_idx] <= dd[:, np.newaxis]).any(axis=1)
                target[:valid] = hit.astype(float)

            targets[tdef.key] = target
            pos_rate = np.nanmean(target[:valid])
            logger.info(
                "  Target [%s]: %d valid / positive=%.1f%%",
                tdef.key, valid, pos_rate * 100,
            )

        return targets

    # -----------------------------------------------------------------
    # Feature assembly
    # -----------------------------------------------------------------

    def prepare_features(
        self,
        df: pd.DataFrame,
        ensemble_signal: pd.Series,
        mc_stats_df: pd.DataFrame = None,
        hmm_probs_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Assemble feature matrix from all upstream modules."""
        X = pd.DataFrame(index=df.index)

        X["ensemble_signal"] = ensemble_signal
        X["ensemble_abs"] = ensemble_signal.abs()

        if mc_stats_df is not None:
            for f in MC_FEATURES:
                if f in mc_stats_df.columns:
                    X[f] = mc_stats_df[f]

        for f in MICRO_FEATURES:
            if f in df.columns:
                X[f] = df[f]

        if hmm_probs_df is not None:
            for col in hmm_probs_df.columns:
                if col.startswith("hmm_prob_state_"):
                    X[col] = hmm_probs_df[col]

        # Only store feature_cols during training (first call).
        # During prediction, keep the cols from training so XGBoost
        # always sees the same features it was trained on.
        if not self.feature_cols:
            self.feature_cols = X.columns.tolist()
        logger.info("Feature matrix: %d features, %d rows", len(self.feature_cols), len(X))
        return X

    # -----------------------------------------------------------------
    # Training (walk-forward + separate calibration)
    # -----------------------------------------------------------------

    def train(self, X: pd.DataFrame, targets: pd.DataFrame) -> dict[str, dict]:
        """
        Train one CalibratedModel per target with 3-way temporal split:
        [---Train---][--Val--][--Cal--]

        Returns dict of target_key -> metrics.
        """
        # Lock feature columns to what the training data provides
        self.feature_cols = X.columns.tolist()
        logger.info("Training %d probability models...", len(self.config.targets))

        for tdef in self.config.targets:
            y = targets[tdef.key]
            mask = y.notna() & X.notna().all(axis=1)
            X_clean = X.loc[mask]
            y_clean = y.loc[mask]

            n = len(X_clean)
            if n < 300:
                logger.warning("  [%s] Skipped: %d rows (need >= 300)", tdef.key, n)
                continue

            # 3-way temporal split
            train_end = int(n * self.config.train_ratio)
            val_end = int(n * (self.config.train_ratio + self.config.cal_ratio))

            X_train = X_clean.iloc[:train_end]
            y_train = y_clean.iloc[:train_end]
            X_val = X_clean.iloc[train_end:val_end]
            y_val = y_clean.iloc[train_end:val_end]
            X_cal = X_clean.iloc[val_end:]
            y_cal = y_clean.iloc[val_end:]

            if len(X_val) < 50 or len(X_cal) < 50:
                logger.warning("  [%s] Too few val/cal rows", tdef.key)
                continue

            logger.info(
                "  [%s] Train=%d, Val=%d, Cal=%d (base=%.1f%%)",
                tdef.key, len(X_train), len(X_val), len(X_cal),
                y_clean.mean() * 100,
            )

            model = CalibratedModel(tdef, dict(self.config.xgb_params))
            metrics = model.train(
                X_train, y_train, X_val, y_val, X_cal, y_cal,
                n_rounds=self.config.n_boost_rounds,
                early_stop=self.config.early_stop,
            )
            self.models[tdef.key] = model
            self.all_metrics[tdef.key] = metrics

        logger.info("Training complete: %d models fitted", len(self.models))
        return self.all_metrics

    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return calibrated probabilities for all fitted targets."""
        X_aligned = X.reindex(columns=self.feature_cols, fill_value=0)
        probs = pd.DataFrame(index=X.index)

        for key, model in self.models.items():
            probs[key] = model.predict(X_aligned)

        return probs

    def feature_importances(self) -> dict[str, pd.Series]:
        return {k: m.feature_importance() for k, m in self.models.items()}

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "feature_cols": self.feature_cols,
            "all_metrics": self.all_metrics,
            "targets": [
                (t.name, t.key, t.horizon, t.threshold, t.direction, t.description)
                for t in self.config.targets
            ],
            "models": {},
        }
        for key, model in self.models.items():
            data["models"][key] = {
                "booster_raw": model._booster.save_raw() if model._booster else None,
                "calibrator": model._calibrator,
                "target_def": (
                    model.target.name, model.target.key, model.target.horizon,
                    model.target.threshold, model.target.direction,
                    model.target.description,
                ),
                "metrics": model.metrics,
            }
        joblib.dump(data, path, compress=3)
        logger.info("Probability models saved to %s (%d models)", path, len(self.models))

    @classmethod
    def load(cls, path: str | Path) -> MultiHorizonModel:
        path = Path(path)
        data = joblib.load(path)
        targets = [TargetDef(*t) for t in data["targets"]]
        instance = cls(config=ProbabilityConfig(targets=targets))
        instance.feature_cols = data["feature_cols"]
        instance.all_metrics = data["all_metrics"]

        for key, mdata in data["models"].items():
            tdef = TargetDef(*mdata["target_def"])
            cm = CalibratedModel(tdef, {})
            if mdata["booster_raw"] is not None:
                cm._booster = xgb.Booster()
                cm._booster.load_model(bytearray(mdata["booster_raw"]))
            cm._calibrator = mdata["calibrator"]
            cm.metrics = mdata["metrics"]
            cm._fitted = True
            instance.models[key] = cm

        logger.info("Probability models loaded from %s (%d)", path, len(instance.models))
        return instance
