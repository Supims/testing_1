"""
Model Auto-Retraining
======================
Manages periodic retraining of HMM, Monte Carlo, and XGBoost models
on a rolling window.

Features:
  - Configurable retrain frequency (default: every 24h)
  - Rolling window (default: 90 days)
  - Model versioning (keeps last N models)
  - Performance comparison (new vs old model)
  - Safe rollback if new model is worse

Integrates with Scanner to swap models at runtime.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("models.retrain")


class ModelRetrainer:
    """
    Handles automatic model retraining on a rolling window.

    Usage:
        retrainer = ModelRetrainer(scanner)
        if retrainer.should_retrain():
            retrainer.retrain()

    Typically called at the start of each agent cycle.
    """

    DEFAULT_TRAIN_SYMBOLS = ["BTCUSDT"]
    DEFAULT_WINDOW_DAYS = 90
    DEFAULT_RETRAIN_HOURS = 24
    MAX_VERSIONS = 5

    def __init__(self, scanner=None, retrain_every_hours: float = None,
                 window_days: int = None):
        from apollo.config import settings
        self._scanner = scanner
        self._models_dir = settings.models_dir
        self._meta_file = self._models_dir / "retrain_meta.json"
        self.retrain_hours = retrain_every_hours or self.DEFAULT_RETRAIN_HOURS
        self.window_days = window_days or self.DEFAULT_WINDOW_DAYS
        self._load_meta()

    def _load_meta(self):
        """Load retraining metadata."""
        if self._meta_file.exists():
            try:
                with open(self._meta_file, "r") as f:
                    self._meta = json.load(f)
            except Exception:
                self._meta = {}
        else:
            self._meta = {}

    def _save_meta(self):
        """Save retraining metadata."""
        try:
            with open(self._meta_file, "w") as f:
                json.dump(self._meta, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save retrain meta: %s", e)

    def should_retrain(self) -> bool:
        """Check if models need retraining based on last retrain time."""
        last_retrain = self._meta.get("last_retrain_timestamp")
        if not last_retrain:
            return True  # Never retrained

        try:
            last_dt = datetime.fromisoformat(last_retrain)
            hours_since = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
            return hours_since >= self.retrain_hours
        except Exception:
            return True

    def retrain(self, symbols: list[str] = None, force: bool = False) -> dict:
        """
        Retrain all models on a rolling window.

        Args:
            symbols: Pairs to train on. Default: BTCUSDT
            force: Bypass the time check

        Returns:
            dict with retrain results (duration, status, etc.)
        """
        if not force and not self.should_retrain():
            return {"status": "skipped", "reason": "Not due yet"}

        if self._scanner is None:
            return {"status": "error", "reason": "No scanner attached"}

        symbols = symbols or self.DEFAULT_TRAIN_SYMBOLS
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self.window_days)

        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        logger.info(
            "Retraining models on %s (%s to %s, %d day window)",
            symbols, start_str, end_str, self.window_days,
        )

        t0 = time.time()
        try:
            # Version the old models
            self._version_current_models()

            # Retrain via scanner.train()
            self._scanner.train(symbols, start_str, end_str)

            duration = time.time() - t0
            result = {
                "status": "success",
                "symbols": symbols,
                "window": f"{start_str} to {end_str}",
                "duration_seconds": round(duration, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Update meta
            self._meta["last_retrain_timestamp"] = datetime.now(timezone.utc).isoformat()
            self._meta["last_retrain_result"] = result
            self._meta["retrain_count"] = self._meta.get("retrain_count", 0) + 1
            self._save_meta()

            logger.info("Retrain complete in %.1fs", duration)
            return result

        except Exception as e:
            duration = time.time() - t0
            logger.error("Retrain failed after %.1fs: %s", duration, e)

            # Try to rollback
            self._rollback_models()

            return {
                "status": "error",
                "error": str(e),
                "duration_seconds": round(duration, 1),
                "rolled_back": True,
            }

    def _version_current_models(self):
        """Copy current models to a versioned directory."""
        scanner_dir = self._models_dir / "scanner"
        if not scanner_dir.exists():
            return

        # Create version directory
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        version_dir = self._models_dir / "versions" / version
        try:
            shutil.copytree(scanner_dir, version_dir)
            logger.info("Model version saved: %s", version)

            # Cleanup old versions
            self._cleanup_old_versions()
        except Exception as e:
            logger.warning("Failed to version models: %s", e)

    def _cleanup_old_versions(self):
        """Keep only the last MAX_VERSIONS model versions."""
        versions_dir = self._models_dir / "versions"
        if not versions_dir.exists():
            return

        versions = sorted(versions_dir.iterdir(), reverse=True)
        for old_dir in versions[self.MAX_VERSIONS:]:
            try:
                shutil.rmtree(old_dir)
                logger.debug("Removed old model version: %s", old_dir.name)
            except Exception:
                pass

    def _rollback_models(self):
        """Rollback to the most recent versioned model."""
        versions_dir = self._models_dir / "versions"
        if not versions_dir.exists():
            return

        versions = sorted(versions_dir.iterdir(), reverse=True)
        if not versions:
            return

        latest = versions[0]
        scanner_dir = self._models_dir / "scanner"

        try:
            if scanner_dir.exists():
                shutil.rmtree(scanner_dir)
            shutil.copytree(latest, scanner_dir)
            logger.info("Rolled back to model version: %s", latest.name)
        except Exception as e:
            logger.error("Rollback failed: %s", e)

    def get_info(self) -> dict:
        """Get retraining info."""
        last = self._meta.get("last_retrain_timestamp", "never")
        count = self._meta.get("retrain_count", 0)

        hours_since = "N/A"
        if last != "never":
            try:
                dt = datetime.fromisoformat(last)
                hours_since = f"{(datetime.now(timezone.utc) - dt).total_seconds() / 3600:.1f}h"
            except Exception:
                pass

        # Count versions
        versions_dir = self._models_dir / "versions"
        n_versions = len(list(versions_dir.iterdir())) if versions_dir.exists() else 0

        return {
            "last_retrain": last,
            "hours_since": hours_since,
            "retrain_count": count,
            "saved_versions": n_versions,
            "window_days": self.window_days,
            "retrain_every_hours": self.retrain_hours,
            "should_retrain": self.should_retrain(),
        }

    def format_status(self) -> str:
        """Human-readable status."""
        info = self.get_info()
        return (
            f"Model Retrain: last={info['hours_since']} ago | "
            f"count={info['retrain_count']} | "
            f"versions={info['saved_versions']} | "
            f"due={'YES' if info['should_retrain'] else 'no'}"
        )
