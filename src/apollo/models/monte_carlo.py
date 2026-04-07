"""
Monte Carlo Simulator (EGARCH-Bootstrap)
=========================================
Generates realistic future price paths by combining conditional volatility
forecasting (EGARCH) with empirical residual bootstrapping, conditioned
on the current HMM market regime.

Pipeline:
  1. Fit EGARCH(1,1) with Student-t innovations on historical returns
  2. Extract standardized residuals
  3. Segregate residuals by HMM regime
  4. Simulate: r_t = sigma_t * epsilon_t, P_t = P_0 * exp(sum(r))

Fallback chain: EGARCH -> GARCH -> constant volatility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from arch import arch_model

from apollo.errors import ModelError

logger = logging.getLogger("models.monte_carlo")


@dataclass
class MCConfig:
    """Monte Carlo simulation configuration."""
    horizon: int = 24
    n_scenarios: int = 500
    block_size: int = 5
    min_regime_samples: int = 50
    vol_scale: float = 100.0
    winsorize_sigma: float = 4.0
    max_return_per_step: float = 0.05
    seed: int | None = None


class MonteCarloSimulator:
    """
    Hybrid GARCH-Bootstrap Monte Carlo simulator.
    Regime-conditioned residuals for realistic scenario generation.
    """

    def __init__(self, config: MCConfig = None):
        self.config = config or MCConfig()
        self._model_result = None
        self._resid_by_regime: dict[int, np.ndarray] = {}
        self._global_resid: np.ndarray = np.array([])
        self._cond_vol: Optional[np.ndarray] = None
        self._fallback_vol: float = 0.01
        self._fitted = False
        self._rng = np.random.default_rng(self.config.seed)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # =====================================================================
    # Fitting
    # =====================================================================

    def fit(self, returns: pd.Series, regimes: pd.Series) -> MonteCarloSimulator:
        """
        Fit EGARCH model and build regime-conditioned residual pools.

        Args:
            returns: Log returns (NOT scaled).
            regimes: Integer regime labels (same index).

        Returns:
            self (chainable).
        """
        df = pd.concat([
            returns.rename("ret"),
            regimes.rename("regime"),
        ], axis=1).dropna()

        if len(df) < 100:
            raise ModelError(
                f"Need >= 100 rows for GARCH, got {len(df)}"
            )

        scaled = df["ret"] * self.config.vol_scale

        # Try EGARCH -> GARCH -> constant vol
        self._model_result = self._fit_garch(scaled)

        if self._model_result is not None:
            std_resid = self._model_result.std_resid
            df = df.loc[std_resid.index]
            df["std_resid"] = std_resid.values
        else:
            # Constant vol fallback
            hist_vol = df["ret"].std()
            self._fallback_vol = hist_vol * self.config.vol_scale
            df["std_resid"] = df["ret"] / hist_vol
            logger.warning("Using constant-volatility fallback")

        # Segregate residuals by regime
        for rid in sorted(df["regime"].unique()):
            r = df.loc[df["regime"] == rid, "std_resid"].dropna().values
            if len(r) >= self.config.min_regime_samples:
                self._resid_by_regime[int(rid)] = self._winsorize(r, f"regime_{rid}")
                logger.info("  Regime %s: %d residuals", rid, len(r))
            else:
                logger.warning("  Regime %s: %d samples (below min), using global", rid, len(r))

        self._global_resid = self._winsorize(
            df["std_resid"].dropna().values, "global"
        )

        # Cache vol forecast
        self._cond_vol = self._forecast_vol()
        self._fitted = True

        logger.info(
            "MC fit complete: %d global residuals, %d regime buckets",
            len(self._global_resid), len(self._resid_by_regime),
        )
        return self

    def _fit_garch(self, scaled_returns: pd.Series):
        """Try EGARCH, fall back to GARCH, return None on total failure."""
        for vol_type in ["EGARCH", "GARCH"]:
            try:
                kwargs = {"vol": vol_type, "p": 1, "q": 1, "dist": "t", "rescale": False}
                if vol_type == "EGARCH":
                    kwargs["o"] = 1
                am = arch_model(scaled_returns, **kwargs)
                result = am.fit(disp="off", show_warning=False)
                logger.info("%s(1,1) fit OK. BIC=%.2f", vol_type, result.bic)
                return result
            except Exception as e:
                logger.warning("%s failed: %s", vol_type, e)
        logger.error("All GARCH models failed")
        return None

    def _forecast_vol(self) -> np.ndarray:
        """Compute per-step conditional volatility forecast."""
        cfg = self.config
        if self._model_result is None:
            return np.full(cfg.horizon, self._fallback_vol / cfg.vol_scale)

        try:
            forecasts = self._model_result.forecast(
                horizon=cfg.horizon,
                method="simulation",
                simulations=1000,
                reindex=False,
            )
            var_scaled = forecasts.variance.iloc[-1].values
            vol = np.sqrt(np.maximum(var_scaled, 1e-12)) / cfg.vol_scale

            # Cap at 2x last fitted vol
            last_vol = self._model_result.conditional_volatility.iloc[-1] / cfg.vol_scale
            max_vol = last_vol * 2.0
            n_capped = int(np.sum(vol > max_vol))
            vol = np.minimum(vol, max_vol)

            if n_capped > 0:
                logger.warning("Capped %d/%d vol steps at %.4f%%", n_capped, cfg.horizon, max_vol * 100)

            return vol

        except Exception as e:
            logger.warning("Vol forecast failed (%s), using last fitted vol", e)
            last_v = self._model_result.conditional_volatility.iloc[-1]
            return np.full(cfg.horizon, last_v / cfg.vol_scale)

    def _winsorize(self, resid: np.ndarray, label: str = "") -> np.ndarray:
        """Clip at +/- winsorize_sigma."""
        cap = self.config.winsorize_sigma
        n_clipped = int(np.sum(np.abs(resid) > cap))
        result = np.clip(resid, -cap, cap)
        if n_clipped > 0:
            logger.info("  Winsorized %d residuals at +/-%.0f sigma (%s)", n_clipped, cap, label)
        return result

    # =====================================================================
    # Simulation
    # =====================================================================

    def simulate(self, current_price: float, regime_id: int) -> np.ndarray:
        """
        Generate N future price paths.

        Args:
            current_price: Current asset price.
            regime_id: Current HMM regime (integer).

        Returns:
            ndarray of shape (n_scenarios, horizon) - price paths.
        """
        if not self._fitted:
            raise ModelError("MC not fitted. Call fit() first.")

        cfg = self.config
        resid = self._resid_by_regime.get(int(regime_id), self._global_resid)
        if len(resid) == 0:
            resid = self._global_resid

        logger.info(
            "Simulating %d paths x %d steps (regime=%d, n_resid=%d)",
            cfg.n_scenarios, cfg.horizon, regime_id, len(resid),
        )

        # Block bootstrap
        sampled = self._block_bootstrap(resid)

        # r_t = sigma_t * epsilon_t
        returns = self._cond_vol[np.newaxis, :] * sampled
        returns = np.clip(returns, -cfg.max_return_per_step, cfg.max_return_per_step)

        # Price paths: P_t = P_0 * exp(cumsum(r))
        paths = current_price * np.exp(np.cumsum(returns, axis=1))
        paths = np.nan_to_num(paths, nan=current_price, posinf=current_price, neginf=current_price)

        return paths

    def _block_bootstrap(self, resid: np.ndarray) -> np.ndarray:
        """Vectorized block bootstrap: (n_scenarios, horizon)."""
        n_resid = len(resid)
        bs = min(self.config.block_size, n_resid)
        n_blocks = int(np.ceil(self.config.horizon / bs))
        max_start = max(1, n_resid - bs)

        starts = self._rng.integers(0, max_start, size=(self.config.n_scenarios, n_blocks))

        rows = []
        for row_starts in starts:
            blocks = np.concatenate([resid[s:s + bs] for s in row_starts])
            rows.append(blocks[:self.config.horizon])

        return np.array(rows)

    # =====================================================================
    # Statistics
    # =====================================================================

    @staticmethod
    def path_statistics(paths: np.ndarray, current_price: float) -> dict:
        """
        Compute comprehensive statistics from MC paths.
        """
        final = paths[:, -1]
        rets = (final - current_price) / current_price

        running_max = np.maximum.accumulate(paths, axis=1)
        drawdowns = (paths - running_max) / running_max
        max_dd = np.min(drawdowns, axis=1)

        var_5 = np.percentile(rets, 5)
        cvar_5 = float(np.mean(rets[rets <= var_5])) if np.any(rets <= var_5) else float(var_5)

        expected = float(np.mean(rets))

        return {
            "expected_return": expected,
            "median_return": float(np.median(rets)),
            "VaR_5pct": float(var_5),
            "CVaR_5pct": cvar_5,
            "best_case_95pct": float(np.percentile(rets, 95)),
            "prob_profit": float(np.mean(rets > 0)),
            "expected_max_drawdown": float(np.mean(max_dd)),
            "path_vol": float(np.std(rets)),
            "risk_reward_ratio": float(abs(expected / cvar_5)) if cvar_5 != 0 else 0.0,
        }

    @staticmethod
    def cone_percentiles(
        paths: np.ndarray,
        percentiles: list[int] = None,
    ) -> dict[int, np.ndarray]:
        """Projection cone percentiles at each time step."""
        if percentiles is None:
            percentiles = [5, 10, 25, 50, 75, 90, 95]
        return {p: np.percentile(paths, p, axis=0) for p in percentiles}

    # =====================================================================
    # Persistence
    # =====================================================================

    def save(self, path: str | Path) -> None:
        if not self._fitted:
            raise ModelError("Cannot save unfitted MC model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model_result": self._model_result,
            "resid_by_regime": self._resid_by_regime,
            "global_resid": self._global_resid,
            "cond_vol": self._cond_vol,
            "config": self.config,
        }, path, compress=3)
        logger.info("MC model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> MonteCarloSimulator:
        path = Path(path)
        data = joblib.load(path)
        instance = cls(config=data["config"])
        instance._model_result = data["model_result"]
        instance._resid_by_regime = data["resid_by_regime"]
        instance._global_resid = data["global_resid"]
        instance._cond_vol = data["cond_vol"]
        instance._fitted = True
        logger.info("MC model loaded from %s", path)
        return instance
