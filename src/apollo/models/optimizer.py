"""
Strategy Optimizer
====================
Walk-forward optimization of ensemble weights per HMM regime.

Uses Scipy minimize (L-BFGS-B) to maximize rolling Sharpe ratio.
No external dependency on Optuna -- pure scipy.

Usage:
    from apollo.models.optimizer import StrategyOptimizer
    opt = StrategyOptimizer()
    best_weights = opt.optimize(signals_df, hmm_probs_df, returns)
    # best_weights: dict[int, dict[str, float]]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from apollo.models.strategies import ALL_STRATEGY_NAMES

logger = logging.getLogger("models.optimizer")


@dataclass
class OptimizerConfig:
    """Configuration for walk-forward optimization."""
    train_days: int = 60       # Walk-forward training window
    test_days: int = 20        # Walk-forward test window
    step_days: int = 10        # Roll step
    n_restarts: int = 5        # Random restarts for each optimization
    min_sharpe: float = -0.1   # Minimum Sharpe below which we keep priors
    annualization: float = np.sqrt(365 * 24)  # hourly -> annual Sharpe


@dataclass
class OptimizationResult:
    """Result of a single walk-forward fold."""
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    weights_per_regime: dict[int, dict[str, float]] = field(default_factory=dict)


class StrategyOptimizer:
    """
    Walk-forward optimizer that tunes ensemble weights per regime
    to maximize Sharpe ratio.

    Reduces overfitting via:
      - Walk-forward cross-validation (no future leakage)
      - Weight bounds [-1.5, +1.5] (allows inverting signals)
      - L2 regularization (shrinks to equal weight)
      - Averaging across folds
    """

    def __init__(self, config: OptimizerConfig = None):
        self.config = config or OptimizerConfig()
        self.results: list[OptimizationResult] = []

    def _sharpe_from_weights(
        self,
        w: np.ndarray,
        signals: np.ndarray,
        probs: np.ndarray,
        returns: np.ndarray,
        n_strategies: int,
        n_states: int,
        reg_lambda: float = 0.01,
    ) -> float:
        """
        Negative Sharpe ratio (for minimization).

        Args:
            w: flat array of shape (n_states * n_strategies,)
            signals: (T, n_strategies) strategy signals
            probs: (T, n_states) regime probs
            returns: (T,) actual returns
            n_strategies: number of strategies
            n_states: number of HMM states
            reg_lambda: L2 regularization strength
        """
        W = w.reshape(n_states, n_strategies)

        # Ensemble signal for each bar
        T = len(returns)
        ensemble = np.zeros(T)
        for sid in range(n_states):
            state_signal = signals @ W[sid]  # (T,)
            ensemble += state_signal * probs[:, sid]

        # Compress to [-1, 1]
        ensemble = np.tanh(ensemble)

        # PnL = ensemble[t-1] * return[t]  (shift by 1 to avoid lookahead)
        pnl = ensemble[:-1] * returns[1:]

        if len(pnl) < 20 or np.std(pnl) < 1e-12:
            return 0.0  # Can't compute Sharpe

        mean_ret = np.mean(pnl)
        std_ret = np.std(pnl)
        sharpe = (mean_ret / std_ret) * self.config.annualization

        # L2 regularization: pulls weights toward 0 (equal weight proxy)
        penalty = reg_lambda * np.sum(w ** 2)

        return -(sharpe - penalty)  # Negative for minimization

    def _optimize_fold(
        self,
        signals: np.ndarray,
        probs: np.ndarray,
        returns: np.ndarray,
        n_strategies: int,
        n_states: int,
    ) -> tuple[np.ndarray, float]:
        """Optimize weights for one fold. Returns (best_weights, best_sharpe)."""
        n_params = n_states * n_strategies
        bounds = [(-1.5, 1.5)] * n_params

        best_w = None
        best_score = np.inf

        for restart in range(self.config.n_restarts):
            if restart == 0:
                # Start from equal weights
                w0 = np.full(n_params, 1.0 / n_strategies)
            else:
                # Random start
                w0 = np.random.uniform(-0.5, 0.5, n_params)

            try:
                result = minimize(
                    self._sharpe_from_weights,
                    w0,
                    args=(signals, probs, returns, n_strategies, n_states),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 200, "ftol": 1e-8},
                )
                if result.fun < best_score:
                    best_score = result.fun
                    best_w = result.x
            except Exception as e:
                logger.debug("Restart %d failed: %s", restart, e)

        if best_w is None:
            best_w = np.full(n_params, 1.0 / n_strategies)

        return best_w, -best_score  # Return positive Sharpe

    def optimize(
        self,
        signals_df: pd.DataFrame,
        hmm_probs_df: pd.DataFrame,
        returns_series: pd.Series,
    ) -> dict[int, dict[str, float]]:
        """
        Run walk-forward optimization.

        Args:
            signals_df: DataFrame with strategy signal columns
            hmm_probs_df: DataFrame with hmm_prob_state_N columns
            returns_series: Series of fwd returns (close.pct_change())

        Returns:
            Optimized weights per regime:
            {0: {"trend": 1.2, "mean_reversion": -0.3, ...}, ...}
        """
        cfg = self.config
        available = [s for s in ALL_STRATEGY_NAMES if s in signals_df.columns]
        n_strategies = len(available)
        prob_cols = sorted([c for c in hmm_probs_df.columns if "hmm_prob_state_" in c])
        n_states = len(prob_cols)

        if n_strategies == 0 or n_states == 0:
            logger.warning("No strategies or states available for optimization")
            return {}

        # Align all DataFrames
        idx = signals_df.index.intersection(hmm_probs_df.index).intersection(returns_series.index)
        sig_arr = signals_df.loc[idx, available].values
        prob_arr = hmm_probs_df.loc[idx, prob_cols].values
        ret_arr = returns_series.loc[idx].values

        total_bars = len(idx)
        bars_per_day = 24  # 1h candles
        train_bars = cfg.train_days * bars_per_day
        test_bars = cfg.test_days * bars_per_day
        step_bars = cfg.step_days * bars_per_day

        if total_bars < train_bars + test_bars:
            logger.warning(
                "Not enough data for optimization: %d bars < %d needed",
                total_bars, train_bars + test_bars,
            )
            return {}

        # Walk-forward folds
        self.results = []
        all_weights = []
        oos_sharpes = []

        fold = 0
        cursor = 0
        while cursor + train_bars + test_bars <= total_bars:
            train_end = cursor + train_bars
            test_end = train_end + test_bars

            # Train
            train_sig = sig_arr[cursor:train_end]
            train_prob = prob_arr[cursor:train_end]
            train_ret = ret_arr[cursor:train_end]

            best_w, is_sharpe = self._optimize_fold(
                train_sig, train_prob, train_ret, n_strategies, n_states,
            )

            # Test (out-of-sample)
            test_sig = sig_arr[train_end:test_end]
            test_prob = prob_arr[train_end:test_end]
            test_ret = ret_arr[train_end:test_end]
            oos_sharpe = -self._sharpe_from_weights(
                best_w, test_sig, test_prob, test_ret, n_strategies, n_states, reg_lambda=0,
            )

            train_dates = idx[cursor], idx[train_end - 1]
            test_dates = idx[train_end], idx[test_end - 1]

            W = best_w.reshape(n_states, n_strategies)
            weights_dict = {}
            for sid in range(n_states):
                weights_dict[sid] = {available[j]: float(W[sid, j]) for j in range(n_strategies)}

            result = OptimizationResult(
                fold=fold,
                train_start=str(train_dates[0]),
                train_end=str(train_dates[1]),
                test_start=str(test_dates[0]),
                test_end=str(test_dates[1]),
                in_sample_sharpe=is_sharpe,
                out_of_sample_sharpe=oos_sharpe,
                weights_per_regime=weights_dict,
            )
            self.results.append(result)
            all_weights.append(best_w)
            oos_sharpes.append(oos_sharpe)

            logger.info(
                "Fold %d: IS Sharpe=%.3f, OOS Sharpe=%.3f",
                fold, is_sharpe, oos_sharpe,
            )

            fold += 1
            cursor += step_bars

        if not all_weights:
            logger.warning("No valid folds completed")
            return {}

        # Average weights across folds (weighted by OOS Sharpe if positive)
        oos_arr = np.array(oos_sharpes)
        positive_mask = oos_arr > self.config.min_sharpe
        if positive_mask.sum() == 0:
            logger.warning("All folds had negative OOS Sharpe -- using equal weights")
            avg_w = np.full(n_states * n_strategies, 1.0 / n_strategies)
        else:
            # Weight by positive OOS Sharpes
            valid_weights = [w for w, ok in zip(all_weights, positive_mask) if ok]
            valid_sharpes = oos_arr[positive_mask]
            sharpe_weights = valid_sharpes / valid_sharpes.sum()
            avg_w = sum(w * s for w, s in zip(valid_weights, sharpe_weights))

        # Build output
        W_final = avg_w.reshape(n_states, n_strategies)
        optimized = {}
        for sid in range(n_states):
            optimized[sid] = {
                available[j]: round(float(W_final[sid, j]), 4)
                for j in range(n_strategies)
            }

        avg_oos = float(np.mean(oos_arr))
        best_oos = float(np.max(oos_arr))
        logger.info(
            "Optimization complete: %d folds, avg OOS Sharpe=%.3f, best=%.3f",
            len(self.results), avg_oos, best_oos,
        )

        return optimized

    def summary(self) -> str:
        """Human-readable summary of optimization results."""
        if not self.results:
            return "No optimization results yet."

        lines = ["Walk-Forward Optimization Results", "=" * 40]
        for r in self.results:
            lines.append(
                f"Fold {r.fold}: IS={r.in_sample_sharpe:+.3f} "
                f"OOS={r.out_of_sample_sharpe:+.3f} "
                f"({r.test_start[:10]} to {r.test_end[:10]})"
            )

        oos = [r.out_of_sample_sharpe for r in self.results]
        lines.append(f"\nAvg OOS Sharpe: {np.mean(oos):+.3f}")
        lines.append(f"Std OOS Sharpe: {np.std(oos):.3f}")
        lines.append(f"Best OOS Sharpe: {np.max(oos):+.3f}")
        lines.append(f"Folds with positive OOS: {sum(1 for s in oos if s > 0)}/{len(oos)}")
        return "\n".join(lines)
