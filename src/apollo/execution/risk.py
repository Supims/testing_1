"""
Risk Dashboard + Dynamic Position Sizing
==========================================
Computes Monte Carlo-derived risk metrics and suggests position sizing.
Informational only -- does NOT make Go/No-Go decisions.

Outputs:
  - VaR / CVaR (5th percentile)
  - Direction-adjusted returns (net of fees)
  - Dynamic SL/TP from MC percentiles
  - Payoff ratio, Kelly fraction
  - Dynamic position sizing (regime + confidence + drawdown scaling)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("execution.risk")


@dataclass
class RiskConfig:
    """Risk parameters."""
    fee_rate: float = 0.0004         # 0.04% taker per side
    slippage: float = 0.0002         # 0.02% assumed per side
    max_risk_per_trade: float = 0.02  # 2% max capital at risk
    reference_capital: float = 10000.0


# Regime-based scaling for position sizing
REGIME_SIZE_SCALE = {
    "High Volatility (Trending)": 0.8,
    "Low Volatility (Trending)": 1.0,
    "High Volatility (Ranging)": 0.5,
    "Low Volatility (Quiet Range)": 0.6,
}

CONFIDENCE_SCALE = {
    "HIGH": 1.0,
    "MEDIUM": 0.6,
    "LOW": 0.3,
}


class RiskDashboard:
    """
    Monte Carlo-powered risk profiler with dynamic sizing.
    """

    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self._total_fees = (self.config.fee_rate + self.config.slippage) * 2

    def compute_profile(
        self,
        price_paths: np.ndarray,
        current_price: float,
        ensemble_signal: float = 0.0,
    ) -> dict:
        """
        Compute risk profile from MC paths.

        Args:
            price_paths: (n_scenarios, horizon) from MonteCarloSimulator.
            current_price: Current asset price.
            ensemble_signal: Ensemble conviction for direction.

        Returns:
            dict with all risk metrics.
        """
        direction = 1 if ensemble_signal > 0 else -1
        direction_str = "LONG" if direction == 1 else "SHORT"

        final_prices = price_paths[:, -1]

        # Direction-adjusted returns
        if direction == 1:
            returns = (final_prices - current_price) / current_price
            adverse = (np.min(price_paths, axis=1) - current_price) / current_price
            favorable = (np.max(price_paths, axis=1) - current_price) / current_price
        else:
            returns = (current_price - final_prices) / current_price
            adverse = (current_price - np.max(price_paths, axis=1)) / current_price
            favorable = (current_price - np.min(price_paths, axis=1)) / current_price

        # Net of fees
        net_returns = returns - self._total_fees
        net_adverse = adverse - self._total_fees

        # Statistics
        expected = float(np.mean(net_returns))
        median = float(np.median(net_returns))
        std = float(np.std(net_returns))
        var_5 = float(np.percentile(net_returns, 5))
        cvar_5 = float(np.mean(net_returns[net_returns <= var_5])) if np.any(net_returns <= var_5) else var_5
        prob_profit = float(np.mean(net_returns > 0))

        # Dynamic SL/TP
        sl_buffer = 0.001
        sl_distance = abs(float(np.percentile(net_adverse, 5))) + sl_buffer
        tp_distance = max(float(np.percentile(favorable, 75)), self._total_fees * 3)

        if direction == 1:
            sl_price = current_price * (1 - sl_distance)
            tp_price = current_price * (1 + tp_distance)
        else:
            sl_price = current_price * (1 + sl_distance)
            tp_price = current_price * (1 - tp_distance)

        payoff = tp_distance / sl_distance if sl_distance > 0 else 1.0

        # Kelly fraction
        kelly = 0.0
        if payoff > 0:
            kelly = (prob_profit * payoff - (1 - prob_profit)) / payoff
        risk_frac = max(0, min(kelly, self.config.max_risk_per_trade))

        # Position sizing
        capital = self.config.reference_capital
        risk_amount = capital * risk_frac
        suggested_size = risk_amount / sl_distance if sl_distance > 0 else 0
        suggested_size = min(suggested_size, capital * 0.5)

        return {
            "direction": direction_str,
            "signal_strength": float(ensemble_signal),
            "current_price": float(current_price),
            "n_scenarios": int(price_paths.shape[0]),
            "horizon_steps": int(price_paths.shape[1]),
            "expected_return_pct": expected * 100,
            "median_return_pct": median * 100,
            "return_std_pct": std * 100,
            "prob_profit_pct": prob_profit * 100,
            "var_5pct": var_5 * 100,
            "cvar_5pct": cvar_5 * 100,
            "sl_price": float(sl_price),
            "tp_price": float(tp_price),
            "sl_distance_pct": sl_distance * 100,
            "tp_distance_pct": tp_distance * 100,
            "payoff_ratio": float(payoff),
            "kelly_fraction_pct": risk_frac * 100,
            "suggested_size_usd": float(suggested_size),
            "risk_amount_usd": float(risk_amount),
        }

    @staticmethod
    def dynamic_size(
        kelly: float,
        regime_label: str,
        confidence: str,
        current_drawdown_pct: float,
        capital: float,
    ) -> float:
        """
        Compute regime/confidence/drawdown-scaled position size.

        Args:
            kelly: Kelly fraction (0-1).
            regime_label: Current HMM regime label string.
            confidence: "HIGH", "MEDIUM", or "LOW".
            current_drawdown_pct: Current portfolio drawdown (e.g., 5.0 = 5%).
            capital: Total available capital.

        Returns:
            Suggested position size in USD.
        """
        base = kelly * capital

        regime_scale = REGIME_SIZE_SCALE.get(regime_label, 0.7)
        conf_scale = CONFIDENCE_SCALE.get(confidence, 0.5)

        # Progressive drawdown reduction
        if current_drawdown_pct > 15:
            dd_scale = 0.0
        elif current_drawdown_pct > 10:
            dd_scale = 0.25
        elif current_drawdown_pct > 5:
            dd_scale = 0.5
        else:
            dd_scale = 1.0

        size = base * regime_scale * conf_scale * dd_scale
        return min(size, capital * 0.05)  # Hard cap: 5% of capital

    @staticmethod
    def format_profile(profile: dict) -> str:
        """ASCII-formatted risk profile display."""
        d = profile
        lines = [
            "-" * 50,
            f"  {d['direction']} @ ${d['current_price']:,.2f} "
            f"(Signal: {d['signal_strength']:+.3f})",
            f"  {d['n_scenarios']} scenarios x {d['horizon_steps']} steps",
            "-" * 50,
            f"  E[return]:  {d['expected_return_pct']:+.2f}% "
            f"(median: {d['median_return_pct']:+.2f}%)",
            f"  P(profit):  {d['prob_profit_pct']:.1f}% "
            f"(sigma: {d['return_std_pct']:.2f}%)",
            f"  VaR 5%:     {d['var_5pct']:+.2f}% "
            f"| CVaR 5%: {d['cvar_5pct']:+.2f}%",
            "-" * 50,
            f"  SL: ${d['sl_price']:,.2f} ({d['sl_distance_pct']:.2f}%)",
            f"  TP: ${d['tp_price']:,.2f} ({d['tp_distance_pct']:.2f}%)",
            f"  Payoff: {d['payoff_ratio']:.2f}:1",
            f"  Kelly: {d['kelly_fraction_pct']:.2f}% "
            f"-> ${d['suggested_size_usd']:,.0f}",
            "-" * 50,
        ]
        return "\n".join(lines)
