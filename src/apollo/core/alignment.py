"""
Pair Alignment
===============
Cross-pair correlation, clustering, and signal contradiction detection.
Used to validate that a trade on one pair doesn't contradict its cluster.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("core.alignment")


class PairAlignment:
    """Cross-pair analysis for signal validation."""

    def correlation_matrix(
        self,
        returns_dict: dict[str, pd.Series],
        window: int = 168,  # 7 days of hourly data
    ) -> pd.DataFrame:
        """
        Build a rolling correlation matrix from log returns.

        Args:
            returns_dict: {symbol: pd.Series of log returns}
            window: rolling window for correlation
        """
        df = pd.DataFrame(returns_dict)
        if df.empty or len(df) < window:
            return pd.DataFrame()
        return df.corr()

    def cluster_pairs(
        self,
        corr_matrix: pd.DataFrame,
        n_clusters: int = 5,
    ) -> dict[int, list[str]]:
        """
        Cluster pairs by correlation similarity.
        Uses simple threshold-based clustering (no scipy dependency).
        """
        if corr_matrix.empty:
            return {}

        symbols = list(corr_matrix.columns)
        n = len(symbols)
        if n <= n_clusters:
            return {i: [s] for i, s in enumerate(symbols)}

        # Simple agglomerative: group pairs with correlation > 0.6
        assigned = set()
        clusters = {}
        cluster_id = 0

        for i, sym_i in enumerate(symbols):
            if sym_i in assigned:
                continue
            group = [sym_i]
            assigned.add(sym_i)
            for j, sym_j in enumerate(symbols):
                if sym_j in assigned:
                    continue
                if corr_matrix.loc[sym_i, sym_j] > 0.6:
                    group.append(sym_j)
                    assigned.add(sym_j)
            clusters[cluster_id] = group
            cluster_id += 1

        return clusters

    def detect_divergences(
        self,
        signals_dict: dict[str, float],
        corr_matrix: pd.DataFrame = None,
    ) -> list[dict]:
        """
        Find pairs with divergent signals despite high correlation.
        These are potential stat-arb opportunities.
        """
        if corr_matrix is None or corr_matrix.empty:
            return []

        divergences = []
        symbols = list(signals_dict.keys())

        for i, sym_a in enumerate(symbols):
            for j, sym_b in enumerate(symbols):
                if j <= i:
                    continue
                if sym_a not in corr_matrix.columns or sym_b not in corr_matrix.columns:
                    continue

                corr = corr_matrix.loc[sym_a, sym_b]
                sig_a = signals_dict[sym_a]
                sig_b = signals_dict[sym_b]

                # High correlation but divergent signals
                if corr > 0.7 and sig_a * sig_b < -0.1:
                    divergences.append({
                        "pair_a": sym_a,
                        "pair_b": sym_b,
                        "correlation": float(corr),
                        "signal_a": float(sig_a),
                        "signal_b": float(sig_b),
                        "divergence": float(abs(sig_a - sig_b)),
                    })

        divergences.sort(key=lambda x: x["divergence"], reverse=True)
        return divergences[:5]

    def cross_validate_signal(
        self,
        symbol: str,
        direction: str,
        signals_dict: dict[str, float],
        clusters: dict[int, list[str]],
    ) -> float:
        """
        Check if the signal for this symbol contradicts its cluster.

        Returns:
            0.0 = no contradiction (cluster agrees)
            1.0 = strong contradiction (cluster disagrees)
        """
        # Find which cluster this symbol belongs to
        my_cluster = []
        for _, members in clusters.items():
            if symbol in members:
                my_cluster = [m for m in members if m != symbol]
                break

        if not my_cluster:
            return 0.0

        # Check how many cluster members agree/disagree
        target_sign = 1.0 if direction == "LONG" else -1.0
        agree = 0
        disagree = 0

        for peer in my_cluster:
            peer_signal = signals_dict.get(peer, 0)
            if abs(peer_signal) < 0.05:  # Neutral, skip
                continue
            if np.sign(peer_signal) == target_sign:
                agree += 1
            else:
                disagree += 1

        total = agree + disagree
        if total == 0:
            return 0.0

        contradiction = disagree / total
        return round(contradiction, 2)

    def beta_to_btc(
        self,
        pair_returns: pd.Series,
        btc_returns: pd.Series,
    ) -> float:
        """Calculate beta of a pair relative to BTC."""
        aligned = pd.DataFrame({
            "pair": pair_returns, "btc": btc_returns,
        }).dropna()

        if len(aligned) < 30:
            return 1.0

        cov = np.cov(aligned["pair"], aligned["btc"])
        if cov[1, 1] == 0:
            return 1.0
        return float(cov[0, 1] / cov[1, 1])
