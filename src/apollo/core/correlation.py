"""
Cross-Pair Correlation Engine
================================
Portfolio-level risk management: detects correlated positions
and injects correlation warnings into AI prompts.

Features:
  - Rolling correlation matrix from 1h returns
  - Cluster detection (groups of highly correlated pairs)
  - Concentration risk scoring
  - Anti-correlation opportunities (hedge pairs)
  - Prompt block generation for AI context

Data: Pure Binance public data (futures klines), no API keys needed.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("core.correlation")

# Correlation thresholds
HIGH_CORR_THRESHOLD = 0.75      # Warn if opening two positions with corr > this
CLUSTER_THRESHOLD = 0.70        # Group pairs with corr > this into a cluster
HEDGE_THRESHOLD = -0.40         # Strong negative correlation = potential hedge
LOOKBACK_HOURS = 168            # 7 days of 1h bars


class CorrelationEngine:
    """
    Computes rolling correlations between pairs and provides
    portfolio-level risk insights to the AI.

    Usage:
        engine = CorrelationEngine()
        matrix = engine.compute(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        prompt_block = engine.to_prompt_block(matrix, open_positions)
    """

    MAX_CACHE_SIZE = 100  # Max symbols to keep in cache before eviction

    def __init__(self, lookback_hours: int = LOOKBACK_HOURS):
        self.lookback_hours = lookback_hours
        self._cache: dict[str, pd.Series] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=30)

    def _is_cache_valid(self) -> bool:
        if self._cache_time is None:
            return False
        return (datetime.now(timezone.utc) - self._cache_time) < self._cache_ttl

    def _fetch_returns(self, symbol: str) -> Optional[pd.Series]:
        """Fetch 1h klines and compute log returns."""
        try:
            from apollo.data.client import BinanceClient
            client = BinanceClient()
            end = datetime.now(timezone.utc)
            start = end - timedelta(hours=self.lookback_hours)

            df = client.get_futures_klines(
                symbol, "1h",
                start.strftime("%Y-%m-%d %H:%M:%S"),
                end.strftime("%Y-%m-%d %H:%M:%S"),
            )
            if df is None or df.empty or len(df) < 24:
                return None

            # Log returns for better statistical properties
            returns = np.log(df["close"].astype(float) / df["close"].astype(float).shift(1))
            returns = returns.dropna()
            return returns
        except Exception as e:
            logger.debug("Failed to fetch returns for %s: %s", symbol, e)
            return None

    def compute(self, symbols: list[str]) -> dict:
        """
        Compute correlation matrix for given symbols.

        Returns dict with:
          - matrix: correlation DataFrame
          - clusters: list of correlated groups
          - hedges: list of negatively correlated pairs
          - concentration: risk score
        """
        # Use cache if valid
        if self._is_cache_valid() and all(s in self._cache for s in symbols):
            returns_dict = {s: self._cache[s] for s in symbols if s in self._cache}
        else:
            # Fetch in parallel
            returns_dict = {}
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(self._fetch_returns, sym): sym
                    for sym in symbols
                }
                for future in as_completed(futures):
                    sym = futures[future]
                    result = future.result()
                    if result is not None and len(result) > 24:
                        returns_dict[sym] = result

            # Update cache (with size limit)
            self._cache = returns_dict
            if len(self._cache) > self.MAX_CACHE_SIZE:
                # Evict oldest entries by keeping only requested symbols
                self._cache = {s: self._cache[s] for s in symbols if s in self._cache}
            self._cache_time = datetime.now(timezone.utc)

        if len(returns_dict) < 2:
            return {"matrix": pd.DataFrame(), "clusters": [], "hedges": [],
                    "concentration": 0.0}

        # Build returns DataFrame — align on common index
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()

        if len(returns_df) < 24:
            return {"matrix": pd.DataFrame(), "clusters": [], "hedges": [],
                    "concentration": 0.0}

        # Correlation matrix
        corr_matrix = returns_df.corr()

        # Detect clusters
        clusters = self._find_clusters(corr_matrix)

        # Detect hedges
        hedges = self._find_hedges(corr_matrix)

        # Concentration risk score
        concentration = self._concentration_score(corr_matrix)

        logger.info(
            "Correlation: %d pairs, %d clusters, %d hedges, concentration=%.2f",
            len(symbols), len(clusters), len(hedges), concentration,
        )

        return {
            "matrix": corr_matrix,
            "clusters": clusters,
            "hedges": hedges,
            "concentration": concentration,
        }

    def _find_clusters(self, corr: pd.DataFrame) -> list[dict]:
        """Find groups of highly correlated pairs."""
        visited = set()
        clusters = []

        for sym in corr.columns:
            if sym in visited:
                continue
            # Find all symbols correlated > threshold with this one
            correlated = [
                s for s in corr.columns
                if s != sym and abs(corr.loc[sym, s]) >= CLUSTER_THRESHOLD
            ]
            if correlated:
                group = [sym] + correlated
                visited.update(group)
                avg_corr = np.mean([
                    abs(corr.loc[a, b])
                    for i, a in enumerate(group)
                    for b in group[i+1:]
                ])
                clusters.append({
                    "members": group,
                    "avg_correlation": round(float(avg_corr), 3),
                    "size": len(group),
                })

        return clusters

    def _find_hedges(self, corr: pd.DataFrame) -> list[dict]:
        """Find negatively correlated pairs (potential hedges)."""
        hedges = []
        seen = set()

        for sym_a in corr.columns:
            for sym_b in corr.columns:
                if sym_a >= sym_b:
                    continue
                pair_key = (sym_a, sym_b)
                if pair_key in seen:
                    continue
                c = corr.loc[sym_a, sym_b]
                if c <= HEDGE_THRESHOLD:
                    seen.add(pair_key)
                    hedges.append({
                        "pair": [sym_a, sym_b],
                        "correlation": round(float(c), 3),
                    })

        return sorted(hedges, key=lambda h: h["correlation"])

    def _concentration_score(self, corr: pd.DataFrame) -> float:
        """
        Portfolio concentration risk score (0-1).
        Higher = more correlated portfolio (more dangerous).
        Based on average absolute pairwise correlation.
        """
        n = len(corr.columns)
        if n < 2:
            return 0.0

        # Average of upper triangle (exclude diagonal)
        upper = []
        for i in range(n):
            for j in range(i+1, n):
                upper.append(abs(corr.iloc[i, j]))

        return round(float(np.mean(upper)), 3)

    def check_position_conflict(
        self, symbol: str, direction: str,
        open_positions: list[dict],
        corr_data: dict = None,
    ) -> dict:
        """
        Check if opening a position conflicts with existing ones.

        Returns:
          - conflicts: list of conflicting positions
          - risk_level: "OK" / "WARNING" / "BLOCK"
          - reason: explanation
        """
        if not open_positions or not corr_data:
            return {"conflicts": [], "risk_level": "OK", "reason": ""}

        matrix = corr_data.get("matrix", pd.DataFrame())
        if matrix.empty or symbol not in matrix.columns:
            return {"conflicts": [], "risk_level": "OK", "reason": ""}

        conflicts = []
        for pos in open_positions:
            pos_sym = pos.get("symbol", "")
            pos_dir = pos.get("direction", "")
            if pos_sym not in matrix.columns:
                continue

            corr_val = matrix.loc[symbol, pos_sym]

            # Same direction + high positive correlation = dangerous concentration
            if direction == pos_dir and corr_val >= HIGH_CORR_THRESHOLD:
                conflicts.append({
                    "symbol": pos_sym,
                    "direction": pos_dir,
                    "correlation": round(float(corr_val), 3),
                    "type": "SAME_DIRECTION_CORRELATED",
                })

            # Opposite direction + high negative correlation = also dangerous
            if direction != pos_dir and corr_val <= -HIGH_CORR_THRESHOLD:
                conflicts.append({
                    "symbol": pos_sym,
                    "direction": pos_dir,
                    "correlation": round(float(corr_val), 3),
                    "type": "OPPOSITE_DIRECTION_ANTI_CORRELATED",
                })

        if not conflicts:
            return {"conflicts": [], "risk_level": "OK", "reason": ""}

        risk = "WARNING" if len(conflicts) == 1 else "BLOCK"
        reasons = [
            f"{c['symbol']}({c['direction']}) corr={c['correlation']}"
            for c in conflicts
        ]
        return {
            "conflicts": conflicts,
            "risk_level": risk,
            "reason": f"Correlated positions: {', '.join(reasons)}",
        }

    def to_prompt_block(
        self,
        corr_data: dict,
        open_positions: list[dict] = None,
    ) -> str:
        """Generate a prompt block for the AI about correlations."""
        if not corr_data or corr_data.get("matrix", pd.DataFrame()).empty:
            return ""

        lines = ["=== CROSS-PAIR CORRELATION ==="]
        lines.append(
            f"Concentration Risk: {corr_data['concentration']:.2f} "
            f"({'HIGH' if corr_data['concentration'] > 0.6 else 'MODERATE' if corr_data['concentration'] > 0.4 else 'LOW'})"
        )

        # Clusters
        clusters = corr_data.get("clusters", [])
        if clusters:
            for cl in clusters[:3]:
                members = ", ".join(cl["members"][:5])
                lines.append(
                    f"Cluster: [{members}] avg_corr={cl['avg_correlation']:.2f}"
                )

        # Hedges
        hedges = corr_data.get("hedges", [])
        if hedges:
            for h in hedges[:3]:
                lines.append(
                    f"Hedge pair: {h['pair'][0]} / {h['pair'][1]} "
                    f"corr={h['correlation']:.2f}"
                )

        # Position conflicts
        if open_positions:
            lines.append("Position conflicts:")
            matrix = corr_data.get("matrix", pd.DataFrame())
            for pos in open_positions:
                sym = pos.get("symbol", "")
                d = pos.get("direction", "")
                if sym in matrix.columns:
                    # Find highest correlation with other open positions
                    for other_pos in open_positions:
                        other_sym = other_pos.get("symbol", "")
                        if other_sym != sym and other_sym in matrix.columns:
                            c = matrix.loc[sym, other_sym]
                            if abs(c) >= HIGH_CORR_THRESHOLD:
                                lines.append(
                                    f"  ⚠ {sym}({d}) ↔ {other_sym}({other_pos.get('direction','')}) "
                                    f"corr={c:.2f}"
                                )
            if len(lines) == len(["=== CROSS-PAIR CORRELATION ==="]) + 1 + len(clusters) + len(hedges) + 1:
                lines.append("  None detected")

        return "\n".join(lines)
