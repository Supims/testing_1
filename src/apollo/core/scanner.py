"""
Scanner -- Pipeline Facade
============================
Orchestrates the full analysis pipeline:
  DataProvider -> FeaturePipeline -> HMM -> Strategies -> Ensemble ->
  MonteCarlo -> Scorecard -> Enrichment -> Probability -> Risk -> PairAnalysis

Supports multi-timeframe (1h decision, 15m/5m confirmation).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("core.scanner")


# -- MTF feature list --------------------------------------------------------

MTF_15M_FEATURES = ["rsi_14", "ema_slope_20", "volume_zscore"]
MTF_5M_FEATURES = ["rsi_14", "mom_10"]


class Scanner:
    """
    Production pipeline. Analyze any pair or batch of pairs in one call.

    Usage:
        scanner = Scanner()
        scanner.train(["BTCUSDT", "ETHUSDT"], "2025-10-01", "2026-02-01")
        results = scanner.scan(["BTCUSDT", "ETHUSDT"])
    """

    def __init__(self, interval: str = "1h", mc_horizon: int = 24,
                 mc_scenarios: int = 500, enable_mtf: bool = True):
        self.interval = interval
        self.mc_horizon = mc_horizon
        self.mc_scenarios = mc_scenarios
        self.enable_mtf = enable_mtf

        # Components (lazy-loaded)
        self._hmm = None
        self._mc = None
        self._probability_model = None
        self._risk_dashboard = None
        self._feature_pipeline = None
        self._state_labels = {}
        self._is_trained = False

    # -----------------------------------------------------------------------
    # Data helpers
    # -----------------------------------------------------------------------

    def _get_feature_df(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch raw data and compute features. Correct pipeline."""
        from apollo.data.provider import MarketDataProvider
        from apollo.features.pipeline import FeaturePipeline

        provider = MarketDataProvider()
        if self._feature_pipeline is None:
            self._feature_pipeline = FeaturePipeline()

        # Step 1: Enriched raw data (OHLCV + funding + OI + spot)
        raw_df, status = provider.get_enriched_dataset(symbol, self.interval, start, end)
        if raw_df.empty:
            return pd.DataFrame()

        # Step 2: Compute features (technical + microstructure + derived)
        feature_df, _meta = self._feature_pipeline.compute(raw_df)
        return feature_df

    def _get_mtf_features(self, symbol: str, start: str, end: str,
                          main_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fetch 15m & 5m data, compute subset of features, resample to 1h,
        and merge into main DataFrame as mtf_15m_* / mtf_5m_* columns.

        Uses futures-only klines (no spot/premium/OI) to avoid errors
        on tokens that don't have spot pairs.
        """
        from apollo.data.provider import MarketDataProvider
        from apollo.features.pipeline import FeaturePipeline

        provider = MarketDataProvider()
        pipe = FeaturePipeline()
        result_df = main_df.copy()

        # -- 15m timeframe --
        try:
            raw_15m, _ = provider.get_klines(symbol, "15m", start, end, source="futures")
            if raw_15m is not None and not raw_15m.empty and len(raw_15m) >= 50:
                feat_15m, _ = pipe.compute(raw_15m)
                for col in MTF_15M_FEATURES:
                    if col in feat_15m.columns:
                        # Resample to 1h: take last value per hour
                        resampled = feat_15m[col].resample("1h").last()
                        result_df[f"mtf_15m_{col}"] = resampled.reindex(
                            result_df.index
                        ).ffill()
        except Exception as e:
            logger.debug("15m MTF data not available for %s: %s", symbol, e)

        # -- 5m timeframe --
        try:
            raw_5m, _ = provider.get_klines(symbol, "5m", start, end, source="futures")
            if raw_5m is not None and not raw_5m.empty and len(raw_5m) >= 50:
                feat_5m, _ = pipe.compute(raw_5m)
                for col in MTF_5M_FEATURES:
                    if col in feat_5m.columns:
                        resampled = feat_5m[col].resample("1h").last()
                        result_df[f"mtf_5m_{col}"] = resampled.reindex(
                            result_df.index
                        ).ffill()
        except Exception as e:
            logger.debug("5m MTF data not available for %s: %s", symbol, e)

        # Count how many MTF columns were added
        mtf_cols = [c for c in result_df.columns if c.startswith("mtf_")]
        if mtf_cols:
            logger.info("MTF: added %d columns for %s: %s", len(mtf_cols), symbol,
                        ", ".join(mtf_cols))
        return result_df

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def train(self, symbols: list[str], start: str, end: str):
        """
        Train HMM + Monte Carlo + probability models on historical data.

        Args:
            symbols: pairs to train on (typically just BTCUSDT)
            start, end: training window
        """
        from apollo.models.regime import RegimeDetector
        from apollo.models.strategies import compute_all
        from apollo.models.ensemble import StaticEnsemble
        from apollo.models.monte_carlo import MonteCarloSimulator
        from apollo.execution.probability import MultiHorizonModel
        from apollo.execution.risk import RiskDashboard

        logger.info("Training pipeline on %s (%s to %s)", symbols, start, end)

        self._risk_dashboard = RiskDashboard()

        # Get features for first symbol (regime training)
        symbol = symbols[0]
        df = self._get_feature_df(symbol, start, end)
        if df.empty or len(df) < 200:
            raise ValueError(f"Insufficient data for training: {len(df)} rows")

        # HMM: fit then predict
        self._hmm = RegimeDetector()
        self._hmm.fit(df)
        regime_df = self._hmm.predict(df)

        # Extract regime info for downstream
        regime_series = regime_df["hmm_regime"].fillna(0).astype(int) if "hmm_regime" in regime_df.columns else pd.Series(0, index=df.index)
        self._state_labels = self._hmm.label_map
        logger.info("HMM trained: %s", self._state_labels)

        # Strategies (no regime_info needed -- they just read feature columns)
        signals = compute_all(df)

        # Ensemble: needs label_map and hmm_probs
        ensemble = StaticEnsemble(self._state_labels)
        ensemble_signal = ensemble.compute(signals, regime_df)

        # Monte Carlo: fit GARCH on returns per regime
        returns = df["close"].pct_change().dropna()
        self._mc = MonteCarloSimulator()
        self._mc.fit(returns, regime_series.reindex(returns.index).fillna(0).astype(int))

        # Probability model: train XGBoost
        self._probability_model = MultiHorizonModel()
        targets = self._probability_model.build_targets(df)
        X = self._probability_model.prepare_features(df, ensemble_signal)
        self._probability_model.train(X, targets)

        self._is_trained = True
        logger.info("Training complete")

    # -----------------------------------------------------------------------
    # Single-pair analysis
    # -----------------------------------------------------------------------

    def analyze_single(self, symbol: str, start: str, end: str) -> dict:
        """Analyze a single pair. Returns structured dict."""
        from apollo.models.strategies import compute_all
        from apollo.models.ensemble import StaticEnsemble
        from apollo.models.scorecard import StrategyScorecard
        from apollo.models.enrichment import SignalEnrichment
        from apollo.execution.risk import RiskDashboard

        # 1h features (main decision timeframe)
        df = self._get_feature_df(symbol, start, end)
        if df.empty or len(df) < 100:
            logger.warning("Insufficient data for %s (%d rows)", symbol, len(df) if not df.empty else 0)
            return {"symbol": symbol, "error": "insufficient_data"}

        # Multi-timeframe enrichment
        if self.enable_mtf:
            try:
                df = self._get_mtf_features(symbol, start, end, df)
            except Exception as e:
                logger.warning("MTF enrichment failed for %s: %s", symbol, e)

        # HMM regime
        regime_df = pd.DataFrame(index=df.index)
        regime_id = 0
        regime_info_output = {}

        if self._hmm is not None and self._hmm.is_fitted:
            try:
                regime_df = self._hmm.predict(df)
                regime_series = regime_df["hmm_regime"].fillna(0).astype(int) if "hmm_regime" in regime_df.columns else pd.Series(0, index=df.index)
                regime_id = int(regime_series.iloc[-1])
                current_label = regime_df["hmm_regime_label"].iloc[-1] if "hmm_regime_label" in regime_df.columns else "Unknown"
                is_ood = bool(regime_df["hmm_ood"].iloc[-1]) if "hmm_ood" in regime_df.columns else False

                regime_info_output = {
                    "label": current_label,
                    "state_id": regime_id,
                    "is_ood": is_ood,
                }
            except Exception as e:
                logger.warning("HMM predict failed for %s: %s", symbol, e)

        # Provide a default regime when HMM is not available
        if not regime_info_output:
            regime_info_output = {
                "label": "Unknown (no HMM)",
                "state_id": 0,
                "is_ood": False,
            }

        # Strategies
        signals = compute_all(df)

        # Ensemble
        ensemble = StaticEnsemble(self._state_labels or {0: "Unknown"})
        ensemble_signal = ensemble.compute(signals, regime_df)

        # Monte Carlo
        close_arr = df["close"].values
        try:
            if self._mc is not None and self._mc.is_fitted:
                paths = self._mc.simulate(close_arr[-1], regime_id)
            else:
                paths = np.random.randn(self.mc_scenarios, self.mc_horizon) * 0.01 * close_arr[-1] + close_arr[-1]
        except Exception as e:
            logger.warning("MC failed for %s: %s", symbol, e)
            paths = np.random.randn(self.mc_scenarios, self.mc_horizon) * 0.01 * close_arr[-1] + close_arr[-1]

        # Risk
        risk_dashboard = self._risk_dashboard or RiskDashboard()
        ens_val = float(ensemble_signal.iloc[-1]) if len(ensemble_signal) > 0 else 0.0
        risk_profile = risk_dashboard.compute_profile(paths, close_arr[-1], ens_val)

        # Scorecard
        try:
            scorecard = StrategyScorecard()
            sc_result = scorecard.compute(signals, df["close"])
            if isinstance(sc_result, pd.DataFrame):
                # Take last row as summary
                sc_summary = sc_result.iloc[-1].to_dict() if len(sc_result) > 0 else {}
            else:
                sc_summary = sc_result if isinstance(sc_result, dict) else {}
        except Exception:
            sc_summary = {}

        # Enrichment
        try:
            enrichment = SignalEnrichment()
            en_result = enrichment.compute(signals)
            if isinstance(en_result, pd.DataFrame):
                en_summary = en_result.iloc[-1].to_dict() if len(en_result) > 0 else {}
            else:
                en_summary = en_result if isinstance(en_result, dict) else {}
        except Exception:
            en_summary = {}

        # Probabilities
        probs = {}
        if self._probability_model is not None:
            try:
                X = self._probability_model.prepare_features(df, ensemble_signal)
                prob_df = self._probability_model.predict(X)
                if not prob_df.empty:
                    probs = prob_df.iloc[-1].to_dict()
            except Exception as e:
                logger.warning("Probability prediction failed for %s: %s", symbol, e)

        # Build latest signal values
        latest_signals = {}
        for name, series in signals.items():
            if isinstance(series, pd.Series) and len(series) > 0:
                latest_signals[name] = float(series.iloc[-1])
            elif isinstance(series, (int, float)):
                latest_signals[name] = float(series)

        if len(ensemble_signal) > 0:
            latest_signals["ensemble"] = float(ensemble_signal.iloc[-1])

        # MTF info
        mtf_info = {}
        mtf_cols = [c for c in df.columns if c.startswith("mtf_")]
        for col in mtf_cols:
            last_val = df[col].iloc[-1]
            if pd.notna(last_val):
                mtf_info[col] = float(last_val)

        return {
            "symbol": symbol,
            "current_price": float(close_arr[-1]),
            "regime": regime_info_output,
            "signals": latest_signals,
            "probabilities": probs,
            "risk": risk_profile,
            "scorecard": sc_summary,
            "enrichment": en_summary,
            "mtf": mtf_info,
        }

    # -----------------------------------------------------------------------
    # Multi-pair scan
    # -----------------------------------------------------------------------

    def scan(self, symbols: list[str] = None, start: str = None,
             end: str = None) -> dict:
        """
        Full multi-pair scan.

        Returns:
            {
                "scan_id": str,
                "timestamp": str,
                "results": [per-pair dicts],
                "scorecard_summary": dict,
                "enrichment_summary": dict,
            }
        """
        from apollo.data.discovery import PairDiscovery

        scan_id = str(uuid.uuid4())[:12]

        if symbols is None:
            discovery = PairDiscovery()
            symbols = discovery.discover(max_pairs=20)

        if end is None:
            end = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        if start is None:
            start = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")

        logger.info("Scanning %d pairs (%s to %s)...", len(symbols), start[:10], end[:10])

        # Fetch on-chain intelligence in parallel (free APIs, no keys)
        onchain_map = {}
        try:
            from apollo.data.onchain import OnChainIntelligence
            intel = OnChainIntelligence()
            profiles = intel.fetch_batch(symbols, max_symbols=10)
            onchain_map = {p.symbol: p for p in profiles}
        except Exception as e:
            logger.warning("On-chain fetch failed: %s", e)

        results = []
        for symbol in symbols:
            try:
                result = self.analyze_single(symbol, start, end)
                # Attach on-chain data
                oc_profile = onchain_map.get(symbol)
                if oc_profile:
                    result["onchain"] = oc_profile.to_dict()
                    result["onchain_prompt"] = oc_profile.to_prompt_block()
                results.append(result)
            except Exception as e:
                logger.error("Scan failed for %s: %s", symbol, e)
                results.append({"symbol": symbol, "error": str(e)})

        # Aggregate scorecard and enrichment from first successful result
        scorecard_summary = {}
        enrichment_summary = {}
        for r in results:
            if "scorecard" in r and r["scorecard"]:
                scorecard_summary = r["scorecard"]
                break
        for r in results:
            if "enrichment" in r and r["enrichment"]:
                enrichment_summary = r["enrichment"]
                break

        # Global on-chain context
        onchain_global = ""
        for p in onchain_map.values():
            if p.global_market or p.stablecoins:
                onchain_global = p.to_prompt_block()
                break

        # Cross-pair correlation analysis
        correlation_data = {}
        correlation_prompt = ""
        try:
            from apollo.core.correlation import CorrelationEngine
            corr_engine = CorrelationEngine()
            valid_symbols = [r["symbol"] for r in results if "error" not in r]
            if len(valid_symbols) >= 3:
                correlation_data = corr_engine.compute(valid_symbols)
                correlation_prompt = corr_engine.to_prompt_block(correlation_data)
        except Exception as e:
            logger.debug("Correlation analysis failed: %s", e)

        # Macro events calendar
        events_prompt = ""
        events_risk_multiplier = 1.0
        try:
            from apollo.core.events import MacroEventCalendar
            calendar = MacroEventCalendar()
            events_prompt = calendar.to_prompt_block()
            events_risk_multiplier = calendar.get_risk_multiplier()
        except Exception as e:
            logger.debug("Events calendar failed: %s", e)

        n_ok = sum(1 for r in results if "error" not in r)
        logger.info("Scan complete: %d/%d pairs analyzed", n_ok, len(symbols))

        return {
            "scan_id": scan_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_pairs": len(results),
            "results": results,
            "scorecard_summary": scorecard_summary,
            "enrichment_summary": enrichment_summary,
            "onchain_global": onchain_global,
            "correlation_prompt": correlation_prompt,
            "correlation_data": correlation_data,
            "events_prompt": events_prompt,
            "events_risk_multiplier": events_risk_multiplier,
        }

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save_models(self, path: str):
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        if self._hmm is not None:
            self._hmm.save(save_path / "hmm.joblib")
        if self._mc is not None:
            self._mc.save(save_path / "monte_carlo.joblib")
        if self._probability_model is not None:
            self._probability_model.save(save_path / "probability.joblib")

        import json
        meta = {
            "state_labels": {str(k): v for k, v in self._state_labels.items()},
            "interval": self.interval,
            "enable_mtf": self.enable_mtf,
        }
        (save_path / "scanner_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        logger.info("Models saved to %s", save_path)

    @classmethod
    def load(cls, path: str) -> Scanner:
        load_path = Path(path)
        import json

        meta = json.loads((load_path / "scanner_meta.json").read_text(encoding="utf-8"))
        scanner = cls(
            interval=meta.get("interval", "1h"),
            enable_mtf=meta.get("enable_mtf", True),
        )
        scanner._state_labels = {int(k): v for k, v in meta.get("state_labels", {}).items()}

        from apollo.models.regime import RegimeDetector
        from apollo.models.monte_carlo import MonteCarloSimulator
        from apollo.execution.probability import MultiHorizonModel

        hmm_path = load_path / "hmm.joblib"
        if hmm_path.exists():
            scanner._hmm = RegimeDetector.load(hmm_path)

        mc_path = load_path / "monte_carlo.joblib"
        if mc_path.exists():
            scanner._mc = MonteCarloSimulator.load(mc_path)

        prob_path = load_path / "probability.joblib"
        if prob_path.exists():
            scanner._probability_model = MultiHorizonModel.load(prob_path)

        scanner._is_trained = True
        logger.info("Scanner loaded from %s", load_path)
        return scanner
