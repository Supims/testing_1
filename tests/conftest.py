"""
Shared test fixtures.
Provides reusable synthetic data, temp directories, and sample contracts.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone
from pathlib import Path


# -- Force test config (no real .env) ------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Ensure tests never read real .env or write to real dirs."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-12345")
    monkeypatch.setenv("APOLLO_AI_PROVIDER", "google")
    monkeypatch.setenv("APOLLO_DAILY_BUDGET", "50.0")
    monkeypatch.setenv("APOLLO_WEEKLY_BUDGET", "200.0")
    monkeypatch.setenv("APOLLO_DEFAULT_TIER", "2")


# -- Synthetic OHLCV data -----------------------------------------------------

@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """
    500 bars of realistic 1h OHLCV data.
    Contains a trend, a ranging period, and a sharp move.
    """
    np.random.seed(42)
    n = 500
    timestamps = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")

    returns = np.random.normal(0.0001, 0.005, n)
    returns[100:200] += 0.002       # trend
    returns[350] = -0.08            # sharp drop
    returns[400:450] = np.random.normal(0.0, 0.002, 50)  # mean reversion

    price = 50000 * np.exp(np.cumsum(returns))
    high = price * (1 + np.abs(np.random.normal(0, 0.003, n)))
    low = price * (1 - np.abs(np.random.normal(0, 0.003, n)))
    volume = np.random.lognormal(10, 1, n)

    df = pd.DataFrame({
        "open": np.roll(price, 1),
        "high": high,
        "low": low,
        "close": price,
        "volume": volume,
    }, index=timestamps)
    df.iloc[0, 0] = df.iloc[0, 3]
    return df


@pytest.fixture
def synthetic_ohlcv_with_micro(synthetic_ohlcv) -> pd.DataFrame:
    """OHLCV + microstructure columns (funding, OI, taker volume, etc.)."""
    n = len(synthetic_ohlcv)
    np.random.seed(42)

    df = synthetic_ohlcv.copy()
    df["taker_buy_volume"] = df["volume"] * np.random.uniform(0.3, 0.7, n)
    df["taker_sell_volume"] = df["volume"] - df["taker_buy_volume"]
    df["funding_rate"] = np.random.normal(0.0001, 0.0003, n)
    df["open_interest"] = np.random.lognormal(12, 0.5, n)
    df["premium_index"] = np.random.normal(0.0, 0.001, n)
    df["spot_close"] = df["close"] * (1 + np.random.normal(0, 0.0005, n))
    return df


# -- Sample domain objects -----------------------------------------------------

@pytest.fixture
def sample_regime():
    from apollo.types import RegimeInfo
    return RegimeInfo(
        label="High Volatility (Trending)",
        state_id=2,
        probabilities={0: 0.05, 1: 0.10, 2: 0.80, 3: 0.05},
        is_ood=False,
        ood_score=1.2,
    )


@pytest.fixture
def sample_probabilities():
    from apollo.types import ProbabilityProfile
    return ProbabilityProfile(
        prob_up_1p5_12h=0.45,
        prob_up_1p5_24h=0.52,
        prob_up_3p0_48h=0.38,
        prob_dd_1p0_24h=0.25,
        prob_dd_2p0_24h=0.12,
    )


@pytest.fixture
def sample_risk():
    from apollo.types import RiskProfile
    return RiskProfile(
        direction="LONG",
        expected_return_pct=1.5,
        median_return_pct=1.2,
        return_std_pct=3.0,
        prob_profit_pct=62.0,
        var_5pct=-4.2,
        cvar_5pct=-6.1,
        max_drawdown_pct=-8.5,
        sl_price=48500.0,
        tp_price=52500.0,
        sl_distance_pct=3.0,
        tp_distance_pct=5.0,
        payoff_ratio=1.67,
        kelly_fraction_pct=3.2,
    )


@pytest.fixture
def sample_pair_analysis(sample_regime, sample_probabilities, sample_risk):
    from apollo.types import PairAnalysis, StrategyOutput
    return PairAnalysis(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc),
        regime=sample_regime,
        strategies=StrategyOutput(
            signals={
                "Trend": 0.65,
                "MR": -0.1,
                "Squeeze": 0.3,
                "SmartMoney": 0.0,
                "Basis": 0.0,
                "Breakout": 0.45,
                "FundingMom": 0.0,
                "OIDivergence": 0.15,
                "LiqCascade": 0.0,
                "VolProfile": 0.2,
            },
            ensemble_signal=0.42,
        ),
        probabilities=sample_probabilities,
        risk=sample_risk,
        opportunity_score=0.42,
        risk_score=-0.15,
    )


# -- Temp dirs -----------------------------------------------------------------

@pytest.fixture
def tmp_models_dir(tmp_path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    return d


@pytest.fixture
def tmp_data_dir(tmp_path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    return d


# -- Pipeline-computed features for integration tests --------------------------

@pytest.fixture
def synthetic_features(synthetic_ohlcv_with_micro) -> pd.DataFrame:
    """Run the FeaturePipeline on micro data to get a ready-to-use feature df."""
    from apollo.features.pipeline import FeaturePipeline
    pipe = FeaturePipeline(warmup_bars=30)
    result, _ = pipe.compute(synthetic_ohlcv_with_micro)
    # Merge back OHLCV for strategies that need close/volume
    for col in ["open", "high", "low", "close", "volume"]:
        result[col] = synthetic_ohlcv_with_micro[col].loc[result.index]
    return result
