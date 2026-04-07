"""
Feature Pipeline -- Technical Indicators
=========================================
Pure stateless functions for computing technical analysis features.
Each function takes a Series/DataFrame and returns a Series/DataFrame.
No side effects, no state, fully testable in isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# =========================================================================
# Returns
# =========================================================================

def log_return(close: pd.Series) -> pd.Series:
    """Log returns of the asset. Used by HMM, GARCH, and autocorrelation."""
    return np.log(close / close.shift(1))


def autocorrelation(series: pd.Series, window: int = 20, lag: int = 1) -> pd.Series:
    """Rolling autocorrelation. Critical for HMM regime detection."""
    return series.rolling(window).apply(
        lambda x: pd.Series(x).autocorr(lag=lag), raw=False
    )


# =========================================================================
# Trend / Momentum
# =========================================================================

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def rsi_normalized(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI normalized to [-1, +1] range. (rsi - 50) / 50."""
    return (rsi(close, period) - 50) / 50.0


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": histogram,
    }, index=close.index)


def adx_full(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14) -> pd.DataFrame:
    """
    Average Directional Index with +DI and -DI.
    Returns 3 columns: adx, adx_pos, adx_neg.
    TrendSignal strategy needs all three.
    """
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr_val = tr.ewm(alpha=1 / period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr_val)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr_val)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1 / period, min_periods=period).mean()

    return pd.DataFrame({
        "adx": adx_val,
        "adx_pos": plus_di,
        "adx_neg": minus_di,
    }, index=close.index)


def cci(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 14) -> pd.Series:
    """Williams %R."""
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Stochastic oscillator (%K and %D)."""
    hh = high.rolling(k_period).max()
    ll = low.rolling(k_period).min()
    k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d}, index=close.index)


def ichimoku(high: pd.Series, low: pd.Series,
             conv_period: int = 9, base_period: int = 26) -> pd.DataFrame:
    """
    Ichimoku conversion and base lines only.
    Senkou spans are excluded (they use forward-shifted values = lookahead).
    """
    conv = (high.rolling(conv_period).max() + low.rolling(conv_period).min()) / 2
    base = (high.rolling(base_period).max() + low.rolling(base_period).min()) / 2
    return pd.DataFrame({
        "ichimoku_conv": conv,
        "ichimoku_base": base,
    }, index=high.index)


# =========================================================================
# Volatility
# =========================================================================

def bollinger_bands(close: pd.Series, period: int = 20,
                    std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: upper, middle, lower, bandwidth, and %B."""
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper = ma + std * sd
    lower = ma - std * sd
    bandwidth = (upper - lower) / ma.replace(0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({
        "bb_upper": upper,
        "bb_middle": ma,
        "bb_lower": lower,
        "bb_bandwidth": bandwidth,
        "bb_pct_b": pct_b,
    }, index=close.index)


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """Average True Range."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def natr(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> pd.Series:
    """Normalized ATR (percentage of close). Comparable across symbols."""
    return (atr(high, low, close, period) / close) * 100


def garman_klass_vol(high: pd.Series, low: pd.Series,
                     open_: pd.Series, close: pd.Series,
                     period: int = 20) -> pd.Series:
    """Garman-Klass volatility estimator."""
    log_hl = (np.log(high / low)) ** 2
    log_co = (np.log(close / open_)) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(np.clip(gk.rolling(period).mean(), a_min=0, a_max=None))


def parkinson_vol(high: pd.Series, low: pd.Series,
                  period: int = 20) -> pd.Series:
    """Parkinson volatility estimator."""
    log_hl = (np.log(high / low)) ** 2
    factor = 1 / (4 * np.log(2))
    return np.sqrt(factor * log_hl.rolling(period).mean())


def realized_vol(close: pd.Series, period: int = 20) -> pd.Series:
    """Realized (close-to-close) volatility."""
    returns = np.log(close / close.shift(1))
    return returns.rolling(period).std()


# =========================================================================
# Volume
# =========================================================================

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff())
    return (volume * direction).cumsum()


def obv_ema(close: pd.Series, volume: pd.Series, span: int = 20) -> pd.Series:
    """OBV smoothed with EMA."""
    return obv(close, volume).ewm(span=span, adjust=False).mean()


def cmf(high: pd.Series, low: pd.Series, close: pd.Series,
        volume: pd.Series, period: int = 20) -> pd.Series:
    """Chaikin Money Flow."""
    hl_range = (high - low).replace(0, np.nan)
    mf_mult = ((close - low) - (high - close)) / hl_range
    mf_vol = mf_mult * volume
    return mf_vol.rolling(period).sum() / volume.rolling(period).sum()


def vwap_rolling(high: pd.Series, low: pd.Series, close: pd.Series,
                 volume: pd.Series, period: int = 20) -> pd.Series:
    """Rolling VWAP (does not explode on long series unlike cumulative)."""
    tp = (high + low + close) / 3
    return (tp * volume).rolling(period).sum() / volume.rolling(period).sum()


def vwap_distance(close: pd.Series, high: pd.Series, low: pd.Series,
                  volume: pd.Series, period: int = 20) -> pd.Series:
    """Distance from rolling VWAP as percentage."""
    v = vwap_rolling(high, low, close, volume, period)
    return (close - v) / v.replace(0, np.nan) * 100


# =========================================================================
# Microstructure (crypto-specific)
# =========================================================================

def taker_delta(taker_buy_volume: pd.Series, volume: pd.Series) -> pd.Series:
    """Net taker delta (buy - sell). Positive = aggressive buyers."""
    taker_sell = volume - taker_buy_volume
    return taker_buy_volume - taker_sell


def taker_delta_ema(taker_buy_volume: pd.Series, volume: pd.Series,
                    span: int = 20) -> pd.Series:
    """Smoothed taker delta."""
    return taker_delta(taker_buy_volume, volume).ewm(span=span, adjust=False).mean()


def cvd_rolling(taker_buy_volume: pd.Series, volume: pd.Series,
                window: int = 24) -> pd.Series:
    """Rolling Cumulative Volume Delta (sum over window)."""
    delta = taker_delta(taker_buy_volume, volume)
    return delta.rolling(window).sum()


def taker_ratio(taker_buy_volume: pd.Series, volume: pd.Series) -> pd.Series:
    """Taker buy/sell ratio. >0.5 = buyer dominant."""
    return taker_buy_volume / volume.replace(0, np.nan)


def funding_zscore(funding_rate: pd.Series, window: int = 168) -> pd.Series:
    """Z-score of funding rate (168 = 1 week of hourly bars)."""
    mu = funding_rate.rolling(window).mean()
    sigma = funding_rate.rolling(window).std()
    return (funding_rate - mu) / sigma.replace(0, np.nan)


def basis_zscore(series: pd.Series, window: int = 100) -> pd.Series:
    """Z-score of futures-spot basis."""
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std()
    return (series - mu) / sigma.replace(0, np.nan)


def oi_velocity(open_interest: pd.Series,
                periods: list[int] = None) -> pd.DataFrame:
    """Rate of change of open interest at multiple periods."""
    if periods is None:
        periods = [5, 20, 60]
    result = {}
    for p in periods:
        result[f"oi_vel_{p}"] = open_interest.pct_change(p) * 100
    return pd.DataFrame(result, index=open_interest.index)


def vol_oi_ratio(volume: pd.Series, open_interest: pd.Series) -> pd.Series:
    """Volume / Open Interest ratio. High = aggressive activity."""
    return volume / open_interest.replace(0, np.nan)


def spot_fut_vol_ratio(futures_vol: pd.Series,
                       spot_vol: pd.Series, span: int = 20) -> pd.DataFrame:
    """Futures vs spot volume ratio + EMA smoothing."""
    ratio = futures_vol / spot_vol.replace(0, np.nan)
    return pd.DataFrame({
        "vol_fut_spot_ratio": ratio,
        "vol_fut_spot_ratio_ema": ratio.ewm(span=span, adjust=False).mean(),
    }, index=futures_vol.index)


# =========================================================================
# Targets (forward returns -- CONTAIN LOOKAHEAD BY DESIGN)
# =========================================================================

def generate_targets(close: pd.Series,
                     horizons: list[int] = None) -> pd.DataFrame:
    """
    Forward log returns for training.
    WARNING: these columns CONTAIN LOOKAHEAD BIAS by definition.
    Drop them before inference/backtesting.
    """
    if horizons is None:
        horizons = [1, 5, 12]
    result = {}
    for h in horizons:
        fwd = np.log(close.shift(-h) / close)
        # Force last h rows to NaN to make the lookahead explicit
        fwd.iloc[-h:] = np.nan
        result[f"target_return_{h}f"] = fwd
        result[f"target_dir_{h}f"] = (fwd > 0).astype(float)
        result[f"target_dir_{h}f"].iloc[-h:] = np.nan
    return pd.DataFrame(result, index=close.index)
