"""
Error Hierarchy
===============
Every failure maps to one of these exceptions.
Callers can catch at the granularity they need:
    - BaseError           -> catch everything
    - DataError           -> exchange/cache problems
    - ModelError          -> HMM/XGBoost/GARCH failures
    - StrategyError       -> signal computation errors
    - ExecutionError      -> trade execution failures
    - BudgetError         -> AI spending limits exceeded
    - ConfigError         -> missing/invalid configuration
"""


class BaseError(Exception):
    """Base exception for all project errors."""


class ConfigError(BaseError):
    """Missing or invalid configuration (API keys, paths, params)."""


class DataError(BaseError):
    """Data ingestion or caching failure."""


class DataStaleError(DataError):
    """Data was returned but is stale (from cache fallback)."""


class DataUnavailableError(DataError):
    """No data available — exchange down and no cache."""


class ModelError(BaseError):
    """Model training, prediction, or loading failure."""


class RegimeError(ModelError):
    """HMM regime detection specific failure."""


class CalibrationError(ModelError):
    """Probability calibration failure (XGBoost + Isotonic)."""


class StrategyError(BaseError):
    """Strategy signal computation failure."""


class ExecutionError(BaseError):
    """Trade execution failure (paper or live)."""


class PositionError(ExecutionError):
    """Position management error (duplicate, sizing, etc)."""


class BudgetError(BaseError):
    """AI token budget exceeded."""


class AlertError(BaseError):
    """Notification delivery failure (non-fatal)."""
