"""
utils.py
--------
Miscellaneous utility functions shared across the library.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def align_series(
    *series: pd.Series,
    fill_method: Optional[str] = "ffill",
    fill_limit: int = 5,
) -> tuple[pd.Series, ...]:
    """Align multiple pd.Series to a common DatetimeIndex.

    Args:
        *series: Any number of pd.Series objects with DatetimeIndex.
        fill_method: Method for filling gaps after alignment ("ffill", "bfill",
            or None to leave NaNs). Default "ffill".
        fill_limit: Maximum number of consecutive periods to fill. Default 5.

    Returns:
        Tuple of aligned pd.Series objects in the same order as input.
    """
    common_idx = series[0].index
    for s in series[1:]:
        common_idx = common_idx.intersection(s.index)

    aligned = []
    for s in series:
        s_aligned = s.reindex(common_idx)
        if fill_method == "ffill":
            s_aligned = s_aligned.ffill(limit=fill_limit)
        elif fill_method == "bfill":
            s_aligned = s_aligned.bfill(limit=fill_limit)
        aligned.append(s_aligned)

    return tuple(aligned)


def prices_to_returns(
    prices: pd.DataFrame,
    log_returns: bool = False,
) -> pd.DataFrame:
    """Convert price DataFrame to return DataFrame.

    Args:
        prices: Adjusted-close price DataFrame (dates × tickers).
        log_returns: If True, compute log returns; otherwise simple returns.

    Returns:
        pd.DataFrame of returns with the same columns; first row is NaN.
    """
    if log_returns:
        return np.log(prices / prices.shift(1))
    return prices.pct_change()


def rolling_cov_matrix(
    returns: pd.DataFrame,
    window: int = 252,
) -> Optional[pd.DataFrame]:
    """Compute the trailing rolling covariance matrix on the most recent ``window`` days.

    Args:
        returns: Daily return DataFrame (dates × tickers).
        window: Rolling look-back in trading days (default 252).

    Returns:
        Covariance matrix as pd.DataFrame (tickers × tickers), or None if
        insufficient data.
    """
    if len(returns) < window:
        logger.warning(
            "Insufficient data for rolling covariance: %d rows, need %d.",
            len(returns),
            window,
        )
        return None
    recent = returns.iloc[-window:]
    return recent.cov()


def winsorise(
    series: pd.Series,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.Series:
    """Winsorise extreme values in a Series to the given percentile bounds.

    Args:
        series: pd.Series of factor scores or returns.
        lower_pct: Lower percentile bound (default 0.01 = 1st percentile).
        upper_pct: Upper percentile bound (default 0.99 = 99th percentile).

    Returns:
        pd.Series with values clipped to [lower_pct, upper_pct] quantiles.
    """
    lo = series.quantile(lower_pct)
    hi = series.quantile(upper_pct)
    return series.clip(lower=lo, upper=hi)


def get_last_trading_days(
    prices: pd.DataFrame,
    freq: str = "ME",  # pandas >= 2.2: use "ME" for month-end
) -> pd.DatetimeIndex:
    """Extract the last trading day of each period from a price DataFrame.

    Args:
        prices: Price DataFrame with DatetimeIndex of trading days.
        freq: Pandas offset alias for the period (default "M" = monthly).

    Returns:
        pd.DatetimeIndex of last trading days for each period.
    """
    return prices.resample(freq).last().index


def format_pct(value: float, decimals: int = 2) -> str:
    """Format a decimal value as a percentage string.

    Args:
        value: Decimal value (e.g., 0.1234 for 12.34%).
        decimals: Number of decimal places (default 2).

    Returns:
        Formatted string (e.g., "12.34%").
    """
    return f"{value * 100:.{decimals}f}%"


def annualize_return(
    cumulative_return: float,
    n_years: float,
) -> float:
    """Convert a total cumulative return to a compound annual growth rate.

    Args:
        cumulative_return: Total return over the period (e.g., 0.5 = 50%).
        n_years: Number of years in the period.

    Returns:
        Annualized return as a decimal.
    """
    if n_years <= 0:
        return np.nan
    return (1 + cumulative_return) ** (1.0 / n_years) - 1.0
