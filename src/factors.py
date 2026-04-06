"""
factors.py
----------
Cross-sectional momentum factor definitions.

IMPORTANT – NO LOOK-AHEAD BIAS
-------------------------------
Every factor function accepts a ``date`` argument and must use ONLY price
data that is strictly BEFORE ``date``. An explicit assertion is included in
each function to enforce this. The ``prices`` DataFrame passed in may contain
the full history, but internal slicing ensures no future information leaks.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


def _prior_prices(prices: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """Return the subset of ``prices`` strictly before ``date``.

    Args:
        prices: Full price history DataFrame.
        date: The rebalance (signal computation) date.

    Returns:
        DataFrame containing only rows with index < ``date``.

    Raises:
        AssertionError: If the resulting slice is empty or the last index is
            not strictly before ``date``.
    """
    subset = prices.loc[prices.index < date]
    assert not subset.empty, f"No price data available before {date}."
    # Enforce that the last data point used is at least 1 trading day before date.
    assert subset.index[-1] < date, (
        f"Look-ahead detected: last data point {subset.index[-1]} is not "
        f"strictly before rebalance date {date}."
    )
    return subset


def momentum_12_1(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    long_window: int = 252,
    skip: int = 21,
) -> pd.Series:
    """Compute the 12-1 month momentum factor for all tickers as of ``date``.

    The signal is the cumulative return from t-``long_window`` to t-``skip``
    trading days before ``date`` (skipping the most recent month to avoid
    the 1-month reversal effect).

    Only data strictly BEFORE ``date`` is used (no look-ahead).

    Args:
        prices: Full adjusted-close price DataFrame (dates × tickers).
        date: The rebalance date for which to compute the signal.
        long_window: Number of trading days defining the look-back start
            (default 252 ≈ 12 months).
        skip: Number of most-recent trading days to skip (default 21 ≈ 1 month).

    Returns:
        pd.Series indexed by ticker containing the 12-1 momentum score.
        Tickers with insufficient history return NaN.
    """
    # ── No-look-ahead guard ───────────────────────────────────────────────────
    hist = _prior_prices(prices, date)

    scores: dict[str, float] = {}
    for ticker in hist.columns:
        col = hist[ticker].dropna()

        # Need at least long_window + skip observations
        if len(col) < long_window + skip:
            scores[ticker] = np.nan
            continue

        # t-skip is the "end" of the formation period
        end_idx = len(col) - skip
        # t-long_window is the "start" of the formation period
        start_idx = len(col) - long_window

        if start_idx < 0 or end_idx <= start_idx:
            scores[ticker] = np.nan
            continue

        p_start = col.iloc[start_idx]
        p_end = col.iloc[end_idx]

        if p_start <= 0:
            scores[ticker] = np.nan
        else:
            scores[ticker] = (p_end / p_start) - 1.0

    result = pd.Series(scores, name="mom_12_1")
    return result


def momentum_6_1(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    long_window: int = 126,
    skip: int = 21,
) -> pd.Series:
    """Compute the 6-1 month momentum factor for all tickers as of ``date``.

    The signal is the cumulative return from t-``long_window`` to t-``skip``
    trading days before ``date`` (skipping the most recent month).

    Only data strictly BEFORE ``date`` is used (no look-ahead).

    Args:
        prices: Full adjusted-close price DataFrame (dates × tickers).
        date: The rebalance date for which to compute the signal.
        long_window: Number of trading days defining the look-back start
            (default 126 ≈ 6 months).
        skip: Number of most-recent trading days to skip (default 21 ≈ 1 month).

    Returns:
        pd.Series indexed by ticker containing the 6-1 momentum score.
        Tickers with insufficient history return NaN.
    """
    # ── No-look-ahead guard ───────────────────────────────────────────────────
    hist = _prior_prices(prices, date)

    scores: dict[str, float] = {}
    for ticker in hist.columns:
        col = hist[ticker].dropna()

        if len(col) < long_window + skip:
            scores[ticker] = np.nan
            continue

        end_idx = len(col) - skip
        start_idx = len(col) - long_window

        if start_idx < 0 or end_idx <= start_idx:
            scores[ticker] = np.nan
            continue

        p_start = col.iloc[start_idx]
        p_end = col.iloc[end_idx]

        if p_start <= 0:
            scores[ticker] = np.nan
        else:
            scores[ticker] = (p_end / p_start) - 1.0

    result = pd.Series(scores, name="mom_6_1")
    return result


def residual_momentum(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    date: pd.Timestamp,
    window: int = 252,
    skip: int = 21,
) -> pd.Series:
    """Compute the residual momentum factor for all tickers as of ``date``.

    For each stock, OLS-regress daily excess returns on the benchmark's daily
    excess returns over the formation window [t-window, t-skip]. The factor
    score is the cumulative sum of OLS residuals over that window (i.e., the
    portion of stock return unexplained by the market).

    Uses statsmodels OLS. Only data strictly BEFORE ``date`` is used.

    NOTE: We use raw returns here as a proxy for excess returns when a
    risk-free series is not passed in directly. The benchmark alignment is
    the key market-beta neutralisation.

    Args:
        prices: Full adjusted-close price DataFrame (dates × tickers).
        benchmark_prices: Full adjusted-close price Series for the benchmark.
        date: The rebalance date for which to compute the signal.
        window: Formation window in trading days (default 252 ≈ 12 months).
        skip: Most-recent days to skip (default 21 ≈ 1 month).

    Returns:
        pd.Series indexed by ticker containing the residual momentum score.
        Tickers with insufficient history return NaN.
    """
    # ── No-look-ahead guard ───────────────────────────────────────────────────
    hist = _prior_prices(prices, date)
    bm_hist = benchmark_prices.loc[benchmark_prices.index < date]

    assert not bm_hist.empty, "No benchmark data available before date."
    assert bm_hist.index[-1] < date, "Look-ahead in benchmark series."

    # Compute daily benchmark returns
    bm_ret = bm_hist.pct_change().dropna()

    scores: dict[str, float] = {}
    for ticker in hist.columns:
        col = hist[ticker].dropna()

        if len(col) < window + skip + 1:
            scores[ticker] = np.nan
            continue

        # Formation window: from t-window to t-skip
        stock_ret = col.pct_change().dropna()

        # Align on dates in the formation window
        # end = last day of formation window (skip days before last available)
        end_loc = len(stock_ret) - skip
        start_loc = end_loc - window

        if start_loc < 0 or end_loc <= start_loc:
            scores[ticker] = np.nan
            continue

        stock_window = stock_ret.iloc[start_loc:end_loc]

        # Align benchmark to same dates
        bm_window = bm_ret.reindex(stock_window.index).dropna()
        stock_aligned = stock_window.reindex(bm_window.index).dropna()

        if len(stock_aligned) < 60:  # Need at least 60 observations for regression
            scores[ticker] = np.nan
            continue

        # OLS regression: stock_ret = alpha + beta * bm_ret + epsilon
        X = sm.add_constant(bm_window.loc[stock_aligned.index])
        model = sm.OLS(stock_aligned, X)
        results = model.fit()
        residuals = results.resid

        # Cumulative sum of residuals = residual momentum score
        scores[ticker] = residuals.sum()

    result = pd.Series(scores, name="residual_mom")
    return result


def compute_all_factors(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    date: pd.Timestamp,
    eligible_tickers: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Compute all three momentum factors for a given rebalance date.

    Calls ``momentum_12_1``, ``momentum_6_1``, and ``residual_momentum``
    and joins the results into a single DataFrame.

    Only data strictly BEFORE ``date`` is used in every sub-computation.

    Args:
        prices: Full adjusted-close price DataFrame (dates × tickers).
        benchmark_prices: Full adjusted-close price Series for the benchmark.
        date: The rebalance date.
        eligible_tickers: If provided, restrict output to this list of tickers.

    Returns:
        pd.DataFrame with columns ["ticker", "mom_12_1", "mom_6_1",
        "residual_mom"] and one row per ticker.
    """
    # Optionally restrict to eligible universe
    if eligible_tickers is not None:
        avail = [t for t in eligible_tickers if t in prices.columns]
        prices_sub = prices[avail]
    else:
        prices_sub = prices

    m12 = momentum_12_1(prices_sub, date)
    m6 = momentum_6_1(prices_sub, date)
    rm = residual_momentum(prices_sub, benchmark_prices, date)

    df = pd.DataFrame(
        {
            "mom_12_1": m12,
            "mom_6_1": m6,
            "residual_mom": rm,
        }
    )
    df.index.name = "ticker"
    df = df.reset_index()
    return df
