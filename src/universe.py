"""
universe.py
-----------
Constructs the investable universe at each monthly rebalance date, applying
minimum history and minimum price filters.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def apply_universe_filters(
    prices_df: pd.DataFrame,
    min_history_days: int = 504,
    min_price: float = 5.0,
    rebalance_freq: str = "ME",  # pandas >= 2.2: use "ME" for month-end
) -> dict[pd.Timestamp, list[str]]:
    """Build a map of {rebalance_date → eligible_tickers} using point-in-time filters.

    At each monthly rebalance date a ticker is eligible only if:
      1. It has at least ``min_history_days`` non-NaN observations strictly
         **before** the rebalance date (no look-ahead).
      2. Its most-recent price strictly before the rebalance date is at or
         above ``min_price``.

    Args:
        prices_df: DataFrame of adjusted close prices; DatetimeIndex × ticker columns.
        min_history_days: Minimum number of valid (non-NaN) price observations
            required before the rebalance date.
        min_price: Minimum share price required on the most recent available date
            before the rebalance date.
        rebalance_freq: Pandas offset alias for rebalance frequency (default "M"
            = last trading day of each month).

    Returns:
        Dictionary mapping each rebalance ``pd.Timestamp`` to a list of eligible
        ticker strings.
    """
    # Generate rebalance dates: last trading day of each month within the sample
    # We resample the index to find the last observation in each period.
    rebalance_dates: pd.DatetimeIndex = (
        prices_df.resample(rebalance_freq).last().index
    )

    eligible_map: dict[pd.Timestamp, list[str]] = {}

    for reb_date in rebalance_dates:
        # ── Point-in-time slice: use only data STRICTLY BEFORE reb_date ──────
        # This is the core no-look-ahead guard for universe construction.
        hist = prices_df.loc[prices_df.index < reb_date]

        if hist.empty:
            eligible_map[reb_date] = []
            continue

        eligible: list[str] = []
        for ticker in prices_df.columns:
            col = hist[ticker].dropna()

            # Filter 1: minimum history
            if len(col) < min_history_days:
                continue

            # Filter 2: minimum price on most recent available day
            last_price = col.iloc[-1]
            if last_price < min_price:
                continue

            eligible.append(ticker)

        eligible_map[reb_date] = eligible

        logger.debug(
            "Rebalance %s: %d eligible tickers (of %d total).",
            reb_date.date(),
            len(eligible),
            len(prices_df.columns),
        )

    logger.info(
        "Universe filters applied over %d rebalance dates. "
        "Avg eligible tickers: %.1f.",
        len(rebalance_dates),
        sum(len(v) for v in eligible_map.values()) / max(len(eligible_map), 1),
    )
    return eligible_map
