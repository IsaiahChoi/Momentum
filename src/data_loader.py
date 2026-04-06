"""
data_loader.py
--------------
Functions for fetching and cleaning market data from Yahoo Finance.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Maximum consecutive trading days to forward-fill
_MAX_FFILL_DAYS: int = 5
# Maximum fraction of missing values tolerated per ticker
_MAX_MISSING_FRAC: float = 0.05


def fetch_price_data(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download daily adjusted-close prices for a list of tickers.

    Uses yfinance bulk download, then applies forward-fill (up to
    ``_MAX_FFILL_DAYS``) and drops tickers with excessive missing data.

    Args:
        tickers: List of Yahoo Finance ticker symbols.
        start: Start date string in "YYYY-MM-DD" format.
        end: End date string in "YYYY-MM-DD" format.

    Returns:
        pd.DataFrame with DatetimeIndex (trading days) and one column per
        surviving ticker containing adjusted close prices.
    """
    logger.info("Downloading price data for %d tickers from %s to %s …", len(tickers), start, end)

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
        threads=True,
    )

    # yfinance returns multi-level columns when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        prices: pd.DataFrame = raw["Close"].copy()
    else:
        # Single ticker edge-case
        prices = raw[["Close"]].copy()
        prices.columns = tickers[:1]

    prices.index = pd.to_datetime(prices.index)
    prices.sort_index(inplace=True)

    # ── Forward-fill up to _MAX_FFILL_DAYS ───────────────────────────────────
    prices = prices.ffill(limit=_MAX_FFILL_DAYS)

    # ── Drop tickers with too many missing values ─────────────────────────────
    n_rows = len(prices)
    missing_frac = prices.isna().sum() / n_rows
    bad_tickers = missing_frac[missing_frac > _MAX_MISSING_FRAC].index.tolist()
    if bad_tickers:
        logger.warning(
            "Dropping %d tickers with >%.0f%% missing data: %s",
            len(bad_tickers),
            _MAX_MISSING_FRAC * 100,
            bad_tickers,
        )
    prices = prices.drop(columns=bad_tickers)

    logger.info("Price data ready: %d rows × %d tickers.", len(prices), prices.shape[1])
    return prices


def fetch_risk_free_rate(
    start: str,
    end: str,
) -> pd.Series:
    """Download the 13-week T-bill annualised yield and convert to daily rate.

    The ^IRX ticker returns an annualised yield in *percent*. This function
    divides by 100 to get a decimal annual yield and then converts to a
    daily rate using simple (not compound) division by 252.

    Args:
        start: Start date string in "YYYY-MM-DD" format.
        end: End date string in "YYYY-MM-DD" format.

    Returns:
        pd.Series indexed by date containing the daily risk-free rate
        (expressed as a decimal, e.g. 0.00019 for ~5% annual).
    """
    logger.info("Downloading risk-free rate (^IRX) …")
    raw = yf.download("^IRX", start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].squeeze()
    else:
        close = raw["Close"]

    close.index = pd.to_datetime(close.index)
    close = close.sort_index().ffill(limit=_MAX_FFILL_DAYS)

    # Convert annualised % → daily decimal rate
    daily_rf: pd.Series = (close / 100.0) / 252.0
    daily_rf.name = "risk_free_daily"
    logger.info("Risk-free rate fetched: %d trading days.", len(daily_rf))
    return daily_rf


def fetch_sector_map(tickers: list[str]) -> dict[str, str]:
    """Retrieve the GICS sector string for each ticker via yfinance .info.

    Makes individual HTTP requests for each ticker; retries once on failure.
    Tickers where the sector cannot be determined are mapped to "Unknown".

    Args:
        tickers: List of Yahoo Finance ticker symbols.

    Returns:
        Dictionary mapping ticker symbol → GICS sector string
        (e.g. ``{"AAPL": "Technology", "JPM": "Financial Services"}``).
    """
    sector_map: dict[str, str] = {}
    logger.info("Fetching sector info for %d tickers …", len(tickers))

    for ticker in tqdm(tickers, desc="Sector map", unit="ticker"):
        for attempt in range(2):
            try:
                info = yf.Ticker(ticker).info
                sector_map[ticker] = info.get("sector", "Unknown") or "Unknown"
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 0:
                    time.sleep(1)
                else:
                    logger.warning("Could not fetch sector for %s: %s", ticker, exc)
                    sector_map[ticker] = "Unknown"

    return sector_map
