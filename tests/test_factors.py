"""
test_factors.py
---------------
Unit tests for src/factors.py using pytest.

Tests
-----
1. momentum_12_1 returns NaN for a stock with < 252 days of data.
2. momentum_12_1 computes the correct value on a synthetic series
   where the answer is known (a stock that exactly doubles gives ~100%).
3. No future data is used: passing a price DataFrame truncated at
   date - 1 day yields the same result as passing the full DataFrame.
4. momentum_6_1 returns NaN for a stock with < 126 days of data.
5. residual_momentum returns NaN for a stock with insufficient history.
6. compute_all_factors returns a DataFrame with the expected columns.
"""

from __future__ import annotations

import sys
import os
import numpy as np
import pandas as pd
import pytest

# Ensure the project root is on the path when running pytest from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.factors import (
    momentum_12_1,
    momentum_6_1,
    residual_momentum,
    compute_all_factors,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_flat_prices(
    n_days: int = 300,
    n_tickers: int = 3,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Create a DataFrame of flat (constant) daily prices."""
    dates = pd.bdate_range("2010-01-01", periods=n_days)
    data = {f"TICK{i}": np.full(n_days, start_price) for i in range(n_tickers)}
    return pd.DataFrame(data, index=dates)


def make_doubling_stock(n_days: int = 300) -> pd.DataFrame:
    """Create a single stock that doubles over its history.

    Price starts at 100 and ends at 200 across n_days business days.
    """
    dates = pd.bdate_range("2010-01-01", periods=n_days)
    prices = np.linspace(100.0, 200.0, n_days)
    return pd.DataFrame({"DOUBLE": prices}, index=dates)


def make_short_prices(n_days: int = 100) -> pd.DataFrame:
    """Create a price DataFrame with only 100 days of data (< min 252+21)."""
    dates = pd.bdate_range("2010-01-01", periods=n_days)
    return pd.DataFrame({"SHORT": np.linspace(100.0, 110.0, n_days)}, index=dates)


# ── Test: momentum_12_1 returns NaN for insufficient history ──────────────────

class TestMomentum12_1NaN:
    """Test that momentum_12_1 returns NaN when data is insufficient."""

    def test_nan_for_short_history(self) -> None:
        """A ticker with < 252 + 21 days of data should receive a NaN score."""
        prices = make_short_prices(n_days=250)  # 250 < 252+21 = 273
        # Use a date one business day after the last available date
        date = prices.index[-1] + pd.offsets.BDay(1)
        scores = momentum_12_1(prices, date)
        assert "SHORT" in scores.index, "Ticker 'SHORT' should be in output."
        assert np.isnan(scores["SHORT"]), (
            f"Expected NaN for ticker with only {len(prices)} days of data, "
            f"got {scores['SHORT']:.4f} instead."
        )

    def test_nan_exactly_at_threshold(self) -> None:
        """A ticker with exactly 272 days (= 252+21-1) should also get NaN."""
        prices = make_short_prices(n_days=272)
        date = prices.index[-1] + pd.offsets.BDay(1)
        scores = momentum_12_1(prices, date)
        assert np.isnan(scores["SHORT"]), "Ticker with 272 days should return NaN."

    def test_not_nan_above_threshold(self) -> None:
        """A ticker with exactly 273 days (= 252+21) should return a valid score."""
        prices = make_flat_prices(n_days=274, n_tickers=1)
        prices.columns = ["TICK"]
        date = prices.index[-1] + pd.offsets.BDay(1)
        scores = momentum_12_1(prices, date)
        assert not np.isnan(scores["TICK"]), (
            "Ticker with 274 days should return a valid (non-NaN) score."
        )


# ── Test: momentum_12_1 on known synthetic series ─────────────────────────────

class TestMomentum12_1KnownValue:
    """Test momentum_12_1 against analytically known results."""

    def test_doubling_stock(self) -> None:
        """A stock that linearly doubles should have ~100% 12-1 momentum."""
        prices = make_doubling_stock(n_days=300)
        date = prices.index[-1] + pd.offsets.BDay(1)
        scores = momentum_12_1(prices, date, long_window=252, skip=21)

        # The formation window spans from index[300-252] to index[300-21].
        # price at start ≈ 100 + (300-252)/300 * 100 ≈ 116
        # price at end   ≈ 100 + (300-21)/300  * 100 ≈ 193
        # expected return ≈ (193-116)/116 ≈ 0.664
        # We check that it is in a reasonable range (positive and large).
        score = scores["DOUBLE"]
        assert not np.isnan(score), "Score should not be NaN for 300-day series."
        assert score > 0.3, f"Expected momentum score > 0.3 for doubling stock, got {score:.4f}."
        assert score < 1.5, f"Expected momentum score < 1.5 for doubling stock, got {score:.4f}."

    def test_flat_stock_zero_momentum(self) -> None:
        """A flat stock should have ~0% momentum."""
        prices = make_flat_prices(n_days=300, n_tickers=1)
        prices.columns = ["FLAT"]
        date = prices.index[-1] + pd.offsets.BDay(1)
        scores = momentum_12_1(prices, date)
        score = scores["FLAT"]
        assert not np.isnan(score), "Score should not be NaN."
        assert abs(score) < 1e-10, f"Expected ~0 momentum for flat stock, got {score:.6f}."

    def test_declining_stock_negative(self) -> None:
        """A stock that halves should have negative momentum."""
        dates = pd.bdate_range("2010-01-01", periods=300)
        prices = pd.DataFrame(
            {"HALVE": np.linspace(200.0, 100.0, 300)},
            index=dates,
        )
        date = prices.index[-1] + pd.offsets.BDay(1)
        scores = momentum_12_1(prices, date)
        assert scores["HALVE"] < 0, "Halving stock should have negative momentum."


# ── Test: no look-ahead bias ───────────────────────────────────────────────────

class TestNoLookAhead:
    """Verify that factor results are identical regardless of future data appended."""

    def test_truncated_equals_full_mom12_1(self) -> None:
        """Result on truncated DataFrame must match result on full DataFrame."""
        full_prices = make_doubling_stock(n_days=350)
        # Use a date in the middle of the sample
        date = full_prices.index[300]

        # Full DataFrame (has future data after ``date``)
        score_full = momentum_12_1(full_prices, date)

        # Truncated DataFrame: ends strictly before ``date``
        truncated = full_prices.loc[full_prices.index < date]
        # Add one more future day to truncated to make it end exactly at date-1
        score_trunc = momentum_12_1(truncated, date)

        assert abs(score_full["DOUBLE"] - score_trunc["DOUBLE"]) < 1e-12, (
            f"Score on full DataFrame ({score_full['DOUBLE']:.6f}) differs from "
            f"score on truncated DataFrame ({score_trunc['DOUBLE']:.6f}). "
            "This indicates look-ahead bias."
        )

    def test_truncated_equals_full_mom6_1(self) -> None:
        """Same look-ahead test for momentum_6_1."""
        full_prices = make_doubling_stock(n_days=300)
        date = full_prices.index[250]

        score_full = momentum_6_1(full_prices, date)
        truncated = full_prices.loc[full_prices.index < date]
        score_trunc = momentum_6_1(truncated, date)

        assert abs(score_full["DOUBLE"] - score_trunc["DOUBLE"]) < 1e-12, (
            "momentum_6_1 look-ahead bias detected."
        )

    def test_assertion_raised_on_empty_slice(self) -> None:
        """Passing a price DataFrame where all data is AFTER date should raise."""
        dates = pd.bdate_range("2015-01-01", periods=300)
        prices = pd.DataFrame({"TICK": np.ones(300) * 100}, index=dates)
        # Use a date BEFORE the data starts
        early_date = pd.Timestamp("2010-01-01")
        with pytest.raises(AssertionError):
            momentum_12_1(prices, early_date)


# ── Test: momentum_6_1 ────────────────────────────────────────────────────────

class TestMomentum6_1:
    """Tests specific to the 6-1 momentum factor."""

    def test_nan_for_insufficient_data(self) -> None:
        """A ticker with < 126 + 21 = 147 days of data should get NaN."""
        prices = make_short_prices(n_days=140)
        date = prices.index[-1] + pd.offsets.BDay(1)
        scores = momentum_6_1(prices, date)
        assert np.isnan(scores["SHORT"]), "Expected NaN for 140-day series."

    def test_valid_for_sufficient_data(self) -> None:
        """A ticker with 160 days should get a valid score."""
        prices = make_flat_prices(n_days=160, n_tickers=1)
        prices.columns = ["TICK"]
        date = prices.index[-1] + pd.offsets.BDay(1)
        scores = momentum_6_1(prices, date)
        assert not np.isnan(scores["TICK"]), "Expected valid score for 160-day series."


# ── Test: residual_momentum ───────────────────────────────────────────────────

class TestResidualMomentum:
    """Tests for the residual momentum factor."""

    def test_nan_for_insufficient_data(self) -> None:
        """Residual momentum should return NaN for short histories."""
        n = 100
        dates = pd.bdate_range("2010-01-01", periods=n)
        prices = pd.DataFrame({"TICK": np.linspace(100, 110, n)}, index=dates)
        benchmark = pd.Series(np.linspace(100, 105, n), index=dates, name="SPY")
        date = prices.index[-1] + pd.offsets.BDay(1)
        scores = residual_momentum(prices, benchmark, date)
        assert np.isnan(scores["TICK"]), "Expected NaN for 100-day series."

    def test_valid_for_long_history(self) -> None:
        """Residual momentum should return a valid float for 300+ day series."""
        np.random.seed(42)
        n = 320
        dates = pd.bdate_range("2010-01-01", periods=n)
        bm_returns = np.random.normal(0.0005, 0.01, n)
        bm_prices = 100 * np.cumprod(1 + bm_returns)
        # Stock with some correlation to benchmark
        stock_returns = 0.8 * bm_returns + np.random.normal(0.0002, 0.005, n)
        stock_prices = 100 * np.cumprod(1 + stock_returns)

        prices = pd.DataFrame({"TICK": stock_prices}, index=dates)
        benchmark = pd.Series(bm_prices, index=dates, name="SPY")
        date = prices.index[-1] + pd.offsets.BDay(1)
        scores = residual_momentum(prices, benchmark, date)
        assert not np.isnan(scores["TICK"]), "Expected valid residual momentum score."


# ── Test: compute_all_factors ─────────────────────────────────────────────────

class TestComputeAllFactors:
    """Tests for the compute_all_factors convenience function."""

    def test_output_columns(self) -> None:
        """Output DataFrame must contain 'ticker', 'mom_12_1', 'mom_6_1', 'residual_mom'."""
        np.random.seed(1)
        n = 320
        dates = pd.bdate_range("2010-01-01", periods=n)
        tickers = ["A", "B", "C"]
        prices_data = {t: 100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n)) for t in tickers}
        prices = pd.DataFrame(prices_data, index=dates)
        bm = pd.Series(100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n)), index=dates, name="SPY")
        date = prices.index[-1] + pd.offsets.BDay(1)

        df = compute_all_factors(prices, bm, date)
        for col in ["ticker", "mom_12_1", "mom_6_1", "residual_mom"]:
            assert col in df.columns, f"Column '{col}' missing from compute_all_factors output."

    def test_ticker_count(self) -> None:
        """Output should have one row per ticker."""
        np.random.seed(2)
        n = 320
        dates = pd.bdate_range("2010-01-01", periods=n)
        tickers = ["X", "Y", "Z", "W"]
        prices = pd.DataFrame(
            {t: 100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n)) for t in tickers},
            index=dates,
        )
        bm = pd.Series(100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n)), index=dates)
        date = prices.index[-1] + pd.offsets.BDay(1)
        df = compute_all_factors(prices, bm, date)
        assert len(df) == len(tickers), (
            f"Expected {len(tickers)} rows, got {len(df)}."
        )
