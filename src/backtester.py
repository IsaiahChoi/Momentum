"""
backtester.py
-------------
Event-driven monthly momentum backtester.

NO LOOK-AHEAD BIAS POLICY
--------------------------
- Factor signals are computed using only price data strictly before each
  rebalance date (enforced inside factors.py via assertion).
- Positions are entered at the CLOSE of the trading day AFTER the rebalance
  date (i.e., the rebalance date itself is the signal date; execution is the
  next business day's close).
- Daily portfolio returns are computed from the execution date forward until
  the next execution date.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .factors import compute_all_factors, momentum_12_1, momentum_6_1, residual_momentum
from .portfolio import construct_long_short, get_all_quantile_weights

logger = logging.getLogger(__name__)

# ── Internal constants ────────────────────────────────────────────────────────
_FACTOR_FUNCS = {
    "mom_12_1": momentum_12_1,
    "mom_6_1": momentum_6_1,
    "residual_mom": residual_momentum,
}


class MomentumBacktester:
    """Event-driven monthly cross-sectional momentum backtester.

    Computes factor signals on each monthly rebalance date, constructs
    long–short quantile portfolios, and tracks daily returns including
    transaction costs.

    Args:
        prices: Adjusted-close price DataFrame (dates × tickers).
        benchmark_prices: Adjusted-close price Series for the benchmark (SPY).
        risk_free: Daily risk-free rate Series (decimal, aligned with prices).
        sector_map: Dictionary mapping ticker → GICS sector string.
        universe_filter: Output of ``apply_universe_filters``; maps each
            rebalance date to a list of eligible tickers.
        config: Module-level config (used for TRANSACTION_COST_BPS).
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        benchmark_prices: pd.Series,
        risk_free: pd.Series,
        sector_map: dict[str, str],
        universe_filter: dict[pd.Timestamp, list[str]],
        config: Any,
    ) -> None:
        self.prices = prices
        self.benchmark_prices = benchmark_prices
        self.risk_free = risk_free
        self.sector_map = sector_map
        self.universe_filter = universe_filter
        self.config = config
        self._trading_days = prices.index

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_factor_scores(
        self,
        factor_name: str,
        rebalance_date: pd.Timestamp,
        eligible_tickers: list[str],
    ) -> pd.Series:
        """Compute factor scores for eligible tickers on a given rebalance date.

        Only data strictly BEFORE ``rebalance_date`` is used.

        Args:
            factor_name: One of "mom_12_1", "mom_6_1", "residual_mom".
            rebalance_date: The signal computation date.
            eligible_tickers: Tickers in the investable universe on this date.

        Returns:
            pd.Series of factor scores indexed by ticker.
        """
        avail = [t for t in eligible_tickers if t in self.prices.columns]
        prices_sub = self.prices[avail]

        if factor_name == "residual_mom":
            return residual_momentum(prices_sub, self.benchmark_prices, rebalance_date)
        else:
            func = _FACTOR_FUNCS[factor_name]
            return func(prices_sub, rebalance_date)

    def _daily_returns(
        self, weights: dict[str, float], start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.Series:
        """Compute the daily return of an equal-weighted portfolio between two dates.

        Args:
            weights: Dictionary {ticker: portfolio_weight} (weights sum to 1.0).
            start_date: First day to compute returns (inclusive).
            end_date: Last day to compute returns (exclusive — next rebalance).

        Returns:
            pd.Series of daily portfolio returns indexed by date.
        """
        if not weights:
            idx = self._trading_days[
                (self._trading_days >= start_date) & (self._trading_days < end_date)
            ]
            return pd.Series(0.0, index=idx)

        tickers = list(weights.keys())
        w = np.array([weights[t] for t in tickers])

        # Compute daily returns for the holding period
        price_window = self.prices.loc[
            (self.prices.index >= start_date) & (self.prices.index < end_date), tickers
        ]

        if price_window.empty:
            return pd.Series(dtype=float)

        rets = price_window.pct_change()
        # Drop the first row (NaN from pct_change) but keep the rest
        rets = rets.iloc[1:] if len(rets) > 1 else rets

        # Forward-fill any remaining NaNs (halted trading etc.)
        rets = rets.ffill().fillna(0.0)

        portfolio_ret = rets.values @ w
        return pd.Series(portfolio_ret, index=rets.index)

    def _compute_turnover(
        self,
        old_weights: dict[str, float],
        new_weights: dict[str, float],
    ) -> float:
        """Compute one-way turnover between consecutive portfolios.

        Turnover = 0.5 * sum(|new_weight - old_weight|) for each ticker.

        Args:
            old_weights: Previous period weights {ticker: weight}.
            new_weights: New period weights {ticker: weight}.

        Returns:
            Turnover as a fraction in [0, 1].
        """
        all_tickers = set(old_weights) | set(new_weights)
        turnover = 0.0
        for t in all_tickers:
            old_w = old_weights.get(t, 0.0)
            new_w = new_weights.get(t, 0.0)
            turnover += abs(new_w - old_w)
        return turnover / 2.0  # one-way

    # ── Public interface ──────────────────────────────────────────────────────

    def run(
        self,
        factor_name: str,
        n_quantiles: int = 5,
    ) -> pd.DataFrame:
        """Run the full backtest for a given factor without sector neutralisation.

        On each monthly rebalance date:
          1. Compute factor scores for eligible tickers using only prior data.
          2. Assign tickers to ``n_quantiles`` equal-count quantile buckets.
          3. Enter positions at next trading day's close.
          4. Hold until next rebalance execution date.
          5. Deduct transaction costs based on portfolio turnover.

        Args:
            factor_name: One of "mom_12_1", "mom_6_1", "residual_mom".
            n_quantiles: Number of quantile buckets (default 5).

        Returns:
            pd.DataFrame indexed by trading date with columns
            ["q1", "q2", …, "q{n_quantiles}", "long_short"] containing
            daily gross returns (before transaction cost spread).
            Also stores turnover series in ``self.turnover_`` after the run.
        """
        if factor_name not in _FACTOR_FUNCS and factor_name != "residual_mom":
            raise ValueError(f"Unknown factor: {factor_name}")

        rebalance_dates = sorted(self.universe_filter.keys())
        all_daily_returns: list[pd.DataFrame] = []
        turnover_records: list[tuple[pd.Timestamp, float]] = []

        # Previous weights for turnover calculation: {q: {ticker: weight}}
        prev_long_weights: dict[str, float] = {}
        prev_short_weights: dict[str, float] = {}

        tc_rate = self.config.TRANSACTION_COST_BPS / 10_000.0  # convert bps to decimal

        logger.info("Running backtest: factor=%s, n_quantiles=%d", factor_name, n_quantiles)

        for i, reb_date in enumerate(tqdm(rebalance_dates, desc=f"Backtest [{factor_name}]")):
            eligible = self.universe_filter.get(reb_date, [])
            if len(eligible) < 2 * n_quantiles:
                continue

            # ── Signal computation: strictly before reb_date ─────────────────
            scores = self._compute_factor_scores(factor_name, reb_date, eligible)
            scores = scores.dropna()

            if len(scores) < 2 * n_quantiles:
                continue

            # ── Quantile assignment ───────────────────────────────────────────
            q_weights = get_all_quantile_weights(scores, n_quantiles=n_quantiles)

            # ── Execution date: NEXT trading day after reb_date ──────────────
            # This enforces that we do not trade at the signal price.
            future_days = self._trading_days[self._trading_days > reb_date]
            if len(future_days) == 0:
                continue
            exec_date = future_days[0]

            # ── End of holding period: execution date of NEXT rebalance ──────
            if i + 1 < len(rebalance_dates):
                next_reb = rebalance_dates[i + 1]
                next_future = self._trading_days[self._trading_days > next_reb]
                end_date = next_future[0] if len(next_future) > 0 else self._trading_days[-1]
            else:
                end_date = self._trading_days[-1] + pd.Timedelta(days=1)

            # ── Compute daily returns for each quantile ───────────────────────
            period_dfs: dict[str, pd.Series] = {}
            for q_idx in range(n_quantiles):
                col_name = f"q{q_idx + 1}"
                w = q_weights.get(q_idx, {})
                period_dfs[col_name] = self._daily_returns(w, exec_date, end_date)

            # ── Long–short portfolio ──────────────────────────────────────────
            top_q = n_quantiles - 1
            bot_q = 0
            long_w = q_weights.get(top_q, {})
            short_w = q_weights.get(bot_q, {})

            long_ret = self._daily_returns(long_w, exec_date, end_date)
            short_ret = self._daily_returns(short_w, exec_date, end_date)

            # Align on common dates
            common_idx = long_ret.index.intersection(short_ret.index)
            ls_ret = long_ret.reindex(common_idx) - short_ret.reindex(common_idx)

            # ── Transaction costs ─────────────────────────────────────────────
            # Long leg
            long_turnover = self._compute_turnover(prev_long_weights, long_w)
            short_turnover = self._compute_turnover(prev_short_weights, short_w)
            total_tc = (long_turnover + short_turnover) * tc_rate  # total cost

            # Deduct TC from the first day of the holding period
            if len(ls_ret) > 0:
                ls_ret.iloc[0] -= total_tc

            turnover_records.append((reb_date, long_turnover + short_turnover))
            prev_long_weights = long_w
            prev_short_weights = short_w

            # ── Combine into period DataFrame ─────────────────────────────────
            # Determine common date range across all quantiles
            all_qs = list(period_dfs.values()) + [ls_ret]
            if not all_qs or all(s.empty for s in all_qs):
                continue

            common = all_qs[0].index
            for s in all_qs[1:]:
                common = common.intersection(s.index)

            period_df = pd.DataFrame(
                {col: period_dfs[col].reindex(common) for col in period_dfs},
                index=common,
            )
            period_df["long_short"] = ls_ret.reindex(common)
            all_daily_returns.append(period_df)

        if not all_daily_returns:
            logger.warning("No return data generated for factor %s.", factor_name)
            return pd.DataFrame()

        result = pd.concat(all_daily_returns).sort_index()
        # Remove any duplicate dates (overlap at rebalance boundaries)
        result = result[~result.index.duplicated(keep="first")]

        # Store turnover series as instance attribute
        self.turnover_ = pd.Series(
            {d: v for d, v in turnover_records},
            name="turnover",
        )

        logger.info(
            "Backtest complete: %d trading days, factor=%s.",
            len(result),
            factor_name,
        )
        return result

    def run_sector_neutral(
        self,
        factor_name: str,
        n_quantiles: int = 5,
    ) -> pd.DataFrame:
        """Run backtest with within-sector quantile sorting (sector-neutral).

        Identical to ``run`` except tickers are sorted into quantiles *within*
        each GICS sector. The final long leg is an equal-weight average across
        all sector long portfolios (and similarly for short). This eliminates
        sector timing bets.

        Args:
            factor_name: One of "mom_12_1", "mom_6_1", "residual_mom".
            n_quantiles: Number of quantile buckets per sector (default 5).

        Returns:
            pd.DataFrame identical in structure to ``run``'s return value.
        """
        rebalance_dates = sorted(self.universe_filter.keys())
        all_daily_returns: list[pd.DataFrame] = []
        turnover_records: list[tuple[pd.Timestamp, float]] = []

        prev_long_weights: dict[str, float] = {}
        prev_short_weights: dict[str, float] = {}
        tc_rate = self.config.TRANSACTION_COST_BPS / 10_000.0

        logger.info(
            "Running sector-neutral backtest: factor=%s, n_quantiles=%d",
            factor_name,
            n_quantiles,
        )

        for i, reb_date in enumerate(tqdm(rebalance_dates, desc=f"SN Backtest [{factor_name}]")):
            eligible = self.universe_filter.get(reb_date, [])
            if len(eligible) < 2 * n_quantiles:
                continue

            scores = self._compute_factor_scores(factor_name, reb_date, eligible)
            scores = scores.dropna()

            if len(scores) < 2 * n_quantiles:
                continue

            # ── Within-sector quantile sorting ────────────────────────────────
            long_tickers_all: list[str] = []
            short_tickers_all: list[str] = []

            sectors = {self.sector_map.get(t, "Unknown") for t in scores.index}

            for sector in sectors:
                sector_tickers = [
                    t for t in scores.index
                    if self.sector_map.get(t, "Unknown") == sector
                ]
                sector_scores = scores.loc[sector_tickers]
                if len(sector_scores) < 2 * n_quantiles:
                    continue
                try:
                    labels = pd.qcut(sector_scores, q=n_quantiles, labels=False, duplicates="drop")
                except Exception:
                    continue
                long_tickers_all.extend(labels[labels == labels.max()].index.tolist())
                short_tickers_all.extend(labels[labels == labels.min()].index.tolist())

            if not long_tickers_all or not short_tickers_all:
                continue

            # Equal-weight across sectors
            long_w = {t: 1.0 / len(long_tickers_all) for t in long_tickers_all}
            short_w = {t: 1.0 / len(short_tickers_all) for t in short_tickers_all}

            # Execution date
            future_days = self._trading_days[self._trading_days > reb_date]
            if len(future_days) == 0:
                continue
            exec_date = future_days[0]

            if i + 1 < len(rebalance_dates):
                next_reb = rebalance_dates[i + 1]
                next_future = self._trading_days[self._trading_days > next_reb]
                end_date = next_future[0] if len(next_future) > 0 else self._trading_days[-1]
            else:
                end_date = self._trading_days[-1] + pd.Timedelta(days=1)

            long_ret = self._daily_returns(long_w, exec_date, end_date)
            short_ret = self._daily_returns(short_w, exec_date, end_date)
            common_idx = long_ret.index.intersection(short_ret.index)
            ls_ret = long_ret.reindex(common_idx) - short_ret.reindex(common_idx)

            long_turnover = self._compute_turnover(prev_long_weights, long_w)
            short_turnover = self._compute_turnover(prev_short_weights, short_w)
            total_tc = (long_turnover + short_turnover) * tc_rate

            if len(ls_ret) > 0:
                ls_ret.iloc[0] -= total_tc

            turnover_records.append((reb_date, long_turnover + short_turnover))
            prev_long_weights = long_w
            prev_short_weights = short_w

            # Build placeholder quantile columns (full sector-neutral version
            # does not split into individual quantiles, so we only have long_short)
            period_df = pd.DataFrame({"long_short": ls_ret}, index=ls_ret.index)
            for q in range(n_quantiles):
                period_df[f"q{q + 1}"] = np.nan
            all_daily_returns.append(period_df)

        if not all_daily_returns:
            logger.warning("No return data generated for sector-neutral factor %s.", factor_name)
            return pd.DataFrame()

        result = pd.concat(all_daily_returns).sort_index()
        result = result[~result.index.duplicated(keep="first")]

        self.turnover_sn_ = pd.Series(
            {d: v for d, v in turnover_records},
            name="turnover_sn",
        )

        logger.info(
            "Sector-neutral backtest complete: %d trading days, factor=%s.",
            len(result),
            factor_name,
        )
        return result
