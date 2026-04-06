"""
portfolio.py
------------
Portfolio construction utilities: equal-weighted and risk-parity long–short
quantile portfolios.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def construct_long_short(
    factor_scores: pd.Series,
    n_quantiles: int = 5,
    weighting: str = "equal",
) -> dict[str, dict[str, float]]:
    """Build equal-weighted long–short portfolio weights from factor scores.

    Tickers are ranked by their factor score and divided into ``n_quantiles``
    equal-count buckets. The **top** quantile is long and the **bottom**
    quantile is short, each with equal weights summing to 1.0.

    Args:
        factor_scores: pd.Series indexed by ticker with factor signal values.
            NaN entries are dropped before ranking.
        n_quantiles: Number of equal-count quantile buckets (default 5).
        weighting: Currently only "equal" is supported.

    Returns:
        Dictionary with keys "long" and "short", each mapping to a
        ``{ticker: weight}`` dictionary. Weights on each side sum to 1.0.

    Raises:
        ValueError: If fewer than ``2 * n_quantiles`` valid tickers exist.
    """
    valid = factor_scores.dropna()
    if len(valid) < 2 * n_quantiles:
        raise ValueError(
            f"Insufficient tickers ({len(valid)}) for {n_quantiles} quantiles."
        )

    # Rank-based quantile assignment (ties broken by average)
    quantile_labels = pd.qcut(valid, q=n_quantiles, labels=False, duplicates="drop")

    top_q = quantile_labels.max()  # highest label = top quantile
    bot_q = quantile_labels.min()  # lowest label  = bottom quantile

    long_tickers = quantile_labels[quantile_labels == top_q].index.tolist()
    short_tickers = quantile_labels[quantile_labels == bot_q].index.tolist()

    if weighting == "equal":
        long_w = {t: 1.0 / len(long_tickers) for t in long_tickers}
        short_w = {t: 1.0 / len(short_tickers) for t in short_tickers}
    else:
        raise NotImplementedError(f"Weighting scheme '{weighting}' not implemented.")

    return {"long": long_w, "short": short_w}


def construct_long_short_optimized(
    factor_scores: pd.Series,
    cov_matrix: pd.DataFrame,
    target_vol: float = 0.10,
) -> dict[str, dict[str, float]]:
    """Build a mean-variance optimised long–short portfolio.

    Maximises the expected return (proportional to factor score) subject to:
      - Portfolio volatility (annualised) ≤ ``target_vol``
      - Long-side gross exposure = 1.0
      - Short-side gross exposure = 1.0
      - All long weights ≥ 0, all short weights ≥ 0 (no shorting on long side,
        no longing on short side)

    The net portfolio is: long_weight - short_weight.

    Args:
        factor_scores: pd.Series indexed by ticker with factor signal values.
            NaN entries are dropped. Used as expected return proxy.
        cov_matrix: Square covariance matrix of daily returns (tickers × tickers).
            Must cover at least the tickers in ``factor_scores``.
        target_vol: Maximum annualised portfolio volatility (default 0.10 = 10%).

    Returns:
        Dictionary with keys "long" and "short", each mapping to a
        ``{ticker: weight}`` dictionary.
    """
    valid = factor_scores.dropna()
    tickers = valid.index.tolist()

    # Restrict cov_matrix to available tickers
    available = [t for t in tickers if t in cov_matrix.index and t in cov_matrix.columns]
    if len(available) < 4:
        logger.warning("Too few tickers for optimisation; falling back to equal-weight.")
        return construct_long_short(valid, n_quantiles=5, weighting="equal")

    mu = valid.loc[available].values            # shape (n,)
    cov = cov_matrix.loc[available, available].values  # shape (n, n)
    n = len(available)

    # Decision variable: w = [w_long (n,), w_short (n,)]
    # Net weights: w_long - w_short
    # Objective: -( mu @ (w_long - w_short) )  ← maximise expected return
    def neg_return(w: np.ndarray) -> float:
        w_net = w[:n] - w[n:]
        return -float(mu @ w_net)

    # Annual vol constraint: sqrt(252 * w_net' cov w_net) <= target_vol
    def vol_constraint(w: np.ndarray) -> float:
        w_net = w[:n] - w[n:]
        variance = float(w_net @ cov @ w_net) * 252
        return target_vol**2 - variance  # ≥ 0 means constraint satisfied

    constraints = [
        {"type": "ineq", "fun": vol_constraint},
        # Long gross exposure = 1.0
        {"type": "eq", "fun": lambda w: np.sum(w[:n]) - 1.0},
        # Short gross exposure = 1.0
        {"type": "eq", "fun": lambda w: np.sum(w[n:]) - 1.0},
    ]

    # Bounds: all weights ≥ 0
    bounds = [(0.0, 1.0)] * (2 * n)

    w0 = np.concatenate([np.ones(n) / n, np.ones(n) / n])
    result = minimize(
        neg_return,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    if not result.success:
        logger.warning("Optimisation did not converge: %s. Falling back to equal-weight.", result.message)
        return construct_long_short(valid, n_quantiles=5, weighting="equal")

    w_long = result.x[:n]
    w_short = result.x[n:]

    long_dict = {t: float(w_long[i]) for i, t in enumerate(available) if w_long[i] > 1e-6}
    short_dict = {t: float(w_short[i]) for i, t in enumerate(available) if w_short[i] > 1e-6}

    return {"long": long_dict, "short": short_dict}


def get_all_quantile_weights(
    factor_scores: pd.Series,
    n_quantiles: int = 5,
) -> dict[int, dict[str, float]]:
    """Assign equal weights within each of ``n_quantiles`` quantile buckets.

    Args:
        factor_scores: pd.Series indexed by ticker with factor signal values.
            NaN entries are dropped before ranking.
        n_quantiles: Number of equal-count quantile buckets (default 5).

    Returns:
        Dictionary mapping quantile index (0 = lowest, n_quantiles-1 = highest)
        to a ``{ticker: weight}`` dictionary where weights within each quantile
        sum to 1.0.
    """
    valid = factor_scores.dropna()
    if len(valid) < n_quantiles:
        return {q: {} for q in range(n_quantiles)}

    quantile_labels = pd.qcut(valid, q=n_quantiles, labels=False, duplicates="drop")
    result: dict[int, dict[str, float]] = {}

    for q in range(n_quantiles):
        members = quantile_labels[quantile_labels == q].index.tolist()
        if members:
            result[q] = {t: 1.0 / len(members) for t in members}
        else:
            result[q] = {}

    return result
