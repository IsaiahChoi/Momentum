"""
cross-sectional-momentum-factor-library
========================================
A library for computing, backtesting, and evaluating classic cross-sectional
equity momentum factors.

Modules
-------
- data_loader : Fetch price data, risk-free rates, and sector maps from yfinance.
- universe    : Apply point-in-time universe filters.
- factors     : Compute momentum factor signals (12-1, 6-1, residual momentum).
- backtester  : Run monthly long–short quantile backtests.
- portfolio   : Construct long–short portfolio weights.
- analytics   : Performance statistics, Fama-French regression, multiple testing.
- utils       : Shared helper functions.
"""

from .data_loader import fetch_price_data, fetch_risk_free_rate, fetch_sector_map
from .universe import apply_universe_filters
from .factors import momentum_12_1, momentum_6_1, residual_momentum, compute_all_factors
from .portfolio import construct_long_short, get_all_quantile_weights
from .backtester import MomentumBacktester
from .analytics import (
    compute_stats,
    compute_rolling_sharpe,
    fama_french_regression,
    multiple_testing_adjustment,
    sub_period_analysis,
)
from .utils import (
    align_series,
    prices_to_returns,
    rolling_cov_matrix,
    winsorise,
    get_last_trading_days,
)

__all__ = [
    "fetch_price_data",
    "fetch_risk_free_rate",
    "fetch_sector_map",
    "apply_universe_filters",
    "momentum_12_1",
    "momentum_6_1",
    "residual_momentum",
    "compute_all_factors",
    "construct_long_short",
    "get_all_quantile_weights",
    "MomentumBacktester",
    "compute_stats",
    "compute_rolling_sharpe",
    "fama_french_regression",
    "multiple_testing_adjustment",
    "sub_period_analysis",
    "align_series",
    "prices_to_returns",
    "rolling_cov_matrix",
    "winsorise",
    "get_last_trading_days",
]
