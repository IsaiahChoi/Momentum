"""
config.py
---------
Global configuration constants for the cross-sectional momentum factor library.

NOTE ON SURVIVORSHIP BIAS
--------------------------
The tickers below were selected from companies that were S&P 500 constituents
*before* 2005 and remain publicly traded. This approach introduces **residual
survivorship bias**: firms that delisted, went bankrupt, or were acquired
between 2005 and 2024 are under-represented. As a result, backtest returns
may be overstated relative to a fully survivorship-bias-free universe.
For production-grade research, use a point-in-time index membership database
(e.g., Compustat or Bloomberg's historical constituent data).
"""

# ── Date range ────────────────────────────────────────────────────────────────
START_DATE: str = "2005-01-01"
END_DATE: str = "2024-12-31"

# ── Market data ───────────────────────────────────────────────────────────────
# 13-week T-bill (annualized yield, used as risk-free rate proxy)
RISK_FREE_TICKER: str = "^IRX"
BENCHMARK: str = "SPY"

# ── Transaction costs ─────────────────────────────────────────────────────────
# One-way cost in basis points applied to gross turnover on each rebalance date
TRANSACTION_COST_BPS: int = 10  # 10 bps one-way

# ── Rebalancing ───────────────────────────────────────────────────────────────
# "ME" = last trading day of each calendar month (pandas >= 2.2; old alias was "M")
REBALANCE_FREQ: str = "ME"  # pandas >= 2.2 uses "ME" (month-end) instead of deprecated "M"

# ── Universe ─────────────────────────────────────────────────────────────────
# ~150 large-cap US equities that were S&P 500 members before 2005.
# RESIDUAL SURVIVORSHIP BIAS WARNING: see module docstring above.
UNIVERSE_TICKERS: list[str] = [
    # Technology
    "AAPL", "MSFT", "IBM", "INTC", "TXN", "ADI", "AMAT", "KLAC", "LRCX",
    "MU", "HPQ", "GLW", "CSCO", "QCOM", "AVGO", "MCHP", "XLNX",
    # Communication Services
    "T", "VZ", "CMCSA", "DIS", "NFLX", "FOXA", "OMC", "IPG",
    # Consumer Discretionary
    "AMZN", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "BBY", "F", "GM",
    "WHR", "HAS", "MAT", "YUM", "DRI", "MHK",
    # Consumer Staples
    "PG", "KO", "PEP", "MO", "PM", "CL", "KMB", "GIS", "K", "CPB",
    "HSY", "SJM", "CAG", "TSN", "HRL", "MKC", "CHD",
    # Energy
    "XOM", "CVX", "COP", "SLB", "HAL", "BKR", "PSX", "VLO", "MPC",
    "OXY", "DVN", "EOG", "PXD", "HES",
    # Financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BK", "STT", "USB", "PNC",
    "AXP", "MET", "PRU", "AFL", "TRV", "ALL", "CB", "MMC", "AON", "BLK",
    "SCHW", "CME", "ICE",
    # Health Care
    "JNJ", "PFE", "MRK", "ABT", "MDT", "UNH", "HUM", "CI", "CVS", "MCK",
    "BMY", "AMGN", "GILD", "BIIB", "BAX", "BSX", "SYK", "ZBH", "BDX",
    "CAH", "HCA", "DHR", "IQV",
    # Industrials
    "GE", "HON", "MMM", "BA", "CAT", "DE", "EMR", "ETN", "PH", "ROK",
    "ITW", "DOV", "GD", "LMT", "NOC", "RTX", "FDX", "UPS", "CSX", "NSC",
    "UNP", "WM", "RSG",
    # Materials
    "LIN", "APD", "ECL", "SHW", "PPG", "NEM", "FCX", "NUE", "PKG",
    "IP", "WRK",
    # Real Estate
    "PLD", "AMT", "SPG", "O", "PSA", "EQR", "AVB",
    # Utilities
    "NEE", "DUK", "SO", "AEP", "EXC", "SRE", "D", "PCG", "XEL",
]

# ── Factor parameters ─────────────────────────────────────────────────────────
MOM_12_1_LONG_WINDOW: int = 252   # ~12 months of trading days
MOM_12_1_SKIP: int = 21            # skip most recent ~1 month
MOM_6_1_LONG_WINDOW: int = 126    # ~6 months of trading days
MOM_6_1_SKIP: int = 21
RESIDUAL_MOM_WINDOW: int = 252
RESIDUAL_MOM_SKIP: int = 21

# ── Portfolio construction ────────────────────────────────────────────────────
N_QUANTILES: int = 5
MIN_HISTORY_DAYS: int = 504       # ~2 years of trading days
MIN_PRICE: float = 5.0            # minimum share price filter
