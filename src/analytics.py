"""
analytics.py
------------
Performance analytics, risk-adjusted statistics, Fama-French regression,
multiple-testing adjustments, and sub-period analysis.
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


# ── Core performance statistics ───────────────────────────────────────────────

def compute_stats(
    returns_series: pd.Series,
    risk_free_series: pd.Series,
    turnover_series: Optional[pd.Series] = None,
) -> dict[str, float]:
    """Compute comprehensive risk-adjusted performance statistics.

    All metrics are computed on the long–short (or any) daily return series.
    Monthly statistics (skewness, kurtosis, hit rate) are derived from
    returns resampled to month-end.

    Args:
        returns_series: Daily return series (decimal, e.g. 0.01 = 1%).
        risk_free_series: Daily risk-free rate series (decimal). Aligned to
            ``returns_series`` by index.
        turnover_series: Optional monthly turnover series (fraction). Used to
            compute average monthly turnover. If None, reported as NaN.

    Returns:
        Dictionary with the following keys:
          - annualized_return (float): Geometric annualized return.
          - annualized_vol (float): Annualized standard deviation.
          - sharpe_ratio (float): Annualized Sharpe ratio.
          - max_drawdown (float): Maximum peak-to-trough drawdown (negative).
          - calmar_ratio (float): Annualized return / |max drawdown|.
          - sortino_ratio (float): Annualized Sortino ratio (MAR = risk-free).
          - skewness (float): Skewness of monthly returns.
          - kurtosis (float): Excess kurtosis of monthly returns.
          - avg_monthly_turnover (float): Mean monthly turnover (or NaN).
          - hit_rate (float): Fraction of months with positive return.
    """
    # Align risk-free to return dates
    rf_aligned = risk_free_series.reindex(returns_series.index).ffill().fillna(0.0)
    excess = returns_series - rf_aligned

    n = len(returns_series)
    if n == 0:
        return {k: np.nan for k in [
            "annualized_return", "annualized_vol", "sharpe_ratio",
            "max_drawdown", "calmar_ratio", "sortino_ratio",
            "skewness", "kurtosis", "avg_monthly_turnover", "hit_rate",
        ]}

    # ── Annualized return (geometric) ─────────────────────────────────────────
    total_ret = (1 + returns_series).prod()
    n_years = n / _TRADING_DAYS_PER_YEAR
    ann_return = total_ret ** (1.0 / n_years) - 1.0

    # ── Annualized volatility ─────────────────────────────────────────────────
    ann_vol = returns_series.std() * np.sqrt(_TRADING_DAYS_PER_YEAR)

    # ── Sharpe ratio ──────────────────────────────────────────────────────────
    ann_excess = excess.mean() * _TRADING_DAYS_PER_YEAR
    sharpe = ann_excess / ann_vol if ann_vol > 0 else np.nan

    # ── Maximum drawdown ──────────────────────────────────────────────────────
    cum_ret = (1 + returns_series).cumprod()
    rolling_max = cum_ret.cummax()
    drawdown = (cum_ret - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # ── Calmar ratio ──────────────────────────────────────────────────────────
    calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan

    # ── Sortino ratio (downside deviation, MAR = daily risk-free) ────────────
    daily_mar = rf_aligned
    downside = returns_series - daily_mar
    downside_neg = downside[downside < 0]
    downside_dev = np.sqrt((downside_neg**2).mean()) * np.sqrt(_TRADING_DAYS_PER_YEAR) if len(downside_neg) > 0 else np.nan
    sortino = ann_excess / downside_dev if downside_dev and downside_dev > 0 else np.nan

    # ── Monthly statistics ────────────────────────────────────────────────────
    monthly = returns_series.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    skew = float(stats.skew(monthly.dropna()))
    kurt = float(stats.kurtosis(monthly.dropna()))  # excess kurtosis
    hit_rate = float((monthly > 0).mean())

    # ── Turnover ──────────────────────────────────────────────────────────────
    avg_turnover = float(turnover_series.mean()) if turnover_series is not None else np.nan

    return {
        "annualized_return": float(ann_return),
        "annualized_vol": float(ann_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "calmar_ratio": float(calmar),
        "sortino_ratio": float(sortino),
        "skewness": float(skew),
        "kurtosis": float(kurt),
        "avg_monthly_turnover": float(avg_turnover),
        "hit_rate": float(hit_rate),
    }


def compute_rolling_sharpe(
    returns_series: pd.Series,
    risk_free_series: pd.Series,
    window: int = 36,
) -> pd.Series:
    """Compute a rolling Sharpe ratio on a monthly return series.

    The input daily returns are first converted to monthly, then a rolling
    ``window``-month Sharpe is computed.

    Args:
        returns_series: Daily return series (decimal).
        risk_free_series: Daily risk-free rate series (decimal).
        window: Rolling window in months (default 36).

    Returns:
        pd.Series of rolling annualised Sharpe ratios indexed by month-end date.
    """
    monthly = returns_series.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    rf_monthly = risk_free_series.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    rf_aligned = rf_monthly.reindex(monthly.index).fillna(0.0)

    excess_monthly = monthly - rf_aligned

    rolling_mean = excess_monthly.rolling(window).mean()
    rolling_std = excess_monthly.rolling(window).std()

    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(12)  # annualise monthly
    rolling_sharpe.name = "rolling_sharpe"
    return rolling_sharpe


# ── Fama-French regression ────────────────────────────────────────────────────

def _load_ff_factors() -> pd.DataFrame:
    """Download and parse Fama-French 5-Factor + Momentum daily data.

    Downloads from Kenneth French's data library:
      - FF 5-Factor daily CSV zip
      - Momentum (UMD) daily CSV zip

    Returns:
        pd.DataFrame with DatetimeIndex and columns
        ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "UMD"]
        in decimal (divided by 100 from raw percentage).
    """
    ff5_url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    )
    mom_url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Momentum_Factor_daily_CSV.zip"
    )

    def _fetch_csv_from_zip(url: str, skip_rows: int) -> pd.DataFrame:
        """Fetch and parse a CSV from a remote ZIP file."""
        logger.info("Downloading FF data from %s …", url)
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
            with zf.open(csv_name) as f:
                raw = f.read().decode("latin-1")

        # Find the header row
        lines = raw.splitlines()
        # Find the line that starts with a date-like entry (8-digit YYYYMMDD)
        start_line = 0
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped and stripped[:8].isdigit() and len(stripped[:8]) == 8:
                start_line = idx
                break

        csv_content = "\n".join(lines[start_line:])
        # Find end of data (some files have footnotes after the last date)
        data_lines = []
        for line in csv_content.splitlines():
            parts = line.strip().split(",")
            if parts and parts[0].strip().isdigit() and len(parts[0].strip()) == 8:
                data_lines.append(line)

        df = pd.read_csv(
            io.StringIO("\n".join(data_lines)),
            header=None,
        )
        # First column is date, rest are factor returns
        return df

    try:
        ff5_raw = _fetch_csv_from_zip(ff5_url, skip_rows=0)
        # Columns: Date, Mkt-RF, SMB, HML, RMW, CMA, RF
        ff5_raw.columns = ["Date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
        ff5_raw["Date"] = pd.to_datetime(ff5_raw["Date"].astype(str), format="%Y%m%d")
        ff5_raw = ff5_raw.set_index("Date")
        ff5_raw = ff5_raw.apply(pd.to_numeric, errors="coerce") / 100.0

        mom_raw = _fetch_csv_from_zip(mom_url, skip_rows=0)
        mom_raw.columns = ["Date", "UMD"]
        mom_raw["Date"] = pd.to_datetime(mom_raw["Date"].astype(str), format="%Y%m%d")
        mom_raw = mom_raw.set_index("Date")
        mom_raw = mom_raw.apply(pd.to_numeric, errors="coerce") / 100.0

        ff_combined = ff5_raw.join(mom_raw, how="inner")
        logger.info("FF factors loaded: %d daily observations.", len(ff_combined))
        return ff_combined

    except Exception as exc:
        logger.error("Failed to load Fama-French data: %s", exc)
        raise


def fama_french_regression(
    returns_series: pd.Series,
    ff_factors_df: Optional[pd.DataFrame] = None,
) -> dict[str, object]:
    """Regress long–short returns on Fama-French 5-factor + Momentum model.

    Uses statsmodels OLS. The alpha is annualised. t-statistics are from the
    HAC (Newey-West) robust standard errors.

    Args:
        returns_series: Daily long–short return series (decimal).
        ff_factors_df: Pre-loaded Fama-French factor DataFrame. If None,
            the function attempts to download it automatically.

    Returns:
        Dictionary with keys:
          - alpha_annualised (float): Annualised regression intercept.
          - alpha_tstat (float): t-statistic of the alpha (HAC robust).
          - factor_loadings (dict): {factor_name: beta_coefficient}.
          - factor_tstats (dict): {factor_name: t-statistic}.
          - r_squared (float): OLS R².
          - n_obs (int): Number of observations.
    """
    if ff_factors_df is None:
        ff_factors_df = _load_ff_factors()

    # Align on common dates
    common_idx = returns_series.index.intersection(ff_factors_df.index)
    if len(common_idx) < 60:
        logger.warning("Insufficient overlap between returns and FF factors.")
        return {}

    y = returns_series.loc[common_idx] - ff_factors_df.loc[common_idx, "RF"]
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]
    available_cols = [c for c in factor_cols if c in ff_factors_df.columns]
    X = sm.add_constant(ff_factors_df.loc[common_idx, available_cols])

    model = sm.OLS(y, X)
    # Newey-West HAC standard errors with 5 lags
    result = model.fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    alpha_daily = result.params["const"]
    alpha_ann = (1 + alpha_daily) ** _TRADING_DAYS_PER_YEAR - 1.0
    alpha_tstat = result.tvalues["const"]

    loadings = {col: float(result.params[col]) for col in available_cols if col in result.params}
    tstats = {col: float(result.tvalues[col]) for col in available_cols if col in result.tvalues}

    return {
        "alpha_annualised": float(alpha_ann),
        "alpha_tstat": float(alpha_tstat),
        "factor_loadings": loadings,
        "factor_tstats": tstats,
        "r_squared": float(result.rsquared),
        "n_obs": int(result.nobs),
    }


# ── Multiple-testing adjustment ───────────────────────────────────────────────

def multiple_testing_adjustment(
    sharpe_ratios_dict: dict[str, float],
    n_obs: int = 252 * 10,
) -> pd.DataFrame:
    """Apply Bonferroni and Benjamini-Hochberg corrections to Sharpe ratio p-values.

    Under the null hypothesis of zero mean, the Sharpe ratio * sqrt(T)
    is approximately t-distributed. We convert each Sharpe to a one-sided
    p-value, then apply multiple testing corrections.

    Args:
        sharpe_ratios_dict: Dictionary {factor_name: annualized_sharpe_ratio}.
        n_obs: Number of daily observations used to estimate each Sharpe
            (default 252 * 10 = 10 years). Used to compute t-statistics.

    Returns:
        pd.DataFrame with columns:
          factor, sharpe, t_stat, p_value_raw,
          p_value_bonferroni, p_value_bh,
          significant_bonferroni (bool), significant_bh (bool)
    """
    factor_names = list(sharpe_ratios_dict.keys())
    sharpes = [sharpe_ratios_dict[f] for f in factor_names]

    # Daily Sharpe ≈ annualized Sharpe / sqrt(252)
    daily_sharpes = [s / np.sqrt(_TRADING_DAYS_PER_YEAR) for s in sharpes]
    t_stats = [s * np.sqrt(n_obs) for s in daily_sharpes]

    # One-sided p-values (H1: Sharpe > 0)
    p_values = [float(1 - stats.t.cdf(t, df=n_obs - 1)) for t in t_stats]

    # Bonferroni
    reject_bonf, p_bonf, _, _ = multipletests(p_values, method="bonferroni", alpha=0.05)
    # Benjamini-Hochberg FDR
    reject_bh, p_bh, _, _ = multipletests(p_values, method="fdr_bh", alpha=0.05)

    df = pd.DataFrame(
        {
            "factor": factor_names,
            "sharpe": sharpes,
            "t_stat": t_stats,
            "p_value_raw": p_values,
            "p_value_bonferroni": list(p_bonf),
            "p_value_bh": list(p_bh),
            "significant_bonferroni": list(reject_bonf),
            "significant_bh": list(reject_bh),
        }
    )
    return df.set_index("factor")


# ── Sub-period analysis ───────────────────────────────────────────────────────

def sub_period_analysis(
    returns_series: pd.Series,
    risk_free_series: pd.Series,
    breakpoints: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Compute performance statistics for distinct sub-periods.

    Splits the return series at the provided breakpoints and computes
    ``compute_stats`` for each sub-period independently.

    Args:
        returns_series: Daily return series (decimal).
        risk_free_series: Daily risk-free rate series (decimal).
        breakpoints: List of date strings (YYYY-MM-DD) defining sub-period
            boundaries. If None, defaults to
            ["2008-06-01", "2010-06-01", "2020-01-01", "2020-06-01"].

    Returns:
        pd.DataFrame where each row corresponds to a sub-period and columns
        are the metrics from ``compute_stats``.
    """
    if breakpoints is None:
        breakpoints = ["2008-06-01", "2010-06-01", "2020-01-01", "2020-06-01"]

    # Build period boundaries
    boundaries = (
        [returns_series.index[0]]
        + [pd.Timestamp(b) for b in breakpoints]
        + [returns_series.index[-1] + pd.Timedelta(days=1)]
    )

    records: list[dict] = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        mask = (returns_series.index >= start) & (returns_series.index < end)
        sub = returns_series.loc[mask]
        if sub.empty:
            continue
        period_label = f"{start.date()} to {(end - pd.Timedelta(days=1)).date()}"
        stats_dict = compute_stats(sub, risk_free_series)
        stats_dict["period"] = period_label
        stats_dict["n_days"] = len(sub)
        records.append(stats_dict)

    result = pd.DataFrame(records).set_index("period")
    return result
