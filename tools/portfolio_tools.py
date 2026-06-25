"""
Portfolio analytics tools for Portfolio Sentinel.

Design principles
-----------------
These five functions are the only entry points the Groq language model may
invoke. The design is intentionally restrictive:

* **Typed arguments only.** The model passes ``tickers`` (list of strings),
  ``weights`` (list of floats), ``lookback_days`` (int), etc. It never sends
  free-form SQL, Python expressions, or arbitrary queries. This keeps the
  attack surface narrow and the audit trail simple.

* **Echo inputs in outputs.** Every return dict repeats the inputs that were
  used, so the model can quote them accurately without having to infer them
  from context.

* **Return denominators alongside ratios.** A Sharpe ratio without its
  component volatility and return is unverifiable. Every ratio is accompanied
  by its numerator and denominator where practical.

* **Business logic in Python, not in prompts.** Formulae, thresholds
  (``concentration_flag``), and grain definitions live here. The system prompt
  instructs the model to never invent numbers; numbers come only from tool
  results.

* **No yfinance imports.** All market data flows through
  ``tools.market_data``. This enforces the caching and fallback contract
  defined in that module.

Grain definitions
-----------------
* **Price row:** one calendar date → one float representing the adjusted
  closing price for that date. Only trading days are present; weekends and
  holidays are absent from the series.

* **Daily return:** ``r_t = (P_t / P_{t-1}) - 1``, computed via
  ``pd.Series.pct_change()``. The first row is NaN and is dropped before any
  downstream calculation.

* **Portfolio return:** ``r_portfolio_t = sum(w_i * r_i_t)`` for all
  tickers ``i`` in the portfolio, where weights are normalised to sum to 1.
  Only dates present in all ticker series are used (inner join).

* **Trading day:** any date with a non-NaN adjusted-close price returned by
  Yahoo Finance. The count of trading days is always derived from the aligned
  price matrix, not from a calendar formula.

Default universe
----------------
Only tickers for which ``get_prices`` returns a non-empty series are
included. Alignment is performed on the *intersection* (inner join) of all
ticker date indexes so that missing data for one ticker does not silently
produce NaN portfolio returns.

Exception paths
---------------
* ``get_as_of_snapshot`` bypasses the in-memory price cache because it must
  slice the price history to a specific historical end date. The cache stores
  the most-recent N days of history and cannot be re-parameterised by end date
  without storing multiple copies per ticker. This function therefore calls
  yfinance (or the fallback) with a fixed 1095-day lookback and slices
  in-memory.

* Tools that raise ``ValueError`` (e.g., fewer than 20 trading days) surface
  the error message as the tool result string so the agent can relay it to the
  user rather than crashing.
"""

from __future__ import annotations

from datetime import datetime, date
from typing import Any

import numpy as np
import pandas as pd

from tools.market_data import get_prices, get_info

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR: int = 252
RISK_FREE_RATE_ANNUAL: float = 0.04
BENCHMARK_TICKER: str = "^GSPC"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _normalise_weights(weights: list[float]) -> list[float]:
    """Return weights scaled to sum to 1.0.

    Parameters
    ----------
    weights:
        Raw portfolio weights. May be unnormalised (e.g. 25, 20, 25 rather
        than 0.25, 0.20, 0.25). All values must be non-negative.

    Returns
    -------
    list[float]
        Weights normalised so their sum equals 1.0, preserving relative
        proportions.

    Raises
    ------
    ValueError
        If any weight is strictly negative, or if the sum of all weights is
        zero.
    """
    if any(w < 0 for w in weights):
        raise ValueError(
            "All portfolio weights must be non-negative. "
            f"Received: {weights}"
        )
    total = sum(weights)
    if total == 0:
        raise ValueError("Weights must not all be zero.")
    return [w / total for w in weights]


def _portfolio_returns(
    tickers: list[str],
    weights: list[float],
    lookback_days: int,
) -> pd.Series:
    """Compute the daily weighted portfolio return series.

    Parameters
    ----------
    tickers:
        List of Yahoo Finance ticker symbols.
    weights:
        Normalised portfolio weights (must already sum to 1.0).
    lookback_days:
        Calendar-day lookback passed to ``get_prices``.

    Returns
    -------
    pd.Series
        Daily portfolio returns with a ``DatetimeIndex``, named ``"portfolio"``.
        Only dates present in **all** ticker series are included (inner join).
    """
    price_frames: list[pd.Series] = []
    for ticker in tickers:
        price_frames.append(get_prices(ticker, lookback_days).rename(ticker))

    prices_df = pd.concat(price_frames, axis=1, join="inner")
    daily_returns = prices_df.pct_change().dropna()

    portfolio_ret = pd.Series(
        np.dot(daily_returns.values, weights),
        index=daily_returns.index,
        name="portfolio",
    )
    return portfolio_ret


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------


def get_portfolio_summary(
    tickers: list[str],
    weights: list[float],
    lookback_days: int = 365,
) -> dict[str, Any]:
    """Summarise overall portfolio performance over a lookback window.

    Computes total return, annualised return, benchmark return (S&P 500), and
    active return for the specified portfolio.

    GRAIN: One row per trading day. Aggregated to a single scalar for the full
    window.

    Parameters
    ----------
    tickers:
        List of Yahoo Finance ticker symbols (e.g. ``["AAPL", "MSFT"]``).
    weights:
        Portfolio allocation weights. Will be normalised to sum to 1.0. Must
        be the same length as ``tickers``.
    lookback_days:
        Number of calendar days of history to use. Default is 365.

    Returns
    -------
    dict
        Keys: ``tickers``, ``weights_normalised``, ``lookback_days``,
        ``total_return``, ``annualised_return``, ``benchmark_ticker``,
        ``benchmark_return``, ``active_return``, ``start_date``, ``end_date``,
        ``trading_days``.

    Raises
    ------
    ValueError
        If ``tickers`` and ``weights`` have different lengths.
    """
    if len(tickers) != len(weights):
        raise ValueError(
            f"tickers and weights must have the same length. "
            f"Got {len(tickers)} tickers and {len(weights)} weights."
        )

    norm_weights = _normalise_weights(weights)
    port_returns = _portfolio_returns(tickers, norm_weights, lookback_days)
    n = len(port_returns)

    total_return: float = float((1 + port_returns).prod() - 1)
    annualised_return: float = float((1 + total_return) ** (TRADING_DAYS_PER_YEAR / n) - 1)

    bench_prices = get_prices(BENCHMARK_TICKER, lookback_days)
    bench_returns = bench_prices.pct_change().dropna()
    bench_returns_aligned = bench_returns.reindex(port_returns.index).dropna()
    bench_total: float = float((1 + bench_returns_aligned).prod() - 1)
    bench_n = len(bench_returns_aligned)
    bench_annualised: float = float((1 + bench_total) ** (TRADING_DAYS_PER_YEAR / bench_n) - 1) if bench_n > 0 else 0.0

    active_return: float = float(annualised_return - bench_annualised)

    return {
        "tickers": tickers,
        "weights_normalised": norm_weights,
        "lookback_days": lookback_days,
        "total_return": total_return,
        "annualised_return": annualised_return,
        "benchmark_ticker": BENCHMARK_TICKER,
        "benchmark_return": bench_annualised,
        "active_return": active_return,
        "start_date": port_returns.index[0].strftime("%Y-%m-%d"),
        "end_date": port_returns.index[-1].strftime("%Y-%m-%d"),
        "trading_days": n,
    }


def get_risk_metrics(
    tickers: list[str],
    weights: list[float],
    lookback_days: int = 365,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Compute portfolio risk metrics over a lookback window.

    Calculates annualised volatility, Sharpe ratio, maximum drawdown,
    Value-at-Risk, and Conditional Value-at-Risk (Expected Shortfall).

    GRAIN: One row per trading day. Risk statistics are computed over the full
    aligned return series.

    Parameters
    ----------
    tickers:
        List of Yahoo Finance ticker symbols.
    weights:
        Portfolio allocation weights. Will be normalised to sum to 1.0.
    lookback_days:
        Number of calendar days of history to use. Default is 365.
    confidence_level:
        Quantile for VaR and CVaR computation (e.g. 0.95 means 5th-percentile
        loss). Default is 0.95.

    Returns
    -------
    dict
        Keys: ``tickers``, ``weights_normalised``, ``lookback_days``,
        ``confidence_level``, ``volatility_annual``, ``sharpe_ratio``,
        ``max_drawdown``, ``var_95``, ``cvar_95``, ``risk_free_rate_used``,
        ``trading_days``.

    Raises
    ------
    ValueError
        If fewer than 20 trading days of aligned data are available.
    """
    if len(tickers) != len(weights):
        raise ValueError(
            f"tickers and weights must have the same length. "
            f"Got {len(tickers)} tickers and {len(weights)} weights."
        )

    norm_weights = _normalise_weights(weights)
    port_returns = _portfolio_returns(tickers, norm_weights, lookback_days)
    n = len(port_returns)

    if n < 20:
        raise ValueError(
            f"Insufficient trading data: only {n} trading days available. "
            f"At least 20 are required for reliable risk estimates."
        )

    vol_daily: float = float(port_returns.std())
    vol_annual: float = float(vol_daily * np.sqrt(TRADING_DAYS_PER_YEAR))

    total_return: float = float((1 + port_returns).prod() - 1)
    ann_return: float = float((1 + total_return) ** (TRADING_DAYS_PER_YEAR / n) - 1)
    sharpe: float = float((ann_return - RISK_FREE_RATE_ANNUAL) / vol_annual) if vol_annual > 0 else 0.0

    cumulative = (1 + port_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1
    max_dd: float = float(drawdown.min())

    var_quantile = 1.0 - confidence_level
    var_95: float = float(np.percentile(port_returns.values, var_quantile * 100))
    cvar_95: float = float(port_returns[port_returns <= var_95].mean())

    return {
        "tickers": tickers,
        "weights_normalised": norm_weights,
        "lookback_days": lookback_days,
        "confidence_level": confidence_level,
        "volatility_annual": vol_annual,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "risk_free_rate_used": RISK_FREE_RATE_ANNUAL,
        "trading_days": n,
    }


def get_correlation_matrix(
    tickers: list[str],
    lookback_days: int = 365,
) -> dict[str, Any]:
    """Compute the Pearson correlation matrix of daily returns.

    GRAIN: One row per trading day. Correlations are computed over the inner
    join of all ticker return series.

    Parameters
    ----------
    tickers:
        List of Yahoo Finance ticker symbols. At least two tickers are
        recommended for a meaningful correlation matrix.
    lookback_days:
        Number of calendar days of history to use. Default is 365.

    Returns
    -------
    dict
        Keys: ``tickers``, ``lookback_days``, ``matrix`` (list of lists,
        row-major, same order as ``tickers``), ``avg_pairwise_correlation``,
        ``trading_days``.
    """
    price_frames: list[pd.Series] = []
    for ticker in tickers:
        price_frames.append(get_prices(ticker, lookback_days).rename(ticker))

    prices_df = pd.concat(price_frames, axis=1, join="inner")
    daily_returns = prices_df.pct_change().dropna()
    corr_matrix = daily_returns.corr()
    n = len(daily_returns)

    matrix_list: list[list[float]] = [
        [float(corr_matrix.iloc[i, j]) for j in range(len(tickers))]
        for i in range(len(tickers))
    ]

    n_tickers = len(tickers)
    off_diag_vals: list[float] = [
        matrix_list[i][j]
        for i in range(n_tickers)
        for j in range(n_tickers)
        if i != j
    ]
    avg_pairwise: float = float(np.mean(off_diag_vals)) if off_diag_vals else 0.0

    return {
        "tickers": tickers,
        "lookback_days": lookback_days,
        "matrix": matrix_list,
        "avg_pairwise_correlation": avg_pairwise,
        "trading_days": n,
    }


def get_sector_exposure(
    tickers: list[str],
    weights: list[float],
) -> dict[str, Any]:
    """Compute portfolio weight allocation broken down by GICS sector.

    Fetches sector metadata from Yahoo Finance info for each ticker, groups
    tickers by sector, and sums their normalised weights.

    GRAIN: One row per ticker. Aggregated to one row per sector.

    Parameters
    ----------
    tickers:
        List of Yahoo Finance ticker symbols.
    weights:
        Portfolio allocation weights. Will be normalised to sum to 1.0.

    Returns
    -------
    dict
        Keys: ``tickers``, ``weights_normalised``, ``sectors`` (list of dicts
        sorted by ``weight_in_portfolio`` descending, each with
        ``sector_name``, ``weight_in_portfolio``, ``tickers_in_sector``),
        ``dominant_sector``, ``dominant_sector_weight``,
        ``concentration_flag`` (``True`` if any single sector exceeds 50% of
        the portfolio).
    """
    if len(tickers) != len(weights):
        raise ValueError(
            f"tickers and weights must have the same length. "
            f"Got {len(tickers)} tickers and {len(weights)} weights."
        )

    norm_weights = _normalise_weights(weights)

    sector_map: dict[str, list[tuple[str, float]]] = {}
    for ticker, w in zip(tickers, norm_weights):
        info = get_info(ticker)
        sector = info.get("sector") or "Unknown"
        sector_map.setdefault(sector, []).append((ticker, w))

    sectors_list: list[dict] = []
    for sector_name, ticker_weights in sector_map.items():
        sector_weight = sum(w for _, w in ticker_weights)
        sector_tickers = sorted(
            [{"ticker": t, "weight": float(w)} for t, w in ticker_weights],
            key=lambda x: x["weight"],
            reverse=True,
        )
        sectors_list.append(
            {
                "sector_name": sector_name,
                "weight_in_portfolio": float(sector_weight),
                "tickers_in_sector": sector_tickers,
            }
        )

    sectors_list.sort(key=lambda x: x["weight_in_portfolio"], reverse=True)

    dominant = sectors_list[0] if sectors_list else {"sector_name": "Unknown", "weight_in_portfolio": 0.0}
    concentration_flag: bool = any(s["weight_in_portfolio"] > 0.5 for s in sectors_list)

    return {
        "tickers": tickers,
        "weights_normalised": norm_weights,
        "sectors": sectors_list,
        "dominant_sector": dominant["sector_name"],
        "dominant_sector_weight": dominant["weight_in_portfolio"],
        "concentration_flag": concentration_flag,
    }


def get_as_of_snapshot(
    tickers: list[str],
    weights: list[float],
    as_of_date: str,
) -> dict[str, Any]:
    """Compute portfolio performance metrics as of a specific historical date.

    EXCEPTION PATH: This function bypasses the in-memory price cache. The
    cache stores the most-recent N days of prices and cannot represent an
    arbitrary historical end date without storing multiple copies per ticker.
    To compute a point-in-time snapshot, this function fetches 1095 days of
    history (approximately 3 years) and then slices to ``as_of_date``.

    GRAIN: One row per trading day up to and including ``as_of_date``.
    Aggregated to a single scalar for the window.

    Parameters
    ----------
    tickers:
        List of Yahoo Finance ticker symbols.
    weights:
        Portfolio allocation weights. Will be normalised to sum to 1.0.
    as_of_date:
        Historical end date in ``YYYY-MM-DD`` format. Must not be in the
        future relative to today.

    Returns
    -------
    dict
        Same keys as ``get_portfolio_summary`` plus ``as_of_date`` echoed.
        Keys: ``tickers``, ``weights_normalised``, ``as_of_date``,
        ``total_return``, ``annualised_return``, ``benchmark_ticker``,
        ``benchmark_return``, ``active_return``, ``start_date``, ``end_date``,
        ``trading_days``.

    Raises
    ------
    ValueError
        If ``as_of_date`` cannot be parsed as ``YYYY-MM-DD``, or if it
        represents a future date, or if no price data is available up to that
        date.
    """
    if len(tickers) != len(weights):
        raise ValueError(
            f"tickers and weights must have the same length. "
            f"Got {len(tickers)} tickers and {len(weights)} weights."
        )

    try:
        snapshot_dt = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(
            f"as_of_date '{as_of_date}' is not in YYYY-MM-DD format."
        )

    if snapshot_dt > date.today():
        raise ValueError(
            f"as_of_date '{as_of_date}' is in the future. "
            f"Only historical dates are supported."
        )

    norm_weights = _normalise_weights(weights)

    LOOKBACK_FULL = 1095
    price_frames: list[pd.Series] = []
    for ticker in tickers:
        prices = get_prices(ticker, LOOKBACK_FULL)
        sliced = prices.loc[:as_of_date]
        if sliced.empty:
            raise ValueError(
                f"No price data available for '{ticker}' up to {as_of_date}."
            )
        price_frames.append(sliced.rename(ticker))

    prices_df = pd.concat(price_frames, axis=1, join="inner")
    daily_returns = prices_df.pct_change().dropna()
    n = len(daily_returns)

    port_returns = pd.Series(
        np.dot(daily_returns.values, norm_weights),
        index=daily_returns.index,
        name="portfolio",
    )

    total_return: float = float((1 + port_returns).prod() - 1)
    ann_return: float = float((1 + total_return) ** (TRADING_DAYS_PER_YEAR / n) - 1) if n > 0 else 0.0

    bench_prices = get_prices(BENCHMARK_TICKER, LOOKBACK_FULL)
    bench_sliced = bench_prices.loc[:as_of_date]
    bench_returns = bench_sliced.pct_change().dropna()
    bench_aligned = bench_returns.reindex(port_returns.index).dropna()
    bench_total: float = float((1 + bench_aligned).prod() - 1)
    bench_n = len(bench_aligned)
    bench_ann: float = float((1 + bench_total) ** (TRADING_DAYS_PER_YEAR / bench_n) - 1) if bench_n > 0 else 0.0

    return {
        "tickers": tickers,
        "weights_normalised": norm_weights,
        "as_of_date": as_of_date,
        "total_return": total_return,
        "annualised_return": ann_return,
        "benchmark_ticker": BENCHMARK_TICKER,
        "benchmark_return": bench_ann,
        "active_return": float(ann_return - bench_ann),
        "start_date": port_returns.index[0].strftime("%Y-%m-%d"),
        "end_date": port_returns.index[-1].strftime("%Y-%m-%d"),
        "trading_days": n,
    }
