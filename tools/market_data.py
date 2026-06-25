"""
Market data access layer for Portfolio Sentinel.

Caching strategy
----------------
All price and info lookups go through this module. Two layers of caching ensure
that the application is both fast within a session and resilient to upstream
failures.

1. In-memory cache (``_price_cache``, ``_info_cache``)
   A plain dict keyed by ``"TICKER_lookbackdays"`` for prices and ``"TICKER"``
   for info. Populated on first request during a process lifetime. Entries are
   never evicted — the process is assumed to be short-lived (ASGI server).

2. Fallback cache (``_fallback``)
   Loaded once at import time from ``fallback_cache.json`` in the project root,
   if the file exists. Keys follow the same convention as the in-memory cache.
   The fallback is consulted only when the live yfinance call fails. If both the
   live fetch and the fallback are unavailable, a ``ValueError`` is raised with a
   clear message so the caller can surface a meaningful error to the user rather
   than a raw traceback.

Fallback cache format
---------------------
The JSON file is produced by ``scripts/build_fallback_cache.py``. Price entries
are stored under the key ``"prices_TICKER_lookbackdays"`` as::

    {"dates": ["2024-01-02", ...], "values": [185.2, ...]}

Info entries are stored under the key ``"info_TICKER"`` as a flat dict.

Public API
----------
Only two functions are exported. Every other module in the project must import
from here — no other file may import yfinance directly.

- ``get_prices(ticker, lookback_days)`` → ``pd.Series``
- ``get_info(ticker)`` → ``dict``
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_price_cache: dict[str, pd.Series] = {}
_info_cache: dict[str, dict] = {}

_FALLBACK_PATH = Path(__file__).resolve().parent.parent / "fallback_cache.json"
_fallback: dict[str, Any] = {}

if _FALLBACK_PATH.exists():
    try:
        with _FALLBACK_PATH.open("r", encoding="utf-8") as _fh:
            _fallback = json.load(_fh)
    except (json.JSONDecodeError, OSError):
        _fallback = {}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def get_prices(ticker: str, lookback_days: int = 365) -> pd.Series:
    """Return the daily adjusted-close price series for *ticker*.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker symbol (e.g. ``"AAPL"``, ``"CRH"``).
    lookback_days:
        Number of calendar days of history to fetch, counting backwards from
        today. Default is 365.

    Returns
    -------
    pd.Series
        Daily adjusted-close prices with a ``DatetimeIndex``. The series is
        named with the ticker symbol. Only trading days are present (weekends
        and market holidays are absent).

    Raises
    ------
    ValueError
        If yfinance returns empty data and no fallback entry exists for this
        ticker/lookback combination.
    """
    cache_key = f"{ticker}_{lookback_days}"
    if cache_key in _price_cache:
        return _price_cache[cache_key]

    end = datetime.today()
    start = end - timedelta(days=lookback_days)

    try:
        raw = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            raise ValueError(f"yfinance returned empty data for {ticker}")

        close = raw["Close"]
        if hasattr(close, "squeeze"):
            close = close.squeeze()

        series = pd.Series(close.values, index=close.index, name=ticker, dtype=float)
        _price_cache[cache_key] = series
        return series

    except Exception as live_exc:
        fallback_key = f"prices_{ticker}_{lookback_days}"
        if fallback_key in _fallback:
            entry = _fallback[fallback_key]
            idx = pd.to_datetime(entry["dates"])
            series = pd.Series(entry["values"], index=idx, name=ticker, dtype=float)
            _price_cache[cache_key] = series
            return series

        raise ValueError(
            f"Failed to fetch price data for '{ticker}'. "
            f"Live error: {live_exc}. "
            f"No fallback entry found under key '{fallback_key}'."
        ) from live_exc


def get_info(ticker: str) -> dict:
    """Return the Yahoo Finance info dict for *ticker*.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker symbol.

    Returns
    -------
    dict
        Info dict containing at minimum the keys ``sector``, ``longName``,
        ``marketCap``, and ``currency``. If yfinance fails, a stub dict with
        those four keys is returned from the fallback cache. If no fallback
        exists, a stub with empty/zero values is returned so callers are not
        blocked by missing metadata.
    """
    if ticker in _info_cache:
        return _info_cache[ticker]

    _STUB_KEYS = ("sector", "longName", "marketCap", "currency")

    try:
        info = yf.Ticker(ticker).info
        if not info or "symbol" not in info:
            raise ValueError(f"yfinance returned empty info for {ticker}")
        _info_cache[ticker] = info
        return info

    except Exception:
        fallback_key = f"info_{ticker}"
        if fallback_key in _fallback:
            info = _fallback[fallback_key]
            _info_cache[ticker] = info
            return info

        stub: dict = {k: "" if k != "marketCap" else 0 for k in _STUB_KEYS}
        stub["longName"] = ticker
        _info_cache[ticker] = stub
        return stub
