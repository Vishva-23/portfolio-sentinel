"""
NewsAPI integration for Portfolio Sentinel.

Fetches recent news articles for a ticker symbol and caches results in memory
so repeated calls within one session do not hit the API again. All failures
are handled gracefully — if the API key is missing or the request fails, an
empty list is returned and a warning is logged. The rest of the application
continues to function normally.

Cache key format: ``"TICKER_YYYY-MM-DD"`` so the cache is invalidated
automatically when the date changes between sessions.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from typing import Any

from tools.market_data import get_info

logger = logging.getLogger("portfolio_sentinel")

_news_cache: dict[str, list[dict]] = {}


def fetch_news(ticker: str, days_back: int = 7) -> list[dict]:
    """Fetch recent news articles for a ticker symbol via NewsAPI.

    Uses the ticker symbol and the company long name from yfinance to build a
    targeted query. Results are cached in memory keyed by ticker and today's
    date so the API is not called more than once per ticker per session.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker symbol (e.g. ``"NVDA"``).
    days_back:
        Number of days of news history to request. Default is 7.

    Returns
    -------
    list[dict]
        List of article dicts, each with keys: ``title``, ``description``,
        ``url``, ``published_at``, ``source``. Returns an empty list if
        NEWSAPI_KEY is not set, if the API call fails, or if no articles are
        found.
    """
    cache_key = f"{ticker}_{date.today().isoformat()}"
    if cache_key in _news_cache:
        return _news_cache[cache_key]

    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        logger.warning(
            "NEWSAPI_KEY is not set — get_news_context will return empty results "
            "for ticker '%s'. Set the key in .env to enable news fetching.",
            ticker,
        )
        _news_cache[cache_key] = []
        return []

    try:
        from newsapi import NewsApiClient

        info = get_info(ticker)
        company_name = info.get("longName") or ticker
        query = f"{ticker} OR \"{company_name}\""

        from_date = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        client = NewsApiClient(api_key=api_key)
        response = client.get_everything(
            q=query,
            language="en",
            sort_by="publishedAt",
            from_param=from_date,
            page_size=20,
        )

        articles: list[dict] = []
        for article in response.get("articles", []):
            articles.append(
                {
                    "title": article.get("title") or "",
                    "description": article.get("description") or "",
                    "url": article.get("url") or "",
                    "published_at": (
                        article.get("publishedAt") or ""
                    )[:10],
                    "source": (article.get("source") or {}).get("name") or "",
                }
            )

        _news_cache[cache_key] = articles
        return articles

    except Exception as exc:
        logger.warning(
            "NewsAPI fetch failed for ticker '%s': %s — returning empty results.",
            ticker,
            exc,
        )
        _news_cache[cache_key] = []
        return []
