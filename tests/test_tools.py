"""
Unit tests for portfolio_tools.py.

All tests use deterministic synthetic data — no network calls are made.
get_prices is mocked to return a fixed price series generated from a seeded
random walk. get_info is mocked to return a Technology sector stub.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tools.portfolio_tools import (
    _normalise_weights,
    get_portfolio_summary,
    get_risk_metrics,
    get_correlation_matrix,
    get_sector_exposure,
    get_as_of_snapshot,
    get_news_context,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_price_series(ticker: str, n: int = 300, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.015, size=n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    idx = pd.date_range(end=date.today() - timedelta(days=1), periods=n, freq="B")
    return pd.Series(prices, index=idx, name=ticker, dtype=float)


def _mock_get_prices(ticker: str, lookback_days: int = 365) -> pd.Series:
    seed = abs(hash(ticker)) % (2**31)
    return _make_price_series(ticker, n=min(lookback_days, 300), seed=seed)


def _mock_get_info(ticker: str) -> dict:
    return {
        "sector": "Technology",
        "longName": "Test Corp",
        "marketCap": 1_000_000_000,
        "currency": "USD",
        "symbol": ticker,
    }


TICKERS = ["AAPL", "MSFT", "NVDA"]
WEIGHTS = [0.4, 0.35, 0.25]


# ---------------------------------------------------------------------------
# _normalise_weights
# ---------------------------------------------------------------------------

class TestNormaliseWeights:
    def test_sums_to_one(self) -> None:
        result = _normalise_weights([1.0, 2.0, 1.0])
        assert abs(sum(result) - 1.0) < 1e-9

    def test_proportions_preserved(self) -> None:
        result = _normalise_weights([50.0, 50.0])
        assert abs(result[0] - 0.5) < 1e-9
        assert abs(result[1] - 0.5) < 1e-9

    def test_already_normalised(self) -> None:
        weights = [0.25, 0.25, 0.25, 0.25]
        result = _normalise_weights(weights)
        assert abs(sum(result) - 1.0) < 1e-9

    def test_raises_on_negative(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _normalise_weights([0.5, -0.1, 0.6])

    def test_raises_on_all_zero(self) -> None:
        with pytest.raises(ValueError, match="zero"):
            _normalise_weights([0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# get_portfolio_summary
# ---------------------------------------------------------------------------

@patch("tools.portfolio_tools.get_prices", side_effect=_mock_get_prices)
class TestGetPortfolioSummary:
    def test_expected_keys(self, _mock) -> None:
        result = get_portfolio_summary(TICKERS, WEIGHTS)
        expected_keys = {
            "tickers", "weights_normalised", "lookback_days",
            "total_return", "annualised_return",
            "benchmark_ticker", "benchmark_return", "active_return",
            "start_date", "end_date", "trading_days",
        }
        assert expected_keys.issubset(result.keys())

    def test_total_return_is_float(self, _mock) -> None:
        result = get_portfolio_summary(TICKERS, WEIGHTS)
        assert isinstance(result["total_return"], float)

    def test_trading_days_positive(self, _mock) -> None:
        result = get_portfolio_summary(TICKERS, WEIGHTS)
        assert result["trading_days"] > 0

    def test_weights_normalised_sum(self, _mock) -> None:
        result = get_portfolio_summary(TICKERS, WEIGHTS)
        assert abs(sum(result["weights_normalised"]) - 1.0) < 1e-9

    def test_raises_on_length_mismatch(self, _mock) -> None:
        with pytest.raises(ValueError):
            get_portfolio_summary(TICKERS, [0.5, 0.5])


# ---------------------------------------------------------------------------
# get_risk_metrics
# ---------------------------------------------------------------------------

@patch("tools.portfolio_tools.get_prices", side_effect=_mock_get_prices)
class TestGetRiskMetrics:
    def test_expected_keys(self, _mock) -> None:
        result = get_risk_metrics(TICKERS, WEIGHTS)
        expected_keys = {
            "volatility_annual", "sharpe_ratio", "max_drawdown",
            "var_95", "cvar_95", "risk_free_rate_used", "trading_days",
        }
        assert expected_keys.issubset(result.keys())

    def test_sharpe_ratio_is_float(self, _mock) -> None:
        result = get_risk_metrics(TICKERS, WEIGHTS)
        assert isinstance(result["sharpe_ratio"], float)

    def test_volatility_positive(self, _mock) -> None:
        result = get_risk_metrics(TICKERS, WEIGHTS)
        assert result["volatility_annual"] > 0

    def test_max_drawdown_nonpositive(self, _mock) -> None:
        result = get_risk_metrics(TICKERS, WEIGHTS)
        assert result["max_drawdown"] <= 0

    def test_var_less_than_cvar(self, _mock) -> None:
        result = get_risk_metrics(TICKERS, WEIGHTS)
        assert result["cvar_95"] <= result["var_95"]


# ---------------------------------------------------------------------------
# get_correlation_matrix
# ---------------------------------------------------------------------------

@patch("tools.portfolio_tools.get_prices", side_effect=_mock_get_prices)
class TestGetCorrelationMatrix:
    def test_matrix_is_square(self, _mock) -> None:
        result = get_correlation_matrix(TICKERS)
        m = result["matrix"]
        assert len(m) == len(TICKERS)
        for row in m:
            assert len(row) == len(TICKERS)

    def test_diagonal_is_one(self, _mock) -> None:
        result = get_correlation_matrix(TICKERS)
        m = result["matrix"]
        for i in range(len(TICKERS)):
            assert abs(m[i][i] - 1.0) < 1e-9

    def test_avg_pairwise_in_range(self, _mock) -> None:
        result = get_correlation_matrix(TICKERS)
        assert -1.0 <= result["avg_pairwise_correlation"] <= 1.0

    def test_trading_days_positive(self, _mock) -> None:
        result = get_correlation_matrix(TICKERS)
        assert result["trading_days"] > 0


# ---------------------------------------------------------------------------
# get_sector_exposure
# ---------------------------------------------------------------------------

@patch("tools.portfolio_tools.get_info", side_effect=_mock_get_info)
class TestGetSectorExposure:
    def test_expected_keys(self, _mock) -> None:
        result = get_sector_exposure(TICKERS, WEIGHTS)
        expected_keys = {
            "tickers", "weights_normalised", "sectors",
            "dominant_sector", "dominant_sector_weight", "concentration_flag",
        }
        assert expected_keys.issubset(result.keys())

    def test_weights_normalised_sum(self, _mock) -> None:
        result = get_sector_exposure(TICKERS, WEIGHTS)
        assert abs(sum(result["weights_normalised"]) - 1.0) < 1e-9

    def test_concentration_flag_is_bool(self, _mock) -> None:
        result = get_sector_exposure(TICKERS, WEIGHTS)
        assert isinstance(result["concentration_flag"], bool)

    def test_sector_weights_sum_to_one(self, _mock) -> None:
        result = get_sector_exposure(TICKERS, WEIGHTS)
        sector_total = sum(s["weight_in_portfolio"] for s in result["sectors"])
        assert abs(sector_total - 1.0) < 1e-9

    def test_single_sector_all_technology(self, _mock) -> None:
        result = get_sector_exposure(TICKERS, WEIGHTS)
        assert result["dominant_sector"] == "Technology"


# ---------------------------------------------------------------------------
# get_as_of_snapshot
# ---------------------------------------------------------------------------

@patch("tools.portfolio_tools.get_prices", side_effect=_mock_get_prices)
class TestGetAsOfSnapshot:
    def test_as_of_date_echoed(self, _mock) -> None:
        as_of = (date.today() - timedelta(days=60)).strftime("%Y-%m-%d")
        result = get_as_of_snapshot(TICKERS, WEIGHTS, as_of)
        assert result["as_of_date"] == as_of

    def test_expected_keys(self, _mock) -> None:
        as_of = (date.today() - timedelta(days=60)).strftime("%Y-%m-%d")
        result = get_as_of_snapshot(TICKERS, WEIGHTS, as_of)
        expected_keys = {
            "tickers", "weights_normalised", "as_of_date",
            "total_return", "annualised_return", "trading_days",
        }
        assert expected_keys.issubset(result.keys())

    def test_raises_on_future_date(self, _mock) -> None:
        future = (date.today() + timedelta(days=10)).strftime("%Y-%m-%d")
        with pytest.raises(ValueError, match="future"):
            get_as_of_snapshot(TICKERS, WEIGHTS, future)

    def test_raises_on_invalid_format(self, _mock) -> None:
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            get_as_of_snapshot(TICKERS, WEIGHTS, "01/01/2024")

    def test_total_return_is_float(self, _mock) -> None:
        as_of = (date.today() - timedelta(days=60)).strftime("%Y-%m-%d")
        result = get_as_of_snapshot(TICKERS, WEIGHTS, as_of)
        assert isinstance(result["total_return"], float)


# ---------------------------------------------------------------------------
# get_news_context
# ---------------------------------------------------------------------------

@patch("tools.portfolio_tools.retrieve_context", return_value=["chunk1", "chunk2"])
@patch("tools.portfolio_tools.embed_and_store", return_value=2)
@patch("tools.portfolio_tools.fetch_news", return_value=[{"title": "Test", "description": "Desc", "url": "http://x.com", "published_at": "2025-01-01", "source": "Reuters"}])
class TestGetNewsContext:
    def test_expected_keys(self, _fn, _es, _rc) -> None:
        result = get_news_context(["AAPL"], "recent earnings")
        assert {"tickers", "query", "results", "total_articles_fetched", "note"}.issubset(result.keys())

    def test_one_result_per_ticker(self, _fn, _es, _rc) -> None:
        result = get_news_context(["AAPL", "MSFT"], "earnings")
        assert len(result["results"]) == 2

    def test_result_has_ticker_and_chunks(self, _fn, _es, _rc) -> None:
        result = get_news_context(["AAPL"], "earnings")
        assert result["results"][0]["ticker"] == "AAPL"
        assert "retrieved_chunks" in result["results"][0]

    def test_total_articles_fetched(self, _fn, _es, _rc) -> None:
        result = get_news_context(["AAPL", "MSFT"], "earnings")
        assert result["total_articles_fetched"] == 2

    def test_empty_when_no_news(self, _fn, _es, _rc) -> None:
        _fn.return_value = []
        _rc.return_value = []
        result = get_news_context(["AAPL"], "earnings")
        assert result["total_articles_fetched"] == 0
        assert result["results"][0]["retrieved_chunks"] == []

    def test_note_is_string(self, _fn, _es, _rc) -> None:
        result = get_news_context(["AAPL"], "test")
        assert isinstance(result["note"], str)

    def test_query_echoed(self, _fn, _es, _rc) -> None:
        result = get_news_context(["AAPL"], "why did NVDA drop")
        assert result["query"] == "why did NVDA drop"

    def test_embed_not_called_when_no_articles(self, _fn, _es, _rc) -> None:
        _fn.return_value = []
        get_news_context(["AAPL"], "test")
        _es.assert_not_called()
