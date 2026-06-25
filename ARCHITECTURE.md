# Architecture

## Layers

Portfolio Sentinel is organised into three layers. Each layer has a single, well-defined responsibility and communicates with the layer below it through a narrow interface.

**Layer 1 — Chat UI** (`static/index.html`)
Renders the conversation, maintains conversation history in a JavaScript variable, and sends POST requests to the FastAPI server. The UI holds no business logic. It displays tool-derived numbers as text and never computes anything itself.

**Layer 2 — FastAPI server + Groq agent** (`app.py`, `agent/agent.py`)
Receives the user message and conversation history, calls the Groq API with the tool schemas and system prompt, dispatches tool calls to the Python functions, and returns the final model response. The server owns request validation (Pydantic models) and error handling (global exception handler). The agent owns the tool dispatch table and the conversation loop.

**Layer 3 — Typed tools + market data** (`tools/portfolio_tools.py`, `tools/market_data.py`)
Computes all metrics. `market_data.py` is the sole gateway to yfinance; no other file may import yfinance directly. `portfolio_tools.py` implements the five typed functions, the two private helpers, and all business-logic constants.

---

## LLM boundary

The Groq model operates entirely within Layer 2. It can see:

- **Tool schemas** — the five function definitions in JSON, including parameter names, types, and descriptions.
- **Tool results** — the JSON dicts returned by each tool function after dispatch.
- **Conversation history** — prior assistant and user messages from this session.
- **System prompt** — the read-only analytics mandate.

The model never sees:

- Raw price arrays or DataFrames.
- The `fallback_cache.json` file or its contents.
- Python source code.
- The yfinance response structure.
- Environment variables or API keys.

This boundary means a compromised or hallucinating model cannot access data that was not explicitly placed in its context as a tool result.

---

## Tool design principles

**Typed arguments only.** Every parameter is a concrete Python type: `list[str]`, `list[float]`, `int`, `float`, `str`. The model selects values from what the user said and the conversation context; it never generates executable code or freeform queries.

**Echo inputs in outputs.** Every return dict repeats the `tickers`, `weights_normalised`, and `lookback_days` that were used. The model can quote these without inference, and callers can verify that the correct inputs were used.

**Return denominators alongside ratios.** The Sharpe ratio is returned alongside `volatility_annual`, `annualised_return`, and `risk_free_rate_used`. The active return is accompanied by `benchmark_return`. This lets the model give attributable explanations and lets reviewers spot formula errors.

**Business logic in Python, not in prompts.** The Sharpe formula, the `concentration_flag` threshold (50%), the VaR quantile inversion, and the inner-join alignment strategy are all encoded in Python. The system prompt does not contain formulae. Changing a calculation requires a code change and a test, not a prompt edit.

---

## Grain definitions

**Price row:** one calendar date mapped to one float, the adjusted closing price for that trading day. Weekends and market holidays are absent from the series. Prices are in the security's native currency.

**Daily return:** `r_t = (P_t / P_{t-1}) - 1`, computed via `pd.Series.pct_change()`. The first element (which would require `P_{-1}`) is NaN and is dropped before any aggregation.

**Portfolio return:** `r_portfolio_t = Σ (w_i × r_i_t)` summed over all tickers `i`, where weights are normalised to sum to 1. Computed as a dot product of the daily returns matrix and the weight vector.

**Trading day:** any date with a non-NaN adjusted-close price present in the Yahoo Finance response. The count of trading days (`n`) is always derived from the aligned return series length, not from a calendar formula.

---

## Default universe

A ticker is in scope for a given calculation if and only if:

1. `get_prices` returns a non-empty series for it (valid prices exist for the lookback window).
2. Its return series has at least one non-NaN value after `pct_change()`.
3. The date is present in the intersection (inner join) of all ticker series in the portfolio.

Tickers that fail condition 1 raise a `ValueError` from `market_data.get_prices`. Conditions 2 and 3 are enforced by the `dropna()` and `join="inner"` steps in `_portfolio_returns`.

---

## Exception paths

### 1. `get_as_of_snapshot` — bypasses the in-memory price cache

The in-memory price cache (`_price_cache` in `market_data.py`) is keyed by `TICKER_lookbackdays`. A cache entry stores the most-recent `lookback_days` of price history relative to the moment of the first fetch during the process lifetime. It cannot be parameterised by an arbitrary end date without storing a separate copy per `(ticker, end_date)` pair, which would grow unboundedly.

`get_as_of_snapshot` needs prices up to a specific historical date. It fetches a fixed 1095-day window (approximately 3 years) — large enough to cover any plausible `as_of_date` — and then slices the resulting series with `series.loc[:as_of_date]` entirely in memory. The cache stores the 1095-day series under the key `TICKER_1095`, which may be reused by subsequent snapshot calls but represents the full available window, not a window ending at `as_of_date`.

The trade-off: one additional yfinance call per unique ticker on the first snapshot request, in exchange for correctness of historical slices.

### 2. Fallback cache — resilience against yfinance upstream failures

yfinance is an unofficial, reverse-engineered Yahoo Finance client with no published SLA. It has been known to break after Yahoo Finance API changes, to rate-limit aggressively, and to return empty responses during market hours or on weekends. A demo or CI run that depends on live yfinance calls is fragile.

`fallback_cache.json` is a JSON file pre-populated by `scripts/build_fallback_cache.py`. It is loaded once at import time into the `_fallback` dict in `market_data.py`. If a live yfinance call fails for any reason, the module attempts to serve the data from `_fallback` before raising. If both the live call and the fallback are unavailable, a `ValueError` with a descriptive message is raised.

The fallback is not committed to version control (it is `.gitignore`d) because it contains time-specific market data. It is regenerated before each demo or CI run via the build script.

---

## Risk-free rate

`RISK_FREE_RATE_ANNUAL = 0.04` approximates the ECB deposit facility rate as of mid-2025. It is a module-level constant in `tools/portfolio_tools.py`, not fetched from a live source, so it does not introduce a network dependency into the Sharpe ratio calculation.

To update it: change the constant at the top of `tools/portfolio_tools.py`. The value is echoed as `risk_free_rate_used` in the `get_risk_metrics` return dict so the model and the user always know which rate was applied.

---

## Known limitations

- **yfinance data is end-of-day, not real-time.** Prices reflect the previous trading session's close. Intraday questions cannot be answered.

- **Sector classifications from Yahoo Finance may lag official GICS updates.** Yahoo Finance sourced sector data is not always synchronised with the official MSCI/S&P GICS classification schedule. A company that changed sectors recently may still appear under its old classification.

- **CRH trades on the NYSE in USD.** No currency conversion is required when mixing CRH with other USD-denominated holdings in the default portfolio.

- **VaR and CVaR use historical simulation, not a parametric model.** Historical simulation is accurate for fat-tailed return distributions and does not assume normality, but it requires sufficient history (at least 20 trading days are enforced; 252+ is recommended) and can be slow to react to sudden volatility regime changes.
