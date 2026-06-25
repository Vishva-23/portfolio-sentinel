# Portfolio Sentinel

A conversational portfolio analytics agent: chat in natural language, get precise historical performance and risk metrics powered by typed Python tools and a Groq LLM.

## What this is

Portfolio Sentinel is a read-only analytics assistant that lets you ask questions about a stock portfolio in plain English. The system uses the tool-calling pattern: the language model receives five strongly-typed tool schemas and may invoke them to retrieve computed metrics. The model never sees raw price data, never writes or executes code, and never accesses any data source outside the five tools. Every number it quotes comes directly from a tool result.

This design matters in regulated and high-stakes contexts. Because the model can only invoke pre-defined, auditable functions, the blast radius of any LLM hallucination is limited to description, not data. The tools enforce business logic — formulae, thresholds, grain definitions — in Python, where they can be tested and reviewed. The model's role is interpretation and communication, not computation.

## Architecture

The system is organised into three layers separated by a strict LLM boundary. The **chat UI** (a single HTML file) sends natural language messages to the **FastAPI server**, which invokes the **Groq agent**. The agent calls `llama-3.3-70b-versatile` with tool schemas; when the model issues a tool call, the agent dispatches it to one of the five typed Python functions in `tools/portfolio_tools.py`, which in turn fetches market data through `tools/market_data.py` (the only file that may call yfinance). The LLM never crosses the boundary into the data layer — it sees tool schemas and JSON results, nothing else.

## The five tools

| Tool | What it answers | Key metric |
|---|---|---|
| `get_portfolio_summary` | How has this portfolio performed? | Annualised return, active return vs S&P 500 |
| `get_risk_metrics` | How risky is this portfolio? | Sharpe ratio, max drawdown, VaR 95%, CVaR 95% |
| `get_correlation_matrix` | How correlated are the holdings? | Average pairwise correlation |
| `get_sector_exposure` | What sectors am I exposed to? | Sector weight breakdown, concentration flag |
| `get_as_of_snapshot` | How did it look on a specific past date? | Point-in-time total and annualised return |

## Quick start

1. **Clone and create a virtual environment**
   ```bash
   git clone <repo-url>
   cd portfolio-sentinel
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your API key**
   ```bash
   cp .env.example .env
   # Edit .env and set GROQ_API_KEY=your_actual_key
   ```

4. **Pre-populate the fallback cache** (recommended before demos or CI)
   ```bash
   python scripts/build_fallback_cache.py
   ```

5. **Start the server**
   ```bash
   uvicorn app:app --reload
   ```

6. **Open the UI**
   Navigate to [http://localhost:8000](http://localhost:8000)

## Example questions

- "What is the total and annualised return of my portfolio over the past year?"
- "What was my portfolio's Sharpe ratio and maximum drawdown over the last 6 months?"
- "What is the 95% VaR and CVaR for this portfolio?"
- "Show me the correlation matrix between all holdings."
- "Which sectors am I most exposed to, and is there any concentration risk?"
- "How did my portfolio perform as of 2024-06-30?"
- "Compare my annualised return to the S&P 500 benchmark."
- "How did the portfolio's risk metrics look over the last 90 days?"

## Design decisions

### Why typed tools instead of free-form SQL or code

Allowing the model to generate and execute arbitrary queries or code creates an unbounded attack surface: a prompt-injected or hallucinated query could read unintended data, modify state, or crash the process. Typed tool functions accept only declared parameters with known types. The model cannot write new logic — it can only choose which function to call and which arguments to pass. This makes the system auditable, testable, and safe to run in front of users.

### Why the point-in-time tool is an exception

`get_as_of_snapshot` bypasses the in-memory price cache. The cache stores the most-recent N days of history keyed by `TICKER_lookbackdays`. It cannot represent a specific historical end date without storing multiple copies per ticker for each possible `as_of_date`. The snapshot function fetches a fixed 1095-day lookback and slices in-memory to the requested date, trading a cache miss for correctness. See `ARCHITECTURE.md` for the full exception rationale.

### What this project deliberately does not do

Portfolio Sentinel is intentionally read-only. It does not execute trades, place orders, or connect to any brokerage API. It does not generate price predictions or forecasts. It does not give investment advice. The system prompt instructs the model to decline such requests explicitly. This is a design constraint, not a technical limitation — removing it would require deliberate changes to both the agent configuration and the tool layer.

## Stack

- **Python 3.11+**
- **FastAPI** — ASGI web framework and REST API
- **Groq** (`llama-3.3-70b-versatile`) — LLM with native tool-calling
- **yfinance** — Yahoo Finance market data adapter
- **pandas** — price series alignment and return computation
- **numpy** — vectorised statistics (VaR, CVaR, volatility)

---

The default portfolio includes **CRH** (CRH plc on the NYSE), Ireland's largest company by market cap, as the international market representation.
