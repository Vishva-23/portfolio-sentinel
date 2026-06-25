"""
Groq-powered tool-calling agent for Portfolio Sentinel.

The agent receives a user message and full conversation history, calls the
Groq API with the five typed tool schemas, dispatches any tool calls to the
corresponding Python functions, and returns the final assistant text.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from eval.eval_logger import get_request_id, log_eval_entry

from tools.portfolio_tools import (
    get_portfolio_summary,
    get_risk_metrics,
    get_correlation_matrix,
    get_sector_exposure,
    get_as_of_snapshot,
    get_news_context,
)

load_dotenv()

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )
        _client = Groq(api_key=api_key)
    return _client


MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are Portfolio Sentinel, a read-only portfolio analytics assistant. "
    "You answer questions about portfolio performance, risk, sector exposure, "
    "and correlations using only the five tools available to you. "
    "You never give investment advice or price predictions. "
    "You never make up numbers — every figure in your response must come from "
    "a tool result. If a user asks something the tools cannot answer, say so "
    "clearly. When referencing numbers, always state the time period they cover."
)

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_portfolio_summary",
            "description": (
                "Summarise overall portfolio performance over a lookback window. "
                "Returns total return, annualised return, S&P 500 benchmark return, "
                "and active return (portfolio minus benchmark). Use this as the "
                "starting point for any general performance question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of Yahoo Finance ticker symbols "
                            "(e.g. [\"AAPL\", \"MSFT\", \"CRH\"])."
                        ),
                    },
                    "weights": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": (
                            "Portfolio allocation weights in the same order as tickers. "
                            "Will be normalised to sum to 1. "
                            "E.g. [0.25, 0.20, 0.25, 0.20, 0.10]."
                        ),
                    },
                    "lookback_days": {
                        "type": "integer",
                        "description": (
                            "Number of calendar days of history to analyse. "
                            "Default is 365 (approximately one year). "
                            "Use 90 for a quarter, 180 for six months."
                        ),
                    },
                },
                "required": ["tickers", "weights"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_risk_metrics",
            "description": (
                "Compute portfolio risk statistics: annualised volatility, Sharpe ratio, "
                "maximum drawdown, Value-at-Risk (VaR), and Conditional VaR (CVaR / "
                "Expected Shortfall). Use when the user asks about risk, volatility, "
                "drawdown, or VaR."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of Yahoo Finance ticker symbols.",
                    },
                    "weights": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": (
                            "Portfolio allocation weights in the same order as tickers. "
                            "Will be normalised to sum to 1."
                        ),
                    },
                    "lookback_days": {
                        "type": "integer",
                        "description": (
                            "Number of calendar days of history to analyse. Default 365."
                        ),
                    },
                    "confidence_level": {
                        "type": "number",
                        "description": (
                            "Confidence level for VaR and CVaR calculation. "
                            "0.95 means 5th-percentile loss. Default 0.95."
                        ),
                    },
                },
                "required": ["tickers", "weights"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_correlation_matrix",
            "description": (
                "Compute the Pearson correlation matrix of daily returns across tickers. "
                "Returns the full matrix and the average pairwise correlation. "
                "Use when the user asks about diversification, correlation, or "
                "how holdings move relative to each other."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of Yahoo Finance ticker symbols.",
                    },
                    "lookback_days": {
                        "type": "integer",
                        "description": (
                            "Number of calendar days of history to analyse. Default 365."
                        ),
                    },
                },
                "required": ["tickers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sector_exposure",
            "description": (
                "Show portfolio weight allocation broken down by GICS sector. "
                "Identifies the dominant sector and flags high concentration. "
                "Use when the user asks about sector breakdown, diversification "
                "by sector, or concentration risk."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of Yahoo Finance ticker symbols.",
                    },
                    "weights": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": (
                            "Portfolio allocation weights in the same order as tickers. "
                            "Will be normalised to sum to 1."
                        ),
                    },
                },
                "required": ["tickers", "weights"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_as_of_snapshot",
            "description": (
                "Compute portfolio performance metrics as of a specific historical date. "
                "Returns the same metrics as get_portfolio_summary but calculated only "
                "up to and including the given date. Use when the user asks how the "
                "portfolio performed at a specific point in the past."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of Yahoo Finance ticker symbols.",
                    },
                    "weights": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": (
                            "Portfolio allocation weights in the same order as tickers. "
                            "Will be normalised to sum to 1."
                        ),
                    },
                    "as_of_date": {
                        "type": "string",
                        "description": (
                            "Historical end date in YYYY-MM-DD format. "
                            "Must be a past date, not today or the future."
                        ),
                    },
                },
                "required": ["tickers", "weights", "as_of_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news_context",
            "description": (
                "Retrieve recent news headlines and context for portfolio tickers "
                "using semantic search. Use this when the user asks why a stock "
                "moved, what the current sentiment is, or for qualitative context "
                "alongside quantitative metrics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of Yahoo Finance ticker symbols to fetch news for.",
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural-language description of what information to find. "
                            "For example: 'why did NVDA drop this week' or "
                            "'recent earnings news for AAPL'."
                        ),
                    },
                },
                "required": ["tickers", "query"],
            },
        },
    },
]

_TOOL_DISPATCH: dict[str, Any] = {
    "get_portfolio_summary": get_portfolio_summary,
    "get_risk_metrics": get_risk_metrics,
    "get_correlation_matrix": get_correlation_matrix,
    "get_sector_exposure": get_sector_exposure,
    "get_as_of_snapshot": get_as_of_snapshot,
    "get_news_context": get_news_context,
}


def _execute_tool(name: str, arguments: dict) -> str:
    """Dispatch a tool call and return the result as a JSON string.

    Times the call, writes an eval log entry on success, and records the tool
    name in the per-request ContextVar accumulator. If the tool raises, the
    exception message is returned as the tool result so the model can relay it
    to the user rather than crashing the request.
    """
    fn = _TOOL_DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        t0 = time.time()
        result = fn(**arguments)
        latency_ms = round((time.time() - t0) * 1000, 2)
        log_eval_entry(
            request_id=get_request_id(),
            tool_name=name,
            inputs=arguments,
            outputs=result,
            latency_ms=latency_ms,
            grounded=True,
        )
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def run_agent(
    tickers: list[str],
    weights: list[float],
    message: str,
    conversation_history: list[dict],
) -> tuple[str, list[dict]]:
    """Run the agent for one user turn.

    Parameters
    ----------
    tickers:
        Current portfolio tickers (passed into the conversation context).
    weights:
        Current portfolio weights (passed into the conversation context).
    message:
        The user's latest message.
    conversation_history:
        Full prior conversation as a list of ``{"role": ..., "content": ...}``
        dicts. This list is mutated and returned with the new turn appended.

    Returns
    -------
    tuple[str, list[dict]]
        ``(reply_text, updated_conversation_history)``
    """
    history = list(conversation_history)
    history.append({"role": "user", "content": message})

    norm_weights = [round(w / sum(weights), 6) for w in weights]
    portfolio_context = (
        f"\n\nThe user's current portfolio is: tickers={tickers}, "
        f"weights={norm_weights}. "
        "Always pass these exact tickers and weights when calling tools, "
        "unless the user explicitly asks about a different set of tickers."
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT + portfolio_context}] + history

    response = _get_client().chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOL_SCHEMAS,
        tool_choice="auto",
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        messages.append(
            {
                "role": "assistant",
                "content": response_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            tool_name = tc.function.name
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            tool_result = _execute_tool(tool_name, arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                }
            )

        final_response = _get_client().chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        reply = final_response.choices[0].message.content or ""
    else:
        reply = response_message.content or ""

    history.append({"role": "assistant", "content": reply})
    return reply, history
