"""
Benchmark runner for Portfolio Sentinel.

Runs a fixed set of questions against the live agent and checks which tool
each question triggers. Pass/fail is determined by whether the expected tool
appears in the list of tools actually called during the agent's response.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from agent.agent import run_agent
from eval.eval_logger import get_request_tools, reset_request_tools, set_request_id

BENCHMARK_QUESTIONS: list[dict[str, str]] = [
    {
        "question": "What is the total return of my portfolio over the last year?",
        "expected_tool": "get_portfolio_summary",
    },
    {
        "question": "What is the Sharpe ratio and maximum drawdown?",
        "expected_tool": "get_risk_metrics",
    },
    {
        "question": "Show me the correlation matrix for my portfolio.",
        "expected_tool": "get_correlation_matrix",
    },
    {
        "question": "Which sectors am I most exposed to?",
        "expected_tool": "get_sector_exposure",
    },
    {
        "question": "What would my portfolio have been worth on 2025-01-01?",
        "expected_tool": "get_as_of_snapshot",
    },
    {
        "question": "How volatile is my portfolio compared to the benchmark?",
        "expected_tool": "get_risk_metrics",
    },
    {
        "question": "Am I concentrated in any single sector?",
        "expected_tool": "get_sector_exposure",
    },
    {
        "question": "What was my annualised return over the last 180 days?",
        "expected_tool": "get_portfolio_summary",
    },
]


def run_benchmark(
    tickers: list[str],
    weights: list[float],
) -> list[dict[str, Any]]:
    """Run all benchmark questions and return per-question results.

    Each question is run with an empty conversation history so results are
    independent. The expected tool is compared against the tools actually
    called during the agent's turn, as recorded by the eval logger's
    per-request ContextVar accumulator.

    Parameters
    ----------
    tickers:
        Portfolio ticker symbols to pass to the agent.
    weights:
        Portfolio weights to pass to the agent.

    Returns
    -------
    list[dict]
        One result dict per question with keys: question (str),
        expected_tool (str), actual_tools_called (list[str]),
        passed (bool), latency_ms (float).
    """
    results: list[dict[str, Any]] = []

    for item in BENCHMARK_QUESTIONS:
        bench_request_id = f"benchmark-{str(uuid.uuid4())[:8]}"
        set_request_id(bench_request_id)
        reset_request_tools()

        t0 = time.time()
        run_agent(
            tickers=tickers,
            weights=weights,
            message=item["question"],
            conversation_history=[],
        )
        latency_ms = round((time.time() - t0) * 1000, 2)

        actual_tools = get_request_tools()
        passed = item["expected_tool"] in actual_tools

        results.append(
            {
                "question": item["question"],
                "expected_tool": item["expected_tool"],
                "actual_tools_called": actual_tools,
                "passed": passed,
                "latency_ms": latency_ms,
            }
        )

    return results
