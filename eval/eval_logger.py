"""
Evaluation logger for Portfolio Sentinel.

Every tool call made by the agent is appended as a JSON line to
eval/eval_log.jsonl. The log is the audit trail that lets a human reviewer
verify that numbers quoted in agent responses came directly from tool results.

grounded flag
-------------
grounded=True is always passed by the calling code for now. Automated
groundedness verification is not implemented — a human reviewer is expected to
inspect the log and change grounded=False on any entry where the agent's
response did not faithfully quote the tool result. Do not automate this flag
without a reliable verification method.

Per-request tool accumulator
-----------------------------
This module also maintains a per-async-task list of tool names called during
the current HTTP request, using contextvars.ContextVar. app.py calls
reset_request_tools() and set_request_id() before invoking run_agent, then
calls get_request_tools() afterwards to populate the structured request log.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_EVAL_DIR = Path(__file__).resolve().parent
_EVAL_LOG_PATH = _EVAL_DIR / "eval_log.jsonl"

# ContextVar stores None until reset_request_tools() is called for the first
# time in a request context. Using None as the default avoids sharing a single
# mutable list across all contexts.
_request_tools: ContextVar[list[str] | None] = ContextVar(
    "_request_tools", default=None
)

_current_request_id: ContextVar[str] = ContextVar(
    "_current_request_id", default=""
)


def set_request_id(request_id: str) -> None:
    """Bind a request ID to the current async context."""
    _current_request_id.set(request_id)


def get_request_id() -> str:
    """Return the request ID bound to the current async context."""
    return _current_request_id.get()


def reset_request_tools() -> None:
    """Reset the per-request tool accumulator. Call at the start of each request."""
    _request_tools.set([])


def record_tool_call(name: str) -> None:
    """Append a tool name to the current request's accumulator."""
    current = _request_tools.get()
    if current is None:
        _request_tools.set([name])
    else:
        current.append(name)


def get_request_tools() -> list[str]:
    """Return tool names called so far in the current request context."""
    current = _request_tools.get()
    return list(current) if current is not None else []


def log_eval_entry(
    request_id: str,
    tool_name: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    latency_ms: float,
    grounded: bool = True,
) -> None:
    """Append one evaluation record to eval/eval_log.jsonl.

    Also records the tool name in the per-request ContextVar accumulator so
    app.py can include tool_calls_made in the structured request log without
    any additional bookkeeping.

    Parameters
    ----------
    request_id:
        UUID of the parent HTTP request. Links this entry back to app.log.
    tool_name:
        Name of the tool that was called.
    inputs:
        Arguments passed to the tool (the raw arguments dict).
    outputs:
        Return value of the tool (the result dict before JSON serialisation).
    latency_ms:
        Execution time of the tool call in milliseconds.
    grounded:
        Whether the agent response faithfully quoted this tool result.
        Always True for now — human reviewer audits this later.
    """
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "tool_name": tool_name,
        "inputs": inputs,
        "outputs": outputs,
        "latency_ms": latency_ms,
        "grounded": grounded,
    }
    with _EVAL_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
    record_tool_call(tool_name)


def load_eval_log() -> list[dict]:
    """Read eval_log.jsonl and return all entries as a list of dicts.

    Silently skips lines that are not valid JSON.
    """
    if not _EVAL_LOG_PATH.exists():
        return []
    entries: list[dict] = []
    with _EVAL_LOG_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def summarise_eval_log() -> dict[str, Any]:
    """Summarise the evaluation log.

    Returns
    -------
    dict
        Keys: total_calls (int), calls_per_tool (dict[str, int]),
        avg_latency_ms (float), grounded_rate (float 0-1).
    """
    entries = load_eval_log()
    if not entries:
        return {
            "total_calls": 0,
            "calls_per_tool": {},
            "avg_latency_ms": 0.0,
            "grounded_rate": 1.0,
        }

    calls_per_tool: dict[str, int] = {}
    total_latency = 0.0
    grounded_count = 0

    for e in entries:
        tool = e.get("tool_name", "unknown")
        calls_per_tool[tool] = calls_per_tool.get(tool, 0) + 1
        total_latency += float(e.get("latency_ms", 0.0))
        if e.get("grounded", True):
            grounded_count += 1

    n = len(entries)
    return {
        "total_calls": n,
        "calls_per_tool": calls_per_tool,
        "avg_latency_ms": round(total_latency / n, 2),
        "grounded_rate": round(grounded_count / n, 4),
    }
