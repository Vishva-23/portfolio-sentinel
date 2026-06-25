"""
FastAPI application entry point for Portfolio Sentinel.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

from agent.agent import MODEL, run_agent  # noqa: E402
from eval.eval_logger import (  # noqa: E402
    get_request_tools,
    load_eval_log,
    reset_request_tools,
    set_request_id,
    summarise_eval_log,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent
_STATIC_DIR = _PROJECT_ROOT / "static"
_LOGS_DIR = _PROJECT_ROOT / "logs"
_LOGS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Logging — two handlers: rotating JSON file + plain console
# ---------------------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if isinstance(record.msg, dict):
            return json.dumps(record.msg, default=str)
        return json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
            },
            default=str,
        )


_file_handler = logging.handlers.RotatingFileHandler(
    _LOGS_DIR / "app.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_file_handler.setFormatter(_JsonFormatter())

# Dedicated logger that writes only JSON to the rotating file
_file_logger = logging.getLogger("portfolio_sentinel.file")
_file_logger.setLevel(logging.INFO)
_file_logger.addHandler(_file_handler)
_file_logger.propagate = False

# Console logger for human-readable startup/error messages
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger("portfolio_sentinel")

# ---------------------------------------------------------------------------
# Default portfolio
# ---------------------------------------------------------------------------

DEFAULT_TICKERS: list[str] = ["AAPL", "MSFT", "NVDA", "GOOGL", "A5G.IR"]
DEFAULT_WEIGHTS: list[float] = [0.25, 0.20, 0.25, 0.20, 0.10]

# May be replaced at startup if A5G.IR fails to fetch
_ACTIVE_TICKERS: list[str] = list(DEFAULT_TICKERS)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    tickers: list[str] = Field(default_factory=lambda: list(_ACTIVE_TICKERS))
    weights: list[float] = Field(default_factory=lambda: list(DEFAULT_WEIGHTS))
    conversation_history: list[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    conversation_history: list[dict]
    request_id: str


class BenchmarkRequest(BaseModel):
    tickers: list[str] = Field(default_factory=lambda: list(_ACTIVE_TICKERS))
    weights: list[float] = Field(default_factory=lambda: list(DEFAULT_WEIGHTS))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Portfolio Sentinel", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.on_event("startup")
async def startup_event() -> None:
    global _ACTIVE_TICKERS
    from tools.market_data import get_prices

    if "A5G.IR" in DEFAULT_TICKERS:
        try:
            get_prices("A5G.IR", 5)
        except Exception as exc:
            logger.warning(
                "A5G.IR price fetch failed (%s) — substituting CRH in active tickers",
                exc,
            )
            _ACTIVE_TICKERS = [
                "CRH" if t == "A5G.IR" else t for t in DEFAULT_TICKERS
            ]

    ticker_str = ", ".join(
        f"{t} ({w:.0%})" for t, w in zip(_ACTIVE_TICKERS, DEFAULT_WEIGHTS)
    )
    logger.info("Portfolio Sentinel started. Active portfolio: %s", ticker_str)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "model": MODEL,
            "tickers_loaded": len(_ACTIVE_TICKERS),
            "cache_populated": (_PROJECT_ROOT / "fallback_cache.json").exists(),
            "default_tickers": _ACTIVE_TICKERS,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    request_id = str(uuid.uuid4())
    start = time.time()
    status = "ok"
    error_detail: str | None = None

    set_request_id(request_id)
    reset_request_tools()

    try:
        reply, updated_history = run_agent(
            tickers=request.tickers,
            weights=request.weights,
            message=request.message,
            conversation_history=request.conversation_history,
        )
    except Exception as exc:
        status = "error"
        error_detail = str(exc)
        raise
    finally:
        latency_ms = round((time.time() - start) * 1000, 2)
        tools_called = get_request_tools()
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "tickers": request.tickers,
            "message_preview": request.message[:80],
            "tool_calls_made": tools_called,
            "latency_ms": latency_ms,
            "status": status,
        }
        if error_detail:
            log_entry["error"] = error_detail
        _file_logger.info(log_entry)

    return ChatResponse(
        reply=reply,
        conversation_history=updated_history,
        request_id=request_id,
    )


@app.get("/eval/summary")
async def eval_summary() -> JSONResponse:
    return JSONResponse(summarise_eval_log())


@app.get("/eval/log")
async def eval_log_endpoint() -> JSONResponse:
    entries = load_eval_log()
    return JSONResponse(entries[-50:])


@app.post("/benchmark")
async def benchmark(request: BenchmarkRequest) -> JSONResponse:
    from benchmark.runner import run_benchmark

    results = run_benchmark(tickers=request.tickers, weights=request.weights)
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    return JSONResponse(
        {
            "results": results,
            "summary": {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": round(passed / total, 4) if total else 0.0,
            },
        }
    )
