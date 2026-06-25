"""
FastAPI application entry point for Portfolio Sentinel.
"""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from agent.agent import run_agent  # noqa: E402  (must load .env before importing agent)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("portfolio_sentinel")

# ---------------------------------------------------------------------------
# Default portfolio
# ---------------------------------------------------------------------------

DEFAULT_TICKERS: list[str] = ["AAPL", "MSFT", "NVDA", "GOOGL", "CRH"]
DEFAULT_WEIGHTS: list[float] = [0.25, 0.20, 0.25, 0.20, 0.10]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    tickers: list[str] = DEFAULT_TICKERS
    weights: list[float] = DEFAULT_WEIGHTS
    conversation_history: list[dict] = []


class ChatResponse(BaseModel):
    reply: str
    conversation_history: list[dict]


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

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.on_event("startup")
async def startup_event() -> None:
    ticker_str = ", ".join(
        f"{t} ({w:.0%})" for t, w in zip(DEFAULT_TICKERS, DEFAULT_WEIGHTS)
    )
    logger.info("Portfolio Sentinel started. Default portfolio: %s", ticker_str)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    reply, updated_history = run_agent(
        tickers=request.tickers,
        weights=request.weights,
        message=request.message,
        conversation_history=request.conversation_history,
    )
    return ChatResponse(reply=reply, conversation_history=updated_history)
