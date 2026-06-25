"""
RAG (Retrieval-Augmented Generation) pipeline for Portfolio Sentinel.

Provides two operations:

1. embed_and_store — takes raw news articles, encodes their text using a local
   sentence-transformers model, and upserts the embeddings into a ChromaDB
   collection keyed by ticker symbol.

2. retrieve_context — takes a natural-language query, encodes it with the same
   model, and returns the top-k most semantically similar article chunks from
   the ticker's collection.

ChromaDB persists to a local directory (``chroma_store/`` in the project root)
so embeddings survive server restarts without re-fetching or re-encoding.

The sentence-transformers model (``all-MiniLM-L6-v2``) is loaded lazily on
first use to avoid slowing down server startup.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import chromadb

logger = logging.getLogger("portfolio_sentinel")

_CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_store"
_CHROMA_DIR.mkdir(exist_ok=True)

_client = chromadb.PersistentClient(path=str(_CHROMA_DIR))

_model: Any = None


def _get_model() -> Any:
    """Lazy-load the sentence-transformers model on first use."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _article_id(article: dict) -> str:
    """Derive a stable, unique ID for an article from its URL or content."""
    key = article.get("url") or (article.get("title", "") + article.get("published_at", ""))
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:16]


def _article_text(article: dict) -> str:
    """Concatenate title and description into a single embeddable string."""
    title = article.get("title") or ""
    desc = article.get("description") or ""
    return f"{title}. {desc}".strip(". ")


def embed_and_store(ticker: str, articles: list[dict]) -> int:
    """Embed article text and upsert into the ticker's ChromaDB collection.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker symbol used as the collection namespace.
    articles:
        List of article dicts as returned by ``fetch_news``. Each must have
        at minimum a ``title`` key.

    Returns
    -------
    int
        Number of documents stored (after deduplication by article ID).
    """
    if not articles:
        return 0

    model = _get_model()
    collection = _client.get_or_create_collection(f"news_{ticker}")

    texts = [_article_text(a) for a in articles]
    ids = [_article_id(a) for a in articles]

    # De-duplicate within this batch (same URL appearing twice in one fetch)
    seen: set[str] = set()
    unique_texts: list[str] = []
    unique_ids: list[str] = []
    for doc_id, text in zip(ids, texts):
        if doc_id not in seen:
            seen.add(doc_id)
            unique_texts.append(text)
            unique_ids.append(doc_id)

    embeddings = model.encode(unique_texts, show_progress_bar=False)

    collection.upsert(
        documents=unique_texts,
        embeddings=[e.tolist() for e in embeddings],
        ids=unique_ids,
    )
    return len(unique_ids)


def retrieve_context(ticker: str, query: str, top_k: int = 3) -> list[str]:
    """Retrieve the most semantically similar article chunks for a query.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker symbol identifying the ChromaDB collection.
    query:
        Natural-language question or topic to search for.
    top_k:
        Maximum number of chunks to return. Automatically capped at the
        number of documents in the collection.

    Returns
    -------
    list[str]
        List of article text strings ranked by semantic similarity to the
        query. Returns an empty list if the collection does not exist or
        contains no documents.
    """
    try:
        collection = _client.get_collection(f"news_{ticker}")
    except Exception:
        return []

    count = collection.count()
    if count == 0:
        return []

    model = _get_model()
    query_embedding = model.encode([query], show_progress_bar=False)[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, count),
    )
    docs: list[str] = results.get("documents", [[]])[0]
    return docs
