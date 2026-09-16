"""RAG tool over the destination knowledge base."""

from __future__ import annotations

import threading
from typing import Annotated

from tripmate.config import get_settings
from tripmate.models import ToolResult
from tripmate.rag.store import VectorStore, build_store
from tripmate.tools.registry import tool

SUPPORTED_CITIES: tuple[str, ...] = ("tokyo", "reykjavik", "bangkok", "barcelona")
TOOL_NAME = "search_destination_guide"

_store: VectorStore | None = None
_store_lock = threading.Lock()


def set_store(store: VectorStore) -> None:
    """Inject a store. Used by tests and by application startup."""
    global _store
    _store = store


def reset_store() -> None:
    global _store
    _store = None


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = build_store()
    return _store


@tool
def search_destination_guide(
    query: Annotated[
        str,
        "The travel question or topic, e.g. 'visa requirements' or 'what to pack'.",
    ],
    city: Annotated[
        str | None,
        "Destination city to restrict the search to. One of: tokyo, reykjavik, "
        "bangkok, barcelona. Omit when the question is not about a specific city.",
    ] = None,
) -> ToolResult:
    """Search the destination knowledge base for visa, weather-season, customs, packing and safety guidance."""
    if not query or not query.strip():
        return ToolResult.no_data(TOOL_NAME, "empty query")

    settings = get_settings()
    chunks = _get_store().search(
        query=query,
        k=settings.rag_top_k,
        city=city,
        min_score=settings.rag_min_score,
    )

    if not chunks:
        target = city.strip().lower() if city else "that topic"
        return ToolResult.no_data(
            TOOL_NAME,
            f"no destination guide content found for {target}",
            available_cities=list(SUPPORTED_CITIES),
        )

    return ToolResult.ok(
        TOOL_NAME,
        {
            "chunks": [
                {
                    "ref": chunk.ref,
                    "city": chunk.city,
                    "section": chunk.section,
                    "text": chunk.text,
                    "score": round(chunk.score, 4),
                }
                for chunk in chunks
            ]
        },
    )
