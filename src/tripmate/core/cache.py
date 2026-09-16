"""Semantic query cache.

Answers are replayed when a new query is close enough to one already answered.
This is the implemented answer to "how do you avoid redundant LLM calls", rather
than a paragraph about it.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Protocol

from tripmate.config import Settings, get_settings
from tripmate.db import CacheRow, Database
from tripmate.models import Citation

FLOAT_FORMAT = "f"


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> Any: ...


@dataclass(frozen=True)
class CachedAnswer:
    answer: str
    citations: list[Citation]
    similarity: float


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    magnitude = math.sqrt(sum(a * a for a in left)) * math.sqrt(
        sum(b * b for b in right)
    )
    return numerator / magnitude if magnitude else 0.0


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}{FLOAT_FORMAT}", *vector)


def _unpack(blob: bytes) -> list[float]:
    count = len(blob) // struct.calcsize(FLOAT_FORMAT)
    return list(struct.unpack(f"<{count}{FLOAT_FORMAT}", blob))


class SemanticCache:
    def __init__(
        self, db: Database, embedder: Embedder, threshold: float, enabled: bool = True
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._threshold = threshold
        self._enabled = enabled

    def _embed(self, text: str) -> list[float]:
        vector = next(iter(self._embedder.embed([text])))
        return [float(value) for value in vector]

    def lookup(self, query: str) -> CachedAnswer | None:
        if not self._enabled or not query.strip():
            return None

        target = self._embed(query)
        with self._db.session() as session:
            rows = session.query(CacheRow).all()

        best: CachedAnswer | None = None
        for row in rows:
            similarity = cosine(target, _unpack(row.embedding))
            if similarity < self._threshold:
                continue
            if best is None or similarity > best.similarity:
                best = CachedAnswer(
                    answer=row.answer,
                    citations=[Citation(**c) for c in (row.citations or [])],
                    similarity=similarity,
                )
        return best

    def store(self, query: str, answer: str, citations: list[Citation]) -> None:
        if not self._enabled or not query.strip():
            return

        with self._db.session() as session:
            session.add(
                CacheRow(
                    query_text=query,
                    embedding=_pack(self._embed(query)),
                    answer=answer,
                    citations=[c.model_dump() for c in citations],
                )
            )


def build_cache(db: Database, settings: Settings | None = None) -> SemanticCache:
    from fastembed import TextEmbedding

    settings = settings or get_settings()
    return SemanticCache(
        db=db,
        embedder=TextEmbedding(model_name=settings.embedding_model),
        threshold=settings.semantic_cache_threshold,
        enabled=settings.semantic_cache_enabled,
    )
