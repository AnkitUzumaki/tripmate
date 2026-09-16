"""Vector store. One Protocol, one implementation.

The Protocol is the swap point — Chroma to pgvector or Qdrant changes this file
and nothing else. A second implementation today would be speculative.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import chromadb
from fastembed import TextEmbedding

from tripmate.config import Settings, get_settings
from tripmate.models import Chunk

COLLECTION_NAME = "destinations"
COSINE_SPACE = {"hnsw:space": "cosine"}


@runtime_checkable
class VectorStore(Protocol):
    def add(self, chunks: list[Chunk]) -> int: ...

    def search(
        self, query: str, k: int, city: str | None = None, min_score: float = 0.0
    ) -> list[Chunk]: ...

    def count(self) -> int: ...


class ChromaStore:
    """Persistent Chroma collection with local ONNX embeddings."""

    def __init__(
        self,
        path: str,
        embedding_model: str,
        collection: str = COLLECTION_NAME,
    ) -> None:
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata=COSINE_SPACE
        )
        self._embedder = TextEmbedding(model_name=embedding_model)

    @property
    def embedder(self) -> TextEmbedding:
        """The embedding model, shared with the semantic cache so both use one instance."""
        return self._embedder

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._embedder.embed(texts)]

    def count(self) -> int:
        return self._collection.count()

    def add(self, chunks: list[Chunk]) -> int:
        """Add only chunks whose content hash is not already stored. Returns how many."""
        if not chunks:
            return 0

        known = set(self._collection.get(ids=[c.id for c in chunks])["ids"])
        fresh = [c for c in chunks if c.id not in known]
        if not fresh:
            return 0

        self._collection.add(
            ids=[c.id for c in fresh],
            embeddings=self._embed([c.text for c in fresh]),
            documents=[c.text for c in fresh],
            metadatas=[
                {"city": c.city, "section": c.section, "source_file": c.source_file}
                for c in fresh
            ],
        )
        return len(fresh)

    def search(
        self, query: str, k: int, city: str | None = None, min_score: float = 0.0
    ) -> list[Chunk]:
        """Semantic search, optionally pre-filtered by city.

        The city filter is what keeps precision usable as the corpus grows: without it,
        20k chunks return another city's visa section for a Tokyo query.
        """
        if not query.strip():
            return []

        response = self._collection.query(
            query_embeddings=self._embed([query]),
            n_results=k,
            where={"city": city.strip().lower()} if city else None,
        )

        if not response["ids"] or not response["ids"][0]:
            return []

        results: list[Chunk] = []
        for chunk_id, document, metadata, distance in zip(
            response["ids"][0],
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
        ):
            similarity = 1.0 - float(distance)
            if similarity < min_score:
                continue
            results.append(
                Chunk(
                    id=chunk_id,
                    text=document,
                    city=str(metadata["city"]),
                    section=str(metadata["section"]),
                    source_file=str(metadata["source_file"]),
                    score=similarity,
                )
            )
        return sorted(results, key=lambda c: c.score, reverse=True)


def build_store(settings: Settings | None = None) -> ChromaStore:
    settings = settings or get_settings()
    return ChromaStore(
        path=settings.chroma_path, embedding_model=settings.embedding_model
    )
