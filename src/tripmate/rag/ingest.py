"""Idempotent ingest of the destination data pack into the vector store."""

from __future__ import annotations

from pathlib import Path

from tripmate.config import Settings, get_settings
from tripmate.rag.chunker import load_all
from tripmate.rag.store import build_store


def ingest(directory: str | Path | None = None, settings: Settings | None = None) -> int:
    """Ingest every guide. Returns the number of chunks newly added."""
    settings = settings or get_settings()
    chunks = load_all(Path(directory or settings.destinations_dir))
    return build_store(settings).add(chunks)


def main() -> None:
    settings = get_settings()
    added = ingest(settings=settings)
    total = build_store(settings).count()
    print(f"ingested {added} new chunk(s); {total} total in {settings.chroma_path}")


if __name__ == "__main__":
    main()
