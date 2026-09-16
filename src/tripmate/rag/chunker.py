"""Section-aware chunking of destination guides.

Each guide has a title line and five ALL-CAPS section headings. One section is one
chunk: they are already semantically self-contained, so no overlap is needed.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from tripmate.models import Chunk

SECTION_HEADING_RE = re.compile(r"^([A-Z][A-Z&\s]{3,})$")
TITLE_PREFIX = "DESTINATION GUIDE:"
ID_LENGTH = 16


def chunk_id(city: str, section: str, body: str) -> str:
    """Deterministic content hash. Re-ingesting unchanged content is a no-op."""
    digest = hashlib.sha256(f"{city}|{section}|{body}".encode("utf-8"))
    return digest.hexdigest()[:ID_LENGTH]


def _extract_city(lines: list[str]) -> str:
    for line in lines:
        if line.startswith(TITLE_PREFIX):
            location = line[len(TITLE_PREFIX):].strip()
            return location.split(",")[0].strip().lower()
    raise ValueError(f"no '{TITLE_PREFIX}' title line found")


def _split_sections(lines: list[str]) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    for line in lines:
        stripped = line.strip()
        if SECTION_HEADING_RE.match(stripped):
            sections.append((stripped, []))
        elif sections and stripped:
            sections[-1][1].append(stripped)
    return [(name, " ".join(body)) for name, body in sections if body]


def parse_guide(path: Path) -> list[Chunk]:
    """Parse one destination guide into one chunk per section."""
    lines = path.read_text(encoding="utf-8").splitlines()
    city = _extract_city(lines)
    title = city.title()

    return [
        Chunk(
            id=chunk_id(city, section, body),
            text=f"{title} — {section}\n{body}",
            city=city,
            section=section,
            source_file=path.name,
        )
        for section, body in _split_sections(lines)
    ]


def load_all(directory: Path) -> list[Chunk]:
    """Parse every `*.txt` guide in `directory`, ordered by filename."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"destinations directory not found: {directory}")

    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.txt")):
        chunks.extend(parse_guide(path))
    return chunks
