from pathlib import Path

import pytest

from tripmate.rag.chunker import chunk_id, load_all, parse_guide

GUIDE = """DESTINATION GUIDE: TOKYO, JAPAN

Note: This is a simplified reference document.

VISA & ENTRY
Many nationalities can enter Japan visa-free for short tourist stays.

BEST TIME TO VISIT
Spring is popular for cherry blossoms.

PACKING TIPS
Layered clothing works well given seasonal variation.
"""


@pytest.fixture()
def guide_file(tmp_path: Path) -> Path:
    path = tmp_path / "tokyo.txt"
    path.write_text(GUIDE, encoding="utf-8")
    return path


def test_parse_guide_returns_one_chunk_per_section(guide_file: Path):
    assert len(parse_guide(guide_file)) == 3


def test_parse_guide_extracts_city_from_the_title_line(guide_file: Path):
    assert {c.city for c in parse_guide(guide_file)} == {"tokyo"}


def test_parse_guide_preserves_section_headings(guide_file: Path):
    sections = [c.section for c in parse_guide(guide_file)]
    assert sections == ["VISA & ENTRY", "BEST TIME TO VISIT", "PACKING TIPS"]


def test_chunk_text_embeds_the_city_name_for_retrieval(guide_file: Path):
    first = parse_guide(guide_file)[0]
    assert first.text.startswith("Tokyo — VISA & ENTRY")


def test_parse_guide_excludes_the_disclaimer_note(guide_file: Path):
    assert all("simplified reference" not in c.text for c in parse_guide(guide_file))


def test_chunk_ids_are_stable_across_calls(guide_file: Path):
    assert [c.id for c in parse_guide(guide_file)] == [c.id for c in parse_guide(guide_file)]


def test_chunk_id_changes_when_body_changes():
    assert chunk_id("tokyo", "PACKING TIPS", "a") != chunk_id("tokyo", "PACKING TIPS", "b")


def test_load_all_reads_every_txt_file(tmp_path: Path):
    (tmp_path / "tokyo.txt").write_text(GUIDE, encoding="utf-8")
    (tmp_path / "osaka.txt").write_text(GUIDE.replace("TOKYO", "OSAKA"), encoding="utf-8")

    assert len(load_all(tmp_path)) == 6


def test_load_all_raises_when_directory_is_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_all(tmp_path / "nope")


def test_real_data_pack_yields_twenty_chunks():
    chunks = load_all(Path("data/destinations"))
    assert len(chunks) == 20
    assert {c.city for c in chunks} == {"tokyo", "reykjavik", "bangkok", "barcelona"}
