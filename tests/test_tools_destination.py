from pathlib import Path

import pytest

from tripmate.rag.chunker import load_all
from tripmate.rag.store import ChromaStore
from tripmate.tools.destination import (
    SUPPORTED_CITIES,
    reset_store,
    search_destination_guide,
    set_store,
)


@pytest.fixture(autouse=True, scope="module")
def _store(tmp_path_factory):
    store = ChromaStore(
        path=str(tmp_path_factory.mktemp("chroma_tool")),
        embedding_model="BAAI/bge-small-en-v1.5",
    )
    store.add(load_all(Path("data/destinations")))
    set_store(store)
    yield
    reset_store()


def test_returns_ok_status_for_a_covered_topic():
    assert search_destination_guide("visa requirements for Japan").status == "ok"


def test_result_data_contains_chunk_dicts_with_refs():
    result = search_destination_guide("visa requirements for Japan")
    assert "ref" in result.data["chunks"][0]


def test_city_argument_restricts_results():
    result = search_destination_guide("what should I pack?", city="reykjavik")
    assert {c["city"] for c in result.data["chunks"]} == {"reykjavik"}


def test_unknown_city_returns_no_data_with_supported_cities():
    result = search_destination_guide("visa rules", city="paris")
    assert result.status == "no_data"
    assert set(result.data["available_cities"]) == set(SUPPORTED_CITIES)


def test_unknown_city_reason_names_the_city():
    assert "paris" in search_destination_guide("visa", city="paris").reason


def test_empty_query_returns_no_data():
    assert search_destination_guide("   ").status == "no_data"


def test_tool_carries_a_derived_schema():
    schema = search_destination_guide._tool_spec.schema
    assert schema["function"]["name"] == "search_destination_guide"
    assert "city" in schema["function"]["parameters"]["properties"]


def test_city_is_optional_in_the_schema():
    schema = search_destination_guide._tool_spec.schema
    assert "city" not in schema["function"]["parameters"].get("required", [])
