# tests/test_store.py
from pathlib import Path

import pytest

from tripmate.rag.chunker import load_all
from tripmate.rag.store import ChromaStore

CITIES = {"tokyo", "reykjavik", "bangkok", "barcelona"}


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> ChromaStore:
    path = tmp_path_factory.mktemp("chroma")
    store = ChromaStore(path=str(path), embedding_model="BAAI/bge-small-en-v1.5")
    store.add(load_all(Path("data/destinations")))
    return store


def test_store_holds_every_chunk_from_the_data_pack(store: ChromaStore):
    assert store.count() == 20


def test_re_adding_the_same_chunks_adds_nothing(store: ChromaStore):
    assert store.add(load_all(Path("data/destinations"))) == 0


def test_visa_query_retrieves_a_visa_section(store: ChromaStore):
    results = store.search("do I need a visa for Japan?", k=3, min_score=0.0)
    assert any("VISA" in chunk.section for chunk in results)


def test_city_filter_restricts_results_to_that_city(store: ChromaStore):
    results = store.search("what should I pack?", k=5, city="bangkok", min_score=0.0)
    assert {chunk.city for chunk in results} == {"bangkok"}


def test_city_filter_is_case_insensitive(store: ChromaStore):
    results = store.search("packing", k=5, city="Bangkok", min_score=0.0)
    assert {chunk.city for chunk in results} == {"bangkok"}


def test_scores_are_similarities_between_zero_and_one(store: ChromaStore):
    results = store.search("visa requirements", k=3, min_score=0.0)
    assert all(0.0 <= chunk.score <= 1.0 for chunk in results)


def test_results_are_ordered_by_descending_similarity(store: ChromaStore):
    scores = [c.score for c in store.search("local customs", k=4, min_score=0.0)]
    assert scores == sorted(scores, reverse=True)


def test_high_score_floor_filters_out_unrelated_queries(store: ChromaStore):
    assert store.search("how do I refinance a mortgage?", k=4, min_score=0.9) == []


def test_unknown_city_filter_returns_nothing(store: ChromaStore):
    assert store.search("visa", k=4, city="paris", min_score=0.0) == []


def test_top_k_limits_the_number_of_results(store: ChromaStore):
    assert len(store.search("travel", k=2, min_score=0.0)) <= 2
