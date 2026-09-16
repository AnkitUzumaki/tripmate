import pytest

from tripmate.core.cache import SemanticCache, cosine
from tripmate.db import Database
from tripmate.models import Citation


class StubEmbedder:
    """Maps known phrases to fixed vectors so similarity is exact and testable."""

    VECTORS = {
        "visa for japan": [1.0, 0.0, 0.0],
        "japan visa rules": [0.99, 0.141, 0.0],   # cosine ~0.99 with the above
        "what to pack for iceland": [0.0, 1.0, 0.0],
    }

    def embed(self, texts):
        return [self.VECTORS.get(text.strip().lower(), [0.0, 0.0, 1.0])
                for text in texts]


@pytest.fixture()
def cache(tmp_path) -> SemanticCache:
    db = Database(url=f"sqlite:///{tmp_path}/cache.db")
    db.create_all()
    return SemanticCache(db=db, embedder=StubEmbedder(), threshold=0.95)


def test_cosine_of_identical_vectors_is_one():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_of_a_zero_vector_is_zero():
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_lookup_misses_on_an_empty_cache(cache: SemanticCache):
    assert cache.lookup("visa for japan") is None


def test_lookup_hits_on_the_exact_same_query(cache: SemanticCache):
    cache.store("visa for japan", "No visa needed.", [])
    assert cache.lookup("visa for japan").answer == "No visa needed."


def test_lookup_hits_on_a_semantically_similar_query(cache: SemanticCache):
    cache.store("visa for japan", "No visa needed.", [])
    hit = cache.lookup("japan visa rules")

    assert hit is not None
    assert hit.similarity >= 0.95


def test_lookup_misses_on_an_unrelated_query(cache: SemanticCache):
    cache.store("visa for japan", "No visa needed.", [])
    assert cache.lookup("what to pack for iceland") is None


def test_citations_survive_the_round_trip(cache: SemanticCache):
    cache.store("visa for japan", "No visa needed.",
                [Citation(city="tokyo", section="VISA & ENTRY")])

    assert cache.lookup("visa for japan").citations[0].ref == "tokyo/VISA & ENTRY"


def test_a_disabled_cache_never_hits(tmp_path):
    db = Database(url=f"sqlite:///{tmp_path}/off.db")
    db.create_all()
    disabled = SemanticCache(db=db, embedder=StubEmbedder(), threshold=0.95,
                             enabled=False)

    disabled.store("visa for japan", "answer", [])

    assert disabled.lookup("visa for japan") is None


def test_empty_query_is_never_cached(cache: SemanticCache):
    cache.store("   ", "answer", [])
    assert cache.lookup("   ") is None
