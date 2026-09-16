import pytest

from tripmate.db import Database, SessionStore
from tripmate.models import TraceEvent


@pytest.fixture()
def store(tmp_path) -> SessionStore:
    db = Database(url=f"sqlite:///{tmp_path}/test.db")
    db.create_all()
    return SessionStore(db)


def test_history_is_empty_for_a_new_session(store: SessionStore):
    assert store.history("s1") == []


def test_add_turn_returns_an_increasing_id(store: SessionStore):
    first = store.add_turn("s1", "user", "hello")
    second = store.add_turn("s1", "assistant", "hi")
    assert second > first


def test_history_returns_messages_oldest_first(store: SessionStore):
    store.add_turn("s1", "user", "one")
    store.add_turn("s1", "assistant", "two")

    assert [m["content"] for m in store.history("s1")] == ["one", "two"]


def test_history_uses_llm_message_shape(store: SessionStore):
    store.add_turn("s1", "user", "hello")
    assert store.history("s1")[0] == {"role": "user", "content": "hello"}


def test_history_is_scoped_to_one_session(store: SessionStore):
    store.add_turn("s1", "user", "one")
    store.add_turn("s2", "user", "two")

    assert len(store.history("s1")) == 1


def test_history_limit_keeps_the_most_recent_turns(store: SessionStore):
    for index in range(5):
        store.add_turn("s1", "user", f"m{index}")

    recent = store.history("s1", limit=2)
    assert [m["content"] for m in recent] == ["m3", "m4"]


def test_turn_records_cost_and_token_counts(store: SessionStore):
    turn_id = store.add_turn("s1", "assistant", "hi", prompt_tokens=10,
                             completion_tokens=5, cost_usd=0.002)

    with store.db.session() as session:
        from tripmate.db import TurnRow
        row = session.get(TurnRow, turn_id)
        assert (row.prompt_tokens, row.cost_usd) == (10, 0.002)


def test_trace_events_are_persisted_against_their_turn(store: SessionStore):
    turn_id = store.add_turn("s1", "assistant", "hi")
    store.add_trace(turn_id, [
        TraceEvent(seq=1, event_type="LLM_CALL", timestamp=1.0, payload={"a": 1})
    ])

    with store.db.session() as session:
        from tripmate.db import TraceRow
        rows = session.query(TraceRow).filter_by(turn_id=turn_id).all()
        assert rows[0].event_type == "LLM_CALL"


def test_ensure_session_is_idempotent(store: SessionStore):
    store.ensure_session("s1")
    store.ensure_session("s1")

    with store.db.session() as session:
        from tripmate.db import SessionRow
        assert session.query(SessionRow).count() == 1
