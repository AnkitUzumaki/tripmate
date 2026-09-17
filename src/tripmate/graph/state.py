"""The graph's shared state.

Every node is a function of this TypedDict and returns a partial update. The previous
implementation threaded a mutable message list and an `allowed_refs` set through three
private methods by hand; as graph state both are inspectable at every step and each
node becomes independently testable.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """State carried between nodes for one turn.

    `messages` uses LangGraph's `add_messages` reducer, so nodes return only the
    messages they add and the framework appends them. Conversation history across
    turns comes from the checkpointer, not from this field being manually reloaded.
    """

    messages: Annotated[list[AnyMessage], add_messages]

    # The user's question for this turn, kept separately from `messages` because the
    # cache keys on it and the last human message is not always the current query
    # once history is reloaded.
    query: str

    # `city/SECTION` refs of chunks actually retrieved this turn. validate_citations
    # strips any citation not in here — this is the anti-fabrication whitelist.
    # A list rather than a set because graph state is serialised by the checkpointer.
    allowed_refs: list[str]

    # Refs as plain strings, not Citation objects: the checkpointer serialises
    # state, and custom types there are forward-incompatible. The facade
    # rebuilds Citation at the boundary.
    citations: list[str]
    answer: str
    was_cached: bool

    # Decided once by check_cache and reused by store_cache. Counting messages in
    # both places was wrong: a tool round inflates the count, so the store node
    # silently stopped caching any answer that used a tool.
    is_first_turn: bool
