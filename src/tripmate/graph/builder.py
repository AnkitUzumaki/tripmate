"""Compile the agent graph.

    START -> check_cache -(hit)------------------------> END
                         -(miss)-> agent <-> tools -> collect_refs
                                     |
                                  (no tool calls)
                                     v
                          validate_citations -> store_cache -> END

`tools_condition` is LangGraph's built-in router: it inspects the last message for tool
calls and routes accordingly. Dynamic tool selection is therefore an edge in the graph,
not a branch inside a method.
"""

from __future__ import annotations

from typing import Callable

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tripmate.core.cache import SemanticCache
from tripmate.core.trace import Tracer
from tripmate.graph.nodes import (
    CACHE_HIT,
    PROCEED,
    make_agent_node,
    make_check_cache,
    make_collect_refs,
    make_store_cache,
    make_validate_citations,
    route_after_cache,
)
from tripmate.graph.state import AgentState
from tripmate.graph.tool_adapter import to_langchain_tools
from tripmate.tools.registry import ToolRegistry


def build_graph(
    model,
    registry: ToolRegistry,
    tracer_of: Callable[[], Tracer],
    cache: SemanticCache | None = None,
    checkpointer=None,
    model_name: str = "",
):
    """Wire and compile the graph.

    `model` must already have tools bound (see `bind_model`), because the agent node
    calls it directly and the whole point is that the model — not our code — decides
    which tools to request.
    """
    tools = to_langchain_tools(registry)

    graph = StateGraph(AgentState)
    graph.add_node("check_cache", make_check_cache(cache, tracer_of))
    graph.add_node("agent", make_agent_node(model, tracer_of, model_name))
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("collect_refs", make_collect_refs(tracer_of))
    graph.add_node("validate_citations", make_validate_citations(tracer_of))
    graph.add_node("store_cache", make_store_cache(cache))

    graph.add_edge(START, "check_cache")
    graph.add_conditional_edges(
        "check_cache",
        route_after_cache,
        {CACHE_HIT: END, PROCEED: "agent"},
    )
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: "validate_citations"},
    )
    graph.add_edge("tools", "collect_refs")
    graph.add_edge("collect_refs", "agent")
    graph.add_edge("validate_citations", "store_cache")
    graph.add_edge("store_cache", END)

    return graph.compile(checkpointer=checkpointer)


def bind_model(model, registry: ToolRegistry):
    """Attach the registry's tool schemas to the model."""
    return model.bind_tools(to_langchain_tools(registry))
