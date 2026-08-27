"""Assemble the workflow graph. Stubs pass state through; edges are real."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from assist.graph.edges import after_guard, after_merge, after_retrieve, route_reply
from assist.graph.state import TurnState

_CYCLE_NODES = frozenset({"retrieve", "broaden_constraints"})

# Identity stubs. Named here so later tasks own `nodes/<stage>.py` until T24.
_PASSTHROUGH_NODES = (
    "load_session",
    "guard",
    "intent",
    "merge_constraints",
    "resolve_people",
    "broaden_constraints",
    "rank",
    "validate_availability",
    "reply_template",
    "reply_generative",
    "reply_clarify",
    "reply_refusal",
    "sanitize_picks",
    "mint_chips",
    "persist",
)


async def passthrough(_state: TurnState) -> dict[str, object]:
    """Identity stub. T24 replaces each named node with the real stage."""
    return {}


async def retrieve(state: TurnState) -> dict[str, object]:
    """Stub retrieve. Increments the cycle counter; later tasks fill candidates."""
    attempts = int(state.get("retrieve_attempts") or 0)
    return {"retrieve_attempts": attempts + 1}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_mermaid_path() -> Path:
    return _repo_root() / "docs" / "graph.mmd"


def _adjacency(compiled: CompiledStateGraph[Any, Any, Any, Any]) -> dict[str, list[str]]:
    adj: dict[str, list[str]] = defaultdict(list)
    for edge in compiled.get_graph().edges:
        adj[edge.source].append(edge.target)
    return dict(adj)


def _simple_cycles(adj: Mapping[str, list[str]]) -> list[tuple[str, ...]]:
    """Directed simple cycles. Each cycle is reported once, from its min node."""
    found: list[tuple[str, ...]] = []
    for start in sorted(adj):
        stack: list[tuple[str, list[str], set[str]]] = [(start, [start], {start})]
        while stack:
            current, path, seen = stack.pop()
            for nxt in adj.get(current, ()):
                if nxt == start and len(path) >= 2:
                    found.append(tuple(path))
                elif nxt not in seen and nxt > start:
                    stack.append((nxt, [*path, nxt], seen | {nxt}))
    return found


def _assert_workflow_invariants(compiled: CompiledStateGraph[Any, Any, Any, Any]) -> None:
    # False, not None: None would inherit a parent checkpointer as a subgraph.
    if compiled.checkpointer is not False:
        msg = "assist graph must compile with checkpointer=False"
        raise RuntimeError(msg)

    stage_nodes = {name for name in compiled.nodes if not str(name).startswith("__")}
    participants: set[str] = set()
    for cycle in _simple_cycles(_adjacency(compiled)):
        nodes = {name for name in cycle if name in stage_nodes}
        if not nodes:
            continue
        if nodes != _CYCLE_NODES:
            msg = f"unexpected graph cycle among {sorted(nodes)}"
            raise RuntimeError(msg)
        participants.update(nodes)
    if participants != _CYCLE_NODES:
        msg = "graph must contain exactly the retrieve↔broaden cycle"
        raise RuntimeError(msg)


def _add_node(graph: StateGraph[TurnState], name: str, fn: Any) -> None:
    """LangGraph `_Node` overloads are sync; mypy rejects `async def` against TurnState.

    `fn` is Any for that seam only. Runtime still registers the async callable.
    """
    graph.add_node(name, fn)


def build_graph() -> CompiledStateGraph[TurnState, None, TurnState, TurnState]:
    graph: StateGraph[TurnState] = StateGraph(TurnState)

    for name in _PASSTHROUGH_NODES:
        _add_node(graph, name, passthrough)
    _add_node(graph, "retrieve", retrieve)

    graph.add_edge(START, "load_session")
    graph.add_edge("load_session", "guard")
    graph.add_conditional_edges(
        "guard",
        after_guard,
        {"intent": "intent", "refusal": "reply_refusal"},
    )
    graph.add_edge("intent", "merge_constraints")
    graph.add_conditional_edges(
        "merge_constraints",
        after_merge,
        {"resolve_people": "resolve_people", "retrieve": "retrieve"},
    )
    graph.add_edge("resolve_people", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        after_retrieve,
        {"broaden": "broaden_constraints", "rank": "rank"},
    )
    graph.add_edge("broaden_constraints", "retrieve")
    graph.add_edge("rank", "validate_availability")
    graph.add_conditional_edges(
        "validate_availability",
        route_reply,
        {
            "template": "reply_template",
            "generative": "reply_generative",
            "clarify": "reply_clarify",
            "refusal": "reply_refusal",
        },
    )
    for reply in (
        "reply_template",
        "reply_generative",
        "reply_clarify",
        "reply_refusal",
    ):
        graph.add_edge(reply, "sanitize_picks")
    graph.add_edge("sanitize_picks", "mint_chips")
    graph.add_edge("mint_chips", "persist")
    graph.add_edge("persist", END)

    compiled = graph.compile(checkpointer=False, name="assist_turn")
    _assert_workflow_invariants(compiled)
    return compiled


def export_mermaid(path: Path | None = None) -> str:
    """Write the compiled graph as Mermaid. Path defaults to docs/graph.mmd."""
    compiled = build_graph()
    mermaid = compiled.get_graph().draw_mermaid(with_styles=False)
    target = path if path is not None else default_mermaid_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not mermaid.endswith("\n"):
        mermaid = mermaid + "\n"
    target.write_text(mermaid, encoding="utf-8")
    return mermaid
