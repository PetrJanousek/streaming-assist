"""Assemble the workflow graph. Stubs pass state through; edges are real."""

from __future__ import annotations

import inspect
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


async def retrieve(_state: TurnState) -> dict[str, object]:
    """Stub retrieve. The graph wrapper owns `retrieve_attempts`; T17 fills hits."""
    return {}


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
    """Directed simple cycles, including self-loops. Each cycle from its min node."""
    found: list[tuple[str, ...]] = []
    for start in sorted(adj):
        stack: list[tuple[str, list[str], set[str]]] = [(start, [start], {start})]
        while stack:
            current, path, seen = stack.pop()
            for nxt in adj.get(current, ()):
                if nxt == start:
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


def _as_update(raw: object) -> dict[str, object]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    msg = f"graph node must return a mapping, got {type(raw)!r}"
    raise TypeError(msg)


def _wrap_node(name: str, fn: Any) -> Any:
    """Own the retrieve cap in the graph. A node cannot skip, reset, or raise it.

    `retrieve_attempts` increments on every retrieve visit, ignoring the node's
    update. `retrieve_max_attempts` is stripped from every node's update so only
    the turn's initial state (and settings) can set the cap. T17/T24 must keep
    registering nodes through `_add_node` — a raw `graph.add_node` drops this.
    """

    async def wrapped(state: TurnState) -> dict[str, object]:
        raw = fn(state)
        if inspect.isawaitable(raw):
            raw = await raw
        result = _as_update(raw)
        result.pop("retrieve_max_attempts", None)
        if name == "retrieve":
            attempts = int(state.get("retrieve_attempts") or 0)
            result["retrieve_attempts"] = attempts + 1
        else:
            result.pop("retrieve_attempts", None)
        return result

    wrapped_fn: Any = wrapped
    wrapped_fn.__name__ = getattr(fn, "__name__", name)
    wrapped_fn.__qualname__ = getattr(fn, "__qualname__", wrapped.__name__)
    wrapped_fn.__doc__ = getattr(fn, "__doc__", None)
    # inspect.unwrap; call-time tool checks need the impl, not this wrapper.
    wrapped_fn.__wrapped__ = fn
    return wrapped_fn


def _add_node(graph: StateGraph[TurnState], name: str, fn: Any) -> None:
    """LangGraph `_Node` overloads are sync; mypy rejects `async def` against TurnState.

    `fn` is Any for that seam only. Runtime still registers the async callable.
    """
    graph.add_node(name, _wrap_node(name, fn))


def build_graph(
    *,
    node_overrides: Mapping[str, Any] | None = None,
) -> CompiledStateGraph[TurnState, None, TurnState, TurnState]:
    graph: StateGraph[TurnState] = StateGraph(TurnState)
    overrides = dict(node_overrides or {})

    for name in _PASSTHROUGH_NODES:
        _add_node(graph, name, overrides.pop(name, passthrough))
    _add_node(graph, "retrieve", overrides.pop("retrieve", retrieve))
    if overrides:
        unknown = ", ".join(sorted(overrides))
        msg = f"unknown node overrides: {unknown}"
        raise ValueError(msg)

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
