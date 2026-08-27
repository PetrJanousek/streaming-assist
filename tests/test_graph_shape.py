"""Architectural invariants: workflow, not an agent."""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import assist.graph.edges as edges_mod
from assist.domain.catalog import Candidate
from assist.domain.constraints import ConstraintState
from assist.domain.context import ServerUserCtx
from assist.domain.enums import (
    DegradedReason,
    DeviceClass,
    MaturityRating,
    MediaType,
    Package,
    Route,
)
from assist.graph.build import build_graph, export_mermaid
from assist.graph.edges import after_guard, after_retrieve, route_reply
from assist.graph.state import PersonSoft, empty_turn_state

EDGES_PATH = Path(edges_mod.__file__).resolve()

_FORBIDDEN_IMPORTS = frozenset(
    {
        "assist.llm",
        "langchain_anthropic",
        "langchain_openai",
        "anthropic",
        "openai",
        "httpx",
        "redis",
        "elasticsearch",
        "asyncpg",
        "sqlalchemy",
        "aiohttp",
        "requests",
        "socket",
        "urllib",
        "http.client",
    }
)
_CYCLE_NODES = frozenset({"retrieve", "broaden_constraints"})
_IGNORE_PREFIX = "__"


def _ctx() -> ServerUserCtx:
    return ServerUserCtx(
        user_id="u1",
        profile_id="p1",
        geo="US",
        package=Package.BASIC,
        maturity_max=MaturityRating.PG_13,
        kids_flag=False,
        device_class=DeviceClass.WEB,
    )


def _candidate(catalog_id: str = "ttl_1") -> Candidate:
    return Candidate(
        catalog_id=catalog_id,
        title=catalog_id,
        media_type=MediaType.FILM,
        score=1.0,
    )


def _imported_modules(path: Path) -> set[str]:
    """Return imported module names from AST. Not a substring scan of the source."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            modules.add(node.module)
            if node.module == "assist":
                for alias in node.names:
                    modules.add(f"assist.{alias.name}")
    return modules


def _is_forbidden(mod: str) -> bool:
    for banned in _FORBIDDEN_IMPORTS:
        if mod == banned or mod.startswith(banned + "."):
            return True
    return False


def _adjacency(compiled: Any) -> dict[str, list[str]]:
    adj: dict[str, list[str]] = defaultdict(list)
    for edge in compiled.get_graph().edges:
        adj[edge.source].append(edge.target)
    return dict(adj)


def _simple_cycles(adj: Mapping[str, list[str]]) -> list[tuple[str, ...]]:
    """Enumerate directed simple cycles. Independent of assist.graph.build."""
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


def _has_tools(obj: object, seen: set[int] | None = None) -> bool:
    if obj is None:
        return False
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return False
    seen.add(oid)

    name = type(obj).__name__
    if name in {"ToolNode", "AgentExecutor"}:
        return True

    tools = getattr(obj, "tools", None)
    if tools:
        return True
    kwargs = getattr(obj, "kwargs", None)
    if isinstance(kwargs, dict) and kwargs.get("tools"):
        return True

    for attr in ("bound", "runnable", "first", "last", "middle", "mapper", "func", "afunc"):
        child = getattr(obj, attr, None)
        if child is not None and child is not obj and _has_tools(child, seen):
            return True
    steps = getattr(obj, "steps", None)
    if isinstance(steps, (list, tuple)):
        for item in steps:
            if _has_tools(item, seen):
                return True
    return False


def test_edges_imports_nothing_from_llm() -> None:
    modules = _imported_modules(EDGES_PATH)
    llm_hits = [mod for mod in sorted(modules) if _is_forbidden(mod)]
    assert llm_hits == [], f"edges.py imported forbidden modules: {llm_hits}"
    assert not any(mod == "assist.llm" or mod.startswith("assist.llm.") for mod in modules)


def test_graph_compiles_without_checkpointer() -> None:
    compiled = build_graph()
    assert compiled.checkpointer is False
    assert compiled.get_graph() is not None


def test_only_cycle_is_retrieve_broaden() -> None:
    compiled = build_graph()
    stage_nodes = {name for name in compiled.nodes if not str(name).startswith(_IGNORE_PREFIX)}
    cycles = _simple_cycles(_adjacency(compiled))
    assert cycles, "expected the retrieve↔broaden cycle"

    participants: set[str] = set()
    for cycle in cycles:
        nodes = {name for name in cycle if name in stage_nodes}
        if not nodes:
            continue
        assert nodes == _CYCLE_NODES, f"unexpected cycle among {sorted(nodes)}: {cycle}"
        participants.update(nodes)
    assert participants == _CYCLE_NODES


async def test_stub_turn_runs_end_to_end() -> None:
    compiled = build_graph()
    result = await compiled.ainvoke(empty_turn_state(_ctx(), text="something cozy"))
    assert result["retrieve_attempts"] >= 1
    assert "ctx" in result
    # Empty-candidate stub path takes the bounded broaden cycle, then persist.
    assert result["retrieve_attempts"] == result["retrieve_max_attempts"]


async def test_retrieve_attempts_cap_is_enforced() -> None:
    compiled = build_graph()
    cap = 2
    visits = {"retrieve": 0, "broaden_constraints": 0}
    async for update in compiled.astream(
        empty_turn_state(_ctx(), retrieve_max_attempts=cap),
        stream_mode="updates",
    ):
        for name in update:
            if name in visits:
                visits[name] += 1
    assert visits["retrieve"] == cap
    assert visits["broaden_constraints"] == cap - 1

    cap_one = await compiled.ainvoke(empty_turn_state(_ctx(), retrieve_max_attempts=1))
    assert cap_one["retrieve_attempts"] == 1


async def test_seeded_candidates_skip_broaden() -> None:
    compiled = build_graph()
    visits = {"retrieve": 0, "broaden_constraints": 0}
    async for update in compiled.astream(
        empty_turn_state(_ctx(), candidates=(_candidate(),)),
        stream_mode="updates",
    ):
        for name in update:
            if name in visits:
                visits[name] += 1
    assert visits["retrieve"] == 1
    assert visits["broaden_constraints"] == 0


async def test_blocked_turn_skips_retrieve() -> None:
    compiled = build_graph()
    result = await compiled.ainvoke(empty_turn_state(_ctx(), safety_blocked=True))
    assert result["retrieve_attempts"] == 0


def test_no_node_is_given_tools() -> None:
    compiled = build_graph()
    for name, node in compiled.nodes.items():
        if str(name).startswith(_IGNORE_PREFIX):
            continue
        bound = getattr(node, "bound", node)
        assert not _has_tools(bound), f"node {name!r} has tools"


def test_after_retrieve_predicates() -> None:
    ctx = _ctx()
    empty = empty_turn_state(ctx, retrieve_attempts=1, retrieve_max_attempts=2, candidates=())
    assert after_retrieve(empty) == "broaden"
    capped = empty_turn_state(ctx, retrieve_attempts=2, retrieve_max_attempts=2, candidates=())
    assert after_retrieve(capped) == "rank"
    hits = empty_turn_state(
        ctx,
        retrieve_attempts=1,
        retrieve_max_attempts=2,
        candidates=(_candidate(),),
    )
    assert after_retrieve(hits) == "rank"


def test_after_retrieve_tolerates_both_media_type_sentinels() -> None:
    ctx = _ctx()
    for media_type in (None, MediaType.ANY):
        state = empty_turn_state(
            ctx,
            constraints=ConstraintState(media_type=media_type),
            retrieve_attempts=1,
            retrieve_max_attempts=2,
            candidates=(),
        )
        assert after_retrieve(state) == "broaden"


def test_after_guard_and_route_reply() -> None:
    ctx = _ctx()
    assert after_guard(empty_turn_state(ctx)) == "intent"
    assert after_guard(empty_turn_state(ctx, safety_blocked=True)) == "refusal"
    assert after_guard(empty_turn_state(ctx, degraded_reason=DegradedReason.SAFETY_BLOCK)) == (
        "refusal"
    )
    assert route_reply(empty_turn_state(ctx)) == "template"
    assert route_reply(empty_turn_state(ctx, route=Route.GENERATIVE)) == "generative"
    assert route_reply(empty_turn_state(ctx, person_ambiguous=True)) == "clarify"
    assert route_reply(empty_turn_state(ctx, route=Route.SAFETY)) == "refusal"


def test_person_soft_hint_routes_to_people() -> None:
    from assist.graph.edges import after_merge

    ctx = _ctx()
    assert after_merge(empty_turn_state(ctx)) == "retrieve"
    hinted = empty_turn_state(ctx, person_soft=PersonSoft(free_hint="spy guy from the 90s"))
    assert after_merge(hinted) == "resolve_people"


def test_export_mermaid_writes_file(tmp_path: Path) -> None:
    target = tmp_path / "graph.mmd"
    mermaid = export_mermaid(target)
    assert target.is_file()
    text = target.read_text(encoding="utf-8")
    assert text == mermaid
    assert "retrieve" in text
    assert "broaden_constraints" in text
    assert "graph " in text.lower()
