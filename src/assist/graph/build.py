"""Assemble the workflow graph. Real nodes, real edges, no checkpointer."""

from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from langchain_core.exceptions import ModelRateLimitError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from assist.config import Settings
from assist.config import settings as default_settings
from assist.domain.enums import DegradedReason, Route
from assist.graph.edges import after_guard, after_merge, after_retrieve, route_reply
from assist.graph.state import TurnState
from assist.obs.logging import get_logger
from assist.stores.session import ChipInvalid, ChipRecord, Session, SessionBindError

_CYCLE_NODES = frozenset({"retrieve", "broaden_constraints"})

# Stage names. T24 binds the real node for each; tests may still override one.
_STAGE_NODES = (
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

log = get_logger("assist.graph.build")


class _ChipLookup:
    """T15 ChipSource. LangGraph runs nodes in sibling tasks, so ContextVar
    set in `load_session` is not visible to `intent`. Chip ids are unique.
    """

    def __init__(self) -> None:
        self._records: dict[str, ChipRecord] = {}

    def remember(self, session: Session) -> None:
        self._records.update(session.issued_chips)

    def lookup_chip(self, chip_id: str) -> ChipRecord:
        record = self._records.get(chip_id)
        if record is None:
            raise ChipInvalid(chip_id)
        expires_at = record.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if expires_at <= datetime.now(UTC):
            raise ChipInvalid(chip_id)
        return record


@dataclass(frozen=True)
class GraphDeps:
    """I/O bound into nodes. Omit a field to leave that stage unwired.

    Production should pass `GraphDeps.live()`. `build_graph()` with no deps
    still compiles: retrieve/session I/O no-ops so T09 shape tests stay pure.
    `src/assist/main.py` (T13) currently calls `build_graph()` with no deps.
    """

    sessions: Any = None
    cache: Any = None
    es: Any = None
    embedder: Any = None
    catalog: Any = None
    events: Any = None
    people: Any = None
    model: Any = None
    settings: Settings | None = None

    @classmethod
    def live(cls, *, settings: Settings | None = None) -> GraphDeps:
        """Construct store clients from env. Caller owns shutdown."""
        cfg = settings if settings is not None else default_settings
        from redis.asyncio import Redis

        from assist.llm.gateway import get_chat_model
        from assist.nodes.people import EsPeopleSearcher
        from assist.stores.cache import CacheStore
        from assist.stores.catalog_client import PostgresCatalogClient
        from assist.stores.db import Database, TurnEventRepository
        from assist.stores.embed_client import EmbedClient
        from assist.stores.es import create_client
        from assist.stores.session import SessionRepository

        redis = Redis.from_url(cfg.redis_url, decode_responses=True)
        cache = CacheStore(redis)
        database = Database.from_settings(cfg)
        es = create_client(url=cfg.elasticsearch_url)
        return cls(
            sessions=SessionRepository(redis, ttl_s=cfg.session_ttl_s),
            cache=cache,
            es=es,
            embedder=EmbedClient(
                base_url=cfg.embedder_url,
                timeout_ms=cfg.embedder_timeout_ms,
                retries=cfg.embedder_retries,
            ),
            catalog=PostgresCatalogClient(database=database, cache=cache),
            events=TurnEventRepository(database.session_factory),
            people=EsPeopleSearcher(es),
            model=get_chat_model(settings=cfg),
            settings=cfg,
        )


async def passthrough(_state: TurnState) -> dict[str, object]:
    """Identity. Kept for T09 scratch graphs and override fallbacks."""
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


def _make_load_session(sessions: Any, chips: _ChipLookup) -> Any:
    async def load_session(state: TurnState) -> dict[str, object]:
        if sessions is None:
            return {}
        ctx = state.get("ctx")
        session_id = str(state.get("session_id") or "")
        if not session_id or ctx is None:
            return {}
        user_id = getattr(ctx, "user_id", "")
        profile_id = getattr(ctx, "profile_id", "")
        if not user_id or not profile_id:
            return {}
        try:
            session = await sessions.load(session_id, user_id, profile_id)
        except SessionBindError:
            raise
        except Exception:
            log.warning("load_session_failed", session_id=session_id)
            current = state.get("degraded_reason") or DegradedReason.NONE
            if current in (DegradedReason.NONE, None):
                return {"degraded_reason": DegradedReason.SESSION_STORE_UNAVAILABLE}
            return {}
        chips.remember(session)
        # Union, not just the last turn: a title shown two taps ago is still
        # "seen" for MORE_RESULTS exclusion. TurnSummary.pick_ids already
        # records what was shown each turn -- no new storage needed.
        seen: set[str] = set()
        for turn in session.turns:
            seen.update(turn.pick_ids)
        return {
            "constraints": session.constraints,
            "turn_count": session.turn_count,
            "seen_catalog_ids": tuple(sorted(seen)),
        }

    return load_session


def _make_guard() -> Any:
    async def guard_node(state: TurnState) -> dict[str, object]:
        # A pre-set block is graph input (T09 / safety edge). Re-inspecting
        # empty text would clear the flag and send a blocked turn into retrieve.
        if state.get("safety_blocked"):
            return {}
        if state.get("degraded_reason") is DegradedReason.SAFETY_BLOCK:
            return {}
        if state.get("route") is Route.SAFETY:
            return {}
        from assist.nodes.guard import guard

        return await guard(state)

    return guard_node


def _make_retrieve(es: Any, embedder: Any) -> Any:
    async def retrieve(state: TurnState) -> dict[str, object]:
        if es is None:
            # Unwired retrieve must not wipe seeded candidates (T09 cycle tests).
            return {}
        from assist.nodes.retrieval import retrieve as retrieve_titles

        return await retrieve_titles(state, es=es, embedder=embedder)

    return retrieve


def _make_availability(catalog: Any) -> Any:
    async def validate(state: TurnState) -> dict[str, object]:
        if catalog is None:
            # Unwired validator must not drop seeded candidates (T17/T19 graph tests).
            return {}
        from assist.nodes.availability import validate_availability

        return await validate_availability(state, client=catalog)

    return validate


def _wired_nodes(deps: GraphDeps) -> dict[str, Any]:
    # Node modules import the LLM gateway. Keep those imports behind this
    # call so `import assist.graph.edges` (via package init) stays llm-free.
    from assist.nodes.chips import make_chips_node
    from assist.nodes.intent import make_intent_node
    from assist.nodes.merge import merge_constraints
    from assist.nodes.people import make_people_node
    from assist.nodes.persist import make_persist_node
    from assist.nodes.rank import rank
    from assist.nodes.reply import make_reply_node
    from assist.nodes.retrieval import broaden_constraints
    from assist.nodes.sanitize import make_sanitize_node
    from assist.nodes.templates import make_template_node

    cfg = deps.settings
    chips = _ChipLookup()
    return {
        "load_session": _make_load_session(deps.sessions, chips),
        "guard": _make_guard(),
        "intent": make_intent_node(
            chips=chips if deps.sessions is not None else None,
            cache=deps.cache,
            model=deps.model,
            settings=cfg,
        ),
        "merge_constraints": merge_constraints,
        "resolve_people": make_people_node(searcher=deps.people, settings=cfg),
        "broaden_constraints": broaden_constraints,
        "rank": rank,
        "validate_availability": _make_availability(deps.catalog),
        "reply_template": make_template_node(slot="template", settings=cfg),
        "reply_generative": make_reply_node(model=deps.model, settings=cfg),
        "reply_clarify": make_template_node(slot="clarify", settings=cfg),
        "reply_refusal": make_template_node(slot="refusal", settings=cfg),
        "sanitize_picks": make_sanitize_node(settings=cfg),
        "mint_chips": make_chips_node(sessions=deps.sessions, settings=cfg),
        "persist": make_persist_node(sessions=deps.sessions, events=deps.events),
        "retrieve": _make_retrieve(deps.es, deps.embedder),
    }


def _degraded_state(state: TurnState, reason: DegradedReason) -> TurnState:
    degraded = dict(state)
    degraded["degraded_reason"] = reason
    degraded["reply"] = str(state.get("reply") or "")
    degraded["picks"] = state.get("picks") or ()
    degraded["chips"] = state.get("chips") or ()
    return cast(TurnState, degraded)


async def ainvoke_turn(
    compiled: CompiledStateGraph[TurnState, None, TurnState, TurnState],
    state: TurnState,
    *,
    timeout_s: float | None = None,
    settings: Settings | None = None,
) -> TurnState:
    """Invoke one turn. Timeouts and unexpected errors degrade; they never raise.

    `ChipInvalid` and `SessionBindError` still propagate — those are 400s, not 500s.
    """
    cfg = settings if settings is not None else default_settings
    timeout = timeout_s if timeout_s is not None else cfg.hard_timeout_ms / 1000.0
    try:
        result = await asyncio.wait_for(compiled.ainvoke(state), timeout=timeout)
        return cast(TurnState, result)
    except TimeoutError:
        log.warning("hard_timeout", session_id=state.get("session_id"))
        return _degraded_state(state, DegradedReason.HARD_TIMEOUT)
    except ModelRateLimitError:
        log.warning("provider_throttle", session_id=state.get("session_id"))
        return _degraded_state(state, DegradedReason.PROVIDER_THROTTLE)
    except (ChipInvalid, SessionBindError):
        raise
    except Exception:
        log.exception("graph_failed", session_id=state.get("session_id"))
        return _degraded_state(state, DegradedReason.RETRIEVAL_UNAVAILABLE)


def build_graph(
    *,
    node_overrides: Mapping[str, Any] | None = None,
    deps: GraphDeps | None = None,
) -> CompiledStateGraph[TurnState, None, TurnState, TurnState]:
    graph: StateGraph[TurnState] = StateGraph(TurnState)
    nodes = _wired_nodes(deps if deps is not None else GraphDeps())
    overrides = dict(node_overrides or {})
    for name, fn in overrides.items():
        if name not in nodes:
            msg = f"unknown node overrides: {name}"
            raise ValueError(msg)
        nodes[name] = fn

    for name in _STAGE_NODES:
        _add_node(graph, name, nodes.pop(name))
    _add_node(graph, "retrieve", nodes.pop("retrieve"))
    if nodes:
        unknown = ", ".join(sorted(nodes))
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
