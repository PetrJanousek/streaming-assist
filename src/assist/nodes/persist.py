"""Persist the turn: session write-back + fire-and-forget AssistTurnEvent.

Turns go through `Session.append_turn` only. `Session.turns` has no max_length,
so assigning a long tuple and calling `save()` would store more than 6.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Literal, Protocol

from assist.domain.catalog import Pick
from assist.domain.constraints import ConstraintState
from assist.domain.context import ServerUserCtx
from assist.domain.enums import DegradedReason, Route
from assist.graph.state import TurnState
from assist.obs.logging import get_logger
from assist.stores.db import TurnEvent
from assist.stores.session import Session, TurnSummary

log = get_logger(__name__)


class TurnEventSink(Protocol):
    async def record(self, event: TurnEvent) -> None: ...


class SessionStore(Protocol):
    async def load(self, session_id: str, user_id: str, profile_id: str) -> Session: ...

    async def save(self, session: Session) -> None: ...


def _pick_ids(state: TurnState) -> tuple[str, ...]:
    raw = state.get("picks") or ()
    if not isinstance(raw, (list, tuple)):
        return ()
    ids: list[str] = []
    for item in raw:
        if isinstance(item, Pick):
            ids.append(item.catalog_id)
            continue
        catalog_id = getattr(item, "catalog_id", None)
        if catalog_id:
            ids.append(str(catalog_id))
    return tuple(ids)


def _message_type(state: TurnState) -> Literal["text", "chip"]:
    raw = state.get("message_type")
    if raw == "chip":
        return "chip"
    if raw == "text":
        return "text"
    return "chip" if state.get("chip_id") else "text"


def turn_summary_from_state(state: TurnState) -> TurnSummary:
    route = state.get("route")
    intent = state.get("intent_source")
    return TurnSummary(
        message_type=_message_type(state),
        text=str(state.get("text") or ""),
        reply=str(state.get("reply") or ""),
        pick_ids=_pick_ids(state),
        route=route if isinstance(route, Route) else None,
        intent_source=str(intent) if intent else None,
    )


def turn_event_from_state(state: TurnState) -> TurnEvent:
    route = state.get("route")
    if not isinstance(route, Route):
        route = Route.TEMPLATE
    reason = state.get("degraded_reason")
    if not isinstance(reason, DegradedReason):
        reason = DegradedReason.NONE
    raw_timings = state.get("timings") or {}
    latency: dict[str, int] = {}
    if isinstance(raw_timings, dict):
        for key, value in raw_timings.items():
            try:
                latency[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
    return TurnEvent(
        trace_id=str(state.get("trace_id") or "-"),
        session_id=str(state.get("session_id") or ""),
        route=route,
        intent_source=str(state.get("intent_source") or ""),
        degraded_reason=reason,
        stage_latency_ms=latency,
        tokens_in=int(state.get("tokens_in") or 0),
        tokens_out=int(state.get("tokens_out") or 0),
        cost_usd=float(state.get("cost_usd") or 0.0),
    )


def append_turn(session: Session, state: TurnState) -> Session:
    """Single write path for history. Never assign `turns=` and save."""
    constraints = state.get("constraints")
    if isinstance(constraints, ConstraintState):
        session = session.with_constraints(constraints)
    ctx = state.get("ctx")
    if isinstance(ctx, ServerUserCtx):
        session = session.with_ctx_echo(ctx)
    return session.append_turn(turn_summary_from_state(state))


async def persist(
    state: TurnState,
    *,
    session: Session | None = None,
    sessions: SessionStore | None = None,
    events: TurnEventSink | None = None,
) -> dict[str, object]:
    """LangGraph node. Degrades on session I/O; analytics failures never raise."""
    t0 = time.perf_counter()
    ctx = state.get("ctx")
    sess = session
    store_failed = False

    if sess is None and sessions is not None and isinstance(ctx, ServerUserCtx):
        session_id = str(state.get("session_id") or "")
        if session_id:
            try:
                sess = await sessions.load(session_id, ctx.user_id, ctx.profile_id)
            except Exception:
                log.warning("persist_session_load_failed", session_id=session_id)
                store_failed = True
                sess = None

    if sess is not None:
        sess = append_turn(sess, state)
        if sessions is not None:
            try:
                await sessions.save(sess)
            except Exception:
                log.warning("persist_session_save_failed", session_id=sess.session_id)
                store_failed = True
    elif sessions is not None:
        store_failed = True

    if events is not None:
        try:
            await events.record(turn_event_from_state(state))
        except Exception:
            log.exception(
                "turn_event_write_failed",
                session_id=str(state.get("session_id") or ""),
            )

    timings = dict(state.get("timings") or {})
    timings["persist"] = int((time.perf_counter() - t0) * 1000)
    out: dict[str, object] = {"timings": timings}
    if sess is not None:
        out["turn_count"] = sess.turn_count
    if store_failed:
        current = state.get("degraded_reason") or DegradedReason.NONE
        if current in (DegradedReason.NONE, None):
            out["degraded_reason"] = DegradedReason.SESSION_STORE_UNAVAILABLE
    return out


def make_persist_node(
    *,
    sessions: SessionStore | None = None,
    events: TurnEventSink | None = None,
) -> Callable[[TurnState], Awaitable[dict[str, object]]]:
    """Bind stores for the graph. T24 wires Redis + Postgres."""

    async def _node(state: TurnState) -> dict[str, object]:
        return await persist(state, sessions=sessions, events=events)

    return _node
