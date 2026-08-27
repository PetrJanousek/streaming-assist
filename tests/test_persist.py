"""Persist node: append_turn only, AssistTurnEvent payload."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from assist.domain.catalog import Pick
from assist.domain.constraints import ConstraintState
from assist.domain.context import ServerUserCtx
from assist.domain.enums import (
    DegradedReason,
    DeviceClass,
    GenreId,
    MaturityRating,
    Package,
    Route,
)
from assist.graph.state import empty_turn_state
from assist.nodes.persist import persist, turn_event_from_state
from assist.stores.db import TurnEvent
from assist.stores.session import TURN_HISTORY_CAP, Session, TurnSummary

PERSIST_SRC = Path(__file__).resolve().parents[1] / "src" / "assist" / "nodes" / "persist.py"
_FORBIDDEN_IMPORTS = frozenset(
    {
        "assist.llm",
        "langchain_anthropic",
        "langchain_openai",
        "anthropic",
        "openai",
    }
)


class FakeSessions:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.saves = 0
        self.save_should_fail = False
        self.load_should_fail = False

    async def load(self, session_id: str, user_id: str, profile_id: str) -> Session:
        if self.load_should_fail:
            raise ConnectionError("redis down")
        return self.session

    async def save(self, session: Session) -> None:
        if self.save_should_fail:
            raise ConnectionError("redis down")
        self.saves += 1
        self.session = session


class FakeEvents:
    def __init__(self) -> None:
        self.events: list[TurnEvent] = []
        self.fail = False

    async def record(self, event: TurnEvent) -> None:
        if self.fail:
            raise RuntimeError("postgres down")
        self.events.append(event)


def _ctx() -> ServerUserCtx:
    return ServerUserCtx(
        user_id="u1",
        profile_id="p1",
        geo="US",
        package=Package.PREMIUM,
        maturity_max=MaturityRating.R,
        kids_flag=False,
        device_class=DeviceClass.WEB,
    )


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_persist_imports_no_llm() -> None:
    imported = _imported_modules(PERSIST_SRC)
    assert not any(
        mod == banned or mod.startswith(banned + ".")
        for mod in imported
        for banned in _FORBIDDEN_IMPORTS
    )


async def test_turn_event_carries_route_source_reason_latency_tokens_cost() -> None:
    session = Session.create(user_id="u1", profile_id="p1", session_id="s-evt")
    store = FakeSessions(session)
    events = FakeEvents()
    state = empty_turn_state(
        _ctx(),
        session_id="s-evt",
        trace_id="tr-evt",
        text="something cozy",
        reply="Here are a few titles that match.",
        picks=(Pick(catalog_id="ttl_a"), Pick(catalog_id="ttl_b")),
        route=Route.TEMPLATE,
        intent_source="rules",
        degraded_reason=DegradedReason.NONE,
        timings={"retrieve": 40, "rank": 5, "sanitize": 2},
        tokens_in=120,
        tokens_out=35,
        cost_usd=0.000295,
        constraints=ConstraintState(genres_include=(GenreId.COMEDY,)),
    )
    out = await persist(state, sessions=store, events=events)

    assert out["turn_count"] == 1
    assert store.saves == 1
    assert len(events.events) == 1
    event = events.events[0]
    assert event.trace_id == "tr-evt"
    assert event.session_id == "s-evt"
    assert event.route is Route.TEMPLATE
    assert event.intent_source == "rules"
    assert event.degraded_reason is DegradedReason.NONE
    assert event.stage_latency_ms["retrieve"] == 40
    assert event.stage_latency_ms["rank"] == 5
    assert event.stage_latency_ms["sanitize"] == 2
    assert event.tokens_in == 120
    assert event.tokens_out == 35
    assert event.cost_usd == pytest.approx(0.000295)

    saved = store.session
    assert saved.turn_count == 1
    assert len(saved.turns) == 1
    assert saved.turns[0].pick_ids == ("ttl_a", "ttl_b")
    assert saved.turns[0].route is Route.TEMPLATE
    assert saved.turns[0].intent_source == "rules"
    assert saved.constraints.genres_include == (GenreId.COMEDY,)
    assert saved.server_ctx_echo is not None
    assert saved.server_ctx_echo.device_class is DeviceClass.WEB


async def test_persist_uses_append_turn_and_caps_history_at_six() -> None:
    session = Session.create(user_id="u1", profile_id="p1", session_id="s-hist")
    for i in range(20):
        session = session.model_copy(
            update={
                "turns": (
                    *session.turns,
                    TurnSummary(message_type="text", text=f"old-{i}"),
                ),
                "turn_count": i + 1,
            }
        )
    assert len(session.turns) == 20

    store = FakeSessions(session)
    events = FakeEvents()
    state = empty_turn_state(
        _ctx(),
        session_id="s-hist",
        text="new turn",
        reply="ok",
        route=Route.TEMPLATE,
        intent_source="chip",
    )
    await persist(state, sessions=store, events=events)

    saved = store.session
    assert len(saved.turns) == TURN_HISTORY_CAP
    assert saved.turns[-1].text == "new turn"
    assert saved.turns[-1].reply == "ok"
    assert [t.text for t in saved.turns] == [
        "old-15",
        "old-16",
        "old-17",
        "old-18",
        "old-19",
        "new turn",
    ]


async def test_persist_via_append_turn_from_empty_caps_at_six() -> None:
    session = Session.create(user_id="u1", profile_id="p1", session_id="s-six")
    store = FakeSessions(session)
    events = FakeEvents()
    for i in range(8):
        state = empty_turn_state(
            _ctx(),
            session_id="s-six",
            text=f"t{i}",
            reply=f"r{i}",
            route=Route.TEMPLATE,
            intent_source="rules",
        )
        await persist(state, sessions=store, events=events)
    assert len(store.session.turns) == TURN_HISTORY_CAP
    assert store.session.turn_count == 8
    assert [t.text for t in store.session.turns] == [f"t{i}" for i in range(2, 8)]


async def test_session_save_failure_degrades_and_still_records_event() -> None:
    store = FakeSessions(Session.create(user_id="u1", profile_id="p1", session_id="s-fail"))
    store.save_should_fail = True
    events = FakeEvents()
    state = empty_turn_state(
        _ctx(),
        session_id="s-fail",
        text="hi",
        route=Route.GENERATIVE,
        intent_source="llm",
        tokens_in=10,
        tokens_out=4,
        cost_usd=0.001,
        timings={"intent": 80},
    )
    out = await persist(state, sessions=store, events=events)
    assert out["degraded_reason"] is DegradedReason.SESSION_STORE_UNAVAILABLE
    assert len(events.events) == 1
    assert events.events[0].route is Route.GENERATIVE
    assert events.events[0].tokens_in == 10


async def test_event_write_failure_does_not_raise() -> None:
    store = FakeSessions(Session.create(user_id="u1", profile_id="p1", session_id="s-evfail"))
    events = FakeEvents()
    events.fail = True
    state = empty_turn_state(
        _ctx(),
        session_id="s-evfail",
        text="hi",
        route=Route.TEMPLATE,
        intent_source="rules",
    )
    out = await persist(state, sessions=store, events=events)
    assert out["turn_count"] == 1
    assert store.saves == 1
    assert "degraded_reason" not in out


def test_turn_event_from_state_fills_required_analytics_fields() -> None:
    state = empty_turn_state(
        _ctx(),
        session_id="s1",
        trace_id="tr1",
        route=Route.CLARIFY,
        intent_source="chip",
        degraded_reason=DegradedReason.PERSON_AMBIGUOUS,
        timings={"people": 11},
        tokens_in=1,
        tokens_out=2,
        cost_usd=0.0,
    )
    event = turn_event_from_state(state)
    assert event.route is Route.CLARIFY
    assert event.intent_source == "chip"
    assert event.degraded_reason is DegradedReason.PERSON_AMBIGUOUS
    assert event.stage_latency_ms == {"people": 11}
    assert event.tokens_in == 1
    assert event.tokens_out == 2
    assert event.cost_usd == 0.0
