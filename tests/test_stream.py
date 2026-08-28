"""SSE stage stream: progressive frames, no unvalidated catalog_id, chip turns."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

from fastapi.testclient import TestClient

from assist.api.deps import AppResources, default_profile_catalog
from assist.api.routes_stream import _catalog_ids_in
from assist.domain.catalog import Candidate, Pick
from assist.domain.constraints import ConstraintDelta, ConstraintState
from assist.domain.enums import GenreId, MediaType, MoodId, SpeechAct
from assist.graph.state import ReplyChip, TurnState
from assist.main import create_app
from assist.stores.ratelimit import RateLimitDecision, RateLimited
from assist.stores.session import Session, SessionBindError

LEAKED_ID = "s-leaked"
PLAYABLE_ID = "s-ok"


class PingRedis:
    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


class FakeCache:
    async def get_idempotent(self, raw_key: str) -> str | None:
        return None

    async def set_idempotent(self, raw_key: str, payload: str) -> None:
        return None


class FakeRateLimiter:
    def __init__(self, *, allowed: bool = True) -> None:
        self.allowed = allowed

    async def acquire(self, scope: str, subject: str, *, cost: int = 1) -> RateLimitDecision:
        if not self.allowed:
            raise RateLimited(scope=scope, subject=subject, retry_after_ms=250)
        return RateLimitDecision(allowed=True, remaining=19.0, retry_after_ms=0, limit=20)


class FakeSessionStore:
    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    async def load(self, session_id: str, user_id: str, profile_id: str) -> Session:
        raw = self._data.get(session_id)
        if raw is None:
            return Session.create(session_id=session_id, user_id=user_id, profile_id=profile_id)
        session = Session.model_validate_json(raw)
        if session.user_id != user_id or session.profile_id != profile_id:
            raise SessionBindError(
                session_id=session_id,
                bound_user_id=session.user_id,
                bound_profile_id=session.profile_id,
                user_id=user_id,
                profile_id=profile_id,
            )
        return session

    async def save(self, session: Session) -> None:
        self._data[session.session_id] = session.model_dump_json()


def _candidate(catalog_id: str, title: str) -> Candidate:
    return Candidate(
        catalog_id=catalog_id,
        title=title,
        media_type=MediaType.FILM,
        release_year=2007,
        genres=(GenreId.COMEDY,),
        score=0.9,
    )


class StageStreamGraph:
    """Yields node updates with sleeps so the client sees frames over time."""

    def __init__(self, *, delay_s: float = 0.06) -> None:
        self.delay_s = delay_s
        self.calls: list[TurnState] = []

    async def ainvoke(self, state: TurnState, **_kwargs: object) -> TurnState:
        out = dict(state)
        async for update in self.astream(state):
            for payload in update.values():
                if isinstance(payload, Mapping):
                    out.update(payload)
        return out  # type: ignore[return-value]

    async def astream(
        self, state: TurnState, **_kwargs: object
    ) -> AsyncIterator[dict[str, object]]:
        self.calls.append(state)
        constraints = ConstraintState(genres_include=(GenreId.COMEDY,))
        leaked = _candidate(LEAKED_ID, "Should Never Ship")
        playable = _candidate(PLAYABLE_ID, "Superbad")
        await asyncio.sleep(self.delay_s)
        yield {"merge_constraints": {"constraints": constraints}}
        await asyncio.sleep(self.delay_s)
        yield {"retrieve": {"candidates": (leaked, playable)}}
        await asyncio.sleep(self.delay_s)
        yield {"rank": {"candidates": (leaked, playable)}}
        await asyncio.sleep(self.delay_s)
        yield {
            "validate_availability": {
                "candidates": (playable,),
                "entitled_ids": (PLAYABLE_ID,),
            }
        }
        await asyncio.sleep(self.delay_s)
        yield {
            "sanitize_picks": {
                "picks": (Pick(catalog_id=PLAYABLE_ID, reason_short="funny"),),
                "reply": "Try Superbad.",
            }
        }
        yield {
            "mint_chips": {
                "chips": (ReplyChip(id="c_refine", label="Funnier"),),
            }
        }
        yield {"persist": {}}


class StickyStreamGraph:
    """Keeps whatever constraints arrived on state and adds one field per turn."""

    def __init__(self, sessions: FakeSessionStore) -> None:
        self.sessions = sessions
        self.calls: list[TurnState] = []

    async def ainvoke(self, state: TurnState, **_kwargs: object) -> TurnState:
        out = dict(state)
        async for update in self.astream(state):
            for payload in update.values():
                if isinstance(payload, Mapping):
                    out.update(payload)
        return out  # type: ignore[return-value]

    async def astream(
        self, state: TurnState, **_kwargs: object
    ) -> AsyncIterator[dict[str, object]]:
        self.calls.append(state)
        incoming = state.get("constraints")
        constraints = incoming if isinstance(incoming, ConstraintState) else ConstraintState.empty()
        if not constraints.genres_include:
            constraints = constraints.model_copy(update={"genres_include": (GenreId.COMEDY,)})
        elif not constraints.moods:
            constraints = constraints.model_copy(update={"moods": (MoodId.FUNNY,)})
        elif constraints.year_min is None:
            constraints = constraints.model_copy(update={"year_min": 1990})

        playable = _candidate(PLAYABLE_ID, "Superbad")
        yield {"merge_constraints": {"constraints": constraints}}
        yield {"rank": {"candidates": (playable,)}}
        yield {
            "validate_availability": {
                "candidates": (playable,),
                "entitled_ids": (PLAYABLE_ID,),
            }
        }
        yield {
            "sanitize_picks": {
                "picks": (Pick(catalog_id=PLAYABLE_ID, reason_short="funny"),),
                "reply": "Still in comedy.",
            }
        }

        ctx = state.get("ctx")
        session_id = str(state.get("session_id") or "")
        user_id = getattr(ctx, "user_id", "")
        profile_id = getattr(ctx, "profile_id", "")
        session = await self.sessions.load(session_id, user_id, profile_id)
        session = session.with_constraints(constraints)
        session, chip = session.mint_chip(
            label="Funnier" if not constraints.moods else "Nineties",
            delta=ConstraintDelta(),
            speech_act=SpeechAct.REFINE_MOOD,
        )
        await self.sessions.save(session)
        yield {
            "mint_chips": {
                "chips": (ReplyChip(id=chip.chip_id, label=chip.label),),
                "constraints": constraints,
                "reply": "Still in comedy.",
            }
        }
        yield {"persist": {}}


class BoomStreamGraph:
    async def ainvoke(self, state: TurnState, **_kwargs: object) -> TurnState:
        raise RuntimeError("graph exploded")

    async def astream(
        self, state: TurnState, **_kwargs: object
    ) -> AsyncIterator[dict[str, object]]:
        raise RuntimeError("graph exploded")
        yield {}  # pragma: no cover — makes this an async generator


def make_resources(**overrides: Any) -> AppResources:
    sessions = overrides.get("sessions") or FakeSessionStore()
    payload: dict[str, Any] = {
        "redis": PingRedis(),
        "cache": FakeCache(),
        "sessions": sessions,
        "rate_limiter": FakeRateLimiter(),
        "graph": StageStreamGraph(),
        "profiles": default_profile_catalog(),
    }
    payload.update(overrides)
    return AppResources(**payload)


def make_client(resources: AppResources | None = None) -> Iterator[TestClient]:
    app = create_app(resources=resources or make_resources())
    with TestClient(app) as client:
        yield client


def _auth(token: str = "dev-adult") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _as_int(value: object) -> int:
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def _as_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _as_list(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


def _chips_of_frame(events: list[tuple[str, dict[str, object]]]) -> list[dict[str, object]]:
    payload = next(p for _e, p in events if p.get("stage") == "reply")
    chips: list[dict[str, object]] = []
    for item in _as_list(payload.get("chips")):
        chips.append(_as_dict(item))
    return chips


def _constraints_of_frame(events: list[tuple[str, dict[str, object]]]) -> dict[str, object]:
    payload = next(p for _e, p in events if p.get("stage") == "constraints")
    return _as_dict(payload.get("constraints"))


def _text_body(text: str = "a cozy comedy", session_id: str | None = None) -> dict[str, object]:
    body: dict[str, object] = {"message": {"type": "text", "text": text}}
    if session_id is not None:
        body["session_id"] = session_id
    return body


def _read_sse(
    client: TestClient, body: dict[str, object]
) -> tuple[list[tuple[str, dict[str, object]]], str, list[float]]:
    times: list[float] = []
    raw_parts: list[str] = []
    events: list[tuple[str, dict[str, object]]] = []
    t0 = time.perf_counter()
    with client.stream(
        "POST",
        "/v1/assist/turn/stream",
        json=body,
        headers={**_auth(), "Accept": "text/event-stream"},
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        buffer = ""
        for chunk in response.iter_text():
            now = time.perf_counter() - t0
            buffer += chunk
            raw_parts.append(chunk)
            while "\n\n" in buffer:
                raw, buffer = buffer.split("\n\n", 1)
                event_name = "message"
                data_lines: list[str] = []
                for line in raw.split("\n"):
                    if line.startswith("event:"):
                        event_name = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line.split(":", 1)[1].strip())
                if data_lines:
                    events.append((event_name, json.loads("\n".join(data_lines))))
                    times.append(now)
        if buffer.strip():
            event_name = "message"
            data_lines = []
            for line in buffer.split("\n"):
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data_lines.append(line.split(":", 1)[1].strip())
            if data_lines:
                events.append((event_name, json.loads("\n".join(data_lines))))
                times.append(time.perf_counter() - t0)
    return events, "".join(raw_parts), times


def test_demo_page_is_served_without_build() -> None:
    with TestClient(create_app(resources=make_resources())) as client:
        index = client.get("/")
        js = client.get("/app.js")
        css = client.get("/style.css")
    assert index.status_code == 200
    assert "text/html" in index.headers["content-type"]
    assert "/app.js" in index.text
    assert 'id="composer"' in index.text
    assert js.status_code == 200
    assert "turn/stream" in js.text
    assert "chip_id" in js.text
    assert css.status_code == 200
    assert "--accent" in css.text


def test_stream_requires_bearer() -> None:
    with TestClient(create_app(resources=make_resources())) as client:
        response = client.post("/v1/assist/turn/stream", json=_text_body())
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "unauthorized"


def test_stream_frames_arrive_progressively() -> None:
    graph = StageStreamGraph(delay_s=0.07)
    with TestClient(create_app(resources=make_resources(graph=graph))) as client:
        events, _raw, times = _read_sse(client, _text_body())

    stages = [payload.get("stage") for _event, payload in events]
    assert stages[0] == "start"
    assert "constraints" in stages
    assert "candidates" in stages
    assert "validated" in stages
    assert "picks" in stages
    assert "reply" in stages
    assert stages[-1] == "done"
    assert len(events) >= 6
    # Emission time is stamped on each frame as t_ms. TestClient buffers the
    # ASGI body, so client-arrival times collapse; server emission must not.
    emitted = [_as_int(payload["t_ms"]) for _event, payload in events]
    assert emitted == sorted(emitted)
    assert emitted[0] < 80
    assert emitted[-1] - emitted[0] >= 200
    by_stage = {str(payload.get("stage")): _as_int(payload["t_ms"]) for _e, payload in events}
    assert by_stage["constraints"] < by_stage["picks"]
    assert by_stage["candidates"] < by_stage["validated"]
    assert times[-1] >= 0.20


def test_stream_never_emits_unvalidated_catalog_id() -> None:
    graph = StageStreamGraph(delay_s=0.0)
    with TestClient(create_app(resources=make_resources(graph=graph))) as client:
        events, raw, _times = _read_sse(client, _text_body())

    assert LEAKED_ID not in raw
    pre_sanitize_stages = {"start", "constraints", "candidates", "validated", "reply"}
    for event_name, payload in events:
        stage = str(payload.get("stage") or event_name)
        ids = _catalog_ids_in(payload)
        if stage in pre_sanitize_stages:
            assert ids == [], f"unvalidated id on {stage}: {ids}"
        if stage == "candidates":
            assert "count" in payload
            assert "catalog_id" not in json.dumps(payload)
        if stage == "validated":
            for card in _as_list(payload.get("cards")):
                card_dict = _as_dict(card)
                assert "catalog_id" not in card_dict
                assert "title" in card_dict
        if stage == "picks":
            assert ids == [PLAYABLE_ID]
        if stage == "done":
            pick_ids = {
                str(_as_dict(item)["catalog_id"]) for item in _as_list(payload.get("picks"))
            }
            assert pick_ids == {PLAYABLE_ID}

    chips = _chips_of_frame(events)
    assert chips == [{"id": "c_refine", "label": "Funnier"}]
    for chip in chips:
        assert set(chip) <= {"id", "label"}


def test_three_turn_chip_conversation_keeps_constraints() -> None:
    sessions = FakeSessionStore()
    graph = StickyStreamGraph(sessions)
    resources = make_resources(sessions=sessions, graph=graph)
    with TestClient(create_app(resources=resources)) as client:
        turn1, _raw1, _t1 = _read_sse(client, _text_body("comedy please"))
        done1 = turn1[-1][1]
        session_id = str(done1["session_id"])
        chip1 = str(_chips_of_frame(turn1)[0]["id"])

        turn2, _raw2, _t2 = _read_sse(
            client,
            {
                "session_id": session_id,
                "message": {"type": "chip", "chip_id": chip1},
            },
        )
        chip2 = str(_chips_of_frame(turn2)[0]["id"])

        turn3, _raw3, _t3 = _read_sse(
            client,
            {
                "session_id": session_id,
                "message": {"type": "chip", "chip_id": chip2},
            },
        )

    constraints1 = _constraints_of_frame(turn1)
    constraints2 = _constraints_of_frame(turn2)
    constraints3 = _constraints_of_frame(turn3)

    assert "comedy" in _as_list(constraints1["genres_include"])
    assert "comedy" in _as_list(constraints2["genres_include"])
    assert "funny" in _as_list(constraints2["moods"])
    assert "comedy" in _as_list(constraints3["genres_include"])
    assert "funny" in _as_list(constraints3["moods"])
    assert constraints3["year_min"] == 1990

    assert graph.calls[1]["message_type"] == "chip"
    assert graph.calls[1]["chip_id"] == chip1
    incoming2 = graph.calls[1]["constraints"]
    assert isinstance(incoming2, ConstraintState)
    assert GenreId.COMEDY in incoming2.genres_include
    incoming3 = graph.calls[2]["constraints"]
    assert isinstance(incoming3, ConstraintState)
    assert GenreId.COMEDY in incoming3.genres_include
    assert MoodId.FUNNY in incoming3.moods


def test_stream_graph_error_is_degraded_done_not_500() -> None:
    with TestClient(create_app(resources=make_resources(graph=BoomStreamGraph()))) as client:
        events, _raw, _times = _read_sse(client, _text_body())
    done = events[-1][1]
    assert events[-1][0] == "done"
    meta = _as_dict(done["meta"])
    assert meta["degraded"] is True
    assert meta["degraded_reason"] == "retrieval_unavailable"
    assert done["picks"] == []


def test_unknown_chip_on_stream_is_400() -> None:
    with TestClient(create_app(resources=make_resources())) as client:
        first, _raw, _t = _read_sse(client, _text_body())
        session_id = first[-1][1]["session_id"]
        response = client.post(
            "/v1/assist/turn/stream",
            json={"session_id": session_id, "message": {"type": "chip", "chip_id": "c_missing"}},
            headers=_auth(),
        )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "chip_invalid"
