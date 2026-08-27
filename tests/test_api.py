"""API surface: turn contract, errors, idempotency, ops routes."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from assist.api.deps import AppResources, ProfileCatalog, SeededProfile, default_profile_catalog
from assist.domain.constraints import ConstraintDelta
from assist.domain.context import ServerUserCtx
from assist.domain.enums import DeviceClass, MaturityRating, Package, SpeechAct
from assist.graph.build import build_graph
from assist.graph.state import TurnState
from assist.main import create_app
from assist.stores.ratelimit import RateLimitDecision, RateLimited
from assist.stores.session import Session, SessionBindError


class PingRedis:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok

    async def ping(self) -> bool:
        if not self.ok:
            raise ConnectionError("redis down")
        return True

    async def aclose(self) -> None:
        return None


class FakeCache:
    def __init__(self) -> None:
        self.idem: dict[str, str] = {}

    async def get_idempotent(self, raw_key: str) -> str | None:
        return self.idem.get(raw_key)

    async def set_idempotent(self, raw_key: str, payload: str) -> None:
        self.idem[raw_key] = payload


class FakeRateLimiter:
    def __init__(self, *, allowed: bool = True, retry_after_ms: int = 250) -> None:
        self.allowed = allowed
        self.retry_after_ms = retry_after_ms
        self.calls: list[tuple[str, str]] = []

    async def acquire(self, scope: str, subject: str, *, cost: int = 1) -> RateLimitDecision:
        self.calls.append((scope, subject))
        if not self.allowed:
            raise RateLimited(scope=scope, subject=subject, retry_after_ms=self.retry_after_ms)
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
        existing = self._data.get(session.session_id)
        if existing is not None:
            bound = Session.model_validate_json(existing)
            if bound.user_id != session.user_id or bound.profile_id != session.profile_id:
                raise SessionBindError(
                    session_id=session.session_id,
                    bound_user_id=bound.user_id,
                    bound_profile_id=bound.profile_id,
                    user_id=session.user_id,
                    profile_id=session.profile_id,
                )
        self._data[session.session_id] = session.model_dump_json()


class RecordingGraph:
    def __init__(self, reply: str = "Here are some titles.") -> None:
        self.calls: list[TurnState] = []
        self.reply = reply

    async def ainvoke(self, state: TurnState, **_kwargs: object) -> TurnState:
        self.calls.append(state)
        out = dict(state)
        out["reply"] = self.reply
        out["route"] = "template"
        return out  # type: ignore[return-value]


class SleepGraph:
    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.calls = 0

    async def ainvoke(self, state: TurnState, **_kwargs: object) -> TurnState:
        self.calls += 1
        await asyncio.sleep(self.delay_s)
        return state


class BoomGraph:
    async def ainvoke(self, state: TurnState, **_kwargs: object) -> TurnState:
        raise RuntimeError("graph exploded")


def make_resources(**overrides: Any) -> AppResources:
    payload: dict[str, Any] = {
        "redis": PingRedis(),
        "cache": FakeCache(),
        "sessions": FakeSessionStore(),
        "rate_limiter": FakeRateLimiter(),
        "graph": RecordingGraph(),
        "profiles": default_profile_catalog(),
    }
    payload.update(overrides)
    return AppResources(**payload)


def make_client(resources: AppResources | None = None) -> Iterator[TestClient]:
    app = create_app(resources=resources or make_resources())
    with TestClient(app) as client:
        yield client


@pytest.fixture
def graph() -> RecordingGraph:
    return RecordingGraph()


@pytest.fixture
def resources(graph: RecordingGraph) -> AppResources:
    return make_resources(graph=graph)


@pytest.fixture
def client(resources: AppResources) -> Iterator[TestClient]:
    yield from make_client(resources)


def _auth(token: str = "dev-adult") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _turn(
    text: str = "something cozy",
    *,
    session_id: str | None = None,
    client_hints: dict[str, object] | None = None,
) -> dict[str, object]:
    body: dict[str, object] = {
        "session_id": session_id,
        "message": {"type": "text", "text": text},
    }
    if client_hints is not None:
        body["client_hints"] = client_hints
    return body


def test_healthz_ok_when_redis_down() -> None:
    with TestClient(create_app(resources=make_resources(redis=PingRedis(ok=False)))) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readyz_fails_when_redis_down() -> None:
    with TestClient(create_app(resources=make_resources(redis=PingRedis(ok=False)))) as client:
        response = client.get("/readyz")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["redis"] == "down"


def test_readyz_ok_when_redis_pings(client: TestClient) -> None:
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "redis": "ok"}


def test_dev_profiles_lists_fixture_tokens(client: TestClient) -> None:
    response = client.get("/dev/profiles")
    assert response.status_code == 200
    tokens = {row["token"] for row in response.json()["profiles"]}
    assert tokens == {"dev-adult", "dev-kids", "dev-basic"}
    for row in response.json()["profiles"]:
        assert "device_bound" not in row
        assert row["device_class"] in {"tv", "mobile", "web", "tablet"}


def test_turn_requires_bearer(client: TestClient) -> None:
    response = client.post("/v1/assist/turn", json=_turn())
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["type"] == "unauthorized"
    assert body["picks"] == []
    assert body["chips"] == []
    assert "meta" in body


def test_turn_unknown_token_is_401(client: TestClient) -> None:
    response = client.post("/v1/assist/turn", json=_turn(), headers=_auth("nope"))
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "unauthorized"


def test_turn_returns_contract_shape(client: TestClient, graph: RecordingGraph) -> None:
    response = client.post("/v1/assist/turn", json=_turn(), headers=_auth())
    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Here are some titles."
    assert body["picks"] == []
    assert body["chips"] == []
    assert body["session_id"]
    assert body["meta"]["trace_id"]
    assert body["meta"]["degraded"] is False
    assert len(graph.calls) == 1
    assert graph.calls[0]["ctx"].user_id == "user_adult"


def test_idempotency_key_replays_without_rerunning_graph(
    client: TestClient, graph: RecordingGraph
) -> None:
    headers = {**_auth(), "Idempotency-Key": str(uuid4())}
    first = client.post("/v1/assist/turn", json=_turn(), headers=headers)
    second = client.post("/v1/assist/turn", json=_turn("a different message"), headers=headers)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.content == second.content
    assert len(graph.calls) == 1


def test_rate_limited_body_has_usable_shape() -> None:
    resources = make_resources(rate_limiter=FakeRateLimiter(allowed=False, retry_after_ms=400))
    with TestClient(create_app(resources=resources)) as client:
        response = client.post("/v1/assist/turn", json=_turn(), headers=_auth())
    assert response.status_code == 429
    body = response.json()
    assert body["error"]["type"] == "rate_limited"
    assert body["error"]["retry_after_ms"] == 400
    assert body["picks"] == []
    assert body["chips"] == []
    assert body["reply"] == ""
    assert "meta" in body
    assert body["meta"]["degraded"] is True
    assert "Retry-After" in response.headers


def test_unknown_chip_is_400_chip_invalid() -> None:
    sessions = FakeSessionStore()
    resources = make_resources(sessions=sessions)
    with TestClient(create_app(resources=resources)) as client:
        created = client.post("/v1/assist/turn", json=_turn(), headers=_auth())
        session_id = created.json()["session_id"]
        response = client.post(
            "/v1/assist/turn",
            json={
                "session_id": session_id,
                "message": {"type": "chip", "chip_id": "c_missing"},
            },
            headers=_auth(),
        )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "chip_invalid"
    assert body["picks"] == []


def test_valid_chip_reaches_graph() -> None:
    sessions = FakeSessionStore()
    graph = RecordingGraph()
    resources = make_resources(sessions=sessions, graph=graph)
    session = Session.create(session_id="s-chip", user_id="user_adult", profile_id="profile_adult")
    session, chip = session.mint_chip(
        label="Funnier",
        delta=ConstraintDelta(),
        speech_act=SpeechAct.REFINE_MOOD,
    )
    sessions._data[session.session_id] = session.model_dump_json()

    with TestClient(create_app(resources=resources)) as client:
        response = client.post(
            "/v1/assist/turn",
            json={
                "session_id": session.session_id,
                "message": {"type": "chip", "chip_id": chip.chip_id},
            },
            headers=_auth(),
        )
    assert response.status_code == 200
    assert len(graph.calls) == 1
    assert graph.calls[0]["chip_id"] == chip.chip_id
    assert graph.calls[0]["message_type"] == "chip"


def test_empty_text_is_400_validation(client: TestClient) -> None:
    response = client.post(
        "/v1/assist/turn",
        json={"message": {"type": "text", "text": "  "}},
        headers=_auth(),
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "validation"


def test_graph_exception_is_degraded_200_not_500() -> None:
    resources = make_resources(graph=BoomGraph())
    with TestClient(create_app(resources=resources)) as client:
        response = client.post("/v1/assist/turn", json=_turn(), headers=_auth())
    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["degraded"] is True
    assert body["meta"]["degraded_reason"] == "retrieval_unavailable"
    assert body["picks"] == []


def test_hard_timeout_returns_degraded_body() -> None:
    resources = make_resources(graph=SleepGraph(0.2), hard_timeout_s=0.05)
    with TestClient(create_app(resources=resources)) as client:
        response = client.post("/v1/assist/turn", json=_turn(), headers=_auth())
    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["degraded"] is True
    assert body["meta"]["degraded_reason"] == "hard_timeout"
    assert body["picks"] == []


def test_missing_device_binding_is_503_degraded_not_client_hint() -> None:
    unbound = SeededProfile(
        token="dev-unbound",
        user_id="user_unbound",
        profile_id="profile_unbound",
        geo="US",
        package=Package.PREMIUM,
        maturity_max=MaturityRating.R,
        kids=False,
        device_class=DeviceClass.WEB,
        device_bound=False,
    )
    graph = RecordingGraph()
    resources = make_resources(
        graph=graph,
        profiles=ProfileCatalog((unbound,)),
    )
    with TestClient(create_app(resources=resources)) as client:
        response = client.post(
            "/v1/assist/turn",
            json=_turn(client_hints={"device_class": "tv"}),
            headers=_auth("dev-unbound"),
        )
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "degraded"
    assert body["picks"] == []
    assert graph.calls == []


def test_stats_counts_turns_and_errors(client: TestClient) -> None:
    client.post("/v1/assist/turn", json=_turn(), headers=_auth())
    client.post("/v1/assist/turn", json=_turn())
    snapshot = client.get("/stats").json()
    assert snapshot["turns"] >= 1
    assert snapshot["unauthorized"] >= 1


def test_real_stub_graph_turn_runs() -> None:
    resources = make_resources(graph=build_graph())
    with TestClient(create_app(resources=resources)) as client:
        response = client.post("/v1/assist/turn", json=_turn(), headers=_auth())
    assert response.status_code == 200
    body = response.json()
    assert "session_id" in body
    assert "picks" in body
    assert "chips" in body
    assert "meta" in body


def test_ctx_on_graph_is_server_user_ctx(client: TestClient, graph: RecordingGraph) -> None:
    client.post("/v1/assist/turn", json=_turn(), headers=_auth())
    ctx = graph.calls[0]["ctx"]
    assert isinstance(ctx, ServerUserCtx)
    assert ctx.profile_id == "profile_adult"
