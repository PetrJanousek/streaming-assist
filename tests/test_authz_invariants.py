"""Client input is never authority. Assert on the constructed ServerUserCtx."""

from __future__ import annotations

import inspect

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from assist.api.deps import (
    AppResources,
    build_server_user_ctx,
    default_profile_catalog,
)
from assist.domain.constraints import ConstraintDelta
from assist.domain.context import ServerUserCtx
from assist.domain.enums import DeviceClass, MaturityRating, Package
from assist.graph.state import TurnState
from assist.main import create_app
from assist.stores.ratelimit import RateLimitDecision
from assist.stores.session import Session, SessionBindError

# Kids fixture: TV / BASIC / US / PG / kids=True. The client will send the opposite.
_KIDS = next(p for p in default_profile_catalog().list_all() if p.token == "dev-kids")

_SPOOF = {
    "device_class": "mobile",
    "geo": "DE",
    "package": "premium",
    "maturity": "NC-17",
    "maturity_max": "NC-17",
    "kids": False,
    "kids_flag": False,
}


class _PingRedis:
    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


class _Cache:
    async def get_idempotent(self, raw_key: str) -> str | None:
        return None

    async def set_idempotent(self, raw_key: str, payload: str) -> None:
        return None


class _Limiter:
    async def acquire(self, scope: str, subject: str, *, cost: int = 1) -> RateLimitDecision:
        return RateLimitDecision(allowed=True, remaining=19.0, retry_after_ms=0, limit=20)


class _Sessions:
    async def load(self, session_id: str, user_id: str, profile_id: str) -> Session:
        return Session.create(session_id=session_id, user_id=user_id, profile_id=profile_id)

    async def save(self, session: Session) -> None:
        if session.session_id == "__bind__":
            raise SessionBindError(
                session_id=session.session_id,
                bound_user_id="other",
                bound_profile_id="other",
                user_id=session.user_id,
                profile_id=session.profile_id,
            )


class RecordingGraph:
    def __init__(self) -> None:
        self.calls: list[TurnState] = []

    async def ainvoke(self, state: TurnState, **_kwargs: object) -> TurnState:
        self.calls.append(state)
        out = dict(state)
        out["reply"] = "ok"
        out["route"] = "template"
        return out  # type: ignore[return-value]


def _resources(graph: RecordingGraph, playable_now: object) -> AppResources:
    return AppResources(
        redis=_PingRedis(),
        cache=_Cache(),
        sessions=_Sessions(),
        rate_limiter=_Limiter(),
        graph=graph,
        profiles=default_profile_catalog(),
        playable_now=playable_now,  # type: ignore[arg-type]
    )


def test_builder_signature_rejects_client_hints() -> None:
    params = list(inspect.signature(build_server_user_ctx).parameters)
    assert params == ["profile"]
    ctx = build_server_user_ctx(_KIDS)
    assert ctx.device_class == DeviceClass.TV
    assert ctx.geo == "US"
    assert ctx.package == Package.BASIC
    assert ctx.maturity_max == MaturityRating.PG
    assert ctx.kids_flag is True


def test_server_user_ctx_forbids_extra_and_has_no_hint_fields() -> None:
    fields = set(ServerUserCtx.model_fields)
    assert "client_hints" not in fields
    assert fields == {
        "user_id",
        "profile_id",
        "geo",
        "package",
        "maturity_max",
        "kids_flag",
        "device_class",
    }
    with pytest.raises(ValidationError):
        ServerUserCtx(
            user_id="u",
            profile_id="p",
            geo="US",
            package=Package.BASIC,
            maturity_max=MaturityRating.PG,
            kids_flag=True,
            device_class=DeviceClass.TV,
            extra_key="nope",  # type: ignore[call-arg]
        )


def test_constraint_delta_rejects_authz_keys() -> None:
    for payload in (
        {"geo": "DE"},
        {"package": "premium"},
        {"kids_flag": True},
        {"device_class": "tv"},
        {"maturity_max": "NC-17"},
    ):
        with pytest.raises(ValidationError):
            ConstraintDelta.model_validate(payload)


def test_constructed_ctx_and_playable_now_ignore_client_spoof() -> None:
    """AuthZ fields on ctx come from the seeded profile, never the request body.

    playable_now is the T20 seam: it is invoked with the constructed ctx object
    (or not at all). Either way it cannot see client_hints — those are not on
    ServerUserCtx and are not passed into the graph state.
    """
    graph = RecordingGraph()
    seen_by_playable: list[ServerUserCtx] = []

    def playable_now(ctx: ServerUserCtx) -> None:
        seen_by_playable.append(ctx)

    body: dict[str, object] = {
        "session_id": None,
        "message": {"type": "text", "text": "cozy"},
        "client_hints": _SPOOF,
        **_SPOOF,
    }
    with TestClient(create_app(resources=_resources(graph, playable_now))) as client:
        response = client.post(
            "/v1/assist/turn",
            json=body,
            headers={"Authorization": "Bearer dev-kids"},
        )
    assert response.status_code == 200

    assert len(graph.calls) == 1
    ctx = graph.calls[0]["ctx"]
    assert isinstance(ctx, ServerUserCtx)

    assert ctx.user_id == _KIDS.user_id
    assert ctx.profile_id == _KIDS.profile_id
    assert ctx.device_class == DeviceClass.TV
    assert ctx.device_class.value != "mobile"
    assert ctx.geo == "US"
    assert ctx.geo != "DE"
    assert ctx.package == Package.BASIC
    assert ctx.package.value != "premium"
    assert ctx.maturity_max == MaturityRating.PG
    assert ctx.maturity_max.value != "NC-17"
    assert ctx.kids_flag is True

    assert "client_hints" not in graph.calls[0]

    assert seen_by_playable == [ctx]
    spy_ctx = seen_by_playable[0]
    assert spy_ctx is ctx
    assert spy_ctx.device_class == DeviceClass.TV
    assert spy_ctx.geo == "US"
    assert spy_ctx.package == Package.BASIC
    assert spy_ctx.maturity_max == MaturityRating.PG
    assert spy_ctx.kids_flag is True
    assert not hasattr(spy_ctx, "client_hints")


def test_adult_profile_not_widened_by_kids_false_spoof() -> None:
    graph = RecordingGraph()
    seen: list[ServerUserCtx] = []
    with TestClient(create_app(resources=_resources(graph, seen.append))) as client:
        response = client.post(
            "/v1/assist/turn",
            json={
                "message": {"type": "text", "text": "cozy"},
                "client_hints": {
                    "device_class": "tv",
                    "kids": True,
                    "geo": "FR",
                    "package": "basic",
                    "maturity_max": "TV-Y",
                },
            },
            headers={"Authorization": "Bearer dev-adult"},
        )
    assert response.status_code == 200
    ctx = graph.calls[0]["ctx"]
    assert ctx.device_class == DeviceClass.WEB
    assert ctx.kids_flag is False
    assert ctx.geo == "US"
    assert ctx.package == Package.PREMIUM
    assert ctx.maturity_max == MaturityRating.R
    assert seen[0] is ctx
