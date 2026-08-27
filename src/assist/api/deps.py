"""Auth → ServerUserCtx. Client hints never enter this module's builder."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Protocol

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict

from assist.domain.context import ServerUserCtx
from assist.domain.enums import DegradedReason, DeviceClass, MaturityRating, Package
from assist.stores.session import Session

_bearer = HTTPBearer(auto_error=False)


class Unauthorized(Exception):
    """Missing or unknown bearer token. Maps to HTTP 401."""

    error_type = "unauthorized"


class DegradedError(Exception):
    """Service cannot honour this turn. Maps to HTTP 503 degraded."""

    error_type = "degraded"

    def __init__(self, message: str, *, reason: DegradedReason | None = None) -> None:
        self.reason = reason
        super().__init__(message)


class SeededProfile(BaseModel):
    """Dev fixture standing in for profile + entitlement + device registry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    token: str
    user_id: str
    profile_id: str
    geo: str
    package: Package
    maturity_max: MaturityRating
    kids: bool
    device_class: DeviceClass
    label: str = ""
    # False simulates a device-registry miss. Fail closed — never use client_hints.
    device_bound: bool = True


class ProfileCatalog:
    def __init__(self, profiles: Sequence[SeededProfile]) -> None:
        self._profiles = tuple(profiles)
        self._by_token = {p.token: p for p in self._profiles}

    def get_by_token(self, token: str) -> SeededProfile | None:
        return self._by_token.get(token)

    def list_all(self) -> tuple[SeededProfile, ...]:
        return self._profiles


def default_profile_catalog() -> ProfileCatalog:
    return ProfileCatalog(
        (
            SeededProfile(
                token="dev-adult",
                user_id="user_adult",
                profile_id="profile_adult",
                geo="US",
                package=Package.PREMIUM,
                maturity_max=MaturityRating.R,
                kids=False,
                device_class=DeviceClass.WEB,
                label="Adult US premium (web)",
            ),
            SeededProfile(
                token="dev-kids",
                user_id="user_kids",
                profile_id="profile_kids",
                geo="US",
                package=Package.BASIC,
                maturity_max=MaturityRating.PG,
                kids=True,
                device_class=DeviceClass.TV,
                label="Kids US basic (tv)",
            ),
            SeededProfile(
                token="dev-basic",
                user_id="user_basic",
                profile_id="profile_basic",
                geo="DE",
                package=Package.BASIC,
                maturity_max=MaturityRating.PG_13,
                kids=False,
                device_class=DeviceClass.MOBILE,
                label="Adult DE basic (mobile)",
            ),
        )
    )


def build_server_user_ctx(profile: SeededProfile) -> ServerUserCtx:
    """Trusted AuthZ floor. The signature takes a profile and nothing else.

    client_hints, geo spoofing, package, maturity and kids from the body have
    no parameter here, so they cannot reach ServerUserCtx or playable_now.
    """
    if not profile.device_bound:
        raise DegradedError(
            "trusted device identity unavailable",
            reason=DegradedReason.RETRIEVAL_UNAVAILABLE,
        )
    return ServerUserCtx(
        user_id=profile.user_id,
        profile_id=profile.profile_id,
        geo=profile.geo,
        package=profile.package,
        maturity_max=profile.maturity_max,
        kids_flag=profile.kids,
        device_class=profile.device_class,
    )


class GraphLike(Protocol):
    # LangGraph CompiledStateGraph.ainvoke is a forest of overloads; we call ainvoke(state).
    def ainvoke(self, *args: Any, **kwargs: Any) -> Awaitable[Any]: ...


class RateLimiterLike(Protocol):
    def acquire(self, scope: str, subject: str, *, cost: int = 1) -> Awaitable[object]: ...


class IdempotencyStore(Protocol):
    def get_idempotent(self, raw_key: str) -> Awaitable[str | None]: ...

    def set_idempotent(self, raw_key: str, payload: str) -> Awaitable[None]: ...


class SessionStore(Protocol):
    def load(self, session_id: str, user_id: str, profile_id: str) -> Awaitable[Session]: ...

    def save(self, session: Session) -> Awaitable[None]: ...


class RedisPing(Protocol):
    def ping(self) -> Awaitable[object]: ...

    def aclose(self) -> Awaitable[None]: ...


@dataclass
class Stats:
    turns: int = 0
    rate_limited: int = 0
    unauthorized: int = 0
    chip_invalid: int = 0
    degraded: int = 0
    started_monotonic: float = field(default_factory=time.monotonic)


PlayableNowFn = Callable[[ServerUserCtx], object]


@dataclass
class AppResources:
    redis: RedisPing
    cache: IdempotencyStore
    sessions: SessionStore
    rate_limiter: RateLimiterLike
    graph: GraphLike
    profiles: ProfileCatalog
    stats: Stats = field(default_factory=Stats)
    hard_timeout_s: float | None = None
    # Optional spy for tests / future T20. Production leaves this None.
    playable_now: PlayableNowFn | None = None


def resources_of(request: Request) -> AppResources:
    return request.app.state.resources  # type: ignore[no-any-return]


async def get_server_user_ctx(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> ServerUserCtx:
    """Bearer token → seeded profile → ServerUserCtx. No request body is read."""
    if creds is None or not creds.credentials:
        raise Unauthorized()
    resources = resources_of(request)
    profile = resources.profiles.get_by_token(creds.credentials)
    if profile is None:
        raise Unauthorized()
    ctx = build_server_user_ctx(profile)
    request.state.server_user_ctx = ctx
    # T20's playable_now will take TurnState.ctx. If a spy is installed, it
    # sees this same object — never client_hints.
    if resources.playable_now is not None:
        resources.playable_now(ctx)
    return ctx
