"""Scale overlay: three replicas share Redis session + rate-limit state."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from redis.asyncio import Redis
from testcontainers.community.redis import RedisContainer

from assist.api.deps import AppResources, default_profile_catalog
from assist.domain.constraints import AddOp, ConstraintDelta, ConstraintState
from assist.domain.enums import GenreId, SpeechAct
from assist.graph.state import TurnState
from assist.main import create_app
from assist.stores.cache import CacheStore
from assist.stores.ratelimit import RateLimiter
from assist.stores.session import Session, SessionRepository, TurnSummary

ROOT = Path(__file__).resolve().parents[1]
NGINX_CONF = ROOT / "docker" / "nginx" / "default.conf"
SCALE_OVERLAY = ROOT / "docker-compose.scale.yml"
BASE_COMPOSE = ROOT / "docker-compose.yml"
README = ROOT / "README.md"
N_REPLICAS = 3


class _PingRedis:
    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


class _FakeCache:
    def __init__(self) -> None:
        self.idem: dict[str, str] = {}

    async def get_idempotent(self, raw_key: str) -> str | None:
        return self.idem.get(raw_key)

    async def set_idempotent(self, raw_key: str, payload: str) -> None:
        self.idem[raw_key] = payload


class _FakeRateLimiter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def acquire(self, scope: str, subject: str, *, cost: int = 1) -> object:
        self.calls.append((scope, subject))
        return object()


class _FakeSessionStore:
    async def load(self, session_id: str, user_id: str, profile_id: str) -> Session:
        return Session.create(session_id=session_id, user_id=user_id, profile_id=profile_id)

    async def save(self, session: Session) -> None:
        return None


class RecordingGraph:
    def __init__(self) -> None:
        self.calls: list[TurnState] = []

    async def ainvoke(self, state: TurnState, **_kwargs: object) -> TurnState:
        self.calls.append(state)
        out = dict(state)
        out["reply"] = "ok"
        out["route"] = "template"
        return out  # type: ignore[return-value]


def _auth(token: str = "dev-adult") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _turn(text: str = "something cozy", *, session_id: str | None = None) -> dict[str, object]:
    body: dict[str, object] = {"message": {"type": "text", "text": text}}
    if session_id is not None:
        body["session_id"] = session_id
    return body


def _round_robin(n: int) -> Iterator[int]:
    """Force replica index 0, 1, ..., n-1, 0, ... — no sticky sessions."""
    i = 0
    while True:
        yield i % n
        i += 1


@pytest.fixture(scope="module")
def redis_url() -> Iterator[str]:
    with RedisContainer("redis:7-alpine") as container:
        host = container.get_container_host_ip()
        port = int(container.get_exposed_port(6379))
        yield f"redis://{host}:{port}/0"


@pytest.fixture
async def redis_client(redis_url: str) -> AsyncIterator[Redis]:
    client = Redis.from_url(redis_url, decode_responses=True)
    try:
        await client.flushdb()
        yield client
    finally:
        await client.flushdb()
        await client.aclose()


def _resources(
    redis: Redis,
    *,
    graph: RecordingGraph | None = None,
    rps: int = 5,
    burst: int = 20,
) -> tuple[AppResources, RecordingGraph]:
    recorder = graph if graph is not None else RecordingGraph()
    resources = AppResources(
        redis=redis,
        cache=CacheStore(redis),
        sessions=SessionRepository(redis),
        rate_limiter=RateLimiter(redis, rps=rps, burst=burst),
        graph=recorder,
        profiles=default_profile_catalog(),
    )
    return resources, recorder


@asynccontextmanager
async def _clients_for(
    redis: Redis,
    *,
    n: int = N_REPLICAS,
    rps: int = 5,
    burst: int = 20,
) -> AsyncIterator[tuple[list[AsyncClient], list[RecordingGraph]]]:
    graphs = [RecordingGraph() for _ in range(n)]
    apps = [_resources(redis, graph=graphs[i], rps=rps, burst=burst)[0] for i in range(n)]
    clients: list[AsyncClient] = []
    try:
        for resources in apps:
            app = create_app(resources=resources)
            transport = ASGITransport(app=app)
            clients.append(AsyncClient(transport=transport, base_url="http://replica"))
        yield clients, graphs
    finally:
        for client in clients:
            await client.aclose()


# ---------------------------------------------------------------------------
# Session: started on replica i, continued on replica i+1
# ---------------------------------------------------------------------------


async def test_session_round_robin_across_three_replicas(redis_client: Redis) -> None:
    repos = [SessionRepository(redis_client) for _ in range(N_REPLICAS)]
    rr = _round_robin(N_REPLICAS)
    sid = f"s-rr-{uuid4().hex}"

    first = next(rr)
    session = Session.create(session_id=sid, user_id="u1", profile_id="p1")
    session = session.with_constraints(ConstraintState(genres_include=(GenreId.COMEDY,)))
    session, chip = session.mint_chip(
        label="Funnier",
        delta=ConstraintDelta(moods=AddOp(values=("funny",))),
        speech_act=SpeechAct.REFINE_MOOD,
    )
    await repos[first].save(session)

    second = next(rr)
    assert second != first
    loaded = await repos[second].load(sid, "u1", "p1")
    assert loaded.constraints.genres_include == (GenreId.COMEDY,)
    found = loaded.lookup_chip(chip.chip_id)
    assert found.label == "Funnier"

    loaded = loaded.append_turn(TurnSummary(message_type="text", text="more", reply="ok"))
    await repos[second].save(loaded)

    third = next(rr)
    assert third != second
    again = await repos[third].load(sid, "u1", "p1")
    assert again.turn_count == 1
    assert again.constraints.genres_include == (GenreId.COMEDY,)
    assert again.lookup_chip(chip.chip_id).chip_id == chip.chip_id


async def test_http_session_id_continues_on_next_replica(redis_client: Redis) -> None:
    async with _clients_for(redis_client) as (clients, graphs):
        rr = _round_robin(len(clients))
        i0 = next(rr)
        created = await clients[i0].post(
            "/v1/assist/turn", json=_turn("cozy comedy"), headers=_auth()
        )
        assert created.status_code == 200, created.text
        session_id = created.json()["session_id"]
        assert session_id
        assert graphs[i0].calls[0]["session_id"] == session_id

        # Persist a sticky constraint the way the graph's persist node would.
        # The route also heartbeats the pre-graph snapshot; write after the
        # response so replica 1 observes the Redis document, not memory.
        repo = SessionRepository(redis_client)
        stored = await repo.load(session_id, "user_adult", "profile_adult")
        stored = stored.with_constraints(ConstraintState(genres_include=(GenreId.COMEDY,)))
        stored = stored.append_turn(
            TurnSummary(message_type="text", text="cozy comedy", reply="ok")
        )
        await repo.save(stored)

        i1 = next(rr)
        assert i1 != i0
        follow = await clients[i1].post(
            "/v1/assist/turn",
            json=_turn("something funnier", session_id=session_id),
            headers=_auth(),
        )
        assert follow.status_code == 200, follow.text
        assert follow.json()["session_id"] == session_id
        assert graphs[i1].calls[0]["session_id"] == session_id
        constraints = graphs[i1].calls[0]["constraints"]
        assert constraints.genres_include == (GenreId.COMEDY,)
        assert graphs[i1].calls[0]["turn_count"] == 1


# ---------------------------------------------------------------------------
# Rate limiter: one shared budget, not N replica budgets
# ---------------------------------------------------------------------------


async def test_rate_limiter_shared_budget_across_replicas(redis_client: Redis) -> None:
    burst = 3
    limiters = [RateLimiter(redis_client, rps=1, burst=burst) for _ in range(N_REPLICAS)]
    subject = f"scale-rl-{uuid4().hex}"
    rr = _round_robin(N_REPLICAS)

    allowed: list[bool] = []
    replica_hits: list[int] = []
    for _ in range(burst + 1):
        idx = next(rr)
        replica_hits.append(idx)
        allowed.append((await limiters[idx].allow("user", subject)).allowed)

    assert replica_hits[:burst] == [0, 1, 2]
    assert replica_hits[burst] == 0
    assert allowed[:burst] == [True, True, True]
    # Replica 0's second request is denied — its local budget would still
    # have had 2 tokens if the bucket were per-process.
    assert allowed[burst] is False


async def test_shared_budget_is_not_multiplied_by_replica_count(redis_client: Redis) -> None:
    burst = 5
    n_hits = 15
    limiters = [RateLimiter(redis_client, rps=1, burst=burst) for _ in range(N_REPLICAS)]
    subject = f"scale-conc-{uuid4().hex}"
    barrier = asyncio.Barrier(n_hits)

    async def hit(limiter: RateLimiter) -> bool:
        await barrier.wait()
        return (await limiter.allow("user", subject)).allowed

    results = await asyncio.gather(*[hit(limiters[i % N_REPLICAS]) for i in range(n_hits)])
    assert sum(1 for ok in results if ok) == burst
    # Per-replica buckets would have allowed burst * N_REPLICAS = 15.


async def test_http_rate_limit_is_one_bucket_across_replicas(redis_client: Redis) -> None:
    burst = 3
    async with _clients_for(redis_client, rps=1, burst=burst) as (clients, _graphs):
        rr = _round_robin(len(clients))
        statuses: list[int] = []
        last = None
        for _ in range(burst + 1):
            idx = next(rr)
            last = await clients[idx].post(
                "/v1/assist/turn", json=_turn(f"q-{uuid4().hex}"), headers=_auth()
            )
            statuses.append(last.status_code)
        assert statuses == [200, 200, 200, 429]
        assert last is not None
        body = last.json()
        assert body["error"]["type"] == "rate_limited"
        assert body["picks"] == []
        assert body["chips"] == []
        assert "Retry-After" in last.headers


# ---------------------------------------------------------------------------
# nginx + overlay + README
# ---------------------------------------------------------------------------


def test_nginx_config_passthrough_and_dynamic_upstream() -> None:
    text = NGINX_CONF.read_text(encoding="utf-8")
    assert "resolver 127.0.0.11" in text
    assert "set $upstream http://api:8000;" in text
    assert "proxy_pass $upstream;" in text
    # Static proxy_pass of the name would freeze the first replica IP.
    assert "proxy_pass http://api:8000;" not in text
    assert "proxy_set_header X-Request-Id $req_id;" in text
    assert "add_header X-Request-Id $req_id always;" in text
    assert "add_header X-Upstream-Addr $upstream_addr always;" in text
    assert "map $http_x_request_id $req_id" in text
    assert "proxy_buffering off;" in text


def test_scale_overlay_hides_api_port_and_mounts_nginx() -> None:
    text = SCALE_OVERLAY.read_text(encoding="utf-8")
    assert "ports: !reset []" in text
    assert "docker/nginx/default.conf:/etc/nginx/conf.d/default.conf" in text
    assert "configs: !reset []" in text
    assert "--scale api=3" in text
    assert "/healthz" in text


def test_readme_has_required_sections() -> None:
    text = README.read_text(encoding="utf-8")
    required = (
        "## Quickstart",
        "## Seeding",
        "## Architecture",
        "## What is synthetic",
        "## Cost",
        "## Model-tier comparison",
        "## Scale",
        "```mermaid",
        "docker compose -f docker-compose.yml -f docker-compose.scale.yml",
        "--scale api=3",
        "dev-adult",
        "ANTHROPIC_API_KEY",
        "claude-haiku-4-5",
        "claude-sonnet-5",
        "claude-opus-5",
        "Availability windows",
        "pop_28d",
    )
    missing = [heading for heading in required if heading not in text]
    assert missing == []


def test_api_echoes_inbound_x_request_id() -> None:
    resources = AppResources(
        redis=_PingRedis(),
        cache=_FakeCache(),
        sessions=_FakeSessionStore(),
        rate_limiter=_FakeRateLimiter(),
        graph=RecordingGraph(),
        profiles=default_profile_catalog(),
    )
    with TestClient(create_app(resources=resources)) as client:
        response = client.get("/healthz", headers={"X-Request-Id": "gw-trace-1"})
    assert response.status_code == 200
    assert response.headers["X-Request-Id"] == "gw-trace-1"


def test_api_mints_x_request_id_when_missing() -> None:
    resources = AppResources(
        redis=_PingRedis(),
        cache=_FakeCache(),
        sessions=_FakeSessionStore(),
        rate_limiter=_FakeRateLimiter(),
        graph=RecordingGraph(),
        profiles=default_profile_catalog(),
    )
    with TestClient(create_app(resources=resources)) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.headers["X-Request-Id"]


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_scale_overlay_compose_config_validates() -> None:
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(BASE_COMPOSE),
            "-f",
            str(SCALE_OVERLAY),
            "--profile",
            "scale",
            "config",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    rendered = result.stdout
    assert "gateway:" in rendered
    assert "default.conf" in rendered or "nginx" in rendered.lower()


@pytest.mark.skipif(os.environ.get("SCALE_LIVE") != "1", reason="SCALE_LIVE!=1")
def test_live_gateway_round_robin_and_shared_session() -> None:
    """Optional: hit a running `docker compose --profile scale --scale api=3`."""
    import httpx

    base = os.environ.get("SCALE_LIVE_URL", "http://127.0.0.1")
    upstreams: list[str] = []
    with httpx.Client(base_url=base, timeout=5.0) as client:
        for _ in range(6):
            response = client.get("/healthz", headers={"X-Request-Id": "scale-live"})
            assert response.status_code == 200
            assert response.headers.get("X-Request-Id") == "scale-live"
            addr = response.headers.get("X-Upstream-Addr", "")
            if addr:
                upstreams.append(addr)
        created = client.post(
            "/v1/assist/turn",
            json=_turn("cozy comedy"),
            headers=_auth(),
        )
        assert created.status_code == 200
        session_id = created.json()["session_id"]
        follow = client.post(
            "/v1/assist/turn",
            json=_turn("something funnier", session_id=session_id),
            headers=_auth(),
        )
        assert follow.status_code == 200
        assert follow.json()["session_id"] == session_id
    assert len(set(upstreams)) >= 2, f"expected >1 upstream, got {upstreams!r}"


def test_round_robin_helper_covers_all_replicas() -> None:
    rr = _round_robin(N_REPLICAS)
    seen = [next(rr) for _ in range(N_REPLICAS * 2)]
    assert seen == [0, 1, 2, 0, 1, 2]
