"""Session repository, in-session chips, and the four Redis caches."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta

import pytest
from redis.asyncio import Redis
from testcontainers.community.redis import RedisContainer

from assist.domain.constraints import AddOp, ConstraintDelta, ConstraintState, SetOp
from assist.domain.context import ServerUserCtx
from assist.domain.enums import (
    DeviceClass,
    GenreId,
    MaturityRating,
    MoodId,
    Package,
    SpeechAct,
)
from assist.stores.cache import (
    AVAIL_TTL_S,
    IDEMPOTENCY_TTL_S,
    INTENT_TTL_S,
    RESPONSE_TTL_S,
    CacheStore,
    availability_key,
    constraints_hash,
    ctx_hash,
    idempotency_key,
    intent_cache_key,
    response_cache_key,
)
from assist.stores.session import (
    TURN_HISTORY_CAP,
    ChipInvalid,
    ChipRecord,
    Session,
    SessionBindError,
    SessionRepository,
    TurnSummary,
    session_key,
)


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


def _ctx(**overrides: object) -> ServerUserCtx:
    payload: dict[str, object] = {
        "user_id": "u1",
        "profile_id": "p1",
        "geo": "US",
        "package": Package.PREMIUM,
        "maturity_max": MaturityRating.R,
        "kids_flag": False,
        "device_class": DeviceClass.WEB,
    }
    payload.update(overrides)
    return ServerUserCtx.model_validate(payload)


def _delta() -> ConstraintDelta:
    return ConstraintDelta(moods=AddOp(values=("funny",)))


def _assert_ttl_ms(pttl: int, ttl_s: int, *, slack_ms: int = 5000) -> None:
    assert pttl > 0
    assert pttl <= ttl_s * 1000
    assert pttl >= ttl_s * 1000 - slack_ms, f"pttl={pttl} not near {ttl_s}s"


def test_unknown_chip_raises_chip_invalid() -> None:
    session = Session.create(user_id="u1", profile_id="p1")
    with pytest.raises(ChipInvalid) as exc:
        session.lookup_chip("c_missing")
    assert exc.value.error_type == "chip_invalid"
    assert exc.value.chip_id == "c_missing"


def test_expired_chip_raises_chip_invalid() -> None:
    session = Session.create(user_id="u1", profile_id="p1")
    expired = ChipRecord(
        chip_id="c_old",
        label="funnier",
        delta=_delta(),
        speech_act=SpeechAct.REFINE_MOOD,
        minted_turn=0,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    session = session.model_copy(update={"issued_chips": {"c_old": expired}})
    with pytest.raises(ChipInvalid) as exc:
        session.lookup_chip("c_old")
    assert exc.value.error_type == "chip_invalid"


def test_mint_and_lookup_chip_roundtrip() -> None:
    session = Session.create(user_id="u1", profile_id="p1")
    session, chip = session.mint_chip(
        label="Something funnier",
        delta=_delta(),
        speech_act=SpeechAct.REFINE_MOOD,
    )
    found = session.lookup_chip(chip.chip_id)
    assert found.chip_id == chip.chip_id
    assert found.label == "Something funnier"
    assert found.delta == _delta()
    assert found.speech_act == SpeechAct.REFINE_MOOD
    assert chip.chip_id.startswith("c_")


def test_turn_history_caps_at_six() -> None:
    session = Session.create(user_id="u1", profile_id="p1")
    for i in range(8):
        session = session.append_turn(
            TurnSummary(message_type="text", text=f"turn-{i}", reply=f"r{i}")
        )
    assert session.turn_count == 8
    assert len(session.turns) == TURN_HISTORY_CAP
    assert [t.text for t in session.turns] == [f"turn-{i}" for i in range(2, 8)]


async def test_load_miss_is_new_session_and_does_not_write(redis_client: Redis) -> None:
    repo = SessionRepository(redis_client)
    session = await repo.load("s-new", "u1", "p1")
    assert session.session_id == "s-new"
    assert session.constraints == ConstraintState()
    assert await redis_client.get(session_key("s-new")) is None


async def test_save_load_roundtrip_preserves_chips_and_constraints(
    redis_client: Redis,
) -> None:
    repo = SessionRepository(redis_client)
    session = Session.create(session_id="s1", user_id="u1", profile_id="p1")
    session = session.with_constraints(
        ConstraintState(genres_include=(GenreId.COMEDY,), local_originals_only=True)
    )
    session, chip = session.mint_chip(
        label="More like this",
        delta=ConstraintDelta(local_originals_only=SetOp(value=False)),
        speech_act=SpeechAct.TOGGLE_LOCAL_ORIGINALS,
    )
    session = session.append_turn(
        TurnSummary(message_type="chip", text=chip.label, pick_ids=("ttl_1",))
    )
    session = session.with_ctx_echo(_ctx())
    await repo.save(session)

    loaded = await repo.load("s1", "u1", "p1")
    assert loaded.constraints.genres_include == (GenreId.COMEDY,)
    assert loaded.constraints.local_originals_only is True
    assert loaded.turn_count == 1
    assert loaded.lookup_chip(chip.chip_id).delta == chip.delta
    assert loaded.server_ctx_echo is not None
    assert loaded.server_ctx_echo.device_class == DeviceClass.WEB
    # Chips live on the session key, not a parallel keyspace.
    assert set(await redis_client.keys("*")) == {session_key("s1")}


async def test_cross_profile_bind_is_rejected(redis_client: Redis) -> None:
    repo = SessionRepository(redis_client)
    session = Session.create(session_id="s-bind", user_id="u1", profile_id="p1")
    await repo.save(session)

    with pytest.raises(SessionBindError) as wrong_profile:
        await repo.load("s-bind", "u1", "p-other")
    assert wrong_profile.value.error_type == "session_bind_rejected"

    with pytest.raises(SessionBindError) as wrong_user:
        await repo.load("s-bind", "u-other", "p1")
    assert wrong_user.value.error_type == "session_bind_rejected"

    thief = Session.create(session_id="s-bind", user_id="u1", profile_id="p-thief")
    with pytest.raises(SessionBindError):
        await repo.save(thief)

    kept = await repo.load("s-bind", "u1", "p1")
    assert kept.profile_id == "p1"


async def test_session_ttl_via_pttl_and_slides_on_load(redis_client: Redis) -> None:
    repo = SessionRepository(redis_client, ttl_s=10)
    session = Session.create(session_id="s-ttl", user_id="u1", profile_id="p1")
    await repo.save(session)
    key = session_key("s-ttl")
    _assert_ttl_ms(int(await redis_client.pttl(key)), 10, slack_ms=2000)

    await redis_client.expire(key, 2)
    remaining = int(await redis_client.pttl(key))
    assert remaining <= 2000
    await repo.load("s-ttl", "u1", "p1")
    slid = int(await redis_client.pttl(key))
    assert slid > remaining
    assert slid > 5000


async def test_default_session_ttl_is_24h(redis_client: Redis) -> None:
    repo = SessionRepository(redis_client)
    session = Session.create(session_id="s-24h", user_id="u1", profile_id="p1")
    await repo.save(session)
    _assert_ttl_ms(int(await redis_client.pttl(session_key("s-24h"))), 86400)


async def test_cache_roundtrips_and_ttls(redis_client: Redis) -> None:
    cache = CacheStore(redis_client)
    state = ConstraintState(moods=(MoodId.FUNNY,))
    ctx = _ctx()
    c_hash = constraints_hash(state)
    u_hash = ctx_hash(ctx)

    await cache.set_intent("cozy tonight", c_hash, '{"moods":["cozy"]}')
    assert await cache.get_intent("cozy tonight", c_hash) == '{"moods":["cozy"]}'
    assert await cache.get_intent("other", c_hash) is None
    _assert_ttl_ms(
        int(await redis_client.pttl(intent_cache_key("cozy tonight", c_hash))),
        INTENT_TTL_S,
    )

    await cache.set_response("cozy tonight", c_hash, u_hash, '{"reply":"ok"}')
    assert await cache.get_response("cozy tonight", c_hash, u_hash) == '{"reply":"ok"}'
    _assert_ttl_ms(
        int(await redis_client.pttl(response_cache_key("cozy tonight", c_hash, u_hash))),
        RESPONSE_TTL_S,
    )

    await cache.set_availability("s1", "premium", "US", False)
    assert await cache.get_availability("s1", "premium", "US") is False
    assert await cache.get_availability("s1", "basic", "US") is None
    _assert_ttl_ms(
        int(await redis_client.pttl(availability_key("s1", "premium", "US"))),
        AVAIL_TTL_S,
        slack_ms=2000,
    )

    await cache.set_availability_many(
        [("s2", "premium", "US", True), ("s3", "premium", "US", False)]
    )
    many = await cache.get_availability_many(
        [("s2", "premium", "US"), ("s3", "premium", "US"), ("s-miss", "premium", "US")]
    )
    assert many == [True, False, None]

    await cache.set_idempotent("idem-1", '{"session_id":"s1"}')
    assert await cache.get_idempotent("idem-1") == '{"session_id":"s1"}'
    assert await cache.get_idempotent("idem-miss") is None
    _assert_ttl_ms(
        int(await redis_client.pttl(idempotency_key("idem-1"))),
        IDEMPOTENCY_TTL_S,
    )
