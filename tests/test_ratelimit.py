"""Lua token-bucket rate limiter: atomicity, isolation, TTL."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from uuid import uuid4

import pytest
from redis.asyncio import Redis
from testcontainers.community.redis import RedisContainer

from assist.stores.ratelimit import RateLimited, RateLimiter, rate_limit_key


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


async def test_burst_then_deny(redis_client: Redis) -> None:
    limiter = RateLimiter(redis_client, rps=5, burst=3)
    subject = f"seq-{uuid4().hex}"
    allowed = [await limiter.allow("user", subject) for _ in range(4)]
    assert [d.allowed for d in allowed] == [True, True, True, False]
    assert allowed[-1].retry_after_ms > 0
    assert allowed[-1].limit == 3


async def test_acquire_raises_rate_limited(redis_client: Redis) -> None:
    limiter = RateLimiter(redis_client, rps=1, burst=1)
    subject = f"acq-{uuid4().hex}"
    await limiter.acquire("user", subject)
    with pytest.raises(RateLimited) as exc:
        await limiter.acquire("user", subject)
    assert exc.value.error_type == "rate_limited"
    assert exc.value.retry_after_ms > 0


async def test_buckets_are_isolated_by_scope_and_subject(redis_client: Redis) -> None:
    limiter = RateLimiter(redis_client, rps=1, burst=1)
    a = f"iso-{uuid4().hex}"
    b = f"iso-{uuid4().hex}"
    assert (await limiter.allow("user", a)).allowed is True
    assert (await limiter.allow("user", a)).allowed is False
    assert (await limiter.allow("user", b)).allowed is True
    assert (await limiter.allow("ip", a)).allowed is True


async def test_atomic_under_50_concurrent_calls(redis_client: Redis) -> None:
    burst = 20
    limiter = RateLimiter(redis_client, rps=1, burst=burst)
    subject = f"conc-{uuid4().hex}"
    barrier = asyncio.Barrier(50)

    async def hit() -> bool:
        await barrier.wait()
        return (await limiter.allow("user", subject)).allowed

    results = await asyncio.gather(*[hit() for _ in range(50)])
    allowed = sum(1 for ok in results if ok)
    denied = sum(1 for ok in results if not ok)
    assert allowed == burst, f"expected exactly {burst} allows, got {allowed}"
    assert denied == 50 - burst


async def test_rate_limit_key_ttl_via_pttl(redis_client: Redis) -> None:
    # burst/rps * 2 = 8s rolling TTL
    limiter = RateLimiter(redis_client, rps=5, burst=20)
    subject = f"ttl-{uuid4().hex}"
    await limiter.allow("user", subject)
    pttl = int(await redis_client.pttl(rate_limit_key("user", subject)))
    assert pttl > 0
    assert pttl <= 8000
    assert pttl > 3000
