"""Redis token-bucket rate limiter.

The bucket lives in Lua so INCR-and-check cannot race across api replicas.
Key layout: `rl:{scope}:{subject}` (plan §4.3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.commands.core import AsyncScript

from assist.config import settings
from assist.obs.logging import get_logger

log = get_logger(__name__)

# TIME + HSET + EXPIRE in one EVAL: two replicas cannot both observe "tokens
# remaining" and both consume the last token.
_TOKEN_BUCKET_LUA = """
local rate = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])

local t = redis.call('TIME')
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

local data = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])

if tokens == nil or ts == nil then
  tokens = burst
  ts = now_ms
end

local elapsed_ms = now_ms - ts
if elapsed_ms < 0 then
  elapsed_ms = 0
end

tokens = math.min(burst, tokens + (elapsed_ms * rate) / 1000.0)
ts = now_ms

local allowed = 0
local retry_after_ms = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  local deficit = cost - tokens
  if rate > 0 then
    retry_after_ms = math.ceil((deficit / rate) * 1000)
  else
    retry_after_ms = ttl * 1000
  end
end

redis.call('HSET', KEYS[1], 'tokens', tokens, 'ts', ts)
redis.call('EXPIRE', KEYS[1], ttl)
return {allowed, tostring(tokens), retry_after_ms}
"""


def rate_limit_key(scope: str, subject: str) -> str:
    return f"rl:{scope}:{subject}"


class RateLimited(Exception):
    """Caller exceeded the bucket. Maps to HTTP 429 rate_limited."""

    error_type = "rate_limited"

    def __init__(self, *, scope: str, subject: str, retry_after_ms: int) -> None:
        self.scope = scope
        self.subject = subject
        self.retry_after_ms = retry_after_ms
        super().__init__(f"rate limited: {scope}:{subject}")


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    remaining: float
    retry_after_ms: int
    limit: int


class RateLimiter:
    """Token bucket. Defaults come from RATE_LIMIT_RPS / RATE_LIMIT_BURST."""

    def __init__(
        self,
        redis: Redis,
        *,
        rps: int | None = None,
        burst: int | None = None,
    ) -> None:
        self._redis = redis
        self._rps = settings.rate_limit_rps if rps is None else rps
        self._burst = settings.rate_limit_burst if burst is None else burst
        if self._rps < 1 or self._burst < 1:
            raise ValueError("rps and burst must be >= 1")
        self._script: AsyncScript = redis.register_script(_TOKEN_BUCKET_LUA)
        # Idle buckets expire after they would have refilled twice. Shorter
        # would snap an empty bucket back to `burst` before refill completes.
        self._ttl_s = max(1, math.ceil(self._burst / self._rps) * 2)

    async def allow(self, scope: str, subject: str, *, cost: int = 1) -> RateLimitDecision:
        if cost < 1:
            raise ValueError("cost must be >= 1")
        key = rate_limit_key(scope, subject)
        raw = await self._script(
            keys=[key],
            args=[str(self._rps), str(self._burst), str(cost), str(self._ttl_s)],
        )
        if not isinstance(raw, (list, tuple)) or len(raw) < 3:
            raise RuntimeError("token bucket lua returned unexpected result")
        allowed = int(raw[0]) == 1
        remaining = float(raw[1])
        retry_after_ms = int(raw[2])
        if not allowed:
            log.info(
                "rate_limited",
                scope=scope,
                subject=subject,
                remaining=remaining,
                retry_after_ms=retry_after_ms,
            )
        return RateLimitDecision(
            allowed=allowed,
            remaining=remaining,
            retry_after_ms=retry_after_ms,
            limit=self._burst,
        )

    async def acquire(self, scope: str, subject: str, *, cost: int = 1) -> RateLimitDecision:
        decision = await self.allow(scope, subject, cost=cost)
        if not decision.allowed:
            raise RateLimited(scope=scope, subject=subject, retry_after_ms=decision.retry_after_ms)
        return decision
