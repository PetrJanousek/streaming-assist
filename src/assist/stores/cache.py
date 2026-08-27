"""Intent, response, availability, and idempotency caches (plan §4.3).

TTLs are the spec values, not env knobs: intent 1h, response 5m,
availability 45s, idempotency 5m. Callers inject a Redis client.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from redis.asyncio import Redis

from assist.domain.constraints import ConstraintState
from assist.domain.context import ServerUserCtx

INTENT_TTL_S = 3600
RESPONSE_TTL_S = 300
AVAIL_TTL_S = 45
IDEMPOTENCY_TTL_S = 300


def _sha1(*parts: str) -> str:
    hasher = hashlib.sha1(usedforsecurity=False)
    for i, part in enumerate(parts):
        if i:
            hasher.update(b"\x1f")
        hasher.update(part.encode())
    return hasher.hexdigest()


def constraints_hash(state: ConstraintState) -> str:
    return _sha1(state.model_dump_json())


def ctx_hash(ctx: ServerUserCtx) -> str:
    return _sha1(ctx.model_dump_json())


def intent_cache_key(norm_text: str, constraints_hash_value: str) -> str:
    return f"cache:intent:{_sha1(norm_text, constraints_hash_value)}"


def response_cache_key(norm_text: str, constraints_hash_value: str, ctx_hash_value: str) -> str:
    return f"cache:resp:{_sha1(norm_text, constraints_hash_value, ctx_hash_value)}"


def availability_key(catalog_id: str, package: str, geo: str) -> str:
    return f"cache:avail:{catalog_id}:{package}:{geo}"


def idempotency_key(raw_key: str) -> str:
    return f"idem:{raw_key}"


def _as_str(raw: str | bytes) -> str:
    return raw.decode() if isinstance(raw, bytes) else raw


def _decode_bool(raw: str | bytes | None) -> bool | None:
    if raw is None:
        return None
    value = _as_str(raw)
    if value == "1":
        return True
    if value == "0":
        return False
    return None


class CacheStore:
    """Four Redis caches with fixed TTLs. Values are opaque JSON strings except avail."""

    def __init__(
        self,
        redis: Redis,
        *,
        intent_ttl_s: int = INTENT_TTL_S,
        response_ttl_s: int = RESPONSE_TTL_S,
        avail_ttl_s: int = AVAIL_TTL_S,
        idem_ttl_s: int = IDEMPOTENCY_TTL_S,
    ) -> None:
        self._redis = redis
        self._intent_ttl_s = intent_ttl_s
        self._response_ttl_s = response_ttl_s
        self._avail_ttl_s = avail_ttl_s
        self._idem_ttl_s = idem_ttl_s

    async def get_intent(self, norm_text: str, constraints_hash_value: str) -> str | None:
        raw = await self._redis.get(intent_cache_key(norm_text, constraints_hash_value))
        return None if raw is None else _as_str(raw)

    async def set_intent(self, norm_text: str, constraints_hash_value: str, payload: str) -> None:
        await self._redis.set(
            intent_cache_key(norm_text, constraints_hash_value),
            payload,
            ex=self._intent_ttl_s,
        )

    async def get_response(
        self, norm_text: str, constraints_hash_value: str, ctx_hash_value: str
    ) -> str | None:
        raw = await self._redis.get(
            response_cache_key(norm_text, constraints_hash_value, ctx_hash_value)
        )
        return None if raw is None else _as_str(raw)

    async def set_response(
        self,
        norm_text: str,
        constraints_hash_value: str,
        ctx_hash_value: str,
        payload: str,
    ) -> None:
        await self._redis.set(
            response_cache_key(norm_text, constraints_hash_value, ctx_hash_value),
            payload,
            ex=self._response_ttl_s,
        )

    async def get_availability(self, catalog_id: str, package: str, geo: str) -> bool | None:
        raw = await self._redis.get(availability_key(catalog_id, package, geo))
        return _decode_bool(raw)

    async def set_availability(
        self, catalog_id: str, package: str, geo: str, playable: bool
    ) -> None:
        await self._redis.set(
            availability_key(catalog_id, package, geo),
            "1" if playable else "0",
            ex=self._avail_ttl_s,
        )

    async def get_availability_many(
        self, items: Sequence[tuple[str, str, str]]
    ) -> list[bool | None]:
        if not items:
            return []
        keys = [availability_key(catalog_id, package, geo) for catalog_id, package, geo in items]
        raw = await self._redis.mget(keys)
        return [_decode_bool(value) for value in raw]

    async def set_availability_many(self, items: Sequence[tuple[str, str, str, bool]]) -> None:
        if not items:
            return
        async with self._redis.pipeline(transaction=False) as pipe:
            for catalog_id, package, geo, playable in items:
                pipe.set(
                    availability_key(catalog_id, package, geo),
                    "1" if playable else "0",
                    ex=self._avail_ttl_s,
                )
            await pipe.execute()

    async def get_idempotent(self, raw_key: str) -> str | None:
        raw = await self._redis.get(idempotency_key(raw_key))
        return None if raw is None else _as_str(raw)

    async def set_idempotent(self, raw_key: str, payload: str) -> None:
        await self._redis.set(idempotency_key(raw_key), payload, ex=self._idem_ttl_s)
