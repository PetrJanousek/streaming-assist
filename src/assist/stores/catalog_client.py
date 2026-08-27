"""Catalog rights client. The seam for a future catalog-service container.

`playable_now` is fail-closed: a missing row, an expired window, a timeout,
or any exception means not playable. Cached True cannot outlive a known
window_end — that is how a stale cache is blocked from resurrecting a title
whose window already ended.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Protocol

from assist.config import settings
from assist.domain.context import ServerUserCtx
from assist.domain.enums import DeviceClass, Package
from assist.obs.logging import get_logger
from assist.stores.cache import AVAIL_TTL_S
from assist.stores.db import AvailabilityWindow, Database

log = get_logger("assist.stores.catalog_client")


class CatalogClientError(Exception):
    """Lookup failed. Callers must treat every affected id as not playable."""


class CatalogClient(Protocol):
    """Fine-rights check. Swap the impl; the node keeps this method."""

    async def playable_now(
        self, catalog_ids: Sequence[str], ctx: ServerUserCtx
    ) -> Mapping[str, bool]:
        """Return per-id flags. Missing keys are not playable."""
        ...


class AvailabilityCache(Protocol):
    async def get_availability_many(
        self, items: Sequence[tuple[str, str, str]]
    ) -> list[bool | None]: ...

    async def set_availability_many(self, items: Sequence[tuple[str, str, str, bool]]) -> None: ...


class WindowLookup(Protocol):
    async def __call__(
        self, catalog_ids: list[str], package: Package, geo: str
    ) -> Sequence[AvailabilityWindow]: ...


def window_is_playable(window: AvailabilityWindow | None, now: datetime) -> bool:
    """Playable flag and current window. Missing window is not playable."""
    if window is None:
        return False
    if not window.playable:
        return False
    return window.window_start <= now <= window.window_end


def _true_outlives_cache(window: AvailabilityWindow, now: datetime, ttl_s: int) -> bool:
    """Refuse to cache True when the window ends inside the cache TTL."""
    return window.window_end >= now + timedelta(seconds=ttl_s)


def _device_ok(ctx: ServerUserCtx) -> bool:
    # No per-device column in the T04 schema. An unexpected value still
    # fails closed so a missing trusted device cannot entitle a title.
    return isinstance(ctx.device_class, DeviceClass)


async def evaluate_playable_now(
    catalog_ids: Sequence[str],
    ctx: ServerUserCtx,
    *,
    lookup: WindowLookup,
    cache: AvailabilityCache | None = None,
    now: datetime | None = None,
    timeout_s: float | None = None,
    avail_ttl_s: int = AVAIL_TTL_S,
) -> dict[str, bool]:
    """Batch rights check. Cache hits skip `lookup`; failures drop the id."""
    now = now if now is not None else datetime.now(UTC)
    timeout_s = timeout_s if timeout_s is not None else settings.catalog_timeout_ms / 1000.0

    log.info(
        "playable_now",
        n=len(catalog_ids),
        geo=ctx.geo,
        package=ctx.package.value,
        device_class=ctx.device_class.value,
        maturity_max=ctx.maturity_max.value,
        kids_flag=ctx.kids_flag,
    )

    if not _device_ok(ctx):
        return {catalog_id: False for catalog_id in catalog_ids}
    if not catalog_ids:
        return {}

    package = ctx.package.value
    geo = ctx.geo
    ids = list(catalog_ids)
    flags = await _cache_get(cache, ids, package, geo)

    result: dict[str, bool] = {}
    misses: list[str] = []
    for catalog_id, cached in zip(ids, flags, strict=True):
        if cached is None:
            misses.append(catalog_id)
        else:
            result[catalog_id] = cached

    if not misses:
        return _apply_device(result, ctx)

    windows_by_id: dict[str, AvailabilityWindow] = {}
    try:
        windows = await _lookup_windows(lookup, misses, ctx.package, ctx.geo, timeout_s)
    except CatalogClientError:
        log.exception("catalog_lookup_failed_closed", n=len(misses))
        for catalog_id in misses:
            result[catalog_id] = False
        return _apply_device(result, ctx)

    for window in windows:
        windows_by_id[window.catalog_id] = window

    to_cache: list[tuple[str, str, str, bool]] = []
    for catalog_id in misses:
        found = windows_by_id.get(catalog_id)
        playable = window_is_playable(found, now)
        result[catalog_id] = playable
        if playable:
            # found is not None: window_is_playable is False for None.
            if found is not None and _true_outlives_cache(found, now, avail_ttl_s):
                to_cache.append((catalog_id, package, geo, True))
        else:
            to_cache.append((catalog_id, package, geo, False))

    await _cache_put(cache, to_cache)
    return _apply_device(result, ctx)


def _apply_device(flags: dict[str, bool], ctx: ServerUserCtx) -> dict[str, bool]:
    if _device_ok(ctx):
        return flags
    return dict.fromkeys(flags, False)


async def _cache_get(
    cache: AvailabilityCache | None,
    catalog_ids: list[str],
    package: str,
    geo: str,
) -> list[bool | None]:
    if cache is None:
        return [None] * len(catalog_ids)
    try:
        raw = await cache.get_availability_many(
            [(catalog_id, package, geo) for catalog_id in catalog_ids]
        )
    except Exception:
        log.exception("avail_cache_get_failed")
        return [None] * len(catalog_ids)
    if len(raw) != len(catalog_ids):
        log.info("avail_cache_malformed", expected=len(catalog_ids), got=len(raw))
        return [None] * len(catalog_ids)
    return list(raw)


async def _cache_put(
    cache: AvailabilityCache | None,
    items: list[tuple[str, str, str, bool]],
) -> None:
    if cache is None or not items:
        return
    try:
        await cache.set_availability_many(items)
    except Exception:
        log.exception("avail_cache_set_failed")


async def _lookup_windows(
    lookup: WindowLookup,
    catalog_ids: list[str],
    package: Package,
    geo: str,
    timeout_s: float,
) -> list[AvailabilityWindow]:
    try:
        async with asyncio.timeout(timeout_s):
            return list(await lookup(catalog_ids, package, geo))
    except TimeoutError as exc:
        raise CatalogClientError("catalog lookup timed out") from exc
    except CatalogClientError:
        raise
    except Exception as exc:
        raise CatalogClientError("catalog lookup failed") from exc


class PostgresCatalogClient:
    """Postgres availability windows + Redis cache. Fail closed on I/O errors."""

    def __init__(
        self,
        database: Database | None = None,
        cache: AvailabilityCache | None = None,
        *,
        timeout_ms: int | None = None,
        clock: Callable[[], datetime] | None = None,
        lookup: WindowLookup | None = None,
    ) -> None:
        if database is None and lookup is None:
            msg = "PostgresCatalogClient requires database or lookup"
            raise ValueError(msg)
        self._database = database
        self._cache = cache
        self._timeout_ms = timeout_ms if timeout_ms is not None else settings.catalog_timeout_ms
        self._clock = clock if clock is not None else (lambda: datetime.now(UTC))
        self._lookup = lookup

    async def playable_now(self, catalog_ids: Sequence[str], ctx: ServerUserCtx) -> dict[str, bool]:
        lookup = self._lookup if self._lookup is not None else self._db_lookup
        return await evaluate_playable_now(
            catalog_ids,
            ctx,
            lookup=lookup,
            cache=self._cache,
            now=self._clock(),
            timeout_s=self._timeout_ms / 1000.0,
        )

    async def _db_lookup(
        self, catalog_ids: list[str], package: Package, geo: str
    ) -> list[AvailabilityWindow]:
        if self._database is None:
            raise CatalogClientError("catalog database is not configured")
        async with self._database.session() as session:
            return await self._database.availability(session).list_for_package_geo(
                catalog_ids, package, geo
            )
