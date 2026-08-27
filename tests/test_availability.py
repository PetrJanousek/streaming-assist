"""Availability node + CatalogClient. Fail closed; never substitute a drop."""

from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import AsyncIterator, Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest
from sqlalchemy import text
from testcontainers.community.postgres import PostgresContainer

from assist.domain.catalog import Candidate, Title
from assist.domain.context import ServerUserCtx
from assist.domain.enums import DeviceClass, GenreId, MaturityRating, MediaType, Package
from assist.graph.state import TurnState, empty_turn_state
from assist.nodes.availability import playable_now, validate_availability
from assist.stores.cache import AVAIL_TTL_S
from assist.stores.catalog_client import (
    PostgresCatalogClient,
    evaluate_playable_now,
    window_is_playable,
)
from assist.stores.db import AvailabilityWindow, Database

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
FAR = datetime(2099, 12, 31, 23, 59, 59, tzinfo=UTC)
PAST = datetime(2015, 6, 1, tzinfo=UTC)
OPEN_START = datetime(2000, 1, 1, tzinfo=UTC)

_SPOOF = {
    "device_class": "mobile",
    "geo": "DE",
    "package": "premium",
    "maturity": "NC-17",
    "kids": False,
}


def _ctx(
    *,
    geo: str = "US",
    package: Package = Package.BASIC,
    maturity_max: MaturityRating = MaturityRating.PG,
    kids_flag: bool = True,
    device_class: DeviceClass = DeviceClass.TV,
) -> ServerUserCtx:
    return ServerUserCtx(
        user_id="user_kids",
        profile_id="profile_kids",
        geo=geo,
        package=package,
        maturity_max=maturity_max,
        kids_flag=kids_flag,
        device_class=device_class,
    )


def _candidate(catalog_id: str, title: str = "Title") -> Candidate:
    return Candidate(
        catalog_id=catalog_id,
        title=title,
        media_type=MediaType.FILM,
        release_year=2010,
        genres=(GenreId.DRAMA,),
        score=1.0,
    )


def _window(
    catalog_id: str,
    *,
    playable: bool = True,
    package: Package = Package.BASIC,
    geo: str = "US",
    window_start: datetime = OPEN_START,
    window_end: datetime = FAR,
) -> AvailabilityWindow:
    return AvailabilityWindow(
        catalog_id=catalog_id,
        package=package,
        geo=geo,
        window_start=window_start,
        window_end=window_end,
        playable=playable,
    )


def _title(catalog_id: str) -> Title:
    return Title(
        catalog_id=catalog_id,
        media_type=MediaType.FILM,
        title=catalog_id,
        synopsis="",
        release_year=2010,
        maturity_rank=4,
        origins=("United States",),
        genres=(GenreId.DRAMA,),
    )


class FakeCache:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str, str], bool] = {}
        self.gets = 0
        self.sets = 0
        self.set_items: list[tuple[str, str, str, bool]] = []

    async def get_availability_many(
        self, items: Sequence[tuple[str, str, str]]
    ) -> list[bool | None]:
        self.gets += 1
        return [self.store.get((catalog_id, package, geo)) for catalog_id, package, geo in items]

    async def set_availability_many(self, items: Sequence[tuple[str, str, str, bool]]) -> None:
        self.sets += 1
        self.set_items.extend(items)
        for catalog_id, package, geo, playable in items:
            self.store[(catalog_id, package, geo)] = playable

    def expire_all(self) -> None:
        self.store.clear()


class CountingLookup:
    def __init__(self, windows: dict[str, AvailabilityWindow] | None = None) -> None:
        self.windows = dict(windows or {})
        self.calls = 0
        self.last_ids: tuple[str, ...] = ()
        self.last_package: Package | None = None
        self.last_geo: str | None = None
        self.packages: list[Package] = []
        self.geos: list[str] = []

    async def __call__(
        self, catalog_ids: list[str], package: Package, geo: str
    ) -> list[AvailabilityWindow]:
        self.calls += 1
        self.last_ids = tuple(catalog_ids)
        self.last_package = package
        self.last_geo = geo
        self.packages.append(package)
        self.geos.append(geo)
        matched: list[AvailabilityWindow] = []
        for catalog_id in catalog_ids:
            window = self.windows.get(catalog_id)
            if window is None:
                continue
            if window.package is package and window.geo == geo:
                matched.append(window)
        return matched


class BoomLookup:
    async def __call__(
        self, catalog_ids: list[str], package: Package, geo: str
    ) -> list[AvailabilityWindow]:
        raise RuntimeError("catalog down")


class SlowLookup:
    def __init__(self, delay_s: float = 1.0) -> None:
        self.delay_s = delay_s
        self.calls = 0

    async def __call__(
        self, catalog_ids: list[str], package: Package, geo: str
    ) -> list[AvailabilityWindow]:
        self.calls += 1
        await asyncio.sleep(self.delay_s)
        return []


class RecordingClient:
    def __init__(self, flags: dict[str, bool] | None = None) -> None:
        self.flags = dict(flags or {})
        self.ctxs: list[ServerUserCtx] = []
        self.ids: list[tuple[str, ...]] = []

    async def playable_now(self, catalog_ids: Sequence[str], ctx: ServerUserCtx) -> dict[str, bool]:
        self.ctxs.append(ctx)
        self.ids.append(tuple(catalog_ids))
        return {catalog_id: self.flags.get(catalog_id, False) for catalog_id in catalog_ids}


class BoomClient:
    def __init__(self) -> None:
        self.ctxs: list[ServerUserCtx] = []

    async def playable_now(self, catalog_ids: Sequence[str], ctx: ServerUserCtx) -> dict[str, bool]:
        self.ctxs.append(ctx)
        raise RuntimeError("catalog client exploded")


class ExtraIdClient:
    async def playable_now(self, catalog_ids: Sequence[str], ctx: ServerUserCtx) -> dict[str, bool]:
        out = {catalog_id: True for catalog_id in catalog_ids}
        out["not-in-list"] = True
        return out


def _kept(out: dict[str, object]) -> tuple[Candidate, ...]:
    raw = out["candidates"]
    assert isinstance(raw, tuple)
    return cast(tuple[Candidate, ...], raw)


# ---------------------------------------------------------------------------
# window predicate
# ---------------------------------------------------------------------------


def test_window_missing_is_not_playable() -> None:
    assert window_is_playable(None, NOW) is False


def test_window_flag_false_is_not_playable() -> None:
    assert window_is_playable(_window("s1", playable=False), NOW) is False


def test_window_expired_is_not_playable_even_if_flag_true() -> None:
    window = _window("s1", playable=True, window_end=PAST)
    assert window_is_playable(window, NOW) is False


def test_window_not_yet_open_is_not_playable() -> None:
    window = _window("s1", playable=True, window_start=FAR, window_end=FAR)
    assert window_is_playable(window, NOW) is False


def test_window_open_and_flagged_is_playable() -> None:
    assert window_is_playable(_window("s1", playable=True), NOW) is True


# ---------------------------------------------------------------------------
# evaluate_playable_now (CatalogClient core)
# ---------------------------------------------------------------------------


async def test_lookup_uses_server_ctx_package_and_geo() -> None:
    lookup = CountingLookup({"s1": _window("s1", package=Package.BASIC, geo="US")})
    ctx = _ctx(geo="US", package=Package.BASIC, device_class=DeviceClass.TV)
    flags = await evaluate_playable_now(["s1"], ctx, lookup=lookup, now=NOW)
    assert flags == {"s1": True}
    assert lookup.last_package is Package.BASIC
    assert lookup.last_geo == "US"
    assert lookup.last_package is ctx.package
    assert lookup.last_geo == ctx.geo


async def test_package_gated_title_drops_for_basic() -> None:
    lookup = CountingLookup({"s1": _window("s1", playable=False, package=Package.BASIC, geo="US")})
    flags = await evaluate_playable_now(["s1"], _ctx(package=Package.BASIC), lookup=lookup, now=NOW)
    assert flags == {"s1": False}


async def test_geo_restricted_title_drops_for_home_geo() -> None:
    # Row exists only for XX; home geo has no window → fail closed.
    lookup = CountingLookup({"s1": _window("s1", geo="XX", playable=True)})
    flags = await evaluate_playable_now(["s1"], _ctx(geo="US"), lookup=lookup, now=NOW)
    assert flags == {"s1": False}
    assert lookup.last_geo == "US"


async def test_missing_row_is_not_playable() -> None:
    flags = await evaluate_playable_now(["ghost"], _ctx(), lookup=CountingLookup(), now=NOW)
    assert flags == {"ghost": False}


async def test_lookup_exception_fails_closed() -> None:
    flags = await evaluate_playable_now(["s1"], _ctx(), lookup=BoomLookup(), now=NOW)
    assert flags == {"s1": False}


async def test_lookup_timeout_fails_closed() -> None:
    slow = SlowLookup(delay_s=1.0)
    flags = await evaluate_playable_now(["s1"], _ctx(), lookup=slow, now=NOW, timeout_s=0.05)
    assert flags == {"s1": False}
    assert slow.calls == 1


async def test_error_is_not_written_to_cache() -> None:
    cache = FakeCache()
    flags = await evaluate_playable_now(["s1"], _ctx(), lookup=BoomLookup(), cache=cache, now=NOW)
    assert flags == {"s1": False}
    assert cache.store == {}
    assert cache.sets == 0


async def test_cache_hit_skips_lookup() -> None:
    lookup = CountingLookup({"s1": _window("s1")})
    cache = FakeCache()
    ctx = _ctx()
    first = await evaluate_playable_now(["s1"], ctx, lookup=lookup, cache=cache, now=NOW)
    assert first == {"s1": True}
    assert lookup.calls == 1
    second = await evaluate_playable_now(["s1"], ctx, lookup=lookup, cache=cache, now=NOW)
    assert second == {"s1": True}
    assert lookup.calls == 1
    assert cache.gets == 2


async def test_do_not_cache_true_when_window_ends_within_ttl() -> None:
    end = NOW + timedelta(seconds=10)
    lookup = CountingLookup(
        {"s1": _window("s1", playable=True, window_start=OPEN_START, window_end=end)}
    )
    cache = FakeCache()
    first = await evaluate_playable_now(
        ["s1"], _ctx(), lookup=lookup, cache=cache, now=NOW, avail_ttl_s=AVAIL_TTL_S
    )
    assert first == {"s1": True}
    assert lookup.calls == 1
    assert cache.store == {}

    later = NOW + timedelta(seconds=11)
    second = await evaluate_playable_now(
        ["s1"], _ctx(), lookup=lookup, cache=cache, now=later, avail_ttl_s=AVAIL_TTL_S
    )
    assert second == {"s1": False}
    assert lookup.calls == 2


async def test_stale_cache_after_ttl_does_not_resurrect_unplayable() -> None:
    lookup = CountingLookup({"s1": _window("s1", playable=True)})
    cache = FakeCache()
    ctx = _ctx()
    first = await evaluate_playable_now(["s1"], ctx, lookup=lookup, cache=cache, now=NOW)
    assert first == {"s1": True}
    assert ("s1", "basic", "US") in cache.store

    lookup.windows["s1"] = _window("s1", playable=False, window_end=PAST)
    still_cached = await evaluate_playable_now(["s1"], ctx, lookup=lookup, cache=cache, now=NOW)
    assert still_cached == {"s1": True}
    assert lookup.calls == 1

    cache.expire_all()
    after_ttl = await evaluate_playable_now(["s1"], ctx, lookup=lookup, cache=cache, now=NOW)
    assert after_ttl == {"s1": False}
    assert lookup.calls == 2


async def test_cached_false_does_not_flip_to_true_inside_ttl() -> None:
    lookup = CountingLookup({"s1": _window("s1", playable=False)})
    cache = FakeCache()
    ctx = _ctx()
    first = await evaluate_playable_now(["s1"], ctx, lookup=lookup, cache=cache, now=NOW)
    assert first == {"s1": False}
    lookup.windows["s1"] = _window("s1", playable=True)
    second = await evaluate_playable_now(["s1"], ctx, lookup=lookup, cache=cache, now=NOW)
    assert second == {"s1": False}
    assert lookup.calls == 1


async def test_postgres_catalog_client_records_timeout() -> None:
    client = PostgresCatalogClient(lookup=SlowLookup(delay_s=1.0), timeout_ms=50, clock=lambda: NOW)
    flags = await client.playable_now(["s1"], _ctx())
    assert flags == {"s1": False}


# ---------------------------------------------------------------------------
# node: drop, never substitute; server ctx only
# ---------------------------------------------------------------------------


async def test_non_playable_title_never_survives() -> None:
    ctx = _ctx()
    client = RecordingClient({"alive": True, "dead": False, "also-dead": False})
    state = empty_turn_state(
        ctx,
        candidates=(
            _candidate("alive", "Alive"),
            _candidate("dead", "Dead"),
            _candidate("also-dead", "Also Dead"),
        ),
    )
    out = await validate_availability(state, client=client)
    kept = _kept(out)
    entitled = out["entitled_ids"]
    assert [c.catalog_id for c in kept] == ["alive"]
    assert [c.title for c in kept] == ["Alive"]
    assert entitled == ("alive",)
    assert all(isinstance(c, Candidate) for c in kept)


async def test_drop_does_not_substitute_a_different_title() -> None:
    ctx = _ctx()
    client = RecordingClient({"a": True, "b": False, "c": True})
    original = (
        _candidate("a", "Alpha"),
        _candidate("b", "Bravo"),
        _candidate("c", "Charlie"),
    )
    out = await validate_availability(empty_turn_state(ctx, candidates=original), client=client)
    kept = _kept(out)
    assert [c.catalog_id for c in kept] == ["a", "c"]
    assert [c.title for c in kept] == ["Alpha", "Charlie"]
    assert kept[0] is original[0]
    assert kept[1] is original[2]
    assert "Bravo" not in [c.title for c in kept]


async def test_extra_true_id_from_client_is_not_added() -> None:
    ctx = _ctx()
    original = (_candidate("a", "Alpha"),)
    out = await validate_availability(
        empty_turn_state(ctx, candidates=original), client=ExtraIdClient()
    )
    kept = _kept(out)
    assert [c.catalog_id for c in kept] == ["a"]
    assert out["entitled_ids"] == ("a",)


async def test_client_exception_drops_every_candidate() -> None:
    ctx = _ctx()
    client = BoomClient()
    original = (_candidate("a", "Alpha"), _candidate("b", "Bravo"))
    out = await validate_availability(empty_turn_state(ctx, candidates=original), client=client)
    assert out["candidates"] == ()
    assert out["entitled_ids"] == ()
    assert client.ctxs == [ctx]


async def test_missing_client_fails_closed() -> None:
    state = empty_turn_state(_ctx(), candidates=(_candidate("a"),))
    out = await validate_availability(state, client=None)
    assert out["candidates"] == ()
    assert out["entitled_ids"] == ()


async def test_missing_ctx_fails_closed() -> None:
    client = RecordingClient({"a": True})
    state = empty_turn_state(_ctx(), candidates=(_candidate("a"),))
    raw = dict(state)
    raw.pop("ctx")
    out = await validate_availability(cast(TurnState, raw), client=client)
    assert out["candidates"] == ()
    assert out["entitled_ids"] == ()
    assert client.ctxs == []


async def test_playable_now_runs_on_empty_candidates_and_receives_ctx() -> None:
    ctx = _ctx()
    client = RecordingClient()
    out = await validate_availability(empty_turn_state(ctx, candidates=()), client=client)
    assert out["candidates"] == ()
    assert out["entitled_ids"] == ()
    assert client.ctxs == [ctx]
    assert client.ids == [()]


async def test_device_class_used_is_server_bound() -> None:
    ctx = _ctx(device_class=DeviceClass.TV)
    client = RecordingClient({"s1": True})
    state = dict(empty_turn_state(ctx, candidates=(_candidate("s1"),)))
    state["client_hints"] = _SPOOF
    out = await validate_availability(cast(TurnState, state), client=client)
    assert out["entitled_ids"] == ("s1",)
    assert len(client.ctxs) == 1
    seen = client.ctxs[0]
    assert seen is ctx
    assert seen.device_class is DeviceClass.TV
    assert seen.device_class.value != "mobile"
    assert not hasattr(seen, "client_hints")


async def test_playable_now_receives_server_geo_package_maturity_kids() -> None:
    ctx = _ctx(
        geo="US",
        package=Package.BASIC,
        maturity_max=MaturityRating.PG,
        kids_flag=True,
        device_class=DeviceClass.TV,
    )
    client = RecordingClient({"s1": True})
    state = dict(empty_turn_state(ctx, candidates=(_candidate("s1"),)))
    state["client_hints"] = _SPOOF
    await validate_availability(cast(TurnState, state), client=client)
    seen = client.ctxs[0]
    assert seen.geo == "US"
    assert seen.geo != "DE"
    assert seen.package is Package.BASIC
    assert seen.package.value != "premium"
    assert seen.maturity_max is MaturityRating.PG
    assert seen.maturity_max.value != "NC-17"
    assert seen.kids_flag is True
    assert seen.device_class is DeviceClass.TV


async def test_playable_now_wrapper_strict_true_only() -> None:
    class LooseClient:
        async def playable_now(
            self, catalog_ids: Sequence[str], ctx: ServerUserCtx
        ) -> dict[str, object]:
            return {"s1": "yes", "s2": True, "s3": False}

    ctx = _ctx()
    flags = await playable_now(["s1", "s2", "s3"], ctx, LooseClient())  # type: ignore[arg-type]
    assert flags == {"s1": False, "s2": True, "s3": False}


async def test_order_of_survivors_matches_rank_order() -> None:
    ctx = _ctx()
    client = RecordingClient({"c": True, "a": True, "b": False})
    original = (
        _candidate("b", "B"),
        _candidate("c", "C"),
        _candidate("a", "A"),
    )
    out = await validate_availability(empty_turn_state(ctx, candidates=original), client=client)
    assert [c.catalog_id for c in _kept(out)] == ["c", "a"]
    assert out["entitled_ids"] == ("c", "a")


# ---------------------------------------------------------------------------
# Postgres batch helper + PostgresCatalogClient (real DB)
# ---------------------------------------------------------------------------


def _run_alembic_upgrade(dsn: str) -> None:
    env = os.environ.copy()
    env["POSTGRES_DSN"] = dsn
    proc = subprocess.run(
        ["uv", "run", "alembic", "upgrade", "head"],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        output = (proc.stdout or "") + (proc.stderr or "")
        raise RuntimeError(f"alembic upgrade head failed ({proc.returncode}):\n{output}")


@pytest.fixture(scope="module")
def postgres_dsn() -> Iterator[str]:
    with PostgresContainer(
        "postgres:16-alpine",
        username="assist",
        password="assist",
        dbname="assist",
        driver="asyncpg",
    ) as container:
        dsn = container.get_connection_url()
        _run_alembic_upgrade(dsn)
        yield dsn


@pytest.fixture
async def database(postgres_dsn: str) -> AsyncIterator[Database]:
    db = Database.from_dsn(postgres_dsn, pool_size=2, max_overflow=1)
    async with db.engine.begin() as conn:
        await conn.execute(text("TRUNCATE titles, availability RESTART IDENTITY CASCADE"))
    try:
        yield db
    finally:
        await db.dispose()


async def test_list_for_package_geo_returns_only_matching_rows(database: Database) -> None:
    async with database.session() as session:
        await database.titles(session).upsert(_title("s1"))
        await database.titles(session).upsert(_title("s2"))
        await database.availability(session).upsert(_window("s1", package=Package.BASIC, geo="US"))
        await database.availability(session).upsert(
            _window("s1", package=Package.PREMIUM, geo="US", playable=False)
        )
        await database.availability(session).upsert(_window("s2", package=Package.BASIC, geo="DE"))

    async with database.session() as session:
        rows = await database.availability(session).list_for_package_geo(
            ["s1", "s2", "s-miss"], Package.BASIC, "US"
        )

    assert {row.catalog_id for row in rows} == {"s1"}
    assert rows[0].package is Package.BASIC
    assert rows[0].geo == "US"
    assert rows[0].playable is True


async def test_postgres_client_drops_unplayable_and_uses_ctx_rights(
    database: Database,
) -> None:
    async with database.session() as session:
        await database.titles(session).upsert(_title("live"))
        await database.titles(session).upsert(_title("dead"))
        await database.titles(session).upsert(_title("gated"))
        await database.availability(session).upsert(
            _window("live", package=Package.BASIC, geo="US")
        )
        await database.availability(session).upsert(
            _window("dead", playable=False, package=Package.BASIC, geo="US", window_end=PAST)
        )
        await database.availability(session).upsert(
            _window("gated", playable=True, package=Package.PREMIUM, geo="US")
        )

    cache = FakeCache()
    client = PostgresCatalogClient(database, cache, timeout_ms=2000, clock=lambda: NOW)
    ctx = _ctx(package=Package.BASIC, geo="US", device_class=DeviceClass.TV)
    flags = await client.playable_now(["live", "dead", "gated", "ghost"], ctx)
    assert flags == {"live": True, "dead": False, "gated": False, "ghost": False}

    again = await client.playable_now(["live", "dead"], ctx)
    assert again == {"live": True, "dead": False}
    assert cache.gets >= 2
