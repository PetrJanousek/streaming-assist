"""Postgres schema, repositories, and fire-and-forget turn_events writes."""

from __future__ import annotations

import os
import subprocess
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.pool import QueuePool
from testcontainers.community.postgres import PostgresContainer

from assist.config import settings
from assist.domain.catalog import Person, Title
from assist.domain.enums import (
    CreditRole,
    DegradedReason,
    DeviceClass,
    GenreId,
    MaturityRating,
    MediaType,
    MoodId,
    Package,
    Route,
    SpeechAct,
)
from assist.stores.db import (
    CATALOG_TABLES,
    AvailabilityWindow,
    CreditRecord,
    Database,
    GoldenQuery,
    PhraseTemplate,
    ProfileRecord,
    TaxonomyEntry,
    TurnEvent,
    create_db_engine,
)

ROOT = Path(__file__).resolve().parents[1]
TRUNCATE_SQL = "TRUNCATE " + ", ".join(CATALOG_TABLES) + " RESTART IDENTITY CASCADE"


def _run_alembic_upgrade(dsn: str) -> str:
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
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"alembic upgrade head failed ({proc.returncode}):\n{output}")
    return output


@pytest.fixture(scope="session")
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
        await conn.execute(text(TRUNCATE_SQL))
    try:
        yield db
    finally:
        await db.dispose()


def _title(**overrides: object) -> Title:
    payload: dict[str, object] = {
        "catalog_id": "s1",
        "media_type": MediaType.FILM,
        "title": "The Matrix",
        "synopsis": "A hacker learns the truth.",
        "release_year": 1999,
        "runtime_min": 136,
        "maturity_rank": 6,
        "origins": ("United States",),
        "genres": (GenreId.SCIFI, GenreId.ACTION),
        "moods": (MoodId.TENSE, MoodId.DARK),
        "local_original": False,
        "pop_28d": 0.91,
    }
    payload.update(overrides)
    return Title.model_validate(payload)


def _person(**overrides: object) -> Person:
    payload: dict[str, object] = {
        "person_id": "p1",
        "name": "Keanu Reeves",
        "name_norm": "keanu reeves",
        "roles": (CreditRole.ACTOR,),
        "credit_count": 12,
        "active_year_min": 1986,
        "active_year_max": 2021,
        "popularity": 8.2,
    }
    payload.update(overrides)
    return Person.model_validate(payload)


def test_pool_sized_from_config() -> None:
    engine = create_db_engine(
        settings.postgres_dsn,
        pool_size=7,
        max_overflow=3,
    )
    try:
        pool = engine.sync_engine.pool
        assert isinstance(pool, QueuePool)
        assert pool.size() == 7
        assert settings.postgres_pool_size == 5
        assert settings.postgres_max_overflow == 10
    finally:
        engine.sync_engine.dispose()


async def test_alembic_upgrade_creates_all_tables(database: Database) -> None:
    async with database.engine.connect() as conn:
        result = await conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        )
        names = {row[0] for row in result}
    assert set(CATALOG_TABLES) <= names
    assert "alembic_version" in names
    async with database.engine.connect() as conn:
        version = await conn.scalar(text("SELECT version_num FROM alembic_version"))
    assert version == "0001"


def test_alembic_upgrade_is_idempotent(postgres_dsn: str) -> None:
    _run_alembic_upgrade(postgres_dsn)


async def test_titles_round_trip(database: Database) -> None:
    title = _title()
    enrichment: dict[str, object] = {
        "tags": ["slow-burn", "dystopia"],
        "audience": "adult",
        "pace": "fast",
        "era_feel": "90s sci-fi",
        "one_line_hook": "Reality is a lie.",
    }
    indexed_at = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)

    async with database.session() as session:
        await database.titles(session).upsert(title, enrichment=enrichment, indexed_at=indexed_at)

    async with database.session() as session:
        stored = await database.titles(session).get_stored("s1")

    assert stored is not None
    assert stored.title == title
    assert stored.enrichment is not None
    assert stored.enrichment["tags"] == ["slow-burn", "dystopia"]
    assert stored.enrichment["moods"] == ["tense", "dark"]
    assert stored.indexed_at == indexed_at

    updated = _title(synopsis="Updated synopsis.", pop_28d=0.5)
    async with database.session() as session:
        await database.titles(session).upsert(updated)

    async with database.session() as session:
        again = await database.titles(session).get_stored("s1")

    assert again is not None
    assert again.title.synopsis == "Updated synopsis."
    assert again.title.pop_28d == 0.5
    # Re-upsert without enrichment must not wipe the enrichment column.
    assert again.enrichment is not None
    assert again.enrichment["one_line_hook"] == "Reality is a lie."
    assert again.indexed_at == indexed_at


async def test_people_round_trip(database: Database) -> None:
    person = _person()
    async with database.session() as session:
        await database.people(session).upsert(person)

    async with database.session() as session:
        got = await database.people(session).get("p1")

    assert got == person

    bumped = _person(credit_count=99, popularity=9.9)
    async with database.session() as session:
        await database.people(session).upsert(bumped)

    async with database.session() as session:
        got = await database.people(session).get("p1")
    assert got == bumped


async def test_credits_round_trip(database: Database) -> None:
    async with database.session() as session:
        await database.titles(session).upsert(_title())
        await database.people(session).upsert(_person())
        await database.credits(session).upsert(
            CreditRecord(catalog_id="s1", person_id="p1", role=CreditRole.ACTOR)
        )
        await database.credits(session).upsert(
            CreditRecord(catalog_id="s1", person_id="p1", role=CreditRole.DIRECTOR)
        )
        # Idempotent: same PK is a no-op, not an error.
        await database.credits(session).upsert(
            CreditRecord(catalog_id="s1", person_id="p1", role=CreditRole.ACTOR)
        )

    async with database.session() as session:
        credits = await database.credits(session).list_for_title("s1")

    roles = {c.role for c in credits}
    assert roles == {CreditRole.ACTOR, CreditRole.DIRECTOR}
    assert all(c.catalog_id == "s1" and c.person_id == "p1" for c in credits)


async def test_availability_round_trip(database: Database) -> None:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2099, 1, 1, tzinfo=UTC)
    window = AvailabilityWindow(
        catalog_id="s1",
        package=Package.PREMIUM,
        geo="US",
        window_start=start,
        window_end=end,
        playable=True,
    )
    async with database.session() as session:
        await database.titles(session).upsert(_title())
        await database.availability(session).upsert(window)

    async with database.session() as session:
        got = await database.availability(session).get("s1", Package.PREMIUM, "US")

    assert got == window

    closed = window.model_copy(update={"playable": False})
    async with database.session() as session:
        await database.availability(session).upsert(closed)

    async with database.session() as session:
        got = await database.availability(session).get("s1", Package.PREMIUM, "US")
    assert got is not None
    assert got.playable is False
    async with database.session() as session:
        missing = await database.availability(session).get("s1", Package.BASIC, "US")
    assert missing is None


async def test_taxonomy_round_trip(database: Database) -> None:
    entry = TaxonomyEntry(
        kind="genres",
        id="drama",
        label="Drama",
        synonyms=("dramas", "TV Dramas"),
    )
    async with database.session() as session:
        await database.taxonomy(session).upsert(entry)
        await database.taxonomy(session).upsert(
            TaxonomyEntry(kind="moods", id="cozy", label="Cozy", synonyms=("cosy",))
        )

    async with database.session() as session:
        got = await database.taxonomy(session).get("genres", "drama")
        by_kind = await database.taxonomy(session).list_by_kind("genres")

    assert got == entry
    assert by_kind == [entry]


async def test_phrase_bank_round_trip(database: Database) -> None:
    phrase = PhraseTemplate(
        id="chip.refine_mood.funnier",
        speech_act=SpeechAct.REFINE_MOOD,
        kind="chip",
        template="Something funnier",
    )
    async with database.session() as session:
        await database.phrase_bank(session).upsert(phrase)

    async with database.session() as session:
        got = await database.phrase_bank(session).get(phrase.id)
        listed = await database.phrase_bank(session).list_by_speech_act(SpeechAct.REFINE_MOOD)

    assert got == phrase
    assert listed == [phrase]


async def test_profiles_round_trip(database: Database) -> None:
    profile = ProfileRecord(
        profile_id="prof_adult",
        token="tok_adult",
        maturity_max=MaturityRating.R,
        kids=False,
        geo="US",
        package=Package.PREMIUM,
        device_class=DeviceClass.TV,
    )
    async with database.session() as session:
        await database.profiles(session).upsert(profile)

    async with database.session() as session:
        by_id = await database.profiles(session).get("prof_adult")
        by_token = await database.profiles(session).get_by_token("tok_adult")

    assert by_id == profile
    assert by_token == profile


async def test_golden_queries_round_trip(database: Database) -> None:
    query = GoldenQuery(
        id="g1",
        text="something cozy for tonight",
        expect_ids=("s1", "s2"),
        expect_class="mood_genre",
        slice="mood",
    )
    async with database.session() as session:
        await database.golden_queries(session).upsert(query)

    async with database.session() as session:
        got = await database.golden_queries(session).get("g1")
        listed = await database.golden_queries(session).list_all()

    assert got == query
    assert listed == [query]


async def test_turn_events_round_trip(database: Database) -> None:
    event_id = uuid4()
    event = TurnEvent(
        id=event_id,
        trace_id="tr-1",
        session_id="sess-1",
        route=Route.TEMPLATE,
        intent_source="rules",
        degraded_reason=DegradedReason.NONE,
        stage_latency_ms={"retrieve": 12, "rank": 3},
        tokens_in=10,
        tokens_out=4,
        cost_usd=0.0011,
    )
    await database.turn_events.record(event)
    got = await database.turn_events.get(event_id)
    assert got is not None
    assert got.trace_id == "tr-1"
    assert got.session_id == "sess-1"
    assert got.route == Route.TEMPLATE
    assert got.intent_source == "rules"
    assert got.degraded_reason == DegradedReason.NONE
    assert got.stage_latency_ms == {"retrieve": 12, "rank": 3}
    assert got.tokens_in == 10
    assert got.tokens_out == 4
    assert got.cost_usd == pytest.approx(0.0011)
    assert got.created_at is not None


async def test_turn_events_insert_failure_does_not_raise(database: Database) -> None:
    event = TurnEvent(
        id=uuid4(),
        trace_id="tr-dup",
        session_id="sess-dup",
        route=Route.TEMPLATE,
        intent_source="rules",
        degraded_reason=DegradedReason.NONE,
    )
    await database.turn_events.record(event)
    # Duplicate PK is a real INSERT failure; the caller must not see it.
    await database.turn_events.record(event)
    got = await database.turn_events.get(event.id)
    assert got is not None
    assert got.trace_id == "tr-dup"


async def test_titles_get_missing_returns_none(database: Database) -> None:
    async with database.session() as session:
        assert await database.titles(session).get("missing") is None
