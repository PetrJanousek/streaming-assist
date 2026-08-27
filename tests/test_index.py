"""Index job: embedding text, versioned bulk index, resume, alias swap."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import subprocess
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.error import URLError
from urllib.request import urlopen

import httpx
import pytest
from elasticsearch import AsyncElasticsearch
from sqlalchemy import text
from testcontainers.community.postgres import PostgresContainer

from assist.domain.catalog import Person, Title
from assist.domain.enums import CreditRole, GenreId, MediaType, MoodId
from assist.jobs.index import (
    build_embedding_text,
    index_catalog,
    person_document,
    resolve_target_index,
    title_document,
)
from assist.stores.db import CATALOG_TABLES, CreditRecord, Database, TitleRecord
from assist.stores.embed_client import EMBED_DIM, EmbedClient
from assist.stores.es import (
    PEOPLE_ALIAS,
    TITLES_ALIAS,
    close_client,
    create_client,
    create_next_index,
    indices_for_alias,
    swap_alias,
    titles_bm25_body,
    titles_knn_body,
    versioned_index_names,
)
from assist.stores.mappings import EMBEDDING_DIMS, PEOPLE_INDEX_BODY, TITLES_INDEX_BODY

ROOT = Path(__file__).resolve().parents[1]
TRUNCATE_SQL = "TRUNCATE " + ", ".join(CATALOG_TABLES) + " RESTART IDENTITY CASCADE"

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


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


def _record(title: Title, enrichment: dict[str, object] | None = None) -> TitleRecord:
    return TitleRecord(title=title, enrichment=enrichment)


def test_embedding_text_joins_title_synopsis_tags_people_era() -> None:
    text = build_embedding_text(
        _title(),
        tags=["slow-burn", "dystopia"],
        people_names=["Keanu Reeves", "Lana Wachowski"],
        era_feel="90s sci-fi",
    )
    assert "The Matrix" in text
    assert "A hacker learns the truth." in text
    assert "slow-burn dystopia" in text
    assert "Keanu Reeves Lana Wachowski" in text
    assert "90s sci-fi" in text


def test_embedding_text_drops_empty_parts() -> None:
    text = build_embedding_text(_title(synopsis=""), tags=[], people_names=[], era_feel=None)
    assert text == "The Matrix"
    assert "None" not in text


def test_title_document_matches_mapping_fields() -> None:
    enrichment: dict[str, object] = {
        "tags": ["slow-burn", "dystopia"],
        "audience": "adult",
        "pace": "fast",
        "era_feel": "90s sci-fi",
        "moods": ["tense", "dark"],
    }
    vector = [0.01] * EMBED_DIM
    doc = title_document(
        _record(_title(), enrichment),
        people_ids=["p1"],
        people_names=["Keanu Reeves"],
        embedding=vector,
    )
    for field in (
        "catalog_id",
        "media_type",
        "genres",
        "moods",
        "origins",
        "maturity_rank",
        "local_original",
        "release_year",
        "runtime_min",
        "audience",
        "pace",
        "people_ids",
        "pop_28d",
        "title",
        "synopsis",
        "tags",
        "people_names",
        "era_feel",
        "embedding",
    ):
        assert field in doc
    assert doc["catalog_id"] == "s1"
    assert doc["title"] == "The Matrix"
    assert doc["people_ids"] == ["p1"]
    assert doc["tags"] == ["slow-burn", "dystopia"]
    assert doc["era_feel"] == "90s sci-fi"
    assert doc["moods"] == ["tense", "dark"]
    assert len(doc["embedding"]) == EMBEDDING_DIMS == 384


def test_person_document_has_search_fields() -> None:
    doc = person_document(_person())
    assert doc["person_id"] == "p1"
    assert doc["name"] == "Keanu Reeves"
    assert doc["roles"] == ["actor"]
    assert doc["active_year_min"] == 1986
    assert doc["popularity"] == 8.2


def test_title_document_omits_audience_when_unenriched() -> None:
    doc = title_document(
        _record(_title(), None),
        people_ids=[],
        people_names=[],
        embedding=[0.0] * EMBED_DIM,
    )
    assert "audience" not in doc
    assert "pace" not in doc
    assert doc["tags"] == []
    assert doc["era_feel"] == ""
    assert doc["moods"] == ["tense", "dark"]


# ---------------------------------------------------------------------------
# In-memory ES — enough for create_next_index, bulk, mget, count, swap
# ---------------------------------------------------------------------------


class FakeIndices:
    def __init__(self, owner: FakeEs) -> None:
        self._owner = owner
        self.store: dict[str, dict[str, Any]] = {}
        self.aliases: dict[str, set[str]] = {}
        self.update_alias_calls: list[list[Mapping[str, Any]]] = []
        self.delete_calls: list[str] = []
        self.alias_cardinalities: dict[str, list[int]] = {}

    async def create(
        self,
        *,
        index: str,
        mappings: Mapping[str, Any] | None = None,
        settings: Mapping[str, Any] | None = None,
        aliases: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if index in self.store:
            raise RuntimeError(f"index exists: {index}")
        self.store[index] = {"mappings": dict(mappings or {}), "settings": dict(settings or {})}
        self._owner.docs.setdefault(index, {})
        if aliases:
            for alias in aliases:
                self.aliases.setdefault(alias, set()).add(index)
        return {"acknowledged": True}

    async def exists(self, *, index: str) -> bool:
        return index in self.store

    async def get(
        self,
        *,
        index: str,
        ignore_unavailable: bool | None = None,
        allow_no_indices: bool | None = None,
        features: str | None = None,
    ) -> dict[str, Any]:
        del ignore_unavailable, allow_no_indices, features
        return {name: body for name, body in self.store.items() if fnmatch.fnmatch(name, index)}

    async def get_alias(
        self,
        *,
        name: str,
        ignore_unavailable: bool | None = None,
    ) -> dict[str, Any]:
        del ignore_unavailable
        out: dict[str, Any] = {}
        for idx in self.aliases.get(name, ()):
            out[idx] = {"aliases": {name: {}}}
        return out

    async def exists_alias(self, *, name: str) -> bool:
        return bool(self.aliases.get(name))

    async def update_aliases(self, *, actions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        self.update_alias_calls.append(list(actions))
        next_aliases = {k: set(v) for k, v in self.aliases.items()}
        for action in actions:
            if "remove" in action:
                spec = action["remove"]
                assert isinstance(spec, Mapping)
                next_aliases.setdefault(str(spec["alias"]), set()).discard(str(spec["index"]))
            if "add" in action:
                spec = action["add"]
                assert isinstance(spec, Mapping)
                next_aliases.setdefault(str(spec["alias"]), set()).add(str(spec["index"]))
        self.aliases = next_aliases
        for alias, idxs in self.aliases.items():
            self.alias_cardinalities.setdefault(alias, []).append(len(idxs))
        return {"acknowledged": True}

    async def refresh(self, *, index: str) -> dict[str, Any]:
        return {"_shards": {"total": 1, "successful": 1, "failed": 0}, "index": index}

    async def delete(self, *, index: str) -> dict[str, Any]:
        self.delete_calls.append(index)
        raise AssertionError("index job must not delete indices; swap leaves them for rollback")


class FakeEs:
    def __init__(self) -> None:
        self.docs: dict[str, dict[str, dict[str, Any]]] = {}
        self.indices = FakeIndices(self)

    async def bulk(
        self,
        *,
        operations: Sequence[Mapping[str, Any]] | None = None,
        body: Sequence[Mapping[str, Any]] | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        rows = list(operations if operations is not None else body or ())
        items: list[dict[str, Any]] = []
        i = 0
        while i < len(rows):
            meta = rows[i]
            action = next(iter(meta))
            spec = meta[action]
            assert isinstance(spec, Mapping)
            idx = str(spec["_index"])
            doc_id = str(spec["_id"])
            source = dict(rows[i + 1])
            self.docs.setdefault(idx, {})[doc_id] = source
            items.append({action: {"_id": doc_id, "result": "created", "status": 201}})
            i += 2
        return {"errors": False, "items": items}

    async def mget(
        self,
        *,
        index: str,
        ids: Sequence[str],
        source: bool | None = None,
    ) -> dict[str, Any]:
        del source
        table = self.docs.get(index, {})
        return {
            "docs": [{"_id": doc_id, "found": doc_id in table, "_index": index} for doc_id in ids]
        }

    async def count(self, *, index: str, **_kwargs: Any) -> dict[str, Any]:
        targets = self.indices.aliases.get(index)
        if targets:
            return {"count": sum(len(self.docs.get(name, {})) for name in targets)}
        return {"count": len(self.docs.get(index, {}))}


def _es(fake: FakeEs) -> AsyncElasticsearch:
    return cast(AsyncElasticsearch, fake)


def _vec_for(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    vals = [((digest[i % 32] / 127.5) - 1.0) for i in range(EMBED_DIM)]
    norm = sum(v * v for v in vals) ** 0.5 or 1.0
    return [v / norm for v in vals]


def _embedder(handler: httpx.MockTransport | None = None, *, retries: int = 0) -> EmbedClient:
    def default_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        texts = [str(t) for t in payload["texts"]]
        return httpx.Response(200, json={"vectors": [_vec_for(t) for t in texts]})

    transport = handler if handler is not None else httpx.MockTransport(default_handler)
    return EmbedClient(
        base_url="http://embedder.test",
        timeout_ms=2000,
        retries=retries,
        backoff_s=0.0,
        transport=transport,
    )


# ---------------------------------------------------------------------------
# Postgres (testcontainer) — scan helpers + job I/O
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


async def _seed_catalog(db: Database) -> None:
    titles = (
        _title(),
        _title(
            catalog_id="s2",
            title="Heat",
            synopsis="A cop hunts a thief across Los Angeles.",
            genres=(GenreId.CRIME, GenreId.THRILLER),
            moods=(MoodId.TENSE,),
            pop_28d=0.7,
        ),
        _title(
            catalog_id="s3",
            title="Spirited Away",
            synopsis="A girl works in a bathhouse for spirits.",
            media_type=MediaType.FILM,
            genres=(GenreId.ANIME,),
            moods=(MoodId.ADVENTUROUS,),
            pop_28d=0.8,
        ),
    )
    enrichments: dict[str, dict[str, object]] = {
        "s1": {
            "tags": ["slow-burn", "dystopia"],
            "audience": "adult",
            "pace": "fast",
            "era_feel": "90s sci-fi",
            "moods": ["tense", "dark"],
        },
        "s2": {
            "tags": ["heist"],
            "audience": "adult",
            "pace": "medium",
            "era_feel": "90s crime",
            "moods": ["tense"],
        },
        "s3": {
            "tags": ["spirits"],
            "audience": "family",
            "pace": "medium",
            "era_feel": "early 2000s fantasy",
            "moods": ["adventurous"],
        },
    }
    people = (
        _person(),
        _person(
            person_id="p2",
            name="Lana Wachowski",
            name_norm="lana wachowski",
            roles=(CreditRole.DIRECTOR,),
            credit_count=4,
        ),
        _person(
            person_id="p3",
            name="Al Pacino",
            name_norm="al pacino",
            credit_count=20,
            popularity=9.1,
        ),
    )
    credits = (
        CreditRecord(catalog_id="s1", person_id="p1", role=CreditRole.ACTOR),
        CreditRecord(catalog_id="s1", person_id="p2", role=CreditRole.DIRECTOR),
        CreditRecord(catalog_id="s2", person_id="p3", role=CreditRole.ACTOR),
    )
    async with db.session() as session:
        for title in titles:
            await db.titles(session).upsert(title, enrichment=enrichments[title.catalog_id])
        for person in people:
            await db.people(session).upsert(person)
        for credit in credits:
            await db.credits(session).upsert(credit)


async def test_scan_pages_titles_in_id_order(database: Database) -> None:
    await _seed_catalog(database)
    async with database.session() as session:
        repo = database.titles(session)
        assert await repo.count() == 3
        first = await repo.scan(limit=2)
        assert [row.title.catalog_id for row in first] == ["s1", "s2"]
        rest = await repo.scan(after_id=first[-1].title.catalog_id, limit=2)
        assert [row.title.catalog_id for row in rest] == ["s3"]
        assert rest[0].enrichment is not None
        assert rest[0].enrichment["era_feel"] == "early 2000s fantasy"


async def test_mark_indexed_sets_indexed_at(database: Database) -> None:
    await _seed_catalog(database)
    when = datetime(2026, 8, 27, 15, 0, tzinfo=UTC)
    async with database.session() as session:
        await database.titles(session).mark_indexed(["s1", "s2"], when)
    async with database.session() as session:
        one = await database.titles(session).get_stored("s1")
        two = await database.titles(session).get_stored("s2")
        three = await database.titles(session).get_stored("s3")
    assert one is not None and one.indexed_at == when
    assert two is not None and two.indexed_at == when
    assert three is not None and three.indexed_at is None


async def test_people_scan_and_credits_batch(database: Database) -> None:
    await _seed_catalog(database)
    async with database.session() as session:
        assert await database.people(session).count() == 3
        page = await database.people(session).scan(limit=2)
        assert [p.person_id for p in page] == ["p1", "p2"]
        many = await database.people(session).get_many(["p1", "p3"])
        assert {p.person_id for p in many} == {"p1", "p3"}
        credits = await database.credits(session).list_for_titles(["s1", "s2"])
        assert {c.catalog_id for c in credits} == {"s1", "s2"}
        assert await database.credits(session).list_for_titles([]) == []


# ---------------------------------------------------------------------------
# Job against FakeEs
# ---------------------------------------------------------------------------


async def test_index_job_counts_match_and_swaps(database: Database) -> None:
    await _seed_catalog(database)
    fake = FakeEs()
    async with _embedder() as embedder:
        result = await index_catalog(db=database, es=_es(fake), embedder=embedder, batch_size=2)

    assert result.swapped is True
    assert result.titles_index == "titles_v1"
    assert result.people_index == "people_v1"
    assert result.titles_indexed == 3
    assert result.people_indexed == 3
    assert result.titles_skipped == 0
    assert len(fake.docs["titles_v1"]) == 3
    assert len(fake.docs["people_v1"]) == 3
    assert await indices_for_alias(_es(fake), TITLES_ALIAS) == ("titles_v1",)
    assert await indices_for_alias(_es(fake), PEOPLE_ALIAS) == ("people_v1",)
    matrix = fake.docs["titles_v1"]["s1"]
    assert matrix["title"] == "The Matrix"
    assert "Keanu Reeves" in matrix["people_names"]
    assert matrix["era_feel"] == "90s sci-fi"
    assert len(matrix["embedding"]) == 384
    assert fake.indices.delete_calls == []
    assert fake.indices.update_alias_calls
    titles_actions = fake.indices.update_alias_calls[0]
    assert any("add" in action for action in titles_actions)


async def test_rerun_creates_new_version_and_keeps_previous(database: Database) -> None:
    await _seed_catalog(database)
    fake = FakeEs()
    es = _es(fake)
    async with _embedder() as embedder:
        first = await index_catalog(db=database, es=es, embedder=embedder)
        second = await index_catalog(db=database, es=es, embedder=embedder)

    assert first.titles_index == "titles_v1"
    assert second.titles_index == "titles_v2"
    assert second.people_index == "people_v2"
    assert second.swapped is True
    assert await indices_for_alias(es, TITLES_ALIAS) == ("titles_v2",)
    assert "titles_v1" in fake.indices.store
    assert "people_v1" in fake.indices.store
    assert "titles_v1" in fake.docs
    assert len(fake.docs["titles_v1"]) == 3
    assert second.titles_previous == ("titles_v1",)
    assert fake.indices.delete_calls == []
    # Each swap is one update_aliases call (remove+add together).
    assert all(n >= 1 for n in fake.indices.alias_cardinalities[TITLES_ALIAS])


async def test_resume_skips_already_indexed_docs(database: Database) -> None:
    await _seed_catalog(database)
    fake = FakeEs()
    es = _es(fake)
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        payload = json.loads(request.content)
        texts = [str(t) for t in payload["texts"]]
        if calls["n"] == 2:
            return httpx.Response(500, text="embedder down")
        return httpx.Response(200, json={"vectors": [_vec_for(t) for t in texts]})

    with pytest.raises(Exception, match="embedder"):
        async with _embedder(httpx.MockTransport(flaky), retries=0) as embedder:
            await index_catalog(db=database, es=es, embedder=embedder, batch_size=1)

    in_progress = await versioned_index_names(es, TITLES_ALIAS)
    assert in_progress == ("titles_v1",)
    assert await indices_for_alias(es, TITLES_ALIAS) == ()
    assert set(fake.docs["titles_v1"]) == {"s1"}

    async with _embedder() as embedder:
        result = await index_catalog(db=database, es=es, embedder=embedder, batch_size=1)

    assert result.titles_index == "titles_v1"
    assert result.titles_skipped == 1
    assert result.titles_indexed == 2
    assert result.swapped is True
    assert set(fake.docs["titles_v1"]) == {"s1", "s2", "s3"}
    assert await indices_for_alias(es, TITLES_ALIAS) == ("titles_v1",)


async def test_incomplete_run_does_not_swap(database: Database) -> None:
    await _seed_catalog(database)
    fake = FakeEs()

    # Limit titles to 1 but people still need all 3 — wait, limit only applies
    # to titles; people all 3. Both complete relative to expected. Use a
    # broken embedder so titles stay empty and swap is refused.
    def boom(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="nope")

    with pytest.raises(Exception, match="embedder"):
        async with _embedder(httpx.MockTransport(boom), retries=0) as embedder:
            await index_catalog(db=database, es=_es(fake), embedder=embedder)

    assert await indices_for_alias(_es(fake), TITLES_ALIAS) == ()
    assert "titles_v1" in fake.indices.store


async def test_resolve_target_reuses_unaliased_latest() -> None:
    fake = FakeEs()
    es = _es(fake)
    first = await create_next_index(es, TITLES_ALIAS)
    second = await resolve_target_index(es, TITLES_ALIAS, resume=True)
    assert first == second == "titles_v1"
    await swap_alias(es, TITLES_ALIAS, first)
    # Latest is live, so resume starts a new version rather than overwriting it.
    fresh = await resolve_target_index(es, TITLES_ALIAS, resume=True)
    assert fresh == "titles_v2"
    forced = await resolve_target_index(es, TITLES_ALIAS, resume=False)
    assert forced == "titles_v3"
    # Resume always continues the highest unaliased version (v3 here, not v2).
    resumed = await resolve_target_index(es, TITLES_ALIAS, resume=True)
    assert resumed == "titles_v3"


# ---------------------------------------------------------------------------
# Live Elasticsearch — BM25 + kNN on a known title
# ---------------------------------------------------------------------------


def _ping_http(url: str) -> bool:
    try:
        with urlopen(url, timeout=2) as resp:
            return 200 <= int(resp.status) < 300
    except (URLError, OSError, TimeoutError):
        return False


@pytest.fixture(scope="session")
def es_url() -> Iterator[str]:
    from assist.config import settings

    default = settings.elasticsearch_url
    if _ping_http(default):
        yield default
        return
    try:
        from testcontainers.community.elasticsearch import ElasticSearchContainer
    except ImportError as exc:
        pytest.skip(f"elasticsearch unreachable and testcontainers missing: {exc}")
    try:
        container = ElasticSearchContainer("elasticsearch:8.15.5", mem_limit="1G")
        container.with_env("discovery.type", "single-node")
        container.with_env("xpack.security.enabled", "false")
        container.with_env("ES_JAVA_OPTS", "-Xms256m -Xmx256m")
        container.start()
    except Exception as exc:
        pytest.skip(f"elasticsearch unreachable: {exc}")
    url = f"http://{container.get_container_host_ip()}:{container.get_exposed_port(container.port)}"
    try:
        yield url
    finally:
        container.stop()


@pytest.fixture
async def es_client(es_url: str) -> AsyncIterator[AsyncElasticsearch]:
    client = create_client(url=es_url)
    try:
        yield client
    finally:
        await close_client(client)


@pytest.fixture
def isolated_aliases() -> tuple[str, str]:
    suffix = uuid.uuid4().hex[:8]
    return f"t12t{suffix}", f"t12p{suffix}"


async def _cleanup(client: AsyncElasticsearch, *aliases: str) -> None:
    for alias in aliases:
        names = await versioned_index_names(client, alias)
        if names:
            await client.indices.delete(index=",".join(names))


async def test_live_known_title_retrievable_bm25_and_knn(
    database: Database,
    es_client: AsyncElasticsearch,
    isolated_aliases: tuple[str, str],
) -> None:
    titles_alias, people_alias = isolated_aliases
    await _seed_catalog(database)
    recorded: list[Sequence[Mapping[str, Any]]] = []
    original = es_client.indices.update_aliases

    async def _spy(*, actions: Sequence[Mapping[str, Any]] | None = None, **kwargs: Any) -> Any:
        if actions is not None:
            recorded.append(list(actions))
        return await original(actions=actions, **kwargs)

    es_client.indices.update_aliases = _spy  # type: ignore[method-assign]
    try:
        async with _embedder() as embedder:
            result = await index_catalog(
                db=database,
                es=es_client,
                embedder=embedder,
                titles_alias=titles_alias,
                people_alias=people_alias,
                titles_body=TITLES_INDEX_BODY,
                people_body=PEOPLE_INDEX_BODY,
                batch_size=2,
            )
        assert result.swapped is True
        assert result.titles_indexed == 3
        titles_count = int((await es_client.count(index=titles_alias))["count"])
        people_count = int((await es_client.count(index=people_alias))["count"])
        assert titles_count == 3
        assert people_count == 3

        bm25 = titles_bm25_body("The Matrix")
        hits = _es_body_hits(await es_client.search(index=titles_alias, body=bm25))
        assert hits[0] == "s1"

        embed_text = build_embedding_text(
            _title(),
            tags=["slow-burn", "dystopia"],
            people_names=["Keanu Reeves", "Lana Wachowski"],
            era_feel="90s sci-fi",
        )
        knn = titles_knn_body(_vec_for(embed_text), k=3)
        knn_hits = _es_body_hits(await es_client.search(index=titles_alias, body=knn))
        assert knn_hits[0] == "s1"

        # Re-run produces a new version; previous index stays for rollback.
        recorded.clear()
        async with _embedder() as embedder:
            again = await index_catalog(
                db=database,
                es=es_client,
                embedder=embedder,
                titles_alias=titles_alias,
                people_alias=people_alias,
                titles_body=TITLES_INDEX_BODY,
                people_body=PEOPLE_INDEX_BODY,
            )
        assert again.titles_index != result.titles_index
        assert again.swapped is True
        assert await es_client.indices.exists(index=result.titles_index)
        assert await indices_for_alias(es_client, titles_alias) == (again.titles_index,)
        assert len(recorded) >= 1
        first_swap = recorded[0]
        assert any("add" in action for action in first_swap)
    finally:
        es_client.indices.update_aliases = original  # type: ignore[method-assign]
        await _cleanup(es_client, titles_alias, people_alias)


def _es_body_hits(resp: object) -> list[str]:
    body = getattr(resp, "body", resp)
    assert isinstance(body, Mapping)
    hits = body["hits"]["hits"]
    assert isinstance(hits, list)
    ids: list[str] = []
    for hit in hits:
        assert isinstance(hit, Mapping)
        ids.append(str(hit["_id"]))
    return ids
