"""Elasticsearch mappings, bootstrap, atomic alias swap, and query builders."""

from __future__ import annotations

import fnmatch
import json
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any, cast
from urllib.error import URLError
from urllib.request import urlopen

import pytest
from elasticsearch import AsyncElasticsearch

from assist.config import settings
from assist.domain.constraints import ConstraintState
from assist.domain.enums import GenreId, MediaType, MoodId
from assist.stores.es import (
    PEOPLE_ALIAS,
    TITLES_ALIAS,
    alias_swap_actions,
    bootstrap,
    close_client,
    constraint_filters,
    create_client,
    create_next_index,
    filters_from_constraints,
    indices_for_alias,
    next_versioned_name,
    people_name_body,
    rollback_alias,
    swap_alias,
    titles_bm25_body,
    titles_knn_body,
    versioned_index_names,
)
from assist.stores.mappings import EMBEDDING_DIMS, PEOPLE_INDEX_BODY, TITLES_INDEX_BODY
from assist.stores.mappings.people import PEOPLE_INDEX_BODY as PEOPLE_BODY
from assist.stores.mappings.synonyms import CATALOG_SYNONYMS
from assist.stores.mappings.titles import TITLES_INDEX_BODY as TITLES_BODY

# ---------------------------------------------------------------------------
# Mapping contract (no I/O)
# ---------------------------------------------------------------------------


def test_titles_mapping_has_english_synonym_analyzer() -> None:
    analysis = TITLES_BODY["settings"]["analysis"]
    syn = analysis["filter"]["catalog_synonyms"]
    assert syn["type"] == "synonym_graph"
    assert "sci-fi, scifi, sci fi, science fiction" in syn["synonyms"]
    assert "catalog_synonyms" in analysis["analyzer"]["english_search"]["filter"]
    assert "catalog_synonyms" not in analysis["analyzer"]["english_index"]["filter"]
    assert CATALOG_SYNONYMS


def test_titles_mapping_dense_vector_384_cosine_hnsw() -> None:
    emb = TITLES_BODY["mappings"]["properties"]["embedding"]
    assert emb["type"] == "dense_vector"
    assert emb["dims"] == EMBEDDING_DIMS == 384
    assert emb["similarity"] == "cosine"
    assert emb["index"] is True
    assert emb["index_options"]["type"] == "hnsw"


def test_titles_mapping_filterable_keywords() -> None:
    props = TITLES_BODY["mappings"]["properties"]
    for field in (
        "media_type",
        "genres",
        "moods",
        "origins",
        "audience",
        "pace",
        "people_ids",
        "catalog_id",
    ):
        assert props[field]["type"] == "keyword"
    assert props["title"]["analyzer"] == "english_index"
    assert props["title"]["search_analyzer"] == "english_search"
    assert "keyword" in props["title"]["fields"]


def test_people_mapping_edge_ngram_on_name() -> None:
    analysis = PEOPLE_BODY["settings"]["analysis"]
    assert analysis["filter"]["name_edge_ngram"]["type"] == "edge_ngram"
    assert analysis["filter"]["name_edge_ngram"]["min_gram"] == 2
    name = PEOPLE_BODY["mappings"]["properties"]["name"]
    assert name["analyzer"] == "name_edge"
    assert name["search_analyzer"] == "name_search"
    assert "name_edge_ngram" in analysis["analyzer"]["name_edge"]["filter"]
    assert "name_edge_ngram" not in analysis["analyzer"]["name_search"]["filter"]


def test_next_versioned_name_increments() -> None:
    assert next_versioned_name("titles", ()) == "titles_v1"
    assert next_versioned_name("titles", ("titles_v1",)) == "titles_v2"
    assert next_versioned_name("titles", ("titles_v1", "titles_v3")) == "titles_v4"


# ---------------------------------------------------------------------------
# Atomic alias actions (pure)
# ---------------------------------------------------------------------------


def test_alias_swap_actions_are_single_list_with_remove_and_add() -> None:
    actions = alias_swap_actions("titles", "titles_v2", ("titles_v1",))
    removes = [a for a in actions if "remove" in a]
    adds = [a for a in actions if "add" in a]
    assert len(removes) == 1
    assert len(adds) == 1
    assert removes[0]["remove"] == {"index": "titles_v1", "alias": "titles"}
    assert adds[0]["add"] == {"index": "titles_v2", "alias": "titles"}


def test_alias_swap_actions_never_remove_without_add() -> None:
    previous = ("titles_v1", "titles_v2")
    actions = alias_swap_actions("titles", "titles_v3", previous)
    has_remove = any("remove" in a for a in actions)
    has_add = any("add" in a for a in actions)
    assert has_remove
    assert has_add
    assert len(actions) == 3  # two removes + one add, one request


def test_alias_swap_actions_first_point_is_add_only() -> None:
    actions = alias_swap_actions("titles", "titles_v1", ())
    assert actions == [{"add": {"index": "titles_v1", "alias": "titles"}}]


def test_alias_swap_actions_noop_when_already_pointed() -> None:
    assert alias_swap_actions("titles", "titles_v1", ("titles_v1",)) == []


# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------


def test_bm25_constraints_live_in_filter_not_must() -> None:
    filters = constraint_filters(genres_include=("thriller",), media_type="film")
    body = titles_bm25_body("spy", filters)
    bool_q = body["query"]["bool"]
    dumped_filters = json.dumps(bool_q["filter"])
    assert "multi_match" not in dumped_filters
    assert bool_q["must"][0]["multi_match"]["query"] == "spy"
    assert "title^3" in bool_q["must"][0]["multi_match"]["fields"]
    assert "filter" in bool_q


def test_knn_filters_attached_to_knn_clause() -> None:
    vector = [0.0] * EMBEDDING_DIMS
    filters = constraint_filters(moods=("cozy",))
    body = titles_knn_body(vector, filters, k=20)
    assert body["knn"]["field"] == "embedding"
    assert body["knn"]["k"] == 20
    assert len(body["knn"]["query_vector"]) == 384
    assert "filter" in body["knn"]


def test_knn_rejects_wrong_dim() -> None:
    with pytest.raises(ValueError, match="embedding dim"):
        titles_knn_body([0.0, 1.0], ())


def test_filters_from_constraints_skips_any_media_and_applies_and_genres() -> None:
    state = ConstraintState(
        media_type=MediaType.ANY,
        genres_include=(GenreId.THRILLER, GenreId.COMEDY),
        genres_exclude=(GenreId.HORROR,),
        moods=(MoodId.TENSE,),
        local_originals_only=True,
        year_min=1990,
        year_max=1999,
        duration_max_min=120,
        people_include=("p1",),
        people_exclude=("p2",),
    )
    clauses = filters_from_constraints(state, maturity_rank_max=5)
    dumped = json.dumps(clauses)
    assert "media_type" not in dumped
    assert clauses.count({"term": {"genres": "thriller"}}) == 1
    assert clauses.count({"term": {"genres": "comedy"}}) == 1
    assert {"term": {"local_original": True}} in clauses
    assert {"range": {"maturity_rank": {"lte": 5}}} in clauses
    assert {"range": {"release_year": {"gte": 1990, "lte": 1999}}} in clauses
    assert {"range": {"runtime_min": {"lte": 120}}} in clauses
    assert any("must_not" in c.get("bool", {}) for c in clauses)


def test_people_name_body_uses_filter_context_for_role_and_era() -> None:
    body = people_name_body(
        "nolan",
        roles=("director",),
        active_year_min=1990,
        active_year_max=1999,
    )
    bool_q = body["query"]["bool"]
    assert bool_q["must"][0]["match"]["name"]["query"] == "nolan"
    dumped = json.dumps(bool_q["filter"])
    assert "director" in dumped
    assert "must" not in dumped or dumped.index("filter") >= 0


# ---------------------------------------------------------------------------
# In-memory fake — bootstrap + atomic swap without a node
# ---------------------------------------------------------------------------


class FakeIndices:
    def __init__(self) -> None:
        self.store: dict[str, dict[str, Any]] = {}
        self.aliases: dict[str, set[str]] = {}
        self.update_alias_calls: list[list[Mapping[str, Any]]] = []
        # Cardinality of each alias after every update_aliases (atomic apply).
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
        # Apply the whole list, then publish. Same atomicity ES guarantees.
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


class FakeEs:
    def __init__(self) -> None:
        self.indices = FakeIndices()


def _es(fake: FakeEs) -> AsyncElasticsearch:
    # In-memory stand-in; IndicesClient is not a Protocol match for FakeIndices.
    return cast(AsyncElasticsearch, fake)


@pytest.fixture
def fake_es() -> FakeEs:
    return FakeEs()


async def test_bootstrap_creates_v1_and_points_aliases(fake_es: FakeEs) -> None:
    result = await bootstrap(_es(fake_es))
    assert result.titles_index == "titles_v1"
    assert result.people_index == "people_v1"
    assert await indices_for_alias(_es(fake_es), TITLES_ALIAS) == ("titles_v1",)
    assert await indices_for_alias(_es(fake_es), PEOPLE_ALIAS) == ("people_v1",)
    assert "titles_v1" in fake_es.indices.store
    assert "people_v1" in fake_es.indices.store


async def test_bootstrap_twice_is_idempotent(fake_es: FakeEs) -> None:
    first = await bootstrap(_es(fake_es))
    second = await bootstrap(_es(fake_es))
    assert first == second
    names = await versioned_index_names(_es(fake_es), TITLES_ALIAS)
    assert names == ("titles_v1",)


async def test_swap_is_one_update_aliases_call_and_never_empty(fake_es: FakeEs) -> None:
    await bootstrap(_es(fake_es))
    v2 = await create_next_index(_es(fake_es), TITLES_ALIAS, body=TITLES_INDEX_BODY)
    assert v2 == "titles_v2"
    fake_es.indices.update_alias_calls.clear()
    previous = await swap_alias(_es(fake_es), TITLES_ALIAS, v2)
    assert previous == ("titles_v1",)
    assert await indices_for_alias(_es(fake_es), TITLES_ALIAS) == ("titles_v2",)
    assert "titles_v1" in fake_es.indices.store
    assert len(fake_es.indices.update_alias_calls) == 1
    actions = fake_es.indices.update_alias_calls[0]
    assert any("remove" in a for a in actions)
    assert any("add" in a for a in actions)
    assert all(n >= 1 for n in fake_es.indices.alias_cardinalities[TITLES_ALIAS])


async def test_rollback_repoints_and_keeps_new_index(fake_es: FakeEs) -> None:
    await bootstrap(_es(fake_es))
    v2 = await create_next_index(_es(fake_es), TITLES_ALIAS, body=TITLES_INDEX_BODY)
    await swap_alias(_es(fake_es), TITLES_ALIAS, v2)
    await rollback_alias(_es(fake_es), TITLES_ALIAS, "titles_v1")
    assert await indices_for_alias(_es(fake_es), TITLES_ALIAS) == ("titles_v1",)
    assert "titles_v2" in fake_es.indices.store


# ---------------------------------------------------------------------------
# Live Elasticsearch
# ---------------------------------------------------------------------------


def _ping_http(url: str) -> bool:
    try:
        with urlopen(url, timeout=2) as resp:
            return 200 <= int(resp.status) < 300
    except (URLError, OSError, TimeoutError):
        return False


@pytest.fixture(scope="session")
def es_url() -> Iterator[str]:
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
    return f"t06t{suffix}", f"t06p{suffix}"


async def _cleanup(client: AsyncElasticsearch, *aliases: str) -> None:
    # ES 8 refuses wildcard deletes (`action.destructive_requires_name`).
    for alias in aliases:
        names = await versioned_index_names(client, alias)
        if names:
            await client.indices.delete(index=",".join(names))


async def test_live_bootstrap_creates_indices_and_aliases(
    es_client: AsyncElasticsearch,
    isolated_aliases: tuple[str, str],
) -> None:
    titles_alias, people_alias = isolated_aliases
    try:
        result = await bootstrap(
            es_client,
            titles_alias=titles_alias,
            people_alias=people_alias,
            titles_body=TITLES_INDEX_BODY,
            people_body=PEOPLE_INDEX_BODY,
        )
        assert result.titles_index == f"{titles_alias}_v1"
        assert result.people_index == f"{people_alias}_v1"
        assert await es_client.indices.exists(index=result.titles_index)
        assert await es_client.indices.exists(index=result.people_index)
        assert await indices_for_alias(es_client, titles_alias) == (result.titles_index,)
        assert await indices_for_alias(es_client, people_alias) == (result.people_index,)

        mapping = dict(await es_client.indices.get(index=result.titles_index))
        props = mapping[result.titles_index]["mappings"]["properties"]
        assert props["embedding"]["dims"] == 384
        assert props["embedding"]["similarity"] == "cosine"

        people = dict(await es_client.indices.get(index=result.people_index))
        name = people[result.people_index]["mappings"]["properties"]["name"]
        assert name["analyzer"] == "name_edge"
    finally:
        await _cleanup(es_client, titles_alias, people_alias)


async def test_live_bootstrap_twice_is_idempotent(
    es_client: AsyncElasticsearch,
    isolated_aliases: tuple[str, str],
) -> None:
    titles_alias, people_alias = isolated_aliases
    try:
        first = await bootstrap(
            es_client,
            titles_alias=titles_alias,
            people_alias=people_alias,
            titles_body=TITLES_INDEX_BODY,
            people_body=PEOPLE_INDEX_BODY,
        )
        second = await bootstrap(
            es_client,
            titles_alias=titles_alias,
            people_alias=people_alias,
            titles_body=TITLES_INDEX_BODY,
            people_body=PEOPLE_INDEX_BODY,
        )
        assert first == second
        names = await versioned_index_names(es_client, titles_alias)
        assert names == (first.titles_index,)
    finally:
        await _cleanup(es_client, titles_alias, people_alias)


async def test_live_alias_swap_is_one_atomic_request(
    es_client: AsyncElasticsearch,
    isolated_aliases: tuple[str, str],
) -> None:
    titles_alias, people_alias = isolated_aliases
    recorded: list[Sequence[Mapping[str, Any]]] = []
    original = es_client.indices.update_aliases

    async def _spy(*, actions: Sequence[Mapping[str, Any]] | None = None, **kwargs: Any) -> Any:
        if actions is not None:
            recorded.append(list(actions))
        return await original(actions=actions, **kwargs)

    es_client.indices.update_aliases = _spy  # type: ignore[method-assign]
    try:
        await bootstrap(
            es_client,
            titles_alias=titles_alias,
            people_alias=people_alias,
            titles_body=TITLES_INDEX_BODY,
            people_body=PEOPLE_INDEX_BODY,
        )
        v2 = await create_next_index(es_client, titles_alias, body=TITLES_INDEX_BODY)
        recorded.clear()
        previous = await swap_alias(es_client, titles_alias, v2)
        assert previous == (f"{titles_alias}_v1",)
        assert await indices_for_alias(es_client, titles_alias) == (v2,)
        assert await es_client.indices.exists(index=previous[0])
        assert len(recorded) == 1
        actions = recorded[0]
        assert any("remove" in a for a in actions)
        assert any("add" in a for a in actions)
        # After the one request, the alias still resolves (never zero).
        assert len(await indices_for_alias(es_client, titles_alias)) == 1

        await rollback_alias(es_client, titles_alias, previous[0])
        assert await indices_for_alias(es_client, titles_alias) == previous
        assert await es_client.indices.exists(index=v2)
    finally:
        es_client.indices.update_aliases = original  # type: ignore[method-assign]
        await _cleanup(es_client, titles_alias, people_alias)
