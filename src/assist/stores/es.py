"""Async Elasticsearch client, versioned-index bootstrap, and query builders.

The hybrid retrieve node (T17) issues these bodies; this module does not search.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from elasticsearch import AsyncElasticsearch, NotFoundError

from assist.config import settings
from assist.domain.constraints import ConstraintState
from assist.domain.enums import MediaType
from assist.obs.logging import get_logger
from assist.stores.mappings import EMBEDDING_DIMS, PEOPLE_INDEX_BODY, TITLES_INDEX_BODY

log = get_logger("assist.stores.es")

TITLES_ALIAS = "titles"
PEOPLE_ALIAS = "people"

# Boosts live on the query, not the mapping (plan §4.2 / §5.4).
TITLES_TEXT_FIELDS: tuple[str, ...] = (
    "title^3",
    "synopsis",
    "tags",
    "people_names^2",
    "era_feel",
)

TITLES_SOURCE_FIELDS: tuple[str, ...] = (
    "catalog_id",
    "title",
    "media_type",
    "release_year",
    "genres",
    "moods",
    "maturity_rank",
    "runtime_min",
    "origins",
    "local_original",
    "people_ids",
    "pop_28d",
    "audience",
    "pace",
)

# Mapping JSON is open-ended; ES accepts and returns untyped dicts.
type QueryBody = dict[str, Any]


@dataclass(frozen=True)
class BootstrapResult:
    titles_index: str
    people_index: str


def create_client(*, url: str | None = None) -> AsyncElasticsearch:
    """Factory from config. Caller owns close (API lifespan / job teardown)."""
    return AsyncElasticsearch(
        hosts=url or settings.elasticsearch_url,
        request_timeout=30.0,
        retry_on_timeout=True,
        max_retries=3,
    )


async def close_client(client: AsyncElasticsearch) -> None:
    await client.close()


def body_for_alias(alias: str) -> dict[str, Any]:
    if alias == PEOPLE_ALIAS:
        return PEOPLE_INDEX_BODY
    if alias == TITLES_ALIAS:
        return TITLES_INDEX_BODY
    msg = f"no mapping body for alias {alias!r}"
    raise KeyError(msg)


def next_versioned_name(alias: str, existing: Sequence[str]) -> str:
    pat = re.compile(rf"^{re.escape(alias)}_v(\d+)$")
    numbers = [int(m.group(1)) for name in existing if (m := pat.match(name))]
    return f"{alias}_v{max(numbers, default=0) + 1}"


def _as_mapping(resp: object) -> dict[str, Any]:
    # ObjectApiResponse is not a collections.abc.Mapping; Fake clients return dicts.
    body = getattr(resp, "body", resp)
    if isinstance(body, Mapping):
        return dict(body)
    msg = f"unexpected ES response type: {type(resp)!r}"
    raise TypeError(msg)


async def indices_for_alias(client: AsyncElasticsearch, alias: str) -> tuple[str, ...]:
    if not await client.indices.exists_alias(name=alias):
        return ()
    try:
        resp = await client.indices.get_alias(name=alias)
    except NotFoundError:
        return ()
    return tuple(sorted(_as_mapping(resp)))


async def versioned_index_names(client: AsyncElasticsearch, alias: str) -> tuple[str, ...]:
    resp = await client.indices.get(
        index=f"{alias}_v*",
        ignore_unavailable=True,
        allow_no_indices=True,
        features="aliases",
    )
    pat = re.compile(rf"^{re.escape(alias)}_v(\d+)$")
    found = [name for name in _as_mapping(resp) if pat.match(name)]
    return tuple(sorted(found, key=lambda n: int(n.rsplit("_v", 1)[1])))


async def create_next_index(
    client: AsyncElasticsearch,
    alias: str,
    *,
    body: Mapping[str, Any] | None = None,
) -> str:
    """Create `{alias}_vN` without attaching the alias. T12 indexes, then swaps."""
    spec = dict(body) if body is not None else body_for_alias(alias)
    existing = await versioned_index_names(client, alias)
    name = next_versioned_name(alias, existing)
    await client.indices.create(
        index=name,
        settings=spec["settings"],
        mappings=spec["mappings"],
    )
    log.info("es.index.created", index=name, alias=alias)
    return name


def alias_swap_actions(
    alias: str,
    new_index: str,
    previous: Sequence[str],
) -> list[QueryBody]:
    """One `_aliases` actions list. If it contains a remove, it also contains an add.

    ES applies the list atomically; splitting remove/add across requests would
    let the alias resolve to zero indices between calls.
    """
    actions: list[QueryBody] = []
    for old in previous:
        if old != new_index:
            actions.append({"remove": {"index": old, "alias": alias}})
    if new_index not in previous:
        actions.append({"add": {"index": new_index, "alias": alias}})
    return actions


async def swap_alias(
    client: AsyncElasticsearch,
    alias: str,
    new_index: str,
) -> tuple[str, ...]:
    """Point `alias` at `new_index` in one `_aliases` request.

    Remove and add travel together so the alias never has zero targets.
    Returns the indices the alias previously pointed at (rollback inputs).
    """
    previous = await indices_for_alias(client, alias)
    actions = alias_swap_actions(alias, new_index, previous)
    if not actions:
        return previous
    await client.indices.update_aliases(actions=actions)
    log.info("es.alias.swapped", alias=alias, index=new_index, previous=list(previous))
    return previous


async def rollback_alias(
    client: AsyncElasticsearch,
    alias: str,
    previous_index: str,
) -> tuple[str, ...]:
    """Re-point the alias at `previous_index`. Does not delete the failed index."""
    return await swap_alias(client, alias, previous_index)


async def bootstrap(
    client: AsyncElasticsearch,
    *,
    titles_alias: str = TITLES_ALIAS,
    people_alias: str = PEOPLE_ALIAS,
    titles_body: Mapping[str, Any] | None = None,
    people_body: Mapping[str, Any] | None = None,
) -> BootstrapResult:
    """Create titles/people v1 and point aliases. Safe to run twice."""
    titles = await _ensure_live(
        client, titles_alias, titles_body if titles_body is not None else TITLES_INDEX_BODY
    )
    people = await _ensure_live(
        client, people_alias, people_body if people_body is not None else PEOPLE_INDEX_BODY
    )
    log.info("es.bootstrap.ok", titles_index=titles, people_index=people)
    return BootstrapResult(titles_index=titles, people_index=people)


async def _ensure_live(
    client: AsyncElasticsearch,
    alias: str,
    body: Mapping[str, Any],
) -> str:
    current = await indices_for_alias(client, alias)
    if current:
        return current[0]
    existing = await versioned_index_names(client, alias)
    if existing:
        latest = existing[-1]
        await swap_alias(client, alias, latest)
        return latest
    # First create attaches the alias in the same request so a crash between
    # create and alias-add cannot leave a nameless v1 as the only copy.
    name = next_versioned_name(alias, ())
    spec = dict(body)
    await client.indices.create(
        index=name,
        settings=spec["settings"],
        mappings=spec["mappings"],
        aliases={alias: {}},
    )
    log.info("es.index.created", index=name, alias=alias)
    return name


def constraint_filters(
    *,
    media_type: str | None = None,
    genres_include: Sequence[str] = (),
    genres_exclude: Sequence[str] = (),
    moods: Sequence[str] = (),
    origins: Sequence[str] = (),
    local_original: bool | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    runtime_max: int | None = None,
    maturity_rank_max: int | None = None,
    audience: str | None = None,
    pace: str | None = None,
    people_include: Sequence[str] = (),
    people_exclude: Sequence[str] = (),
) -> list[QueryBody]:
    """Non-scoring `filter` clauses. Never `must` — scoring stays BM25/kNN-only."""
    clauses: list[QueryBody] = []
    if media_type and media_type != MediaType.ANY:
        clauses.append({"term": {"media_type": media_type}})
    for genre in genres_include:
        clauses.append({"term": {"genres": genre}})
    if genres_exclude:
        clauses.append({"bool": {"must_not": [{"terms": {"genres": list(genres_exclude)}}]}})
    for mood in moods:
        clauses.append({"term": {"moods": mood}})
    if origins:
        clauses.append({"terms": {"origins": list(origins)}})
    if local_original:
        clauses.append({"term": {"local_original": True}})
    year_range: dict[str, int] = {}
    if year_min is not None:
        year_range["gte"] = year_min
    if year_max is not None:
        year_range["lte"] = year_max
    if year_range:
        clauses.append({"range": {"release_year": year_range}})
    if runtime_max is not None:
        clauses.append({"range": {"runtime_min": {"lte": runtime_max}}})
    if maturity_rank_max is not None:
        clauses.append({"range": {"maturity_rank": {"lte": maturity_rank_max}}})
    if audience:
        clauses.append({"term": {"audience": audience}})
    if pace:
        clauses.append({"term": {"pace": pace}})
    for person_id in people_include:
        clauses.append({"term": {"people_ids": person_id}})
    if people_exclude:
        clauses.append({"bool": {"must_not": [{"terms": {"people_ids": list(people_exclude)}}]}})
    return clauses


def filters_from_constraints(
    state: ConstraintState,
    *,
    maturity_rank_max: int | None = None,
) -> list[QueryBody]:
    """Map a merged ConstraintState onto filter clauses.

    Effective maturity is `min(profile, requested)` — computed by T16, not here.
    Pass the ceiling in as `maturity_rank_max`.
    """
    media_type = None if state.media_type is None else str(state.media_type)
    return constraint_filters(
        media_type=media_type,
        genres_include=tuple(g.value for g in state.genres_include),
        genres_exclude=tuple(g.value for g in state.genres_exclude),
        moods=tuple(m.value for m in state.moods),
        origins=state.origins,
        local_original=True if state.local_originals_only else None,
        year_min=state.year_min,
        year_max=state.year_max,
        runtime_max=state.duration_max_min,
        maturity_rank_max=maturity_rank_max,
        people_include=state.people_include,
        people_exclude=state.people_exclude,
    )


def titles_bm25_body(
    query: str,
    filters: Sequence[Mapping[str, Any]] = (),
    *,
    size: int = 50,
) -> QueryBody:
    scoring: QueryBody
    if query.strip():
        scoring = {
            "multi_match": {
                "query": query,
                "fields": list(TITLES_TEXT_FIELDS),
                "type": "best_fields",
            }
        }
    else:
        scoring = {"match_all": {}}
    bool_query: QueryBody = {"must": [scoring]}
    if filters:
        bool_query["filter"] = list(filters)
    return {
        "size": size,
        "query": {"bool": bool_query},
        "_source": list(TITLES_SOURCE_FIELDS),
    }


def titles_knn_body(
    vector: Sequence[float],
    filters: Sequence[Mapping[str, Any]] = (),
    *,
    k: int = 50,
    num_candidates: int | None = None,
) -> QueryBody:
    if len(vector) != EMBEDDING_DIMS:
        msg = f"embedding dim {len(vector)} != {EMBEDDING_DIMS}"
        raise ValueError(msg)
    knn: QueryBody = {
        "field": "embedding",
        "query_vector": list(vector),
        "k": k,
        "num_candidates": num_candidates if num_candidates is not None else max(k * 4, 100),
    }
    if filters:
        knn["filter"] = filters[0] if len(filters) == 1 else {"bool": {"filter": list(filters)}}
    return {"size": k, "knn": knn, "_source": list(TITLES_SOURCE_FIELDS)}


def people_name_body(
    name: str,
    *,
    size: int = 10,
    roles: Sequence[str] = (),
    active_year_min: int | None = None,
    active_year_max: int | None = None,
) -> QueryBody:
    scoring: QueryBody
    if name.strip():
        scoring = {"match": {"name": {"query": name, "operator": "and"}}}
    else:
        scoring = {"match_all": {}}
    filters: list[QueryBody] = []
    if roles:
        filters.append({"terms": {"roles": list(roles)}})
    if active_year_min is not None:
        filters.append({"range": {"active_year_max": {"gte": active_year_min}}})
    if active_year_max is not None:
        filters.append({"range": {"active_year_min": {"lte": active_year_max}}})
    bool_query: QueryBody = {"must": [scoring]}
    if filters:
        bool_query["filter"] = filters
    return {
        "size": size,
        "query": {"bool": bool_query},
        "sort": [{"popularity": {"order": "desc"}}],
    }
