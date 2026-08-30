"""Retrieval node: hybrid BM25+kNN, Python RRF, franchise cap, bounded broaden."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, cast
from urllib.error import URLError
from urllib.request import urlopen

import pytest
from elasticsearch import AsyncElasticsearch

from assist.config import settings
from assist.domain.catalog import Candidate
from assist.domain.constraints import ConstraintState
from assist.domain.context import ServerUserCtx
from assist.domain.enums import (
    DegradedReason,
    DeviceClass,
    GenreId,
    MaturityRating,
    MediaType,
    MoodId,
    Package,
    RecencyBias,
    maturity_rank,
)
from assist.graph.build import build_graph
from assist.graph.state import TurnState, empty_turn_state
from assist.nodes.retrieval import (
    apply_broaden_ladder,
    broaden_constraints,
    ceiling_maturity_rank,
    diversify,
    franchise_key,
    retrieve,
    rrf_fuse,
    titles_people_join_body,
)
from assist.stores.es import (
    TITLES_ALIAS,
    bootstrap,
    close_client,
    constraint_filters,
    create_client,
    filters_from_constraints,
    titles_bm25_body,
    versioned_index_names,
)
from assist.stores.mappings import EMBEDDING_DIMS, PEOPLE_INDEX_BODY, TITLES_INDEX_BODY

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _ctx(
    *,
    maturity_max: MaturityRating = MaturityRating.PG_13,
    kids_flag: bool = False,
) -> ServerUserCtx:
    return ServerUserCtx(
        user_id="u1",
        profile_id="p1",
        geo="US",
        package=Package.BASIC,
        maturity_max=maturity_max,
        kids_flag=kids_flag,
        device_class=DeviceClass.WEB,
    )


def _state(
    ctx: ServerUserCtx | None = None,
    **overrides: object,
) -> TurnState:
    return empty_turn_state(ctx or _ctx(), **overrides)


def _hit(
    catalog_id: str,
    title: str,
    *,
    media_type: str = "film",
    year: int = 2010,
    genres: list[str] | None = None,
    maturity: int = 4,
    score: float = 1.0,
) -> dict[str, Any]:
    return {
        "_id": catalog_id,
        "_score": score,
        "_source": {
            "catalog_id": catalog_id,
            "title": title,
            "media_type": media_type,
            "release_year": year,
            "genres": genres or ["drama"],
            "maturity_rank": maturity,
        },
    }


class FakeEmbedder:
    def __init__(self, *, fail: bool = False, dim: int = EMBEDDING_DIMS) -> None:
        self.calls: list[list[str]] = []
        self.fail = fail
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("embedder down")
        return [[0.01] * self.dim for _ in texts]


class FakeEs:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.bm25_hits: list[dict[str, Any]] = []
        self.knn_hits: list[dict[str, Any]] = []
        self.people_hits: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self.mood_gated: bool = False

    async def search(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        raw_body = kwargs.get("body")
        body: dict[str, Any] = dict(raw_body) if isinstance(raw_body, Mapping) else {}
        if self.mood_gated:
            blob = json.dumps({"query": body.get("query"), "knn": body.get("knn")})
            if '"moods"' in blob:
                return {"hits": {"hits": []}}
        if "knn" in body:
            hits = self.knn_hits
        elif "sort" in body:
            hits = self.people_hits
        else:
            hits = self.bm25_hits
        return {"hits": {"hits": hits}}


def _bm25_bodies(fake: FakeEs) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for call in fake.calls:
        body = call.get("body")
        if isinstance(body, Mapping) and "query" in body and "knn" not in body:
            if "sort" in body:
                continue
            out.append(dict(body))
    return out


def test_rrf_matches_hand_computed_fusion() -> None:
    # Two lists, k=60. Ranks are 1-based. score = Σ 1/(k + rank).
    k = 60
    ranking_a = ("a", "b", "c")
    ranking_b = ("b", "d", "a")
    expected = {
        "a": 1 / (k + 1) + 1 / (k + 3),
        "b": 1 / (k + 2) + 1 / (k + 1),
        "c": 1 / (k + 3),
        "d": 1 / (k + 2),
    }
    fused = rrf_fuse(ranking_a, ranking_b, k=k)
    got = dict(fused)
    assert set(got) == set(expected)
    for doc_id, score in expected.items():
        assert got[doc_id] == pytest.approx(score)
    # b = 1/61 + 1/62 > a = 1/61 + 1/63 > d = 1/62 > c = 1/63
    assert [doc_id for doc_id, _ in fused] == ["b", "a", "d", "c"]


def test_rrf_duplicate_in_one_list_uses_first_rank() -> None:
    fused = rrf_fuse(("a", "a", "b"), k=60)
    assert dict(fused)["a"] == pytest.approx(1 / 61)
    assert dict(fused)["b"] == pytest.approx(1 / 63)


def test_franchise_key_collapses_colon_sequels() -> None:
    assert franchise_key("Narcos") == franchise_key("Narcos: Mexico")
    assert franchise_key("The Witcher") == "witcher"
    assert franchise_key("  ") == ""


def test_diversify_franchise_cap_holds() -> None:
    from assist.nodes.retrieval import RetrievedHit

    hits = [
        RetrievedHit("s1", "Narcos", MediaType.FILM, 2015, (GenreId.CRIME,), 6),
        RetrievedHit("s2", "Narcos: Mexico", MediaType.FILM, 2018, (GenreId.CRIME,), 6),
        RetrievedHit("s3", "Dark", MediaType.SERIES, 2017, (GenreId.MYSTERY,), 6),
        RetrievedHit("s4", "Narcos: Colombia", MediaType.FILM, 2019, (GenreId.CRIME,), 6),
        RetrievedHit("s5", "Chef's Table", MediaType.SERIES, 2015, (GenreId.DOCUMENTARY,), 5),
    ]
    kept = diversify(hits, cap=1, size=25)
    titles = [hit.title for hit in kept]
    assert titles == ["Narcos", "Dark", "Chef's Table"]
    keys = [franchise_key(hit.title) for hit in kept]
    assert len(keys) == len(set(keys))


def test_broaden_drops_flavor_keeps_maturity_and_exclusions() -> None:
    prior = ConstraintState(
        media_type=MediaType.FILM,
        genres_include=(GenreId.THRILLER,),
        genres_exclude=(GenreId.HORROR,),
        moods=(MoodId.TENSE,),
        year_min=1990,
        year_max=1999,
        duration_max_min=120,
        local_originals_only=True,
        people_include=("p1",),
        people_exclude=("p2",),
        maturity_request_stricter=MaturityRating.PG,
        recency_bias=RecencyBias.TONIGHT,
    )
    out = apply_broaden_ladder(prior)
    assert out.moods == ()
    assert out.year_min is None and out.year_max is None
    assert out.duration_max_min is None
    assert out.local_originals_only is False
    assert out.genres_exclude == ()
    assert out.recency_bias is None
    assert out.media_type is MediaType.FILM
    assert out.genres_include == (GenreId.THRILLER,)
    assert out.people_include == ("p1",)
    assert out.people_exclude == ("p2",)
    assert out.maturity_request_stricter is MaturityRating.PG


def test_broaden_second_rung_drops_people_then_genre() -> None:
    only_people = ConstraintState(people_include=("p1",), people_exclude=("p2",))
    dropped_people = apply_broaden_ladder(only_people)
    assert dropped_people.people_include == ()
    assert dropped_people.people_exclude == ("p2",)

    only_genre = ConstraintState(genres_include=(GenreId.COMEDY,))
    dropped_genre = apply_broaden_ladder(only_genre)
    assert dropped_genre.genres_include == ()


def test_ceiling_uses_stricter_request_not_a_raise() -> None:
    ctx = _ctx(maturity_max=MaturityRating.R)
    stricter = ConstraintState(maturity_request_stricter=MaturityRating.PG)
    assert ceiling_maturity_rank(ctx, stricter) == maturity_rank(MaturityRating.PG)
    raise_attempt = ConstraintState(maturity_request_stricter=MaturityRating.NC_17)
    assert ceiling_maturity_rank(ctx, raise_attempt) == maturity_rank(MaturityRating.R)


def test_bm25_builder_puts_constraints_in_filter_not_must() -> None:
    filters = constraint_filters(genres_include=("thriller",), maturity_rank_max=4)
    body = titles_bm25_body("spy thriller", filters)
    bool_q = body["query"]["bool"]
    dumped_filter = json.dumps(bool_q["filter"])
    dumped_must = json.dumps(bool_q["must"])
    assert "multi_match" in dumped_must
    assert "multi_match" not in dumped_filter
    assert '{"term": {"genres": "thriller"}}' in dumped_filter
    assert '{"term": {"genres": "thriller"}}' not in dumped_must
    assert "maturity_rank" in dumped_filter
    assert "maturity_rank" not in dumped_must


def test_people_join_body_is_filter_only() -> None:
    filters = constraint_filters(people_include=("p-leo",), maturity_rank_max=5)
    body = titles_people_join_body(filters, size=20)
    dumped = json.dumps(body)
    assert "must" not in dumped
    assert "multi_match" not in dumped
    assert body["query"]["bool"]["filter"]
    assert body["sort"][0] == {"pop_28d": {"order": "desc"}}


# ---------------------------------------------------------------------------
# Node with fakes
# ---------------------------------------------------------------------------


async def test_retrieve_issues_bm25_and_knn_through_titles_alias() -> None:
    fake = FakeEs()
    fake.bm25_hits = [_hit("s1", "Spy Game")]
    fake.knn_hits = [_hit("s2", "Tinker Tailor")]
    embedder = FakeEmbedder()
    out = await retrieve(
        _state(query_rewrite="spy", text="ignored"),
        es=cast(Any, fake),
        embedder=embedder,
    )
    assert "retrieve_attempts" not in out
    assert "retrieve_max_attempts" not in out
    indexes = {call.get("index") for call in fake.calls}
    assert indexes == {TITLES_ALIAS}
    kinds = []
    for call in fake.calls:
        body = call["body"]
        if "knn" in body:
            kinds.append("knn")
        else:
            kinds.append("bm25")
    assert "bm25" in kinds
    assert "knn" in kinds
    assert embedder.calls == [["spy"]]
    ids = {c.catalog_id for c in cast(tuple[Candidate, ...], out["candidates"])}
    assert ids == {"s1", "s2"}


async def test_retrieve_puts_live_filters_in_bm25_filter_context() -> None:
    fake = FakeEs()
    fake.bm25_hits = [_hit("s1", "Heat")]
    constraints = ConstraintState(
        media_type=MediaType.FILM,
        genres_include=(GenreId.THRILLER,),
        moods=(MoodId.TENSE,),
    )
    await retrieve(
        _state(
            _ctx(maturity_max=MaturityRating.PG),
            query_rewrite="crime",
            constraints=constraints,
        ),
        es=cast(Any, fake),
        embedder=FakeEmbedder(),
    )
    bodies = _bm25_bodies(fake)
    assert len(bodies) == 1
    bool_q = bodies[0]["query"]["bool"]
    dumped_filter = json.dumps(bool_q["filter"])
    dumped_must = json.dumps(bool_q["must"])
    assert "multi_match" in dumped_must
    assert "multi_match" not in dumped_filter
    assert "thriller" in dumped_filter
    assert "tense" in dumped_filter
    assert '"lte": 4' in dumped_filter or '"lte":4' in dumped_filter.replace(" ", "")
    assert "maturity_rank" in dumped_filter
    assert "maturity_rank" not in dumped_must
    assert "thriller" not in dumped_must


async def test_maturity_restricted_profile_never_sees_over_rated_title() -> None:
    fake = FakeEs()
    fake.bm25_hits = [
        _hit("pg", "Family Spy", maturity=4),
        _hit("r", "Brutal Spy", maturity=6),
        _hit("nc", "Worse Spy", maturity=7),
    ]
    fake.knn_hits = [_hit("r2", "Also Brutal", maturity=6)]
    out = await retrieve(
        _state(_ctx(maturity_max=MaturityRating.PG), query_rewrite="spy"),
        es=cast(Any, fake),
        embedder=FakeEmbedder(),
        size=25,
    )
    candidates = cast(tuple[Candidate, ...], out["candidates"])
    assert [c.catalog_id for c in candidates] == ["pg"]
    bodies = _bm25_bodies(fake)
    dumped_filter = json.dumps(bodies[0]["query"]["bool"]["filter"])
    assert "maturity_rank" in dumped_filter
    assert "4" in dumped_filter


async def test_franchise_cap_holds_on_emitted_cards() -> None:
    fake = FakeEs()
    fake.bm25_hits = [
        _hit("n1", "Narcos", maturity=5),
        _hit("n2", "Narcos: Mexico", maturity=5),
        _hit("n3", "Narcos: Colombia", maturity=5),
        _hit("d1", "Dark", maturity=5),
        _hit("c1", "Chef's Table", maturity=5),
    ]
    out = await retrieve(
        _state(_ctx(maturity_max=MaturityRating.NC_17), query_rewrite="crime"),
        es=cast(Any, fake),
        embedder=None,
        franchise_cap=1,
        size=25,
    )
    titles = [c.title for c in cast(tuple[Candidate, ...], out["candidates"])]
    assert "Narcos" in titles
    assert "Narcos: Mexico" not in titles
    assert "Narcos: Colombia" not in titles
    keys = [franchise_key(t) for t in titles]
    assert len(keys) == len(set(keys))


async def test_people_include_issues_join_and_fuses() -> None:
    fake = FakeEs()
    fake.bm25_hits = [_hit("s1", "The Departed")]
    fake.people_hits = [_hit("s9", "Catch Me If You Can")]
    constraints = ConstraintState(people_include=("p-dicaprio",))
    out = await retrieve(
        _state(query_rewrite="leo", constraints=constraints),
        es=cast(Any, fake),
        embedder=None,
    )
    kinds = []
    for call in fake.calls:
        body = call["body"]
        if "sort" in body:
            kinds.append("people")
        else:
            kinds.append("bm25")
    assert "people" in kinds
    assert "bm25" in kinds
    ids = {c.catalog_id for c in cast(tuple[Candidate, ...], out["candidates"])}
    assert ids == {"s1", "s9"}


async def test_es_failure_degrades_and_does_not_raise() -> None:
    fake = FakeEs()
    fake.error = ConnectionError("es down")
    out = await retrieve(
        _state(query_rewrite="spy"),
        es=cast(Any, fake),
        embedder=FakeEmbedder(),
    )
    assert out["candidates"] == ()
    assert out["degraded_reason"] is DegradedReason.RETRIEVAL_UNAVAILABLE
    assert "retrieve_attempts" not in out


async def test_missing_es_degrades() -> None:
    out = await retrieve(_state(query_rewrite="spy"), es=None)
    assert out["degraded_reason"] is DegradedReason.RETRIEVAL_UNAVAILABLE
    assert out["candidates"] == ()


async def test_embedder_failure_falls_back_to_bm25() -> None:
    fake = FakeEs()
    fake.bm25_hits = [_hit("s1", "Spy Game")]
    out = await retrieve(
        _state(query_rewrite="spy"),
        es=cast(Any, fake),
        embedder=FakeEmbedder(fail=True),
    )
    candidates = cast(tuple[Candidate, ...], out["candidates"])
    assert [c.catalog_id for c in candidates] == ["s1"]
    assert out.get("degraded_reason") is None


async def test_empty_first_attempt_does_not_mark_empty_catalog() -> None:
    fake = FakeEs()
    out = await retrieve(
        _state(query_rewrite="nope", retrieve_attempts=0, retrieve_max_attempts=2),
        es=cast(Any, fake),
        embedder=None,
    )
    assert out == {"candidates": (), "exclude_exhausted": False}


async def test_empty_last_attempt_sets_empty_catalog_match() -> None:
    fake = FakeEs()
    out = await retrieve(
        _state(query_rewrite="nope", retrieve_attempts=1, retrieve_max_attempts=2),
        es=cast(Any, fake),
        embedder=None,
    )
    assert out["candidates"] == ()
    assert out["degraded_reason"] is DegradedReason.EMPTY_CATALOG_MATCH


async def test_emits_at_most_retrieve_size_cards() -> None:
    fake = FakeEs()
    fake.bm25_hits = [_hit(f"s{i}", f"Title {i}") for i in range(40)]
    out = await retrieve(
        _state(_ctx(maturity_max=MaturityRating.NC_17), query_rewrite="title"),
        es=cast(Any, fake),
        embedder=None,
        size=25,
        franchise_cap=99,
    )
    assert len(cast(tuple[Candidate, ...], out["candidates"])) == 25


# ---------------------------------------------------------------------------
# Graph-owned retry (broaden ladder + cap)
# ---------------------------------------------------------------------------


async def test_zero_hits_triggers_exactly_one_broaden_then_stops() -> None:
    fake = FakeEs()
    visits = {"retrieve": 0, "broaden_constraints": 0}

    async def bound_retrieve(state: TurnState) -> dict[str, object]:
        return await retrieve(state, es=cast(Any, fake), embedder=None)

    compiled = build_graph(
        node_overrides={
            "retrieve": bound_retrieve,
            "broaden_constraints": broaden_constraints,
        }
    )
    async for update in compiled.astream(
        _state(retrieve_max_attempts=2, query_rewrite="nope"),
        stream_mode="updates",
    ):
        for name in update:
            if name in visits:
                visits[name] += 1
    assert visits["retrieve"] == 2
    assert visits["broaden_constraints"] == 1


async def test_hits_skip_broaden() -> None:
    fake = FakeEs()
    fake.bm25_hits = [_hit("s1", "Heat")]
    visits = {"retrieve": 0, "broaden_constraints": 0}

    async def bound_retrieve(state: TurnState) -> dict[str, object]:
        return await retrieve(state, es=cast(Any, fake), embedder=None)

    compiled = build_graph(
        node_overrides={
            "retrieve": bound_retrieve,
            "broaden_constraints": broaden_constraints,
        }
    )
    async for update in compiled.astream(
        _state(query_rewrite="heat", retrieve_max_attempts=2),
        stream_mode="updates",
    ):
        for name in update:
            if name in visits:
                visits[name] += 1
    assert visits["retrieve"] == 1
    assert visits["broaden_constraints"] == 0


async def test_broaden_retry_relaxes_mood_and_then_hits() -> None:
    fake = FakeEs()
    fake.mood_gated = True
    fake.bm25_hits = [_hit("s1", "Relaxed Hit")]
    visits = {"retrieve": 0, "broaden_constraints": 0}

    async def bound_retrieve(state: TurnState) -> dict[str, object]:
        return await retrieve(state, es=cast(Any, fake), embedder=None)

    compiled = build_graph(
        node_overrides={
            "retrieve": bound_retrieve,
            "broaden_constraints": broaden_constraints,
        }
    )
    result = await compiled.ainvoke(
        _state(
            query_rewrite="hit",
            retrieve_max_attempts=2,
            constraints=ConstraintState(moods=(MoodId.TENSE,)),
        )
    )
    async for update in compiled.astream(
        _state(
            query_rewrite="hit",
            retrieve_max_attempts=2,
            constraints=ConstraintState(moods=(MoodId.TENSE,)),
        ),
        stream_mode="updates",
    ):
        for name in update:
            if name in visits:
                visits[name] += 1
    assert visits["retrieve"] == 2
    assert visits["broaden_constraints"] == 1
    assert result["constraints"].moods == ()
    ids = [c.catalog_id for c in result["candidates"]]
    assert ids == ["s1"]
    assert result["retrieve_attempts"] == 2


async def test_hostile_retrieve_return_cannot_write_attempt_keys() -> None:
    fake = FakeEs()
    fake.bm25_hits = [_hit("s1", "Heat")]
    out = await retrieve(_state(query_rewrite="heat"), es=cast(Any, fake), embedder=None)
    assert set(out).isdisjoint({"retrieve_attempts", "retrieve_max_attempts"})


# ---------------------------------------------------------------------------
# MORE_RESULTS exclusion (T35): must_not on catalog_id, exhaustion detection
# ---------------------------------------------------------------------------


class FakeEsExclusionAware(FakeEs):
    """bm25 returns hits only for a query with no catalog_id must_not clause.

    Models a filter that genuinely matches titles, all of which are already
    seen -- the disambiguation `_pool_has_any` exists to make.
    """

    async def search(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        raw_body = kwargs.get("body")
        body: dict[str, Any] = dict(raw_body) if isinstance(raw_body, Mapping) else {}
        if "knn" in body or "sort" in body:
            return {"hits": {"hits": []}}
        blob = json.dumps(body.get("query"))
        if '"must_not"' in blob and '"catalog_id"' in blob:
            return {"hits": {"hits": []}}
        return {"hits": {"hits": self.bm25_hits}}


def test_exclude_filter_is_a_must_not_terms_clause_on_catalog_id() -> None:
    from assist.stores.es import exclude_catalog_ids_filter

    clause = exclude_catalog_ids_filter(["a", "b"])
    assert clause == {"bool": {"must_not": [{"terms": {"catalog_id": ["a", "b"]}}]}}


async def test_retrieve_applies_exclusion_filter_when_flagged() -> None:
    fake = FakeEs()
    fake.bm25_hits = [_hit("s1", "Fresh")]
    await retrieve(
        _state(exclude_seen=True, seen_catalog_ids=("s0",)),
        es=cast(Any, fake),
        embedder=None,
    )
    bodies = _bm25_bodies(fake)
    assert bodies
    blob = json.dumps(bodies[0])
    assert '"catalog_id": ["s0"]' in blob


async def test_retrieve_omits_exclusion_filter_when_not_flagged() -> None:
    fake = FakeEs()
    fake.bm25_hits = [_hit("s1", "Fresh")]
    await retrieve(
        _state(exclude_seen=False, seen_catalog_ids=("s0",)),
        es=cast(Any, fake),
        embedder=None,
    )
    bodies = _bm25_bodies(fake)
    assert bodies
    filter_clauses = bodies[0]["query"]["bool"].get("filter", [])
    assert "catalog_id" not in json.dumps(filter_clauses)


async def test_exhaustion_sets_flag_without_degrading() -> None:
    fake = FakeEsExclusionAware()
    fake.bm25_hits = [_hit("s1", "Fresh")]
    out = await retrieve(
        _state(
            exclude_seen=True,
            seen_catalog_ids=("s0",),
            retrieve_attempts=0,
            retrieve_max_attempts=2,
        ),
        es=cast(Any, fake),
        embedder=None,
    )
    assert out["candidates"] == ()
    assert out["exclude_exhausted"] is True
    assert out.get("degraded_reason", DegradedReason.NONE) is DegradedReason.NONE


async def test_genuinely_empty_filter_with_exclusion_flag_is_not_exhausted() -> None:
    # Nothing matches the filter at all, with or without exclusion -- this
    # must broaden exactly like the no-exclusion case, not exhaust.
    fake = FakeEs()
    out = await retrieve(
        _state(
            exclude_seen=True,
            seen_catalog_ids=("s0",),
            retrieve_attempts=0,
            retrieve_max_attempts=2,
        ),
        es=cast(Any, fake),
        embedder=None,
    )
    assert out["candidates"] == ()
    assert out["exclude_exhausted"] is False


async def test_exhaustion_does_not_enter_broaden_and_keeps_constraints() -> None:
    fake = FakeEsExclusionAware()
    fake.bm25_hits = [_hit("s1", "Fresh")]
    visits = {"retrieve": 0, "broaden_constraints": 0}
    prior = ConstraintState(genres_include=(GenreId.HORROR,), year_min=1990, year_max=1999)

    async def bound_retrieve(state: TurnState) -> dict[str, object]:
        return await retrieve(state, es=cast(Any, fake), embedder=None)

    compiled = build_graph(
        node_overrides={
            "retrieve": bound_retrieve,
            "broaden_constraints": broaden_constraints,
        }
    )
    result = await compiled.ainvoke(
        _state(
            exclude_seen=True,
            seen_catalog_ids=("s0",),
            constraints=prior,
            retrieve_max_attempts=2,
        )
    )
    async for update in compiled.astream(
        _state(
            exclude_seen=True,
            seen_catalog_ids=("s0",),
            constraints=prior,
            retrieve_max_attempts=2,
        ),
        stream_mode="updates",
    ):
        for name in update:
            if name in visits:
                visits[name] += 1
    assert visits["retrieve"] == 1
    assert visits["broaden_constraints"] == 0
    assert result["candidates"] == ()
    assert result["exclude_exhausted"] is True
    assert result["degraded_reason"] is DegradedReason.NONE
    # The year/genre filter the user chose survives untouched -- the ladder
    # never ran, so it never had a chance to drop them.
    assert result["constraints"] == prior


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
    pytest.skip("elasticsearch unreachable")


@pytest.fixture
async def es_client(es_url: str) -> AsyncIterator[AsyncElasticsearch]:
    client = create_client(url=es_url)
    try:
        yield client
    finally:
        await close_client(client)


@pytest.fixture
def isolated_alias() -> str:
    return f"t17t{uuid.uuid4().hex[:8]}"


async def _cleanup(client: AsyncElasticsearch, *aliases: str) -> None:
    for alias in aliases:
        names = await versioned_index_names(client, alias)
        if names:
            await client.indices.delete(index=",".join(names))


def _doc(
    catalog_id: str,
    title: str,
    *,
    synopsis: str,
    maturity: int,
    genres: list[str],
) -> dict[str, Any]:
    return {
        "catalog_id": catalog_id,
        "media_type": "film",
        "genres": genres,
        "moods": [],
        "origins": ["United States"],
        "maturity_rank": maturity,
        "local_original": False,
        "release_year": 2010,
        "runtime_min": 100,
        "people_ids": [],
        "pop_28d": 0.5,
        "title": title,
        "synopsis": synopsis,
        "tags": [],
        "people_names": [],
        "era_feel": "",
        # Cosine rejects zero-magnitude vectors; a constant non-zero is enough for BM25 tests.
        "embedding": [0.01] * EMBEDDING_DIMS,
    }


async def test_live_filters_do_not_change_bm25_scores_of_surviving_docs(
    es_client: AsyncElasticsearch,
    isolated_alias: str,
) -> None:
    titles_alias = isolated_alias
    people_alias = isolated_alias.replace("t17t", "t17p", 1)
    token = f"t17score{uuid.uuid4().hex[:6]}"
    try:
        await bootstrap(
            es_client,
            titles_alias=titles_alias,
            people_alias=people_alias,
            titles_body=TITLES_INDEX_BODY,
            people_body=PEOPLE_INDEX_BODY,
        )
        docs = [
            _doc("low", f"{token} alpha", synopsis=f"{token} spy", maturity=3, genres=["drama"]),
            _doc("mid", f"{token} bravo", synopsis=f"{token} spy", maturity=4, genres=["drama"]),
            _doc("high", f"{token} charlie", synopsis=f"{token} spy", maturity=6, genres=["drama"]),
        ]
        for doc in docs:
            await es_client.index(index=titles_alias, id=doc["catalog_id"], document=doc)
        await es_client.indices.refresh(index=titles_alias)

        unfiltered = titles_bm25_body(token, ())
        raw_open = await es_client.search(index=titles_alias, body=unfiltered)
        open_hits = _live_scores(raw_open)
        assert set(open_hits) == {"low", "mid", "high"}

        filtered = titles_bm25_body(
            token,
            constraint_filters(maturity_rank_max=4),
        )
        raw_filt = await es_client.search(index=titles_alias, body=filtered)
        filt_hits = _live_scores(raw_filt)
        assert set(filt_hits) == {"low", "mid"}
        assert "high" not in filt_hits
        for catalog_id in ("low", "mid"):
            assert filt_hits[catalog_id] == pytest.approx(open_hits[catalog_id])
    finally:
        await _cleanup(es_client, titles_alias, people_alias)


def _live_scores(resp: object) -> dict[str, float]:
    body = getattr(resp, "body", resp)
    assert isinstance(body, Mapping)
    hits = body["hits"]["hits"]
    out: dict[str, float] = {}
    for hit in hits:
        src = hit.get("_source") or {}
        catalog_id = src.get("catalog_id") or hit.get("_id")
        score = hit.get("_score")
        if isinstance(catalog_id, str) and isinstance(score, int | float):
            out[catalog_id] = float(score)
    return out


async def test_live_maturity_profile_cannot_see_over_rated_title(
    es_client: AsyncElasticsearch,
    isolated_alias: str,
) -> None:
    titles_alias = isolated_alias
    people_alias = isolated_alias.replace("t17t", "t17p", 1)
    token = f"t17mat{uuid.uuid4().hex[:6]}"
    try:
        await bootstrap(
            es_client,
            titles_alias=titles_alias,
            people_alias=people_alias,
            titles_body=TITLES_INDEX_BODY,
            people_body=PEOPLE_INDEX_BODY,
        )
        await es_client.index(
            index=titles_alias,
            id="ok",
            document=_doc("ok", f"{token} safe", synopsis=token, maturity=4, genres=["drama"]),
        )
        await es_client.index(
            index=titles_alias,
            id="bad",
            document=_doc("bad", f"{token} adult", synopsis=token, maturity=6, genres=["drama"]),
        )
        await es_client.indices.refresh(index=titles_alias)

        out = await retrieve(
            _state(_ctx(maturity_max=MaturityRating.PG), query_rewrite=token),
            es=es_client,
            embedder=None,
            index=titles_alias,
        )
        candidates = cast(tuple[Candidate, ...], out["candidates"])
        assert [c.catalog_id for c in candidates] == ["ok"]
    finally:
        await _cleanup(es_client, titles_alias, people_alias)


class _AliasSpy:
    """Records the index name. `options()` returns self so retrieve still hits search."""

    def __init__(self, inner: AsyncElasticsearch) -> None:
        self._inner = inner
        self.indexes: list[str] = []

    def options(self, **_kwargs: Any) -> _AliasSpy:
        return self

    async def search(self, **kwargs: Any) -> Any:
        index = kwargs.get("index")
        if isinstance(index, str):
            self.indexes.append(index)
        return await self._inner.search(**kwargs)


async def test_live_retrieve_reads_titles_alias_never_versioned_name(
    es_client: AsyncElasticsearch,
) -> None:
    spy = _AliasSpy(es_client)
    out = await retrieve(
        _state(_ctx(maturity_max=MaturityRating.PG), query_rewrite="the matrix"),
        es=spy,
        embedder=None,
    )
    assert spy.indexes
    assert all(index == TITLES_ALIAS for index in spy.indexes)
    assert all("_v" not in index for index in spy.indexes)
    candidates = cast(tuple[Candidate, ...], out.get("candidates", ()))
    if not candidates:
        return
    ids = [c.catalog_id for c in candidates]
    raw = await es_client.search(
        index=TITLES_ALIAS,
        body={
            "size": len(ids),
            "query": {"ids": {"values": ids}},
            "_source": ["catalog_id", "maturity_rank"],
        },
    )
    body = getattr(raw, "body", raw)
    assert isinstance(body, Mapping)
    for hit in body["hits"]["hits"]:
        rank = hit["_source"]["maturity_rank"]
        assert int(rank) <= maturity_rank(MaturityRating.PG)


def test_filters_from_constraints_always_include_profile_ceiling() -> None:
    state = ConstraintState(genres_include=(GenreId.COMEDY,))
    clauses = filters_from_constraints(state, maturity_rank_max=4)
    assert {"range": {"maturity_rank": {"lte": 4}}} in clauses
