"""Hybrid retrieval: concurrent BM25 + kNN, Python RRF, franchise cap, broaden.

T24 wires `retrieve` and `broaden_constraints` into the graph. The graph wrapper
owns `retrieve_attempts` / `retrieve_max_attempts` — this module must not write
either key. ES failure or timeout degrades the turn; it never raises.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from assist.config import settings
from assist.domain.catalog import Candidate
from assist.domain.constraints import ConstraintState
from assist.domain.context import ServerUserCtx
from assist.domain.enums import (
    DegradedReason,
    GenreId,
    MediaType,
    is_stricter_than,
    maturity_rank,
)
from assist.graph.state import TurnState
from assist.obs.logging import get_logger
from assist.stores.es import (
    TITLES_ALIAS,
    TITLES_SOURCE_FIELDS,
    exclude_catalog_ids_filter,
    filters_from_constraints,
    titles_bm25_body,
    titles_knn_body,
)
from assist.stores.mappings import EMBEDDING_DIMS

log = get_logger("assist.nodes.retrieval")

# Colon / slash sequels share a franchise ("Narcos: Mexico" -> "narcos").
_FRANCHISE_SPLIT = re.compile(r"\s*[:/|]\s*")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_LEADING_ARTICLE = re.compile(r"^(the|a|an)\s+")

_RETRIEVAL_REASONS = frozenset(
    {DegradedReason.RETRIEVAL_UNAVAILABLE, DegradedReason.EMPTY_CATALOG_MATCH}
)


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class RetrievedHit:
    catalog_id: str
    title: str
    media_type: MediaType
    release_year: int | None
    genres: tuple[GenreId, ...]
    maturity_rank: int


def query_text(state: TurnState) -> str:
    rewrite = state.get("query_rewrite")
    if isinstance(rewrite, str) and rewrite.strip():
        return rewrite.strip()
    text = state.get("text")
    if isinstance(text, str):
        return text.strip()
    return ""


def ceiling_maturity_rank(ctx: ServerUserCtx, constraints: ConstraintState) -> int:
    """Effective AuthZ ceiling: min(profile, requested_stricter). Never from the client."""
    requested = constraints.maturity_request_stricter
    if requested is not None and is_stricter_than(requested, ctx.maturity_max):
        return maturity_rank(requested)
    return maturity_rank(ctx.maturity_max)


def rrf_fuse(
    *rankings: Sequence[str],
    k: int,
) -> list[tuple[str, float]]:
    """Reciprocal rank fusion. `score = Σ 1/(k + rank)` with 1-based ranks.

    First occurrence in a list wins; later duplicates in the same list are ignored.
    Ties break on `catalog_id` so the order is stable.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def franchise_key(title: str) -> str:
    """Stable family key. Colon sequels collapse; leading articles drop."""
    raw = title.strip().lower()
    if not raw:
        return ""
    head = _FRANCHISE_SPLIT.split(raw, maxsplit=1)[0]
    head = _LEADING_ARTICLE.sub("", head)
    head = _NON_ALNUM.sub(" ", head)
    return " ".join(head.split())


def diversify(
    ordered: Sequence[RetrievedHit],
    *,
    cap: int,
    size: int,
) -> list[RetrievedHit]:
    """Greedy franchise cap across the emitted set. Skipped siblings are dropped."""
    if cap < 1 or size < 1:
        return []
    kept: list[RetrievedHit] = []
    counts: dict[str, int] = {}
    for hit in ordered:
        key = franchise_key(hit.title) or hit.catalog_id
        if counts.get(key, 0) >= cap:
            continue
        kept.append(hit)
        counts[key] = counts.get(key, 0) + 1
        if len(kept) >= size:
            break
    return kept


def apply_broaden_ladder(constraints: ConstraintState) -> ConstraintState:
    """One-shot recall expansion. Maturity and people_exclude never relax."""
    flavor: dict[str, object] = {}
    if constraints.moods:
        flavor["moods"] = ()
    if constraints.duration_max_min is not None:
        flavor["duration_max_min"] = None
    if constraints.local_originals_only:
        flavor["local_originals_only"] = False
    if constraints.year_min is not None or constraints.year_max is not None:
        flavor["year_min"] = None
        flavor["year_max"] = None
    if constraints.recency_bias is not None:
        flavor["recency_bias"] = None
    if constraints.genres_exclude:
        flavor["genres_exclude"] = ()
    if constraints.origins:
        flavor["origins"] = ()
    if constraints.languages:
        flavor["languages"] = ()
    if flavor:
        return constraints.model_copy(update=flavor)

    if constraints.people_include:
        return constraints.model_copy(update={"people_include": ()})
    if constraints.genres_include:
        return constraints.model_copy(update={"genres_include": ()})
    if constraints.media_type is not None and constraints.media_type is not MediaType.ANY:
        return constraints.model_copy(update={"media_type": MediaType.ANY})
    return constraints


def titles_people_join_body(
    filters: Sequence[Mapping[str, Any]] = (),
    *,
    size: int = 50,
) -> dict[str, Any]:
    """Popularity-ranked titles for resolved people. Same filter set, no scoring query."""
    query: dict[str, Any]
    if filters:
        query = {"bool": {"filter": list(filters)}}
    else:
        query = {"match_all": {}}
    return {
        "size": size,
        "query": query,
        "sort": [{"pop_28d": {"order": "desc"}}, {"catalog_id": {"order": "asc"}}],
        "_source": list(TITLES_SOURCE_FIELDS),
    }


def _as_mapping(resp: object) -> dict[str, Any]:
    body = getattr(resp, "body", resp)
    if isinstance(body, Mapping):
        return dict(body)
    return {}


def _constraints_of(state: TurnState) -> ConstraintState:
    raw = state.get("constraints")
    if isinstance(raw, ConstraintState):
        return raw
    return ConstraintState.empty()


def _attempt_cap(state: TurnState) -> int:
    configured = int(settings.retrieve_max_attempts)
    requested = int(state.get("retrieve_max_attempts") or configured)
    cap = min(requested, configured)
    return configured if cap < 1 else cap


def _is_last_attempt(state: TurnState) -> bool:
    attempts = int(state.get("retrieve_attempts") or 0)
    return attempts + 1 >= _attempt_cap(state)


def _parse_genres(raw: object) -> tuple[GenreId, ...]:
    if not isinstance(raw, list):
        return ()
    out: list[GenreId] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        try:
            out.append(GenreId(item))
        except ValueError:
            continue
    return tuple(out)


def _parse_year(raw: object) -> int | None:
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None
    return raw


def _parse_hit(raw: object) -> RetrievedHit | None:
    if not isinstance(raw, Mapping):
        return None
    src_obj = raw.get("_source")
    src: Mapping[str, Any] = src_obj if isinstance(src_obj, Mapping) else {}
    catalog_id = src.get("catalog_id") or raw.get("_id")
    title = src.get("title")
    media_raw = src.get("media_type")
    maturity_raw = src.get("maturity_rank")
    if not isinstance(catalog_id, str) or not catalog_id:
        return None
    if not isinstance(title, str) or not title:
        return None
    if isinstance(maturity_raw, bool) or not isinstance(maturity_raw, int):
        # Missing rank cannot be authorized; drop fail-closed.
        return None
    try:
        media_type = MediaType(media_raw) if isinstance(media_raw, str) else None
    except ValueError:
        media_type = None
    if media_type is None or media_type is MediaType.ANY:
        return None
    return RetrievedHit(
        catalog_id=catalog_id,
        title=title,
        media_type=media_type,
        release_year=_parse_year(src.get("release_year")),
        genres=_parse_genres(src.get("genres")),
        maturity_rank=maturity_raw,
    )


def _ranking_ids(hits: Sequence[RetrievedHit]) -> list[str]:
    return [hit.catalog_id for hit in hits]


def _hits_from_response(resp: object) -> list[RetrievedHit]:
    body = _as_mapping(resp)
    hits_obj = body.get("hits")
    if not isinstance(hits_obj, Mapping):
        return []
    raw_hits = hits_obj.get("hits")
    if not isinstance(raw_hits, list):
        return []
    out: list[RetrievedHit] = []
    seen: set[str] = set()
    for raw in raw_hits:
        hit = _parse_hit(raw)
        if hit is None or hit.catalog_id in seen:
            continue
        seen.add(hit.catalog_id)
        out.append(hit)
    return out


async def _search(
    es: Any,  # duck-typed; see retrieve()
    *,
    index: str,
    body: Mapping[str, Any],
    timeout_s: float,
) -> list[RetrievedHit]:
    payload: dict[str, Any] = {"index": index, "body": dict(body)}
    options = getattr(es, "options", None)
    if callable(options):
        # Per-request timeout; do not inherit the T06 client's 3 retries.
        resp = await options(request_timeout=timeout_s, max_retries=0).search(**payload)
    else:
        resp = await es.search(**payload, request_timeout=timeout_s)
    return _hits_from_response(resp)


async def _embed_query(embedder: Embedder | None, text: str) -> list[float] | None:
    if embedder is None or not text:
        return None
    try:
        vectors = await embedder.embed([text])
    except Exception as exc:
        log.warning("retrieve_embed_failed", error=str(exc) or type(exc).__name__)
        return None
    if not vectors or not isinstance(vectors[0], list):
        return None
    vector = vectors[0]
    if len(vector) != EMBEDDING_DIMS:
        log.warning("retrieve_embed_bad_dim", dim=len(vector))
        return None
    parsed: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, int | float):
            log.warning("retrieve_embed_non_numeric")
            return None
        parsed.append(float(value))
    return parsed


def _degraded(reason: DegradedReason) -> dict[str, object]:
    return {"candidates": (), "degraded_reason": reason, "exclude_exhausted": False}


def _success(
    candidates: Sequence[Candidate],
    state: TurnState,
) -> dict[str, object]:
    update: dict[str, object] = {"candidates": tuple(candidates), "exclude_exhausted": False}
    reason = state.get("degraded_reason")
    if reason in _RETRIEVAL_REASONS:
        update["degraded_reason"] = DegradedReason.NONE
    return update


def _seen_ids_of(state: TurnState) -> tuple[str, ...]:
    raw = state.get("seen_catalog_ids") or ()
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(item for item in raw if isinstance(item, str) and item)


async def _pool_has_any(
    es: Any,
    *,
    index: str,
    text: str,
    filters: Sequence[Mapping[str, Any]],
    timeout_s: float,
) -> bool:
    """Existence check on `filters` alone, without the seen-id exclusion.

    Disambiguates exhaustion (something matches the filter, all of it seen)
    from a genuinely empty filter (broaden should still run). Same text +
    filters as the real query so a text mismatch cannot masquerade as
    exhaustion.
    """
    body = titles_bm25_body(text, filters, size=1)
    try:
        hits = await _search(es, index=index, body=body, timeout_s=timeout_s)
    except Exception as exc:
        log.warning("retrieve_exhaustion_check_failed", error=str(exc) or type(exc).__name__)
        return False
    return bool(hits)


def _to_candidates(
    ordered: Sequence[tuple[str, float]],
    by_id: Mapping[str, RetrievedHit],
    *,
    ceiling: int,
    cap: int,
    size: int,
) -> tuple[Candidate, ...]:
    scores = dict(ordered)
    authorized: list[RetrievedHit] = []
    for catalog_id, _score in ordered:
        hit = by_id.get(catalog_id)
        if hit is None or hit.maturity_rank > ceiling:
            continue
        authorized.append(hit)
    diverse = diversify(authorized, cap=cap, size=size)
    return tuple(
        Candidate(
            catalog_id=hit.catalog_id,
            title=hit.title,
            media_type=hit.media_type,
            release_year=hit.release_year,
            genres=hit.genres,
            score=scores[hit.catalog_id],
        )
        for hit in diverse
    )


async def retrieve(
    state: TurnState,
    *,
    # AsyncElasticsearch or a test double. ES 8 search overloads reject a Protocol.
    es: Any = None,
    embedder: Embedder | None = None,
    index: str = TITLES_ALIAS,
    rrf_k: int | None = None,
    size: int | None = None,
    each_k: int | None = None,
    franchise_cap: int | None = None,
    timeout_s: float | None = None,
) -> dict[str, object]:
    """One hybrid search. Empty hits let the graph broaden; this node does not retry."""
    ctx = state.get("ctx")
    if not isinstance(ctx, ServerUserCtx) or es is None:
        log.warning(
            "retrieve_unavailable",
            has_ctx=isinstance(ctx, ServerUserCtx),
            has_es=es is not None,
        )
        return _degraded(DegradedReason.RETRIEVAL_UNAVAILABLE)

    constraints = _constraints_of(state)
    ceiling = ceiling_maturity_rank(ctx, constraints)
    base_filters = filters_from_constraints(constraints, maturity_rank_max=ceiling)
    exclude_seen = bool(state.get("exclude_seen"))
    seen_ids = _seen_ids_of(state)
    # MORE_RESULTS (T35): must_not on catalog_id, not a separate query. The
    # constraint filter is unchanged -- only what counts as "already shown"
    # is added, so a tap never narrows the user's own filter.
    filters = base_filters
    if exclude_seen and seen_ids:
        filters = [*base_filters, exclude_catalog_ids_filter(seen_ids)]
    text = query_text(state)
    k_rrf = int(settings.rrf_k if rrf_k is None else rrf_k)
    emit_size = int(settings.retrieve_size if size is None else size)
    per_list = int(settings.retrieve_each_k if each_k is None else each_k)
    cap = int(settings.retrieve_franchise_cap if franchise_cap is None else franchise_cap)
    timeout = timeout_s if timeout_s is not None else settings.elasticsearch_timeout_ms / 1000.0

    bm25_body = titles_bm25_body(text, filters, size=per_list)
    people_ids = constraints.people_include
    join_body = titles_people_join_body(filters, size=per_list) if people_ids else None

    async def _bm25() -> list[RetrievedHit]:
        return await _search(es, index=index, body=bm25_body, timeout_s=timeout)

    async def _knn() -> list[RetrievedHit]:
        vector = await _embed_query(embedder, text)
        if vector is None:
            return []
        try:
            knn_body = titles_knn_body(vector, filters, k=per_list)
        except ValueError as exc:
            log.warning("retrieve_knn_body_failed", error=str(exc))
            return []
        return await _search(es, index=index, body=knn_body, timeout_s=timeout)

    async def _people() -> list[RetrievedHit]:
        if join_body is None:
            return []
        return await _search(es, index=index, body=join_body, timeout_s=timeout)

    gathered = await asyncio.gather(_bm25(), _knn(), _people(), return_exceptions=True)
    bm25_out, knn_out, people_out = gathered
    knn_skipped = embedder is None or not text
    people_skipped = join_body is None

    def _ok(label: str, outcome: object, *, skipped: bool) -> list[RetrievedHit]:
        if isinstance(outcome, BaseException):
            log.warning(
                "retrieve_list_failed",
                list=label,
                error=str(outcome) or type(outcome).__name__,
            )
            return []
        if skipped:
            return []
        return list(outcome) if isinstance(outcome, list) else []

    bm25_hits = _ok("bm25", bm25_out, skipped=False)
    knn_hits = _ok("knn", knn_out, skipped=knn_skipped)
    people_hits = _ok("people", people_out, skipped=people_skipped)

    bm25_failed = isinstance(bm25_out, BaseException)
    knn_failed = isinstance(knn_out, BaseException)
    people_failed = isinstance(people_out, BaseException)
    # No usable list when every path that could have hit ES failed or was skipped.
    if bm25_failed and (knn_failed or knn_skipped) and (people_failed or people_skipped):
        return _degraded(DegradedReason.RETRIEVAL_UNAVAILABLE)

    by_id: dict[str, RetrievedHit] = {}
    rankings: list[list[str]] = []
    for hits in (bm25_hits, knn_hits, people_hits):
        if not hits:
            continue
        rankings.append(_ranking_ids(hits))
        for hit in hits:
            by_id.setdefault(hit.catalog_id, hit)

    fused = rrf_fuse(*rankings, k=k_rrf) if rankings else []
    candidates = _to_candidates(
        fused,
        by_id,
        ceiling=ceiling,
        cap=cap,
        size=emit_size,
    )

    log.info(
        "retrieve",
        index=index,
        n_bm25=len(bm25_hits),
        n_knn=len(knn_hits),
        n_people=len(people_hits),
        n_fused=len(fused),
        n_emitted=len(candidates),
        maturity_rank_max=ceiling,
        people_join=join_body is not None,
        last_attempt=_is_last_attempt(state),
    )

    if candidates:
        return _success(candidates, state)

    exhausted = False
    if exclude_seen and seen_ids:
        exhausted = await _pool_has_any(
            es, index=index, text=text, filters=base_filters, timeout_s=timeout
        )
    if exhausted:
        log.info("retrieve_exhausted", index=index, n_seen=len(seen_ids))
        update: dict[str, object] = {"candidates": (), "exclude_exhausted": True}
        reason = state.get("degraded_reason")
        if reason in _RETRIEVAL_REASONS:
            update["degraded_reason"] = DegradedReason.NONE
        return update

    if _is_last_attempt(state):
        return _degraded(DegradedReason.EMPTY_CATALOG_MATCH)
    return {"candidates": (), "exclude_exhausted": False}


async def broaden_constraints(state: TurnState) -> dict[str, object]:
    """Deterministic ladder. Graph calls this once between empty retrieve visits."""
    prior = _constraints_of(state)
    broadened = apply_broaden_ladder(prior)
    log.info(
        "broaden_constraints",
        changed=broadened != prior,
        dropped_moods=bool(prior.moods) and not broadened.moods,
        dropped_years=prior.year_min is not None and broadened.year_min is None,
        kept_maturity=(
            prior.maturity_request_stricter.value
            if prior.maturity_request_stricter is not None
            else None
        ),
    )
    return {"constraints": broadened}
