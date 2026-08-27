"""Deterministic ranker: pop + constraint match + semantic, then franchise cap.

Weights come from config, never from literals. Min-max is per candidate set.
Missing pop uses the set median, or an editorial prior when the set has none.
The node never invents a catalog_id and never writes retrieve_attempts.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from assist.config import Settings
from assist.config import settings as default_settings
from assist.domain.catalog import Candidate
from assist.domain.constraints import ConstraintState
from assist.domain.enums import GenreId, MediaType, MoodId
from assist.graph.state import TurnState
from assist.nodes.retrieval import franchise_key
from assist.obs.logging import get_logger

log = get_logger("assist.nodes.rank")

# Design: client picks.length ∈ [0, 8]. Diversity applies to that window.
_TOP_K = 8
_FRANCHISE_CAP = 1
# Used only when every candidate is missing pop_28d. After min-max the
# component is zero either way; the prior exists so missing pop cannot crash.
_EDITORIAL_POP_PRIOR = 0.0


@dataclass(frozen=True)
class RankWeights:
    """Mixing weights. Construct from settings so a typo cannot drift the formula."""

    pop: float
    constraint: float
    semantic: float

    @classmethod
    def from_settings(cls, cfg: Settings | None = None) -> RankWeights:
        src = cfg if cfg is not None else default_settings
        return cls(
            pop=src.rank_w_pop,
            constraint=src.rank_w_constraint,
            semantic=src.rank_w_semantic,
        )


@dataclass(frozen=True)
class RankFeatures:
    """Optional catalog projection. None means unknown, not zero.

    Candidate is a compact card (T02) and does not carry pop, moods, origins,
    runtime, or people. T24 can hydrate these from the catalog. Live enrichment
    currently has zero rows, so moods stay empty unless a test injects them.
    """

    pop_28d: float | None = None
    media_type: MediaType | None = None
    genres: tuple[GenreId, ...] | None = None
    moods: tuple[MoodId, ...] | None = None
    release_year: int | None = None
    runtime_min: int | None = None
    origins: tuple[str, ...] | None = None
    local_original: bool | None = None
    people_ids: tuple[str, ...] | None = None
    semantic_score: float | None = None


def min_max_norm(values: Sequence[float]) -> list[float]:
    """Scale to [0, 1] over this set. A zero range is no signal, not a crash."""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span == 0.0:
        return [0.0] * len(values)
    return [(value - lo) / span for value in values]


def log_pop(pop: float) -> float:
    """log1p so a few huge pop_28d values cannot flatten the rest."""
    return math.log1p(max(pop, 0.0))


def fill_cold_start(pops: Sequence[float | None]) -> list[float]:
    """Replace missing pop with the set median, else the editorial prior."""
    known = [value for value in pops if value is not None]
    if known:
        prior = float(statistics.median(known))
    else:
        prior = _EDITORIAL_POP_PRIOR
    return [float(value) if value is not None else prior for value in pops]


def constraint_match(features: RankFeatures, constraints: ConstraintState) -> float:
    """Fraction of active soft constraints this title satisfies.

    List includes use coverage (partial credit). Excludes are binary. A missing
    feature scores 0 on that axis — we never invent a match. Constraints we
    cannot evaluate (languages, recency_bias) are skipped, not failed.
    """
    parts: list[float] = []

    media = constraints.media_type
    if media is not None and media is not MediaType.ANY:
        parts.append(1.0 if features.media_type is media else 0.0)

    if constraints.genres_include:
        parts.append(_coverage(features.genres, constraints.genres_include))
    if constraints.genres_exclude:
        parts.append(_exclude_clear(features.genres, constraints.genres_exclude))

    if constraints.moods:
        parts.append(_coverage(features.moods, constraints.moods))

    if constraints.year_min is not None or constraints.year_max is not None:
        parts.append(
            _year_in_window(features.release_year, constraints.year_min, constraints.year_max)
        )

    if constraints.duration_max_min is not None:
        runtime = features.runtime_min
        if runtime is None:
            parts.append(0.0)
        else:
            parts.append(1.0 if runtime <= constraints.duration_max_min else 0.0)

    if constraints.origins:
        parts.append(_origin_coverage(features.origins, constraints.origins))

    if constraints.local_originals_only:
        parts.append(1.0 if features.local_original is True else 0.0)

    if constraints.people_include:
        parts.append(_coverage(features.people_ids, constraints.people_include))
    if constraints.people_exclude:
        parts.append(_exclude_clear(features.people_ids, constraints.people_exclude))

    if not parts:
        return 0.0
    return sum(parts) / len(parts)


def fused_score(
    *,
    pop_norm: float,
    constraint: float,
    semantic_norm: float,
    weights: RankWeights,
) -> float:
    return (
        weights.pop * pop_norm + weights.constraint * constraint + weights.semantic * semantic_norm
    )


def apply_franchise_cap(
    ordered: Sequence[Candidate],
    *,
    cap: int = _FRANCHISE_CAP,
    window: int = _TOP_K,
) -> list[Candidate]:
    """Greedy: at most `cap` per family in the first `window` slots.

    Overflow keeps score order, including siblings pushed out of the window.
    The output is a permutation of the input — no drop, no substitute.
    """
    if cap < 1 or window < 1:
        return list(ordered)
    primary: list[Candidate] = []
    overflow: list[Candidate] = []
    counts: dict[str, int] = {}
    for candidate in ordered:
        if len(primary) >= window:
            overflow.append(candidate)
            continue
        key = franchise_key(candidate.title) or candidate.catalog_id
        if counts.get(key, 0) >= cap:
            overflow.append(candidate)
            continue
        primary.append(candidate)
        counts[key] = counts.get(key, 0) + 1
    return primary + overflow


def rank_candidates(
    candidates: Sequence[Candidate],
    constraints: ConstraintState,
    *,
    features: Mapping[str, RankFeatures] | None = None,
    vector_path: bool = True,
    weights: RankWeights | None = None,
    franchise_cap: int = _FRANCHISE_CAP,
    top_k: int = _TOP_K,
) -> list[Candidate]:
    """Pure ranker. Same input → same order in every process."""
    mix = weights if weights is not None else RankWeights.from_settings()
    extras = features or {}
    unique = _unique_in_order(candidates)
    if not unique:
        return []

    resolved = [
        _resolve_features(candidate, extras.get(candidate.catalog_id)) for candidate in unique
    ]
    filled_pop = fill_cold_start([item.pop_28d for item in resolved])
    pop_norm = min_max_norm([log_pop(value) for value in filled_pop])
    matches = [constraint_match(item, constraints) for item in resolved]
    if vector_path:
        raw_semantic = [
            item.semantic_score if item.semantic_score is not None else 0.0 for item in resolved
        ]
        semantic_norm = min_max_norm(raw_semantic)
    else:
        semantic_norm = [0.0] * len(resolved)

    scored: list[Candidate] = []
    rows = zip(unique, pop_norm, matches, semantic_norm, strict=True)
    for candidate, pop, match, semantic in rows:
        score = fused_score(pop_norm=pop, constraint=match, semantic_norm=semantic, weights=mix)
        scored.append(candidate.model_copy(update={"score": score}))

    # catalog_id is a stable tiebreak. Builtin hash is randomised per process.
    ordered = sorted(scored, key=lambda item: (-item.score, item.catalog_id))
    return apply_franchise_cap(ordered, cap=franchise_cap, window=top_k)


async def rank(
    state: TurnState,
    *,
    features: Mapping[str, RankFeatures] | None = None,
    vector_path: bool = True,
    cfg: Settings | None = None,
    franchise_cap: int | None = None,
    top_k: int = _TOP_K,
) -> dict[str, object]:
    """LangGraph node. Reorders `candidates`; does not add or drop ids."""
    constraints = _constraints_of(state)
    incoming = _candidates_of(state)
    weights = RankWeights.from_settings(cfg)
    cap = _FRANCHISE_CAP if franchise_cap is None else franchise_cap
    ranked = rank_candidates(
        incoming,
        constraints,
        features=features,
        vector_path=vector_path,
        weights=weights,
        franchise_cap=cap,
        top_k=top_k,
    )
    top1, gap = _top1_gap(ranked)
    log.info(
        "rank",
        n_in=len(incoming),
        n_out=len(ranked),
        top1=top1,
        gap=gap,
        vector_path=vector_path,
        w_pop=weights.pop,
        w_constraint=weights.constraint,
        w_semantic=weights.semantic,
    )
    return {"candidates": tuple(ranked), "top1": top1, "gap": gap}


def _candidates_of(state: TurnState) -> tuple[Candidate, ...]:
    raw = state.get("candidates") or ()
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        return ()
    return tuple(item for item in raw if isinstance(item, Candidate))


def _constraints_of(state: TurnState) -> ConstraintState:
    raw = state.get("constraints")
    if isinstance(raw, ConstraintState):
        return raw
    return ConstraintState.empty()


def _unique_in_order(candidates: Sequence[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    out: list[Candidate] = []
    for candidate in candidates:
        if candidate.catalog_id in seen:
            continue
        seen.add(candidate.catalog_id)
        out.append(candidate)
    return out


def _resolve_features(candidate: Candidate, extra: RankFeatures | None) -> RankFeatures:
    base = RankFeatures(
        media_type=candidate.media_type,
        genres=candidate.genres,
        release_year=candidate.release_year,
        semantic_score=candidate.score,
    )
    if extra is None:
        return base
    return RankFeatures(
        pop_28d=extra.pop_28d if extra.pop_28d is not None else base.pop_28d,
        media_type=extra.media_type if extra.media_type is not None else base.media_type,
        genres=extra.genres if extra.genres is not None else base.genres,
        moods=extra.moods if extra.moods is not None else base.moods,
        release_year=extra.release_year if extra.release_year is not None else base.release_year,
        runtime_min=extra.runtime_min if extra.runtime_min is not None else base.runtime_min,
        origins=extra.origins if extra.origins is not None else base.origins,
        local_original=(
            extra.local_original if extra.local_original is not None else base.local_original
        ),
        people_ids=extra.people_ids if extra.people_ids is not None else base.people_ids,
        semantic_score=(
            extra.semantic_score if extra.semantic_score is not None else base.semantic_score
        ),
    )


def _coverage[T](have: tuple[T, ...] | None, wanted: Sequence[T]) -> float:
    if not wanted:
        return 1.0
    if not have:
        return 0.0
    present = set(have)
    return sum(1 for item in wanted if item in present) / len(wanted)


def _exclude_clear[T](have: tuple[T, ...] | None, excluded: Sequence[T]) -> float:
    if not excluded:
        return 1.0
    if not have:
        return 1.0
    present = set(have)
    return 0.0 if any(item in present for item in excluded) else 1.0


def _origin_coverage(have: tuple[str, ...] | None, wanted: Sequence[str]) -> float:
    if not wanted:
        return 1.0
    if not have:
        return 0.0
    present = {item.casefold() for item in have}
    return sum(1 for item in wanted if item.casefold() in present) / len(wanted)


def _year_in_window(year: int | None, year_min: int | None, year_max: int | None) -> float:
    if year is None:
        return 0.0
    if year_min is not None and year < year_min:
        return 0.0
    if year_max is not None and year > year_max:
        return 0.0
    return 1.0


def _top1_gap(ranked: Sequence[Candidate]) -> tuple[float | None, float | None]:
    if not ranked:
        return None, None
    top1 = ranked[0].score
    if len(ranked) < 2:
        return top1, None
    return top1, top1 - ranked[1].score
