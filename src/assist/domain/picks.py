"""Pure pick sanitizer. The model never names a title; it only returns indices/IDs."""

from __future__ import annotations

from collections.abc import Sequence, Set

from assist.domain.catalog import Candidate
from assist.domain.enums import DegradedReason, Route

MAX_PICKS_DEFAULT = 8
MIN_PICKS_GRID = 3
MIN_PICKS_NONE = 0

_NO_GRID_REASONS = frozenset(
    {
        DegradedReason.PERSON_AMBIGUOUS,
        DegradedReason.EMPTY_CATALOG_MATCH,
        DegradedReason.SAFETY_BLOCK,
    }
)
_BEST_EFFORT_REASONS = frozenset(
    {
        DegradedReason.HARD_TIMEOUT,
        DegradedReason.PROVIDER_THROTTLE,
        DegradedReason.GENERATIVE_TIMEOUT,
        DegradedReason.RETRIEVAL_UNAVAILABLE,
        DegradedReason.SESSION_STORE_UNAVAILABLE,
    }
)


def min_picks_for(
    *,
    route: Route | None = None,
    degraded_reason: DegradedReason | None = None,
    entitled_count: int = 0,
) -> int:
    """Call-site `min_picks` from the design.md policy matrix.

    Entitled-empty always forces 0 — there is nothing legal to pad with.
    """
    if entitled_count <= 0:
        return MIN_PICKS_NONE
    reason = degraded_reason if degraded_reason is not DegradedReason.NONE else None
    if reason in _NO_GRID_REASONS or route in {Route.SAFETY, Route.CLARIFY}:
        return MIN_PICKS_NONE
    if reason in _BEST_EFFORT_REASONS:
        return MIN_PICKS_GRID
    if reason is DegradedReason.GENERATIVE_SCHEMA_FAIL:
        return MIN_PICKS_GRID
    return MIN_PICKS_GRID


def sanitize_picks(
    model_ids: Sequence[str],
    candidates: Sequence[Candidate],
    entitled: Set[str],
    min_picks: int,
    max_picks: int = MAX_PICKS_DEFAULT,
) -> tuple[str, ...]:
    """Allowlist intersection + optional ranker pad. Never invents an ID.

    Signature is the design.md contract. Pad only when `min_picks > 0`.
    Non-playable / unknown IDs are dropped, never substituted with a lookalike.
    """
    if max_picks <= 0:
        return ()
    candidate_ids = tuple(c.catalog_id for c in candidates)
    in_candidates = set(candidate_ids)
    allow: list[str] = []
    seen: set[str] = set()
    for catalog_id in model_ids:
        if catalog_id in seen:
            continue
        if catalog_id in in_candidates and catalog_id in entitled:
            allow.append(catalog_id)
            seen.add(catalog_id)
            if len(allow) >= max_picks:
                return tuple(allow)
    if min_picks > 0 and len(allow) < min_picks:
        for catalog_id in candidate_ids:
            if catalog_id in entitled and catalog_id not in seen:
                allow.append(catalog_id)
                seen.add(catalog_id)
                if len(allow) >= min_picks:
                    break
    return tuple(allow[:max_picks])
