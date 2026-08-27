"""Availability node: drop every candidate that is not playable now.

playable_now runs on every response, including an empty candidate list,
so the AuthZ ctx is always the one the validator sees. Failures fail
closed. A dropped title is never replaced with a different one.
"""

from __future__ import annotations

from collections.abc import Sequence

from assist.domain.catalog import Candidate
from assist.domain.context import ServerUserCtx
from assist.graph.state import TurnState
from assist.obs.logging import get_logger
from assist.stores.catalog_client import CatalogClient

log = get_logger("assist.nodes.availability")


async def playable_now(
    catalog_ids: Sequence[str],
    ctx: ServerUserCtx,
    client: CatalogClient,
) -> dict[str, bool]:
    """Fail-closed wrapper around CatalogClient.playable_now.

    Errors and missing keys are not playable. The client receives `ctx`
    and nothing from client_hints.
    """
    try:
        raw = await client.playable_now(catalog_ids, ctx)
    except Exception:
        log.exception("playable_now_failed_closed", n=len(catalog_ids))
        return dict.fromkeys(catalog_ids, False)
    flags: dict[str, bool] = {}
    for catalog_id in catalog_ids:
        flags[catalog_id] = raw.get(catalog_id) is True
    return flags


def _candidates_of(state: TurnState) -> tuple[Candidate, ...]:
    raw = state.get("candidates") or ()
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        return ()
    return tuple(item for item in raw if isinstance(item, Candidate))


async def validate_availability(
    state: TurnState,
    *,
    client: CatalogClient | None = None,
) -> dict[str, object]:
    """Keep playable candidates, in rank order. Never invent a replacement."""
    ctx = state.get("ctx")
    candidates = _candidates_of(state)
    ids = tuple(c.catalog_id for c in candidates)

    if not isinstance(ctx, ServerUserCtx) or client is None:
        log.info(
            "validate_availability_failed_closed",
            has_ctx=isinstance(ctx, ServerUserCtx),
            has_client=client is not None,
            n=len(candidates),
        )
        return {"candidates": (), "entitled_ids": ()}

    flags = await playable_now(ids, ctx, client)

    kept: list[Candidate] = []
    entitled: list[str] = []
    seen: set[str] = set()
    dropped = 0
    for candidate in candidates:
        catalog_id = candidate.catalog_id
        if flags.get(catalog_id) is True and catalog_id not in seen:
            kept.append(candidate)
            entitled.append(catalog_id)
            seen.add(catalog_id)
        else:
            dropped += 1

    log.info(
        "validate_availability",
        kept=len(kept),
        dropped=dropped,
        geo=ctx.geo,
        package=ctx.package.value,
        device_class=ctx.device_class.value,
        maturity_max=ctx.maturity_max.value,
        kids_flag=ctx.kids_flag,
    )
    return {
        "candidates": tuple(kept),
        "entitled_ids": tuple(entitled),
    }
