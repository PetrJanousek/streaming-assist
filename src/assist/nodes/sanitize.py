"""Sanitize node: allowlist picks, then ground reply prose.

Wraps `domain.picks.sanitize_picks` — do not reimplement the allowlist.
The model may only select from candidates; unentitled IDs are dropped, never
substituted. Title-like spans in reply text that are not a candidate/pick
title are stripped, or the reply falls back to a template. No second LLM.
"""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable, Sequence, Set
from typing import cast

from assist.config import Settings
from assist.config import settings as default_settings
from assist.domain.catalog import Candidate, Pick
from assist.domain.enums import DegradedReason, Route
from assist.domain.picks import min_picks_for, sanitize_picks
from assist.graph.state import TurnState
from assist.obs.logging import get_logger

log = get_logger(__name__)

# Generic template names no catalog title. T22 owns the phrase bank; this is
# the fail-closed stand-in so a span miss never ships an off-catalog name.
FALLBACK_REPLY = "Here are a few titles that match."
# After stripping one off-catalog span, leftover shorter than this is not a reply.
_MIN_STRIPPED_CHARS = 24

_QUOTED_RE = re.compile(r'["“”]([^"“”]{2,80})["“”]')
# Capitalized multiword: "The Dark Knight", not a full sentence in title case.
_TITLE_CASE_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9'&:.-]*(?:\s+[A-Z][A-Za-z0-9'&:.-]*)+)\b")


def _candidates_of(state: TurnState) -> tuple[Candidate, ...]:
    raw = state.get("candidates") or ()
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(item for item in raw if isinstance(item, Candidate))


def _entitled_of(state: TurnState) -> frozenset[str]:
    # Missing/unreadable entitled set fails closed — never treat as "all playable".
    raw = state.get("entitled_ids")
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(str(item) for item in raw)


def _as_index(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def model_ids_from_state(state: TurnState, candidates: Sequence[Candidate]) -> tuple[str, ...]:
    """Resolve model selections to catalog ids. Out-of-range indices are dropped."""
    raw_ids = state.get("model_pick_ids") or ()
    if isinstance(raw_ids, (list, tuple)) and raw_ids:
        return tuple(str(item) for item in raw_ids if item)

    raw_indices = state.get("model_pick_indices") or ()
    if not isinstance(raw_indices, (list, tuple)):
        return ()
    n = len(candidates)
    resolved: list[str] = []
    for raw in raw_indices:
        idx = _as_index(raw)
        if idx is None or idx < 0 or idx >= n:
            continue
        resolved.append(candidates[idx].catalog_id)
    return tuple(resolved)


def _allowed_titles(candidates: Sequence[Candidate], pick_ids: Sequence[str]) -> frozenset[str]:
    allowed = {c.title.casefold() for c in candidates if c.title}
    by_id = {c.catalog_id: c for c in candidates}
    for catalog_id in pick_ids:
        candidate = by_id.get(catalog_id)
        if candidate is not None and candidate.title:
            allowed.add(candidate.title.casefold())
    return frozenset(allowed)


def _overlaps(start: int, end: int, ranges: Sequence[tuple[int, int]]) -> bool:
    return any(not (end <= r0 or start >= r1) for r0, r1 in ranges)


def _off_catalog_ranges(reply: str, allowed: Set[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for match in _QUOTED_RE.finditer(reply):
        inner = match.group(1)
        if inner and inner.casefold() not in allowed:
            ranges.append((match.start(), match.end()))
    for match in _TITLE_CASE_RE.finditer(reply):
        span = match.group(0)
        if span.casefold() in allowed:
            continue
        if _overlaps(match.start(), match.end(), ranges):
            continue
        ranges.append((match.start(), match.end()))
    ranges.sort()
    return ranges


def _strip_ranges(text: str, ranges: Sequence[tuple[int, int]]) -> str:
    parts: list[str] = []
    pos = 0
    for start, end in ranges:
        parts.append(text[pos:start])
        pos = end
    parts.append(text[pos:])
    out = "".join(parts)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return out.strip()


def ground_reply(
    reply: str,
    allowed_titles: Set[str],
    *,
    fallback: str = FALLBACK_REPLY,
    min_keep: int = _MIN_STRIPPED_CHARS,
) -> tuple[str, str]:
    """Return (reply, action). action is keep | strip | template.

    One off-catalog span with enough leftover prose is stripped. Several spans,
    or a leftover that is too short, fall back to `fallback`. Never ships a
    free-named title that is not in `allowed_titles`.
    """
    if not reply:
        return reply, "keep"
    ranges = _off_catalog_ranges(reply, allowed_titles)
    if not ranges:
        return reply, "keep"
    if len(ranges) > 1:
        return fallback, "template"
    stripped = _strip_ranges(reply, ranges)
    if len(stripped) < min_keep:
        return fallback, "template"
    return stripped, "strip"


def _truncate(reply: str, max_chars: int) -> str:
    if max_chars <= 0 or len(reply) <= max_chars:
        return reply
    cut = reply[:max_chars]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip()


def _route_of(state: TurnState) -> Route | None:
    route = state.get("route")
    return route if isinstance(route, Route) else None


def _reason_of(state: TurnState) -> DegradedReason:
    reason = state.get("degraded_reason")
    return reason if isinstance(reason, DegradedReason) else DegradedReason.NONE


async def sanitize(
    state: TurnState,
    *,
    settings: Settings | None = None,
    fallback_reply: str | None = None,
) -> dict[str, object]:
    """LangGraph node. Writes `picks` and a grounded `reply`. Never calls a model."""
    t0 = time.perf_counter()
    cfg = settings if settings is not None else default_settings
    fallback = fallback_reply if fallback_reply is not None else FALLBACK_REPLY

    candidates = _candidates_of(state)
    entitled = _entitled_of(state)
    route = _route_of(state)
    reason = _reason_of(state)
    min_picks = min_picks_for(route=route, degraded_reason=reason, entitled_count=len(entitled))
    model_ids = model_ids_from_state(state, candidates)
    allowed_ids = sanitize_picks(
        model_ids,
        candidates,
        cast(Set[str], entitled),
        min_picks,
    )
    picks = tuple(Pick(catalog_id=catalog_id) for catalog_id in allowed_ids)

    reply = str(state.get("reply") or "")
    grounded, action = ground_reply(
        reply, _allowed_titles(candidates, allowed_ids), fallback=fallback
    )
    if action != "keep":
        log.info(
            "reply_title_span",
            action=action,
            reply_len=len(reply),
            grounded_len=len(grounded),
        )
    grounded = _truncate(grounded, cfg.reply_max_chars)

    timings = dict(state.get("timings") or {})
    timings["sanitize"] = int((time.perf_counter() - t0) * 1000)
    return {
        "picks": picks,
        "reply": grounded,
        "min_picks": min_picks,
        "timings": timings,
    }


def make_sanitize_node(
    *,
    settings: Settings | None = None,
    fallback_reply: str | None = None,
) -> Callable[[TurnState], Awaitable[dict[str, object]]]:
    """Bind config for the graph. T24 wires this over the passthrough stub."""

    async def _node(state: TurnState) -> dict[str, object]:
        return await sanitize(state, settings=settings, fallback_reply=fallback_reply)

    return _node
