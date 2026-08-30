"""TurnState — the turn's state. No parallel bookkeeping object."""

from __future__ import annotations

from typing import Literal, TypedDict, cast

from pydantic import BaseModel, ConfigDict

from assist.config import settings
from assist.domain.catalog import Candidate, Person, Pick
from assist.domain.constraints import ConstraintDelta, ConstraintState
from assist.domain.context import ServerUserCtx
from assist.domain.enums import DegradedReason, Route, SpeechAct


class ReplyChip(BaseModel):
    """Client-facing chip. The delta stays on the server (T21)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    label: str


class PersonSoft(BaseModel):
    """Soft person descriptor from intent. Person IDs come only from the index."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str | None = None
    era_year_min: int | None = None
    era_year_max: int | None = None
    free_hint: str | None = None


class TurnState(TypedDict, total=False):
    """Per-turn graph state. `ctx` is an immutable input; nodes must not widen it."""

    ctx: ServerUserCtx
    session_id: str
    trace_id: str
    text: str
    chip_id: str | None
    message_type: Literal["text", "chip"]

    constraints: ConstraintState
    delta: ConstraintDelta | None
    turn_count: int
    # "Show me more" (T35): titles already shown across this session's history
    # (`TurnSummary.pick_ids`, unioned in `_make_load_session`), the flag a
    # MORE_RESULTS chip tap sets to ask retrieval to exclude them, and the
    # signal retrieval sends back when exclusion alone emptied the pool --
    # exhaustion, not a genuinely empty filter, so it must not enter broaden.
    seen_catalog_ids: tuple[str, ...]
    exclude_seen: bool
    exclude_exhausted: bool

    intent_source: Literal["chip", "rules", "llm"] | None
    intent_class: str | None
    query_rewrite: str
    person_mentions: tuple[str, ...]
    person_soft: PersonSoft | None
    person_ambiguous: bool
    people_candidates: tuple[Person, ...]

    candidates: tuple[Candidate, ...]
    entitled_ids: tuple[str, ...]
    retrieve_attempts: int
    retrieve_max_attempts: int
    top1: float | None
    gap: float | None

    route: Route | None
    degraded_reason: DegradedReason
    safety_blocked: bool
    reply: str
    min_picks: int
    model_pick_indices: tuple[int, ...]
    model_pick_ids: tuple[str, ...]
    picks: tuple[Pick, ...]
    chip_speech_acts: tuple[SpeechAct, ...]
    chips: tuple[ReplyChip, ...]

    timings: dict[str, int]
    tokens_in: int
    tokens_out: int
    cost_usd: float


def empty_turn_state(ctx: ServerUserCtx, **overrides: object) -> TurnState:
    """Build a complete TurnState. Callers override only the keys they care about.

    `constraints` starts as `ConstraintState.empty()` (`media_type=None`). That is
    one of two valid 'no media filter' sentinels; `reset_soft` uses `MediaType.ANY`.
    """
    chip_id = overrides.get("chip_id", None)
    message_type: Literal["text", "chip"]
    if "message_type" in overrides:
        message_type = cast(Literal["text", "chip"], overrides["message_type"])
    elif chip_id:
        message_type = "chip"
    else:
        message_type = "text"

    state: dict[str, object] = {
        "ctx": ctx,
        "session_id": "",
        "trace_id": "-",
        "text": "",
        "chip_id": None,
        "message_type": message_type,
        "constraints": ConstraintState.empty(),
        "delta": None,
        "turn_count": 0,
        "seen_catalog_ids": (),
        "exclude_seen": False,
        "exclude_exhausted": False,
        "intent_source": None,
        "intent_class": None,
        "query_rewrite": "",
        "person_mentions": (),
        "person_soft": None,
        "person_ambiguous": False,
        "people_candidates": (),
        "candidates": (),
        "entitled_ids": (),
        "retrieve_attempts": 0,
        "retrieve_max_attempts": settings.retrieve_max_attempts,
        "top1": None,
        "gap": None,
        "route": None,
        "degraded_reason": DegradedReason.NONE,
        "safety_blocked": False,
        "reply": "",
        "min_picks": 0,
        "model_pick_indices": (),
        "model_pick_ids": (),
        "picks": (),
        "chip_speech_acts": (),
        "chips": (),
        "timings": {},
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0.0,
    }
    state.update(overrides)
    if "message_type" not in overrides and state.get("chip_id"):
        state["message_type"] = "chip"
    return cast(TurnState, state)
