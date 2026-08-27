"""Merge node: apply this turn's deltas onto prior constraints.

`domain.constraints.merge` takes one delta per call, so call order is
precedence. Text/rules first, then chip — a chip tap wins a same-field
conflict. Every call passes `ServerUserCtx.maturity_max`; a 3-arg merge
does not clamp, and the delta is not trusted.

Effective maturity is min(profile ceiling, requested_stricter) and is
computed here. Hard AuthZ fields live on `ctx` and never enter a delta.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from assist.domain.constraints import ConstraintDelta, ConstraintState
from assist.domain.constraints import merge as merge_delta
from assist.domain.context import ServerUserCtx
from assist.domain.enums import DeltaSource, MaturityRating, is_stricter_than
from assist.graph.state import TurnState
from assist.obs.logging import get_logger

log = get_logger(__name__)

# Dual-source turns (text + chip in one request) stash the chip here until
# T24 can add a typed TurnState key. T15 owns the lookup; we only apply.
CHIP_DELTA_KEY = "chip_delta"


def effective_maturity(ctx: ServerUserCtx, constraints: ConstraintState) -> MaturityRating:
    """Profile ceiling, optionally pulled down by a stricter user request."""
    requested = constraints.maturity_request_stricter
    if requested is not None and is_stricter_than(requested, ctx.maturity_max):
        return requested
    return ctx.maturity_max


def apply_turn_deltas(
    prior: ConstraintState,
    profile_maturity_max: MaturityRating,
    steps: Sequence[tuple[ConstraintDelta, DeltaSource]],
) -> ConstraintState:
    """Apply `steps` in order. Each `merge` receives the profile ceiling."""
    current = prior
    for delta, source in steps:
        current = merge_delta(
            current,
            delta,
            source,
            profile_maturity_max=profile_maturity_max,
        )
        current = _clamp_maturity(current, profile_maturity_max)
    return current


def _clamp_maturity(constraints: ConstraintState, ceiling: MaturityRating) -> ConstraintState:
    """Drop a request that is not strictly below the server profile ceiling."""
    requested = constraints.maturity_request_stricter
    if requested is None or is_stricter_than(requested, ceiling):
        return constraints
    return constraints.model_copy(update={"maturity_request_stricter": None})


def _as_delta(value: object) -> ConstraintDelta | None:
    return value if isinstance(value, ConstraintDelta) else None


def _text_source(intent_source: str | None) -> DeltaSource:
    if intent_source == "rules":
        return DeltaSource.RULES
    return DeltaSource.TEXT


def _steps_from_state(state: TurnState) -> list[tuple[ConstraintDelta, DeltaSource]]:
    """Build the ordered (delta, source) list. Text/rules before chip.

    Unknown `intent_source` is treated as text so a missing label cannot
    steal chip precedence. Dual-source turns pass the chip via
    `CHIP_DELTA_KEY`; a chip-only turn uses `delta` + `intent_source=chip`.
    """
    delta = _as_delta(state.get("delta"))
    extra_chip = _as_delta(cast(Mapping[str, object], state).get(CHIP_DELTA_KEY))
    intent_source = state.get("intent_source")

    steps: list[tuple[ConstraintDelta, DeltaSource]] = []
    if extra_chip is not None:
        if delta is not None:
            steps.append((delta, _text_source(intent_source)))
        steps.append((extra_chip, DeltaSource.CHIP))
        return steps

    if delta is None:
        return steps
    if intent_source == "chip":
        steps.append((delta, DeltaSource.CHIP))
    else:
        steps.append((delta, _text_source(intent_source)))
    return steps


async def merge_constraints(state: TurnState) -> dict[str, object]:
    """LangGraph node. Returns only `constraints`; `ctx` is never rewritten."""
    ctx = state.get("ctx")
    if not isinstance(ctx, ServerUserCtx):
        raise TypeError("merge_constraints requires ServerUserCtx on state")

    prior = state.get("constraints")
    if not isinstance(prior, ConstraintState):
        prior = ConstraintState.empty()

    ceiling = ctx.maturity_max
    steps = _steps_from_state(state)
    if not steps:
        merged = prior
    else:
        merged = apply_turn_deltas(prior, ceiling, steps)

    log.info(
        "merge_constraints",
        intent_source=state.get("intent_source"),
        sources=[source.value for _, source in steps],
        media_type=merged.media_type.value if merged.media_type is not None else None,
        maturity_request=(
            merged.maturity_request_stricter.value
            if merged.maturity_request_stricter is not None
            else None
        ),
        effective_maturity=effective_maturity(ctx, merged).value,
    )
    return {"constraints": merged}
