"""Pure routing predicates. No I/O, no model, no `assist.llm`."""

from __future__ import annotations

from typing import Literal

from assist.config import settings
from assist.domain.enums import DegradedReason, Route
from assist.graph.state import PersonSoft, TurnState

AfterGuard = Literal["intent", "refusal"]
AfterMerge = Literal["resolve_people", "retrieve"]
AfterRetrieve = Literal["broaden", "rank"]
RouteReply = Literal["template", "generative", "clarify", "refusal"]


def after_guard(state: TurnState) -> AfterGuard:
    """Blocked turns skip intent/retrieve and go straight to the refusal reply."""
    if state.get("safety_blocked"):
        return "refusal"
    if state.get("degraded_reason") is DegradedReason.SAFETY_BLOCK:
        return "refusal"
    if state.get("route") is Route.SAFETY:
        return "refusal"
    return "intent"


def has_person_hint(state: TurnState) -> bool:
    """True when a person path should run. Does not inspect `media_type`."""
    if state.get("person_mentions"):
        return True
    soft = state.get("person_soft")
    if isinstance(soft, PersonSoft) and (
        soft.role is not None
        or soft.era_year_min is not None
        or soft.era_year_max is not None
        or (soft.free_hint is not None and soft.free_hint != "")
    ):
        return True
    constraints = state.get("constraints")
    if constraints is not None and constraints.people_include:
        return True
    return False


def after_merge(state: TurnState) -> AfterMerge:
    if has_person_hint(state):
        return "resolve_people"
    return "retrieve"


def after_retrieve(state: TurnState) -> AfterRetrieve:
    """The one bounded cycle: retrieve → broaden → retrieve, then rank.

    The cap is graph-owned. A node cannot raise it above `settings.retrieve_max_attempts`.
    """
    attempts = int(state.get("retrieve_attempts") or 0)
    configured = int(settings.retrieve_max_attempts)
    requested = int(state.get("retrieve_max_attempts") or configured)
    cap = min(requested, configured)
    if cap < 1:
        cap = configured
    candidates = state.get("candidates") or ()
    if not candidates and attempts < cap:
        return "broaden"
    return "rank"


def route_reply(state: TurnState) -> RouteReply:
    """First-cut router. T22 replaces this with the full design.md table."""
    if state.get("safety_blocked") or state.get("route") is Route.SAFETY:
        return "refusal"
    if state.get("degraded_reason") is DegradedReason.SAFETY_BLOCK:
        return "refusal"
    if state.get("person_ambiguous") or state.get("route") is Route.CLARIFY:
        return "clarify"
    if state.get("route") is Route.GENERATIVE:
        return "generative"
    return "template"
