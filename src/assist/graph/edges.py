"""Pure routing predicates. No I/O, no model, no `assist.llm`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from assist.config import settings
from assist.domain.enums import DegradedReason, Route
from assist.graph.state import PersonSoft, TurnState
from assist.obs.logging import get_logger

AfterGuard = Literal["intent", "refusal"]
AfterMerge = Literal["resolve_people", "retrieve"]
AfterRetrieve = Literal["broaden", "rank"]
RouteReply = Literal["template", "generative", "clarify", "refusal"]

log = get_logger("assist.graph.edges")

# Closed-class names from design.md Router v0. String literals so this
# module never imports `assist.nodes.intent` (that module imports the LLM).
_RULES_TEMPLATE_CLASSES = frozenset(
    {
        "known_title_lookup",
        "pure_genre_facet",
        "pure_decade",
        "duration_only",
    }
)
_MOOD_GENRE_SET = frozenset({"mood_genre"})
_SHED_TO_TEMPLATE = frozenset(
    {
        DegradedReason.GENERATIVE_TIMEOUT,
        DegradedReason.GENERATIVE_SCHEMA_FAIL,
        DegradedReason.PROVIDER_THROTTLE,
        DegradedReason.RETRIEVAL_UNAVAILABLE,
        DegradedReason.SESSION_STORE_UNAVAILABLE,
        DegradedReason.EMPTY_CATALOG_MATCH,
        DegradedReason.HARD_TIMEOUT,
    }
)


@dataclass(frozen=True)
class RouterDecision:
    """Inspectable router result. `route_reply` returns only `next_node`."""

    next_node: RouteReply
    reason: str
    top1: float | None
    gap: float | None
    theta1: float
    theta_gap: float
    high_conf: bool


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
    if candidates:
        return "rank"
    # exclude_exhausted (T35): empty because MORE_RESULTS exclusion consumed
    # everything the filter matches, not because the filter itself is empty.
    # The user chose this filter explicitly -- broadening it here would be
    # the exact silent constraint-drop the ladder exists to avoid elsewhere.
    if state.get("exclude_exhausted"):
        return "rank"
    if attempts < cap:
        return "broaden"
    return "rank"


def decide_route(
    state: TurnState,
    *,
    theta1: float | None = None,
    theta_gap: float | None = None,
) -> RouterDecision:
    """Router v0 from design.md. Pure: no I/O, no model.

    Thresholds default to `settings.router_theta1` / `settings.router_theta_gap`.
    Person-ambiguous is checked before high-confidence so a mood/genre score
    cannot skip T18's clarify chips. Chip is checked after that so a leftover
    `person_ambiguous` flag still clarify-routes; a resolved chip turn clears
    the flag in the people node.
    """
    t1 = settings.router_theta1 if theta1 is None else theta1
    tgap = settings.router_theta_gap if theta_gap is None else theta_gap
    top1 = _as_score(state.get("top1"))
    gap = _as_score(state.get("gap"))
    intent_class = _intent_class_of(state)
    high_conf = (
        top1 is not None
        and gap is not None
        and top1 >= t1
        and gap >= tgap
        and intent_class in _MOOD_GENRE_SET
    )

    def _out(next_node: RouteReply, reason: str) -> RouterDecision:
        return RouterDecision(
            next_node=next_node,
            reason=reason,
            top1=top1,
            gap=gap,
            theta1=t1,
            theta_gap=tgap,
            high_conf=high_conf,
        )

    if _is_safety(state):
        return _out("refusal", "safety")
    if state.get("person_ambiguous") or state.get("route") is Route.CLARIFY:
        return _out("clarify", "person_ambiguous")
    if _is_chip(state):
        return _out("template", "chip")
    if intent_class in _RULES_TEMPLATE_CLASSES:
        return _out("template", "rules_closed_class")
    if high_conf:
        return _out("template", "high_confidence")
    if _is_shed_to_template(state):
        return _out("template", "shed")
    # Honour a pre-set GENERATIVE flag before the empty-catalog shed so the
    # T09 stub (empty candidates + route=GENERATIVE) still reaches reply_generative.
    if state.get("route") is Route.GENERATIVE:
        return _out("generative", "pre_set_generative")
    if not (state.get("candidates") or ()):
        return _out("template", "empty_catalog")
    if intent_class:
        return _out("generative", "free_text")
    return _out("template", "default_template")


def route_reply(state: TurnState) -> RouteReply:
    """Conditional-edge adapter. Decision is pure; the log line is observability."""
    decision = decide_route(state)
    log.info(
        "route_reply",
        next_node=decision.next_node,
        reason=decision.reason,
        top1=decision.top1,
        gap=decision.gap,
        theta1=decision.theta1,
        theta_gap=decision.theta_gap,
        high_conf=decision.high_conf,
        intent_class=_intent_class_of(state) or None,
        intent_source=state.get("intent_source"),
        message_type=state.get("message_type"),
    )
    return decision.next_node


def _is_safety(state: TurnState) -> bool:
    if state.get("safety_blocked"):
        return True
    if state.get("route") is Route.SAFETY:
        return True
    return state.get("degraded_reason") is DegradedReason.SAFETY_BLOCK


def _is_chip(state: TurnState) -> bool:
    if state.get("message_type") == "chip":
        return True
    return state.get("intent_source") == "chip"


def _is_shed_to_template(state: TurnState) -> bool:
    if state.get("route") is Route.DEGRADED_KEYWORD:
        return True
    return state.get("degraded_reason") in _SHED_TO_TEMPLATE


def _as_score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _intent_class_of(state: TurnState) -> str:
    raw = state.get("intent_class")
    if raw is None:
        return ""
    text = str(getattr(raw, "value", raw)).strip()
    return text
