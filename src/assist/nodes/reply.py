"""Generative reply node: one structured call over numbered candidate cards.

The model returns indices, never catalog_id or person_id. Chip labels come
from the phrase bank, not free text. Schema failure has zero retries and
routes to the template path with `degraded_reason=generative_schema_fail`.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig, RunnableLambda
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from assist.config import Settings
from assist.config import settings as default_settings
from assist.domain.catalog import Candidate
from assist.domain.enums import DegradedReason, Route, SpeechAct
from assist.domain.picks import min_picks_for
from assist.graph.state import TurnState
from assist.llm.cost import CostCallbackHandler
from assist.llm.gateway import LLMError, LLMSchemaError, LLMTimeout, structured_output
from assist.llm.prompts import chat_prompt_template
from assist.nodes.templates import PhraseBank, load_phrase_bank, reply_template
from assist.obs.logging import get_logger

log = get_logger("assist.nodes.reply")

_SPEECH_ACTS = ", ".join(act.value for act in SpeechAct)


def _coerce_indices(value: object) -> tuple[int, ...]:
    """Keep raw ints, including out-of-range. Sanitize drops those later."""
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        return ()
    out: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            continue
        out.append(item)
    return tuple(out)


def _coerce_speech_acts(value: object) -> tuple[SpeechAct, ...]:
    """Keep known SpeechAct values only. Unknown labels never become chip copy."""
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        return ()
    acts: list[SpeechAct] = []
    seen: set[SpeechAct] = set()
    for item in value:
        raw = item.value if isinstance(item, SpeechAct) else item
        try:
            act = SpeechAct(str(raw))
        except ValueError:
            continue
        if act not in seen:
            seen.add(act)
            acts.append(act)
    return tuple(acts)


class GroundedReply(BaseModel):
    """Structured generative output. Indices only; no catalog or person ids."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reply: str = Field(
        default="",
        description="Short reply. Name only titles that appear on the numbered list.",
    )
    pick_indices: Annotated[tuple[int, ...], BeforeValidator(_coerce_indices)] = Field(
        default=(),
        description="0-based indices into the numbered candidate list. Never catalog_id.",
    )
    chip_speech_acts: Annotated[tuple[SpeechAct, ...], BeforeValidator(_coerce_speech_acts)] = (
        Field(
            default=(),
            description="Known SpeechAct values only. Do not write chip labels.",
        )
    )


def format_candidate_cards(candidates: Sequence[Candidate]) -> str:
    """Numbered compact cards. Rank order. Never includes catalog_id."""
    if not candidates:
        return "(none)"
    lines: list[str] = []
    for index, card in enumerate(candidates):
        title = " ".join((card.title or "").split()) or "(untitled)"
        year = str(card.release_year) if card.release_year is not None else "?"
        genres = ", ".join(genre.value for genre in card.genres[:3]) or "untagged"
        lines.append(f"[{index}] {title} ({year}) · {card.media_type.value} · {genres}")
    return "\n".join(lines)


def _cfg(settings: Settings | None) -> Settings:
    return settings if settings is not None else default_settings


def _candidates_of(state: TurnState, *, limit: int) -> tuple[Candidate, ...]:
    raw = state.get("candidates") or ()
    if not isinstance(raw, (list, tuple)):
        return ()
    cards = tuple(item for item in raw if isinstance(item, Candidate))
    if limit <= 0:
        return cards
    return cards[:limit]


def _entitled_count(state: TurnState) -> int:
    raw = state.get("entitled_ids")
    if isinstance(raw, (list, tuple, set, frozenset)):
        return len(raw)
    return 0


def _truncate(reply: str, max_chars: int) -> str:
    if max_chars <= 0 or len(reply) <= max_chars:
        return reply
    cut = reply[:max_chars]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip()


def _filter_acts(acts: Sequence[SpeechAct], bank: PhraseBank) -> tuple[SpeechAct, ...]:
    allowed = bank.speech_acts()
    if not allowed:
        return ()
    return tuple(act for act in acts if act in allowed)


def _prompt_payload(
    state: TurnState,
    cards: Sequence[Candidate],
    *,
    max_chars: int,
) -> dict[str, str]:
    n = len(cards)
    return {
        "text": state.get("text") or "",
        "candidates": format_candidate_cards(cards),
        "speech_acts": _SPEECH_ACTS,
        "n": str(n),
        "n_last": str(n - 1 if n else 0),
        "max_chars": str(max_chars),
    }


def _stamp_timings(state: TurnState, t0: float) -> dict[str, int]:
    timings = dict(state.get("timings") or {})
    timings["reply"] = int((time.perf_counter() - t0) * 1000)
    return timings


async def _degrade_to_template(
    state: TurnState,
    reason: DegradedReason,
    *,
    bank: PhraseBank,
    settings: Settings,
    t0: float,
) -> dict[str, object]:
    """Template + ranker pad. min_picks stays 3 when entitled candidates exist."""
    degraded: TurnState = {**state, "degraded_reason": reason}
    out = await reply_template(degraded, bank=bank, settings=settings)
    out["degraded_reason"] = reason
    out["model_pick_indices"] = ()
    out["model_pick_ids"] = ()
    out["timings"] = _stamp_timings(state, t0)
    log.info(
        "reply_generative_degraded",
        degraded_reason=reason.value,
        route=getattr(out.get("route"), "value", out.get("route")),
        reply_len=len(str(out.get("reply") or "")),
        n_candidates=len(_candidates_of(state, limit=settings.retrieve_size)),
    )
    return out


async def reply_generative(
    state: TurnState,
    *,
    bank: PhraseBank | None = None,
    model: BaseChatModel | None = None,
    cost_handler: CostCallbackHandler | None = None,
    settings: Settings | None = None,
) -> dict[str, object]:
    """One structured Haiku call. Schema miss → template, never a retry."""
    cfg = _cfg(settings)
    t0 = time.perf_counter()
    loaded = bank if bank is not None else load_phrase_bank()
    cards = _candidates_of(state, limit=cfg.retrieve_size)
    if not cards:
        return await _degrade_to_template(
            state,
            DegradedReason.EMPTY_CATALOG_MATCH,
            bank=loaded,
            settings=cfg,
            t0=t0,
        )
    try:
        return await _from_llm(
            state,
            cards,
            bank=loaded,
            model=model,
            cost_handler=cost_handler,
            settings=cfg,
            t0=t0,
        )
    except Exception:
        log.exception("reply_generative_failed")
        return await _degrade_to_template(
            state,
            DegradedReason.GENERATIVE_SCHEMA_FAIL,
            bank=loaded,
            settings=cfg,
            t0=t0,
        )


async def _from_llm(
    state: TurnState,
    cards: tuple[Candidate, ...],
    *,
    bank: PhraseBank,
    model: BaseChatModel | None,
    cost_handler: CostCallbackHandler | None,
    settings: Settings,
    t0: float,
) -> dict[str, object]:
    fallback_used = False

    def _fallback(_input: Any) -> GroundedReply:
        nonlocal fallback_used
        fallback_used = True
        return GroundedReply()

    handler = cost_handler if cost_handler is not None else CostCallbackHandler()
    # Gateway helper, not ChatAnthropic.with_structured_output: default method is
    # function_calling and would bind tools (invariant 1). retries=0 is the
    # design.md MVP rule — schema miss goes to template, not a second call.
    chain = chat_prompt_template("reply") | structured_output(
        GroundedReply,
        retries=0,
        fallback=RunnableLambda(_fallback),
        model=model,
        settings=settings,
    )
    config: RunnableConfig = {"callbacks": [handler]}
    try:
        raw = await chain.ainvoke(
            _prompt_payload(state, cards, max_chars=settings.reply_max_chars),
            config=config,
        )
    except LLMTimeout:
        log.info("reply_generative_failed", reason="timeout")
        return await _degrade_to_template(
            state,
            DegradedReason.GENERATIVE_TIMEOUT,
            bank=bank,
            settings=settings,
            t0=t0,
        )
    except LLMSchemaError:
        log.info("reply_generative_failed", reason="schema")
        return await _degrade_to_template(
            state,
            DegradedReason.GENERATIVE_SCHEMA_FAIL,
            bank=bank,
            settings=settings,
            t0=t0,
        )
    except LLMError:
        log.info("reply_generative_failed", reason="gateway")
        return await _degrade_to_template(
            state,
            DegradedReason.GENERATIVE_SCHEMA_FAIL,
            bank=bank,
            settings=settings,
            t0=t0,
        )

    if fallback_used or not isinstance(raw, GroundedReply):
        log.info("reply_generative_failed", reason="schema_fallback")
        return await _degrade_to_template(
            state,
            DegradedReason.GENERATIVE_SCHEMA_FAIL,
            bank=bank,
            settings=settings,
            t0=t0,
        )

    reply = _truncate(raw.reply.strip(), settings.reply_max_chars)
    acts = _filter_acts(raw.chip_speech_acts, bank)
    min_picks = min_picks_for(
        route=Route.GENERATIVE,
        degraded_reason=DegradedReason.NONE,
        entitled_count=_entitled_count(state),
    )
    log.info(
        "reply_generative",
        route=Route.GENERATIVE.value,
        reply_len=len(reply),
        n_candidates=len(cards),
        n_indices=len(raw.pick_indices),
        min_picks=min_picks,
    )
    return {
        "reply": reply,
        "route": Route.GENERATIVE,
        "degraded_reason": DegradedReason.NONE,
        "min_picks": min_picks,
        "model_pick_indices": raw.pick_indices,
        "model_pick_ids": (),
        "chip_speech_acts": acts,
        "tokens_in": handler.tokens_in,
        "tokens_out": handler.tokens_out,
        "cost_usd": handler.cost_usd,
        "timings": _stamp_timings(state, t0),
    }


def make_reply_node(
    *,
    bank: PhraseBank | None = None,
    model: BaseChatModel | None = None,
    cost_handler: CostCallbackHandler | None = None,
    settings: Settings | None = None,
) -> Callable[[TurnState], Awaitable[dict[str, object]]]:
    """Bind I/O for the graph. T24 wires the real model and phrase bank."""

    async def _node(state: TurnState) -> dict[str, object]:
        return await reply_generative(
            state,
            bank=bank,
            model=model,
            cost_handler=cost_handler,
            settings=settings,
        )

    return _node
