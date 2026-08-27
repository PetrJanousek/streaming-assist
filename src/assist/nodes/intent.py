"""Intent node: chip lookup, closed-class rules, else one structured LLM call.

Three sources converge on one ConstraintDelta. The model never names a title
and never emits a catalog_id or person_id; person IDs come only from the index.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Annotated, Any, Literal, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig, RunnableLambda
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

from assist.config import Settings
from assist.domain.constraints import (
    AddOp,
    ConstraintDelta,
    ConstraintState,
    SetOp,
)
from assist.domain.enums import GenreId, MediaType, MoodId, SpeechAct
from assist.graph.state import PersonSoft, TurnState
from assist.llm.cost import CostCallbackHandler
from assist.llm.gateway import LLMError, structured_output
from assist.llm.prompts import chat_prompt_template
from assist.obs.logging import get_logger
from assist.stores.cache import constraints_hash
from assist.stores.session import ChipInvalid, ChipRecord

log = get_logger(__name__)

IntentSource = Literal["chip", "rules", "llm"]


class ChipSource(Protocol):
    def lookup_chip(self, chip_id: str) -> ChipRecord: ...


class IntentCache(Protocol):
    async def get_intent(self, norm_text: str, constraints_hash_value: str) -> str | None: ...

    async def set_intent(
        self, norm_text: str, constraints_hash_value: str, payload: str
    ) -> None: ...


class IntentClass(StrEnum):
    """Appendix A plus the router closed-class names in design.md."""

    MOOD_GENRE = "mood_genre"
    PEOPLE_FUZZY = "people_fuzzy"
    KNOWN_ITEM = "known_item"
    KNOWN_TITLE_LOOKUP = "known_title_lookup"
    PURE_GENRE_FACET = "pure_genre_facet"
    PURE_DECADE = "pure_decade"
    DURATION_ONLY = "duration_only"
    DURATION = "duration"
    RESET = "reset"
    MEDIA_TYPE = "media_type"
    OTHER = "other"


def _parse_intent_class(value: object) -> IntentClass:
    if isinstance(value, IntentClass):
        return value
    if isinstance(value, str):
        try:
            return IntentClass(value)
        except ValueError:
            return IntentClass.OTHER
    return IntentClass.OTHER


class IntentUpdate(BaseModel):
    """Structured intent. Titles and ids are forbidden; the node also strips them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent_class: Annotated[IntentClass, BeforeValidator(_parse_intent_class)] = Field(
        default=IntentClass.OTHER,
        description="Closed class of this turn's request.",
    )
    query_rewrite: str = Field(
        default="",
        description="Search string for retrieval. Not a title guessed from memory.",
    )
    constraint_delta: ConstraintDelta = Field(default_factory=ConstraintDelta)
    person_soft: PersonSoft | None = Field(
        default=None,
        description="Soft person descriptor. Role, era, free_hint only. No ids.",
    )
    person_mentions: tuple[str, ...] = Field(
        default=(),
        description="Person display names only. Never person_id.",
    )
    person_ids_from_index: tuple[str, ...] = Field(
        default=(),
        description="Always empty. The server ignores this field. Do not fill it.",
    )


_FILLER = frozenset({"a", "an", "the", "some", "please", "just", "only"})
_FILM_WORDS = frozenset({"movie", "movies", "film", "films"})
_SERIES_WORDS = frozenset({"series", "show", "shows", "tv"})
_RESET_PHRASES = frozenset(
    {
        "reset",
        "start over",
        "start again",
        "start fresh",
        "clear",
        "clear filters",
        "anything",
        "whatever",
        "never mind",
        "nevermind",
        "forget that",
    }
)
_DURATION = re.compile(
    r"^(?:under|less than|at most|maximum|max|up to)\s+(\d+)\s*"
    r"(minutes?|mins?|hours?|hrs?|m|h)$"
)
_DECADE4 = re.compile(r"^(\d{4})s$")
_DECADE2 = re.compile(r"^(\d{2})s$")
_YEAR_RANGE = re.compile(r"^(\d{4})\s+(\d{4})$")
_QUOTED_TITLE = re.compile(
    r"^\s*(?:(?:watch|play|find|stream)\s+)?[\"'“”](.+?)[\"'“”]\s*$",
    re.IGNORECASE | re.DOTALL,
)

_GENRE_EXTRA: dict[str, GenreId] = {
    "comedies": GenreId.COMEDY,
    "comedic": GenreId.COMEDY,
    "sci fi": GenreId.SCIFI,
    "science fiction": GenreId.SCIFI,
    "kids": GenreId.CHILDREN,
    "childrens": GenreId.CHILDREN,
    "children s": GenreId.CHILDREN,
    "docs": GenreId.DOCUMENTARY,
    "documentaries": GenreId.DOCUMENTARY,
    "thrillers": GenreId.THRILLER,
    "musicals": GenreId.MUSIC,
    "musical": GenreId.MUSIC,
    "indie": GenreId.INDEPENDENT,
    "lgbt": GenreId.LGBTQ,
    "standup": GenreId.STAND_UP,
    "stand up comedy": GenreId.STAND_UP,
    "sport": GenreId.SPORTS,
    "dramas": GenreId.DRAMA,
    "romances": GenreId.ROMANCE,
    "classics": GenreId.CLASSIC,
    "teenage": GenreId.TEEN,
    "faith based": GenreId.FAITH,
    "mysteries": GenreId.MYSTERY,
}


def _genre_aliases() -> tuple[tuple[str, GenreId], ...]:
    pairs = [(genre.value.replace("_", " "), genre) for genre in GenreId]
    pairs.extend(_GENRE_EXTRA.items())
    return tuple(sorted(pairs, key=lambda item: len(item[0]), reverse=True))


_GENRE_ALIASES = _genre_aliases()


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse space. Cache key and rules share this."""
    lowered = text.lower().replace("_", " ")
    cleaned = re.sub(r"[^\w\s-]", " ", lowered).replace("-", " ")
    return " ".join(cleaned.split())


def _empty_intent(text: str) -> IntentUpdate:
    return IntentUpdate(intent_class=IntentClass.OTHER, query_rewrite=text.strip())


def _rules_or_empty(text: str) -> IntentUpdate:
    return match_rules(text) or _empty_intent(text)


def _split_media(tokens: list[str]) -> tuple[list[str], MediaType | None]:
    film = any(token in _FILM_WORDS for token in tokens)
    series = any(token in _SERIES_WORDS for token in tokens)
    if film and series:
        return tokens, None
    if film:
        return [token for token in tokens if token not in _FILM_WORDS], MediaType.FILM
    if series:
        return [token for token in tokens if token not in _SERIES_WORDS], MediaType.SERIES
    return tokens, None


def _decade_bounds(remainder: str) -> tuple[int, int] | None:
    match = _DECADE4.fullmatch(remainder)
    if match:
        start = int(match.group(1))
        if 1900 <= start <= 2020 and start % 10 == 0:
            return start, start + 9
        return None
    match = _DECADE2.fullmatch(remainder)
    if match:
        n = int(match.group(1))
        if n == 0:
            start = 2000
        elif n <= 20:
            start = 2000 + n
        else:
            start = 1900 + n
        return start, start + 9
    match = _YEAR_RANGE.fullmatch(remainder)
    if match:
        lo, hi = int(match.group(1)), int(match.group(2))
        if 1800 <= lo <= hi <= 2100:
            return lo, hi
    return None


def _duration_minutes(remainder: str) -> int | None:
    match = _DURATION.fullmatch(remainder)
    if match is None:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    if unit.startswith("h"):
        return amount * 60
    return amount


def _match_genre(remainder: str) -> GenreId | None:
    for alias, genre in _GENRE_ALIASES:
        if remainder == alias:
            return genre
    return None


def _delta_with_media(
    media: MediaType | None,
    **fields: AddOp | SetOp | None,
) -> ConstraintDelta:
    payload: dict[str, object] = {key: value for key, value in fields.items() if value is not None}
    if media is not None:
        payload["media_type"] = SetOp(value=media.value)
    return ConstraintDelta.model_validate(payload)


def match_rules(text: str) -> IntentUpdate | None:
    """Closed-class matcher. None means the turn needs the LLM (or a cache hit)."""
    quoted = _QUOTED_TITLE.fullmatch(text.strip())
    if quoted:
        title = quoted.group(1).strip()
        if title:
            return IntentUpdate(
                intent_class=IntentClass.KNOWN_TITLE_LOOKUP,
                query_rewrite=title,
            )

    raw = normalize_text(text)
    if not raw:
        return None

    tokens = [token for token in raw.split() if token not in _FILLER]
    if not tokens:
        return None
    remainder = " ".join(tokens)
    if remainder in _RESET_PHRASES:
        return IntentUpdate(
            intent_class=IntentClass.RESET,
            query_rewrite="",
            constraint_delta=ConstraintDelta(reset_soft=True),
        )

    rest_tokens, media = _split_media(tokens)
    rest_tokens = [token for token in rest_tokens if token not in _FILLER]
    rest = " ".join(rest_tokens)

    if not rest:
        if media is None:
            return None
        return IntentUpdate(
            intent_class=IntentClass.MEDIA_TYPE,
            query_rewrite=raw,
            constraint_delta=_delta_with_media(media),
        )

    minutes = _duration_minutes(rest)
    if minutes is not None:
        return IntentUpdate(
            intent_class=IntentClass.DURATION_ONLY,
            query_rewrite=raw,
            constraint_delta=_delta_with_media(media, duration_max_min=SetOp(value=minutes)),
        )

    bounds = _decade_bounds(rest)
    if bounds is not None:
        year_min, year_max = bounds
        return IntentUpdate(
            intent_class=IntentClass.PURE_DECADE,
            query_rewrite=raw,
            constraint_delta=_delta_with_media(
                media,
                year_min=SetOp(value=year_min),
                year_max=SetOp(value=year_max),
            ),
        )

    genre = _match_genre(rest)
    if genre is not None:
        return IntentUpdate(
            intent_class=IntentClass.PURE_GENRE_FACET,
            query_rewrite=raw,
            constraint_delta=_delta_with_media(media, genres_include=AddOp(values=(genre.value,))),
        )

    return None


def _person_soft_or_none(soft: PersonSoft | None) -> PersonSoft | None:
    if soft is None:
        return None
    if (
        soft.role is None
        and soft.era_year_min is None
        and soft.era_year_max is None
        and not soft.free_hint
    ):
        return None
    return soft


def _sanitize_model_intent(update: IntentUpdate) -> IntentUpdate:
    """Drop model-emitted person ids. Person IDs come only from the index (T18)."""
    delta = update.constraint_delta
    dropped_ids = bool(update.person_ids_from_index)
    dropped_ops = delta.people_include is not None or delta.people_exclude is not None
    if dropped_ids:
        log.info("intent_person_ids_dropped", count=len(update.person_ids_from_index))
    if dropped_ops:
        log.info("intent_people_ops_dropped", reason="person ids only come from the index")
        delta = delta.model_copy(update={"people_include": None, "people_exclude": None})
    if not dropped_ids and not dropped_ops:
        return update
    return update.model_copy(update={"constraint_delta": delta, "person_ids_from_index": ()})


def _chip_intent_class(speech_act: SpeechAct) -> IntentClass:
    mapping = {
        SpeechAct.RESET_SOFT: IntentClass.RESET,
        SpeechAct.REFINE_DURATION: IntentClass.DURATION_ONLY,
        SpeechAct.REFINE_GENRE: IntentClass.PURE_GENRE_FACET,
        SpeechAct.PERSON_DISAMBIGUATE: IntentClass.PEOPLE_FUZZY,
        SpeechAct.REFINE_MOOD: IntentClass.MOOD_GENRE,
    }
    return mapping.get(speech_act, IntentClass.OTHER)


def _state_from_update(
    update: IntentUpdate,
    *,
    source: IntentSource,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    cost_usd: float | None = None,
) -> dict[str, object]:
    out: dict[str, object] = {
        "delta": update.constraint_delta,
        "intent_source": source,
        "intent_class": update.intent_class.value,
        "query_rewrite": update.query_rewrite,
        "person_soft": _person_soft_or_none(update.person_soft),
        "person_mentions": tuple(update.person_mentions),
    }
    if tokens_in is not None:
        out["tokens_in"] = tokens_in
        out["tokens_out"] = tokens_out if tokens_out is not None else 0
        out["cost_usd"] = cost_usd if cost_usd is not None else 0.0
    log.info("intent_resolved", intent_source=source, intent_class=update.intent_class.value)
    return out


def _from_chip(state: TurnState, chips: ChipSource | None) -> dict[str, object]:
    chip_id = state.get("chip_id") or ""
    if not chip_id:
        raise ChipInvalid("")
    if chips is None:
        raise ChipInvalid(chip_id)
    record = chips.lookup_chip(chip_id)
    update = IntentUpdate(
        intent_class=_chip_intent_class(record.speech_act),
        query_rewrite="",
        constraint_delta=record.delta,
    )
    return _state_from_update(update, source="chip")


def _constraints_of(state: TurnState) -> ConstraintState:
    current = state.get("constraints")
    if isinstance(current, ConstraintState):
        return current
    return ConstraintState.empty()


async def _cache_get(
    cache: IntentCache | None, text: str, constraints: ConstraintState
) -> IntentUpdate | None:
    if cache is None:
        return None
    try:
        raw = await cache.get_intent(normalize_text(text), constraints_hash(constraints))
    except Exception:
        log.warning("intent_cache_get_failed")
        return None
    if not raw:
        return None
    try:
        return _sanitize_model_intent(IntentUpdate.model_validate_json(raw))
    except ValidationError:
        log.info("intent_cache_invalid")
        return None


async def _cache_put(
    cache: IntentCache | None,
    text: str,
    constraints: ConstraintState,
    update: IntentUpdate,
) -> None:
    if cache is None:
        return
    try:
        await cache.set_intent(
            normalize_text(text),
            constraints_hash(constraints),
            update.model_dump_json(),
        )
    except Exception:
        log.warning("intent_cache_set_failed")


def _is_chip_turn(state: TurnState) -> bool:
    if state.get("message_type") == "chip":
        return True
    chip_id = state.get("chip_id")
    return bool(chip_id)


async def run_intent(
    state: TurnState,
    *,
    chips: ChipSource | None = None,
    cache: IntentCache | None = None,
    model: BaseChatModel | None = None,
    cost_handler: CostCallbackHandler | None = None,
    settings: Settings | None = None,
) -> dict[str, object]:
    """Resolve one ConstraintDelta. Chip and rules never call the model."""
    if _is_chip_turn(state):
        return _from_chip(state, chips)

    text = state.get("text") or ""
    constraints = _constraints_of(state)

    if not text.strip():
        return _state_from_update(_empty_intent(""), source="rules")

    rules_hit = match_rules(text)
    if rules_hit is not None:
        return _state_from_update(rules_hit, source="rules")

    cached = await _cache_get(cache, text, constraints)
    if cached is not None:
        log.info("intent_cache_hit")
        return _state_from_update(cached, source="llm")

    return await _from_llm(
        text,
        constraints,
        cache=cache,
        model=model,
        cost_handler=cost_handler,
        settings=settings,
    )


async def _from_llm(
    text: str,
    constraints: ConstraintState,
    *,
    cache: IntentCache | None,
    model: BaseChatModel | None,
    cost_handler: CostCallbackHandler | None,
    settings: Settings | None,
) -> dict[str, object]:
    fallback_used = False

    def _fallback(_input: Any) -> IntentUpdate:
        nonlocal fallback_used
        fallback_used = True
        return _rules_or_empty(text)

    handler = cost_handler if cost_handler is not None else CostCallbackHandler()
    # Gateway helper, not ChatAnthropic.with_structured_output: default method is
    # function_calling and would bind tools (invariant 1). Always pass fallback so
    # a schema miss degrades instead of raising LLMSchemaError.
    chain = chat_prompt_template("intent") | structured_output(
        IntentUpdate,
        fallback=RunnableLambda(_fallback),
        model=model,
        settings=settings,
    )
    payload = {
        "text": text,
        "constraints_json": constraints.model_dump_json(),
        "genre_ids": ", ".join(item.value for item in GenreId),
        "mood_ids": ", ".join(item.value for item in MoodId),
    }
    config: RunnableConfig = {"callbacks": [handler]}
    try:
        raw = await chain.ainvoke(payload, config=config)
    except LLMError:
        log.info("intent_llm_failed", reason="gateway")
        return _state_from_update(_rules_or_empty(text), source="rules")

    if not isinstance(raw, IntentUpdate):
        log.info("intent_llm_failed", reason="not_intent_update")
        return _state_from_update(_rules_or_empty(text), source="rules")

    if fallback_used:
        return _state_from_update(raw, source="rules")

    update = _sanitize_model_intent(raw)
    await _cache_put(cache, text, constraints, update)
    return _state_from_update(
        update,
        source="llm",
        tokens_in=handler.tokens_in,
        tokens_out=handler.tokens_out,
        cost_usd=handler.cost_usd,
    )


def make_intent_node(
    *,
    chips: ChipSource | None = None,
    cache: IntentCache | None = None,
    model: BaseChatModel | None = None,
    cost_handler: CostCallbackHandler | None = None,
    settings: Settings | None = None,
) -> Callable[[TurnState], Awaitable[dict[str, object]]]:
    """Bind I/O for the graph. T24 wires the real session, cache, and model."""

    async def _node(state: TurnState) -> dict[str, object]:
        return await run_intent(
            state,
            chips=chips,
            cache=cache,
            model=model,
            cost_handler=cost_handler,
            settings=settings,
        )

    return _node


intent = make_intent_node()
