"""Intent node: chip lookup, closed-class rules, else one structured LLM call.

Three sources converge on one ConstraintDelta. The model never names a title
and never emits a catalog_id or person_id; person IDs come only from the index.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Any, Literal, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig, RunnableLambda
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

from assist.config import Settings
from assist.domain.constraints import (
    AddOp,
    ClearOp,
    ConstraintDelta,
    ConstraintState,
    FieldOp,
    RemoveOp,
    ReplaceOp,
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


# --- T29: flat wire schema -------------------------------------------------
#
# `IntentUpdate` above (and its nested `ConstraintDelta`/`FieldOp` union) is
# still what the rest of the graph works with. It is no longer what crosses
# the provider boundary: a discriminated union nested 8 `$defs` deep compiles
# to a ~9.7KB grammar that Anthropic rejects outright (400 invalid_request_
# error, "compiled grammar is too large"). `IntentOpWire`/`IntentUpdateWire`
# below are the only schema the model ever sees. `field` is a plain `str`,
# not an enum, on purpose -- that is most of the size win, and it pushes
# validation of field names onto `to_constraint_delta` (below), which drops
# anything it does not recognize rather than raising.
#
# Because every op carries one scalar `value`, a list field (e.g.
# genres_include) arrives as multiple ops sharing the same (field, op) pair.
# `to_constraint_delta` coalesces those into a single AddOp/RemoveOp/ReplaceOp.


class IntentOpWire(BaseModel):
    """One flat constraint operation as the model emits it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field: str = Field(description="ConstraintDelta field name, e.g. 'origins'.")
    op: Literal["set", "add", "remove", "replace", "clear"] = "set"
    value: str = Field(
        default="",
        description="Scalar payload. Ignored for op=clear. List fields repeat this op.",
    )


class IntentUpdateWire(BaseModel):
    """Flat, provider-facing shape. Adapted to `IntentUpdate` server-side."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent_class: str = Field(default=IntentClass.OTHER.value)
    query_rewrite: str = ""
    ops: tuple[IntentOpWire, ...] = ()
    person_role: str | None = None
    person_era_year_min: int | None = None
    person_era_year_max: int | None = None
    person_free_hint: str | None = None
    person_mentions: tuple[str, ...] = ()
    reset_soft: bool = False


# List-valued ConstraintDelta fields take Add/Remove/Replace; everything else
# (besides reset_soft, which is not a FieldOp at all) is scalar and takes Set
# or Clear. This split is not derivable from ConstraintDelta's own typing (a
# FieldOp slot accepts any of the five op classes regardless of the target's
# real shape) so it is spelled out once, here, and nowhere else.
_LIST_DELTA_FIELDS = frozenset({"genres_include", "genres_exclude", "moods", "origins"})
_INT_DELTA_FIELDS = frozenset({"year_min", "year_max", "duration_max_min"})
_BOOL_DELTA_FIELDS = frozenset({"local_originals_only"})
# Forbidden regardless of validity: person ids come only from the index (T18),
# and the catalog has no language field -- the prompt maps languages to origins.
_FORBIDDEN_DELTA_FIELDS = frozenset({"languages", "people_include", "people_exclude"})
_LIST_OP_CLASSES: dict[str, type[AddOp | RemoveOp | ReplaceOp]] = {
    "add": AddOp,
    "remove": RemoveOp,
    "replace": ReplaceOp,
}


def _coerce_int(value: str) -> int | None:
    try:
        return int(value.strip())
    except (ValueError, AttributeError):
        return None


def _coerce_bool(value: str) -> bool | None:
    lowered = value.strip().lower() if isinstance(value, str) else ""
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def _op_parts(raw: object) -> tuple[str, str, str]:
    """Pull (field, op, value) out of anything -- a real IntentOpWire, a dict,
    or plain garbage. Never raises: malformed input just yields empty parts,
    which the caller then drops."""
    if isinstance(raw, IntentOpWire):
        return raw.field, raw.op, raw.value
    if isinstance(raw, Mapping):
        field = raw.get("field", "")
        op = raw.get("op", "")
        value = raw.get("value", "")
    else:
        field = getattr(raw, "field", "")
        op = getattr(raw, "op", "")
        value = getattr(raw, "value", "")
    field = field if isinstance(field, str) else ""
    op = op if isinstance(op, str) else ""
    if not isinstance(value, str):
        value = "" if value is None else str(value)
    return field, op, value


def to_constraint_delta(
    ops: Sequence[object] | None, *, reset_soft: bool = False
) -> ConstraintDelta:
    """Adapt flat wire ops into a `ConstraintDelta`. Pure: no I/O, no mutation
    of `ops`, deterministic, never raises -- unrecognized fields, ops the
    merge algebra could not use, and bad values are dropped, not coerced."""
    allowed_fields = frozenset(ConstraintDelta.model_fields) - {"reset_soft"}
    scalar_fields = allowed_fields - _LIST_DELTA_FIELDS

    list_ops: dict[str, tuple[str, list[str]]] = {}
    single_ops: dict[str, FieldOp] = {}

    for raw in ops or ():
        field, op, value = _op_parts(raw)
        if field == "reset_soft":
            if op == "set":
                coerced = _coerce_bool(value)
                if coerced is not None:
                    reset_soft = coerced
            continue
        if field not in allowed_fields or field in _FORBIDDEN_DELTA_FIELDS:
            continue
        if op == "clear":
            # Clear is valid on both list and scalar fields; it supersedes
            # any op already collected for this field this turn.
            single_ops[field] = ClearOp()
            list_ops.pop(field, None)
            continue
        if field in _LIST_DELTA_FIELDS:
            if op not in _LIST_OP_CLASSES or not value:
                continue
            existing = list_ops.get(field)
            if existing is not None and existing[0] == op:
                if value not in existing[1]:
                    existing[1].append(value)
            else:
                list_ops[field] = (op, [value])
            single_ops.pop(field, None)
        elif field in scalar_fields:
            if op != "set" or not value:
                continue
            if field in _INT_DELTA_FIELDS:
                coerced_int = _coerce_int(value)
                if coerced_int is None:
                    continue
                single_ops[field] = SetOp(value=coerced_int)
            elif field in _BOOL_DELTA_FIELDS:
                coerced_bool = _coerce_bool(value)
                if coerced_bool is None:
                    continue
                single_ops[field] = SetOp(value=coerced_bool)
            else:
                single_ops[field] = SetOp(value=value)
            list_ops.pop(field, None)

    payload: dict[str, object] = dict(single_ops)
    for field, (op_name, values) in list_ops.items():
        payload[field] = _LIST_OP_CLASSES[op_name](values=tuple(values))
    payload["reset_soft"] = reset_soft
    return ConstraintDelta.model_validate(payload)


def _person_soft_from_wire(wire: IntentUpdateWire) -> PersonSoft | None:
    if (
        wire.person_role is None
        and wire.person_era_year_min is None
        and wire.person_era_year_max is None
        and not wire.person_free_hint
    ):
        return None
    return PersonSoft(
        role=wire.person_role,
        era_year_min=wire.person_era_year_min,
        era_year_max=wire.person_era_year_max,
        free_hint=wire.person_free_hint,
    )


def to_intent_update(wire: IntentUpdateWire) -> IntentUpdate:
    """Adapt the flat provider-facing shape into the internal `IntentUpdate`.
    Everything downstream of this call (merge, cache, eval) is unchanged."""
    return IntentUpdate(
        intent_class=_parse_intent_class(wire.intent_class),
        query_rewrite=wire.query_rewrite,
        constraint_delta=to_constraint_delta(wire.ops, reset_soft=wire.reset_soft),
        person_soft=_person_soft_from_wire(wire),
        person_mentions=tuple(wire.person_mentions),
    )


def _scalar_to_wire_value(value: bool | int | str) -> str:
    # Inverse of _coerce_bool/_coerce_int: bool is checked before int since
    # bool is an int subclass in Python.
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _delta_field_to_ops(field: str, field_op: FieldOp | None) -> list[IntentOpWire]:
    """Expand one ConstraintDelta field's FieldOp back into flat wire ops.
    None -> no ops. ClearOp -> one op, value ignored. Add/Remove/Replace ->
    one op per value, since the wire schema carries one scalar per op."""
    if field_op is None:
        return []
    if isinstance(field_op, ClearOp):
        return [IntentOpWire(field=field, op="clear")]
    if isinstance(field_op, SetOp):
        return [IntentOpWire(field=field, op="set", value=_scalar_to_wire_value(field_op.value))]
    if isinstance(field_op, AddOp | RemoveOp | ReplaceOp):
        return [IntentOpWire(field=field, op=field_op.op, value=v) for v in field_op.values]
    return []


def to_wire(update: IntentUpdate) -> IntentUpdateWire:
    """Adapt the internal `IntentUpdate` into the flat provider-facing shape.
    The inverse of `to_intent_update`: expands `ConstraintDelta`'s `FieldOp`s
    back into flat ops (one op per value for list ops), and folds `reset_soft`
    onto the wire model's own field rather than a synthetic op -- symmetric
    with how `to_constraint_delta` special-cases a `field == "reset_soft"` op,
    but the round trip does not require emitting one."""
    delta = update.constraint_delta
    ops: list[IntentOpWire] = []
    for field in ConstraintDelta.model_fields:
        if field == "reset_soft":
            continue
        ops.extend(_delta_field_to_ops(field, getattr(delta, field)))

    person_soft = update.person_soft
    return IntentUpdateWire(
        intent_class=update.intent_class.value,
        query_rewrite=update.query_rewrite,
        ops=tuple(ops),
        person_role=person_soft.role if person_soft else None,
        person_era_year_min=person_soft.era_year_min if person_soft else None,
        person_era_year_max=person_soft.era_year_max if person_soft else None,
        person_free_hint=person_soft.free_hint if person_soft else None,
        person_mentions=tuple(update.person_mentions),
        reset_soft=delta.reset_soft,
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

    def _fallback(_input: Any) -> IntentUpdateWire:
        # Return type must match the wire schema (structured_output's TSchema),
        # but its content is never read: fallback_used short-circuits below to
        # the same rules-degrade path the old nested-schema fallback used.
        nonlocal fallback_used
        fallback_used = True
        return IntentUpdateWire()

    handler = cost_handler if cost_handler is not None else CostCallbackHandler()
    # Gateway helper, not ChatAnthropic.with_structured_output: default method is
    # function_calling and would bind tools (invariant 1). Always pass fallback so
    # a schema miss degrades instead of raising LLMSchemaError. IntentUpdateWire,
    # not IntentUpdate, is what crosses the provider boundary (T29): the nested
    # FieldOp union compiled to a grammar Anthropic rejected as too large.
    chain = chat_prompt_template("intent") | structured_output(
        IntentUpdateWire,
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
        wire = await chain.ainvoke(payload, config=config)
    except LLMError:
        log.info("intent_llm_failed", reason="gateway")
        return _state_from_update(_rules_or_empty(text), source="rules")

    if not isinstance(wire, IntentUpdateWire):
        log.info("intent_llm_failed", reason="not_intent_update_wire")
        return _state_from_update(_rules_or_empty(text), source="rules")

    if fallback_used:
        return _state_from_update(_rules_or_empty(text), source="rules")

    update = _sanitize_model_intent(to_intent_update(wire))
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
