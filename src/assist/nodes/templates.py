"""Template replies from the phrase bank. Never names a title off the candidate list.

The model is not involved. A missing phrase or a fill failure degrades to a
generic line; this node does not raise. T24 wires the three reply callables
over the passthrough stubs. Chip *deltas* stay in `nodes/chips.py`; this
module owns labels and reply copy only.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from assist.config import Settings
from assist.config import settings as default_settings
from assist.domain.catalog import Candidate
from assist.domain.constraints import ConstraintState
from assist.domain.enums import DegradedReason, Route, SpeechAct
from assist.domain.picks import min_picks_for
from assist.graph.state import TurnState
from assist.obs.logging import get_logger

log = get_logger("assist.nodes.templates")

# Same generic line sanitize.py uses when prose cannot be grounded.
GENERIC_REPLY = "Here are a few titles that match."
_SLOT_RE = re.compile(r"\{([a-z0-9_]+)\}")
_DEFAULT_PACK: tuple[SpeechAct, ...] = (
    SpeechAct.REFINE_MOOD,
    SpeechAct.REFINE_DURATION,
    SpeechAct.RESET_SOFT,
)
_SHED_ROUTES = frozenset(
    {
        DegradedReason.GENERATIVE_TIMEOUT,
        DegradedReason.GENERATIVE_SCHEMA_FAIL,
        DegradedReason.PROVIDER_THROTTLE,
        DegradedReason.RETRIEVAL_UNAVAILABLE,
        DegradedReason.SESSION_STORE_UNAVAILABLE,
        DegradedReason.HARD_TIMEOUT,
    }
)
_INTENT_TO_REPLY_ID: Mapping[str, str] = {
    "mood_genre": "reply.mood",
    "pure_genre_facet": "reply.genre",
    "pure_decade": "reply.decade",
    "duration_only": "reply.duration",
    "duration": "reply.duration",
    "known_title_lookup": "reply.known_title",
    "known_item": "reply.known_title",
    "reset": "reply.reset",
    "refine_origin": "reply.origin",
}


class Phrase(BaseModel):
    """One phrase_bank row. Matches T04's table: id, speech_act, kind, template."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    speech_act: SpeechAct
    kind: str
    template: str


class PhraseBank:
    """In-process bank. File is the seed; T24 may pass DB rows via `from_records`."""

    def __init__(self, phrases: Sequence[Phrase]) -> None:
        self._phrases: tuple[Phrase, ...] = tuple(phrases)
        self._by_id: dict[str, Phrase] = {item.id: item for item in self._phrases}

    @classmethod
    def from_records(cls, records: Sequence[Phrase]) -> PhraseBank:
        return cls(records)

    @classmethod
    def from_path(cls, path: Path) -> PhraseBank:
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw.get("phrases", raw) if isinstance(raw, dict) else raw
        if not isinstance(rows, list):
            return cls(())
        phrases: list[Phrase] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            try:
                phrases.append(Phrase.model_validate(item))
            except ValidationError:
                log.info("phrase_skipped", reason="invalid_row")
        return cls(phrases)

    def get(self, phrase_id: str) -> Phrase | None:
        return self._by_id.get(phrase_id)

    def template(self, phrase_id: str) -> str | None:
        phrase = self._by_id.get(phrase_id)
        return phrase.template if phrase is not None else None

    def chip_label(self, speech_act: SpeechAct) -> str | None:
        for item in self._phrases:
            if item.kind == "chip" and item.speech_act is speech_act:
                return item.template
        return None

    def speech_acts(self) -> frozenset[SpeechAct]:
        return frozenset(item.speech_act for item in self._phrases)

    def __len__(self) -> int:
        return len(self._phrases)


_PHRASES_REL = Path("data") / "phrases" / "bank.json"


def default_phrases_path() -> Path:
    """Resolve the phrase bank from cwd, an editable checkout, or the image.

    An installed package sits under `.venv/lib/.../site-packages`, so a fixed
    `parents[3]` walks into the venv instead of the repo. Search instead.
    """
    for start in (Path.cwd(), Path(__file__).resolve()):
        for parent in [start, *start.parents]:
            candidate = parent / _PHRASES_REL
            if candidate.is_file():
                return candidate
    return Path("/app") / _PHRASES_REL


def load_phrase_bank(path: Path | None = None) -> PhraseBank:
    """Load the committed bank. A missing or corrupt file becomes an empty bank."""
    target = path if path is not None else default_phrases_path()
    try:
        return PhraseBank.from_path(target)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        log.warning("phrase_bank_missing", path=str(target))
        return PhraseBank(())


def fill_template(
    template: str,
    *,
    candidates: Sequence[Candidate],
    constraints: ConstraintState,
) -> str:
    """Substitute closed slots. Unknown or unfillable slots raise ValueError.

    `{title}` / `{title_1..3}` come only from `candidates`. This function never
    invents a catalog title.
    """
    titles = [item.title for item in candidates if item.title]
    mapping: dict[str, str] = {
        "n": str(len(candidates)),
        "mood": _first_enum_label(constraints.moods),
        "genre": _first_enum_label(constraints.genres_include),
        "year": _year_of(candidates),
    }
    if titles:
        mapping["title"] = titles[0]
        mapping["title_list"] = ", ".join(titles[:3])
        for index, title in enumerate(titles[:3], start=1):
            mapping[f"title_{index}"] = title

    needed = set(_SLOT_RE.findall(template))
    missing = needed - mapping.keys()
    if missing:
        raise ValueError(f"unfilled template slots: {sorted(missing)}")
    empty_needed = {key for key in needed if not mapping.get(key)}
    if empty_needed:
        raise ValueError(f"empty template slots: {sorted(empty_needed)}")

    def _sub(match: re.Match[str]) -> str:
        return mapping[match.group(1)]

    return _SLOT_RE.sub(_sub, template)


def _first_enum_label(values: Sequence[object]) -> str:
    if not values:
        return ""
    raw = getattr(values[0], "value", values[0])
    return str(raw).replace("_", " ")


def _year_of(candidates: Sequence[Candidate]) -> str:
    if not candidates:
        return ""
    year = candidates[0].release_year
    return str(year) if year is not None else ""


def _candidates_of(state: TurnState) -> tuple[Candidate, ...]:
    raw = state.get("candidates") or ()
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(item for item in raw if isinstance(item, Candidate))


def _constraints_of(state: TurnState) -> ConstraintState:
    current = state.get("constraints")
    return current if isinstance(current, ConstraintState) else ConstraintState.empty()


def _reason_of(state: TurnState) -> DegradedReason:
    reason = state.get("degraded_reason")
    return reason if isinstance(reason, DegradedReason) else DegradedReason.NONE


def _intent_class_of(state: TurnState) -> str:
    raw = state.get("intent_class")
    if raw is None:
        return ""
    return str(getattr(raw, "value", raw)).strip()


def _entitled_count(state: TurnState) -> int:
    raw = state.get("entitled_ids")
    if isinstance(raw, (list, tuple, set, frozenset)):
        return len(raw)
    return 0


def _pick_reply_id(
    state: TurnState,
    *,
    slot: Literal["template", "clarify", "refusal"],
) -> str:
    if slot == "refusal":
        return "reply.refusal"
    if slot == "clarify":
        return "reply.clarify_person"
    reason = _reason_of(state)
    if reason is DegradedReason.EMPTY_CATALOG_MATCH or not _candidates_of(state):
        return "reply.empty"
    if reason in _SHED_ROUTES:
        return "reply.degraded"
    if state.get("intent_source") == "chip" or state.get("message_type") == "chip":
        return "reply.chip"
    intent = _intent_class_of(state)
    return _INTENT_TO_REPLY_ID.get(intent, "reply.named")


def _chip_acts(
    slot: Literal["template", "clarify", "refusal"],
    *,
    reason: DegradedReason,
    empty: bool,
) -> tuple[SpeechAct, ...]:
    if slot == "refusal":
        return (SpeechAct.SAFE_REFUSE_CONTINUE,)
    if slot == "clarify" or reason is DegradedReason.PERSON_AMBIGUOUS:
        return (SpeechAct.PERSON_DISAMBIGUATE,)
    if empty or reason is DegradedReason.EMPTY_CATALOG_MATCH:
        return (SpeechAct.CLARIFY_GENRE, SpeechAct.CLARIFY_MEDIA_TYPE)
    return _DEFAULT_PACK


def _route_for(
    slot: Literal["template", "clarify", "refusal"],
    *,
    reason: DegradedReason,
) -> Route:
    if slot == "refusal":
        return Route.SAFETY
    if slot == "clarify":
        return Route.CLARIFY
    if reason in _SHED_ROUTES:
        return Route.DEGRADED_KEYWORD
    return Route.TEMPLATE


def render_reply(
    state: TurnState,
    *,
    slot: Literal["template", "clarify", "refusal"],
    bank: PhraseBank,
) -> str:
    """Pick and fill a phrase. Falls back to GENERIC_REPLY; never raises."""
    candidates = _candidates_of(state)
    constraints = _constraints_of(state)
    phrase_id = _pick_reply_id(state, slot=slot)
    template = bank.template(phrase_id)
    if template is None:
        template = bank.template("reply.generic")
    if template is None:
        return GENERIC_REPLY
    try:
        return fill_template(template, candidates=candidates, constraints=constraints)
    except ValueError:
        generic = bank.template("reply.generic") or GENERIC_REPLY
        try:
            return fill_template(generic, candidates=candidates, constraints=constraints)
        except ValueError:
            return GENERIC_REPLY


def _truncate(reply: str, max_chars: int) -> str:
    if max_chars <= 0 or len(reply) <= max_chars:
        return reply
    cut = reply[:max_chars]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip()


def _build_update(
    state: TurnState,
    *,
    slot: Literal["template", "clarify", "refusal"],
    bank: PhraseBank,
    t0: float,
    max_chars: int,
) -> dict[str, object]:
    candidates = _candidates_of(state)
    reason = _reason_of(state)
    empty = not candidates
    if slot == "template" and empty and reason is DegradedReason.NONE:
        reason = DegradedReason.EMPTY_CATALOG_MATCH
    if slot == "clarify":
        reason = reason if reason is not DegradedReason.NONE else DegradedReason.PERSON_AMBIGUOUS
    if slot == "refusal":
        reason = DegradedReason.SAFETY_BLOCK

    route = _route_for(slot, reason=reason)
    reply = _truncate(render_reply(state, slot=slot, bank=bank), max_chars)
    min_picks = min_picks_for(
        route=route,
        degraded_reason=reason,
        entitled_count=_entitled_count(state),
    )
    timings = dict(state.get("timings") or {})
    timings["reply"] = int((time.perf_counter() - t0) * 1000)
    log.info(
        "reply_template",
        slot=slot,
        route=route.value,
        degraded_reason=reason.value,
        reply_len=len(reply),
        n_candidates=len(candidates),
        min_picks=min_picks,
    )
    return {
        "reply": reply,
        "route": route,
        "degraded_reason": reason,
        "min_picks": min_picks,
        "chip_speech_acts": _chip_acts(slot, reason=reason, empty=empty),
        "timings": timings,
    }


async def reply_template(
    state: TurnState,
    *,
    bank: PhraseBank | None = None,
    settings: Settings | None = None,
) -> dict[str, object]:
    """TEMPLATE / degraded-keyword reply. Ranker picks pad later in sanitize."""
    cfg = settings if settings is not None else default_settings
    t0 = time.perf_counter()
    try:
        loaded = bank if bank is not None else load_phrase_bank()
        return _build_update(
            state, slot="template", bank=loaded, t0=t0, max_chars=cfg.reply_max_chars
        )
    except Exception:
        log.exception("reply_template_failed")
        return _failed_update(state, slot="template", t0=t0)


async def reply_clarify(
    state: TurnState,
    *,
    bank: PhraseBank | None = None,
    settings: Settings | None = None,
) -> dict[str, object]:
    """Person-ambiguous clarify copy. min_picks=0; no title guess."""
    cfg = settings if settings is not None else default_settings
    t0 = time.perf_counter()
    try:
        loaded = bank if bank is not None else load_phrase_bank()
        return _build_update(
            state, slot="clarify", bank=loaded, t0=t0, max_chars=cfg.reply_max_chars
        )
    except Exception:
        log.exception("reply_clarify_failed")
        return _failed_update(state, slot="clarify", t0=t0)


async def reply_refusal(
    state: TurnState,
    *,
    bank: PhraseBank | None = None,
    settings: Settings | None = None,
) -> dict[str, object]:
    """Safety refusal copy. min_picks=0."""
    cfg = settings if settings is not None else default_settings
    t0 = time.perf_counter()
    try:
        loaded = bank if bank is not None else load_phrase_bank()
        return _build_update(
            state, slot="refusal", bank=loaded, t0=t0, max_chars=cfg.reply_max_chars
        )
    except Exception:
        log.exception("reply_refusal_failed")
        return _failed_update(state, slot="refusal", t0=t0)


def _failed_update(
    state: TurnState,
    *,
    slot: Literal["template", "clarify", "refusal"],
    t0: float,
) -> dict[str, object]:
    reason = _reason_of(state)
    if slot == "refusal":
        reason = DegradedReason.SAFETY_BLOCK
        route = Route.SAFETY
        reply = "I cannot help with that. Try a different search."
    elif slot == "clarify":
        reason = DegradedReason.PERSON_AMBIGUOUS
        route = Route.CLARIFY
        reply = "I found more than one person. Tap a name to continue."
    else:
        route = Route.TEMPLATE
        reply = GENERIC_REPLY
        if reason is DegradedReason.NONE and not _candidates_of(state):
            reason = DegradedReason.EMPTY_CATALOG_MATCH
    timings = dict(state.get("timings") or {})
    timings["reply"] = int((time.perf_counter() - t0) * 1000)
    return {
        "reply": reply,
        "route": route,
        "degraded_reason": reason,
        "min_picks": min_picks_for(
            route=route,
            degraded_reason=reason,
            entitled_count=_entitled_count(state),
        ),
        "chip_speech_acts": _chip_acts(slot, reason=reason, empty=not _candidates_of(state)),
        "timings": timings,
    }


def make_template_node(
    *,
    slot: Literal["template", "clarify", "refusal"] = "template",
    bank: PhraseBank | None = None,
    settings: Settings | None = None,
) -> Callable[[TurnState], Awaitable[dict[str, object]]]:
    """Bind a phrase bank for the graph. T24 wires this over the stub."""

    async def _node(state: TurnState) -> dict[str, object]:
        if slot == "clarify":
            return await reply_clarify(state, bank=bank, settings=settings)
        if slot == "refusal":
            return await reply_refusal(state, bank=bank, settings=settings)
        return await reply_template(state, bank=bank, settings=settings)

    return _node
