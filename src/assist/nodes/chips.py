"""Mint server-side chips. The client never sees a delta.

ChipRecord holds the authoritative ConstraintDelta by design (T05). This node
projects `{id, label}` only — dumping the record into response-shaped state
would leak the delta and let the client become authority.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from assist.config import Settings
from assist.config import settings as default_settings
from assist.domain.catalog import Candidate, Person, Pick
from assist.domain.constraints import AddOp, ConstraintDelta, ConstraintState, SetOp
from assist.domain.context import ServerUserCtx
from assist.domain.enums import (
    DegradedReason,
    GenreId,
    MediaType,
    MoodId,
    Route,
    SpeechAct,
)
from assist.graph.state import ReplyChip, TurnState
from assist.obs.logging import get_logger
from assist.stores.session import ChipRecord, Session

log = get_logger(__name__)

MAX_CHIPS_PER_TURN = 4


class UnknownSpeechActError(ValueError):
    """Mint refused: `speech_act` is not in the closed SpeechAct enum."""

    def __init__(self, speech_act: str) -> None:
        self.speech_act = speech_act
        super().__init__(f"unknown speech_act: {speech_act}")


class ChipPhrase(BaseModel):
    """Server-held label + delta for one SpeechAct. Not a response object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    speech_act: SpeechAct
    label: str
    delta: ConstraintDelta


class ChipPhraseSource(Protocol):
    def chip_phrase(self, speech_act: SpeechAct) -> ChipPhrase | None: ...


class SessionStore(Protocol):
    async def load(self, session_id: str, user_id: str, profile_id: str) -> Session: ...

    async def save(self, session: Session) -> None: ...


def require_speech_act(value: object) -> SpeechAct:
    """Closed-enum gate. Unknown values are refused at mint time, not coerced."""
    if isinstance(value, SpeechAct):
        return value
    text = str(value) if value is not None else ""
    try:
        return SpeechAct(text)
    except ValueError:
        raise UnknownSpeechActError(text) from None


def to_reply_chip(record: ChipRecord) -> ReplyChip:
    """Client projection. Deliberately drops delta, speech_act, expiry."""
    return ReplyChip(id=record.chip_id, label=record.label)


def _builtin_phrases(*, home_country: str) -> dict[SpeechAct, ChipPhrase]:
    origin = home_country or "United States"
    return {
        SpeechAct.REFINE_MOOD: ChipPhrase(
            speech_act=SpeechAct.REFINE_MOOD,
            label="Something funnier",
            delta=ConstraintDelta(moods=AddOp(values=(MoodId.FUNNY.value,))),
        ),
        SpeechAct.REFINE_GENRE: ChipPhrase(
            speech_act=SpeechAct.REFINE_GENRE,
            label="More comedy",
            delta=ConstraintDelta(genres_include=AddOp(values=(GenreId.COMEDY.value,))),
        ),
        SpeechAct.REFINE_DURATION: ChipPhrase(
            speech_act=SpeechAct.REFINE_DURATION,
            label="Under 90 minutes",
            delta=ConstraintDelta(duration_max_min=SetOp(value=90)),
        ),
        SpeechAct.REFINE_ORIGIN: ChipPhrase(
            speech_act=SpeechAct.REFINE_ORIGIN,
            label="From home country",
            delta=ConstraintDelta(origins=AddOp(values=(origin,))),
        ),
        SpeechAct.TOGGLE_LOCAL_ORIGINALS: ChipPhrase(
            speech_act=SpeechAct.TOGGLE_LOCAL_ORIGINALS,
            label="Local originals",
            delta=ConstraintDelta(local_originals_only=SetOp(value=True)),
        ),
        SpeechAct.RESET_SOFT: ChipPhrase(
            speech_act=SpeechAct.RESET_SOFT,
            label="Start over",
            delta=ConstraintDelta(reset_soft=True),
        ),
        SpeechAct.CLARIFY_GENRE: ChipPhrase(
            speech_act=SpeechAct.CLARIFY_GENRE,
            label="Comedy",
            delta=ConstraintDelta(genres_include=AddOp(values=(GenreId.COMEDY.value,))),
        ),
        SpeechAct.CLARIFY_MEDIA_TYPE: ChipPhrase(
            speech_act=SpeechAct.CLARIFY_MEDIA_TYPE,
            label="Movies",
            delta=ConstraintDelta(media_type=SetOp(value=MediaType.FILM.value)),
        ),
        SpeechAct.SAFE_REFUSE_CONTINUE: ChipPhrase(
            speech_act=SpeechAct.SAFE_REFUSE_CONTINUE,
            label="Something else",
            delta=ConstraintDelta(reset_soft=True),
        ),
        SpeechAct.PERSON_DISAMBIGUATE: ChipPhrase(
            speech_act=SpeechAct.PERSON_DISAMBIGUATE,
            label="Name the person",
            delta=ConstraintDelta(),
        ),
        SpeechAct.MORE_LIKE_PICK: ChipPhrase(
            speech_act=SpeechAct.MORE_LIKE_PICK,
            label="More like this",
            delta=ConstraintDelta(),
        ),
    }


class BuiltinChipPhrases:
    """In-process phrase bank. T22 may replace labels from `data/phrases`."""

    def __init__(self, *, home_country: str | None = None) -> None:
        country = home_country if home_country is not None else default_settings.home_country
        self._phrases = _builtin_phrases(home_country=country)

    def chip_phrase(self, speech_act: SpeechAct) -> ChipPhrase | None:
        return self._phrases.get(speech_act)


def _people_of(state: TurnState) -> tuple[Person, ...]:
    raw = state.get("people_candidates") or ()
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(item for item in raw if isinstance(item, Person))


def _candidates_of(state: TurnState) -> tuple[Candidate, ...]:
    raw = state.get("candidates") or ()
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(item for item in raw if isinstance(item, Candidate))


def _picks_of(state: TurnState) -> tuple[Pick, ...]:
    raw = state.get("picks") or ()
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(item for item in raw if isinstance(item, Pick))


def _constraints_of(state: TurnState) -> ConstraintState:
    current = state.get("constraints")
    return current if isinstance(current, ConstraintState) else ConstraintState.empty()


def _toggle_local(constraints: ConstraintState, base: ChipPhrase) -> ChipPhrase:
    if constraints.local_originals_only:
        return ChipPhrase(
            speech_act=SpeechAct.TOGGLE_LOCAL_ORIGINALS,
            label="All titles",
            delta=ConstraintDelta(local_originals_only=SetOp(value=False)),
        )
    return base


# StrEnum values are storage ids, not display text. Only the ones whose id does
# not survive a plain underscore swap need an entry here.
_GENRE_LABELS: dict[GenreId, str] = {
    GenreId.SCIFI: "sci-fi",
    GenreId.STAND_UP: "stand-up",
    GenreId.LGBTQ: "LGBTQ",
}


def _genre_label(genre: GenreId) -> str:
    return _GENRE_LABELS.get(genre, genre.value.replace("_", " "))


def _pool_genre_phrases(
    base: ChipPhrase,
    constraints: ConstraintState,
    candidates: Sequence[Candidate],
    limit: int,
) -> tuple[ChipPhrase, ...]:
    """Refine-genre chips drawn from the retrieved pool, most common first.

    genres_include is ANDed clause-per-genre in ES, so a genre the pool does not
    contain narrows the result set to nothing. Counting the candidates the user
    is already looking at is what keeps every chip non-empty when tapped.
    """
    if limit <= 0:
        return ()
    already = set(constraints.genres_include) | set(constraints.genres_exclude)
    counts: dict[GenreId, int] = {}
    for candidate in candidates:
        for genre in candidate.genres:
            if genre in already:
                continue
            counts[genre] = counts.get(genre, 0) + 1
    # Ties break on the genre id so the same pool always mints the same chips.
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].value))
    return tuple(
        ChipPhrase(
            speech_act=base.speech_act,
            label=f"More {_genre_label(genre)}",
            delta=ConstraintDelta(genres_include=AddOp(values=(genre.value,))),
        )
        for genre, _ in ranked[:limit]
    )


def _more_like_phrase(base: ChipPhrase, candidate: Candidate | None) -> ChipPhrase:
    if candidate is None:
        return base
    label = f"More like {candidate.title}" if candidate.title else base.label
    if not candidate.genres:
        return ChipPhrase(speech_act=base.speech_act, label=label, delta=base.delta)
    return ChipPhrase(
        speech_act=base.speech_act,
        label=label,
        delta=ConstraintDelta(
            genres_include=AddOp(values=tuple(g.value for g in candidate.genres))
        ),
    )


def _requested_acts(state: TurnState) -> list[object]:
    raw = state.get("chip_speech_acts") or ()
    if isinstance(raw, (list, tuple)) and raw:
        return list(raw)
    route = state.get("route")
    reason = state.get("degraded_reason")
    if route is Route.SAFETY or reason is DegradedReason.SAFETY_BLOCK:
        return [SpeechAct.SAFE_REFUSE_CONTINUE]
    if route is Route.CLARIFY or reason is DegradedReason.PERSON_AMBIGUOUS:
        return [SpeechAct.PERSON_DISAMBIGUATE]
    if reason is DegradedReason.EMPTY_CATALOG_MATCH:
        return [SpeechAct.CLARIFY_GENRE, SpeechAct.CLARIFY_MEDIA_TYPE]
    return []


def mint_one(
    session: Session,
    speech_act: object,
    *,
    phrases: ChipPhraseSource,
    constraints: ConstraintState | None = None,
    people: Sequence[Person] = (),
    picks: Sequence[Pick] = (),
    candidates: Sequence[Candidate] = (),
    max_chips: int = MAX_CHIPS_PER_TURN,
) -> tuple[Session, tuple[ReplyChip, ...]]:
    """Mint ReplyChips for one speech act onto `session` via `Session.mint_chip`.

    Unknown speech acts raise `UnknownSpeechActError` — they are never minted.
    """
    act = require_speech_act(speech_act)
    phrase = phrases.chip_phrase(act)
    if phrase is None:
        raise UnknownSpeechActError(act.value)

    current = constraints if constraints is not None else ConstraintState.empty()
    reply_chips: list[ReplyChip] = []

    if act is SpeechAct.PERSON_DISAMBIGUATE and people:
        for person in people[:max_chips]:
            session, record = session.mint_chip(
                label=person.name,
                delta=ConstraintDelta(people_include=AddOp(values=(person.person_id,))),
                speech_act=act,
            )
            reply_chips.append(to_reply_chip(record))
        return session, tuple(reply_chips)

    if act is SpeechAct.MORE_LIKE_PICK and picks:
        by_id = {c.catalog_id: c for c in candidates}
        for pick in picks[:max_chips]:
            bound = _more_like_phrase(phrase, by_id.get(pick.catalog_id))
            session, record = session.mint_chip(
                label=bound.label, delta=bound.delta, speech_act=act
            )
            reply_chips.append(to_reply_chip(record))
        return session, tuple(reply_chips)

    if act is SpeechAct.REFINE_GENRE:
        # No pool signal means no honest refinement to offer, so mint nothing
        # rather than fall back to a fixed genre the results may not contain.
        options = _pool_genre_phrases(phrase, current, candidates, max_chips)
        for bound in options:
            session, record = session.mint_chip(
                label=bound.label, delta=bound.delta, speech_act=act
            )
            reply_chips.append(to_reply_chip(record))
        return session, tuple(reply_chips)

    if act is SpeechAct.REFINE_MOOD:
        # Candidate carries no moods, so a mood chip cannot be grounded in the
        # pool the way a genre chip can. Skipped until it can be.
        return session, ()

    bound = _toggle_local(current, phrase) if act is SpeechAct.TOGGLE_LOCAL_ORIGINALS else phrase
    session, record = session.mint_chip(label=bound.label, delta=bound.delta, speech_act=act)
    return session, (to_reply_chip(record),)


async def mint_chips(
    state: TurnState,
    *,
    session: Session | None = None,
    sessions: SessionStore | None = None,
    phrases: ChipPhraseSource | None = None,
    settings: Settings | None = None,
) -> dict[str, object]:
    """LangGraph node. Writes `chips` as ReplyChip only. Saves session if given."""
    t0 = time.perf_counter()
    cfg = settings if settings is not None else default_settings
    bank = phrases if phrases is not None else BuiltinChipPhrases(home_country=cfg.home_country)

    sess = session
    ctx = state.get("ctx")
    if sess is None and sessions is not None and isinstance(ctx, ServerUserCtx):
        session_id = str(state.get("session_id") or "")
        if session_id:
            try:
                sess = await sessions.load(session_id, ctx.user_id, ctx.profile_id)
            except Exception:
                log.warning("chip_session_load_failed", session_id=session_id)
                sess = None

    if sess is None:
        log.info("chips_skipped", reason="no_session")
        timings = dict(state.get("timings") or {})
        timings["chips"] = int((time.perf_counter() - t0) * 1000)
        return {"chips": (), "timings": timings}

    people = _people_of(state)
    picks = _picks_of(state)
    candidates = _candidates_of(state)
    constraints = _constraints_of(state)
    minted: list[ReplyChip] = []

    for raw_act in _requested_acts(state):
        if len(minted) >= MAX_CHIPS_PER_TURN:
            break
        try:
            sess, chips = mint_one(
                sess,
                raw_act,
                phrases=bank,
                constraints=constraints,
                people=people,
                picks=picks,
                candidates=candidates,
                max_chips=MAX_CHIPS_PER_TURN - len(minted),
            )
        except UnknownSpeechActError as exc:
            log.info("chip_refused", speech_act=exc.speech_act, reason="unknown")
            continue
        minted.extend(chips)

    minted = minted[:MAX_CHIPS_PER_TURN]
    if sessions is not None:
        try:
            await sessions.save(sess)
        except Exception:
            log.warning("chip_session_save_failed", session_id=sess.session_id)

    timings = dict(state.get("timings") or {})
    timings["chips"] = int((time.perf_counter() - t0) * 1000)
    return {"chips": tuple(minted), "timings": timings}


def make_chips_node(
    *,
    sessions: SessionStore | None = None,
    phrases: ChipPhraseSource | None = None,
    settings: Settings | None = None,
) -> Callable[[TurnState], Awaitable[dict[str, object]]]:
    """Bind session + phrase bank for the graph. T24 wires the real stores."""

    async def _node(state: TurnState) -> dict[str, object]:
        return await mint_chips(state, sessions=sessions, phrases=phrases, settings=settings)

    return _node
