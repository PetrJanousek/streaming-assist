"""People resolver: soft descriptors and name mentions → index person_ids.

The model never emits a person_id. This node is the only path that turns a
soft mention into an id, and every id must already exist in `people`.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import unicodedata
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol

from assist.config import Settings
from assist.config import settings as default_settings
from assist.domain.catalog import Person
from assist.domain.constraints import AddOp, ConstraintDelta, ConstraintState, SetOp, merge
from assist.domain.enums import CreditRole, DegradedReason, DeltaSource, Route, SpeechAct
from assist.domain.picks import MIN_PICKS_GRID, MIN_PICKS_NONE
from assist.graph.state import PersonSoft, TurnState
from assist.obs.logging import get_logger
from assist.stores.es import PEOPLE_ALIAS, people_name_body

log = get_logger("assist.nodes.people")

MAX_CLARIFY = 3
SEARCH_SIZE = 15
# 1-char tokens are not indexed (edge-ngram min_gram=2). AND-matching them
# drops real hits such as "Samuel L. Jackson".
_MIN_TOKEN_LEN = 2
# Prefix of a longer token scores 0.65 + 0.35 * (q_len/n_len), which for
# "chris"/"christina" is ~0.84, above person_theta. An exact given-name
# token must not rank below that, or the clarify list omits Chris Rock.
# Stay under the 0.999 full-name gate so "Chris" still clarifies.
_EXACT_FIRST_NAME = 0.99
_EXACT_LAST_NAME = 0.86
_OUTCOME = Literal["single", "ambiguous", "zero"]

_HINT_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "actress",
        "actor",
        "bloke",
        "chlapek",
        "director",
        "dude",
        "famous",
        "film",
        "from",
        "girl",
        "guy",
        "herec",
        "herecka",
        "in",
        "man",
        "movie",
        "movies",
        "of",
        "old",
        "older",
        "or",
        "spy",
        "star",
        "that",
        "the",
        "this",
        "those",
        "with",
        "woman",
        "year",
        "years",
        "young",
        "younger",
        "80s",
        "90s",
        "70s",
    }
)


class PeopleSearcher(Protocol):
    """Return people documents. Implementations may only yield index rows."""

    async def search(
        self,
        name: str,
        *,
        roles: Sequence[str] = (),
        year_min: int | None = None,
        year_max: int | None = None,
        size: int = SEARCH_SIZE,
    ) -> tuple[Person, ...]: ...


@dataclass(frozen=True)
class AliasBook:
    """Editorial alias → catalog display name. Values are names, never ids."""

    mapping: Mapping[str, str]

    def expand(self, raw: str) -> str:
        folded = _fold(raw)
        if not folded:
            return raw.strip()
        return self.mapping.get(folded, raw.strip())


@dataclass(frozen=True)
class _Scored:
    person: Person
    name_score: float


def default_aliases_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "aliases" / "people.json"


def load_alias_book(path: Path | None = None) -> AliasBook:
    target = path if path is not None else default_aliases_path()
    raw = json.loads(target.read_text(encoding="utf-8"))
    aliases = raw.get("aliases", raw) if isinstance(raw, dict) else {}
    if not isinstance(aliases, dict):
        return AliasBook(mapping={})
    mapping: dict[str, str] = {}
    for key, value in aliases.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        # Aliases resolve to a display name. A p_… value would invent an id.
        folded_key = _fold(key)
        name = value.strip()
        if not folded_key or not name or _looks_like_person_id(name):
            continue
        mapping[folded_key] = name
    return AliasBook(mapping=mapping)


@lru_cache(maxsize=1)
def _default_aliases() -> AliasBook:
    try:
        return load_alias_book()
    except (OSError, json.JSONDecodeError):
        log.warning("people_aliases_missing")
        return AliasBook(mapping={})


def _fold(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = nfkd.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-z0-9]+", " ", ascii_text.lower())
    return " ".join(cleaned.split())


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(tok for tok in _fold(text).split() if len(tok) >= _MIN_TOKEN_LEN)


def _looks_like_person_id(value: str) -> bool:
    return bool(re.fullmatch(r"p_[0-9a-f]{16}", value.strip().lower()))


def _search_text(raw: str) -> str:
    return " ".join(_tokens(raw))


def _token_pair_score(query_tok: str, name_tok: str) -> float:
    if query_tok == name_tok:
        return 1.0
    if len(query_tok) >= _MIN_TOKEN_LEN and name_tok.startswith(query_tok):
        return 0.65 + 0.35 * (len(query_tok) / len(name_tok))
    if len(name_tok) >= _MIN_TOKEN_LEN and query_tok.startswith(name_tok):
        return 0.55
    return 0.0


def name_confidence(query: str, name: str) -> float:
    """Deterministic 0..1 name match. Popularity is not mixed in."""
    q_tokens = _tokens(query)
    n_tokens = _tokens(name)
    if not q_tokens or not n_tokens:
        return 0.0
    if q_tokens == n_tokens:
        return 1.0
    # First+last with a dropped middle initial ("Samuel Jackson" / "Samuel L. Jackson").
    if len(q_tokens) >= 2 and q_tokens[0] == n_tokens[0] and q_tokens[-1] == n_tokens[-1]:
        return 0.96
    n_set = set(n_tokens)
    if all(tok in n_set for tok in q_tokens):
        if len(q_tokens) == 1 and q_tokens[0] == n_tokens[-1]:
            return _EXACT_LAST_NAME
        if len(q_tokens) == 1 and q_tokens[0] == n_tokens[0]:
            return _EXACT_FIRST_NAME
        return 0.90
    used = [False] * len(n_tokens)
    scores: list[float] = []
    for qt in q_tokens:
        best_i = -1
        best = 0.0
        for i, nt in enumerate(n_tokens):
            if used[i]:
                continue
            pair = _token_pair_score(qt, nt)
            if pair > best:
                best = pair
                best_i = i
        if best_i < 0 or best <= 0.0:
            return 0.0
        used[best_i] = True
        scores.append(best)
    return sum(scores) / len(scores)


def person_from_source(src: Mapping[str, object]) -> Person | None:
    """Build a Person from an index `_source`. Missing person_id → drop."""
    person_id = src.get("person_id")
    name = src.get("name")
    if not isinstance(person_id, str) or not person_id:
        return None
    if not isinstance(name, str) or not name.strip():
        return None
    name_norm_raw = src.get("name_norm")
    name_norm = (
        name_norm_raw.strip()
        if isinstance(name_norm_raw, str) and name_norm_raw.strip()
        else " ".join(name.lower().split())
    )
    roles_raw = src.get("roles") or ()
    roles: list[CreditRole] = []
    if isinstance(roles_raw, Sequence) and not isinstance(roles_raw, (str, bytes)):
        for item in roles_raw:
            try:
                roles.append(CreditRole(str(item)))
            except ValueError:
                continue
    credit_raw = src.get("credit_count")
    credit_count = int(credit_raw) if isinstance(credit_raw, int) else 0
    pop_raw = src.get("popularity")
    popularity = float(pop_raw) if isinstance(pop_raw, (int, float)) else 0.0
    ymin = src.get("active_year_min")
    ymax = src.get("active_year_max")
    return Person(
        person_id=person_id,
        name=name.strip(),
        name_norm=name_norm,
        roles=tuple(roles),
        credit_count=credit_count,
        active_year_min=ymin if isinstance(ymin, int) else None,
        active_year_max=ymax if isinstance(ymax, int) else None,
        popularity=popularity,
    )


class EsPeopleSearcher:
    """Query the `people` alias through T06's `people_name_body`."""

    def __init__(
        self,
        client: Any,
        *,
        index: str = PEOPLE_ALIAS,
        timeout_ms: int | None = None,
    ) -> None:
        # Any: AsyncElasticsearch.search is a 50-kwarg overload; tests inject a double.
        self._client = client
        self._index = index
        ms = timeout_ms if timeout_ms is not None else default_settings.elasticsearch_timeout_ms
        self._timeout_s = ms / 1000.0

    async def search(
        self,
        name: str,
        *,
        roles: Sequence[str] = (),
        year_min: int | None = None,
        year_max: int | None = None,
        size: int = SEARCH_SIZE,
    ) -> tuple[Person, ...]:
        body = people_name_body(
            name,
            size=size,
            roles=roles,
            active_year_min=year_min,
            active_year_max=year_max,
        )
        search_kw: dict[str, Any] = {
            "index": self._index,
            "query": body["query"],
            "size": body["size"],
            "sort": body["sort"],
            "track_scores": True,
        }
        timeout_s = self._timeout_s
        async with asyncio.timeout(timeout_s):
            options = getattr(self._client, "options", None)
            if callable(options):
                # Per-request timeout; do not inherit the T06 client's 3 retries.
                resp = await options(request_timeout=timeout_s, max_retries=0).search(**search_kw)
            else:
                resp = await self._client.search(**search_kw, request_timeout=timeout_s)
        payload = getattr(resp, "body", resp)
        if not isinstance(payload, Mapping):
            return ()
        hits_wrap = payload.get("hits")
        if not isinstance(hits_wrap, Mapping):
            return ()
        raw_hits = hits_wrap.get("hits")
        if not isinstance(raw_hits, Sequence):
            return ()
        people: list[Person] = []
        seen: set[str] = set()
        for hit in raw_hits:
            if not isinstance(hit, Mapping):
                continue
            src = hit.get("_source")
            if not isinstance(src, Mapping):
                continue
            person = person_from_source(src)
            if person is None or person.person_id in seen:
                continue
            people.append(person)
            seen.add(person.person_id)
        return tuple(people)


class MemoryPeopleIndex:
    """In-process stand-in for tests. Same filter semantics as people_name_body."""

    def __init__(self, people: Sequence[Person]) -> None:
        self.people = tuple(people)
        self.index_ids = frozenset(person.person_id for person in people)

    async def search(
        self,
        name: str,
        *,
        roles: Sequence[str] = (),
        year_min: int | None = None,
        year_max: int | None = None,
        size: int = SEARCH_SIZE,
    ) -> tuple[Person, ...]:
        q_tokens = _tokens(name)
        role_set = {str(role) for role in roles}
        hits: list[Person] = []
        for person in self.people:
            person_roles = {role.value for role in person.roles}
            if role_set and person_roles.isdisjoint(role_set):
                continue
            if year_min is not None and (
                person.active_year_max is None or person.active_year_max < year_min
            ):
                continue
            if year_max is not None and (
                person.active_year_min is None or person.active_year_min > year_max
            ):
                continue
            if q_tokens and name_confidence(name, person.name) <= 0.0:
                continue
            hits.append(person)
        hits.sort(key=lambda item: item.popularity, reverse=True)
        return tuple(hits[:size])


def _constraints_of(state: TurnState) -> ConstraintState:
    current = state.get("constraints")
    return current if isinstance(current, ConstraintState) else ConstraintState.empty()


def _soft_of(state: TurnState) -> PersonSoft | None:
    soft = state.get("person_soft")
    return soft if isinstance(soft, PersonSoft) else None


def _mentions_of(state: TurnState) -> tuple[str, ...]:
    raw = state.get("person_mentions") or ()
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in raw if str(item).strip())


def _role_values(soft: PersonSoft | None) -> tuple[str, ...]:
    if soft is None or not soft.role:
        return ()
    try:
        return (CreditRole(soft.role).value,)
    except ValueError:
        return ()


def _name_like(text: str) -> str | None:
    tokens = [tok for tok in _tokens(text) if tok not in _HINT_STOP]
    if not tokens:
        return None
    return " ".join(tokens)


def _query_strings(state: TurnState, aliases: AliasBook) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        text = raw.strip()
        if not text:
            return
        candidates = [text, aliases.expand(text)]
        for item in candidates:
            key = _fold(item)
            if not key or key in seen:
                continue
            seen.add(key)
            ordered.append(item)

    for mention in _mentions_of(state):
        _add(mention)
    soft = _soft_of(state)
    if soft is not None and soft.free_hint:
        hint = _name_like(soft.free_hint)
        if hint is not None:
            _add(hint)
    return tuple(ordered)


def _has_descriptor(state: TurnState) -> bool:
    if _mentions_of(state):
        return True
    soft = _soft_of(state)
    if soft is None:
        return False
    return bool(
        soft.role
        or soft.era_year_min is not None
        or soft.era_year_max is not None
        or (soft.free_hint or "").strip()
    )


def _best_name_score(queries: Sequence[str], person: Person) -> float:
    best = 0.0
    for query in queries:
        score = name_confidence(query, person.name)
        if score > best:
            best = score
    return best


def _decide_named(
    scored: Sequence[_Scored], *, theta: float
) -> tuple[_OUTCOME, tuple[Person, ...]]:
    if not scored:
        return "zero", ()
    ranked = sorted(
        scored, key=lambda item: (item.name_score, item.person.popularity), reverse=True
    )
    exact = [item for item in ranked if item.name_score >= 0.999]
    if len(exact) == 1:
        return "single", (exact[0].person,)
    if len(exact) >= 2:
        return "ambiguous", tuple(item.person for item in exact[:MAX_CLARIFY])
    high = [item for item in ranked if item.name_score >= theta]
    if len(high) == 1:
        return "single", (high[0].person,)
    if len(high) >= 2:
        return "ambiguous", tuple(item.person for item in high[:MAX_CLARIFY])
    return "zero", ()


def _decide_soft(hits: Sequence[Person]) -> tuple[_OUTCOME, tuple[Person, ...]]:
    if not hits:
        return "zero", ()
    if len(hits) == 1:
        return "single", (hits[0],)
    return "ambiguous", tuple(hits[:MAX_CLARIFY])


def _era_fallback(constraints: ConstraintState, soft: PersonSoft | None) -> ConstraintState:
    if soft is None:
        return constraints
    delta = ConstraintDelta(
        year_min=SetOp(value=soft.era_year_min)
        if (constraints.year_min is None and soft.era_year_min is not None)
        else None,
        year_max=SetOp(value=soft.era_year_max)
        if (constraints.year_max is None and soft.era_year_max is not None)
        else None,
    )
    if delta.year_min is None and delta.year_max is None:
        return constraints
    return merge(constraints, delta, DeltaSource.RULES)


def _include(constraints: ConstraintState, person_ids: Sequence[str]) -> ConstraintState:
    ids = tuple(pid for pid in person_ids if pid)
    if not ids:
        return constraints
    return merge(
        constraints,
        ConstraintDelta(people_include=AddOp(values=ids)),
        DeltaSource.RULES,
    )


def _timings(state: TurnState, t0: float) -> dict[str, int]:
    timings = dict(state.get("timings") or {})
    timings["people"] = int((time.perf_counter() - t0) * 1000)
    return timings


async def _collect_hits(
    searcher: PeopleSearcher,
    queries: Sequence[str],
    *,
    roles: Sequence[str],
    year_min: int | None,
    year_max: int | None,
) -> tuple[Person, ...]:
    by_id: dict[str, Person] = {}
    names = queries if queries else ("",)
    for name in names:
        text = _search_text(name) if name else ""
        hits = await searcher.search(
            text,
            roles=roles,
            year_min=year_min,
            year_max=year_max,
            size=SEARCH_SIZE,
        )
        for person in hits:
            by_id.setdefault(person.person_id, person)
    return tuple(by_id.values())


async def resolve_people(
    state: TurnState,
    *,
    searcher: PeopleSearcher | None = None,
    aliases: AliasBook | None = None,
    settings: Settings | None = None,
) -> dict[str, object]:
    """LangGraph node. Person ids come only from the searcher / index."""
    t0 = time.perf_counter()
    cfg = settings if settings is not None else default_settings
    theta = float(cfg.person_theta)
    book = aliases if aliases is not None else _default_aliases()
    constraints = _constraints_of(state)
    soft = _soft_of(state)
    queries = _query_strings(state, book)

    if constraints.people_include and not _has_descriptor(state):
        return {}
    if not _has_descriptor(state) and not constraints.people_include:
        return {}

    if searcher is None:
        log.warning("people_resolver_no_searcher")
        return _zero_update(state, constraints, soft, t0, reason="no_searcher")

    roles = _role_values(soft)
    year_min = soft.era_year_min if soft is not None else None
    year_max = soft.era_year_max if soft is not None else None

    try:
        hits = await _collect_hits(
            searcher,
            queries,
            roles=roles,
            year_min=year_min,
            year_max=year_max,
        )
    except Exception:
        log.exception("people_search_failed")
        return _zero_update(
            state,
            constraints,
            soft,
            t0,
            reason="search_failed",
            degraded_reason=DegradedReason.RETRIEVAL_UNAVAILABLE,
        )

    index_ids = {person.person_id for person in hits}

    if queries:
        scored = [
            _Scored(person=person, name_score=_best_name_score(queries, person)) for person in hits
        ]
        scored = [item for item in scored if item.name_score > 0.0]
        outcome, chosen = _decide_named(scored, theta=theta)
    else:
        outcome, chosen = _decide_soft(hits)

    chosen = tuple(person for person in chosen if person.person_id in index_ids)

    log.info(
        "people_resolved",
        outcome=outcome,
        n_hits=len(hits),
        n_chosen=len(chosen),
        names=[person.name for person in chosen],
        theta=theta,
    )
    timings = _timings(state, t0)

    if outcome == "single" and chosen:
        person = chosen[0]
        return {
            "constraints": _include(constraints, (person.person_id,)),
            "person_ambiguous": False,
            "people_candidates": (),
            "min_picks": MIN_PICKS_GRID,
            "picks": (),
            "timings": timings,
        }

    if outcome == "ambiguous" and chosen:
        return {
            "constraints": constraints,
            "person_ambiguous": True,
            "people_candidates": chosen,
            "min_picks": MIN_PICKS_NONE,
            "picks": (),
            "route": Route.CLARIFY,
            "degraded_reason": DegradedReason.PERSON_AMBIGUOUS,
            "chip_speech_acts": (SpeechAct.PERSON_DISAMBIGUATE,),
            "timings": timings,
        }

    return _zero_update(state, constraints, soft, t0, reason="no_confident_match")


def _zero_update(
    state: TurnState,
    constraints: ConstraintState,
    soft: PersonSoft | None,
    t0: float,
    *,
    reason: str,
    degraded_reason: DegradedReason | None = None,
) -> dict[str, object]:
    log.info("people_zero", reason=reason)
    update: dict[str, object] = {
        "constraints": _era_fallback(constraints, soft),
        "person_ambiguous": False,
        "people_candidates": (),
        "min_picks": MIN_PICKS_GRID,
        "picks": (),
        "chip_speech_acts": (SpeechAct.PERSON_DISAMBIGUATE,),
        "timings": _timings(state, t0),
    }
    if degraded_reason is not None:
        update["degraded_reason"] = degraded_reason
    return update


def make_people_node(
    *,
    searcher: PeopleSearcher | None = None,
    aliases: AliasBook | None = None,
    settings: Settings | None = None,
) -> Callable[[TurnState], Awaitable[dict[str, object]]]:
    """Bind the index searcher for the graph. T24 wires the live ES client."""

    async def _node(state: TurnState) -> dict[str, object]:
        return await resolve_people(state, searcher=searcher, aliases=aliases, settings=settings)

    return _node
