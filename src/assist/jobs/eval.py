"""Golden-set eval harness. Invokes the compiled graph per query and reports.

Default mode is a fixture graph: committed catalog subset, cached LLM
responses, seeded latency. Live mode uses GraphDeps.live() against stores.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, ConfigDict, Field, field_validator

from assist.config import Settings
from assist.config import settings as default_settings
from assist.domain.catalog import Person, Pick
from assist.domain.constraints import AddOp, ConstraintDelta, ConstraintState
from assist.domain.context import ServerUserCtx
from assist.domain.enums import (
    CreditRole,
    DegradedReason,
    DeviceClass,
    MaturityRating,
    Package,
    Route,
)
from assist.graph.build import GraphDeps, ainvoke_turn, build_graph
from assist.graph.state import TurnState, empty_turn_state
from assist.jobs.fetch import data_dir
from assist.llm.cost import cost_usd
from assist.nodes.intent import IntentClass, IntentUpdate, normalize_text, to_wire
from assist.nodes.people import MemoryPeopleIndex
from assist.nodes.reply import GroundedReply
from assist.obs.logging import get_logger
from assist.stores.session import Session, SessionBindError

log = get_logger("assist.jobs.eval")

QUERIES_NAME = "queries.jsonl"
CATALOG_NAME = "catalog.json"
REPORT_NAME = "eval-report.md"

SliceName = Literal[
    "mood_genre",
    "person_fuzzy",
    "decade",
    "duration",
    "known_item",
    "reset",
    "adversarial",
    "vague",
]

REQUIRED_SLICES: tuple[SliceName, ...] = (
    "mood_genre",
    "person_fuzzy",
    "decade",
    "duration",
    "known_item",
    "reset",
    "adversarial",
    "vague",
)

# Seeded fixture latency so a re-run with EVAL_SEED is byte-stable.
_LATENCY_STAGES: tuple[str, ...] = (
    "load_session",
    "guard",
    "intent",
    "people",
    "retrieve",
    "rank",
    "availability",
    "reply",
    "sanitize",
    "chips",
    "persist",
)

_SYNTHETIC_FIELDS: tuple[str, ...] = (
    "availability windows (fixture catalog is all playable; live catalog uses hashed windows)",
    "pop_28d (committed in data/golden/catalog.json; live catalog hashes catalog_id)",
    "catalog subset (fixture titles, not the full 8,807-row index)",
    "LLM responses (replayed from the committed golden set, not a live model)",
    "per-stage latency (seeded from EVAL_SEED in fixture mode; live mode uses wall-clock)",
)

_DEFAULT_REPLY = GroundedReply(
    reply="Here are a few that fit what you asked for.",
    pick_indices=(0, 1, 2),
    chip_speech_acts=(),
)


class EvalQuery(BaseModel):
    """One golden utterance. Extra fields drive the LLM cache, not the DB row."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    text: str
    slice: SliceName
    expect_class: str
    expect_ids: tuple[str, ...] = ()
    expect_person_id: str | None = None
    person_name: str | None = None
    moods: tuple[str, ...] = ()
    genres: tuple[str, ...] = ()

    @field_validator("expect_person_id", "person_name", mode="before")
    @classmethod
    def _empty_str_none(cls, value: object) -> object:
        if value == "":
            return None
        return value


class Scorecard(BaseModel):
    """Aggregated metrics. Every field is populated, including zeros."""

    model_config = ConfigDict(frozen=True)

    n_queries: int
    recall_at_8: float
    person_at_1: float
    schema_failure_rate: float
    degraded_rate: float
    usd_per_turn: float
    route_mix: dict[str, float]
    latency_p50_ms: dict[str, float]
    latency_p95_ms: dict[str, float]
    slice_counts: dict[str, int]
    mode: str
    seed: int
    synthetic_fields: tuple[str, ...] = _SYNTHETIC_FIELDS

    def canonical_metrics(self) -> dict[str, object]:
        """Latency-free fingerprint. Used to prove a re-run is deterministic."""
        return {
            "n_queries": self.n_queries,
            "recall_at_8": round(self.recall_at_8, 6),
            "person_at_1": round(self.person_at_1, 6),
            "schema_failure_rate": round(self.schema_failure_rate, 6),
            "degraded_rate": round(self.degraded_rate, 6),
            "usd_per_turn": round(self.usd_per_turn, 8),
            "route_mix": dict(sorted((k, round(v, 6)) for k, v in self.route_mix.items())),
            "slice_counts": dict(sorted(self.slice_counts.items())),
            "mode": self.mode,
            "seed": self.seed,
        }


@dataclass
class QueryOutcome:
    query: EvalQuery
    pick_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    people_ids: tuple[str, ...]
    route: str
    degraded_reason: str
    intent_source: str
    intent_class: str
    timings: dict[str, int]
    tokens_in: int
    tokens_out: int
    cost_usd: float
    recall_at_8: float | None
    person_at_1: bool | None


@dataclass
class _LlmEntry:
    intent: IntentUpdate
    reply: GroundedReply
    intent_tokens_in: int = 180
    intent_tokens_out: int = 60
    reply_tokens_in: int = 420
    reply_tokens_out: int = 80


@dataclass
class MemorySessions:
    _store: dict[str, Session] = field(default_factory=dict)

    async def load(self, session_id: str, user_id: str, profile_id: str) -> Session:
        existing = self._store.get(session_id)
        if existing is None:
            session = Session.create(session_id=session_id, user_id=user_id, profile_id=profile_id)
            self._store[session_id] = session
            return session
        if existing.user_id != user_id or existing.profile_id != profile_id:
            raise SessionBindError(
                session_id=session_id,
                bound_user_id=existing.user_id,
                bound_profile_id=existing.profile_id,
                user_id=user_id,
                profile_id=profile_id,
            )
        return existing

    async def save(self, session: Session) -> None:
        self._store[session.session_id] = session


@dataclass
class MemoryCache:
    store: dict[tuple[str, str], str] = field(default_factory=dict)

    async def get_intent(self, norm_text: str, constraints_hash_value: str) -> str | None:
        return self.store.get((norm_text, constraints_hash_value))

    async def set_intent(self, norm_text: str, constraints_hash_value: str, payload: str) -> None:
        self.store[(norm_text, constraints_hash_value)] = payload


@dataclass
class MemoryEvents:
    events: list[Any] = field(default_factory=list)

    async def record(self, event: Any) -> None:
        self.events.append(event)


class AllPlayable:
    async def playable_now(
        self, catalog_ids: list[str] | tuple[str, ...], ctx: ServerUserCtx
    ) -> dict[str, bool]:
        return {catalog_id: True for catalog_id in catalog_ids}


class FixtureEs:
    """In-memory titles index. Applies filter clauses; scores BM25-ish overlap."""

    def __init__(self, titles: Sequence[Mapping[str, Any]]) -> None:
        self.titles = tuple(dict(item) for item in titles)

    async def search(self, **kwargs: Any) -> dict[str, Any]:
        body = kwargs.get("body")
        if not isinstance(body, Mapping):
            body = {}
        filters = _filters_of(body)
        query = _query_text_of(body)
        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in self.titles:
            if not _doc_matches(doc, filters):
                continue
            scored.append((_bm25_score(doc, query), doc))
        scored.sort(key=lambda item: (-item[0], str(item[1].get("catalog_id") or "")))
        size = body.get("size")
        limit = int(size) if isinstance(size, int) and size > 0 else len(scored)
        hits = [
            {
                "_id": str(doc.get("catalog_id") or ""),
                "_score": score,
                "_source": dict(doc),
            }
            for score, doc in scored[:limit]
        ]
        return {"hits": {"hits": hits}}


class CachedChat(BaseChatModel):
    """Replays committed IntentUpdate / GroundedReply. Never opens a network client."""

    replay: dict[str, Any] = Field(default_factory=dict)
    calls: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "eval-cache"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise AssertionError("raw generate must not run on the eval cache path")

    def with_structured_output(
        self,
        schema: dict[str, Any] | type[Any],
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        name = getattr(schema, "__name__", type(schema).__name__)

        def _run(value: Any) -> Any:
            self.calls.append(name)
            text = _text_from_input(value)
            entry = self.replay.get(normalize_text(text))
            if name == "IntentUpdateWire":
                if entry is None:
                    return to_wire(
                        IntentUpdate(intent_class=IntentClass.OTHER, query_rewrite=text.strip())
                    )
                return to_wire(entry.intent)
            if name == "GroundedReply":
                if entry is None:
                    return _DEFAULT_REPLY
                return entry.reply
            raise TypeError(name)

        return RunnableLambda(_run)


def golden_dir() -> Path:
    return data_dir() / "golden"


def default_queries_path() -> Path:
    return golden_dir() / QUERIES_NAME


def default_catalog_path() -> Path:
    return golden_dir() / CATALOG_NAME


def default_report_path() -> Path:
    return data_dir().parent / "docs" / REPORT_NAME


def load_queries(path: Path | None = None) -> list[EvalQuery]:
    target = path if path is not None else default_queries_path()
    lines = target.read_text(encoding="utf-8").splitlines()
    queries: list[EvalQuery] = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        queries.append(EvalQuery.model_validate_json(stripped))
    return queries


def load_catalog(path: Path | None = None) -> tuple[tuple[dict[str, Any], ...], tuple[Person, ...]]:
    target = path if path is not None else default_catalog_path()
    payload = json.loads(target.read_text(encoding="utf-8"))
    raw_titles = payload.get("titles") if isinstance(payload, dict) else None
    raw_people = payload.get("people") if isinstance(payload, dict) else None
    titles = tuple(item for item in (raw_titles or ()) if isinstance(item, dict))
    people: list[Person] = []
    for item in raw_people or ():
        if not isinstance(item, dict):
            continue
        roles_raw = item.get("roles") or ()
        roles = tuple(
            CreditRole(str(role))
            for role in roles_raw
            if str(role) in {member.value for member in CreditRole}
        )
        name = str(item.get("name") or "")
        people.append(
            Person(
                person_id=str(item.get("person_id") or ""),
                name=name,
                name_norm=name.lower(),
                roles=roles,
                popularity=float(item.get("popularity") or 0.0),
                active_year_min=_as_optional_int(item.get("active_year_min")),
                active_year_max=_as_optional_int(item.get("active_year_max")),
            )
        )
    return titles, tuple(people)


def recall_at_k(expect_ids: Sequence[str], got_ids: Sequence[str], k: int = 8) -> float | None:
    """|expect ∩ top-k| / min(|expect|, k). None when the query has no gold ids."""
    expect = [item for item in expect_ids if item]
    if not expect:
        return None
    denom = min(len(expect), k)
    if denom <= 0:
        return None
    got = [item for item in got_ids[:k] if item]
    hits = len(set(expect) & set(got))
    return hits / denom


def person_at_1(expect_person_id: str | None, people_ids: Sequence[str]) -> bool | None:
    if not expect_person_id:
        return None
    if not people_ids:
        return False
    return people_ids[0] == expect_person_id


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile. Empty → 0 so the report field is never blank."""
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    idx = min(len(ordered) - 1, max(0, round(rank)))
    return ordered[idx]


def llm_cache_for(queries: Sequence[EvalQuery]) -> dict[str, _LlmEntry]:
    """Deterministic IntentUpdate + GroundedReply per golden text."""
    cache: dict[str, _LlmEntry] = {}
    for query in queries:
        intent = _intent_for(query)
        if intent is None:
            continue
        cache[normalize_text(query.text)] = _LlmEntry(intent=intent, reply=_DEFAULT_REPLY)
    return cache


def _intent_for(query: EvalQuery) -> IntentUpdate | None:
    if query.slice in {"reset", "adversarial", "decade", "duration", "known_item"}:
        return None
    if query.expect_class in {
        IntentClass.PURE_GENRE_FACET.value,
        IntentClass.PURE_DECADE.value,
        IntentClass.DURATION_ONLY.value,
        IntentClass.KNOWN_TITLE_LOOKUP.value,
        IntentClass.RESET.value,
    }:
        return None
    fields: dict[str, Any] = {}
    if query.moods:
        fields["moods"] = AddOp(values=query.moods)
    if query.genres:
        fields["genres_include"] = AddOp(values=query.genres)
    delta = ConstraintDelta.model_validate(fields) if fields else ConstraintDelta()
    mentions = (query.person_name,) if query.person_name else ()
    try:
        intent_class = IntentClass(query.expect_class)
    except ValueError:
        intent_class = IntentClass.OTHER
    return IntentUpdate(
        intent_class=intent_class,
        query_rewrite=query.text.strip(),
        constraint_delta=delta,
        person_mentions=mentions,
    )


def _eval_ctx() -> ServerUserCtx:
    return ServerUserCtx(
        user_id="eval-user",
        profile_id="eval-profile",
        geo="US",
        package=Package.PREMIUM,
        maturity_max=MaturityRating.NC_17,
        kids_flag=False,
        device_class=DeviceClass.WEB,
    )


def _as_optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _text_from_input(value: Any) -> str:
    """Prompt templates render before the model; pull the user line back out."""
    if isinstance(value, Mapping):
        text = value.get("text")
        if isinstance(text, str) and text.strip():
            return text
    to_messages = getattr(value, "to_messages", None)
    if callable(to_messages):
        value = to_messages()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            content = getattr(item, "content", item)
            if isinstance(content, str):
                parts.append(content)
        value = "\n".join(parts)
    if isinstance(value, str):
        marker = "User message:"
        if marker in value:
            # The user line is the first block after the marker. Anything after a
            # blank line is prompt scaffolding (T28 appends a reinforcement note),
            # so stop there rather than assuming {text} renders last.
            tail = value.rsplit(marker, 1)[-1].strip()
            return tail.split("\n\n", 1)[0].strip()
        return value.strip()
    return ""


def _filters_of(body: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    query = body.get("query")
    if isinstance(query, Mapping):
        bool_q = query.get("bool")
        if isinstance(bool_q, Mapping):
            filt = bool_q.get("filter")
            if isinstance(filt, list):
                return [item for item in filt if isinstance(item, Mapping)]
    knn = body.get("knn")
    if isinstance(knn, Mapping):
        filt = knn.get("filter")
        if isinstance(filt, list):
            return [item for item in filt if isinstance(item, Mapping)]
        if isinstance(filt, Mapping):
            inner = filt.get("bool")
            if isinstance(inner, Mapping):
                nested = inner.get("filter")
                if isinstance(nested, list):
                    return [item for item in nested if isinstance(item, Mapping)]
            return [filt]
    return []


def _query_text_of(body: Mapping[str, Any]) -> str:
    query = body.get("query")
    if not isinstance(query, Mapping):
        return ""
    bool_q = query.get("bool")
    if not isinstance(bool_q, Mapping):
        return ""
    must = bool_q.get("must")
    if not isinstance(must, list):
        return ""
    for clause in must:
        if not isinstance(clause, Mapping):
            continue
        multi = clause.get("multi_match")
        if isinstance(multi, Mapping):
            raw = multi.get("query")
            if isinstance(raw, str):
                return raw
    return ""


def _field_values(doc: Mapping[str, Any], field: str) -> list[str]:
    raw = doc.get(field)
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, bool):
        return [str(raw).lower()]
    return [str(raw)]


def _doc_matches(doc: Mapping[str, Any], filters: Sequence[Mapping[str, Any]]) -> bool:
    return all(_clause_matches(doc, clause) for clause in filters)


def _clause_matches(doc: Mapping[str, Any], clause: Mapping[str, Any]) -> bool:
    term = clause.get("term")
    if isinstance(term, Mapping) and term:
        field, value = next(iter(term.items()))
        return str(value) in _field_values(doc, str(field))
    terms = clause.get("terms")
    if isinstance(terms, Mapping) and terms:
        field, values = next(iter(terms.items()))
        wanted = {str(item) for item in values} if isinstance(values, list) else {str(values)}
        return bool(wanted.intersection(_field_values(doc, str(field))))
    range_q = clause.get("range")
    if isinstance(range_q, Mapping) and range_q:
        field, bounds = next(iter(range_q.items()))
        raw = doc.get(str(field))
        if isinstance(raw, bool) or not isinstance(raw, int):
            return False
        if not isinstance(bounds, Mapping):
            return False
        gte = bounds.get("gte")
        lte = bounds.get("lte")
        if isinstance(gte, int) and raw < gte:
            return False
        if isinstance(lte, int) and raw > lte:
            return False
        return True
    bool_q = clause.get("bool")
    if isinstance(bool_q, Mapping):
        must_not = bool_q.get("must_not")
        if isinstance(must_not, list):
            return all(
                not _clause_matches(doc, sub) for sub in must_not if isinstance(sub, Mapping)
            )
        nested = bool_q.get("filter")
        if isinstance(nested, list):
            return all(_clause_matches(doc, sub) for sub in nested if isinstance(sub, Mapping))
    return True


def _bm25_score(doc: Mapping[str, Any], query: str) -> float:
    pop = float(doc.get("pop_28d") or 0.0)
    stripped = query.strip().lower()
    if not stripped:
        return pop
    title = str(doc.get("title") or "").lower()
    if title == stripped:
        return 100.0 + pop
    q_tokens = set(stripped.split())
    title_tokens = set(title.split())
    overlap = len(q_tokens & title_tokens)
    names = " ".join(str(item) for item in (doc.get("people_names") or ())).lower()
    name_hit = 2.0 if any(token in names for token in q_tokens if len(token) > 2) else 0.0
    synopsis = str(doc.get("synopsis") or "").lower()
    syn_hit = 0.5 if any(token in synopsis for token in q_tokens if len(token) > 3) else 0.0
    return overlap * 3.0 + name_hit + syn_hit + pop


def _seeded_ms(seed: int, query_id: str, stage: str) -> int:
    digest = hashlib.sha256(f"{seed}:{query_id}:{stage}".encode()).digest()
    return 1 + int.from_bytes(digest[:2], "big") % 50


def _pick_ids(state: TurnState) -> tuple[str, ...]:
    raw = state.get("picks") or ()
    out: list[str] = []
    for item in raw:
        if isinstance(item, Pick):
            out.append(item.catalog_id)
        else:
            catalog_id = getattr(item, "catalog_id", None)
            if catalog_id:
                out.append(str(catalog_id))
    return tuple(out)


def _candidate_ids(state: TurnState) -> tuple[str, ...]:
    raw = state.get("candidates") or ()
    out: list[str] = []
    for item in raw:
        catalog_id = getattr(item, "catalog_id", None)
        if catalog_id:
            out.append(str(catalog_id))
    return tuple(out)


def _people_ids(state: TurnState) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(person_id: str) -> None:
        if person_id and person_id not in seen:
            seen.add(person_id)
            ordered.append(person_id)

    constraints = state.get("constraints")
    if isinstance(constraints, ConstraintState):
        for person_id in constraints.people_include:
            _add(person_id)
    for person in state.get("people_candidates") or ():
        _add(str(getattr(person, "person_id", "") or ""))
    return tuple(ordered)


def _enum_value(value: object, default: str) -> str:
    raw = getattr(value, "value", value)
    if raw is None:
        return default
    text = str(raw)
    return text if text else default


def fixture_deps(
    *,
    titles: Sequence[Mapping[str, Any]],
    people: Sequence[Person],
    cache_entries: Mapping[str, _LlmEntry],
) -> GraphDeps:
    model = CachedChat(replay=dict(cache_entries))
    return GraphDeps(
        sessions=MemorySessions(),
        cache=MemoryCache(),
        es=FixtureEs(titles),
        catalog=AllPlayable(),
        events=MemoryEvents(),
        people=MemoryPeopleIndex(people),
        model=model,
        embedder=None,
        settings=default_settings,
    )


async def run_query(
    compiled: Any,
    query: EvalQuery,
    *,
    seed: int,
    cache_entries: Mapping[str, _LlmEntry],
    freeze_latency: bool,
    timeout_s: float | None = None,
) -> QueryOutcome:
    session_id = f"eval-{seed}-{query.id}"
    state = empty_turn_state(
        _eval_ctx(),
        session_id=session_id,
        text=query.text,
        trace_id=f"eval-{query.id}",
    )
    result = await ainvoke_turn(compiled, state, timeout_s=timeout_s)
    candidate_ids = _candidate_ids(result)
    pick_ids = _pick_ids(result)
    ranked = candidate_ids if candidate_ids else pick_ids
    people_ids = _people_ids(result)
    route = _enum_value(result.get("route"), Route.TEMPLATE.value)
    degraded = _enum_value(result.get("degraded_reason"), DegradedReason.NONE.value)
    intent_source = str(result.get("intent_source") or "")
    intent_class = str(result.get("intent_class") or "")
    raw_timings = result.get("timings") or {}
    timings: dict[str, int] = {}
    if isinstance(raw_timings, dict):
        for key, value in raw_timings.items():
            try:
                timings[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
    if freeze_latency:
        timings = {stage: _seeded_ms(seed, query.id, stage) for stage in _LATENCY_STAGES}
    else:
        for stage in _LATENCY_STAGES:
            timings.setdefault(stage, 0)
    tokens_in = int(result.get("tokens_in") or 0)
    tokens_out = int(result.get("tokens_out") or 0)
    usd = float(result.get("cost_usd") or 0.0)
    entry = cache_entries.get(normalize_text(query.text))
    if usd == 0.0 and entry is not None and intent_source == "llm":
        tokens_in = entry.intent_tokens_in
        tokens_out = entry.intent_tokens_out
        usd = cost_usd(entry.intent_tokens_in, entry.intent_tokens_out)
        if route == Route.GENERATIVE.value:
            tokens_in += entry.reply_tokens_in
            tokens_out += entry.reply_tokens_out
            usd += cost_usd(entry.reply_tokens_in, entry.reply_tokens_out)
    return QueryOutcome(
        query=query,
        pick_ids=pick_ids,
        candidate_ids=candidate_ids,
        people_ids=people_ids,
        route=route,
        degraded_reason=degraded,
        intent_source=intent_source,
        intent_class=intent_class,
        timings=timings,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=usd,
        recall_at_8=recall_at_k(query.expect_ids, ranked),
        person_at_1=person_at_1(query.expect_person_id, people_ids),
    )


def aggregate(
    outcomes: Sequence[QueryOutcome],
    *,
    mode: str,
    seed: int,
) -> Scorecard:
    n = len(outcomes)
    recalls = [item.recall_at_8 for item in outcomes if item.recall_at_8 is not None]
    persons = [item.person_at_1 for item in outcomes if item.person_at_1 is not None]
    schema_fails = sum(
        1
        for item in outcomes
        if item.degraded_reason == DegradedReason.GENERATIVE_SCHEMA_FAIL.value
    )
    degraded = sum(1 for item in outcomes if item.degraded_reason != DegradedReason.NONE.value)
    route_counts = Counter(item.route for item in outcomes)
    route_mix = {route.value: 0.0 for route in Route}
    if n:
        for route, count in route_counts.items():
            route_mix[route] = count / n
    slice_counts: dict[str, int] = {name: 0 for name in REQUIRED_SLICES}
    for item in outcomes:
        slice_counts[item.query.slice] = slice_counts.get(item.query.slice, 0) + 1
    latency_p50: dict[str, float] = {}
    latency_p95: dict[str, float] = {}
    for stage in _LATENCY_STAGES:
        values = [float(item.timings.get(stage, 0)) for item in outcomes]
        latency_p50[stage] = percentile(values, 50)
        latency_p95[stage] = percentile(values, 95)
    usd_mean = (sum(item.cost_usd for item in outcomes) / n) if n else 0.0
    return Scorecard(
        n_queries=n,
        recall_at_8=(sum(recalls) / len(recalls)) if recalls else 0.0,
        person_at_1=(sum(1 for hit in persons if hit) / len(persons)) if persons else 0.0,
        schema_failure_rate=(schema_fails / n) if n else 0.0,
        degraded_rate=(degraded / n) if n else 0.0,
        usd_per_turn=usd_mean,
        route_mix=route_mix,
        latency_p50_ms=latency_p50,
        latency_p95_ms=latency_p95,
        slice_counts=slice_counts,
        mode=mode,
        seed=seed,
    )


def render_report(scorecard: Scorecard) -> str:
    lines = [
        "# Eval report",
        "",
        f"Generated by `jobs eval`. Mode: `{scorecard.mode}`. Seed: `{scorecard.seed}`.",
        "",
        "## Synthetic fixtures",
        "",
        "These fields are fixtures, not live catalog facts:",
        "",
    ]
    for item in scorecard.synthetic_fields:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Scorecard",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| queries | {scorecard.n_queries} |",
            f"| recall@8 | {scorecard.recall_at_8:.3f} |",
            f"| person@1 | {scorecard.person_at_1:.3f} |",
            f"| schema-failure rate | {scorecard.schema_failure_rate:.3f} |",
            f"| degraded rate | {scorecard.degraded_rate:.3f} |",
            f"| USD per turn | {scorecard.usd_per_turn:.6f} |",
            "",
            "### Route mix",
            "",
            "| Route | Share |",
            "|---|---|",
        ]
    )
    for route, share in sorted(scorecard.route_mix.items()):
        lines.append(f"| {route} | {share:.3f} |")
    lines.extend(
        [
            "",
            "### Latency p50 / p95 (ms)",
            "",
            "| Stage | p50 | p95 |",
            "|---|---|---|",
        ]
    )
    for stage in _LATENCY_STAGES:
        p50 = scorecard.latency_p50_ms.get(stage, 0.0)
        p95 = scorecard.latency_p95_ms.get(stage, 0.0)
        lines.append(f"| {stage} | {p50:.1f} | {p95:.1f} |")
    lines.extend(
        [
            "",
            "### Slice counts",
            "",
            "| Slice | N |",
            "|---|---|",
        ]
    )
    for name in REQUIRED_SLICES:
        lines.append(f"| {name} | {scorecard.slice_counts.get(name, 0)} |")
    lines.append("")
    return "\n".join(lines)


def write_report(scorecard: Scorecard, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(scorecard), encoding="utf-8")
    log.info("eval_report_written", path=str(path), n=scorecard.n_queries)
    return path


async def run_async(
    *,
    live: bool = False,
    report_path: Path | None = None,
    queries_path: Path | None = None,
    catalog_path: Path | None = None,
    seed: int | None = None,
    settings: Settings | None = None,
    deps: GraphDeps | None = None,
) -> Scorecard:
    cfg = settings if settings is not None else default_settings
    eval_seed = int(cfg.eval_seed if seed is None else seed)
    queries = load_queries(queries_path)
    cache_entries = llm_cache_for(queries)
    mode = "live" if live and deps is None else "fixture"
    freeze_latency = mode == "fixture"
    owned_live = False
    graph_deps = deps
    if graph_deps is None:
        if live:
            graph_deps = GraphDeps.live(settings=cfg)
            owned_live = True
            mode = "live"
            freeze_latency = False
        else:
            titles, people = load_catalog(catalog_path)
            graph_deps = fixture_deps(titles=titles, people=people, cache_entries=cache_entries)
    compiled = build_graph(deps=graph_deps)
    outcomes: list[QueryOutcome] = []
    try:
        for query in queries:
            outcome = await run_query(
                compiled,
                query,
                seed=eval_seed,
                cache_entries=cache_entries,
                freeze_latency=freeze_latency,
            )
            outcomes.append(outcome)
            log.info(
                "eval_query",
                id=query.id,
                slice=query.slice,
                route=outcome.route,
                degraded_reason=outcome.degraded_reason,
                recall_at_8=outcome.recall_at_8,
                person_at_1=outcome.person_at_1,
            )
    finally:
        if owned_live and graph_deps is not None:
            await _aclose_live(graph_deps)
    scorecard = aggregate(outcomes, mode=mode, seed=eval_seed)
    target = report_path if report_path is not None else default_report_path()
    write_report(scorecard, target)
    return scorecard


async def _aclose_live(deps: GraphDeps) -> None:
    embedder = deps.embedder
    close_embed = getattr(embedder, "aclose", None)
    if callable(close_embed):
        await close_embed()
    es = deps.es
    if es is not None:
        from assist.stores.es import close_client

        closer = getattr(es, "close", None)
        if callable(closer):
            await close_client(es)
    catalog = deps.catalog
    database = getattr(catalog, "database", None)
    dispose = getattr(database, "dispose", None)
    if callable(dispose):
        await dispose()


def run(
    *,
    live: bool = False,
    report_path: Path | None = None,
    queries_path: Path | None = None,
    catalog_path: Path | None = None,
    seed: int | None = None,
    settings: Settings | None = None,
    deps: GraphDeps | None = None,
) -> Scorecard:
    """CLI / Makefile entry. Writes `docs/eval-report.md` by default."""
    return asyncio.run(
        run_async(
            live=live,
            report_path=report_path,
            queries_path=queries_path,
            catalog_path=catalog_path,
            seed=seed,
            settings=settings,
            deps=deps,
        )
    )


def register(app: typer.Typer) -> None:
    @app.command("eval")
    def eval_cmd(
        live: Annotated[
            bool,
            typer.Option("--live/--fixture", help="Live stores vs committed fixture catalog"),
        ] = False,
        report: Annotated[
            Path | None,
            typer.Option("--report", help="Markdown report path (default docs/eval-report.md)"),
        ] = None,
        seed: Annotated[
            int | None,
            typer.Option("--seed", help="Override EVAL_SEED"),
        ] = None,
    ) -> None:
        scorecard = run(live=live, report_path=report, seed=seed)
        log.info("cli_eval_done", **scorecard.canonical_metrics())


def main() -> None:
    run()


if __name__ == "__main__":
    main()
