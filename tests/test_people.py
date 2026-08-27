"""People resolver: names and soft descriptors resolve only against the index."""

from __future__ import annotations

import ast
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast
from urllib.error import URLError
from urllib.request import urlopen

import pytest

from assist.config import settings as default_settings
from assist.domain.catalog import Person
from assist.domain.constraints import AddOp, ConstraintState
from assist.domain.context import ServerUserCtx
from assist.domain.enums import (
    CreditRole,
    DegradedReason,
    DeviceClass,
    MaturityRating,
    Package,
    Route,
    SpeechAct,
)
from assist.graph.state import PersonSoft, empty_turn_state
from assist.nodes.chips import mint_chips
from assist.nodes.people import (
    AliasBook,
    EsPeopleSearcher,
    MemoryPeopleIndex,
    default_aliases_path,
    load_alias_book,
    name_confidence,
    person_from_source,
    resolve_people,
)
from assist.stores.es import PEOPLE_ALIAS, close_client, create_client, people_name_body
from assist.stores.session import Session

ROOT = Path(__file__).resolve().parents[1]
PEOPLE_SRC = ROOT / "src" / "assist" / "nodes" / "people.py"
ALIASES_PATH = ROOT / "data" / "aliases" / "people.json"
GOLDEN_PATH = ROOT / "data" / "aliases" / "golden_people.json"


class FakeSessions:
    def __init__(self, session: Session) -> None:
        self.session = session

    async def load(self, session_id: str, user_id: str, profile_id: str) -> Session:
        return self.session

    async def save(self, session: Session) -> None:
        self.session = session


class RecordingEs:
    def __init__(self, hits: Sequence[Mapping[str, Any]] = ()) -> None:
        self.calls: list[dict[str, Any]] = []
        self.hits = list(hits)

    async def search(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"hits": {"hits": list(self.hits)}}


class BoomSearcher:
    async def search(
        self,
        name: str,
        *,
        roles: Sequence[str] = (),
        year_min: int | None = None,
        year_max: int | None = None,
        size: int = 15,
    ) -> tuple[Person, ...]:
        del name, roles, year_min, year_max, size
        raise RuntimeError("es down")


def _ctx() -> ServerUserCtx:
    return ServerUserCtx(
        user_id="u1",
        profile_id="p1",
        geo="US",
        package=Package.BASIC,
        maturity_max=MaturityRating.PG_13,
        kids_flag=False,
        device_class=DeviceClass.WEB,
    )


def _person(
    person_id: str,
    name: str,
    *,
    roles: tuple[CreditRole, ...] = (CreditRole.ACTOR,),
    popularity: float = 1.0,
    year_min: int | None = 1990,
    year_max: int | None = 2010,
) -> Person:
    return Person(
        person_id=person_id,
        name=name,
        name_norm=" ".join(name.lower().split()),
        roles=roles,
        credit_count=4,
        active_year_min=year_min,
        active_year_max=year_max,
        popularity=popularity,
    )


def _book() -> AliasBook:
    return load_alias_book(ALIASES_PATH)


def _golden() -> list[dict[str, str]]:
    payload = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    queries = payload["queries"]
    assert isinstance(queries, list)
    return [dict(item) for item in queries]


def _ping(url: str) -> bool:
    try:
        with urlopen(url, timeout=2) as resp:
            return 200 <= int(resp.status) < 300
    except (URLError, OSError, TimeoutError):
        return False


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


@pytest.fixture
async def live_searcher() -> AsyncIterator[EsPeopleSearcher]:
    from assist.config import settings

    url = settings.elasticsearch_url
    if not _ping(url):
        pytest.skip("elasticsearch unreachable")
    client = create_client(url=url)
    try:
        exists = await client.indices.exists_alias(name=PEOPLE_ALIAS)
        if not exists:
            pytest.skip("people alias missing")
        count = await client.count(index=PEOPLE_ALIAS)
        body = getattr(count, "body", count)
        n = int(body["count"]) if isinstance(body, Mapping) else 0
        if n < 100:
            pytest.skip("people alias empty")
        yield EsPeopleSearcher(client, index=PEOPLE_ALIAS)
    finally:
        await close_client(client)


# ---------------------------------------------------------------------------
# Name scorer
# ---------------------------------------------------------------------------


def test_name_confidence_exact_and_surname_and_prefix() -> None:
    assert name_confidence("Will Smith", "Will Smith") == 1.0
    assert name_confidence("nolan", "Christopher Nolan") == pytest.approx(0.86)
    assert name_confidence("nolan", "Nolan North") == pytest.approx(0.58)
    # Initials are not indexed (edge-ngram min_gram=2), so they drop out of both sides.
    assert name_confidence("Samuel Jackson", "Samuel L. Jackson") == 1.0
    assert name_confidence("Nic Cage", "Nicolas Cage") >= 0.75
    assert name_confidence("Leo DiCaprio", "Leonardo DiCaprio") >= 0.75
    assert name_confidence("Will Smith", "Willow Smith") < 1.0
    assert name_confidence("the rock", "Dwayne Johnson") == 0.0


def test_aliases_are_names_never_ids() -> None:
    book = _book()
    assert book.mapping
    for canonical in book.mapping.values():
        assert not canonical.lower().startswith("p_")
    assert book.expand("the rock") == "Dwayne Johnson"
    assert book.expand("SRK") == "Shah Rukh Khan"
    assert book.expand("Nolan") == "Christopher Nolan"


def test_default_aliases_path_is_owned_data_dir() -> None:
    assert default_aliases_path() == ALIASES_PATH
    assert ALIASES_PATH.is_file()


# ---------------------------------------------------------------------------
# Index document parsing — never invent an id
# ---------------------------------------------------------------------------


def test_person_from_source_requires_index_person_id() -> None:
    assert person_from_source({"name": "Ghost"}) is None
    assert person_from_source({"person_id": "", "name": "Ghost"}) is None
    parsed = person_from_source(
        {
            "person_id": "p_abc",
            "name": "Ada",
            "name_norm": "ada",
            "roles": ["actor"],
            "popularity": 2.5,
            "credit_count": 3,
            "active_year_min": 1991,
            "active_year_max": 2001,
        }
    )
    assert parsed is not None
    assert parsed.person_id == "p_abc"
    assert parsed.roles == (CreditRole.ACTOR,)


async def test_es_searcher_drops_hits_missing_source_person_id() -> None:
    es = RecordingEs(
        hits=[
            {"_id": "p_from_meta", "_source": {"name": "Invented"}},
            {
                "_id": "ignored",
                "_source": {
                    "person_id": "p_real",
                    "name": "Real Person",
                    "name_norm": "real person",
                    "roles": ["actor"],
                    "popularity": 1.0,
                    "credit_count": 1,
                },
            },
        ]
    )
    searcher = EsPeopleSearcher(es, index=PEOPLE_ALIAS)
    hits = await searcher.search("Real")
    assert [person.person_id for person in hits] == ["p_real"]


async def test_es_searcher_uses_people_alias_and_people_name_body() -> None:
    es = RecordingEs()
    searcher = EsPeopleSearcher(es, index=PEOPLE_ALIAS)
    await searcher.search(
        "nolan",
        roles=("director",),
        year_min=1990,
        year_max=1999,
        size=10,
    )
    assert len(es.calls) == 1
    call = es.calls[0]
    assert call["index"] == PEOPLE_ALIAS
    expected = people_name_body(
        "nolan",
        size=10,
        roles=("director",),
        active_year_min=1990,
        active_year_max=1999,
    )
    assert call["query"] == expected["query"]
    assert call["sort"] == expected["sort"]
    assert call["size"] == expected["size"]


def test_people_py_does_not_construct_person_ids() -> None:
    imported = _imported_modules(PEOPLE_SRC)
    assert "assist.jobs.normalize" not in imported
    tree = ast.parse(PEOPLE_SRC.read_text(encoding="utf-8"), filename=str(PEOPLE_SRC))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else None
            if isinstance(func, ast.Attribute):
                name = func.attr
            assert name != "person_id_for"


# ---------------------------------------------------------------------------
# Resolver outcomes (in-memory index)
# ---------------------------------------------------------------------------


async def test_single_high_confidence_sets_people_include_from_index() -> None:
    ada = _person("p_ada", "Ada Lovelace", popularity=8.0)
    index = MemoryPeopleIndex([ada, _person("p_other", "Other Person", popularity=9.0)])
    state = empty_turn_state(_ctx(), person_mentions=("Ada Lovelace",))
    out = await resolve_people(state, searcher=index, aliases=AliasBook(mapping={}))
    constraints = out["constraints"]
    assert isinstance(constraints, ConstraintState)
    assert constraints.people_include == ("p_ada",)
    assert out["person_ambiguous"] is False
    assert out["min_picks"] == 3
    assert out["picks"] == ()
    assert out["people_candidates"] == ()


async def test_alias_expands_before_search() -> None:
    rock = _person("p_rock", "Dwayne Johnson", popularity=5.0)
    index = MemoryPeopleIndex([rock, _person("p_other", "Stone Phillips", popularity=9.0)])
    state = empty_turn_state(_ctx(), person_mentions=("the rock",))
    out = await resolve_people(state, searcher=index, aliases=_book())
    constraints = out["constraints"]
    assert isinstance(constraints, ConstraintState)
    assert constraints.people_include == ("p_rock",)


async def test_ambiguous_close_candidates_do_not_guess() -> None:
    a = _person("p_ann", "Ann Smith", popularity=5.0)
    b = _person("p_bob", "Bob Smith", popularity=4.8)
    c = _person("p_cam", "Cam Smith", popularity=4.5)
    index = MemoryPeopleIndex([a, b, c])
    state = empty_turn_state(_ctx(), person_mentions=("Smith",))
    out = await resolve_people(state, searcher=index, aliases=AliasBook(mapping={}))
    constraints = out["constraints"]
    assert isinstance(constraints, ConstraintState)
    assert constraints.people_include == ()
    assert out["person_ambiguous"] is True
    assert out["min_picks"] == 0
    assert out["picks"] == ()
    assert out["route"] is Route.CLARIFY
    assert out["degraded_reason"] is DegradedReason.PERSON_AMBIGUOUS
    candidates = out["people_candidates"]
    assert isinstance(candidates, tuple)
    assert 2 <= len(candidates) <= 3
    assert {person.person_id for person in candidates} <= index.index_ids


async def test_ambiguous_produces_clarify_chips_and_empty_picks() -> None:
    a = _person("p_ann", "Ann Smith", popularity=5.0)
    b = _person("p_bob", "Bob Smith", popularity=4.8)
    index = MemoryPeopleIndex([a, b])
    resolved = await resolve_people(
        empty_turn_state(_ctx(), person_mentions=("Smith",)),
        searcher=index,
        aliases=AliasBook(mapping={}),
    )
    store = FakeSessions(Session.create(user_id="u1", profile_id="p1", session_id="s-people"))
    chip_state = empty_turn_state(
        _ctx(),
        session_id=store.session.session_id,
        person_ambiguous=True,
        route=resolved["route"],
        degraded_reason=resolved["degraded_reason"],
        people_candidates=resolved["people_candidates"],
        min_picks=resolved["min_picks"],
        picks=resolved["picks"],
    )
    chips_out = await mint_chips(chip_state, sessions=store)
    chips = chips_out["chips"]
    assert isinstance(chips, tuple)
    assert len(chips) == 2
    labels = {chip.label for chip in chips}
    assert labels == {"Ann Smith", "Bob Smith"}
    for chip in chips:
        assert set(chip.model_dump()) == {"id", "label"}
        delta = store.session.lookup_chip(chip.id).delta
        assert isinstance(delta.people_include, AddOp)
        assert delta.people_include.values[0] in index.index_ids
    assert resolved["picks"] == ()
    assert resolved["min_picks"] == 0


async def test_zero_hits_applies_era_fallback_not_a_guessed_id() -> None:
    index = MemoryPeopleIndex([_person("p_ada", "Ada Lovelace", year_min=1980, year_max=1999)])
    soft = PersonSoft(role="actor", era_year_min=1990, era_year_max=1999, free_hint="older spy guy")
    state = empty_turn_state(_ctx(), person_mentions=("Nobody McGhost",), person_soft=soft)
    out = await resolve_people(state, searcher=index, aliases=AliasBook(mapping={}))
    constraints = out["constraints"]
    assert isinstance(constraints, ConstraintState)
    assert constraints.people_include == ()
    assert constraints.year_min == 1990
    assert constraints.year_max == 1999
    assert out["person_ambiguous"] is False
    assert out["min_picks"] == 3
    assert out["chip_speech_acts"] == (SpeechAct.PERSON_DISAMBIGUATE,)


async def test_soft_descriptor_with_close_popular_people_clarifies() -> None:
    a = _person("p_a", "Star A", popularity=9.0, year_min=1990, year_max=1999)
    b = _person("p_b", "Star B", popularity=8.5, year_min=1991, year_max=1998)
    index = MemoryPeopleIndex([a, b])
    state = empty_turn_state(
        _ctx(),
        person_soft=PersonSoft(role="actor", era_year_min=1990, era_year_max=1999),
    )
    out = await resolve_people(state, searcher=index, aliases=AliasBook(mapping={}))
    constraints = out["constraints"]
    assert isinstance(constraints, ConstraintState)
    assert constraints.people_include == ()
    assert out["person_ambiguous"] is True
    assert out["min_picks"] == 0
    assert out["picks"] == ()
    ids = {person.person_id for person in cast(tuple[Person, ...], out["people_candidates"])}
    assert ids == {"p_a", "p_b"}


async def test_soft_descriptor_unique_hit_includes_index_id() -> None:
    only = _person("p_only", "Only One", popularity=1.0, roles=(CreditRole.DIRECTOR,))
    extra = _person("p_actor", "Some Actor", popularity=9.0, roles=(CreditRole.ACTOR,))
    index = MemoryPeopleIndex([only, extra])
    state = empty_turn_state(
        _ctx(),
        person_soft=PersonSoft(role="director", era_year_min=1990, era_year_max=2010),
    )
    out = await resolve_people(state, searcher=index, aliases=AliasBook(mapping={}))
    constraints = out["constraints"]
    assert isinstance(constraints, ConstraintState)
    assert constraints.people_include == ("p_only",)


async def test_already_resolved_people_include_is_a_noop() -> None:
    index = MemoryPeopleIndex([_person("p_ada", "Ada Lovelace")])
    prior = ConstraintState(people_include=("p_ada",))
    state = empty_turn_state(_ctx(), constraints=prior)
    out = await resolve_people(state, searcher=index, aliases=AliasBook(mapping={}))
    assert out == {}


async def test_model_supplied_person_id_is_ignored() -> None:
    real = _person("p_ada", "Ada Lovelace", popularity=5.0)
    index = MemoryPeopleIndex([real])
    state = empty_turn_state(_ctx(), person_mentions=("p_deadbeefdeadbeef",))
    stuffed = cast(dict[str, object], state)
    stuffed["person_ids_from_index"] = ("p_deadbeefdeadbeef", "p_ada")
    out = await resolve_people(cast(Any, stuffed), searcher=index, aliases=AliasBook(mapping={}))
    constraints = out["constraints"]
    assert isinstance(constraints, ConstraintState)
    assert "p_deadbeefdeadbeef" not in constraints.people_include
    assert constraints.people_include == ()


async def test_resolved_ids_are_always_subset_of_index() -> None:
    people = (
        _person("p_ada", "Ada Lovelace", popularity=8.0),
        _person("p_al", "Alan Turing", popularity=7.0),
        _person("p_ann", "Ann Smith", popularity=6.0),
        _person("p_bob", "Bob Smith", popularity=5.5),
    )
    index = MemoryPeopleIndex(people)
    cases = [
        empty_turn_state(_ctx(), person_mentions=("Ada Lovelace",)),
        empty_turn_state(_ctx(), person_mentions=("Smith",)),
        empty_turn_state(_ctx(), person_mentions=("Nobody",)),
        empty_turn_state(
            _ctx(),
            person_soft=PersonSoft(role="actor", era_year_min=1990, era_year_max=2010),
        ),
    ]
    for state in cases:
        out = await resolve_people(state, searcher=index, aliases=AliasBook(mapping={}))
        if not out:
            continue
        constraints = out.get("constraints")
        if isinstance(constraints, ConstraintState):
            assert set(constraints.people_include) <= index.index_ids
        raw_candidates = out.get("people_candidates") or ()
        assert isinstance(raw_candidates, tuple)
        for person in raw_candidates:
            assert isinstance(person, Person)
            assert person.person_id in index.index_ids


async def test_searcher_failure_degrades_to_era_fallback() -> None:
    soft = PersonSoft(role="actor", era_year_min=1990, era_year_max=1999)
    state = empty_turn_state(_ctx(), person_mentions=("Ada",), person_soft=soft)
    out = await resolve_people(state, searcher=BoomSearcher(), aliases=AliasBook(mapping={}))
    constraints = out["constraints"]
    assert isinstance(constraints, ConstraintState)
    assert constraints.people_include == ()
    assert constraints.year_min == 1990
    assert out["person_ambiguous"] is False


async def test_missing_searcher_degrades_never_raises() -> None:
    state = empty_turn_state(_ctx(), person_mentions=("Ada Lovelace",))
    out = await resolve_people(state, searcher=None, aliases=AliasBook(mapping={}))
    constraints = out["constraints"]
    assert isinstance(constraints, ConstraintState)
    assert constraints.people_include == ()


async def test_theta_from_settings() -> None:
    close = _person("p_ann", "Ann Close", popularity=1.0)
    index = MemoryPeopleIndex([close])
    # "Ann" is a first-name-only match (0.58). Default theta 0.75 → zero.
    state = empty_turn_state(_ctx(), person_mentions=("Ann",))
    out_low = await resolve_people(
        state,
        searcher=index,
        aliases=AliasBook(mapping={}),
        settings=default_settings,
    )
    constraints_low = out_low["constraints"]
    assert isinstance(constraints_low, ConstraintState)
    assert constraints_low.people_include == ()
    high = default_settings.model_copy(update={"person_theta": 0.50})
    out_high = await resolve_people(
        state, searcher=index, aliases=AliasBook(mapping={}), settings=high
    )
    constraints_high = out_high["constraints"]
    assert isinstance(constraints_high, ConstraintState)
    assert constraints_high.people_include == ("p_ann",)


# ---------------------------------------------------------------------------
# Live Elasticsearch via the people alias
# ---------------------------------------------------------------------------


async def test_golden_slice_person_at_1(live_searcher: EsPeopleSearcher) -> None:
    cases = _golden()
    assert len(cases) >= 15
    hits = 0
    misses: list[str] = []
    for case in cases:
        query = case["query"]
        expect = case["expect_name"]
        state = empty_turn_state(_ctx(), person_mentions=(query,))
        out = await resolve_people(state, searcher=live_searcher, aliases=_book())
        constraints = out.get("constraints")
        if not isinstance(constraints, ConstraintState) or len(constraints.people_include) != 1:
            misses.append(
                f"{query!r} -> no single include (ambiguous={out.get('person_ambiguous')})"
            )
            continue
        person_id = constraints.people_include[0]
        confirm = await resolve_people(
            empty_turn_state(_ctx(), person_mentions=(expect,)),
            searcher=live_searcher,
            aliases=_book(),
        )
        confirm_c = confirm.get("constraints")
        if not isinstance(confirm_c, ConstraintState) or confirm_c.people_include != (person_id,):
            got = confirm_c.people_include if isinstance(confirm_c, ConstraintState) else None
            misses.append(f"{query!r} -> {person_id} expected_id={got} name={expect}")
            continue
        hits += 1
    person_at_1 = hits / len(cases)
    assert not misses, f"person@1={person_at_1:.2f} misses={misses}"
    assert hits == len(cases)
    assert person_at_1 == 1.0


async def test_live_ambiguous_smith_clarifies_instead_of_guessing(
    live_searcher: EsPeopleSearcher,
) -> None:
    state = empty_turn_state(_ctx(), person_mentions=("Smith",))
    out = await resolve_people(state, searcher=live_searcher, aliases=AliasBook(mapping={}))
    constraints = out["constraints"]
    assert isinstance(constraints, ConstraintState)
    assert constraints.people_include == ()
    assert out["person_ambiguous"] is True
    assert out["min_picks"] == 0
    assert out["picks"] == ()
    candidates = out["people_candidates"]
    assert isinstance(candidates, tuple)
    assert 2 <= len(candidates) <= 3
    for person in candidates:
        assert person.person_id.startswith("p_")
        assert "smith" in person.name.lower()

    store = FakeSessions(Session.create(user_id="u1", profile_id="p1", session_id="s-smith"))
    chips_out = await mint_chips(
        empty_turn_state(
            _ctx(),
            session_id=store.session.session_id,
            route=Route.CLARIFY,
            degraded_reason=DegradedReason.PERSON_AMBIGUOUS,
            people_candidates=candidates,
            min_picks=0,
            picks=(),
        ),
        sessions=store,
    )
    chips = chips_out["chips"]
    assert isinstance(chips, tuple)
    assert len(chips) >= 2
    for chip in chips:
        assert set(chip.model_dump()) == {"id", "label"}


async def test_live_ids_come_from_people_alias(live_searcher: EsPeopleSearcher) -> None:
    state = empty_turn_state(_ctx(), person_mentions=("Adam Sandler",))
    out = await resolve_people(state, searcher=live_searcher, aliases=_book())
    constraints = out["constraints"]
    assert isinstance(constraints, ConstraintState)
    assert len(constraints.people_include) == 1
    person_id = constraints.people_include[0]
    confirmed = await live_searcher.search("Adam Sandler", size=3)
    assert any(
        person.person_id == person_id and person.name == "Adam Sandler" for person in confirmed
    )
