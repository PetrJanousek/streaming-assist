"""Intent node: chip / rules / cache skip the model; LLM failures degrade."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import Field

from assist.domain.constraints import (
    AddOp,
    ClearOp,
    ConstraintDelta,
    ConstraintState,
    RemoveOp,
    ReplaceOp,
    SetOp,
)
from assist.domain.context import ServerUserCtx
from assist.domain.enums import DeviceClass, GenreId, MaturityRating, MediaType, Package, SpeechAct
from assist.graph.state import PersonSoft, empty_turn_state
from assist.llm.cost import CostCallbackHandler
from assist.llm.gateway import UnavailableChatModel
from assist.llm.prompts import load_prompt
from assist.nodes import intent as intent_mod
from assist.nodes.intent import (
    IntentClass,
    IntentOpWire,
    IntentUpdate,
    IntentUpdateWire,
    match_rules,
    normalize_text,
    run_intent,
    to_constraint_delta,
    to_intent_update,
    to_wire,
)
from assist.stores.cache import constraints_hash
from assist.stores.session import ChipInvalid, Session

INTENT_PATH = Path(intent_mod.__file__).resolve()


class FakeCache:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}
        self.gets = 0
        self.sets = 0

    async def get_intent(self, norm_text: str, constraints_hash_value: str) -> str | None:
        self.gets += 1
        return self.store.get((norm_text, constraints_hash_value))

    async def set_intent(self, norm_text: str, constraints_hash_value: str, payload: str) -> None:
        self.sets += 1
        self.store[(norm_text, constraints_hash_value)] = payload


class _BoomChat(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "boom"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise AssertionError("model must not be called")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise AssertionError("model must not be called")

    def with_structured_output(
        self,
        schema: dict[str, Any] | type[Any],
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        raise AssertionError("model must not be called")


class _FixedChat(BaseChatModel):
    canned: IntentUpdateWire
    call_log: list[int] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fixed"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise AssertionError("raw generate must not run")

    def with_structured_output(
        self,
        schema: dict[str, Any] | type[Any],
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        def _run(_input: Any) -> IntentUpdateWire:
            self.call_log.append(1)
            return self.canned

        return RunnableLambda(_run)


class _ParserFailChat(BaseChatModel):
    call_log: list[int] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "parser-fail"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise AssertionError("raw generate must not run")

    def with_structured_output(
        self,
        schema: dict[str, Any] | type[Any],
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        def _raise(_input: Any) -> Any:
            self.call_log.append(1)
            raise OutputParserException("bad json")

        return RunnableLambda(_raise)


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


@pytest.fixture
def no_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("structured_output must not be called")

    monkeypatch.setattr("assist.nodes.intent.structured_output", _boom)


def test_intent_module_does_not_bind_tools() -> None:
    source = INTENT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(INTENT_PATH))
    imported_gateway_helper = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "with_structured_output":
            pytest.fail("call assist.llm.gateway.structured_output, not with_structured_output")
        if isinstance(node, ast.ImportFrom) and node.module == "assist.llm.gateway":
            imported_gateway_helper = any(alias.name == "structured_output" for alias in node.names)
    assert imported_gateway_helper
    assert "structured_output(" in source


def test_prompt_forbids_titles_and_ids() -> None:
    text = load_prompt("intent")
    assert "Never name a title from memory" in text
    assert "catalog_id" in text
    assert "person_id" in text
    assert "person_ids_from_index as an empty list" in text


@pytest.mark.parametrize(
    ("text", "intent_class", "check"),
    [
        (
            "comedy",
            IntentClass.PURE_GENRE_FACET,
            lambda u: u.constraint_delta.genres_include == AddOp(values=(GenreId.COMEDY.value,)),
        ),
        (
            "comedy movies",
            IntentClass.PURE_GENRE_FACET,
            lambda u: u.constraint_delta.media_type == SetOp(value=MediaType.FILM.value),
        ),
        (
            "90s",
            IntentClass.PURE_DECADE,
            lambda u: (
                u.constraint_delta.year_min == SetOp(value=1990)
                and u.constraint_delta.year_max == SetOp(value=1999)
            ),
        ),
        (
            "the 2010s",
            IntentClass.PURE_DECADE,
            lambda u: (
                u.constraint_delta.year_min == SetOp(value=2010)
                and u.constraint_delta.year_max == SetOp(value=2019)
            ),
        ),
        (
            "under 90 minutes",
            IntentClass.DURATION_ONLY,
            lambda u: u.constraint_delta.duration_max_min == SetOp(value=90),
        ),
        (
            "under 2 hours",
            IntentClass.DURATION_ONLY,
            lambda u: u.constraint_delta.duration_max_min == SetOp(value=120),
        ),
        (
            "movies",
            IntentClass.MEDIA_TYPE,
            lambda u: u.constraint_delta.media_type == SetOp(value=MediaType.FILM.value),
        ),
        (
            '"The Irishman"',
            IntentClass.KNOWN_TITLE_LOOKUP,
            lambda u: u.query_rewrite == "The Irishman",
        ),
        (
            "reset",
            IntentClass.RESET,
            lambda u: u.constraint_delta.reset_soft is True,
        ),
    ],
)
def test_match_rules_closed_classes(text: str, intent_class: IntentClass, check: Any) -> None:
    hit = match_rules(text)
    assert hit is not None
    assert hit.intent_class is intent_class
    assert check(hit)


def test_mixed_query_is_not_a_rules_hit() -> None:
    assert match_rules("90s comedy") is None
    assert match_rules("something cozy for a rainy night") is None


async def test_chip_path_makes_zero_model_calls(no_llm: None) -> None:
    session = Session.create(user_id="u1", profile_id="p1")
    delta = ConstraintDelta(moods=AddOp(values=("funny",)))
    session, chip = session.mint_chip(
        label="Something funnier",
        delta=delta,
        speech_act=SpeechAct.REFINE_MOOD,
    )
    state = empty_turn_state(_ctx(), chip_id=chip.chip_id, message_type="chip", text="ignored")
    out = await run_intent(state, chips=session, cache=FakeCache(), model=_BoomChat())
    assert out["intent_source"] == "chip"
    assert out["delta"] == delta
    assert out["intent_class"] == IntentClass.MOOD_GENRE.value


async def test_unknown_chip_raises_chip_invalid(no_llm: None) -> None:
    session = Session.create(user_id="u1", profile_id="p1")
    state = empty_turn_state(_ctx(), chip_id="c_missing", message_type="chip")
    with pytest.raises(ChipInvalid) as exc:
        await run_intent(state, chips=session, model=_BoomChat())
    assert exc.value.error_type == "chip_invalid"


@pytest.mark.parametrize(
    "text",
    ["comedy", "90s", "under 90 minutes", "movies", '"The Irishman"', "reset"],
)
async def test_rules_path_makes_zero_model_calls(no_llm: None, text: str) -> None:
    state = empty_turn_state(_ctx(), text=text)
    out = await run_intent(state, cache=FakeCache(), model=_BoomChat())
    assert out["intent_source"] == "rules"
    assert isinstance(out["delta"], ConstraintDelta)


async def test_cache_hit_makes_zero_model_calls(no_llm: None) -> None:
    cache = FakeCache()
    text = "something cozy for a rainy night"
    state = empty_turn_state(_ctx(), text=text)
    cached = IntentUpdate(
        intent_class=IntentClass.MOOD_GENRE,
        query_rewrite="cozy rainy night",
        constraint_delta=ConstraintDelta(moods=AddOp(values=("cozy",))),
        person_ids_from_index=("p_should_ignore",),
    )
    await cache.set_intent(
        normalize_text(text),
        constraints_hash(state["constraints"]),
        cached.model_dump_json(),
    )
    out = await run_intent(state, cache=cache, model=_BoomChat())
    assert out["intent_source"] == "llm"
    assert out["query_rewrite"] == "cozy rainy night"
    assert "person_ids_from_index" not in out
    assert cache.gets == 1
    assert cache.sets == 1  # only the pre-seed write in this test


async def test_malformed_model_response_degrades_to_rules() -> None:
    model = _ParserFailChat()
    state = empty_turn_state(_ctx(), text="something cozy for a rainy night")
    out = await run_intent(state, cache=FakeCache(), model=model)
    assert out["intent_source"] == "rules"
    assert isinstance(out["delta"], ConstraintDelta)
    assert model.call_log  # the model ran, then schema failed


async def test_llm_unavailable_degrades_to_rules_not_raise() -> None:
    # Force unavailability explicitly rather than relying on the environment
    # having no ANTHROPIC_API_KEY: a real key must never make this test place
    # a live call (T29 -- a second agent owns the live-provider acceptance
    # criterion; this suite is fakes-only).
    state = empty_turn_state(_ctx(), text="something cozy for a rainy night")
    out = await run_intent(state, cache=FakeCache(), model=UnavailableChatModel())
    assert out["intent_source"] == "rules"
    assert isinstance(out["delta"], ConstraintDelta)


async def test_person_ids_from_index_ignored() -> None:
    # The wire schema has no person_ids_from_index field at all -- the model
    # cannot emit one. This test now covers the still-relevant guarantee that
    # people_include/people_exclude ops never survive the adapter.
    canned = IntentUpdateWire(
        intent_class=IntentClass.PEOPLE_FUZZY.value,
        query_rewrite="older spy guy 90s",
        ops=(
            IntentOpWire(field="people_include", op="add", value="p_abc"),
            IntentOpWire(field="year_min", op="set", value="1990"),
            IntentOpWire(field="year_max", op="set", value="1999"),
        ),
        person_role="actor",
        person_era_year_min=1990,
        person_era_year_max=1999,
        person_free_hint="older spy",
        person_mentions=("the spy guy",),
    )
    model = _FixedChat(canned=canned)
    cache = FakeCache()
    out = await run_intent(
        empty_turn_state(_ctx(), text="the older spy guy from the 90s"),
        cache=cache,
        model=model,
    )
    delta = out["delta"]
    assert isinstance(delta, ConstraintDelta)
    assert delta.people_include is None
    assert delta.people_exclude is None
    assert delta.year_min == SetOp(value=1990)
    assert out["person_mentions"] == ("the spy guy",)
    assert out["person_soft"] == PersonSoft(
        role="actor", era_year_min=1990, era_year_max=1999, free_hint="older spy"
    )
    assert "person_ids_from_index" not in out
    stored = IntentUpdate.model_validate_json(next(iter(cache.store.values())))
    assert stored.person_ids_from_index == ()
    assert stored.constraint_delta.people_include is None


async def test_llm_success_writes_cache_and_records_llm_source() -> None:
    canned = IntentUpdateWire(
        intent_class=IntentClass.MOOD_GENRE.value,
        query_rewrite="cozy rainy night",
        ops=(IntentOpWire(field="moods", op="add", value="cozy"),),
    )
    model = _FixedChat(canned=canned)
    cache = FakeCache()
    text = "something cozy for a rainy night"
    state = empty_turn_state(_ctx(), text=text, constraints=ConstraintState.empty())
    out = await run_intent(state, cache=cache, model=model)
    assert out["intent_source"] == "llm"
    assert model.call_log == [1]
    assert cache.sets == 1
    again = await run_intent(state, cache=cache, model=_BoomChat())
    assert again["intent_source"] == "llm"
    assert again["query_rewrite"] == "cozy rainy night"


async def test_uses_gateway_structured_output_with_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def _fake(
        schema: type[IntentUpdateWire],
        **kwargs: Any,
    ) -> Runnable[Any, IntentUpdateWire]:
        seen["schema"] = schema
        seen["fallback"] = kwargs.get("fallback")

        def _run(_input: Any) -> IntentUpdateWire:
            return IntentUpdateWire(
                intent_class=IntentClass.MOOD_GENRE.value,
                query_rewrite="cozy",
            )

        return RunnableLambda(_run)

    monkeypatch.setattr("assist.nodes.intent.structured_output", _fake)
    out = await run_intent(
        empty_turn_state(_ctx(), text="something cozy for a rainy night"),
        cache=FakeCache(),
    )
    assert seen["schema"] is IntentUpdateWire
    assert seen["fallback"] is not None
    assert out["intent_source"] == "llm"


async def test_cost_callback_is_passed_on_llm_path(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}
    handler = CostCallbackHandler()

    def _fake(schema: type[IntentUpdateWire], **kwargs: Any) -> Runnable[Any, IntentUpdateWire]:
        assert kwargs.get("fallback") is not None

        def _run(_input: Any, config: RunnableConfig) -> IntentUpdateWire:
            seen["config"] = config
            return IntentUpdateWire(intent_class=IntentClass.OTHER.value, query_rewrite="x")

        return RunnableLambda(_run)

    monkeypatch.setattr("assist.nodes.intent.structured_output", _fake)
    await run_intent(
        empty_turn_state(_ctx(), text="something cozy for a rainy night"),
        cache=FakeCache(),
        cost_handler=handler,
    )
    config = seen["config"]
    assert isinstance(config, dict)
    callbacks = config.get("callbacks")
    assert callbacks is not None
    raw_handlers = getattr(callbacks, "handlers", None)
    assert isinstance(raw_handlers, list)
    assert handler in raw_handlers


# --- T29: to_constraint_delta / to_intent_update adapter -------------------


def test_adapter_scalar_ops_produce_correctly_typed_sets() -> None:
    delta = to_constraint_delta(
        [
            IntentOpWire(field="media_type", op="set", value="film"),
            IntentOpWire(field="year_min", op="set", value="1990"),
            IntentOpWire(field="duration_max_min", op="set", value="90"),
        ]
    )
    assert delta.media_type == SetOp(value="film")
    assert delta.year_min == SetOp(value=1990)
    assert delta.duration_max_min == SetOp(value=90)


def test_adapter_bool_field_coerces() -> None:
    delta = to_constraint_delta(
        [IntentOpWire(field="local_originals_only", op="set", value="true")]
    )
    assert delta.local_originals_only == SetOp(value=True)


def test_adapter_coalesces_multiple_ops_on_one_list_field() -> None:
    delta = to_constraint_delta(
        [
            IntentOpWire(field="genres_include", op="add", value="comedy"),
            IntentOpWire(field="genres_include", op="add", value="drama"),
            IntentOpWire(field="genres_include", op="add", value="comedy"),  # dup
        ]
    )
    assert delta.genres_include == AddOp(values=("comedy", "drama"))


def test_adapter_coalesces_remove_and_replace_ops_too() -> None:
    removed = to_constraint_delta(
        [
            IntentOpWire(field="genres_exclude", op="remove", value="horror"),
            IntentOpWire(field="genres_exclude", op="remove", value="crime"),
        ]
    )
    assert removed.genres_exclude == RemoveOp(values=("horror", "crime"))

    replaced = to_constraint_delta(
        [
            IntentOpWire(field="origins", op="replace", value="France"),
            IntentOpWire(field="origins", op="replace", value="Japan"),
        ]
    )
    assert replaced.origins == ReplaceOp(values=("France", "Japan"))


def test_adapter_drops_unknown_field_but_keeps_the_rest() -> None:
    delta = to_constraint_delta(
        [
            IntentOpWire(field="not_a_real_field", op="set", value="whatever"),
            IntentOpWire(field="media_type", op="set", value="series"),
        ]
    )
    assert delta.media_type == SetOp(value="series")


@pytest.mark.parametrize(
    "ops",
    [
        [{"field": "year_min", "op": "bogus_op", "value": "1990"}],
        [{"field": "year_min", "op": "set", "value": ""}],
        [{"field": "year_min", "op": "set", "value": "not-a-number"}],
        [{"field": "genres_include", "op": "set", "value": "comedy"}],  # set on list field
        [{"field": "media_type", "op": "add", "value": "film"}],  # add on scalar field
        [],
        [None],
        ["garbage"],
        [{"field": 123, "op": None, "value": []}],
    ],
)
def test_adapter_malformed_input_drops_cleanly_never_raises(ops: list[object]) -> None:
    delta = to_constraint_delta(ops)
    assert isinstance(delta, ConstraintDelta)
    assert delta == ConstraintDelta()


def test_adapter_completely_garbage_payload_never_raises() -> None:
    assert to_constraint_delta(None) == ConstraintDelta()


def test_adapter_drops_languages_and_people_ops() -> None:
    delta = to_constraint_delta(
        [
            IntentOpWire(field="languages", op="add", value="french"),
            IntentOpWire(field="people_include", op="add", value="p_1"),
            IntentOpWire(field="people_exclude", op="add", value="p_2"),
        ]
    )
    assert delta == ConstraintDelta()


def test_adapter_reset_soft_round_trips() -> None:
    assert to_constraint_delta([], reset_soft=True).reset_soft is True
    assert (
        to_constraint_delta([IntentOpWire(field="reset_soft", op="set", value="true")]).reset_soft
        is True
    )
    assert (
        to_constraint_delta(
            [IntentOpWire(field="reset_soft", op="set", value="false")], reset_soft=True
        ).reset_soft
        is False
    )


def test_adapter_clear_works_on_list_and_scalar_fields() -> None:
    delta = to_constraint_delta(
        [
            IntentOpWire(field="genres_include", op="clear", value=""),
            IntentOpWire(field="year_min", op="clear", value=""),
        ]
    )
    assert delta.genres_include == ClearOp()
    assert delta.year_min == ClearOp()


def test_adapter_does_not_mutate_input_and_is_deterministic() -> None:
    ops = (
        IntentOpWire(field="genres_include", op="add", value="comedy"),
        IntentOpWire(field="media_type", op="set", value="film"),
    )
    snapshot = tuple(ops)
    first = to_constraint_delta(ops)
    second = to_constraint_delta(ops)
    assert ops == snapshot
    assert first == second


@pytest.mark.parametrize(
    ("text", "wire_ops", "check"),
    [
        (
            "Czech movies",
            [
                IntentOpWire(field="media_type", op="set", value="film"),
                IntentOpWire(field="origins", op="add", value="Czech Republic"),
            ],
            lambda d: (
                d.media_type == SetOp(value="film")
                and d.origins == AddOp(values=("Czech Republic",))
            ),
        ),
        (
            "korean thrillers",
            [
                IntentOpWire(field="origins", op="add", value="South Korea"),
                IntentOpWire(field="genres_include", op="add", value="thriller"),
            ],
            lambda d: (
                d.origins == AddOp(values=("South Korea",))
                and d.genres_include == AddOp(values=("thriller",))
            ),
        ),
        (
            "scary movies under 90 minutes",
            [
                IntentOpWire(field="media_type", op="set", value="film"),
                IntentOpWire(field="moods", op="add", value="scary"),
                IntentOpWire(field="duration_max_min", op="set", value="90"),
            ],
            lambda d: (
                d.media_type == SetOp(value="film")
                and d.moods == AddOp(values=("scary",))
                and d.duration_max_min == SetOp(value=90)
            ),
        ),
    ],
)
async def test_end_to_end_flat_wire_matches_old_nested_shape(
    text: str, wire_ops: list[IntentOpWire], check: Any
) -> None:
    canned = IntentUpdateWire(
        intent_class=IntentClass.MOOD_GENRE.value,
        query_rewrite=text,
        ops=tuple(wire_ops),
    )
    model = _FixedChat(canned=canned)
    out = await run_intent(
        empty_turn_state(_ctx(), text=text),
        cache=FakeCache(),
        model=model,
    )
    delta = out["delta"]
    assert isinstance(delta, ConstraintDelta)
    assert check(delta)


def test_to_intent_update_preserves_person_soft() -> None:
    wire = IntentUpdateWire(
        intent_class=IntentClass.PEOPLE_FUZZY.value,
        query_rewrite="rewrite",
        ops=(IntentOpWire(field="year_min", op="set", value="1990"),),
        person_role="actor",
        person_era_year_min=1990,
        person_era_year_max=1999,
        person_free_hint="older spy",
        person_mentions=("some guy",),
    )
    update = to_intent_update(wire)
    assert isinstance(update, IntentUpdate)
    assert update.person_soft == PersonSoft(
        role="actor", era_year_min=1990, era_year_max=1999, free_hint="older spy"
    )
    assert update.person_mentions == ("some guy",)
    assert update.constraint_delta.year_min == SetOp(value=1990)


@pytest.mark.parametrize(
    "delta",
    [
        ConstraintDelta(media_type=SetOp(value="film")),
        ConstraintDelta(genres_include=AddOp(values=("thriller", "drama"))),
        ConstraintDelta(origins=ClearOp()),
        ConstraintDelta(reset_soft=True, moods=AddOp(values=("scary",))),
        ConstraintDelta(
            year_min=SetOp(value=1990),
            year_max=SetOp(value=1999),
            duration_max_min=SetOp(value=90),
            local_originals_only=SetOp(value=True),
        ),
        ConstraintDelta(genres_exclude=ReplaceOp(values=("horror",))),
        ConstraintDelta(recency_bias=SetOp(value="new")),
    ],
)
def test_to_wire_round_trips_constraint_delta(delta: ConstraintDelta) -> None:
    update = IntentUpdate(
        intent_class=IntentClass.MOOD_GENRE, query_rewrite="rt", constraint_delta=delta
    )
    round_tripped = to_intent_update(to_wire(update))
    assert round_tripped.constraint_delta == delta
