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

from assist.domain.constraints import AddOp, ConstraintDelta, ConstraintState, SetOp
from assist.domain.context import ServerUserCtx
from assist.domain.enums import DeviceClass, GenreId, MaturityRating, MediaType, Package, SpeechAct
from assist.graph.state import PersonSoft, empty_turn_state
from assist.llm.cost import CostCallbackHandler
from assist.llm.prompts import load_prompt
from assist.nodes import intent as intent_mod
from assist.nodes.intent import (
    IntentClass,
    IntentUpdate,
    match_rules,
    normalize_text,
    run_intent,
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
    canned: IntentUpdate
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
        def _run(_input: Any) -> IntentUpdate:
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
    state = empty_turn_state(_ctx(), text="something cozy for a rainy night")
    out = await run_intent(state, cache=FakeCache())
    assert out["intent_source"] == "rules"
    assert isinstance(out["delta"], ConstraintDelta)


async def test_person_ids_from_index_ignored() -> None:
    canned = IntentUpdate(
        intent_class=IntentClass.PEOPLE_FUZZY,
        query_rewrite="older spy guy 90s",
        constraint_delta=ConstraintDelta(
            people_include=AddOp(values=("p_abc",)),
            year_min=SetOp(value=1990),
            year_max=SetOp(value=1999),
        ),
        person_soft=PersonSoft(
            role="actor",
            era_year_min=1990,
            era_year_max=1999,
            free_hint="older spy",
        ),
        person_mentions=("the spy guy",),
        person_ids_from_index=("p_abc", "p_def"),
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
    assert out["person_soft"] == canned.person_soft
    assert "person_ids_from_index" not in out
    stored = IntentUpdate.model_validate_json(next(iter(cache.store.values())))
    assert stored.person_ids_from_index == ()
    assert stored.constraint_delta.people_include is None


async def test_llm_success_writes_cache_and_records_llm_source() -> None:
    canned = IntentUpdate(
        intent_class=IntentClass.MOOD_GENRE,
        query_rewrite="cozy rainy night",
        constraint_delta=ConstraintDelta(moods=AddOp(values=("cozy",))),
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
        schema: type[IntentUpdate],
        **kwargs: Any,
    ) -> Runnable[Any, IntentUpdate]:
        seen["schema"] = schema
        seen["fallback"] = kwargs.get("fallback")

        def _run(_input: Any) -> IntentUpdate:
            return IntentUpdate(
                intent_class=IntentClass.MOOD_GENRE,
                query_rewrite="cozy",
            )

        return RunnableLambda(_run)

    monkeypatch.setattr("assist.nodes.intent.structured_output", _fake)
    out = await run_intent(
        empty_turn_state(_ctx(), text="something cozy for a rainy night"),
        cache=FakeCache(),
    )
    assert seen["schema"] is IntentUpdate
    assert seen["fallback"] is not None
    assert out["intent_source"] == "llm"


async def test_cost_callback_is_passed_on_llm_path(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}
    handler = CostCallbackHandler()

    def _fake(schema: type[IntentUpdate], **kwargs: Any) -> Runnable[Any, IntentUpdate]:
        assert kwargs.get("fallback") is not None

        def _run(_input: Any, config: RunnableConfig) -> IntentUpdate:
            seen["config"] = config
            return IntentUpdate(intent_class=IntentClass.OTHER, query_rewrite="x")

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
