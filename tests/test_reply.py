"""Generative reply node: indices only, zero retries, schema miss → template."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import Field, ValidationError

from assist.config import LLMProvider, Settings
from assist.domain.catalog import Candidate, Pick
from assist.domain.context import ServerUserCtx
from assist.domain.enums import (
    DegradedReason,
    DeviceClass,
    GenreId,
    MaturityRating,
    MediaType,
    Package,
    Route,
    SpeechAct,
)
from assist.graph.state import TurnState, empty_turn_state
from assist.llm.cost import CostCallbackHandler
from assist.llm.gateway import LLMTimeout
from assist.llm.prompts import load_prompt
from assist.nodes import reply as reply_mod
from assist.nodes.reply import (
    GroundedReply,
    format_candidate_cards,
    reply_generative,
)
from assist.nodes.sanitize import sanitize

REPLY_PATH = Path(reply_mod.__file__).resolve()
_PROMPTS = Path(__file__).resolve().parents[1] / "src" / "assist" / "llm" / "prompts"
PROMPT_PATH = _PROMPTS / "reply.md"


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
    canned: GroundedReply
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
        def _run(_input: Any) -> GroundedReply:
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


class _TimeoutChat(BaseChatModel):
    call_log: list[int] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "timeout"

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
            raise LLMTimeout("LLM call exceeded 50ms")

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


def _c(
    catalog_id: str,
    title: str,
    *,
    year: int | None = 2019,
    genres: tuple[GenreId, ...] = (GenreId.DRAMA, GenreId.CRIME),
) -> Candidate:
    return Candidate(
        catalog_id=catalog_id,
        title=title,
        media_type=MediaType.FILM,
        release_year=year,
        genres=genres,
        score=1.0,
    )


CANDIDATES = (
    _c("ttl_a", "The Irishman"),
    _c("ttl_b", "Heat", year=1995),
    _c("ttl_c", "Casino", year=1995),
    _c("ttl_d", "Goodfellas", year=1990),
)
ENTITLED = ("ttl_a", "ttl_b", "ttl_c", "ttl_d")


def _state(**overrides: object) -> TurnState:
    payload: dict[str, object] = {
        "text": "something tense for tonight",
        "candidates": CANDIDATES,
        "entitled_ids": ENTITLED,
        "route": Route.GENERATIVE,
        "intent_class": "other",
        "intent_source": "llm",
    }
    payload.update(overrides)
    return empty_turn_state(_ctx(), **payload)


def _merged(state: TurnState, update: dict[str, object]) -> TurnState:
    return cast(TurnState, {**state, **update})


def _pick_ids(out: dict[str, object]) -> tuple[str, ...]:
    picks = out["picks"]
    assert isinstance(picks, tuple)
    return tuple(p.catalog_id for p in picks if isinstance(p, Pick))


def _settings(**kwargs: Any) -> Settings:
    values: dict[str, Any] = {
        "llm_provider": LLMProvider.ANTHROPIC,
        "anthropic_api_key": "sk-test",
    }
    values.update(kwargs)
    return Settings(**values)


def test_reply_module_does_not_bind_tools() -> None:
    source = REPLY_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(REPLY_PATH))
    imported_gateway_helper = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "with_structured_output":
            pytest.fail("call assist.llm.gateway.structured_output, not with_structured_output")
        if isinstance(node, ast.ImportFrom) and node.module == "assist.llm.gateway":
            imported_gateway_helper = any(alias.name == "structured_output" for alias in node.names)
    assert imported_gateway_helper
    assert "structured_output(" in source
    assert "retries=0" in source


def test_prompt_forbids_titles_ids_and_free_chip_labels() -> None:
    text = load_prompt("reply")
    assert PROMPT_PATH.is_file()
    assert "Never invent a title, catalog_id, or person_id from memory" in text
    assert "0-based indices" in text
    assert "Never a catalog_id" in text
    assert "Do not write chip labels" in text
    assert "numbered candidate list" in text


def test_format_candidate_cards_are_numbered_and_omit_ids() -> None:
    blob = format_candidate_cards(CANDIDATES)
    assert blob.startswith("[0] The Irishman")
    assert "[1] Heat" in blob
    assert "[3] Goodfellas" in blob
    for catalog_id in ("ttl_a", "ttl_b", "ttl_c", "ttl_d"):
        assert catalog_id not in blob
    assert "catalog_id" not in blob


def test_grounded_reply_drops_unknown_speech_act_not_schema() -> None:
    parsed = GroundedReply.model_validate(
        {
            "reply": 'Start with "The Irishman".',
            "pick_indices": [0, 1],
            "chip_speech_acts": ["refine_mood", "not_a_real_act", "reset_soft"],
        }
    )
    assert parsed.chip_speech_acts == (SpeechAct.REFINE_MOOD, SpeechAct.RESET_SOFT)


def test_grounded_reply_keeps_out_of_range_indices_for_sanitize() -> None:
    parsed = GroundedReply.model_validate(
        {"reply": "ok", "pick_indices": [0, 99, -1, 1], "chip_speech_acts": []}
    )
    assert parsed.pick_indices == (0, 99, -1, 1)


def test_grounded_reply_forbids_catalog_id_field() -> None:
    with pytest.raises(ValidationError):
        GroundedReply.model_validate({"reply": "x", "pick_indices": [0], "catalog_id": "ttl_a"})


async def test_out_of_range_index_is_dropped_by_sanitize_not_honoured() -> None:
    canned = GroundedReply(
        reply='Start with "The Irishman" if you want a long night in.',
        pick_indices=(0, 99, -1, 1),
        chip_speech_acts=(SpeechAct.REFINE_MOOD,),
    )
    gen = await reply_generative(
        _state(),
        model=_FixedChat(canned=canned),
        settings=_settings(),
    )
    assert gen["model_pick_indices"] == (0, 99, -1, 1)
    assert gen["model_pick_ids"] == ()
    out = await sanitize(_merged(_state(), gen))
    ids = _pick_ids(out)
    # 99 and -1 must not wrap, invent, or substitute; 0 and 1 map to a and b.
    assert ids[0] == "ttl_a"
    assert "ttl_b" in ids
    assert set(ids) <= set(ENTITLED)
    assert len(ids) <= 8


async def test_schema_failure_is_template_plus_ranker_picks() -> None:
    model = _ParserFailChat()
    gen = await reply_generative(_state(), model=model, settings=_settings())
    assert model.call_log == [1]
    assert gen["degraded_reason"] is DegradedReason.GENERATIVE_SCHEMA_FAIL
    assert gen["route"] is Route.DEGRADED_KEYWORD
    assert gen["model_pick_indices"] == ()
    assert gen["model_pick_ids"] == ()
    assert str(gen["reply"]).strip()
    assert "Ghost" not in str(gen["reply"])
    out = await sanitize(_merged(_state(), gen))
    ids = _pick_ids(out)
    assert ids == ("ttl_a", "ttl_b", "ttl_c")
    assert out["min_picks"] == 3


async def test_uses_gateway_structured_output_with_zero_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    calls = {"n": 0}

    def _fake(schema: type[GroundedReply], **kwargs: Any) -> Runnable[Any, GroundedReply]:
        seen["schema"] = schema
        seen["retries"] = kwargs.get("retries")
        seen["fallback"] = kwargs.get("fallback")

        def _run(_input: Any) -> GroundedReply:
            calls["n"] += 1
            return GroundedReply(
                reply='Try "The Irishman" tonight.',
                pick_indices=(0,),
                chip_speech_acts=(SpeechAct.REFINE_DURATION,),
            )

        return RunnableLambda(_run)

    monkeypatch.setattr("assist.nodes.reply.structured_output", _fake)
    out = await reply_generative(_state(), settings=_settings())
    assert seen["schema"] is GroundedReply
    assert seen["retries"] == 0
    assert seen["fallback"] is not None
    assert calls["n"] == 1
    assert out["route"] is Route.GENERATIVE
    assert out["model_pick_indices"] == (0,)


async def test_reply_length_cap_enforced() -> None:
    long_reply = (
        'Start with "The Irishman" if you want a very long night in with friends '
        "and then keep going for hours after that."
    )
    canned = GroundedReply(reply=long_reply, pick_indices=(0,), chip_speech_acts=())
    cfg = _settings(reply_max_chars=40)
    gen = await reply_generative(_state(), model=_FixedChat(canned=canned), settings=cfg)
    assert len(str(gen["reply"])) <= 40
    assert "hours after that" not in str(gen["reply"])


async def test_off_catalog_title_in_model_reply_is_caught_by_sanitize() -> None:
    canned = GroundedReply(
        reply='Tonight try "Ghost Movie" with one of these.',
        pick_indices=(0,),
        chip_speech_acts=(),
    )
    gen = await reply_generative(
        _state(),
        model=_FixedChat(canned=canned),
        settings=_settings(),
    )
    assert "Ghost Movie" in str(gen["reply"])
    out = await sanitize(_merged(_state(), gen))
    assert "Ghost Movie" not in str(out["reply"])
    assert "Tonight try" in str(out["reply"])


async def test_success_writes_indices_not_ids_and_filters_chips() -> None:
    canned = GroundedReply(
        reply='Start with "The Irishman" if you want a long night in.',
        pick_indices=(0, 2),
        chip_speech_acts=(SpeechAct.REFINE_MOOD, SpeechAct.RESET_SOFT),
    )
    gen = await reply_generative(
        _state(),
        model=_FixedChat(canned=canned),
        settings=_settings(),
    )
    assert gen["route"] is Route.GENERATIVE
    assert gen["degraded_reason"] is DegradedReason.NONE
    assert gen["model_pick_indices"] == (0, 2)
    assert gen["model_pick_ids"] == ()
    assert gen["chip_speech_acts"] == (SpeechAct.REFINE_MOOD, SpeechAct.RESET_SOFT)
    assert gen["min_picks"] == 3


async def test_empty_candidates_skip_the_model() -> None:
    out = await reply_generative(
        _state(candidates=(), entitled_ids=()),
        model=_BoomChat(),
        settings=_settings(),
    )
    assert out["degraded_reason"] is DegradedReason.EMPTY_CATALOG_MATCH
    assert out["model_pick_indices"] == ()
    assert str(out["reply"]).strip()


async def test_unavailable_gateway_degrades_to_template() -> None:
    out = await reply_generative(
        _state(),
        settings=_settings(llm_provider=LLMProvider.NONE, anthropic_api_key=None),
    )
    assert out["degraded_reason"] is DegradedReason.GENERATIVE_SCHEMA_FAIL
    assert out["route"] is Route.DEGRADED_KEYWORD
    assert out["model_pick_indices"] == ()
    assert str(out["reply"]).strip()


async def test_timeout_degrades_with_generative_timeout() -> None:
    model = _TimeoutChat()
    out = await reply_generative(_state(), model=model, settings=_settings(llm_timeout_ms=50))
    assert model.call_log == [1]
    assert out["degraded_reason"] is DegradedReason.GENERATIVE_TIMEOUT
    assert out["model_pick_indices"] == ()


async def test_cost_callback_is_passed_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}
    handler = CostCallbackHandler()

    def _fake(schema: type[GroundedReply], **kwargs: Any) -> Runnable[Any, GroundedReply]:
        def _run(_input: Any, config: RunnableConfig) -> GroundedReply:
            seen["config"] = config
            return GroundedReply(reply="ok", pick_indices=(0,))

        return RunnableLambda(_run)

    monkeypatch.setattr("assist.nodes.reply.structured_output", _fake)
    await reply_generative(_state(), cost_handler=handler, settings=_settings())
    config = seen["config"]
    assert isinstance(config, dict)
    callbacks = config.get("callbacks")
    assert callbacks is not None
    raw_handlers = getattr(callbacks, "handlers", None)
    assert isinstance(raw_handlers, list)
    assert handler in raw_handlers


def test_cards_never_include_a_catalog_id_even_if_title_is_blank() -> None:
    blank = _c("ttl_secret_id", "", year=2001)
    blob = format_candidate_cards((blank,))
    assert "ttl_secret_id" not in blob
    assert "[0]" in blob
