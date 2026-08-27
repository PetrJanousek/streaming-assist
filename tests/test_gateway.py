"""LLM gateway: no-key stub, cost math, timeout, schema fallback."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import uuid4

import httpx
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

from assist.config import LLMProvider, Settings
from assist.llm.cost import (
    HAIKU_INPUT_USD_PER_MTOK,
    HAIKU_OUTPUT_USD_PER_MTOK,
    CostCallbackHandler,
    cost_usd,
)
from assist.llm.gateway import (
    LLMSchemaError,
    LLMTimeout,
    LLMUnavailable,
    get_chat_model,
    structured_output,
    with_timeout,
)
from assist.llm.prompts import chat_prompt_template, load_prompt


class _Item(BaseModel):
    n: int = Field(description="an integer")


def _settings(**kwargs: Any) -> Settings:
    values: dict[str, Any] = {
        "llm_provider": LLMProvider.ANTHROPIC,
        "anthropic_api_key": None,
        "llm_timeout_ms": 2500,
    }
    values.update(kwargs)
    return Settings(**values)


def test_import_gateway_does_not_require_a_key() -> None:
    import assist.llm.gateway as gw

    assert callable(gw.get_chat_model)


def test_provider_none_raises_unavailable_not_network() -> None:
    model = get_chat_model(
        settings=_settings(llm_provider=LLMProvider.NONE, anthropic_api_key="sk-x")
    )
    with pytest.raises(LLMUnavailable, match="LLM_PROVIDER=none"):
        try:
            model.invoke("hello")
        except httpx.HTTPError as exc:
            pytest.fail(f"network error leaked: {exc}")


@pytest.mark.asyncio
async def test_missing_key_raises_unavailable_and_never_constructs_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("ChatAnthropic must not be constructed without a key")

    monkeypatch.setattr("assist.llm.gateway.ChatAnthropic", _boom)
    model = get_chat_model(settings=_settings(anthropic_api_key=None))
    with pytest.raises(LLMUnavailable, match="ANTHROPIC_API_KEY"):
        try:
            await model.ainvoke("hello")
        except httpx.HTTPError as exc:
            pytest.fail(f"network error leaked: {exc}")


def test_missing_key_does_not_open_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("httpx must not be used when the key is missing")

    monkeypatch.setattr(httpx, "Client", _boom)
    monkeypatch.setattr(httpx, "AsyncClient", _boom)
    model = get_chat_model(settings=_settings(anthropic_api_key=""))
    with pytest.raises(LLMUnavailable):
        model.invoke([HumanMessage(content="hi")])


def test_factory_returns_chat_anthropic_when_keyed() -> None:
    model = get_chat_model(settings=_settings(anthropic_api_key="sk-test", llm_timeout_ms=2500))
    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-haiku-4-5"
    assert model.default_request_timeout == 2.5
    assert model.max_retries == 0


def test_cost_usd_matches_hand_computed_haiku_figure() -> None:
    # 1000 in + 200 out at $1 / $5 per MTok:
    # 1000/1e6 * 1 + 200/1e6 * 5 = 0.001 + 0.001 = 0.002
    expected = (1000 / 1_000_000) * HAIKU_INPUT_USD_PER_MTOK + (
        200 / 1_000_000
    ) * HAIKU_OUTPUT_USD_PER_MTOK
    assert expected == pytest.approx(0.002)
    assert cost_usd(1000, 200) == pytest.approx(expected)


def test_cost_callback_matches_hand_computed_from_usage_metadata() -> None:
    handler = CostCallbackHandler()
    message = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200},
    )
    result = LLMResult(generations=[[ChatGeneration(message=message)]])
    handler.on_llm_end(result, run_id=uuid4())
    assert handler.tokens_in == 1000
    assert handler.tokens_out == 200
    assert handler.cost_usd == pytest.approx(0.002)


def test_cost_callback_falls_back_to_llm_output_and_accumulates() -> None:
    handler = CostCallbackHandler()
    result = LLMResult(
        generations=[[]],
        llm_output={"usage": {"input_tokens": 300, "output_tokens": 150}},
    )
    handler.on_llm_end(result, run_id=uuid4())
    handler.on_llm_end(result, run_id=uuid4())
    assert handler.tokens_in == 600
    assert handler.tokens_out == 300
    # 600/1e6 * 1 + 300/1e6 * 5 = 0.0006 + 0.0015 = 0.0021
    assert handler.cost_usd == pytest.approx(0.0021)
    handler.reset()
    assert handler.tokens_in == 0
    assert handler.cost_usd == pytest.approx(0.0)


class _SleepyChat(BaseChatModel):
    delay_s: float = 2.0

    @property
    def _llm_type(self) -> str:
        return "sleepy"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        time.sleep(self.delay_s)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="late"))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        await asyncio.sleep(self.delay_s)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="late"))])


@pytest.mark.asyncio
async def test_timeout_enforced_on_sleeping_fake() -> None:
    started = time.perf_counter()
    wrapped = with_timeout(_SleepyChat(delay_s=2.0), timeout_s=0.05)
    with pytest.raises(LLMTimeout, match="50ms"):
        await wrapped.ainvoke([HumanMessage(content="hi")])
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0


class _ParserFailChat(BaseChatModel):
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
            raise OutputParserException("bad json")

        return RunnableLambda(_raise)


def test_structured_output_schema_failure_degrades_to_fallback() -> None:
    def _fallback(_input: Any) -> _Item:
        return _Item(n=7)

    fallback: Runnable[Any, _Item] = RunnableLambda(_fallback)
    chain = structured_output(
        _Item,
        model=_ParserFailChat(),
        fallback=fallback,
        retries=0,
        settings=_settings(anthropic_api_key="sk-test", llm_timeout_ms=2500),
    )
    assert chain.invoke("x") == _Item(n=7)


def test_structured_output_schema_failure_is_typed_without_fallback() -> None:
    chain = structured_output(
        _Item,
        model=_ParserFailChat(),
        retries=0,
        settings=_settings(anthropic_api_key="sk-test"),
    )
    with pytest.raises(LLMSchemaError):
        chain.invoke("x")


@pytest.mark.asyncio
async def test_structured_output_unavailable_never_hits_network() -> None:
    chain = structured_output(
        _Item,
        settings=_settings(llm_provider=LLMProvider.NONE),
    )
    with pytest.raises(LLMUnavailable):
        await chain.ainvoke("x")


def test_load_grounding_prompt() -> None:
    text = load_prompt("grounding")
    assert "catalog_id" in text
    template = chat_prompt_template("grounding")
    rendered = template.format()
    assert "numbered candidate list" in rendered


def test_unknown_prompt_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("does-not-exist")
