"""Provider factory, structured-output helper, retry/fallback, and hard timeout.

The model never gets tools for control flow. Structured output may use the
provider's schema channel; that is decoding, not an agent loop.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import (
    ModelConnectionError,
    ModelRateLimitError,
    OutputParserException,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import BaseModel, SecretStr, ValidationError

from assist.config import LLMProvider, Settings
from assist.config import settings as default_settings
from assist.obs.logging import get_logger

log = get_logger(__name__)

_RETRYABLE = (ModelRateLimitError, ModelConnectionError)
_SCHEMA_ERRORS = (OutputParserException, ValidationError)


class LLMError(Exception):
    """Gateway failure. Callers degrade the turn; they do not 500."""


class LLMUnavailable(LLMError):
    """Provider is off, the key is missing, or this provider is not wired."""


class LLMTimeout(LLMError):
    """The call exceeded `LLM_TIMEOUT_MS`."""


class LLMSchemaError(LLMError):
    """Structured output did not match the requested schema."""


class UnavailableChatModel(BaseChatModel):
    """Stub so the process boots with no API key. Invoke raises `LLMUnavailable`."""

    reason: str = "LLM unavailable"

    @property
    def _llm_type(self) -> str:
        return "unavailable"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        log.info("llm_unavailable", reason=self.reason)
        raise LLMUnavailable(self.reason)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        log.info("llm_unavailable", reason=self.reason)
        raise LLMUnavailable(self.reason)

    def with_structured_output(
        self,
        schema: dict[str, Any] | type[Any],
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        # Stay a stub: schema binding must not open a network client.
        return self


def _cfg(settings: Settings | None) -> Settings:
    return settings if settings is not None else default_settings


def _timeout_s(cfg: Settings) -> float:
    return cfg.llm_timeout_ms / 1000.0


def get_chat_model(*, settings: Settings | None = None) -> BaseChatModel:
    """Return a configured `ChatAnthropic`, or a stub that raises `LLMUnavailable`.

    Importing this module never talks to the network. A missing key or
    `LLM_PROVIDER=none` yields a stub; invoking it raises `LLMUnavailable`.
    """
    cfg = _cfg(settings)
    if cfg.llm_provider is LLMProvider.NONE:
        return UnavailableChatModel(reason="LLM_PROVIDER=none")
    if cfg.llm_provider is LLMProvider.OLLAMA:
        return UnavailableChatModel(reason="LLM_PROVIDER=ollama is not wired")
    key = (cfg.anthropic_api_key or "").strip()
    if not key:
        return UnavailableChatModel(reason="ANTHROPIC_API_KEY is missing")
    # HTTP-client retries would stack on LLM_TIMEOUT_MS; LangChain with_retry
    # owns the few transient retries we do want, under the hard timeout.
    return ChatAnthropic(
        model=cfg.anthropic_model,
        anthropic_api_key=SecretStr(key),
        default_request_timeout=_timeout_s(cfg),
        max_retries=0,
        temperature=0,
    )


def with_timeout(runnable: Runnable[Any, Any], timeout_s: float) -> Runnable[Any, Any]:
    """Hard-cap a runnable at `timeout_s` seconds. Raises `LLMTimeout`."""

    def _sync(value: Any, config: RunnableConfig) -> Any:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(runnable.invoke, value, config)
            try:
                return future.result(timeout=timeout_s)
            except TimeoutError as exc:
                log.warning("llm_timeout", timeout_s=timeout_s)
                raise LLMTimeout(f"LLM call exceeded {int(timeout_s * 1000)}ms") from exc

    async def _async(value: Any, config: RunnableConfig) -> Any:
        try:
            async with asyncio.timeout(timeout_s):
                return await runnable.ainvoke(value, config=config)
        except TimeoutError as exc:
            log.warning("llm_timeout", timeout_s=timeout_s)
            raise LLMTimeout(f"LLM call exceeded {int(timeout_s * 1000)}ms") from exc

    return RunnableLambda(_sync, afunc=_async)


def _raise_schema_error(_: Any) -> Any:
    raise LLMSchemaError("structured output failed schema validation")


def structured_output[TSchema: BaseModel](
    schema: type[TSchema],
    *,
    retries: int = 1,
    fallback: Runnable[Any, TSchema] | None = None,
    model: BaseChatModel | None = None,
    settings: Settings | None = None,
) -> Runnable[Any, TSchema]:
    """Structured call: schema decode, optional retry, fallback, hard timeout.

    Schema failure runs `fallback` if given; otherwise it raises `LLMSchemaError`
    rather than a raw parser exception. Pass `retries=0` on the generative reply
    path (zero retries by design).
    """
    cfg = _cfg(settings)
    timeout_s = _timeout_s(cfg)
    chat = model if model is not None else get_chat_model(settings=cfg)
    if isinstance(chat, UnavailableChatModel):
        return cast("Runnable[Any, TSchema]", with_timeout(chat, timeout_s))

    chain: Runnable[Any, Any] = chat.with_structured_output(schema, method="json_schema")
    if retries > 0:
        chain = chain.with_retry(
            retry_if_exception_type=_RETRYABLE,
            stop_after_attempt=retries + 1,
            wait_exponential_jitter=True,
        )
    degrade: Runnable[Any, Any] = (
        fallback if fallback is not None else RunnableLambda(_raise_schema_error)
    )
    chain = chain.with_fallbacks(
        [degrade],
        exceptions_to_handle=_SCHEMA_ERRORS,
    )
    return cast("Runnable[Any, TSchema]", with_timeout(chain, timeout_s))
