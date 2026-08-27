"""LLM gateway: provider factory, structured output, cost accounting, prompts."""

from assist.llm.cost import (
    HAIKU_INPUT_USD_PER_MTOK,
    HAIKU_OUTPUT_USD_PER_MTOK,
    CostCallbackHandler,
    cost_usd,
)
from assist.llm.gateway import (
    LLMError,
    LLMSchemaError,
    LLMTimeout,
    LLMUnavailable,
    get_chat_model,
    structured_output,
    with_timeout,
)

__all__ = [
    "HAIKU_INPUT_USD_PER_MTOK",
    "HAIKU_OUTPUT_USD_PER_MTOK",
    "CostCallbackHandler",
    "LLMError",
    "LLMSchemaError",
    "LLMTimeout",
    "LLMUnavailable",
    "cost_usd",
    "get_chat_model",
    "structured_output",
    "with_timeout",
]
