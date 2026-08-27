"""Per-turn token and USD accounting. Haiku 4.5 rates are from implementation-plan §6."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# Published Haiku 4.5 list price: $1 input / $5 output per million tokens.
HAIKU_INPUT_USD_PER_MTOK = 1.0
HAIKU_OUTPUT_USD_PER_MTOK = 5.0
_TOKENS_PER_MTOK = 1_000_000


def cost_usd(
    tokens_in: int,
    tokens_out: int,
    *,
    input_usd_per_mtok: float = HAIKU_INPUT_USD_PER_MTOK,
    output_usd_per_mtok: float = HAIKU_OUTPUT_USD_PER_MTOK,
) -> float:
    """USD for a known token count. Callers pass rates when the model is not Haiku."""
    return (tokens_in / _TOKENS_PER_MTOK) * input_usd_per_mtok + (
        tokens_out / _TOKENS_PER_MTOK
    ) * output_usd_per_mtok


def _first_int(data: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        return int(value)
    return 0


def _usage_dict(llm_output: dict[str, Any] | None) -> dict[str, Any]:
    if not llm_output:
        return {}
    usage = llm_output.get("usage") or llm_output.get("token_usage") or {}
    if isinstance(usage, dict):
        return usage
    dumped = getattr(usage, "model_dump", None)
    if callable(dumped):
        result = dumped()
        if isinstance(result, dict):
            return result
    return {}


def tokens_from_result(response: LLMResult) -> tuple[int, int]:
    """Best-effort (input, output) tokens from a LangChain `LLMResult`.

    Prefer `AIMessage.usage_metadata` (one generation) so cached-token extras that
    Anthropic reports on the message are counted once. Fall back to `llm_output`.
    """
    for gen_list in response.generations:
        for gen in gen_list:
            message = getattr(gen, "message", None)
            usage = getattr(message, "usage_metadata", None) if message is not None else None
            if usage:
                return (
                    int(usage.get("input_tokens", 0) or 0),
                    int(usage.get("output_tokens", 0) or 0),
                )
    usage_dict = _usage_dict(response.llm_output)
    return (
        _first_int(usage_dict, "input_tokens", "prompt_tokens"),
        _first_int(usage_dict, "output_tokens", "completion_tokens"),
    )


class CostCallbackHandler(BaseCallbackHandler):
    """Accumulates tokens and USD across the LLM calls of one turn."""

    def __init__(
        self,
        *,
        input_usd_per_mtok: float = HAIKU_INPUT_USD_PER_MTOK,
        output_usd_per_mtok: float = HAIKU_OUTPUT_USD_PER_MTOK,
    ) -> None:
        super().__init__()
        self.tokens_in = 0
        self.tokens_out = 0
        self.input_usd_per_mtok = input_usd_per_mtok
        self.output_usd_per_mtok = output_usd_per_mtok

    @property
    def cost_usd(self) -> float:
        return cost_usd(
            self.tokens_in,
            self.tokens_out,
            input_usd_per_mtok=self.input_usd_per_mtok,
            output_usd_per_mtok=self.output_usd_per_mtok,
        )

    def reset(self) -> None:
        self.tokens_in = 0
        self.tokens_out = 0

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        tokens_in, tokens_out = tokens_from_result(response)
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
