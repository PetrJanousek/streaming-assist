"""Application settings. Every knob is an env var; bad values fail at import."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_DATASET_URL = (
    "https://huggingface.co/datasets/hugginglearners/netflix-shows/resolve/main/netflix_titles.csv"
)
_DEFAULT_DATASET_SHA256 = "b5e2528aaac6a9a1544f68fb1ad004f46fabbf8bc2644d96f7364ca8b66ce959"


class LLMProvider(StrEnum):
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    NONE = "none"


class Settings(BaseSettings):
    """Env-backed settings. Unknown *values* raise; they never fall back."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        env_ignore_empty=True,
    )

    # LLM — empty ANTHROPIC_API_KEY is the documented degraded mode (T08).
    anthropic_api_key: str | None = None
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    anthropic_model: str = "claude-haiku-4-5"
    llm_timeout_ms: int = Field(default=2500, gt=0)

    # Catalog / retrieval
    home_country: str = "United States"
    dataset_url: str = _DEFAULT_DATASET_URL
    dataset_sha256: str = _DEFAULT_DATASET_SHA256
    dataset_fetch_timeout_s: int = Field(default=60, gt=0)
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rrf_k: int = Field(default=60, ge=1)
    router_theta1: float = Field(default=0.55, ge=0.0, le=1.0)
    router_theta_gap: float = Field(default=0.08, ge=0.0, le=1.0)
    rank_w_pop: float = Field(default=0.50, ge=0.0, le=1.0)
    rank_w_constraint: float = Field(default=0.30, ge=0.0, le=1.0)
    rank_w_semantic: float = Field(default=0.20, ge=0.0, le=1.0)

    # Session / limits
    session_ttl_s: int = Field(default=86400, gt=0)
    rate_limit_rps: int = Field(default=5, gt=0)
    rate_limit_burst: int = Field(default=20, gt=0)
    hard_timeout_ms: int = Field(default=8000, gt=0)
    retrieve_max_attempts: int = Field(default=2, ge=1)
    # Design threat model: max input 500 chars. Longer is a cost/injection vector.
    guard_max_chars: int = Field(default=500, ge=1)
    # Post-sanitize reply cap (design.md reliability layer 5: length caps + title-span).
    reply_max_chars: int = Field(default=600, ge=1)

    # Observability
    langsmith_tracing: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    debug_meta: bool = False

    # Stores — host-side defaults; compose will override inside the network (T03).
    postgres_dsn: str = "postgresql+asyncpg://assist:assist@localhost:5432/assist"
    postgres_pool_size: int = Field(default=5, ge=1)
    postgres_max_overflow: int = Field(default=10, ge=0)
    redis_url: str = "redis://localhost:6379/0"
    elasticsearch_url: str = "http://localhost:9200"
    embedder_url: str = "http://localhost:8080"
    embedder_timeout_ms: int = Field(default=5000, gt=0)
    embedder_retries: int = Field(default=2, ge=0)

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @model_validator(mode="after")
    def _rank_weights_sum_to_one(self) -> Self:
        total = self.rank_w_pop + self.rank_w_constraint + self.rank_w_semantic
        if abs(total - 1.0) > 1e-6:
            msg = f"RANK_W_POP + RANK_W_CONSTRAINT + RANK_W_SEMANTIC must sum to 1.0, got {total}"
            raise ValueError(msg)
        return self


settings = Settings()
