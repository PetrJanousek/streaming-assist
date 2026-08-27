"""Catalog-facing domain types. Pure data; no store I/O."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from assist.domain.enums import CreditRole, GenreId, MediaType, MoodId


class Title(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    catalog_id: str
    media_type: MediaType
    title: str
    synopsis: str = ""
    release_year: int | None = None
    runtime_min: int | None = None
    seasons: int | None = None
    maturity_rank: int
    origins: tuple[str, ...] = ()
    genres: tuple[GenreId, ...] = ()
    moods: tuple[MoodId, ...] = ()
    local_original: bool = False
    pop_28d: float = 0.0


class Person(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    person_id: str
    name: str
    name_norm: str
    roles: tuple[CreditRole, ...] = ()
    credit_count: int = 0
    active_year_min: int | None = None
    active_year_max: int | None = None
    popularity: float = 0.0


class Candidate(BaseModel):
    """Compact ranked card. Sequence order is rank order for sanitize_picks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    catalog_id: str
    title: str
    media_type: MediaType
    release_year: int | None = None
    genres: tuple[GenreId, ...] = ()
    score: float = 0.0


class Pick(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    catalog_id: str
    reason_short: str = Field(default="")
