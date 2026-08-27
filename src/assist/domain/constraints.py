"""Constraint state, deltas, and the pure merge algebra from design.md."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from assist.domain.enums import (
    DeltaSource,
    GenreId,
    MaturityRating,
    MediaType,
    MoodId,
    RecencyBias,
    is_stricter_than,
)


class SetOp(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["set"] = "set"
    value: bool | int | str


class AddOp(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["add"] = "add"
    values: tuple[str, ...]


class RemoveOp(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["remove"] = "remove"
    values: tuple[str, ...]


class ClearOp(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["clear"] = "clear"


class ReplaceOp(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["replace"] = "replace"
    values: tuple[str, ...]


FieldOp = Annotated[
    SetOp | AddOp | RemoveOp | ClearOp | ReplaceOp,
    Field(discriminator="op"),
]


class ConstraintState(BaseModel):
    """Sticky session constraints. Hard AuthZ floors live on ServerUserCtx, not here."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    media_type: MediaType | None = None
    genres_include: tuple[GenreId, ...] = ()
    genres_exclude: tuple[GenreId, ...] = ()
    moods: tuple[MoodId, ...] = ()
    year_min: int | None = None
    year_max: int | None = None
    duration_max_min: int | None = None
    origins: tuple[str, ...] = ()
    local_originals_only: bool = False
    languages: tuple[str, ...] = ()
    people_include: tuple[str, ...] = ()
    people_exclude: tuple[str, ...] = ()
    maturity_request_stricter: MaturityRating | None = None
    recency_bias: RecencyBias | None = None

    @classmethod
    def empty(cls) -> ConstraintState:
        return cls()


class ConstraintDelta(BaseModel):
    """Per-field ops. Extra fields are forbidden so AuthZ keys cannot sneak in."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reset_soft: bool = False
    media_type: FieldOp | None = None
    genres_include: FieldOp | None = None
    genres_exclude: FieldOp | None = None
    moods: FieldOp | None = None
    year_min: FieldOp | None = None
    year_max: FieldOp | None = None
    duration_max_min: FieldOp | None = None
    origins: FieldOp | None = None
    local_originals_only: FieldOp | None = None
    languages: FieldOp | None = None
    people_include: FieldOp | None = None
    people_exclude: FieldOp | None = None
    maturity_request_stricter: FieldOp | None = None
    recency_bias: FieldOp | None = None


def merge(
    state: ConstraintState,
    delta: ConstraintDelta,
    source: DeltaSource,
    *,
    profile_maturity_max: MaturityRating | None = None,
) -> ConstraintState:
    """Apply `delta` onto `state`. Returns a new state; never mutates `state`.

    Precedence for one turn is the caller's job: apply text/rules first, then
    chip, so a chip tap wins a same-field conflict. `source` is the label for
    that call; field policy does not change by source.
    """
    # `source` is the caller's precedence label; field policy is source-agnostic.
    if source not in DeltaSource:
        raise TypeError(f"unknown delta source: {source!r}")
    # reset_soft wipes sticky preference fields only; AuthZ is not on this object.
    if delta.reset_soft:
        current = ConstraintState(media_type=MediaType.ANY)
    else:
        current = state

    media_type = _apply_enum(
        current.media_type, delta.media_type, MediaType, default_clear=MediaType.ANY
    )
    genres_include, genres_exclude = _apply_include_exclude(
        current.genres_include,
        current.genres_exclude,
        delta.genres_include,
        delta.genres_exclude,
        _as_genre,
    )
    moods = _apply_list(current.moods, delta.moods, _as_mood)

    year_min = _apply_int(current.year_min, delta.year_min)
    year_max = _apply_int(current.year_max, delta.year_max)
    # Inverted range is almost always an extractor error; keep a valid window.
    if year_min is not None and year_max is not None and year_min > year_max:
        year_min, year_max = year_max, year_min

    duration_max_min = _apply_int(current.duration_max_min, delta.duration_max_min)
    origins = _apply_list(current.origins, delta.origins, _as_nonempty_str)
    local_originals_only = _apply_bool(current.local_originals_only, delta.local_originals_only)
    languages = _apply_list(current.languages, delta.languages, _as_nonempty_str)

    people_include, people_exclude = _apply_include_exclude(
        current.people_include,
        current.people_exclude,
        delta.people_include,
        delta.people_exclude,
        _as_nonempty_str,
    )

    maturity_request_stricter = _apply_maturity(
        current.maturity_request_stricter,
        delta.maturity_request_stricter,
        profile_maturity_max,
    )
    recency_bias = _apply_enum(
        current.recency_bias, delta.recency_bias, RecencyBias, default_clear=None
    )

    return ConstraintState(
        media_type=media_type,
        genres_include=genres_include,
        genres_exclude=genres_exclude,
        moods=moods,
        year_min=year_min,
        year_max=year_max,
        duration_max_min=duration_max_min,
        origins=origins,
        local_originals_only=local_originals_only,
        languages=languages,
        people_include=people_include,
        people_exclude=people_exclude,
        maturity_request_stricter=maturity_request_stricter,
        recency_bias=recency_bias,
    )


def _as_genre(value: object) -> GenreId | None:
    return _as_enum(value, GenreId)


def _as_mood(value: object) -> MoodId | None:
    return _as_enum(value, MoodId)


def _as_enum[E: StrEnum](value: object, enum_cls: type[E]) -> E | None:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError:
            return None
    return None


def _as_nonempty_str(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _apply_list[T](
    current: tuple[T, ...],
    op: FieldOp | None,
    parse: Callable[[object], T | None],
) -> tuple[T, ...]:
    if op is None:
        return current
    if isinstance(op, ClearOp):
        return ()
    if isinstance(op, AddOp):
        return _union(current, _parse_values(op.values, parse))
    if isinstance(op, RemoveOp):
        drop = set(_parse_values(op.values, parse))
        return tuple(item for item in current if item not in drop)
    if isinstance(op, ReplaceOp):
        return _parse_values(op.values, parse)
    # set is a scalar op; ignore it on list fields rather than guess a replace.
    return current


def _parse_values[T](
    values: Sequence[object], parse: Callable[[object], T | None]
) -> tuple[T, ...]:
    out: list[T] = []
    seen: set[T] = set()
    for raw in values:
        item = parse(raw)
        if item is None or item in seen:
            continue
        out.append(item)
        seen.add(item)
    return tuple(out)


def _union[T](current: tuple[T, ...], incoming: tuple[T, ...]) -> tuple[T, ...]:
    seen = set(current)
    out = list(current)
    for item in incoming:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return tuple(out)


def _without[T](seq: tuple[T, ...], banned: set[T]) -> tuple[T, ...]:
    if not banned:
        return seq
    return tuple(item for item in seq if item not in banned)


def _apply_include_exclude[T](
    include: tuple[T, ...],
    exclude: tuple[T, ...],
    include_op: FieldOp | None,
    exclude_op: FieldOp | None,
    parse: Callable[[object], T | None],
) -> tuple[tuple[T, ...], tuple[T, ...]]:
    """Keep the two lists disjoint. Same-turn exclude wins the overlap."""
    if include_op is not None:
        include = _apply_list(include, include_op, parse)
        exclude = _without(exclude, set(include))
    if exclude_op is not None:
        exclude = _apply_list(exclude, exclude_op, parse)
        include = _without(include, set(exclude))
    return include, exclude


def _apply_enum[E: StrEnum](
    current: E | None,
    op: FieldOp | None,
    enum_cls: type[E],
    *,
    default_clear: E | None,
) -> E | None:
    if op is None:
        return current
    if isinstance(op, ClearOp):
        return default_clear
    if isinstance(op, SetOp):
        parsed = _as_enum(op.value, enum_cls)
        return current if parsed is None else parsed
    return current


def _apply_int(current: int | None, op: FieldOp | None) -> int | None:
    if op is None:
        return current
    if isinstance(op, ClearOp):
        return None
    if isinstance(op, SetOp):
        parsed = _as_int(op.value)
        return current if parsed is None else parsed
    return current


def _apply_bool(current: bool, op: FieldOp | None) -> bool:
    if op is None:
        return current
    if isinstance(op, ClearOp):
        return False
    if isinstance(op, SetOp):
        parsed = _as_bool(op.value)
        return current if parsed is None else parsed
    return current


def _apply_maturity(
    current: MaturityRating | None,
    op: FieldOp | None,
    profile_maturity_max: MaturityRating | None,
) -> MaturityRating | None:
    if op is None:
        return current
    if isinstance(op, ClearOp):
        return None
    if not isinstance(op, SetOp):
        return current
    requested = _as_enum(op.value, MaturityRating)
    if requested is None:
        return current
    # Soft-down only: a delta cannot raise (or even match) the profile ceiling.
    if profile_maturity_max is not None and not is_stricter_than(requested, profile_maturity_max):
        return current
    return requested
