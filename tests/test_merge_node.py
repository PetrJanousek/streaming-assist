"""Merge node: precedence, maturity clamp, sticky multi-turn, AuthZ isolation."""

from __future__ import annotations

from typing import Any, cast

import pytest

from assist.domain.constraints import (
    AddOp,
    ClearOp,
    ConstraintDelta,
    ConstraintState,
    RemoveOp,
    ReplaceOp,
    SetOp,
    merge,
)
from assist.domain.context import ServerUserCtx
from assist.domain.enums import (
    DeltaSource,
    DeviceClass,
    GenreId,
    MaturityRating,
    MediaType,
    MoodId,
    Package,
)
from assist.graph.state import TurnState, empty_turn_state
from assist.nodes.merge import (
    CHIP_DELTA_KEY,
    apply_turn_deltas,
    effective_maturity,
    merge_constraints,
)

PROFILE_PG13 = MaturityRating.PG_13
PROFILE_R = MaturityRating.R


def _ctx(
    *,
    maturity_max: MaturityRating = PROFILE_PG13,
    kids_flag: bool = False,
    geo: str = "US",
    package: Package = Package.BASIC,
    device_class: DeviceClass = DeviceClass.WEB,
) -> ServerUserCtx:
    return ServerUserCtx(
        user_id="u1",
        profile_id="p1",
        geo=geo,
        package=package,
        maturity_max=maturity_max,
        kids_flag=kids_flag,
        device_class=device_class,
    )


def _state(
    ctx: ServerUserCtx,
    *,
    constraints: ConstraintState | None = None,
    delta: ConstraintDelta | None = None,
    intent_source: str | None = None,
    chip_delta: ConstraintDelta | None = None,
    **overrides: object,
) -> TurnState:
    extra: dict[str, object] = {}
    if constraints is not None:
        extra["constraints"] = constraints
    if delta is not None:
        extra["delta"] = delta
    if intent_source is not None:
        extra["intent_source"] = intent_source
    if chip_delta is not None:
        extra[CHIP_DELTA_KEY] = chip_delta
    extra.update(overrides)
    return empty_turn_state(ctx, **extra)


async def _run(state: TurnState) -> ConstraintState:
    out = await merge_constraints(state)
    merged = out["constraints"]
    assert isinstance(merged, ConstraintState)
    return merged


# --- T02 carry-forward: 3-arg merge does not clamp; the node must. ---


def test_three_arg_merge_does_not_clamp_maturity() -> None:
    """Proof that skipping profile_maturity_max lets a raise through."""
    leaked = merge(
        ConstraintState.empty(),
        ConstraintDelta(maturity_request_stricter=SetOp(value="NC-17")),
        DeltaSource.TEXT,
    )
    assert leaked.maturity_request_stricter is MaturityRating.NC_17


async def test_raise_maturity_above_profile_is_noop_through_node() -> None:
    ctx = _ctx(maturity_max=PROFILE_PG13)
    prior = ConstraintState.empty()
    merged = await _run(
        _state(
            ctx,
            constraints=prior,
            delta=ConstraintDelta(maturity_request_stricter=SetOp(value="NC-17")),
            intent_source="llm",
        )
    )
    assert merged.maturity_request_stricter is None
    assert merged == prior
    assert effective_maturity(ctx, merged) is PROFILE_PG13
    assert ctx.maturity_max is PROFILE_PG13


async def test_equal_maturity_to_profile_is_noop_through_node() -> None:
    ctx = _ctx(maturity_max=PROFILE_PG13)
    merged = await _run(
        _state(
            ctx,
            delta=ConstraintDelta(maturity_request_stricter=SetOp(value="PG-13")),
            intent_source="rules",
        )
    )
    assert merged.maturity_request_stricter is None
    assert effective_maturity(ctx, merged) is PROFILE_PG13


async def test_stricter_maturity_request_is_kept() -> None:
    ctx = _ctx(maturity_max=PROFILE_R)
    merged = await _run(
        _state(
            ctx,
            delta=ConstraintDelta(maturity_request_stricter=SetOp(value="PG")),
            intent_source="llm",
        )
    )
    assert merged.maturity_request_stricter is MaturityRating.PG
    assert effective_maturity(ctx, merged) is MaturityRating.PG


async def test_every_merge_call_passes_profile_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    real = merge

    def wrapped(
        state: ConstraintState,
        delta: ConstraintDelta,
        source: DeltaSource,
        *,
        profile_maturity_max: MaturityRating | None = None,
    ) -> ConstraintState:
        calls.append(
            {
                "source": source,
                "profile_maturity_max": profile_maturity_max,
            }
        )
        return real(state, delta, source, profile_maturity_max=profile_maturity_max)

    monkeypatch.setattr("assist.nodes.merge.merge_delta", wrapped)
    ctx = _ctx(maturity_max=PROFILE_PG13)
    text = ConstraintDelta(moods=AddOp(values=("funny",)))
    chip = ConstraintDelta(moods=ReplaceOp(values=("cozy",)))
    await _run(_state(ctx, delta=text, intent_source="llm", chip_delta=chip))
    assert len(calls) == 2
    assert [c["source"] for c in calls] == [DeltaSource.TEXT, DeltaSource.CHIP]
    assert all(c["profile_maturity_max"] is PROFILE_PG13 for c in calls)


# --- Precedence: text/rules first, chip second. ---


async def test_chip_delta_beats_conflicting_text_delta_same_turn() -> None:
    ctx = _ctx()
    text = ConstraintDelta(moods=AddOp(values=("funny",)), media_type=SetOp(value="film"))
    chip = ConstraintDelta(moods=ReplaceOp(values=("cozy",)), media_type=SetOp(value="series"))
    merged = await _run(_state(ctx, delta=text, intent_source="llm", chip_delta=chip))
    assert merged.moods == (MoodId.COZY,)
    assert merged.media_type is MediaType.SERIES


async def test_chip_after_rules_wins_same_field() -> None:
    ctx = _ctx()
    rules = ConstraintDelta(genres_include=AddOp(values=("drama",)))
    chip = ConstraintDelta(genres_include=ReplaceOp(values=("comedy",)))
    merged = await _run(_state(ctx, delta=rules, intent_source="rules", chip_delta=chip))
    assert merged.genres_include == (GenreId.COMEDY,)


def test_apply_order_is_text_then_chip() -> None:
    """Reversing the steps would leave the text value; order is the contract."""
    prior = ConstraintState.empty()
    text = ConstraintDelta(media_type=SetOp(value="film"))
    chip = ConstraintDelta(media_type=SetOp(value="series"))
    right = apply_turn_deltas(
        prior,
        PROFILE_PG13,
        ((text, DeltaSource.TEXT), (chip, DeltaSource.CHIP)),
    )
    wrong = apply_turn_deltas(
        prior,
        PROFILE_PG13,
        ((chip, DeltaSource.CHIP), (text, DeltaSource.TEXT)),
    )
    assert right.media_type is MediaType.SERIES
    assert wrong.media_type is MediaType.FILM


async def test_chip_only_turn_uses_chip_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources: list[DeltaSource] = []
    real = merge

    def wrapped(
        state: ConstraintState,
        delta: ConstraintDelta,
        source: DeltaSource,
        *,
        profile_maturity_max: MaturityRating | None = None,
    ) -> ConstraintState:
        sources.append(source)
        return real(state, delta, source, profile_maturity_max=profile_maturity_max)

    monkeypatch.setattr("assist.nodes.merge.merge_delta", wrapped)
    merged = await _run(
        _state(
            _ctx(),
            delta=ConstraintDelta(moods=AddOp(values=("cozy",))),
            intent_source="chip",
        )
    )
    assert sources == [DeltaSource.CHIP]
    assert merged.moods == (MoodId.COZY,)


async def test_unknown_intent_source_is_text_not_chip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources: list[DeltaSource] = []
    real = merge

    def wrapped(
        state: ConstraintState,
        delta: ConstraintDelta,
        source: DeltaSource,
        *,
        profile_maturity_max: MaturityRating | None = None,
    ) -> ConstraintState:
        sources.append(source)
        return real(state, delta, source, profile_maturity_max=profile_maturity_max)

    monkeypatch.setattr("assist.nodes.merge.merge_delta", wrapped)
    await _run(
        _state(
            _ctx(),
            delta=ConstraintDelta(moods=AddOp(values=("funny",))),
            intent_source=None,
        )
    )
    assert sources == [DeltaSource.TEXT]


# --- Multi-turn stickiness. ---


async def test_turn1_constraints_survive_turn3() -> None:
    ctx = _ctx()
    turn1 = await _run(
        _state(
            ctx,
            delta=ConstraintDelta(
                local_originals_only=SetOp(value=True),
                origins=AddOp(values=("CZ",)),
                year_min=SetOp(value=1990),
            ),
            intent_source="llm",
        )
    )
    turn2 = await _run(
        _state(
            ctx,
            constraints=turn1,
            delta=ConstraintDelta(moods=AddOp(values=("funny",))),
            intent_source="rules",
        )
    )
    turn3 = await _run(
        _state(
            ctx,
            constraints=turn2,
            delta=ConstraintDelta(people_include=AddOp(values=("p1",))),
            intent_source="chip",
        )
    )
    assert turn3.local_originals_only is True
    assert turn3.origins == ("CZ",)
    assert turn3.year_min == 1990
    assert turn3.moods == (MoodId.FUNNY,)
    assert turn3.people_include == ("p1",)


async def test_chip_in_later_turn_overrides_conflicting_field_only() -> None:
    ctx = _ctx()
    turn1 = await _run(
        _state(
            ctx,
            delta=ConstraintDelta(
                local_originals_only=SetOp(value=True),
                origins=AddOp(values=("CZ",)),
                moods=AddOp(values=("funny",)),
            ),
            intent_source="llm",
        )
    )
    turn2 = await _run(
        _state(
            ctx,
            constraints=turn1,
            delta=ConstraintDelta(
                local_originals_only=SetOp(value=False),
                origins=ClearOp(),
            ),
            intent_source="chip",
        )
    )
    assert turn2.local_originals_only is False
    assert turn2.origins == ()
    assert turn2.moods == (MoodId.FUNNY,)


# --- Media-type sentinels. ---


@pytest.mark.parametrize("media_type", [None, MediaType.ANY])
async def test_both_no_media_filter_sentinels_survive_unrelated_delta(
    media_type: MediaType | None,
) -> None:
    prior = ConstraintState(media_type=media_type)
    merged = await _run(
        _state(
            _ctx(),
            constraints=prior,
            delta=ConstraintDelta(moods=AddOp(values=("cozy",))),
            intent_source="llm",
        )
    )
    assert merged.media_type is media_type
    assert merged.moods == (MoodId.COZY,)


async def test_reset_soft_from_empty_sentinel_becomes_any() -> None:
    merged = await _run(
        _state(
            _ctx(),
            constraints=ConstraintState.empty(),
            delta=ConstraintDelta(reset_soft=True),
            intent_source="chip",
        )
    )
    assert ConstraintState.empty().media_type is None
    assert merged.media_type is MediaType.ANY
    assert merged.moods == ()


async def test_reset_soft_from_any_sentinel_stays_any() -> None:
    prior = ConstraintState(media_type=MediaType.ANY, moods=(MoodId.FUNNY,))
    merged = await _run(
        _state(
            _ctx(),
            constraints=prior,
            delta=ConstraintDelta(reset_soft=True),
            intent_source="chip",
        )
    )
    assert merged.media_type is MediaType.ANY
    assert merged.moods == ()


async def test_missing_constraints_uses_empty_sentinel() -> None:
    state = empty_turn_state(_ctx(), delta=ConstraintDelta(moods=AddOp(values=("dark",))))
    assert state["constraints"] == ConstraintState.empty()
    assert state["constraints"].media_type is None
    merged = await _run(state)
    assert merged.moods == (MoodId.DARK,)
    assert merged.media_type is None


async def test_none_delta_is_identity() -> None:
    prior = ConstraintState(moods=(MoodId.FUNNY,), media_type=MediaType.FILM)
    merged = await _run(_state(_ctx(), constraints=prior, delta=None))
    assert merged == prior


# --- Hard AuthZ isolation. ---


AUTHZ_KEYS = frozenset(
    {
        "geo",
        "package",
        "maturity_max",
        "kids_flag",
        "device_class",
        "user_id",
        "profile_id",
    }
)

DELTA_CORPUS: tuple[ConstraintDelta, ...] = (
    ConstraintDelta(),
    ConstraintDelta(reset_soft=True),
    ConstraintDelta(media_type=SetOp(value="film")),
    ConstraintDelta(media_type=ClearOp()),
    ConstraintDelta(genres_include=AddOp(values=("drama",))),
    ConstraintDelta(genres_include=RemoveOp(values=("drama",))),
    ConstraintDelta(genres_include=ReplaceOp(values=("comedy",))),
    ConstraintDelta(genres_include=ClearOp()),
    ConstraintDelta(genres_exclude=AddOp(values=("horror",))),
    ConstraintDelta(moods=AddOp(values=("funny",))),
    ConstraintDelta(moods=RemoveOp(values=("funny",))),
    ConstraintDelta(moods=ReplaceOp(values=("cozy",))),
    ConstraintDelta(moods=ClearOp()),
    ConstraintDelta(year_min=SetOp(value=1990), year_max=SetOp(value=1999)),
    ConstraintDelta(year_min=ClearOp(), year_max=ClearOp()),
    ConstraintDelta(duration_max_min=SetOp(value=90)),
    ConstraintDelta(origins=AddOp(values=("CZ",))),
    ConstraintDelta(origins=ClearOp()),
    ConstraintDelta(local_originals_only=SetOp(value=True)),
    ConstraintDelta(local_originals_only=ClearOp()),
    ConstraintDelta(languages=AddOp(values=("en",))),
    ConstraintDelta(people_include=AddOp(values=("p1",))),
    ConstraintDelta(people_exclude=AddOp(values=("p2",))),
    ConstraintDelta(maturity_request_stricter=SetOp(value="PG")),
    ConstraintDelta(maturity_request_stricter=SetOp(value="NC-17")),
    ConstraintDelta(maturity_request_stricter=ClearOp()),
    ConstraintDelta(recency_bias=SetOp(value="tonight")),
    ConstraintDelta(recency_bias=ClearOp()),
    ConstraintDelta(reset_soft=True, moods=AddOp(values=("cozy",))),
)


@pytest.mark.parametrize("delta", DELTA_CORPUS, ids=lambda d: d.model_dump_json())
async def test_hard_authz_fields_unchanged_by_every_delta_shape(
    delta: ConstraintDelta,
) -> None:
    ctx = _ctx(
        maturity_max=PROFILE_PG13,
        kids_flag=True,
        geo="CZ",
        package=Package.PREMIUM,
        device_class=DeviceClass.TV,
    )
    snapshot = ctx.model_dump()
    prior = ConstraintState(
        media_type=MediaType.FILM,
        genres_include=(GenreId.DRAMA,),
        maturity_request_stricter=MaturityRating.PG,
    )
    state = _state(ctx, constraints=prior, delta=delta, intent_source="llm")
    out = await merge_constraints(state)
    assert "ctx" not in out
    assert state["ctx"] is ctx
    assert ctx.model_dump() == snapshot
    merged = out["constraints"]
    assert isinstance(merged, ConstraintState)
    assert AUTHZ_KEYS.isdisjoint(merged.model_dump())
    assert AUTHZ_KEYS.isdisjoint(delta.model_dump())
    # NC-17 is in the corpus; a raise must not survive the node clamp.
    assert merged.maturity_request_stricter in {None, MaturityRating.PG}
    assert effective_maturity(ctx, merged) in {PROFILE_PG13, MaturityRating.PG}


async def test_node_does_not_return_ctx() -> None:
    ctx = _ctx()
    out = await merge_constraints(
        _state(ctx, delta=ConstraintDelta(moods=AddOp(values=("funny",))))
    )
    assert set(out) == {"constraints"}


async def test_missing_ctx_raises() -> None:
    state = cast(TurnState, {"constraints": ConstraintState.empty(), "delta": None})
    with pytest.raises(TypeError, match="ServerUserCtx"):
        await merge_constraints(state)


def test_effective_maturity_ignores_non_stricter_request() -> None:
    ctx = _ctx(maturity_max=PROFILE_PG13)
    leaked = ConstraintState(maturity_request_stricter=MaturityRating.NC_17)
    assert effective_maturity(ctx, leaked) is PROFILE_PG13
