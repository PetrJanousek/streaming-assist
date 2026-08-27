"""Golden cases for the constraint merge algebra (design.md, ≥20 cases)."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

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
    RecencyBias,
)

PROFILE_R = MaturityRating.R
PROFILE_PG13 = MaturityRating.PG_13


def _merge(
    state: ConstraintState,
    delta: ConstraintDelta,
    source: DeltaSource = DeltaSource.TEXT,
    *,
    profile: MaturityRating | None = PROFILE_R,
) -> ConstraintState:
    return merge(state, delta, source, profile_maturity_max=profile)


def test_worked_example_five_steps() -> None:
    """design.md R6 oracle: empty → local originals → funny → chip foreign → reset."""
    state = ConstraintState.empty()
    assert state == ConstraintState()

    state = _merge(
        state,
        ConstraintDelta(
            local_originals_only=SetOp(value=True),
            origins=AddOp(values=("CZ",)),
        ),
    )
    assert state.local_originals_only is True
    assert state.origins == ("CZ",)

    state = _merge(state, ConstraintDelta(moods=AddOp(values=("funny",))))
    assert state.moods == (MoodId.FUNNY,)
    assert state.local_originals_only is True
    assert state.origins == ("CZ",)

    state = _merge(
        state,
        ConstraintDelta(
            local_originals_only=SetOp(value=False),
            origins=ClearOp(),
        ),
        DeltaSource.CHIP,
    )
    assert state.local_originals_only is False
    assert state.origins == ()
    assert state.moods == (MoodId.FUNNY,)

    state = _merge(state, ConstraintDelta(reset_soft=True), DeltaSource.CHIP)
    assert state.moods == ()
    assert state.genres_include == ()
    assert state.genres_exclude == ()
    assert state.people_include == ()
    assert state.people_exclude == ()
    assert state.year_min is None
    assert state.year_max is None
    assert state.origins == ()
    assert state.local_originals_only is False
    assert state.media_type is MediaType.ANY


def test_chip_overrides_text_same_field() -> None:
    state = _merge(
        ConstraintState.empty(),
        ConstraintDelta(moods=AddOp(values=("funny",))),
        DeltaSource.TEXT,
    )
    state = _merge(
        state,
        ConstraintDelta(moods=ReplaceOp(values=("cozy",))),
        DeltaSource.CHIP,
    )
    assert state.moods == (MoodId.COZY,)


def test_reset_soft_clears_soft_fields_only() -> None:
    prior = ConstraintState(
        media_type=MediaType.FILM,
        genres_include=(GenreId.DRAMA,),
        genres_exclude=(GenreId.HORROR,),
        moods=(MoodId.TENSE,),
        year_min=1990,
        year_max=1999,
        duration_max_min=120,
        origins=("US",),
        local_originals_only=True,
        languages=("en",),
        people_include=("p1",),
        people_exclude=("p2",),
        maturity_request_stricter=MaturityRating.PG,
        recency_bias=RecencyBias.TONIGHT,
    )
    ctx = ServerUserCtx(
        user_id="u1",
        profile_id="pr1",
        geo="US",
        package=Package.PREMIUM,
        maturity_max=PROFILE_R,
        kids_flag=False,
        device_class=DeviceClass.WEB,
    )
    after = _merge(prior, ConstraintDelta(reset_soft=True), profile=ctx.maturity_max)
    assert after.genres_include == ()
    assert after.genres_exclude == ()
    assert after.moods == ()
    assert after.year_min is None
    assert after.year_max is None
    assert after.duration_max_min is None
    assert after.origins == ()
    assert after.local_originals_only is False
    assert after.languages == ()
    assert after.people_include == ()
    assert after.people_exclude == ()
    assert after.maturity_request_stricter is None
    assert after.recency_bias is None
    assert after.media_type is MediaType.ANY
    # AuthZ lives on the profile, not on ConstraintState, and is unchanged.
    assert ctx.maturity_max is PROFILE_R
    assert ctx.kids_flag is False
    assert ctx.geo == "US"
    assert ctx.package is Package.PREMIUM
    assert ctx.device_class is DeviceClass.WEB


def test_raise_maturity_above_profile_ceiling_is_noop() -> None:
    prior = ConstraintState.empty()
    after = _merge(
        prior,
        ConstraintDelta(maturity_request_stricter=SetOp(value="R")),
        profile=PROFILE_PG13,
    )
    assert after.maturity_request_stricter is None
    assert after == prior


def test_maturity_equal_to_profile_is_noop() -> None:
    after = _merge(
        ConstraintState.empty(),
        ConstraintDelta(maturity_request_stricter=SetOp(value="PG-13")),
        profile=PROFILE_PG13,
    )
    assert after.maturity_request_stricter is None


def test_maturity_stricter_than_profile_sets() -> None:
    after = _merge(
        ConstraintState.empty(),
        ConstraintDelta(maturity_request_stricter=SetOp(value="PG")),
        profile=PROFILE_R,
    )
    assert after.maturity_request_stricter is MaturityRating.PG


def test_clear_maturity_reverts_to_profile_floor() -> None:
    prior = ConstraintState(maturity_request_stricter=MaturityRating.PG)
    after = _merge(prior, ConstraintDelta(maturity_request_stricter=ClearOp()))
    assert after.maturity_request_stricter is None


def test_merge_does_not_mutate_input_state() -> None:
    state = ConstraintState(
        moods=(MoodId.FUNNY,),
        genres_include=(GenreId.COMEDY,),
        origins=("CZ",),
    )
    snapshot = copy.deepcopy(state)
    dumped = state.model_dump()
    result = _merge(state, ConstraintDelta(moods=AddOp(values=("cozy",))))
    assert state == snapshot
    assert state.model_dump() == dumped
    assert state.moods == (MoodId.FUNNY,)
    assert result.moods == (MoodId.FUNNY, MoodId.COZY)
    assert result is not state


def test_hard_authz_fields_rejected_on_delta() -> None:
    for payload in (
        {"geo": {"op": "set", "value": "US"}},
        {"package": {"op": "set", "value": "premium"}},
        {"kids_flag": {"op": "set", "value": False}},
        {"device_class": {"op": "set", "value": "tv"}},
        {"maturity_max": {"op": "set", "value": "NC-17"}},
        {"profile_id": {"op": "set", "value": "other"}},
    ):
        with pytest.raises(ValidationError):
            ConstraintDelta.model_validate(payload)


def test_add_genres_is_set_union_preserving_order() -> None:
    state = _merge(
        ConstraintState.empty(),
        ConstraintDelta(genres_include=AddOp(values=("drama", "comedy"))),
    )
    state = _merge(state, ConstraintDelta(genres_include=AddOp(values=("comedy", "horror"))))
    assert state.genres_include == (GenreId.DRAMA, GenreId.COMEDY, GenreId.HORROR)


def test_remove_genre() -> None:
    prior = ConstraintState(genres_include=(GenreId.DRAMA, GenreId.COMEDY, GenreId.HORROR))
    after = _merge(prior, ConstraintDelta(genres_include=RemoveOp(values=("comedy",))))
    assert after.genres_include == (GenreId.DRAMA, GenreId.HORROR)


def test_replace_genres_is_not_union() -> None:
    prior = ConstraintState(genres_include=(GenreId.DRAMA, GenreId.COMEDY))
    after = _merge(prior, ConstraintDelta(genres_include=ReplaceOp(values=("horror",))))
    assert after.genres_include == (GenreId.HORROR,)


def test_clear_moods() -> None:
    prior = ConstraintState(moods=(MoodId.FUNNY, MoodId.COZY))
    after = _merge(prior, ConstraintDelta(moods=ClearOp()))
    assert after.moods == ()


def test_unknown_genre_value_skipped() -> None:
    after = _merge(
        ConstraintState.empty(),
        ConstraintDelta(genres_include=AddOp(values=("drama", "not-a-genre", "comedy"))),
    )
    assert after.genres_include == (GenreId.DRAMA, GenreId.COMEDY)


def test_exclude_genre_drops_from_include() -> None:
    prior = ConstraintState(genres_include=(GenreId.COMEDY, GenreId.DRAMA))
    after = _merge(prior, ConstraintDelta(genres_exclude=AddOp(values=("comedy",))))
    assert after.genres_exclude == (GenreId.COMEDY,)
    assert after.genres_include == (GenreId.DRAMA,)


def test_include_genre_drops_from_exclude() -> None:
    prior = ConstraintState(genres_exclude=(GenreId.COMEDY, GenreId.HORROR))
    after = _merge(prior, ConstraintDelta(genres_include=AddOp(values=("comedy",))))
    assert after.genres_include == (GenreId.COMEDY,)
    assert after.genres_exclude == (GenreId.HORROR,)


def test_same_delta_exclude_wins_overlap() -> None:
    after = _merge(
        ConstraintState.empty(),
        ConstraintDelta(
            genres_include=AddOp(values=("comedy", "drama")),
            genres_exclude=AddOp(values=("comedy",)),
        ),
    )
    assert after.genres_include == (GenreId.DRAMA,)
    assert after.genres_exclude == (GenreId.COMEDY,)


def test_people_include_and_exclude() -> None:
    state = _merge(
        ConstraintState.empty(),
        ConstraintDelta(people_include=AddOp(values=("p1", "p2"))),
    )
    state = _merge(state, ConstraintDelta(people_exclude=AddOp(values=("p2",))))
    assert state.people_include == ("p1",)
    assert state.people_exclude == ("p2",)


def test_set_year_range() -> None:
    after = _merge(
        ConstraintState.empty(),
        ConstraintDelta(year_min=SetOp(value=1990), year_max=SetOp(value=1999)),
    )
    assert after.year_min == 1990
    assert after.year_max == 1999


def test_inverted_year_range_swaps() -> None:
    after = _merge(
        ConstraintState.empty(),
        ConstraintDelta(year_min=SetOp(value=2000), year_max=SetOp(value=1980)),
    )
    assert after.year_min == 1980
    assert after.year_max == 2000


def test_set_duration_max() -> None:
    after = _merge(
        ConstraintState.empty(),
        ConstraintDelta(duration_max_min=SetOp(value=90)),
    )
    assert after.duration_max_min == 90


def test_set_and_clear_media_type() -> None:
    state = _merge(ConstraintState.empty(), ConstraintDelta(media_type=SetOp(value="film")))
    assert state.media_type is MediaType.FILM
    state = _merge(state, ConstraintDelta(media_type=ClearOp()))
    assert state.media_type is MediaType.ANY


def test_add_on_scalar_is_noop() -> None:
    prior = ConstraintState(media_type=MediaType.FILM, duration_max_min=90)
    after = _merge(
        prior,
        ConstraintDelta(
            media_type=AddOp(values=("series",)),
            duration_max_min=AddOp(values=("120",)),
        ),
    )
    assert after == prior


def test_set_on_list_is_noop() -> None:
    prior = ConstraintState(moods=(MoodId.FUNNY,))
    after = _merge(prior, ConstraintDelta(moods=SetOp(value="cozy")))
    assert after.moods == (MoodId.FUNNY,)


def test_remove_absent_value_is_noop() -> None:
    prior = ConstraintState(moods=(MoodId.FUNNY,))
    after = _merge(prior, ConstraintDelta(moods=RemoveOp(values=("cozy",))))
    assert after.moods == (MoodId.FUNNY,)


def test_recency_bias_and_languages() -> None:
    after = _merge(
        ConstraintState.empty(),
        ConstraintDelta(
            recency_bias=SetOp(value="tonight"),
            languages=AddOp(values=("en", "cs")),
        ),
    )
    assert after.recency_bias is RecencyBias.TONIGHT
    assert after.languages == ("en", "cs")


def test_local_originals_clear_is_false() -> None:
    prior = ConstraintState(local_originals_only=True)
    after = _merge(prior, ConstraintDelta(local_originals_only=ClearOp()))
    assert after.local_originals_only is False


def test_empty_delta_is_identity() -> None:
    prior = ConstraintState(moods=(MoodId.DARK,), year_min=2010)
    after = _merge(prior, ConstraintDelta())
    assert after == prior


def test_reset_soft_then_ops_in_same_delta() -> None:
    prior = ConstraintState(moods=(MoodId.FUNNY,), genres_include=(GenreId.COMEDY,))
    after = _merge(
        prior,
        ConstraintDelta(reset_soft=True, moods=AddOp(values=("cozy",))),
    )
    assert after.moods == (MoodId.COZY,)
    assert after.genres_include == ()
    assert after.media_type is MediaType.ANY


def test_rules_source_applies_like_text() -> None:
    after = _merge(
        ConstraintState.empty(),
        ConstraintDelta(genres_include=AddOp(values=("thriller",))),
        DeltaSource.RULES,
    )
    assert after.genres_include == (GenreId.THRILLER,)


def test_server_user_ctx_is_frozen() -> None:
    ctx = ServerUserCtx(
        user_id="u",
        profile_id="p",
        geo="CZ",
        package=Package.BASIC,
        maturity_max=MaturityRating.PG,
        kids_flag=True,
        device_class=DeviceClass.TV,
    )
    with pytest.raises(ValidationError):
        ctx.kids_flag = False  # type: ignore[misc]
