"""sanitize_picks allowlist + min_picks policy matrix."""

from __future__ import annotations

import pytest

from assist.domain.catalog import Candidate, Pick, Title
from assist.domain.enums import (
    DegradedReason,
    GenreId,
    MediaType,
    Route,
)
from assist.domain.picks import min_picks_for, sanitize_picks


def _c(catalog_id: str, title: str = "") -> Candidate:
    return Candidate(
        catalog_id=catalog_id,
        title=title or catalog_id,
        media_type=MediaType.FILM,
        genres=(GenreId.DRAMA,),
        score=1.0,
    )


CANDIDATES = (_c("a"), _c("b"), _c("c"), _c("d"), _c("e"))
ENTITLED = frozenset({"a", "b", "c", "d", "e"})


def test_id_not_in_candidates_is_dropped() -> None:
    picks = sanitize_picks(
        model_ids=["zzz", "a"],
        candidates=CANDIDATES,
        entitled=ENTITLED,
        min_picks=0,
    )
    assert picks == ("a",)


def test_unentitled_id_is_dropped() -> None:
    picks = sanitize_picks(
        model_ids=["a", "b"],
        candidates=CANDIDATES,
        entitled=frozenset({"a", "c"}),
        min_picks=0,
    )
    assert picks == ("a",)


def test_min_picks_zero_never_pads() -> None:
    picks = sanitize_picks(
        model_ids=["a"],
        candidates=CANDIDATES,
        entitled=ENTITLED,
        min_picks=0,
    )
    assert picks == ("a",)


def test_min_picks_three_pads_in_rank_order() -> None:
    picks = sanitize_picks(
        model_ids=["d"],
        candidates=CANDIDATES,
        entitled=ENTITLED,
        min_picks=3,
    )
    # model pick kept first; pad walks candidates a, b, c, d, e skipping d
    assert picks == ("d", "a", "b")


def test_empty_entitled_set_returns_empty() -> None:
    picks = sanitize_picks(
        model_ids=["a", "b", "c"],
        candidates=CANDIDATES,
        entitled=frozenset(),
        min_picks=3,
    )
    assert picks == ()


def test_duplicate_model_ids_preserve_first_order() -> None:
    picks = sanitize_picks(
        model_ids=["c", "a", "c", "a"],
        candidates=CANDIDATES,
        entitled=ENTITLED,
        min_picks=0,
    )
    assert picks == ("c", "a")


def test_pad_skips_unentitled_and_already_included() -> None:
    picks = sanitize_picks(
        model_ids=["c"],
        candidates=CANDIDATES,
        entitled=frozenset({"c", "e"}),
        min_picks=3,
    )
    # only c and e are legal; cannot invent a third
    assert picks == ("c", "e")


def test_max_picks_caps_output() -> None:
    picks = sanitize_picks(
        model_ids=["a", "b", "c", "d", "e"],
        candidates=CANDIDATES,
        entitled=ENTITLED,
        min_picks=3,
        max_picks=2,
    )
    assert picks == ("a", "b")


def test_entitled_but_not_in_candidates_dropped() -> None:
    picks = sanitize_picks(
        model_ids=["ghost"],
        candidates=CANDIDATES,
        entitled=frozenset({"ghost", "a"}),
        min_picks=0,
    )
    assert picks == ()


def test_min_picks_policy_matrix() -> None:
    assert min_picks_for(route=Route.TEMPLATE, entitled_count=10) == 3
    assert min_picks_for(route=Route.GENERATIVE, entitled_count=10) == 3
    assert (
        min_picks_for(
            route=Route.CLARIFY,
            degraded_reason=DegradedReason.PERSON_AMBIGUOUS,
            entitled_count=10,
        )
        == 0
    )
    assert (
        min_picks_for(
            degraded_reason=DegradedReason.EMPTY_CATALOG_MATCH,
            entitled_count=10,
        )
        == 0
    )
    assert (
        min_picks_for(
            route=Route.SAFETY,
            degraded_reason=DegradedReason.SAFETY_BLOCK,
            entitled_count=10,
        )
        == 0
    )
    assert (
        min_picks_for(
            route=Route.TEMPLATE,
            degraded_reason=DegradedReason.GENERATIVE_SCHEMA_FAIL,
            entitled_count=10,
        )
        == 3
    )
    assert (
        min_picks_for(
            route=Route.DEGRADED_KEYWORD,
            degraded_reason=DegradedReason.HARD_TIMEOUT,
            entitled_count=4,
        )
        == 3
    )
    assert (
        min_picks_for(
            route=Route.DEGRADED_KEYWORD,
            degraded_reason=DegradedReason.HARD_TIMEOUT,
            entitled_count=0,
        )
        == 0
    )
    assert (
        min_picks_for(
            route=Route.DEGRADED_KEYWORD,
            degraded_reason=DegradedReason.PROVIDER_THROTTLE,
            entitled_count=2,
        )
        == 3
    )
    assert min_picks_for(route=Route.TEMPLATE, entitled_count=0) == 0


def test_min_picks_for_empty_entitled_forces_zero_even_on_success_route() -> None:
    assert (
        min_picks_for(
            route=Route.GENERATIVE,
            degraded_reason=DegradedReason.NONE,
            entitled_count=0,
        )
        == 0
    )


def test_pick_and_title_round_trip_shape() -> None:
    title = Title(
        catalog_id="s1",
        media_type=MediaType.FILM,
        title="Example",
        maturity_rank=5,
        genres=(GenreId.DRAMA,),
    )
    pick = Pick(catalog_id=title.catalog_id, reason_short="quiet drama")
    assert pick.catalog_id == "s1"
    assert title.maturity_rank == 5


@pytest.mark.parametrize(
    "model_ids",
    [["b"], ["b", "missing"]],
)
def test_pad_uses_candidate_sequence_not_sorted_ids(model_ids: list[str]) -> None:
    ranked = (_c("z"), _c("m"), _c("b"), _c("a"))
    picks = sanitize_picks(
        model_ids=model_ids,
        candidates=ranked,
        entitled=frozenset({"z", "m", "b", "a"}),
        min_picks=3,
    )
    assert picks[0] == "b"
    assert picks[1] == "z"
    assert picks[2] == "m"
