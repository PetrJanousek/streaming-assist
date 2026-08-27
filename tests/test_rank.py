"""Rank node: config weights, per-set min-max, vector-off, deterministic ties."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from assist.config import Settings, settings
from assist.domain.catalog import Candidate
from assist.domain.constraints import ConstraintState
from assist.domain.context import ServerUserCtx
from assist.domain.enums import (
    DeviceClass,
    GenreId,
    MaturityRating,
    MediaType,
    MoodId,
    Package,
)
from assist.graph.build import build_graph
from assist.graph.state import TurnState, empty_turn_state
from assist.nodes.rank import (
    RankFeatures,
    RankWeights,
    constraint_match,
    fill_cold_start,
    fused_score,
    log_pop,
    min_max_norm,
    rank,
    rank_candidates,
)

# Documented fixture (see test_documented_fixture_order).
# pop_28d = expm1(x) so log1p is exact: alpha=10, bravo=7, low=0 → norms 1.0, 0.7, 0.0.
_W = RankWeights(pop=0.50, constraint=0.30, semantic=0.20)
_FIXTURE_WEIGHTS = _W


def _ctx() -> ServerUserCtx:
    return ServerUserCtx(
        user_id="u1",
        profile_id="p1",
        geo="US",
        package=Package.BASIC,
        maturity_max=MaturityRating.PG_13,
        kids_flag=False,
        device_class=DeviceClass.WEB,
    )


def _cand(
    catalog_id: str,
    title: str,
    *,
    media_type: MediaType = MediaType.FILM,
    year: int | None = 2015,
    genres: tuple[GenreId, ...] = (GenreId.DRAMA,),
    score: float = 0.0,
) -> Candidate:
    return Candidate(
        catalog_id=catalog_id,
        title=title,
        media_type=media_type,
        release_year=year,
        genres=genres,
        score=score,
    )


def _ids(ranked: list[Candidate] | tuple[Candidate, ...]) -> list[str]:
    return [item.catalog_id for item in ranked]


def _ids_from_state(raw: object) -> list[str]:
    assert isinstance(raw, (list, tuple))
    return _ids(tuple(item for item in raw if isinstance(item, Candidate)))


def _documented_candidates() -> tuple[list[Candidate], dict[str, RankFeatures], ConstraintState]:
    candidates = [
        _cand("s-alpha", "Alpha", score=0.0),
        _cand("s-bravo", "Bravo", score=1.0),
        _cand("s-low", "Low", genres=(GenreId.COMEDY,), year=1990, score=0.5),
    ]
    features = {
        "s-alpha": RankFeatures(pop_28d=math.expm1(10), semantic_score=0.0),
        "s-bravo": RankFeatures(pop_28d=math.expm1(7), semantic_score=1.0),
        "s-low": RankFeatures(pop_28d=0.0, semantic_score=0.5),
    }
    constraints = ConstraintState(genres_include=(GenreId.DRAMA,))
    return candidates, features, constraints


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def test_config_weights_sum_to_one() -> None:
    total = settings.rank_w_pop + settings.rank_w_constraint + settings.rank_w_semantic
    assert total == pytest.approx(1.0)
    loaded = RankWeights.from_settings()
    assert loaded.pop == settings.rank_w_pop
    assert loaded.constraint == settings.rank_w_constraint
    assert loaded.semantic == settings.rank_w_semantic


def test_rank_w_pop_point_four_raises() -> None:
    with pytest.raises(ValidationError, match=r"sum to 1\.0"):
        Settings(rank_w_pop=0.4, rank_w_constraint=0.3, rank_w_semantic=0.2)


def test_rank_uses_config_weights_not_literals() -> None:
    """Pop-only weights put the high-pop title first even if semantic is inverted."""
    cfg = Settings(rank_w_pop=1.0, rank_w_constraint=0.0, rank_w_semantic=0.0)
    candidates = [
        _cand("s-sem", "Semantic", score=1.0),
        _cand("s-pop", "Popular", score=0.0),
    ]
    features = {
        "s-sem": RankFeatures(pop_28d=1.0, semantic_score=1.0),
        "s-pop": RankFeatures(pop_28d=100.0, semantic_score=0.0),
    }
    ranked = rank_candidates(
        candidates,
        ConstraintState.empty(),
        features=features,
        weights=RankWeights.from_settings(cfg),
    )
    assert _ids(ranked) == ["s-pop", "s-sem"]


def test_rank_module_does_not_call_builtin_hash() -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "assist" / "nodes" / "rank.py"
    code = source.read_text(encoding="utf-8")
    # Comments may mention the builtin; the call form `hash(` must not appear.
    code_no_comments = "\n".join(
        line for line in code.splitlines() if not line.lstrip().startswith("#")
    )
    assert "hash(" not in code_no_comments


# ---------------------------------------------------------------------------
# Documented fixture
# ---------------------------------------------------------------------------


def test_documented_fixture_order() -> None:
    """Fixed fixture → stable documented order.

    pop_28d = expm1(x) so log1p is exact. Constraint = drama include.

        id       pop_norm  constraint  semantic_norm
        s-alpha  1.0       1.0         0.0
        s-bravo  0.7       1.0         1.0
        s-low    0.0       0.0         0.5

    With vector (w = 0.50 / 0.30 / 0.20):
        alpha = 0.50*1.0 + 0.30*1.0 + 0.20*0.0 = 0.80
        bravo = 0.50*0.7 + 0.30*1.0 + 0.20*1.0 = 0.85
        low   = 0.50*0.0 + 0.30*0.0 + 0.20*0.5 = 0.10
    order: s-bravo, s-alpha, s-low
    """
    candidates, features, constraints = _documented_candidates()
    ranked = rank_candidates(
        candidates,
        constraints,
        features=features,
        vector_path=True,
        weights=_FIXTURE_WEIGHTS,
    )
    assert _ids(ranked) == ["s-bravo", "s-alpha", "s-low"]
    assert ranked[0].score == pytest.approx(0.85)
    assert ranked[1].score == pytest.approx(0.80)
    assert ranked[2].score == pytest.approx(0.10)


def test_vector_path_off_changes_order_and_does_not_crash() -> None:
    """Same fixture, semantic_norm = 0. Alpha (pop) overtakes bravo.

        alpha = 0.80
        bravo = 0.50*0.7 + 0.30*1.0 = 0.65
        low   = 0.00
    order: s-alpha, s-bravo, s-low
    """
    candidates, features, constraints = _documented_candidates()
    ranked = rank_candidates(
        candidates,
        constraints,
        features=features,
        vector_path=False,
        weights=_FIXTURE_WEIGHTS,
    )
    assert _ids(ranked) == ["s-alpha", "s-bravo", "s-low"]
    assert ranked[0].score == pytest.approx(0.80)
    assert ranked[1].score == pytest.approx(0.65)


def test_vector_path_off_empty_set_does_not_crash() -> None:
    assert rank_candidates((), ConstraintState.empty(), vector_path=False) == []


# ---------------------------------------------------------------------------
# Ties / determinism
# ---------------------------------------------------------------------------


def test_ties_break_on_catalog_id() -> None:
    candidates = [
        _cand("s-zeta", "Zeta", score=0.4),
        _cand("s-alpha", "Alpha", score=0.4),
        _cand("s-mu", "Mu", score=0.4),
    ]
    features = {
        item.catalog_id: RankFeatures(pop_28d=10.0, semantic_score=0.4) for item in candidates
    }
    ranked = rank_candidates(candidates, ConstraintState.empty(), features=features)
    assert _ids(ranked) == ["s-alpha", "s-mu", "s-zeta"]


def test_tiebreak_identical_across_two_processes() -> None:
    """PYTHONHASHSEED must not change order. catalog_id, never hash()."""
    script = textwrap.dedent(
        """
        from assist.domain.catalog import Candidate
        from assist.domain.constraints import ConstraintState
        from assist.domain.enums import GenreId, MediaType
        from assist.nodes.rank import RankFeatures, rank_candidates

        def cand(cid: str) -> Candidate:
            return Candidate(
                catalog_id=cid,
                title=cid,
                media_type=MediaType.FILM,
                release_year=2010,
                genres=(GenreId.DRAMA,),
                score=0.4,
            )

        ids = ["s-zeta", "s-alpha", "s-mu", "s-bravo"]
        candidates = [cand(i) for i in ids]
        features = {i: RankFeatures(pop_28d=10.0, semantic_score=0.4) for i in ids}
        ranked = rank_candidates(candidates, ConstraintState.empty(), features=features)
        print(",".join(c.catalog_id for c in ranked))
        """
    )
    env_a = {**os.environ, "PYTHONHASHSEED": "0"}
    env_b = {**os.environ, "PYTHONHASHSEED": "2147483647"}
    out_a = subprocess.check_output([sys.executable, "-c", script], env=env_a, text=True)
    out_b = subprocess.check_output([sys.executable, "-c", script], env=env_b, text=True)
    assert out_a.strip() == "s-alpha,s-bravo,s-mu,s-zeta"
    assert out_a == out_b


# ---------------------------------------------------------------------------
# Cold start, min-max, constraint match
# ---------------------------------------------------------------------------


def test_minmax_is_per_candidate_set() -> None:
    assert min_max_norm([1.0, 3.0, 5.0]) == pytest.approx([0.0, 0.5, 1.0])
    assert min_max_norm([10.0, 10.0, 10.0]) == [0.0, 0.0, 0.0]
    assert min_max_norm([]) == []


def test_cold_start_uses_median_of_known_pops() -> None:
    filled = fill_cold_start([10.0, None, 30.0])
    assert filled == [10.0, 20.0, 30.0]


def test_cold_start_all_missing_does_not_crash() -> None:
    candidates = [_cand("s-a", "A", score=0.9), _cand("s-b", "B", score=0.1)]
    ranked = rank_candidates(candidates, ConstraintState.empty())
    assert _ids(ranked) == ["s-a", "s-b"]


def test_constraint_match_fraction_and_mood_unenriched() -> None:
    features = RankFeatures(
        media_type=MediaType.FILM,
        genres=(GenreId.DRAMA,),
        release_year=2015,
        moods=(),
    )
    constraints = ConstraintState(
        media_type=MediaType.FILM,
        genres_include=(GenreId.DRAMA,),
        moods=(MoodId.COZY,),
    )
    # film yes, drama yes, cozy no (empty moods — live catalog is unenriched)
    assert constraint_match(features, constraints) == pytest.approx(2.0 / 3.0)


def test_constraint_match_no_active_is_zero() -> None:
    assert constraint_match(RankFeatures(), ConstraintState.empty()) == 0.0


def test_fused_score_uses_passed_weights() -> None:
    mix = RankWeights(pop=0.0, constraint=0.0, semantic=1.0)
    got = fused_score(pop_norm=1.0, constraint=1.0, semantic_norm=0.4, weights=mix)
    assert got == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Franchise cap / no resurrection
# ---------------------------------------------------------------------------


def test_franchise_cap_one_per_top_8() -> None:
    # Score order via semantic: Narcos, Narcos: Mexico, then 7 unique titles.
    names = [
        ("s-narcos", "Narcos"),
        ("s-mexico", "Narcos: Mexico"),
        ("s-ozark", "Ozark"),
        ("s-dark", "Dark"),
        ("s-heist", "Money Heist"),
        ("s-crown", "The Crown"),
        ("s-sherlock", "Sherlock"),
        ("s-bad", "Breaking Bad"),
        ("s-wire", "The Wire"),
    ]
    candidates = [
        _cand(catalog_id, title, score=1.0 - (i * 0.05))
        for i, (catalog_id, title) in enumerate(names)
    ]
    ranked = rank_candidates(candidates, ConstraintState.empty(), vector_path=True)
    top8 = _ids(ranked)[:8]
    assert "s-narcos" in top8
    assert "s-mexico" not in top8
    assert ranked[8].catalog_id == "s-mexico"
    assert set(_ids(ranked)) == {catalog_id for catalog_id, _title in names}


def test_rank_does_not_resurrect_or_drop_ids() -> None:
    candidates = [
        _cand("s-c", "C", score=0.1),
        _cand("s-a", "A", score=0.9),
        _cand("s-b", "B", score=0.5),
    ]
    ranked = rank_candidates(candidates, ConstraintState.empty())
    assert set(_ids(ranked)) == {"s-a", "s-b", "s-c"}
    assert len(ranked) == 3


def test_duplicate_input_ids_are_not_cloned() -> None:
    a = _cand("s-a", "A", score=0.5)
    ranked = rank_candidates([a, a], ConstraintState.empty())
    assert _ids(ranked) == ["s-a"]


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rank_node_writes_top1_gap_not_retrieve_attempts() -> None:
    candidates, features, constraints = _documented_candidates()
    state = empty_turn_state(
        _ctx(),
        candidates=tuple(candidates),
        constraints=constraints,
        retrieve_attempts=1,
        retrieve_max_attempts=2,
    )
    cfg = Settings(rank_w_pop=0.50, rank_w_constraint=0.30, rank_w_semantic=0.20)
    update = await rank(state, features=features, vector_path=True, cfg=cfg)
    assert "retrieve_attempts" not in update
    assert "retrieve_max_attempts" not in update
    assert _ids_from_state(update["candidates"]) == ["s-bravo", "s-alpha", "s-low"]
    assert update["top1"] == pytest.approx(0.85)
    assert update["gap"] == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_rank_node_empty_candidates() -> None:
    update = await rank(empty_turn_state(_ctx()))
    assert update["candidates"] == ()
    assert update["top1"] is None
    assert update["gap"] is None


@pytest.mark.asyncio
async def test_rank_node_vector_off_never_crashes() -> None:
    candidates, features, constraints = _documented_candidates()
    state = empty_turn_state(_ctx(), candidates=tuple(candidates), constraints=constraints)
    cfg = Settings(rank_w_pop=0.50, rank_w_constraint=0.30, rank_w_semantic=0.20)
    update = await rank(state, features=features, vector_path=False, cfg=cfg)
    assert _ids_from_state(update["candidates"]) == ["s-alpha", "s-bravo", "s-low"]


@pytest.mark.asyncio
async def test_rank_as_graph_node_preserves_id_set() -> None:
    candidates, features, constraints = _documented_candidates()
    cfg = Settings(rank_w_pop=0.50, rank_w_constraint=0.30, rank_w_semantic=0.20)

    async def _rank_node(state: TurnState) -> dict[str, object]:
        return await rank(state, features=features, vector_path=True, cfg=cfg)

    graph = build_graph(node_overrides={"rank": _rank_node})
    result = await graph.ainvoke(
        empty_turn_state(_ctx(), candidates=tuple(candidates), constraints=constraints)
    )
    assert _ids_from_state(result["candidates"]) == ["s-bravo", "s-alpha", "s-low"]
    assert result["top1"] == pytest.approx(0.85)


def test_log_pop_matches_documented_fixture() -> None:
    assert log_pop(math.expm1(10)) == pytest.approx(10.0)
    assert log_pop(math.expm1(7)) == pytest.approx(7.0)
    assert log_pop(0.0) == pytest.approx(0.0)
    norms = min_max_norm([10.0, 7.0, 0.0])
    assert norms == pytest.approx([1.0, 0.7, 0.0])
