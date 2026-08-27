"""Sanitize node: allowlist wrapper + reply title-span grounding."""

from __future__ import annotations

import ast
from pathlib import Path

from assist.config import Settings
from assist.domain.catalog import Candidate, Pick
from assist.domain.context import ServerUserCtx
from assist.domain.enums import (
    DegradedReason,
    DeviceClass,
    GenreId,
    MaturityRating,
    MediaType,
    Package,
    Route,
)
from assist.graph.state import TurnState, empty_turn_state
from assist.nodes.sanitize import (
    FALLBACK_REPLY,
    ground_reply,
    model_ids_from_state,
    sanitize,
)

SANITIZE_SRC = Path(__file__).resolve().parents[1] / "src" / "assist" / "nodes" / "sanitize.py"
_FORBIDDEN_IMPORTS = frozenset(
    {
        "assist.llm",
        "langchain_anthropic",
        "langchain_openai",
        "anthropic",
        "openai",
    }
)


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


def _c(catalog_id: str, title: str) -> Candidate:
    return Candidate(
        catalog_id=catalog_id,
        title=title,
        media_type=MediaType.FILM,
        genres=(GenreId.DRAMA,),
        score=1.0,
    )


CANDIDATES = (
    _c("ttl_a", "The Irishman"),
    _c("ttl_b", "Heat"),
    _c("ttl_c", "Casino"),
    _c("ttl_d", "Goodfellas"),
)
ENTITLED = ("ttl_a", "ttl_b", "ttl_c")


def _state(**overrides: object) -> TurnState:
    payload: dict[str, object] = {
        "candidates": CANDIDATES,
        "entitled_ids": ENTITLED,
        "route": Route.GENERATIVE,
    }
    payload.update(overrides)
    return empty_turn_state(_ctx(), **payload)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_sanitize_imports_no_llm() -> None:
    imported = _imported_modules(SANITIZE_SRC)
    assert not any(
        mod == banned or mod.startswith(banned + ".")
        for mod in imported
        for banned in _FORBIDDEN_IMPORTS
    )


def test_out_of_range_index_is_dropped() -> None:
    ids = model_ids_from_state(
        empty_turn_state(_ctx(), model_pick_indices=(0, 99, -1, 1)),
        CANDIDATES,
    )
    assert ids == ("ttl_a", "ttl_b")


async def test_unentitled_pick_is_dropped_not_substituted() -> None:
    state = _state(model_pick_ids=("ttl_d", "ttl_a"), min_picks=3)
    out = await sanitize(state)
    picks = out["picks"]
    assert isinstance(picks, tuple)
    assert all(isinstance(p, Pick) for p in picks)
    ids = tuple(p.catalog_id for p in picks)
    assert "ttl_d" not in ids
    assert "ttl_a" in ids
    # pad uses entitled rank order; never a lookalike for the dropped id
    assert "ttl_d" not in ids


async def test_empty_entitled_fails_closed() -> None:
    state = empty_turn_state(
        _ctx(),
        candidates=CANDIDATES,
        entitled_ids=(),
        model_pick_ids=("ttl_a", "ttl_b"),
        route=Route.TEMPLATE,
    )
    out = await sanitize(state)
    assert out["picks"] == ()
    assert out["min_picks"] == 0


async def test_missing_entitled_ids_fails_closed() -> None:
    state = empty_turn_state(
        _ctx(),
        candidates=CANDIDATES,
        model_pick_ids=("ttl_a",),
        route=Route.TEMPLATE,
    )
    # empty_turn_state sets entitled_ids=(); the node must not treat missing as all-playable
    out = await sanitize(state)
    assert out["picks"] == ()


def test_ground_reply_strips_one_off_catalog_span() -> None:
    reply = 'Tonight try "Ghost Movie" with one of these.'
    grounded, action = ground_reply(reply, {"the irishman", "heat"})
    assert action == "strip"
    assert "Ghost Movie" not in grounded
    assert "Tonight try" in grounded
    assert "with one of these." in grounded


def test_ground_reply_replaces_when_leftover_too_short() -> None:
    reply = '"Ghost Movie" is a masterpiece'
    grounded, action = ground_reply(reply, {"the irishman"})
    assert action == "template"
    assert grounded == FALLBACK_REPLY
    assert "Ghost Movie" not in grounded


def test_ground_reply_replaces_when_multiple_off_catalog_spans() -> None:
    reply = 'Skip "Ghost One" and also skip "Ghost Two" tonight with friends around.'
    grounded, action = ground_reply(reply, {"the irishman"})
    assert action == "template"
    assert grounded == FALLBACK_REPLY


def test_ground_reply_keeps_candidate_title() -> None:
    reply = 'Start with "The Irishman" if you want a long night in.'
    grounded, action = ground_reply(reply, {"the irishman", "heat"})
    assert action == "keep"
    assert grounded == reply


async def test_sanitize_strips_off_catalog_title_in_reply() -> None:
    state = _state(
        model_pick_ids=("ttl_a",),
        reply='Tonight try "Ghost Movie" with one of these.',
    )
    out = await sanitize(state)
    assert "Ghost Movie" not in str(out["reply"])
    assert "Tonight try" in str(out["reply"])
    assert FALLBACK_REPLY not in str(out["reply"])


async def test_sanitize_replaces_reply_when_span_is_the_whole_prose() -> None:
    state = _state(
        model_pick_ids=("ttl_a",),
        reply='"Ghost Movie" is a masterpiece',
    )
    out = await sanitize(state)
    assert out["reply"] == FALLBACK_REPLY


async def test_sanitize_keeps_candidate_title_in_prose() -> None:
    state = _state(
        model_pick_ids=("ttl_a",),
        reply='Start with "The Irishman" if you want a long night in.',
    )
    out = await sanitize(state)
    assert "The Irishman" in str(out["reply"])


async def test_sanitize_truncates_to_reply_max_chars() -> None:
    cfg = Settings(reply_max_chars=40)
    state = _state(
        model_pick_ids=("ttl_a",),
        reply="Start with The Irishman if you want a very long night in with friends.",
    )
    out = await sanitize(state, settings=cfg)
    assert len(str(out["reply"])) <= 40


async def test_safety_route_does_not_pad_picks() -> None:
    state = _state(
        model_pick_ids=("ttl_a",),
        route=Route.SAFETY,
        degraded_reason=DegradedReason.SAFETY_BLOCK,
    )
    out = await sanitize(state)
    picks = out["picks"]
    assert isinstance(picks, tuple)
    ids = tuple(p.catalog_id for p in picks)
    assert ids == ("ttl_a",)
    assert out["min_picks"] == 0
