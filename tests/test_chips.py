"""Chip minting: known SpeechAct only; client payload is id + label."""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from assist.api.schemas import ChipOut, MetaOut, TurnResponse, turn_response_from_state
from assist.domain.catalog import Person
from assist.domain.constraints import AddOp, ConstraintDelta
from assist.domain.context import ServerUserCtx
from assist.domain.enums import (
    CreditRole,
    DegradedReason,
    DeviceClass,
    MaturityRating,
    Package,
    Route,
    SpeechAct,
)
from assist.graph.state import ReplyChip, empty_turn_state
from assist.nodes.chips import (
    BuiltinChipPhrases,
    UnknownSpeechActError,
    mint_chips,
    mint_one,
    require_speech_act,
    to_reply_chip,
)
from assist.stores.session import ChipRecord, Session


class FakeSessions:
    def __init__(self, session: Session) -> None:
        self.session = session

    async def load(self, session_id: str, user_id: str, profile_id: str) -> Session:
        return self.session

    async def save(self, session: Session) -> None:
        self.session = session


CHIPS_SRC = Path(__file__).resolve().parents[1] / "src" / "assist" / "nodes" / "chips.py"
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


def _session() -> Session:
    return Session.create(user_id="u1", profile_id="p1", session_id="s-chips")


def _chips_of(out: dict[str, object]) -> tuple[ReplyChip, ...]:
    raw = out["chips"]
    assert isinstance(raw, tuple)
    chips = cast(tuple[ReplyChip, ...], raw)
    assert all(isinstance(chip, ReplyChip) for chip in chips)
    return chips


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


def test_chips_imports_no_llm() -> None:
    imported = _imported_modules(CHIPS_SRC)
    assert not any(
        mod == banned or mod.startswith(banned + ".")
        for mod in imported
        for banned in _FORBIDDEN_IMPORTS
    )


def test_unknown_speech_act_is_refused_at_mint_time() -> None:
    with pytest.raises(UnknownSpeechActError) as exc:
        require_speech_act("invent_genre")
    assert exc.value.speech_act == "invent_genre"

    session = _session()
    phrases = BuiltinChipPhrases()
    with pytest.raises(UnknownSpeechActError):
        mint_one(session, "not_a_speech_act", phrases=phrases)


def test_known_speech_act_mints_via_session_mint_chip() -> None:
    session = _session()
    session, chips = mint_one(session, SpeechAct.REFINE_MOOD, phrases=BuiltinChipPhrases())
    assert len(chips) == 1
    chip = chips[0]
    assert isinstance(chip, ReplyChip)
    assert set(chip.model_dump()) == {"id", "label"}
    assert "delta" not in chip.model_dump()
    record = session.lookup_chip(chip.id)
    assert record.delta == ConstraintDelta(moods=AddOp(values=("funny",)))
    assert record.speech_act is SpeechAct.REFINE_MOOD
    assert record.label == chip.label


async def test_node_skips_unknown_act_and_mints_known() -> None:
    store = FakeSessions(_session())
    state = empty_turn_state(
        _ctx(),
        session_id=store.session.session_id,
        chip_speech_acts=("invent_genre", SpeechAct.RESET_SOFT),
    )
    out = await mint_chips(state, sessions=store)
    chips = _chips_of(out)
    assert len(chips) == 1
    assert chips[0].label == "Start over"
    assert store.session.lookup_chip(chips[0].id).delta.reset_soft is True


async def test_client_facing_chips_are_id_and_label_only() -> None:
    store = FakeSessions(_session())
    state = empty_turn_state(
        _ctx(),
        session_id=store.session.session_id,
        chip_speech_acts=(SpeechAct.REFINE_DURATION,),
    )
    out = await mint_chips(state, sessions=store)
    chips = _chips_of(out)
    assert len(chips) == 1
    dumped = chips[0].model_dump()
    assert dumped.keys() == {"id", "label"}
    assert "delta" not in dumped
    assert "speech_act" not in dumped

    merged = dict(state)
    merged.update(out)
    body = turn_response_from_state(merged, latency_ms=12, trace_id="tr-chip")
    chip_payload = body.model_dump()["chips"]
    assert chip_payload == [{"id": chips[0].id, "label": chips[0].label}]
    raw = body.model_dump_json()
    parsed = json.loads(raw)
    for item in parsed["chips"]:
        assert set(item) == {"id", "label"}
        assert "delta" not in item
    assert "delta" not in raw


def test_response_shaped_objects_never_carry_delta() -> None:
    """ChipRecord holds delta by design. Response types must not grow that field."""
    assert "delta" in ChipRecord.model_fields
    assert set(ReplyChip.model_fields) == {"id", "label"}
    assert set(ChipOut.model_fields) == {"id", "label"}
    assert "delta" not in ReplyChip.model_fields
    assert "delta" not in ChipOut.model_fields
    assert "delta" not in TurnResponse.model_fields

    record = ChipRecord(
        chip_id="c_secret",
        label="Funnier",
        delta=ConstraintDelta(moods=AddOp(values=("funny",))),
        speech_act=SpeechAct.REFINE_MOOD,
        minted_turn=0,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    assert "delta" in record.model_dump()
    projected = to_reply_chip(record)
    assert projected.model_dump() == {"id": "c_secret", "label": "Funnier"}

    leaked = ChipOut.model_validate(
        {"id": record.chip_id, "label": record.label, "delta": record.delta.model_dump()}
    )
    assert leaked.model_dump() == {"id": "c_secret", "label": "Funnier"}

    body = TurnResponse(
        session_id="s",
        reply="hi",
        chips=[ChipOut(id=record.chip_id, label=record.label)],
        meta=MetaOut(degraded=False, latency_ms=1, trace_id="t"),
    )
    as_json = json.loads(body.model_dump_json())
    assert "delta" not in as_json
    assert as_json["chips"] == [{"id": "c_secret", "label": "Funnier"}]


async def test_person_disambiguate_binds_server_person_id() -> None:
    store = FakeSessions(_session())
    person = Person(
        person_id="p_pacino",
        name="Al Pacino",
        name_norm="al pacino",
        roles=(CreditRole.ACTOR,),
        credit_count=12,
    )
    state = empty_turn_state(
        _ctx(),
        session_id=store.session.session_id,
        route=Route.CLARIFY,
        degraded_reason=DegradedReason.PERSON_AMBIGUOUS,
        people_candidates=(person,),
    )
    out = await mint_chips(state, sessions=store)
    chips = _chips_of(out)
    assert len(chips) == 1
    assert chips[0].label == "Al Pacino"
    assert set(chips[0].model_dump()) == {"id", "label"}
    delta = store.session.lookup_chip(chips[0].id).delta
    assert delta.people_include == AddOp(values=("p_pacino",))
