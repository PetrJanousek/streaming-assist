"""Request and response models for the assist HTTP surface."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from assist.config import settings
from assist.domain.catalog import Pick
from assist.domain.enums import DegradedReason, DeviceClass
from assist.graph.state import ReplyChip, TurnState


class ClientHints(BaseModel):
    """UI/analytics only. Extra keys (geo, package, kids, maturity) are dropped."""

    model_config = ConfigDict(extra="ignore")

    device_class: DeviceClass | None = None
    ui_language: str | None = None
    app_version: str | None = None


class TurnMessage(BaseModel):
    """Client message. Chip follow-ups send chip_id only — never a delta."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["text", "chip"]
    text: str | None = None
    chip_id: str | None = None

    @model_validator(mode="after")
    def _require_payload(self) -> Self:
        if self.type == "text" and not (self.text or "").strip():
            raise ValueError("text message requires non-empty text")
        if self.type == "chip" and not (self.chip_id or "").strip():
            raise ValueError("chip message requires chip_id")
        return self


class TurnRequest(BaseModel):
    """Turn body. AuthZ fields at the top level are ignored, never trusted."""

    model_config = ConfigDict(extra="ignore")

    session_id: str | None = None
    message: TurnMessage
    client_hints: ClientHints | None = None


class PickOut(BaseModel):
    catalog_id: str
    reason_short: str = ""


class ChipOut(BaseModel):
    """Client-facing chip. The delta stays on the server."""

    id: str
    label: str


class MetaOut(BaseModel):
    degraded: bool
    degraded_reason: str | None = None
    latency_ms: int
    trace_id: str
    route: str | None = None
    intent_source: str | None = None
    stage_latency_ms: dict[str, int] | None = None


class TurnResponse(BaseModel):
    session_id: str
    reply: str
    picks: list[PickOut] = Field(default_factory=list)
    chips: list[ChipOut] = Field(default_factory=list)
    meta: MetaOut


class ErrorDetail(BaseModel):
    type: str
    message: str
    retry_after_ms: int | None = None


class ErrorEnvelope(TurnResponse):
    """Turn-shaped body plus an error object so clients can always parse."""

    error: ErrorDetail


def _enum_value(value: object) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    text = str(raw)
    return text if text else None


def _pick_out(raw: Pick | object) -> PickOut:
    if isinstance(raw, Pick):
        return PickOut(catalog_id=raw.catalog_id, reason_short=raw.reason_short)
    catalog_id = getattr(raw, "catalog_id", None)
    reason = getattr(raw, "reason_short", "")
    if catalog_id is None and isinstance(raw, dict):
        catalog_id = raw.get("catalog_id", "")
        reason = raw.get("reason_short", "")
    return PickOut(catalog_id=str(catalog_id or ""), reason_short=str(reason or ""))


def _chip_out(raw: ReplyChip | object) -> ChipOut:
    if isinstance(raw, ReplyChip):
        return ChipOut(id=raw.id, label=raw.label)
    chip_id = getattr(raw, "id", None)
    label = getattr(raw, "label", "")
    if chip_id is None and isinstance(raw, dict):
        chip_id = raw.get("id", "")
        label = raw.get("label", "")
    return ChipOut(id=str(chip_id or ""), label=str(label or ""))


def turn_response_from_state(
    state: TurnState | dict[str, object],
    *,
    latency_ms: int,
    trace_id: str,
) -> TurnResponse:
    reason = state.get("degraded_reason") or DegradedReason.NONE
    reason_value = _enum_value(reason)
    degraded = reason_value is not None and reason_value != DegradedReason.NONE.value
    meta = MetaOut(
        degraded=degraded,
        degraded_reason=reason_value if degraded else None,
        latency_ms=latency_ms,
        trace_id=trace_id,
        route=_enum_value(state.get("route")),
        intent_source=_enum_value(state.get("intent_source")),
    )
    if settings.debug_meta:
        raw_timings = state.get("timings")
        timings = raw_timings if isinstance(raw_timings, dict) else {}
        meta.stage_latency_ms = {str(k): int(v) for k, v in timings.items()}
    raw_picks = state.get("picks")
    raw_chips = state.get("chips")
    picks = [_pick_out(p) for p in (raw_picks if isinstance(raw_picks, (list, tuple)) else ())]
    chips = [_chip_out(c) for c in (raw_chips if isinstance(raw_chips, (list, tuple)) else ())]
    return TurnResponse(
        session_id=str(state.get("session_id") or ""),
        reply=str(state.get("reply") or ""),
        picks=picks,
        chips=chips,
        meta=meta,
    )


def error_envelope(
    *,
    error_type: str,
    message: str,
    trace_id: str,
    latency_ms: int = 0,
    retry_after_ms: int | None = None,
    session_id: str = "",
    degraded: bool = False,
    degraded_reason: str | None = None,
) -> ErrorEnvelope:
    return ErrorEnvelope(
        session_id=session_id,
        reply="",
        picks=[],
        chips=[],
        meta=MetaOut(
            degraded=degraded,
            degraded_reason=degraded_reason,
            latency_ms=latency_ms,
            trace_id=trace_id,
        ),
        error=ErrorDetail(type=error_type, message=message, retry_after_ms=retry_after_ms),
    )
