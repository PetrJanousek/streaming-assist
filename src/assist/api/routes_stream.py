"""SSE turn stream and the no-build demo UI.

Stage frames follow the plan: constraints understood, then candidate count,
then validated cards, then reply. catalog_id is withheld until sanitize_picks
has committed the allowlist. That is the design.md streaming policy, not a
UI preference — a mid-stream leak would let the client see an unplayable id.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Annotated, cast

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from assist.api.deps import AppResources, DegradedError, get_server_user_ctx, resources_of
from assist.api.routes_turn import _acquire_rate_limit, _load_session
from assist.api.schemas import ChipOut, PickOut, TurnRequest, turn_response_from_state
from assist.config import settings
from assist.domain.catalog import Candidate, Pick
from assist.domain.constraints import ConstraintState
from assist.domain.context import ServerUserCtx
from assist.domain.enums import DegradedReason
from assist.graph.state import ReplyChip, TurnState, empty_turn_state
from assist.obs.logging import get_logger
from assist.stores.ratelimit import RateLimited
from assist.stores.session import SessionBindError

log = get_logger("assist.api.stream")

stream_router = APIRouter(prefix="/v1/assist", tags=["assist"])
ui_router = APIRouter(tags=["ui"])

# Nodes whose updates may still hold unvalidated catalog rows.
_PRE_SANITIZE_NODES = frozenset(
    {
        "load_session",
        "guard",
        "intent",
        "merge_constraints",
        "resolve_people",
        "broaden_constraints",
        "retrieve",
        "rank",
        "validate_availability",
        "reply_template",
        "reply_generative",
        "reply_clarify",
        "reply_refusal",
    }
)
_CATALOG_ID_KEYS = frozenset({"catalog_id"})
_ALWAYS_REDACT_KEYS = frozenset(
    {
        "person_id",
        "people_include",
        "people_exclude",
        "model_pick_ids",
        "entitled_ids",
        "candidates",
    }
)

_SSE_HEADERS = {
    "Cache-Control": "no-cache, no-store",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _web_dir() -> Path | None:
    here = Path(__file__).resolve()
    candidates = [Path("/app/web"), Path.cwd() / "web"]
    candidates.extend(parent / "web" for parent in here.parents)
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if (path / "index.html").is_file():
            return path
    return None


def install(app: FastAPI) -> None:
    """Mount the SSE route and demo page. One call from create_app."""
    app.include_router(stream_router)
    app.include_router(ui_router)


def _json_default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raw = getattr(value, "value", None)
    if raw is not None and not isinstance(value, (bytes, bytearray)):
        return raw
    raise TypeError(f"unserializable stream value: {type(value)!r}")


def _sse(event: str, data: Mapping[str, object], *, t0: float) -> str:
    payload = dict(data)
    payload["t_ms"] = int((time.perf_counter() - t0) * 1000)
    body = json.dumps(payload, default=_json_default, separators=(",", ":"))
    return f"event: {event}\ndata: {body}\n\n"


def _redact(obj: object, *, drop: frozenset[str]) -> object:
    if isinstance(obj, Mapping):
        out: dict[str, object] = {}
        for key, value in obj.items():
            if str(key) in drop:
                continue
            out[str(key)] = _redact(value, drop=drop)
        return out
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [_redact(item, drop=drop) for item in obj]
    return obj


def _catalog_ids_in(obj: object) -> list[str]:
    found: list[str] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if str(key) in _CATALOG_ID_KEYS and value not in (None, ""):
                found.append(str(value))
            found.extend(_catalog_ids_in(value))
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for item in obj:
            found.extend(_catalog_ids_in(item))
    return found


def _candidates_of(state: Mapping[str, object]) -> tuple[Candidate, ...]:
    raw = state.get("candidates") or ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(item for item in raw if isinstance(item, Candidate))


def _picks_of(state: Mapping[str, object]) -> tuple[Pick, ...]:
    raw = state.get("picks") or ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    out: list[Pick] = []
    for item in raw:
        if isinstance(item, Pick):
            out.append(item)
            continue
        catalog_id = getattr(item, "catalog_id", None)
        reason = getattr(item, "reason_short", "")
        if catalog_id is None and isinstance(item, Mapping):
            catalog_id = item.get("catalog_id")
            reason = item.get("reason_short", "")
        if catalog_id:
            out.append(Pick(catalog_id=str(catalog_id), reason_short=str(reason or "")))
    return tuple(out)


def _chips_of(state: Mapping[str, object]) -> list[ChipOut]:
    raw = state.get("chips") or ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    out: list[ChipOut] = []
    for item in raw:
        if isinstance(item, ReplyChip):
            out.append(ChipOut(id=item.id, label=item.label))
            continue
        chip_id = getattr(item, "id", None)
        label = getattr(item, "label", "")
        if chip_id is None and isinstance(item, Mapping):
            chip_id = item.get("id")
            label = item.get("label", "")
        if chip_id:
            out.append(ChipOut(id=str(chip_id), label=str(label or "")))
    return out


def _constraints_public(raw: object) -> dict[str, object]:
    if isinstance(raw, ConstraintState):
        data = raw.model_dump(mode="json")
    elif isinstance(raw, Mapping):
        data = dict(raw)
    else:
        return {}
    people_include = data.pop("people_include", ()) or ()
    people_exclude = data.pop("people_exclude", ()) or ()
    n_people = len(tuple(people_include)) + len(tuple(people_exclude))
    public: dict[str, object] = {}
    for key in (
        "media_type",
        "genres_include",
        "genres_exclude",
        "moods",
        "year_min",
        "year_max",
        "duration_max_min",
        "origins",
        "local_originals_only",
        "languages",
        "recency_bias",
    ):
        value = data.get(key)
        if value in (None, (), [], "", False):
            continue
        public[key] = value
    if n_people:
        public["people_count"] = n_people
    return public


def _cards_without_ids(candidates: Sequence[Candidate]) -> list[dict[str, object]]:
    cards: list[dict[str, object]] = []
    for candidate in candidates:
        card: dict[str, object] = {"title": candidate.title}
        if candidate.release_year is not None:
            card["release_year"] = candidate.release_year
        if candidate.media_type is not None:
            card["media_type"] = str(candidate.media_type)
        cards.append(card)
    return cards


def _validated_pick_out(
    picks: Sequence[Pick],
    candidates: Sequence[Candidate],
    entitled: set[str],
) -> list[PickOut]:
    """Only ids that sanitize committed *and* availability entitled."""
    by_id = {c.catalog_id: c for c in candidates}
    out: list[PickOut] = []
    seen: set[str] = set()
    for pick in picks:
        catalog_id = pick.catalog_id
        if catalog_id in seen:
            continue
        if entitled and catalog_id not in entitled:
            continue
        if not entitled:
            # Fail closed: no entitled set means we never learnt playable_now.
            continue
        seen.add(catalog_id)
        candidate = by_id.get(catalog_id)
        reason = pick.reason_short
        if not reason and candidate is not None:
            reason = candidate.title
        out.append(PickOut(catalog_id=catalog_id, reason_short=reason))
    return out


def _entitled_of(state: Mapping[str, object]) -> set[str]:
    raw = state.get("entitled_ids") or ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return set()
    return {str(item) for item in raw if item}


def _guard_payload(
    stage: str, payload: dict[str, object], *, allow_catalog_id: bool
) -> dict[str, object]:
    drop = set(_ALWAYS_REDACT_KEYS)
    if not allow_catalog_id:
        drop.update(_CATALOG_ID_KEYS)
    redacted = _redact(payload, drop=frozenset(drop))
    if not isinstance(redacted, dict):
        return {"stage": stage}
    if not allow_catalog_id and _catalog_ids_in(redacted):
        # Last line of defence: drop the frame body rather than leak an id.
        log.warning("stream_redacted_unvalidated_id", stage=stage)
        return {"stage": stage}
    redacted.setdefault("stage", stage)
    return redacted


def _degraded_done(
    state: Mapping[str, object],
    *,
    reason: DegradedReason,
    latency_ms: int,
    trace_id: str,
) -> dict[str, object]:
    merged = dict(state)
    merged["degraded_reason"] = reason
    merged["picks"] = ()
    body = turn_response_from_state(
        cast(TurnState, merged), latency_ms=latency_ms, trace_id=trace_id
    )
    # Even a degraded done must not invent picks from pre-sanitize state.
    dumped = body.model_dump(mode="json")
    dumped["picks"] = []
    dumped["stage"] = "done"
    return dumped


def _final_done(
    state: Mapping[str, object],
    *,
    sanitized: bool,
    latency_ms: int,
    trace_id: str,
) -> dict[str, object]:
    body = turn_response_from_state(
        cast(TurnState, dict(state)), latency_ms=latency_ms, trace_id=trace_id
    )
    dumped = body.model_dump(mode="json")
    if not sanitized:
        dumped["picks"] = []
    else:
        entitled = _entitled_of(state)
        picks = _validated_pick_out(_picks_of(state), _candidates_of(state), entitled)
        dumped["picks"] = [p.model_dump(mode="json") for p in picks]
    dumped["stage"] = "done"
    return dumped


def _frame_for_node(
    node: str,
    accumulated: Mapping[str, object],
    *,
    sanitized: bool,
    seen: set[str],
) -> dict[str, object] | None:
    if node == "merge_constraints":
        return _guard_payload(
            "constraints",
            {"constraints": _constraints_public(accumulated.get("constraints"))},
            allow_catalog_id=False,
        )
    if node == "rank":
        candidates = _candidates_of(accumulated)
        return _guard_payload(
            "candidates",
            {"count": len(candidates)},
            allow_catalog_id=False,
        )
    if node == "validate_availability":
        candidates = _candidates_of(accumulated)
        return _guard_payload(
            "validated",
            {"count": len(candidates), "cards": _cards_without_ids(candidates)},
            allow_catalog_id=False,
        )
    if node == "sanitize_picks" and sanitized:
        entitled = _entitled_of(accumulated)
        picks = _validated_pick_out(_picks_of(accumulated), _candidates_of(accumulated), entitled)
        by_id = {c.catalog_id: c for c in _candidates_of(accumulated)}
        cards: list[dict[str, object]] = []
        for pick in picks:
            card: dict[str, object] = {
                "catalog_id": pick.catalog_id,
                "reason_short": pick.reason_short,
            }
            candidate = by_id.get(pick.catalog_id)
            if candidate is not None:
                card["title"] = candidate.title
                if candidate.release_year is not None:
                    card["release_year"] = candidate.release_year
                card["media_type"] = str(candidate.media_type)
            cards.append(card)
        return _guard_payload("picks", {"picks": cards}, allow_catalog_id=True)
    if node == "mint_chips" or (node == "persist" and "mint_chips" not in seen):
        return _guard_payload(
            "reply",
            {
                "reply": str(accumulated.get("reply") or ""),
                "chips": [c.model_dump(mode="json") for c in _chips_of(accumulated)],
                "session_id": str(accumulated.get("session_id") or ""),
            },
            allow_catalog_id=False,
        )
    return None


def _apply_update(accumulated: dict[str, object], payload: object) -> None:
    if isinstance(payload, Mapping):
        accumulated.update(dict(payload))


async def _updates_from_graph(
    graph: object, state: TurnState
) -> AsyncIterator[Mapping[str, object]]:
    astream = getattr(graph, "astream", None)
    if callable(astream):
        async for update in astream(state, stream_mode="updates"):
            if isinstance(update, Mapping):
                yield update
        return
    if not hasattr(graph, "ainvoke"):
        raise TypeError("graph has no ainvoke")
    result = await graph.ainvoke(state)
    # ainvoke is already post-sanitize. Replay the public frame sequence
    # from the final state so ids still only appear on the picks frame.
    if isinstance(result, Mapping):
        yield {"__final__": dict(result)}


async def _stream_turn(
    resources: AppResources,
    state: TurnState,
    *,
    trace_id: str,
    t0: float,
) -> AsyncIterator[str]:
    accumulated: dict[str, object] = dict(state)
    seen: set[str] = set()
    timeout = resources.hard_timeout_s
    if timeout is None:
        timeout = settings.hard_timeout_ms / 1000.0
    deadline = time.perf_counter() + float(timeout)

    yield _sse("stage", _guard_payload("start", {}, allow_catalog_id=False), t0=t0)

    agen = _updates_from_graph(resources.graph, state)
    try:
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError
            try:
                update = await asyncio.wait_for(agen.__anext__(), timeout=remaining)
            except StopAsyncIteration:
                break
            for node, payload in update.items():
                node_name = str(node)
                if node_name == "__final__":
                    _apply_update(accumulated, payload)
                    seen.update(
                        {
                            "merge_constraints",
                            "rank",
                            "validate_availability",
                            "sanitize_picks",
                            "mint_chips",
                            "persist",
                        }
                    )
                    for replay in (
                        "merge_constraints",
                        "rank",
                        "validate_availability",
                        "sanitize_picks",
                        "mint_chips",
                    ):
                        frame = _frame_for_node(replay, accumulated, sanitized=True, seen=seen)
                        if frame is not None:
                            yield _sse("stage", frame, t0=t0)
                    continue
                seen.add(node_name)
                _apply_update(accumulated, payload)
                sanitized = "sanitize_picks" in seen
                if node_name in _PRE_SANITIZE_NODES and node_name != "sanitize_picks":
                    # Belt: never let a pre-sanitize update pass through raw.
                    if isinstance(payload, Mapping) and _catalog_ids_in(
                        _redact(payload, drop=_ALWAYS_REDACT_KEYS)
                    ):
                        log.info("stream_dropped_pre_sanitize_ids", node=node_name)
                frame = _frame_for_node(node_name, accumulated, sanitized=sanitized, seen=seen)
                if frame is not None:
                    yield _sse("stage", frame, t0=t0)
        resources.stats.turns += 1
        latency_ms = int((time.perf_counter() - t0) * 1000)
        yield _sse(
            "done",
            _final_done(
                accumulated,
                sanitized="sanitize_picks" in seen,
                latency_ms=latency_ms,
                trace_id=trace_id,
            ),
            t0=t0,
        )
    except TimeoutError:
        log.warning("hard_timeout_stream", session_id=state.get("session_id"))
        latency_ms = int((time.perf_counter() - t0) * 1000)
        yield _sse(
            "done",
            _degraded_done(
                accumulated,
                reason=DegradedReason.HARD_TIMEOUT,
                latency_ms=latency_ms,
                trace_id=trace_id,
            ),
            t0=t0,
        )
    except (RateLimited, SessionBindError, DegradedError):
        raise
    except Exception:
        log.exception("stream_graph_failed", session_id=state.get("session_id"))
        latency_ms = int((time.perf_counter() - t0) * 1000)
        yield _sse(
            "done",
            _degraded_done(
                accumulated,
                reason=DegradedReason.RETRIEVAL_UNAVAILABLE,
                latency_ms=latency_ms,
                trace_id=trace_id,
            ),
            t0=t0,
        )
    finally:
        aclose = getattr(agen, "aclose", None)
        if callable(aclose):
            await aclose()


@stream_router.post("/turn/stream")
async def assist_turn_stream(
    body: TurnRequest,
    request: Request,
    ctx: Annotated[ServerUserCtx, Depends(get_server_user_ctx)],
) -> StreamingResponse:
    resources = resources_of(request)
    trace_id = str(getattr(request.state, "trace_id", "-"))
    t0 = float(getattr(request.state, "t0", time.perf_counter()))

    if body.client_hints is not None:
        log.info(
            "client_hints",
            user_id=ctx.user_id,
            profile_id=ctx.profile_id,
            device_class=body.client_hints.device_class,
            ui_language=body.client_hints.ui_language,
            app_version=body.client_hints.app_version,
        )

    await _acquire_rate_limit(resources, ctx)
    session, store_degraded = await _load_session(resources, body, ctx)

    if body.message.type == "chip":
        session.lookup_chip((body.message.chip_id or "").strip())

    # Persist the session id so a chip follow-up can load it. Do not write the
    # pre-turn snapshot back after the graph — that would clobber persist.
    if not store_degraded:
        try:
            await resources.sessions.save(session)
        except SessionBindError:
            raise
        except Exception:
            log.warning("session_save_failed", session_id=session.session_id)
            store_degraded = True

    state = empty_turn_state(
        ctx,
        session_id=session.session_id,
        trace_id=trace_id,
        text=(body.message.text or "").strip(),
        chip_id=body.message.chip_id if body.message.type == "chip" else None,
        message_type=body.message.type,
        turn_count=session.turn_count,
        constraints=session.constraints,
        degraded_reason=(
            DegradedReason.SESSION_STORE_UNAVAILABLE if store_degraded else DegradedReason.NONE
        ),
    )

    async def events() -> AsyncIterator[bytes]:
        async for chunk in _stream_turn(resources, state, trace_id=trace_id, t0=t0):
            yield chunk.encode("utf-8")

    return StreamingResponse(events(), media_type="text/event-stream", headers=_SSE_HEADERS)


def _web_file(name: str) -> Response:
    root = _web_dir()
    if root is None:
        return Response(status_code=404, content=b"demo ui not packaged")
    path = (root / name).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return Response(status_code=404, content=b"not found")
    if not path.is_file():
        return Response(status_code=404, content=b"not found")
    media = {
        "index.html": "text/html; charset=utf-8",
        "app.js": "text/javascript; charset=utf-8",
        "style.css": "text/css; charset=utf-8",
    }.get(name, "application/octet-stream")
    return FileResponse(path, media_type=media)


@ui_router.get("/")
async def demo_index() -> Response:
    return _web_file("index.html")


@ui_router.get("/app.js")
async def demo_js() -> Response:
    return _web_file("app.js")


@ui_router.get("/style.css")
async def demo_css() -> Response:
    return _web_file("style.css")
