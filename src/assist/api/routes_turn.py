"""POST /v1/assist/turn — invoke the compiled graph, return the response contract."""

from __future__ import annotations

import asyncio
import time
from typing import Annotated, cast
from uuid import uuid4

from fastapi import APIRouter, Depends, Request, Response

from assist.api.deps import AppResources, DegradedError, get_server_user_ctx, resources_of
from assist.api.schemas import TurnRequest, turn_response_from_state
from assist.config import settings
from assist.domain.context import ServerUserCtx
from assist.domain.enums import DegradedReason
from assist.graph.state import TurnState, empty_turn_state
from assist.obs.logging import get_logger
from assist.stores.ratelimit import RateLimited
from assist.stores.session import Session, SessionBindError

log = get_logger("assist.api.turn")

router = APIRouter(prefix="/v1/assist", tags=["assist"])


async def _acquire_rate_limit(resources: AppResources, ctx: ServerUserCtx) -> None:
    try:
        await resources.rate_limiter.acquire("user", ctx.user_id)
    except RateLimited:
        raise
    except Exception:
        log.warning("rate_limiter_unavailable", user_id=ctx.user_id)
        raise DegradedError(
            "rate limiter unavailable",
            reason=DegradedReason.SESSION_STORE_UNAVAILABLE,
        ) from None


async def _load_session(
    resources: AppResources, body: TurnRequest, ctx: ServerUserCtx
) -> tuple[Session, bool]:
    session_id = (body.session_id or "").strip() or str(uuid4())
    try:
        session = await resources.sessions.load(session_id, ctx.user_id, ctx.profile_id)
        return session, False
    except SessionBindError:
        raise
    except Exception:
        log.warning("session_store_unavailable", session_id=session_id)
        return Session.create(
            session_id=session_id, user_id=ctx.user_id, profile_id=ctx.profile_id
        ), True


async def _invoke_graph(resources: AppResources, state: TurnState) -> TurnState:
    timeout = resources.hard_timeout_s
    if timeout is None:
        timeout = settings.hard_timeout_ms / 1000.0
    try:
        result = await asyncio.wait_for(resources.graph.ainvoke(state), timeout=timeout)
        return cast(TurnState, result)
    except TimeoutError:
        log.warning("hard_timeout", session_id=state.get("session_id"))
        degraded = dict(state)
        degraded["degraded_reason"] = DegradedReason.HARD_TIMEOUT
        degraded["reply"] = ""
        degraded["picks"] = ()
        degraded["chips"] = ()
        return cast(TurnState, degraded)
    except Exception:
        log.exception("graph_failed", session_id=state.get("session_id"))
        degraded = dict(state)
        degraded["degraded_reason"] = DegradedReason.RETRIEVAL_UNAVAILABLE
        degraded["reply"] = ""
        degraded["picks"] = ()
        degraded["chips"] = ()
        return cast(TurnState, degraded)


@router.post("/turn")
async def assist_turn(
    body: TurnRequest,
    request: Request,
    ctx: Annotated[ServerUserCtx, Depends(get_server_user_ctx)],
) -> Response:
    resources = resources_of(request)
    trace_id = str(getattr(request.state, "trace_id", "-"))
    t0 = float(getattr(request.state, "t0", time.perf_counter()))

    # Hints are logged and then dropped. They never join ctx or the graph state.
    if body.client_hints is not None:
        log.info(
            "client_hints",
            user_id=ctx.user_id,
            profile_id=ctx.profile_id,
            device_class=body.client_hints.device_class,
            ui_language=body.client_hints.ui_language,
            app_version=body.client_hints.app_version,
        )

    idem_header = (request.headers.get("idempotency-key") or "").strip()
    idem_key = f"{ctx.user_id}:{ctx.profile_id}:{idem_header}" if idem_header else None
    if idem_key is not None:
        try:
            cached = await resources.cache.get_idempotent(idem_key)
        except Exception:
            cached = None
            log.warning("idempotency_lookup_failed")
        if cached is not None:
            return Response(content=cached.encode("utf-8"), media_type="application/json")

    await _acquire_rate_limit(resources, ctx)

    session, store_degraded = await _load_session(resources, body, ctx)

    if body.message.type == "chip":
        session.lookup_chip((body.message.chip_id or "").strip())

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

    result = await _invoke_graph(resources, state)

    try:
        await resources.sessions.save(session)
    except SessionBindError:
        raise
    except Exception:
        log.warning("session_save_failed", session_id=session.session_id)
        current = result.get("degraded_reason") or DegradedReason.NONE
        if current in (DegradedReason.NONE, None):
            merged = dict(result)
            merged["degraded_reason"] = DegradedReason.SESSION_STORE_UNAVAILABLE
            result = cast(TurnState, merged)

    resources.stats.turns += 1
    latency_ms = int((time.perf_counter() - t0) * 1000)
    payload = turn_response_from_state(result, latency_ms=latency_ms, trace_id=trace_id)
    raw = payload.model_dump_json()
    if idem_key is not None:
        try:
            await resources.cache.set_idempotent(idem_key, raw)
        except Exception:
            log.warning("idempotency_store_failed")
    return Response(content=raw.encode("utf-8"), media_type="application/json")
