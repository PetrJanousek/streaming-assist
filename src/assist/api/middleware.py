"""Trace binding, request timing, and HTTP error mapping."""

from __future__ import annotations

import math
import time
from collections.abc import Awaitable, Callable
from typing import cast
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from assist.api.deps import DegradedError, Unauthorized, resources_of
from assist.api.schemas import error_envelope
from assist.obs.logging import get_logger, get_trace_id, trace_scope
from assist.stores.ratelimit import RateLimited
from assist.stores.session import ChipInvalid, SessionBindError

log = get_logger("assist.api")

_JSON = "application/json"


def _trace_id_of(request: Request) -> str:
    return str(getattr(request.state, "trace_id", None) or get_trace_id())


def _latency_ms(request: Request) -> int:
    t0 = getattr(request.state, "t0", None)
    if t0 is None:
        return 0
    return int((time.perf_counter() - float(t0)) * 1000)


def envelope_response(
    *,
    status_code: int,
    error_type: str,
    message: str,
    request: Request,
    retry_after_ms: int | None = None,
    session_id: str = "",
    degraded: bool = False,
    degraded_reason: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = error_envelope(
        error_type=error_type,
        message=message,
        trace_id=_trace_id_of(request),
        latency_ms=_latency_ms(request),
        retry_after_ms=retry_after_ms,
        session_id=session_id,
        degraded=degraded,
        degraded_reason=degraded_reason,
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers=headers,
        media_type=_JSON,
    )


class TraceTimingMiddleware(BaseHTTPMiddleware):
    """Bind trace_id for the request and echo timing. Ops routes stay unauthenticated."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        trace_id = request.headers.get("x-request-id") or str(uuid4())
        request.state.trace_id = trace_id
        request.state.t0 = time.perf_counter()
        with trace_scope(trace_id):
            response = await call_next(request)
        latency_ms = _latency_ms(request)
        response.headers["X-Request-Id"] = trace_id
        response.headers["X-Response-Time-Ms"] = str(latency_ms)
        return response


async def _handle_chip_invalid(request: Request, exc: Exception) -> JSONResponse:
    err = cast(ChipInvalid, exc)
    try:
        resources_of(request).stats.chip_invalid += 1
    except Exception:
        pass
    log.info("chip_invalid", chip_id=err.chip_id)
    return envelope_response(
        status_code=400,
        error_type=err.error_type,
        message=str(err),
        request=request,
    )


async def _handle_unauthorized(request: Request, exc: Exception) -> JSONResponse:
    err = cast(Unauthorized, exc)
    try:
        resources_of(request).stats.unauthorized += 1
    except Exception:
        pass
    return envelope_response(
        status_code=401,
        error_type=err.error_type,
        message="invalid or missing bearer token",
        request=request,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def _handle_rate_limited(request: Request, exc: Exception) -> JSONResponse:
    err = cast(RateLimited, exc)
    try:
        resources_of(request).stats.rate_limited += 1
    except Exception:
        pass
    retry_s = max(1, math.ceil(err.retry_after_ms / 1000))
    log.info("rate_limited", retry_after_ms=err.retry_after_ms, subject=err.subject)
    return envelope_response(
        status_code=429,
        error_type=err.error_type,
        message="rate limited",
        request=request,
        retry_after_ms=err.retry_after_ms,
        degraded=True,
        headers={"Retry-After": str(retry_s)},
    )


async def _handle_degraded(request: Request, exc: Exception) -> JSONResponse:
    err = cast(DegradedError, exc)
    try:
        resources_of(request).stats.degraded += 1
    except Exception:
        pass
    reason = err.reason.value if err.reason is not None else None
    log.warning("degraded", reason=reason, message=str(err))
    return envelope_response(
        status_code=503,
        error_type=err.error_type,
        message=str(err),
        request=request,
        degraded=True,
        degraded_reason=reason,
    )


async def _handle_session_bind(request: Request, exc: Exception) -> JSONResponse:
    err = cast(SessionBindError, exc)
    log.warning("session_bind_rejected", session_id=err.session_id)
    return envelope_response(
        status_code=400,
        error_type=err.error_type,
        message=str(err),
        request=request,
        session_id=err.session_id,
    )


async def _handle_validation(request: Request, exc: Exception) -> JSONResponse:
    err = cast(RequestValidationError, exc)
    return envelope_response(
        status_code=400,
        error_type="validation",
        message="; ".join(
            f"{'.'.join(str(loc) for loc in item.get('loc', ()))}: {item.get('msg')}"
            for item in err.errors()
        )
        or "validation error",
        request=request,
    )


async def _handle_http_exception(request: Request, exc: Exception) -> JSONResponse:
    err = cast(StarletteHTTPException, exc)
    return JSONResponse(status_code=err.status_code, content={"detail": err.detail})


async def _handle_unhandled(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled_error", error=str(exc))
    if request.url.path.rstrip("/") == "/v1/assist/turn":
        try:
            resources_of(request).stats.degraded += 1
        except Exception:
            pass
        # A turn never 500s: return the same parseable shape as a 503 degrade.
        return envelope_response(
            status_code=200,
            error_type="degraded",
            message="turn failed",
            request=request,
            degraded=True,
            degraded_reason="retrieval_unavailable",
        )
    return JSONResponse(status_code=500, content={"error": {"type": "internal"}})


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ChipInvalid, _handle_chip_invalid)
    app.add_exception_handler(Unauthorized, _handle_unauthorized)
    app.add_exception_handler(RateLimited, _handle_rate_limited)
    app.add_exception_handler(DegradedError, _handle_degraded)
    app.add_exception_handler(SessionBindError, _handle_session_bind)
    app.add_exception_handler(RequestValidationError, _handle_validation)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(Exception, _handle_unhandled)
