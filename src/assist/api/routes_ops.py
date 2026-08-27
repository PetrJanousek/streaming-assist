"""Ops routes: liveness, readiness, stats, and the dev profile fixture set."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from assist.api.deps import resources_of
from assist.obs.logging import get_logger

log = get_logger("assist.api.ops")

router = APIRouter(tags=["ops"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Process liveness. Does not consult Redis."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    """Fail when Redis is down. /healthz stays up so the orchestrator can see us."""
    try:
        pong = await resources_of(request).redis.ping()
        if pong is False:
            raise RuntimeError("redis ping returned false")
    except Exception:
        log.warning("readyz_redis_down")
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "redis": "down"},
        )
    return JSONResponse(status_code=200, content={"status": "ok", "redis": "ok"})


@router.get("/stats")
async def stats(request: Request) -> dict[str, int | float]:
    resources = resources_of(request)
    snapshot = resources.stats
    return {
        "turns": snapshot.turns,
        "rate_limited": snapshot.rate_limited,
        "unauthorized": snapshot.unauthorized,
        "chip_invalid": snapshot.chip_invalid,
        "degraded": snapshot.degraded,
        "uptime_s": round(time.monotonic() - snapshot.started_monotonic, 3),
    }


@router.get("/dev/profiles")
async def dev_profiles(request: Request) -> dict[str, object]:
    profiles = [
        p.model_dump(mode="json", exclude={"device_bound"})
        for p in resources_of(request).profiles.list_all()
        if p.device_bound
    ]
    return {"profiles": profiles}
