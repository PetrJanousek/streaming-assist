"""FastAPI application, lifespan, and DI wiring."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis.asyncio import Redis

from assist.api.deps import AppResources, default_profile_catalog
from assist.api.middleware import TraceTimingMiddleware, install_error_handlers
from assist.api.routes_ops import router as ops_router
from assist.api.routes_stream import install as install_stream
from assist.api.routes_turn import router as turn_router
from assist.config import settings
from assist.graph.build import build_graph
from assist.obs.logging import configure_logging, get_logger
from assist.stores.cache import CacheStore
from assist.stores.ratelimit import RateLimiter
from assist.stores.session import SessionRepository

log = get_logger("assist.main")


def _build_resources(redis: Redis) -> AppResources:
    return AppResources(
        redis=redis,
        cache=CacheStore(redis),
        sessions=SessionRepository(redis),
        rate_limiter=RateLimiter(redis),
        graph=build_graph(),
        profiles=default_profile_catalog(),
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging(settings.log_level)
    owned = False
    if getattr(app.state, "resources", None) is None:
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        app.state.resources = _build_resources(redis)
        owned = True
        log.info("resources_ready", redis_url=settings.redis_url)
    try:
        yield
    finally:
        if owned:
            await app.state.resources.redis.aclose()


def create_app(*, resources: AppResources | None = None) -> FastAPI:
    application = FastAPI(
        title="streaming-assist",
        version="0.1.0",
        lifespan=_lifespan,
    )
    if resources is not None:
        application.state.resources = resources
    application.add_middleware(TraceTimingMiddleware)
    install_error_handlers(application)
    application.include_router(ops_router)
    application.include_router(turn_router)
    install_stream(application)
    return application


app = create_app()
