"""JSON structured logging. Every line carries `trace_id` from a contextvar."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import cast

import structlog
from structlog.types import EventDict, WrappedLogger

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


def get_trace_id() -> str:
    return trace_id_var.get()


def bind_trace_id(trace_id: str) -> Token[str]:
    token = trace_id_var.set(trace_id)
    structlog.contextvars.bind_contextvars(trace_id=trace_id)
    return token


def reset_trace_id(token: Token[str]) -> None:
    trace_id_var.reset(token)
    structlog.contextvars.bind_contextvars(trace_id=trace_id_var.get())


@contextmanager
def trace_scope(trace_id: str) -> Iterator[None]:
    token = bind_trace_id(trace_id)
    try:
        yield
    finally:
        reset_trace_id(token)


def _inject_trace_id(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    event_dict.setdefault("trace_id", trace_id_var.get())
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
        force=True,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _inject_trace_id,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    # structlog.get_logger is typed as Any; wrapper_class is BoundLogger above.
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
