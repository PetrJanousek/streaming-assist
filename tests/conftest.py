"""Shared fixtures. T01 ships an empty suite; later tasks add tests here."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from assist.obs.logging import bind_trace_id, configure_logging, reset_trace_id

configure_logging(level="WARNING")


@pytest.fixture(autouse=True)
def _bind_test_trace_id() -> Iterator[None]:
    token = bind_trace_id("test")
    yield
    reset_trace_id(token)
