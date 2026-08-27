"""EmbedClient: request shape, 384-dim check, timeout/retry, no network on empty."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from assist.obs.logging import bind_trace_id, reset_trace_id
from assist.stores.embed_client import (
    EMBED_DIM,
    SERVICE_MAX_TEXTS,
    EmbedClient,
    EmbedderUnavailable,
)

Handler = Callable[[httpx.Request], httpx.Response]


def _vec(fill: float = 0.1) -> list[float]:
    return [fill] * EMBED_DIM


def _ok(texts: list[str], fill: float = 0.1) -> httpx.Response:
    return httpx.Response(200, json={"vectors": [_vec(fill) for _ in texts]})


def _texts(request: httpx.Request) -> list[str]:
    payload = json.loads(request.content)
    raw = payload["texts"]
    assert isinstance(raw, list)
    return [str(item) for item in raw]


def _client(
    handler: Handler,
    *,
    retries: int = 2,
    timeout_ms: int = 200,
    backoff_s: float = 0.0,
) -> EmbedClient:
    return EmbedClient(
        base_url="http://embedder.test",
        timeout_ms=timeout_ms,
        retries=retries,
        backoff_s=backoff_s,
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_empty_texts_skips_http() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("empty embed must not hit the network")

    async with _client(handler) as client:
        assert await client.embed([]) == []


@pytest.mark.asyncio
async def test_embed_posts_texts_and_returns_384d() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = request.read()
        assert request.method == "POST"
        assert request.url.path == "/embed"
        assert b'"texts"' in body
        return _ok(["a", "b"], fill=0.25)

    async with _client(handler) as client:
        vectors = await client.embed(["a", "b"])

    assert len(vectors) == 2
    assert all(len(v) == EMBED_DIM for v in vectors)
    assert vectors[0][0] == 0.25
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_identical_input_returns_identical_vectors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok(_texts(request), fill=0.42)

    async with _client(handler) as client:
        first = await client.embed(["same sentence"])
        second = await client.embed(["same sentence"])
    assert first == second


@pytest.mark.asyncio
async def test_wrong_dimension_is_rejected() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"vectors": [[0.1, 0.2]]})

    async with _client(handler) as client:
        with pytest.raises(EmbedderUnavailable, match="dim 2"):
            await client.embed(["a"])


@pytest.mark.asyncio
async def test_count_mismatch_is_rejected() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"vectors": [_vec(), _vec()]})

    async with _client(handler) as client:
        with pytest.raises(EmbedderUnavailable, match="2 vectors for 1 texts"):
            await client.embed(["a"])


@pytest.mark.asyncio
async def test_malformed_body_is_rejected() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json", headers={"content-type": "text/plain"})

    async with _client(handler) as client:
        with pytest.raises(EmbedderUnavailable, match="malformed"):
            await client.embed(["a"])


@pytest.mark.asyncio
async def test_connect_error_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("down", request=request)
        return _ok(["a"])

    async with _client(handler, retries=2) as client:
        vectors = await client.embed(["a"])
    assert len(vectors) == 1
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_connect_error_exhausted_raises() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("down", request=request)

    async with _client(handler, retries=2) as client:
        with pytest.raises(EmbedderUnavailable, match="down"):
            await client.embed(["a"])
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_timeout_retries_then_raises() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("slow", request=request)

    async with _client(handler, retries=1) as client:
        with pytest.raises(EmbedderUnavailable):
            await client.embed(["a"])
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_http_500_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, text="boom")
        return _ok(["a"])

    async with _client(handler, retries=2) as client:
        vectors = await client.embed(["a"])
    assert len(vectors) == 1
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_http_400_does_not_retry() -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad")

    async with _client(handler, retries=3) as client:
        with pytest.raises(EmbedderUnavailable, match="HTTP 400"):
            await client.embed(["a"])
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_sends_trace_id_header() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("x-request-id", ""))
        return _ok(["a"])

    token = bind_trace_id("trace-t07")
    try:
        async with _client(handler) as client:
            await client.embed(["a"])
    finally:
        reset_trace_id(token)
    assert seen == ["trace-t07"]


@pytest.mark.asyncio
async def test_oversize_payload_is_chunked() -> None:
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts = _texts(request)
        sizes.append(len(texts))
        return _ok(texts)

    n = SERVICE_MAX_TEXTS + 3
    async with _client(handler) as client:
        vectors = await client.embed([f"t{i}" for i in range(n)])
    assert sizes == [SERVICE_MAX_TEXTS, 3]
    assert len(vectors) == n
