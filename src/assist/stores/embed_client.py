"""HTTP client for the embedder service. Timeout and retry; never blocks."""

from __future__ import annotations

import asyncio
from typing import Final, Self

import httpx

from assist.config import settings
from assist.obs.logging import get_logger, get_trace_id

EMBED_DIM: Final[int] = 384
# Must match services/embedder/app.py MAX_TEXTS. Larger payloads are split.
SERVICE_MAX_TEXTS: Final[int] = 256
_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({408, 429, 500, 502, 503, 504})


class EmbedderUnavailable(Exception):
    """Embedder timed out, refused, or returned a bad payload."""


class EmbedClient:
    """Async client for `POST /embed`. Failures raise `EmbedderUnavailable`."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_ms: int | None = None,
        retries: int | None = None,
        backoff_s: float = 0.05,
        transport: httpx.AsyncBaseTransport | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = (base_url or settings.embedder_url).rstrip("/")
        timeout_ms = timeout_ms if timeout_ms is not None else settings.embedder_timeout_ms
        self._retries = retries if retries is not None else settings.embedder_retries
        self._backoff_s = backoff_s
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_ms / 1000.0),
            transport=transport,
        )
        self._log = get_logger(__name__)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed `texts` in service-sized chunks. Empty input skips the network."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        for i in range(0, len(texts), SERVICE_MAX_TEXTS):
            chunk = texts[i : i + SERVICE_MAX_TEXTS]
            vectors.extend(await self._embed_chunk(chunk))
        return vectors

    async def _embed_chunk(self, texts: list[str]) -> list[list[float]]:
        payload = {"texts": texts}
        attempts = self._retries + 1
        last_error: str = "embedder failed"
        for attempt in range(attempts):
            try:
                response = await self._client.post(
                    "/embed",
                    json=payload,
                    headers={"X-Request-Id": get_trace_id()},
                )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = str(exc) or exc.__class__.__name__
                self._log.warning(
                    "embedder_transport_error",
                    attempt=attempt + 1,
                    attempts=attempts,
                    error=last_error,
                )
                if attempt + 1 < attempts:
                    await self._sleep(attempt)
                    continue
                raise EmbedderUnavailable(last_error) from exc

            if response.status_code in _RETRYABLE_STATUS:
                last_error = f"embedder HTTP {response.status_code}"
                self._log.warning(
                    "embedder_retryable_status",
                    attempt=attempt + 1,
                    attempts=attempts,
                    status=response.status_code,
                )
                if attempt + 1 < attempts:
                    await self._sleep(attempt)
                    continue
                raise EmbedderUnavailable(last_error)

            if response.status_code >= 400:
                raise EmbedderUnavailable(
                    f"embedder HTTP {response.status_code}: {response.text[:200]}"
                )

            try:
                body: object = response.json()
            except ValueError as exc:
                raise EmbedderUnavailable("embedder returned a malformed body") from exc
            return _parse_vectors(body, len(texts))

        raise EmbedderUnavailable(last_error)

    async def _sleep(self, attempt: int) -> None:
        delay = self._backoff_s * (2**attempt)
        if delay > 0:
            await asyncio.sleep(delay)


def _parse_vectors(body: object, n_texts: int) -> list[list[float]]:
    if not isinstance(body, dict):
        raise EmbedderUnavailable("embedder body is not an object")
    raw = body.get("vectors")
    if not isinstance(raw, list):
        raise EmbedderUnavailable("embedder body missing vectors")
    if len(raw) != n_texts:
        raise EmbedderUnavailable(f"embedder returned {len(raw)} vectors for {n_texts} texts")
    vectors: list[list[float]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, list) or len(item) != EMBED_DIM:
            dim = len(item) if isinstance(item, list) else -1
            raise EmbedderUnavailable(f"embedder vector {i} has dim {dim}, expected {EMBED_DIM}")
        parsed: list[float] = []
        for value in item:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise EmbedderUnavailable(f"embedder vector {i} is not numeric")
            parsed.append(float(value))
        vectors.append(parsed)
    return vectors
