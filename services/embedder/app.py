"""CPU embedder. Loads a baked local model; never reaches the network."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/model")
MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = 384
MAX_TEXTS = 256
BATCH_SIZE = 32

_model: Any = None
_encode_lock: asyncio.Lock | None = None


def _configure_logging() -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
        force=True,
    )


def _log(event: str, *, trace_id: str = "-", **fields: object) -> None:
    payload = {"event": event, "trace_id": trace_id, **fields}
    logging.getLogger("embedder").info("%s", json.dumps(payload, separators=(",", ":")))


def _trace_id(request: Request) -> str:
    return request.headers.get("x-request-id") or "-"


def _load_model() -> Any:
    # Import here so `import app` stays cheap for health-check wrappers.
    import torch
    from sentence_transformers import SentenceTransformer

    threads = min(4, os.cpu_count() or 1)
    torch.set_num_threads(threads)
    model = SentenceTransformer(MODEL_PATH, local_files_only=True)
    model.encode(["warmup"], normalize_embeddings=True, show_progress_bar=False)
    return model


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _model, _encode_lock
    _configure_logging()
    _encode_lock = asyncio.Lock()
    t0 = time.perf_counter()
    _model = await asyncio.to_thread(_load_model)
    _log("model_loaded", model=MODEL_NAME, path=MODEL_PATH, ms=round((time.perf_counter() - t0) * 1000, 1))
    yield
    _model = None


app = FastAPI(title="embedder", lifespan=lifespan)


class EmbedRequest(BaseModel):
    texts: list[str] = Field(default_factory=list, max_length=MAX_TEXTS)


class EmbedResponse(BaseModel):
    vectors: list[list[float]]


@app.get("/healthz")
def healthz() -> dict[str, object]:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model": MODEL_NAME, "dim": EMBED_DIM}


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest, request: Request) -> EmbedResponse:
    if _model is None or _encode_lock is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    texts = req.texts
    if not texts:
        return EmbedResponse(vectors=[])
    async with _encode_lock:
        t0 = time.perf_counter()
        vectors = await asyncio.to_thread(_encode, texts)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _log(
        "embed",
        trace_id=_trace_id(request),
        n=len(texts),
        dim=EMBED_DIM,
        ms=round(elapsed_ms, 1),
    )
    return EmbedResponse(vectors=vectors)


def _encode(texts: list[str]) -> list[list[float]]:
    assert _model is not None
    arr = _model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return arr.astype("float32").tolist()
