# syntax=docker/dockerfile:1
# Model weights are baked at build time. Runtime is offline (`--network none`).

FROM python:3.12-slim-bookworm AS builder

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# CPU torch first so sentence-transformers does not pull a CUDA wheel.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt \
 && python -c "import torch; assert torch.version.cuda is None, torch.__version__"

ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
ENV EMBED_MODEL=${EMBED_MODEL} \
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

# Save a self-contained folder so runtime loads a path, never the Hub.
RUN python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['EMBED_MODEL']).save('/opt/model')"


FROM python:3.12-slim-bookworm AS runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

COPY --from=builder /usr/local /usr/local
COPY --from=builder --chown=nonroot:nonroot /opt/model /opt/model
COPY --chown=nonroot:nonroot app.py /app/app.py

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    MODEL_PATH=/opt/model \
    EMBED_MODEL=BAAI/bge-small-en-v1.5 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

USER nonroot
WORKDIR /app
EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
