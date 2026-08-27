# streaming-assist

Grounded conversational search assist for a streaming catalog — a conversational layer inside an
existing search bar that turns vague natural-language intent into catalog-grounded title picks,
a short reply, and tappable refinement chips.

The model extracts intent and writes one sentence. It never picks the catalog, never names a title,
and never decides control flow. Retrieval, ranking, availability and constraint state are all
deterministic.

**Status:** scaffolding. Implementation has not started.

## Documentation

| File | What it is |
|---|---|
| [`docs/implementation-plan.md`](docs/implementation-plan.md) | Architecture, technology decisions, build order |
| [`docs/WORKPLAN.md`](docs/WORKPLAN.md) | Task DAG for implementation agents |
| [`docs/design.md`](docs/design.md) | Original design document |
| [`CLAUDE.md`](CLAUDE.md) | Working agreement for agents |

## Stack

Python 3.12 · FastAPI · LangGraph · LangChain + Claude Haiku 4.5 · Elasticsearch (BM25 + kNN, RRF
fused in Python) · Postgres · Redis · `bge-small-en-v1.5` embeddings. Everything runs in Docker;
nothing is installed on the host.

Quickstart lands with T27.
