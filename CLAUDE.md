# Working agreement

Grounded conversational search assist for a streaming catalog. FastAPI + LangGraph, Elasticsearch hybrid retrieval, Postgres catalog, Redis session state. Runs entirely in Docker.

## Read before doing anything

| File | What it is |
|---|---|
| `docs/implementation-plan.md` | Architecture. The *what* and *why*. Source of truth. |
| `docs/WORKPLAN.md` | Task DAG, agent protocol, file ownership. The *who does what, when*. |
| `docs/design.md` | Original design document. Normative for the merge algebra, response contract, router table, and `sanitize_picks`. |

If the plan and this file disagree, the plan wins.

## You need a task ID

You are one of several agents working this repo in parallel. **Do not start work without a task ID from `docs/WORKPLAN.md` §5.** If you were spawned without one, take the lowest-numbered `ready` task (all deps merged, no `task/<ID>-*` branch exists), claim it by creating its branch, and say which one you took.

Do only your task. Touch only the files your task **Owns**. If you need to change a file another task owns, stop and report it — do not edit it.

Full protocol — claim, build, verify, ship, report — is `docs/WORKPLAN.md` §2. Definition of done is §3.

## The one architectural rule

This is a **workflow, not an agent**. The model never decides control flow. Four invariants, enforced by `tests/test_graph_shape.py`:

1. Neither model call is given tools.
2. No conditional edge consults a model. `graph/edges.py` does no I/O and imports nothing from `assist.llm`.
3. Exactly one cycle exists — `retrieve → broaden → retrieve` — bounded by a counter.
4. No LangGraph checkpointer. Cross-turn state is ours, in Redis, under our merge algebra.

Adding a model call to an edge, or a checkpointer, fails CI. If a task seems to require it, the task is wrong — report it.

## Non-negotiables

- **No hallucinated titles.** The model selects from a numbered candidate list by index. It never names a title from memory, and never emits a `catalog_id` or `person_id`.
- **No unplayable pick reaches the client.** `playable_now` runs on every response. Failures fail closed. Non-playable candidates are dropped, never substituted.
- **Client input is never authority.** `client_hints` may influence layout and logging only. Geo, package, maturity, kids and device_class come from the server profile. Chips travel as `chip_id`; the delta stays server-side.
- **Degrade, never error.** A turn returns a degraded body, not a 500. Every `DegradedReason` needs a test that reaches it.

## Conventions

- Python 3.12, `uv` for everything (`uv run`, `uv add` — never `pip`).
- Async throughout. No blocking I/O in a node.
- `ruff` for lint and format, `mypy` for types, `pytest` + `pytest-asyncio` for tests.
- Config via `pydantic-settings` in `src/assist/config.py`. New knob → env var → `.env.example`. Nothing hardcoded.
- Structured logging only (`structlog`, JSON). Every line carries `trace_id`. No `print`.
- Comments explain *why*, not *what*. Match the density of the surrounding code.
- No emoji in code, comments, commit messages or log output.

## Commands

```bash
make up            # start the stack
make seed          # fetch + normalize + enrich + index
make lint typecheck test
make eval          # golden set report
make graph         # regenerate docs/graph.mmd
```

## Reporting

When done, report: task ID, what you built, each acceptance criterion with **actual command output** as evidence, anything deliberately left out, and which tasks are now unblocked. Do not claim a criterion passed without showing it.
