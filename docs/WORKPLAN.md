# Work Plan — task DAG for implementation agents

Companion to `docs/implementation-plan.md` (the *what* and *why*). This file is the *who does what, when* — a dependency graph of independently mergeable tasks.

**Read `docs/implementation-plan.md` first.** It is the source of truth for architecture. If this file and the plan disagree, the plan wins and you should fix this file in your PR.

---

## 1. How to use this file

- **If you are spawning agents:** look at §4. Give each agent exactly one task ID. Only hand out tasks whose dependencies are all `merged`.
- **If you are an agent:** you were given a task ID. Find its card in §5. Follow the protocol in §2. Do **only** that task.

**Do not start work without a task ID.** If you were spawned without one, pick the lowest-numbered task that is `ready` (all deps merged, no existing `task/<ID>-*` branch), claim it per §2.1, and say which one you took.

---

## 2. Agent protocol

### 2.1 Claim

A task is claimed by the **existence of its branch**. No lock files, no coordination server.

```bash
git fetch --all --prune 2>/dev/null || true
git branch -a --list '*task/T04-*'     # non-empty → someone owns it, pick another
git worktree add ../wt-T04 -b task/T04-postgres main
cd ../wt-T04
```

Racing two agents onto one task is possible but cheap to detect: whoever merges second will see the work already present and should close their branch rather than force it.

### 2.2 Build

- Implement **only** the files in the task's **Owns** list. If you need to change a file another task owns, stop and say so in your report instead of editing it.
- Adding a dependency to `pyproject.toml` is always allowed (union-merge; conflicts here are trivial).
- Write the tests named in the task's acceptance criteria. Tests are part of the task, not a follow-up.

### 2.3 Verify

```bash
make lint typecheck test
```

All three must be clean. A red suite is not a mergeable PR.

### 2.4 Ship

Tick your task's checkbox in this file (§5) **inside the same PR**. That one-line edit is the only shared-file write, and conflicts on it resolve trivially.

**Mode L — local fallback (only if `git remote -v` is empty):**

```bash
git add -A && git commit -m "T04: postgres schema, migrations and repositories"
git checkout main && git merge --no-ff task/T04-postgres -m "$(cat <<'EOF'
T04: postgres schema, migrations and repositories

<what changed, 3-6 bullets>
<acceptance criteria, each with evidence>
EOF
)"
git worktree remove ../wt-T04 && git branch -d task/T04-postgres
```

**Mode G — GitHub (THE DEFAULT — `origin` is `PetrJanousek/streaming-assist`, private):**

```bash
git push -u origin task/T04-postgres
gh pr create --title "T04: postgres schema, migrations and repositories" --body "..."
gh pr merge --squash --delete-branch
```

PR body must contain: what changed (3-6 bullets), each acceptance criterion with the actual
command output that proves it, and anything deliberately left out. Squash-merge keeps `main`
one commit per task.

Do not merge a PR whose `make lint typecheck test` output is not in the body. Do not add or
change remotes.

### 2.5 Report

State: task ID, what you built, acceptance criteria with evidence (actual command output, not a claim that it passed), anything you deliberately left out, and which tasks are unblocked now.

---

## 3. Definition of done — every task

1. `make lint typecheck test` clean.
2. New behaviour has tests. Nodes get at least one test with fakes for their I/O.
3. No `TODO`/`FIXME` without an owning task ID in the comment.
4. Public functions have type hints. `Any` needs a reason.
5. Nothing hardcoded that `config.py` should own — new knobs get an env var **and** a line in `.env.example`.
6. Your checkbox in §5 is ticked in the same PR.
7. `README.md` updated if you changed how the thing is run or seeded.

**Never commit:** secrets, `.env`, `data/raw/`, model weights, `uv.lock` conflicts left unresolved.

---

## 4. Dependency graph and parallelism

```
                              T01 scaffold
                                   │
                    ┌──────────────┴──────────────┐
                 T02 domain                   T03 docker
                    │                             │
    ┌────────┬──────┼───────┬─────────┬───────────┴──┐
  T04 pg  T05 redis │  T06 es    T07 embedder   T08 llm-gw
    │        │      │     │           │              │
    │        │   T09 graph skeleton ──┴──────────────┤
    │        │      │                                │
    ├────────┴──────┼────────┬─────────┬─────────────┤
 T10 data-job    T13 api  T14 guard  T15 intent  T16 merge
    │                                                │
    ├──────────────┬─────────────┐                   │
 T11 enrich    T12 index     T20 avail   T21 sanitize/chips/persist
                   │             │             │
              ┌────┴────┐        │             │
          T17 retrieve T18 people│             │
               │                 │             │
           T19 rank ─────────────┴─────────────┘
               │
        T22 router + template reply
               │
        T23 generative reply
               │
        T24 full graph wiring + e2e
               │
      ┌────────┼────────┐
  T25 UI   T26 eval   T27 hardening
```

### Useful agent count per stage

| Stage | Tasks that can run concurrently | Agents worth spawning |
|---|---|---|
| 1 | T01 | **1** — everything blocks on it |
| 2 | T02, T03 | 2 |
| 3 | T04, T05, T06, T07, T08, T09 | **6** — widest point |
| 4 | T10, T13, T14, T15, T16 | 5 |
| 5 | T11, T12, T20, T21 | 4 |
| 6 | T17, T18 | 2 |
| 7 | T19 | 1 |
| 8 | T22 → T23 → T24 | **1** — strictly sequential assembly |
| 9 | T25, T26, T27 | 3 |

Spawning more than the right-hand column at any stage just produces agents that wait or collide. Stages 1 and 8 are genuinely serial — that is the shape of the work, not a planning artifact.

---

## 5. Tasks

Legend: `[ ]` not started · `[~]` branch exists · `[x]` merged to main

---

### [x] T01 — Project scaffold
**Deps:** none · **Branch:** `task/T01-scaffold`

**Owns:** `pyproject.toml`, `Makefile`, `.env.example`, `.gitignore`, `ruff.toml`, `mypy.ini`, `src/assist/__init__.py`, `src/assist/config.py`, `src/assist/obs/logging.py`, `tests/conftest.py`

**Do:**
- Dependency groups in `pyproject.toml`: runtime (fastapi, uvicorn, pydantic, pydantic-settings, langgraph, langchain, langchain-anthropic, redis, sqlalchemy[asyncio], asyncpg, alembic, elasticsearch, httpx, structlog, typer) and dev (pytest, pytest-asyncio, ruff, mypy, testcontainers).
- `config.py` with pydantic-settings covering every env var in implementation-plan §8. Fail loudly on malformed values, never silently default a typo.
- `obs/logging.py`: structlog JSON renderer, `trace_id` bound per request via contextvar.
- `Makefile`: `lint typecheck test fmt` (docker targets come in T03).

**Acceptance:**
- `uv sync` succeeds from clean.
- `make lint typecheck test` clean with an empty suite.
- `python -c "from assist.config import settings; print(settings.model_dump())"` prints defaults with no env set.

---

### [x] T02 — Domain layer
**Deps:** T01 · **Branch:** `task/T02-domain`

**Owns:** `src/assist/domain/*`, `tests/test_merge_algebra.py`, `tests/test_sanitize_picks.py`

**Do:**
- `enums.py`: `GenreId` (the ~20 canonical genres — see plan §3.1), `MoodId` (~16), `Audience`, `Pace`, `SpeechAct`, `DegradedReason`, `Route`, maturity ladder.
- `constraints.py`: `FieldOp` union (`set|add|remove|clear|replace`), `ConstraintDelta`, `ConstraintState`, and `merge(state, delta, source) -> ConstraintState` implementing the precedence and per-field policy in design.md § "Merge algebra".
- `context.py`: `ServerUserCtx` — frozen.
- `catalog.py`: `Title`, `Person`, `Candidate`, `Pick`.
- `picks.py`: `sanitize_picks(...)` with the exact signature in design.md, plus the `min_picks` policy matrix.

**Acceptance:**
- **≥20 merge golden cases**, including: the worked example in design.md (5 steps), chip-overrides-text precedence, `reset_soft` clearing soft fields only, and an attempt to raise `maturity_max` above the profile ceiling that must be a **no-op**.
- `sanitize_picks` tests cover: model returns an ID not in candidates (dropped), an unentitled ID (dropped), `min_picks=0` never pads, `min_picks=3` pads in rank order, empty entitled set returns empty.
- `merge` is pure — a test asserts the input state is not mutated.

---

### [x] T03 — Docker + compose stack
**Deps:** T01 · **Branch:** `task/T03-docker`

**Owns:** `docker-compose.yml`, `docker-compose.scale.yml`, `docker/api.Dockerfile`, `docker/jobs.Dockerfile`, `.dockerignore`, Makefile docker targets

**Do:**
- Compose with all six services per plan §1, including the `embedder` service entry pointing at `services/embedder/Dockerfile` (**T07 creates that file — the service will not build until T07 lands; that is expected**).
- Healthchecks on every service; `depends_on: {condition: service_healthy}`.
- ES: `discovery.type=single-node`, `xpack.security.enabled=false`, `ES_JAVA_OPTS=-Xms512m -Xmx512m`. Redis: `--maxmemory 256mb --maxmemory-policy allkeys-lru`. Named volumes for pg/es/redis.
- Profiles: default, `tools`, `scale`. `scale` overlay adds nginx in front of N api replicas.
- Multi-stage api image using uv, non-root user.
- Makefile: `up down logs ps shell seed`.

**Acceptance:**
- `docker compose up -d` on a clean machine → `postgres`, `elasticsearch`, `redis` all healthy (embedder/api may fail until T07/T13 — note this in the PR).
- `docker compose config` validates.
- Total stack RSS under ~2.5GB measured with `docker stats`.

---

### [x] T04 — Postgres: schema, migrations, repositories
**Deps:** T01, T02 · **Branch:** `task/T04-postgres`

**Owns:** `src/assist/stores/db.py`, `migrations/*`, `alembic.ini`, `tests/test_db.py`

**Do:** async SQLAlchemy models + alembic migration for every table in plan §4.1. Repositories for titles, people, credits, availability, taxonomy, phrase_bank, profiles, golden_queries, turn_events. Connection pool sized from config. `turn_events` writes are fire-and-forget and must never fail a request.

**Acceptance:** `alembic upgrade head` on empty DB creates all tables; round-trip test per repository against a testcontainer Postgres; a deliberately failing `turn_events` insert does not raise to the caller.

---

### [x] T05 — Redis: session, chips, caches, rate limiter
**Deps:** T01, T02 · **Branch:** `task/T05-redis`

**Owns:** `src/assist/stores/session.py`, `cache.py`, `ratelimit.py`, `tests/test_session.py`, `tests/test_ratelimit.py`

**Do:** Session repository (load/save, 24h sliding TTL, `ConstraintState` + last 6 turns + `issued_chips` + `turn_count`), chip mint/lookup **inside the session object**, the three caches (intent 1h, response 5m, availability 45s) and idempotency (5m) per plan §4.3, and a Redis token-bucket rate limiter as a **Lua script** (atomic, correct across replicas).

**Acceptance:** cross-profile session bind is rejected; unknown/expired `chip_id` raises the `chip_invalid` error type; turn history caps at 6; rate limiter test proves atomicity under 50 concurrent calls; TTLs asserted via `PTTL`.

---

### [x] T06 — Elasticsearch client, mappings, index bootstrap
**Deps:** T01, T02 · **Branch:** `task/T06-es`

**Owns:** `src/assist/stores/es.py`, `src/assist/stores/mappings/*`, `tests/test_es.py`

**Do:** async ES client from config. `titles_v1` and `people_v1` mappings per plan §4.2 (english analyzer + synonym pack, `dense_vector` 384 cosine HNSW, edge-ngram on person names). Versioned-index creation with **atomic alias swap** and rollback. Query *builders* only — the actual hybrid query lives in T17.

**Acceptance:** bootstrap creates indices and points aliases; running it twice is idempotent; alias swap is atomic (test asserts the alias never resolves to zero indices mid-swap).

---

### [x] T07 — Embedder service
**Deps:** T01 · **Branch:** `task/T07-embedder`

**Owns:** `services/embedder/*` (including its `Dockerfile`), `src/assist/stores/embed_client.py`, `tests/test_embed_client.py`

**Do:** Tiny FastAPI service wrapping `BAAI/bge-small-en-v1.5` via sentence-transformers. `POST /embed {texts: [...]} -> {vectors: [[384 floats]]}`, batching, `/healthz`. **Model weights baked into the image at build time** — the running container must never reach the network. Client in `assist` with timeout and retry.

**Acceptance:** `docker build` then `docker run --network none` still serves `/embed`; 384-dim output; identical input → identical vector; batch of 64 under 500ms on CPU.

---

### [x] T08 — LLM gateway
**Deps:** T01, T02 · **Branch:** `task/T08-llm-gateway`

**Owns:** `src/assist/llm/gateway.py`, `cost.py`, `llm/prompts/*`, `tests/test_gateway.py`

**Do:** Provider factory (`LLM_PROVIDER`, default anthropic / `claude-haiku-4-5`) returning a configured `ChatAnthropic`. Hard timeout from `LLM_TIMEOUT_MS`. `.with_structured_output(...)` helper, `.with_retry()`/`.with_fallbacks()` wiring so schema failure degrades rather than raises. `BaseCallbackHandler` accumulating tokens + USD per turn. **`LLM_PROVIDER=none` must return a stub that raises a typed `LLMUnavailable`** so the whole system runs with no API key.

**Acceptance:** with no `ANTHROPIC_API_KEY`, importing and calling the gateway raises `LLMUnavailable` and never a network error; cost callback numbers match a hand-computed figure for a known token count; timeout is enforced (test with a sleeping fake).

---

### [x] T09 — Graph skeleton
**Deps:** T01, T02 · **Branch:** `task/T09-graph`

**Owns:** `src/assist/graph/state.py`, `edges.py`, `build.py`, `src/assist/nodes/__init__.py`, `tests/test_graph_shape.py`

**Do:** `TurnState` TypedDict (immutable `ServerUserCtx` input, constraints, candidates, picks, chips, route, degraded_reason, retrieve_attempts, timings). `build_graph()` assembling a **stub** node per stage that passes state through, with the real conditional edges and the bounded `retrieve → broaden → retrieve` cycle already wired. Mermaid export helper. No checkpointer.

**Acceptance:**
- Graph compiles; a stub turn runs end to end.
- `test_graph_shape.py` asserts: `graph/edges.py` imports nothing from `assist.llm`; the only cycle is retrieve↔broaden; `retrieve_attempts` cap is enforced; no checkpointer is configured.
- `make graph` writes a Mermaid diagram to `docs/graph.mmd`.

---

### [x] T10 — Data job: fetch + normalize
**Deps:** T02, T04 · **Branch:** `task/T10-data-job`

**Owns:** `src/assist/jobs/cli.py`, `fetch.py`, `normalize.py`, `data/taxonomy/*`, `tests/test_normalize.py`

**Do:** `fetch` downloads the dataset (URL in plan §3.1) with checksum verification, falling back to a committed 500-row sample. `normalize` loads CSV → Postgres: 42 raw genre labels → ~20 `GenreId` via a **committed mapping file**, maturity ladder, `duration` → runtime/seasons, comma-split people → `people` + `credits` with `active_year_min/max`, deterministic synthesized availability windows and `pop_28d` seeded by `catalog_id`, `local_original` from `HOME_COUNTRY` + deterministic 20%.

**Acceptance:** `jobs normalize` loads 8,807 titles and ~41k people; the 4 dirty rating rows are quarantined not crashed; re-running is idempotent; synthesized availability yields ~85% playable (asserted within a tolerance band); a fixed `catalog_id` produces the same availability and popularity across runs.

---

### [x] T11 — Enrichment job
**Deps:** T08, T10 · **Branch:** `task/T11-enrich`

**Owns:** `src/assist/jobs/enrich.py`, `data/enriched/*`, `tests/test_enrich.py`

**Do:** One structured Haiku call per title → the `Enrichment` schema in plan §3.3, written to `titles.enrichment` jsonb. Concurrency semaphore of 8, **resumable** (skip already-enriched), `--limit N` default 2500, `--dry-run` cost estimate. Export/import a committed JSONL artifact so a fresh clone seeds with zero API spend.

**Acceptance:** killing mid-run and re-running loses no work and re-does nothing; a title whose enrichment call fails still ends up indexed with empty moods; `--dry-run` prints a token and USD estimate; the committed artifact imports without an API key.

---

### [x] T12 — Index job
**Deps:** T06, T07, T10 · **Branch:** `task/T12-index`

**Owns:** `src/assist/jobs/index.py`, `tests/test_index.py`

**Do:** Read titles from Postgres, build embedding text (title + synopsis + tags + people + era_feel), call the embedder in batches, bulk-index into a fresh `titles_vN`, build `people_vN`, then swap aliases. Progress logging, resumable.

**Acceptance:** after `jobs index`, ES doc count matches the Postgres title count; a known title is retrievable by both BM25 and kNN; alias swap leaves the previous index intact for rollback; re-running produces a new version and swaps cleanly.

---

### [x] T13 — API surface + middleware
**Deps:** T05, T09 · **Branch:** `task/T13-api`

**Owns:** `src/assist/main.py`, `src/assist/api/*`, `tests/test_api.py`, `tests/test_authz_invariants.py`

**Do:** `POST /v1/assist/turn` per design.md § "API / Interface Changes" — invokes the compiled graph, returns `{session_id, reply, picks, chips, meta}`. Bearer token → seeded profile → `ServerUserCtx` (`deps.py`). Ops routes `/healthz /readyz /stats /dev/profiles`. Middleware: rate limit (T05), `Idempotency-Key`, request timing, trace_id binding. Error mapping: 400 `chip_invalid` / 401 / 429 `rate_limited` / 503 `degraded`.

**Acceptance:**
- `test_authz_invariants.py` proves `client_hints.device_class`, geo, package, maturity and kids sent by the client **never** reach `ServerUserCtx` or `playable_now` — assert on the constructed context, not just the response.
- Same `Idempotency-Key` twice returns a byte-identical body without re-running the graph.
- 429 body still carries a usable shape.
- `/readyz` fails when Redis is down; `/healthz` does not.

---

### [x] T14 — Guard node
**Deps:** T09 · **Branch:** `task/T14-guard`

**Owns:** `src/assist/nodes/guard.py`, `data/guard/*`, `tests/test_guard.py`

**Do:** Rules + heuristics: prompt-injection markers, jailbreak phrasings, adult/piracy/competitor terms, absurd length, control characters. Fails **closed** to a refusal route. Runs before any model sees the text.

**Acceptance:** a committed adversarial corpus (≥40 strings) is blocked; a benign corpus (≥40) passes; blocked turns produce `route=safety`, `min_picks=0`, and reach no model call (asserted with a fake gateway that raises if called).

---

### [x] T15 — Intent node
**Deps:** T05, T08, T09 · **Branch:** `task/T15-intent`

**Owns:** `src/assist/nodes/intent.py`, `llm/prompts/intent.md`, `tests/test_intent.py`

**Do:** Three sources converging on one `ConstraintDelta`: chip → server-held delta lookup (no NLU); rules → closed-class matcher (bare genre, decade, duration, media type, known title); else one structured LLM call returning `IntentUpdate` (design.md Appendix A). Intent cache checked before the model. `intent_source` recorded on state.

**Acceptance:** chip path makes zero model calls; rules path makes zero model calls; a cache hit makes zero model calls; a malformed model response degrades to rules rather than raising; `person_ids_from_index` from the model is **ignored** (person IDs only ever come from the index — assert this).

---

### [x] T16 — Merge node
**Deps:** T02, T09 · **Branch:** `task/T16-merge`

**Owns:** `src/assist/nodes/merge.py`, `tests/test_merge_node.py`

**Do:** Thin node applying `domain.constraints.merge` with the right precedence and writing the new state. Effective maturity is `min(profile.maturity_max, requested_stricter)` and is computed here, not trusted from the delta.

**Acceptance:** multi-turn sequence test — constraints from turn 1 survive turn 3; a chip delta beats a conflicting text delta in the same turn; hard AuthZ fields are unchanged by every delta shape in the corpus.

---

### [x] T17 — Retrieval node
**Deps:** T06, T09, T12 · **Branch:** `task/T17-retrieval`

**Owns:** `src/assist/nodes/retrieval.py`, `tests/test_retrieval.py`

**Do:** Concurrent BM25 + kNN against the same filtered set (constraints as `filter` clauses, never `must`), optional people→titles join, **RRF fusion in Python** (`k=RRF_K`), franchise/series diversification, emit 20–25 compact candidate cards. Deterministic broaden ladder for the zero-hit retry, bounded by `RETRIEVE_MAX_ATTEMPTS`.

**Acceptance:** filters never alter BM25 scoring (asserted); RRF output matches a hand-computed fusion on a fixed fixture; zero hits triggers exactly one broaden retry then stops; franchise cap holds; a maturity-restricted profile can never see an over-rated title.

---

### [x] T18 — People resolver node
**Deps:** T06, T09, T12 · **Branch:** `task/T18-people`

**Owns:** `src/assist/nodes/people.py`, `data/aliases/*`, `tests/test_people.py`

**Do:** Resolve `person_soft` descriptors (role + era + popularity) and name mentions against `people_v1`. θ_person = 0.75: single high-confidence → `people_include`; 2–3 close → **clarify chips, no guess**, `min_picks=0`; zero → era+genre fallback. Person IDs come only from the index.

**Acceptance:** a golden slice of ≥15 person-fuzzy queries reports person@1; the ambiguous case produces clarify chips and empty picks rather than a wrong confident pick; no code path can construct a `person_id` that is not in the index.

---

### [x] T19 — Rank node
**Deps:** T09, T17 · **Branch:** `task/T19-rank`

**Owns:** `src/assist/nodes/rank.py`, `tests/test_rank.py`

**Do:** `0.50·pop_norm + 0.30·constraint_match + 0.20·semantic_norm` with weights from config; min-max normalisation **per candidate set**; cold-start fallback to editorial/global-median prior; greedy franchise cap of 1 per top-8; `semantic_norm = 0` when the vector path is off.

**Acceptance:** weights sum-check; a fixed candidate fixture produces a stable documented ordering; disabling the vector path changes ranking but never crashes; ties broken deterministically.

---

### [x] T20 — Availability node + CatalogClient
**Deps:** T04, T05, T09 · **Branch:** `task/T20-availability`

**Owns:** `src/assist/nodes/availability.py`, `src/assist/stores/catalog_client.py`, `tests/test_availability.py`

**Do:** `playable_now(ServerUserCtx)` batched over candidates, Redis-cached 45s, behind a `CatalogClient` interface (the seam for a future separate service). Non-playable candidates are **dropped, never substituted**. Client failure fails closed to "not playable" — never fail open.

**Acceptance:** a non-playable title never survives; cache hit avoids the DB (asserted with a counting fake); a `CatalogClient` raising an exception drops the candidate rather than admitting it; device_class used is the server-bound one.

---

### [x] T21 — Sanitize, chips, persist nodes
**Deps:** T02, T04, T05, T09 · **Branch:** `task/T21-sanitize-chips-persist`

**Owns:** `src/assist/nodes/sanitize.py`, `chips.py`, `persist.py`, `tests/test_chips.py`, `tests/test_persist.py`

**Do:** `sanitize` wraps `domain.picks.sanitize_picks` and adds the reply-prose title-span check (strip an off-catalog title span, or fall back to template). `chips` mints `chip_id`s with server-held deltas from the phrase bank for a known `SpeechAct` only. `persist` writes the session back and fires an `AssistTurnEvent`.

**Acceptance:** a reply naming a title not in candidates is stripped or replaced (test both branches); a chip whose `speech_act` is unknown is refused at mint time; the client-facing chip payload contains **only** `id` and `label` — asserted on the serialized response; turn event carries route, intent_source, degraded_reason, per-stage latency, tokens and cost.

---

### [x] T22 — Router edge + template reply
**Deps:** T19, T20, T21 · **Branch:** `task/T22-router-template`

**Owns:** `src/assist/graph/edges.py` (route predicate), `src/assist/nodes/templates.py`, `data/phrases/*`, `tests/test_router.py`

**Do:** Route predicate per design.md "Router v0" — chip/safety/rules/high-confidence → TEMPLATE, person-ambiguous → clarify, else GENERATIVE. Thresholds `ROUTER_THETA1`/`ROUTER_THETA_GAP` from config, `top1` and `gap` logged every turn. Phrase bank of template replies and chip labels covering every `SpeechAct`.

**Acceptance:** the routing table in design.md is covered case by case; the predicate is pure (no I/O, no model — enforced by `test_graph_shape.py`); every `SpeechAct` has at least one phrase; **this is the first end-to-end turn with zero LLM calls** — demonstrate it in the PR with real output.

---

### [x] T23 — Generative reply node
**Deps:** T08, T22 · **Branch:** `task/T23-generative-reply`

**Owns:** `src/assist/nodes/reply.py`, `llm/prompts/reply.md`, `tests/test_reply.py`

**Do:** One structured call over 20–25 **numbered** candidate cards returning `{reply, pick_indices, chip_speech_acts}`. The model returns **indices, never IDs or titles**. Chip labels come from the phrase bank. **Zero retries** — schema failure routes to template with `degraded_reason=generative_schema_fail`.

**Acceptance:** an out-of-range index is dropped by sanitize, not honoured; a schema failure produces a template reply plus ranker picks with the right `degraded_reason`; reply length cap enforced; a model reply naming an off-catalog title is caught by T21's span check.

---

### [x] T24 — Full graph wiring + e2e
**Deps:** T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23 · **Branch:** `task/T24-wiring`

**Owns:** `src/assist/graph/build.py` (real nodes replacing stubs), `tests/test_pipeline_e2e.py`

**Do:** Replace every stub with the real node. Hard timeout (`HARD_TIMEOUT_MS`) with a degraded response rather than a 500. Verify every `DegradedReason` is reachable.

**Acceptance:** multi-turn conversation against the real stack (fake LLM) produces sticky constraints, valid picks and working chips; **every `DegradedReason` variant has a test that reaches it**; a turn never raises to the client — worst case is a degraded body; p50 latency on the template path recorded in the PR.

---

### [ ] T25 — SSE streaming + demo UI
**Deps:** T13, T24 · **Branch:** `task/T25-ui`

**Owns:** `src/assist/api/routes_stream.py`, `web/*`, `tests/test_stream.py`

**Do:** SSE endpoint streaming LangGraph node updates as stage frames (constraints → candidates → validated cards → reply). Single-page UI served by FastAPI, no build step: search input, streamed reply, title cards, tappable chips that post `chip_id`.

**Acceptance:** browser demo completes a 3-turn conversation with sticky constraints via chip taps; stream frames arrive progressively (asserted on timing, not just content); **no unvalidated `catalog_id` is ever sent to the client** — including mid-stream.

---

### [ ] T26 — Golden set + eval harness
**Deps:** T24 · **Branch:** `task/T26-eval`

**Owns:** `src/assist/jobs/eval.py`, `data/golden/*`, `tests/test_eval.py`

**Do:** ~60 golden queries stratified across mood/genre, person-fuzzy (≥15%), decade, duration, known-item, reset, adversarial, and vague. `jobs eval` **invokes the compiled graph** per query and reports recall@8, person@1, schema-failure rate, route mix, degraded rate, latency p50/p95 per stage, and USD per turn. Markdown report to `docs/eval-report.md`.

**Acceptance:** `make eval` produces the report with every metric populated; re-running with a fixed seed and cached LLM responses is deterministic; the report header states which fields are synthetic fixtures.

---

### [ ] T27 — Hardening, scale overlay, README
**Deps:** T24 · **Branch:** `task/T27-hardening`

**Owns:** `docker-compose.scale.yml`, `docker/nginx/*`, `README.md`, `docs/runbook.md`

**Do:** Prove `docker compose --profile scale up --scale api=3` behaves identically to one replica (session and rate limit shared through Redis). nginx routing + `X-Request-Id` passthrough. README: quickstart, seeding, architecture (embedding the generated Mermaid), what is synthetic, cost notes, and the model-tier comparison.

**Acceptance:** with 3 replicas, a session started on one replica continues correctly on another (asserted by forcing round-robin); the rate limiter enforces one shared budget across replicas, not three; a fresh clone can go from `git clone` to a working demo following only the README.

---

## 6. File ownership map

Two agents must never edit the same file concurrently. If your task needs a file it does not own, **report it instead of editing it**.

| Path | Owner |
|---|---|
| `pyproject.toml` | shared — dependency additions only, union-merge |
| `docs/WORKPLAN.md` | shared — your own checkbox line only |
| `src/assist/config.py` | T01, then additive by any task (one setting per task, append) |
| `src/assist/domain/**` | T02 |
| `docker-compose.yml` | T03 (T27 owns the `scale` overlay only) |
| `src/assist/stores/db.py`, `migrations/**` | T04 |
| `src/assist/stores/session.py`, `cache.py`, `ratelimit.py` | T05 |
| `src/assist/stores/es.py`, `mappings/**` | T06 |
| `services/embedder/**` | T07 |
| `src/assist/llm/**` | T08 (T15, T23 add their own prompt files only) |
| `src/assist/graph/state.py`, `build.py` | T09, then T24 |
| `src/assist/graph/edges.py` | T09, then T22 |
| `src/assist/nodes/<one file per task>` | see each card |
| `src/assist/api/**` | T13 (T25 adds `routes_stream.py` only) |
| `src/assist/jobs/**` | T10, T11, T12, T26 — one file each |
| `README.md` | T27 (others append a section only if they change how it runs) |
