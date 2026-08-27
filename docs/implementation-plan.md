# Implementation Plan — Conversational Search Assistant

| Field | Value |
|-------|-------|
| **Author** | Petr Janousek |
| **Date** | 2026-08-27 |
| **Status** | Plan / for review |
| **Implements** | `design-conversational-search-assistant.md`, `AIE_assigment_petr_janousek.pdf` |
| **Step 1 goal** | Full hot path, running end-to-end, `docker compose up` on a clean machine |
| **Step 2 goal (not built now)** | Horizontal scale behind a gateway with rate limiting |

---

## 0. Deltas from the design document

These are deliberate, and each one is a scope decision, not a change of architecture.

| Design doc says | This build does | Why |
|---|---|---|
| Czech-first, Czech eval gates, Czech phrase bank | **English only** | Explicitly dropped from the assignment. Every Czech-specific mechanism (ICU analyzer, morphology, Czech eval slice) collapses to its English equivalent. The *architecture* is unchanged — language is a config axis, not a structural one. |
| Serving in Rust/Go preferred, FastAPI acceptable | **Python 3.12 + FastAPI + LangGraph + LangChain** | Requested. FastAPI was already the doc's acceptable-for-MVP option. |
| "Thin in-house pipeline; **no** heavy agent framework" | **LangGraph as a workflow engine** — a `StateGraph` with explicit, hand-written edges. Plus LangChain narrowly for the model factory, structured output, prompt templates, retry/fallback and cost callbacks. **No agent, no tool-calling loop, no model-directed control flow.** | The doc's objection is to *agents* — LLM-directed control flow — not to expressing a deterministic pipeline as a graph. Every edge in this graph is a pure Python predicate over typed state; the model never chooses the next step. In exchange we get free per-node tracing, per-node latency for `meta.stage_latency_ms`, and node-level streaming that makes the progressive UI nearly free. See §7 for the boundary and the discipline rules that keep this a workflow. |
| Real auth, entitlement service, device registry | **Simulated `ServerUserCtx`** — a `/dev/profiles` fixture set, selected by a bearer token that maps to a seeded profile row | Auth is not the interesting part and building it would prove nothing. The *invariant* is preserved and testable: `client_hints` can never influence `playable_now`. |
| Live catalog service with real-time rights | **Postgres as source of truth + synthesized availability windows** behind a `CatalogClient` interface | Keeps the validator a real, cache-backed, fallible call. The interface is the seam where a separate `catalog-service` container drops in later (§10). |
| OpenSearch | **Elasticsearch 8.x**, single node, basic license | Requested. RRF is fused in Python, not by ES — see §5.4. |
| Vector store = "after BM25 ships" | **Both from day one** | The hybrid + RRF fusion is the load-bearing anti-hallucination mechanism; shipping it late would leave the interesting half unbuilt. |

---

## 1. Component topology

```
                         ┌──────────────┐
   browser ──────────────│  demo UI     │  (static page served by api)
                         └──────┬───────┘
                                │ POST /v1/assist/turn
                         ┌──────▼───────────────────────────────┐
      [step 2: gateway]  │  api   FastAPI + LangGraph workflow  │
      nginx/Traefik ───▶ │  stateless, N replicas               │
      + rate limit       │  ┌─ graph: guard → intent → merge    │
                         │  │  → retrieve → rank → validate     │
                         │  │  → route → generate → sanitize    │
                         │  └─ llm gateway (timeouts, cost, FB) │
                         └───┬────────┬─────────┬───────────┬───┘
                             │        │         │           │
                  ┌──────────▼──┐ ┌───▼─────┐ ┌─▼────────┐ ┌▼──────────┐
                  │ redis       │ │ elastic │ │ postgres │ │ embedder  │
                  │ session     │ │ search  │ │ catalog  │ │ bge-small │
                  │ chips       │ │ BM25 +  │ │ people   │ │ CPU, 384d │
                  │ 3 caches    │ │ kNN 384d│ │ avail.   │ │ FastAPI   │
                  │ rate limit  │ │         │ │ taxonomy │ │           │
                  └─────────────┘ └─────────┘ │ phrases  │ └───────────┘
                                              │ eval set │
                                              │ events   │
                                              └──────────┘
                  ┌───────────────────────────────────────┐
                  │ jobs   (profile: tools, run-once)     │
                  │  fetch → normalize → enrich → embed   │
                  │  → index    |    eval                 │
                  └───────────────────────────────────────┘
```

### Containers

| Service | Image | Role | Notes |
|---|---|---|---|
| `api` | local build, `python:3.12-slim` + uv | The whole hot path (LangGraph `StateGraph`) + demo UI + `/healthz`, `/readyz` | Stateless. `--scale api=3` must work unchanged. |
| `postgres` | `postgres:16-alpine` | Source of truth: titles, people, credits, availability windows, taxonomy, phrase bank, golden set, turn events | Volume-backed. |
| `elasticsearch` | `elasticsearch:8.15.x` | Hybrid index: BM25 + `dense_vector` HNSW kNN | `discovery.type=single-node`, `xpack.security.enabled=false`, `ES_JAVA_OPTS=-Xms512m -Xmx512m`. |
| `redis` | `redis:7-alpine` | Session + chip map, intent cache, response cache, availability cache, idempotency, rate-limit buckets | `--maxmemory 256mb --maxmemory-policy allkeys-lru`. |
| `embedder` | local build | `BAAI/bge-small-en-v1.5` on CPU behind `POST /embed` | Model **baked into the image at build time** so runtime needs no network. 384-dim, ~130MB, ~5ms/query on CPU. Separate container so `api` stays light and both `api` and `jobs` share one loaded copy — and so it scales independently in step 2. |
| `jobs` | same image as `api` | One-shot CLI: `fetch`, `normalize`, `enrich`, `embed`, `index`, `eval`, `seed-all` | Compose profile `tools`; never runs in `up`. |

Compose profiles: default (`api`, `postgres`, `elasticsearch`, `redis`, `embedder`), `tools` (`jobs`), `scale` (adds `gateway`, step 2).

Everything is healthchecked with `depends_on: { condition: service_healthy }` so a cold `docker compose up` works on the first try with no ordering races.

---

## 2. Repository layout

Proposed location: `~/Projects/rust/2026/streaming-assist/` (confirm — see §12).

```
streaming-assist/
├── docker-compose.yml
├── docker-compose.scale.yml        # step 2 overlay: gateway + replicas
├── Makefile                        # up / seed / eval / logs / down / test
├── .env.example
├── pyproject.toml                  # uv
├── README.md
│
├── src/assist/
│   ├── main.py                     # FastAPI app, lifespan, DI wiring
│   ├── config.py                   # pydantic-settings; every knob is env
│   │
│   ├── api/
│   │   ├── routes_turn.py          # POST /v1/assist/turn  (+ SSE variant)
│   │   ├── routes_ops.py           # /healthz /readyz /stats /dev/profiles
│   │   ├── schemas.py              # request/response pydantic models
│   │   ├── deps.py                 # auth → ServerUserCtx, request id
│   │   └── middleware.py           # rate limit, idempotency, timing, logs
│   │
│   ├── graph/
│   │   ├── state.py                # TurnState TypedDict — the turn's state, period
│   │   ├── edges.py                # pure routing predicates; no LLM, no I/O
│   │   └── build.py                # StateGraph assembly, compile, cycle assert
│   │
│   ├── nodes/                      # one file per node; nodes own their stage logic
│   │   ├── guard.py                # 2. safety / injection pre-filter
│   │   ├── intent.py               # 3. chip / rules / LLM IntentUpdate
│   │   ├── merge.py                # 4. applies domain/constraints merge algebra
│   │   ├── people.py               # 5. person/alias resolver
│   │   ├── retrieval.py            # 6. ES hybrid query + RRF fusion + broaden
│   │   ├── rank.py                 # 7. deterministic score + franchise cap
│   │   ├── availability.py         # 8. playable_now validator + cache
│   │   ├── reply.py                # 9. template / generative / clarify / refusal
│   │   ├── templates.py            #    phrase bank
│   │   ├── sanitize.py             # 10. wraps domain/picks.sanitize_picks
│   │   ├── chips.py                # 11. server-minted chips
│   │   └── persist.py              # 12. session write-back + turn event
│   │
│   ├── llm/
│   │   ├── gateway.py              # LangChain factory, timeout, retry, fallback
│   │   ├── cost.py                 # callback handler → token/$ per turn
│   │   └── prompts/                # versioned prompt text, not inline strings
│   │
│   ├── stores/
│   │   ├── session.py              # redis session + chip repository
│   │   ├── cache.py                # intent / response / availability caches
│   │   ├── ratelimit.py            # redis token bucket (Lua)
│   │   ├── es.py                   # ES client + query builders + mappings
│   │   └── db.py                   # async SQLAlchemy, models, repositories
│   │
│   ├── domain/                     # pure; imported by nodes AND by jobs/
│   │   ├── constraints.py          # ConstraintState / Delta / FieldOp + merge()
│   │   ├── picks.py                # sanitize_picks() — signature fixed by design doc
│   │   ├── catalog.py              # Title, Person, Candidate, Pick
│   │   ├── context.py              # ServerUserCtx
│   │   └── enums.py                # GenreId, MoodId, SpeechAct, DegradedReason
│   │
│   ├── obs/
│   │   ├── logging.py              # structlog JSON, trace_id on every line
│   │   └── events.py               # AssistTurnEvent → postgres (async)
│   │
│   └── jobs/
│       ├── cli.py                  # typer entrypoint
│       ├── fetch.py                # download dataset → data/raw/
│       ├── normalize.py            # CSV → postgres, taxonomy mapping, synth
│       ├── enrich.py               # one-pass LLM enrichment (batched, resumable)
│       ├── index.py                # embed + bulk index into ES
│       └── eval.py                 # runs the compiled graph per query + report
│
├── data/
│   ├── raw/                        # gitignored, populated by `jobs fetch`
│   ├── taxonomy/                   # genre + mood closed enums, synonym packs
│   ├── phrases/                    # template reply + chip label bank
│   └── golden/                     # eval query set, committed
│
├── web/                            # demo UI: index.html + app.js + style.css
└── tests/
    ├── test_merge_algebra.py       # ≥20 golden cases (doc PR 2 acceptance)
    ├── test_sanitize_picks.py
    ├── test_router.py
    ├── test_authz_invariants.py    # client_hints can never reach playable_now
    ├── test_graph_shape.py         # edges.py imports no llm/; one bounded cycle only
    └── test_pipeline_e2e.py        # fake LLM, real ES/redis/pg via testcontainers
```

---

## 3. Data: source, normalization, enrichment

### 3.1 Source — verified reachable

`https://huggingface.co/datasets/hugginglearners/netflix-shows/resolve/main/netflix_titles.csv` — 3.4 MB, HTTP 200, no auth. Checked 2026-08-27.

**8,807 rows** — 6,131 movies, 2,676 series. **40,948 distinct people** (cast + director). 42 raw genre labels. Real maturity ratings. Years 1925–2021.

| CSV column | Maps to | Notes |
|---|---|---|
| `show_id` | `catalog_id` | Already unique (`s1`…`s8807`). |
| `type` | `media_type` | `Movie` → `film`, `TV Show` → `series`. |
| `title` | `title` | Indexed as `text` (english analyzer) + `keyword` subfield. |
| `director`, `cast` | `people` + `credits` | Comma-split. 7,982 rows have cast. This is what makes the **person-fuzzy query class real** — "the spy film with the older guy from the 90s" resolves against actual names and actual credit years. |
| `country` | `origins` | 122 countries; multi-valued. |
| `release_year` | `release_year` | Drives decade/era constraints. |
| `rating` | `maturity_rank` | TV-Y/TV-G/G → PG → PG-13/TV-14 → R/TV-MA/NC-17, mapped to an integer ladder. **4 rows are dirty** (a duration string leaked into the rating column) — the normalizer quarantines them. |
| `duration` | `runtime_min` \| `seasons` | `"90 min"` vs `"2 Seasons"`, branch on `media_type`. |
| `listed_in` | `genres` | 42 labels → **~20 canonical `GenreId`s** via a committed mapping (`International Movies`/`International TV Shows` become an origin signal, not a genre; `TV Dramas`+`Dramas` collapse to `drama`). |
| `description` | `synopsis` | 100% populated. Primary embedding + BM25 text. |

### 3.2 Synthesized fields (deterministic, seeded by `catalog_id`)

The dataset has no availability, no popularity, no moods. Three fill strategies, in increasing order of interestingness:

1. **Availability windows** — deterministic hash of `catalog_id`: ~85% playable now, ~7% window-expired, ~5% package-gated (`basic` vs `premium`), ~3% geo-restricted. This exists so the validator has something to actually drop, and so the "picks that passed retrieval but failed `playable_now`" counter is non-zero and observable. **Documented as synthetic** in the README — it is a fixture, not a claim.
2. **Popularity (`pop_28d`)** — deterministic pseudo-random, skewed by recency and genre prior. Feeds the 0.50 weight of the ranker.
3. **Moods, tags, audience, era descriptors** — **the LLM enrichment pass**, §3.3. This is the doc's "single pass LLM call which adds metadata like tags, era, who's it for."

`local_original` is `origins ∋ HOME_COUNTRY` (env, default `United States`) **and** a deterministic ~20% flag, so `local_originals_only` is a meaningful filter rather than a country synonym.

### 3.3 Enrichment pass (offline, one time)

For each title, one Haiku call producing a strict schema:

```python
class Enrichment(BaseModel):
    moods: list[MoodId]            # closed enum ~16: cozy, tense, funny, bleak,
                                   # feelgood, thought_provoking, dark, uplifting…
    tags: list[str]                # ≤8 free descriptors, lowercase, for BM25 recall
    audience: Audience             # kids | family | teen | adult
    pace: Pace                     # slow | medium | fast
    era_feel: str | None           # "90s spy thriller", "cold war", …
    one_line_hook: str             # ≤90 chars, used by template replies
```

- Input: title, year, type, genres, country, cast head, synopsis. ~350 in / ~180 out tokens.
- **Batched + concurrent** (semaphore 8) with **resumable checkpointing** — writes per-title into `titles.enrichment` jsonb, skips already-enriched rows. Killing the job and re-running is safe.
- `--limit N` flag. Default seed target: **2,500 titles** (≈$1.75 at Haiku 4.5 rates, ~6 min wall clock). Full 8,807 ≈ $6.
- The enriched artifact is **committed as JSONL** so a fresh clone can seed with `ENRICH=skip` and needs no API key at all for the offline path.
- Failures are non-fatal: a title without enrichment still indexes, just with empty moods. The mood axis degrades, retrieval does not break.

### 3.4 Person index

40,948 people from cast + director. For each: `person_id`, name, normalized name, role set (`actor`/`director`), credit count, `active_year_min`/`max` derived from their titles' release years, popularity prior = credit count × mean title popularity.

`active_year_min/max` is what answers "the older guy from the 90s": the soft-descriptor path filters people by role + active era + popularity, returns the top 2–3, and if they're close the pipeline **clarifies with person chips instead of guessing** — exactly the doc's failure UX.

---

## 4. Data model

### 4.1 Postgres

```sql
titles(catalog_id pk, media_type, title, synopsis, release_year, runtime_min,
       seasons, maturity_rank, origins text[], genres text[], local_original bool,
       pop_28d float, enrichment jsonb, indexed_at)
people(person_id pk, name, name_norm, roles text[], credit_count,
       active_year_min, active_year_max, popularity)
credits(catalog_id, person_id, role, pk(catalog_id, person_id, role))
availability(catalog_id, package, geo, window_start, window_end, playable bool)
taxonomy(kind, id, label, synonyms text[])          -- genres, moods, origins
phrase_bank(id, speech_act, kind, template)          -- reply + chip templates
profiles(profile_id pk, token, maturity_max, kids, geo, package, device_class)
golden_queries(id, text, expect_ids text[], expect_class, slice)
turn_events(id, trace_id, session_id, route, intent_source, degraded_reason,
            stage_latency_ms jsonb, tokens_in, tokens_out, cost_usd, created_at)
```

### 4.2 Elasticsearch — `titles_v1` (alias `titles`)

Filterable keywords: `media_type`, `genres`, `moods`, `origins`, `maturity_rank`, `local_original`, `release_year`, `runtime_min`, `audience`, `pace`, `people_ids`.
Text (english analyzer + synonym pack): `title^3`, `synopsis`, `tags`, `people_names^2`, `era_feel`.
Vector: `embedding` — `dense_vector`, 384 dims, cosine, HNSW.

Index built under a versioned name and swapped via alias, so a reindex is atomic and rollback is one API call.

`people_v1`: `person_id`, `name` (text + edge-ngram), `roles`, `active_year_min/max`, `popularity`.

### 4.3 Redis

| Key | Value | TTL |
|---|---|---|
| `sess:{session_id}` | `ConstraintState` + last 6 turns + `issued_chips` map + `turn_count` | 24h, sliding |
| `cache:intent:{sha1(norm_text ‖ constraints_hash)}` | `IntentUpdate` JSON | 1h |
| `cache:resp:{sha1(norm_text ‖ constraints_hash ‖ ctx_hash)}` | full response | 5m |
| `cache:avail:{catalog_id}:{package}:{geo}` | bool | 45s |
| `idem:{Idempotency-Key}` | full response | 5m |
| `rl:{scope}:{subject}` | token bucket state | rolling |

Chips live **inside** the session object, not as separate keys — one round trip, atomic with the turn write-back, and the AuthZ invariant ("client sends `chip_id` only, server owns the delta") falls out for free.

---

## 5. The hot path

A LangGraph `StateGraph` over a typed `TurnState`. **Every edge is a pure Python predicate — the model never selects a node.**

```
                          load_session
                               │
                             guard ──────────[blocked]──────────┐
                               │                                │
                    ┌──────────┴──────────┐                     │
              (message.type / rules hit?) │                     │
                    │          │          │                     │
            intent_from_chip  ...rules  ...llm                  │
                    └──────────┬──────────┘                     │
                               │                                │
                       merge_constraints                        │
                               │                                │
                     [person hint present?]                     │
                               │                                │
                        resolve_people                          │
                               │                                │
                            retrieve ◀───┐                      │
                               │         │ [0 hits & retries<1] │
                               ├─────────┘ broaden_constraints  │
                               │                                │
                             rank                               │
                               │                                │
                    validate_availability                       │
                               │                                │
                        [route predicate]                       │
              ┌────────┬───────┴───────┬────────┐               │
        reply_template  reply_generative  reply_clarify  ◀──────┘ reply_refusal
              └────────┴───────┬───────┴────────┘
                               │
                        sanitize_picks
                               │
                          mint_chips
                               │
                            persist
```

**Discipline rules that keep this a workflow and not an agent** — each one is a test, not a convention:

1. **No node has tool-calling authority.** The two model calls (`intent_from_llm`, `reply_generative`) are plain structured-output calls. Neither is given tools.
2. **No conditional edge consults the model.** `edges.py` contains pure functions over `TurnState` with no I/O. A test asserts the module imports nothing from `llm/`.
3. **Exactly one cycle exists** — `retrieve → broaden_constraints → retrieve` on zero hits — and it is bounded by a `retrieve_attempts` counter in state with a hard cap of 1. Broadening is a deterministic constraint relaxation ladder, not a model decision. A test asserts the graph has no other cycle.
4. **No checkpointer.** The graph is per-turn and stateless across turns. Sticky constraints stay in *our* Redis session object under *our* merge algebra — letting LangGraph's checkpointer own cross-turn memory would hand the framework the exact guarantee the design doc says must be deterministic.

**On code organization:** nodes own their stage logic. LangGraph is a committed dependency, not something to hold at arm's length behind adapters — `TurnState` *is* the turn's state, with no parallel bookkeeping object shadowing it. The exception is `domain/`: pure types plus the constraint merge algebra, which stay framework-free because the offline jobs import them and because the design document specifies `sanitize_picks` as a pure function with a fixed signature. That is two real callers, not a hedge against a library migration.

| # | Stage | Module | Notes |
|---|---|---|---|
| 1 | Auth → `ServerUserCtx` | `api/deps.py` | Bearer token → seeded profile row. `client_hints` are copied to the log record and **nowhere else**. Enforced by `test_authz_invariants.py`. |
| 2 | Session load | `stores/session.py` | One Redis GET. Reject cross-profile bind. New session if absent. |
| 3 | Guard | `nodes/guard.py` | Rules + heuristics: injection markers, jailbreak patterns, adult/piracy/competitor terms, absurd length. Fails **closed** to a refusal template. Runs **before** any model sees the text. |
| 4 | IntentUpdate | `nodes/intent.py` | Chip → server delta lookup, no NLU. Rules hit (bare genre, decade, duration, known title) → rules delta. Else **one** structured Haiku call. Intent cache checked first. |
| 5 | Merge | `nodes/merge.py` → `domain/constraints.py` | Pure typed algebra. Precedence `chip > text delta > prior`. Hard AuthZ fields are unreachable from a delta. **≥20 golden cases**, including the "raise my own maturity ceiling" attempt, which must be a no-op. |
| 6 | Retrieve | `nodes/retrieval.py` | §5.4 below. |
| 7 | Rank | `nodes/rank.py` | `0.50·pop_norm + 0.30·constraint_match + 0.20·semantic_norm`; franchise/series cap 1 per top-8. |
| 8 | Validate | `nodes/availability.py` | `playable_now(ServerUserCtx)` per candidate, Redis-cached 45s, batched. Non-playable are dropped, never substituted. |
| 9 | Route | `graph/edges.py` (conditional edge, pure) | Chip/safety/rules/high-confidence → TEMPLATE. Person-ambiguous → TEMPLATE + clarify chips, `min_picks=0`. Else → GENERATIVE. `θ₁=0.55`, `θ_gap=0.08`, both logged every turn. |
| 10 | Generate | `nodes/reply.py` | One structured Haiku call over 20–25 numbered candidate cards. Model may only return **indices from the list**. Chip labels chosen from the phrase bank, not written free. |
| 11 | Sanitize | `nodes/sanitize.py` → `domain/picks.py` | Pure allowlist intersection + `min_picks` policy matrix + title-span check on the reply prose. **No second LLM, ever.** |
| 12 | Mint chips, persist | `nodes/chips.py`, `nodes/persist.py` | New `chip_id`s with server-held deltas, session write-back, `AssistTurnEvent` fire-and-forget to Postgres. |

Stage 1 runs in the FastAPI dependency, *before* the graph — `ServerUserCtx` is an immutable input to `TurnState`, so no node can widen it. Stages 2–12 are nodes.

Per-node latency comes from LangGraph's own node events rather than hand-rolled timing contexts, and feeds `meta.stage_latency_ms` when `DEBUG_META=1`. Because every turn traverses the same named nodes, "p95 of `retrieve`" is a well-defined number — which is the observability property an agent loop cannot give you.

### 5.4 Retrieval detail

Two queries issued concurrently against the same filtered candidate set:

1. **BM25** — `multi_match` over `title^3 / synopsis / tags / people_names^2 / era_feel`, with the merged constraints as `filter` clauses (never `must`, so they don't perturb scoring).
2. **kNN** — `embedding` vector from the `embedder` service for the LLM's `query_rewrite`, same filters attached to the kNN clause.
3. Plus a **people→titles join** when the resolver produced `people_include`.

Fused with **RRF in Python** — `score = Σ 1/(k + rank_i)`, `k=60`, ~15 lines.

> **Why not ES's own RRF retriever:** it is a licensed (Platinum/Enterprise) feature. On the basic license it is unavailable, and building the demo on a trial license that expires in 30 days would make the "runs on a clean machine" property false. Fusing in Python costs nothing, keeps the fusion weights inspectable, and is exactly what the design document describes anyway.

Then: franchise diversification, top 20–25 compact cards to the ranker/LLM.

---

## 6. LLM usage and cost

| Call | Model | When | Tokens (in/out) | Cost/call |
|---|---|---|---|---|
| IntentUpdate | `claude-haiku-4-5` | Free-text turns that miss rules + cache | ~300 / ~150 | ~$0.0011 |
| GroundedGenerate | `claude-haiku-4-5` | GENERATIVE route only | ~1,100 / ~180 | ~$0.0020 |
| Enrichment | `claude-haiku-4-5` | Offline, once | ~350 / ~180 | ~$0.0013 |

Haiku 4.5 is $1 / $5 per MTok. A worst-case turn (both calls, no cache) is **~$0.003**; a chip tap or rules hit is **$0**. Seeding 2,500 enriched titles is **~$1.75**.

`ANTHROPIC_MODEL` is one env var — swapping to `claude-sonnet-5` or `claude-opus-5` for a quality comparison is a restart, not a code change. That comparison is worth running once and putting in the README.

**No API key?** The gateway degrades to rules-only intent + template replies and the whole system still serves turns. That is the doc's degradation path, and it doubles as the offline demo mode.

---

## 7. Where LangGraph / LangChain are used — and where they are not

**LangGraph — used as a workflow engine only:**
- `StateGraph` over a typed `TurnState`, with hand-written edges (§5).
- Node-level streaming (`astream`, `stream_mode="updates"`) → drives the SSE progressive UI. Each node completion is a frame: constraints understood → candidates found → validated cards → reply. This is most of P7 for free.
- Node-level events → `meta.stage_latency_ms` and the per-stage percentiles.
- Compiled graph is exported to Mermaid in the README, so the architecture diagram is generated from the code rather than drawn beside it and left to rot.
- Nodes own their stage logic and `TurnState` is the state — no adapter layer, no shadow state object. We picked the dependency; we use it.
- The eval harness **invokes the compiled graph** per golden query and reads the final state, rather than reassembling stages by hand. Faithful to the real path, and it means the graph has one production entry point rather than two.

**LangChain — used narrowly:**
- `ChatAnthropic` behind a factory in `llm/gateway.py` — provider is one env var.
- `.with_structured_output(IntentUpdate)` / `(GroundedReply)` — schema enforcement, the doc's "JSON-schema output and validator" layer 1.
- `ChatPromptTemplate` with versioned prompt files.
- `.with_retry()` and `.with_fallbacks()` — the fallback chain *is* the degradation path: structured call → schema failure → template reply, declaratively.
- A `BaseCallbackHandler` for per-turn token and cost accounting into `AssistTurnEvent`.

**Not used, deliberately:**
- `create_react_agent` / any prebuilt agent, tool-calling loops, model-directed routing — rejected in the design doc for latency, cost, constraint drift and jailbreak surface. Nothing in this build takes a next-step decision from a model.
- LangGraph checkpointers / cross-turn memory — session state is ours (§5, rule 4).
- LangChain retrievers and vector stores — we need filter-first hybrid queries with hand-controlled RRF; the abstraction would hide precisely the part that matters.
- LangChain memory — session state is a typed server-side object with a merge algebra, not a message buffer.
- **LangSmith tracing — off by default.** It ships prompts and retrieved content to an external service. Opt-in behind `LANGSMITH_TRACING=1`, documented as a data-egress decision rather than a default.

---

## 8. Configuration

Single `.env`, everything overridable. `.env.example` committed with working defaults for every non-secret value.

```
ANTHROPIC_API_KEY=            # optional; absent → rules+template degraded mode
LLM_PROVIDER=anthropic        # anthropic | ollama | none
ANTHROPIC_MODEL=claude-haiku-4-5
LLM_TIMEOUT_MS=2500
HOME_COUNTRY=United States
EMBED_MODEL=BAAI/bge-small-en-v1.5
RRF_K=60
ROUTER_THETA1=0.55
ROUTER_THETA_GAP=0.08
RANK_W_POP=0.50 / RANK_W_CONSTRAINT=0.30 / RANK_W_SEMANTIC=0.20
SESSION_TTL_S=86400
RATE_LIMIT_RPS=5 / RATE_LIMIT_BURST=20
HARD_TIMEOUT_MS=8000
RETRIEVE_MAX_ATTEMPTS=2       # the one bounded graph cycle; 1 = no broaden-and-retry
LANGSMITH_TRACING=0           # opt-in; sends prompts + retrieved content off-box
```

Thresholds and ranker weights being env-tunable is not decoration — it is what makes the eval harness able to sweep them.

---

## 9. Build order

Each phase ends in something runnable. Nothing is "done" until its acceptance line passes.

| Phase | Contents | Acceptance |
|---|---|---|
| **P0 — Skeleton** | compose (5 services, healthchecks), Makefile, config, logging, `/healthz`, `/readyz`, `POST /v1/assist/turn` returning a hardcoded shape, **`TurnState` + a 3-node stub graph** (wiring the graph now is far cheaper than retrofitting it at P4) | `docker compose up` on a clean machine → `make smoke` green; graph renders to Mermaid |
| **P1 — Data** | `fetch`, `normalize` (taxonomy mapping, synthesized availability/popularity, dirty-row quarantine), Postgres schema + migrations, person index build | `make seed` → 8,807 titles, ~41k people, availability rows in Postgres |
| **P2 — Retrieval** | ES mappings, embedder container, `enrich`, `index`, BM25 + kNN + Python RRF, filter-first constraints, ranker, franchise cap | `jobs eval --retrieval-only` reports recall@8 on the golden set |
| **P3 — State** | `ConstraintState`, `ConstraintDelta`, merge algebra, Redis session + chip repository | ≥20 merge golden cases pass, incl. the maturity-raise no-op |
| **P4 — Deterministic path** | Guard, rules intent, availability validator, template replies, phrase bank, chip minting, `sanitize_picks`, router — all wired as graph nodes + pure edge predicates; bounded broaden-and-retry cycle | **First real end-to-end turn with zero LLM calls.** Sticky constraints work across turns via chips. Graph-shape tests green (no model in `edges.py`, one bounded cycle). |
| **P5 — LLM path** | LLM gateway, IntentUpdate call, grounded reply call, structured-output enforcement, fallback chain, intent + response caches, cost accounting | Free-text multi-turn conversation works; killing `ANTHROPIC_API_KEY` degrades instead of erroring |
| **P6 — People** | Alias resolver, soft-descriptor path (role + era + popularity), clarify chips, `min_picks=0` handling | "the spy film with the older guy from the 90s" either resolves or clarifies — never invents |
| **P7 — UI + eval** | Demo page (input, cards, tap chips, SSE progressive render), golden set, eval runner reporting recall@k, person@1, schema-fail rate, latency percentiles, $/turn | `make eval` prints the scorecard; browser demo is usable end-to-end |
| **P8 — Hardening** | Rate limiter, idempotency, hard timeout + degraded paths, `degraded_reason` enum coverage, `turn_events` analytics, README | Every `degraded_reason` is reachable by a test; `docker compose up --scale api=3` behaves identically |

---

## 10. Step 2 readiness — what P0–P8 already does for it

Not building the gateway now. But these choices make it a config change rather than a refactor, and each costs nothing today:

| Step 2 need | Built in now |
|---|---|
| Multiple API replicas | `api` holds **zero** local state. Session, chips, caches, rate-limit buckets all in Redis. `--scale api=3` is expected to work at P8 and is tested there. |
| Rate limiting | **Redis token bucket in middleware, from P8.** Shared across replicas the moment there is more than one. The gateway later becomes pure routing + TLS, not a stateful component; per-user/per-session/per-IP scopes are already distinct. |
| Gateway | `docker-compose.scale.yml` overlay adds nginx (or Traefik) in front. No app change — the app already trusts `X-Request-Id` / forwarded headers correctly. |
| Independent scaling of the expensive parts | `embedder` is already its own container; LLM calls already go through one gateway module with a timeout and concurrency budget. |
| Catalog as a separate service | Availability sits behind `CatalogClient`. Extracting it to its own container is moving one file and swapping the implementation for an HTTP one. |
| Backpressure / shedding | The router already sheds to TEMPLATE on LLM timeout/throttle. Under load that becomes the load-shedding lever, not an error path. |
| Observability at scale | `trace_id` on every log line, per-stage latency + cost already recorded per turn in `turn_events`. Prometheus/OTel export is an exporter, not instrumentation work. |

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| Dataset URL rots | `jobs fetch` verifies a checksum and falls back to a committed 500-row sample so the build never hard-fails. Enriched artifact is committed regardless. |
| ES container is the memory hog on a laptop | Capped at 512MB heap, single node, no replicas. Total stack should sit under ~2.5GB. |
| Haiku structured-output failures on the reply call | Retries = 0 by design; failure routes to template + ranker picks with `degraded_reason=generative_schema_fail`. The failure rate is a first-class eval metric, not a surprise. |
| Enrichment cost/time surprises someone re-running seed | `--limit`, resumable checkpoints, and a committed pre-enriched artifact so the default path spends $0. |
| Synthetic availability/popularity read as real | Stated in the README, in the plan, and in the eval report header. They are fixtures that make the validator and ranker observable. |
| Latency with two LLM calls | Only the GENERATIVE route makes two, and only on a cold cache. Chips, rules hits, and cache hits make zero. The eval harness reports the route mix, so the claim is measured rather than asserted. |
| LangChain / LangGraph version churn | Pinned in `uv.lock`. Accepted as a normal dependency risk — we are not paying an abstraction tax up front to hedge a migration that probably never happens. |
| Graph drifts toward an agent under pressure | The four discipline rules in §5 are enforced by `test_graph_shape.py`, not by memory. Adding a model call to an edge fails CI. |

---

## 12. Open questions

**Decided 2026-08-27:** orchestration is a **LangGraph `StateGraph` used as a workflow** — explicit edges, pure predicates, no agent. Rationale and the discipline rules that keep it from drifting into an agent are in §5 and §7. The alternative considered and rejected was a ReAct agent: it makes sticky constraints advisory rather than enforced, multiplies per-turn cost and latency, widens the injection surface, and would have contradicted the design document's own `A1` rejection.

Still open:

1. **Code location.** I proposed `~/Projects/rust/2026/streaming-assist/`. Confirm or name another path.
2. **Streaming.** I've planned SSE that streams *stage progress* (skeleton → validated cards → reply), which now falls out of LangGraph's node events almost for free. Token-level streaming conflicts with structured output on the reply call. Confirm stage-level is what you want.
3. **Full 8,807-title enrichment, or 2,500?** 2,500 keeps seed cost ~$1.75 and covers the golden set comfortably. Full coverage is ~$6 and makes retrieval demos denser.
4. **Eval golden set size.** I'll hand-write ~60 queries stratified across the doc's classes (mood/genre, person-fuzzy, decade, duration, known-item, reset, adversarial/injection, vague). More is better but it's manual labour — say if you want 150.

---

## 13. First deliverable

On go, I start at **P0** and stop at the end of **P1** for a checkpoint: compose stack up, healthy, seeded, and `psql` showing the real catalog — before any retrieval or LLM work. That's the cheapest point at which the "portable, no host installs" claim is either true or false.
