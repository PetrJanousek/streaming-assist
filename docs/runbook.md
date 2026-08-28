# Runbook — streaming-assist

Operations notes for the local Docker stack. The architecture lives in
[`implementation-plan.md`](implementation-plan.md). This file is *how to run it*.

## 1. Start the stack

Stores only (Postgres, Elasticsearch, Redis):

```bash
make up
```

Full stack (stores + embedder + api):

```bash
make up-all
```

Wait until every service is healthy:

```bash
docker compose ps
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/readyz
```

`/healthz` is process liveness. It does not talk to Redis.
`/readyz` fails when Redis is down. Use `/readyz` for "can this replica serve a turn".

## 2. Seed the catalog

```bash
make seed
```

Equivalent without Make:

```bash
docker compose --profile tools run --rm jobs seed-all
```

`seed-all` is fetch → normalize → enrich → index.

Default `ENRICH=skip` imports the committed JSONL. That path spends $0 and needs no API key.

Re-run is safe. Normalize is idempotent. Enrich skips titles that already have a payload. Index writes a new versioned ES index and swaps the alias.

## 3. Serve a turn

Bearer tokens map to seeded profiles. They are fixtures, not real auth.

| Token | Profile |
|---|---|
| `dev-adult` | US premium, maturity R, web |
| `dev-kids` | US basic, maturity PG, kids, tv |
| `dev-basic` | DE basic, maturity PG-13, mobile |

```bash
curl -sS http://127.0.0.1:8000/v1/assist/turn \
  -H "Authorization: Bearer dev-adult" \
  -H "Content-Type: application/json" \
  -d '{"message":{"type":"text","text":"a cozy comedy under 100 minutes"}}'
```

Chip follow-ups send only `chip_id`. The delta stays in Redis on the session.

```bash
curl -sS http://127.0.0.1:8000/v1/assist/turn \
  -H "Authorization: Bearer dev-adult" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"<id>","message":{"type":"chip","chip_id":"<chip>"}}'
```

Unknown or expired `chip_id` → HTTP 400 `chip_invalid`.
Missing or unknown bearer → HTTP 401.
Token-bucket exceeded → HTTP 429 `rate_limited` with a usable body.

A turn never returns HTTP 500. The worst case is a degraded body.

## 4. Scale to three API replicas

The API holds no local state. Session, chips, caches, and rate-limit buckets live in Redis.

```bash
docker compose -f docker-compose.yml -f docker-compose.scale.yml \
  --profile scale up -d --wait --wait-timeout 180 --scale api=3
```

Host traffic then goes to `http://127.0.0.1/` (nginx). `api:8000` is unpublished.

nginx re-resolves the Docker DNS name `api` on every request. That is what makes round-robin work. Confirm three replicas and a passing gateway:

```bash
docker compose ps
curl -sS -D - -o /dev/null -H "X-Request-Id: runbook-1" http://127.0.0.1/healthz
```

The response echoes `X-Request-Id` and adds `X-Upstream-Addr` (the replica that served the request). Two sequential `/healthz` calls should not always show the same upstream address.

Suggested Makefile target, not added this stage (T26 owns the Makefile):

```make
scale:
	$(COMPOSE) -f docker-compose.yml -f docker-compose.scale.yml \
	  --profile scale up -d --wait --wait-timeout 180 --scale api=3
```

## 5. Rate limit

Defaults: `RATE_LIMIT_RPS=5`, `RATE_LIMIT_BURST=20`, keyed `rl:user:{user_id}`.

The bucket is a Redis Lua script. Three replicas share one budget. They do not each get a full burst.

A 429 body still matches the turn shape (`reply`, `picks`, `chips`, `meta`) plus `error.type=rate_limited` and `Retry-After`.

## 6. Logs and traces

Every log line is JSON and carries `trace_id`.

- Client may send `X-Request-Id`. nginx forwards it. The API binds it as `trace_id`.
- If the header is missing, nginx mints `$request_id` and the API accepts that value.
- The API echoes `X-Request-Id` and `X-Response-Time-Ms` on the response.

```bash
make logs
# or
docker compose logs -f --tail=200 api gateway
```

Grep a single turn:

```bash
docker compose logs api | grep '"trace_id":"runbook-1"'
```

## 7. Degraded mode

Empty `ANTHROPIC_API_KEY` or `LLM_PROVIDER=none` does not crash the process. Intent falls back to rules. Reply falls back to templates. The turn still returns a body.

Other closed failure modes (all tested in `tests/test_pipeline_e2e.py`):

| `degraded_reason` | Typical cause |
|---|---|
| `hard_timeout` | Turn exceeded `HARD_TIMEOUT_MS` (8s) |
| `generative_timeout` / `generative_schema_fail` | Reply model call failed; template + ranker picks |
| `provider_throttle` | Upstream 429 from the LLM provider |
| `retrieval_unavailable` | ES or embedder down |
| `session_store_unavailable` | Redis session I/O failed |
| `safety_block` | Guard refused the text |
| `empty_catalog_match` | Filters matched nothing after broaden |
| `person_ambiguous` | Two or three close people hits; clarify chips, no guess |

## 8. Elasticsearch memory

The ES container is capped at a 512MB heap and ~1GB RAM. Total stack RSS should stay under ~2.5GB.

If the node will not start on Docker Desktop, raise the VM memory, then:

```bash
docker compose up -d elasticsearch
docker compose logs elasticsearch
```

A reindex (`jobs index` / `make seed`) writes `titles_vN` and swaps the `titles` alias. The previous index stays for rollback.

## 9. Stop and reset

```bash
make down
```

That command includes the `tools` and `scale` profiles so leftover `jobs` / `gateway` containers go away.

Named volumes (`pgdata`, `esdata`, `redisdata`) survive `down`. To wipe catalog state:

```bash
docker compose --profile tools --profile scale down -v
```

## 10. Common failures

| Symptom | What to check |
|---|---|
| `/readyz` is 503, `/healthz` is 200 | Redis is down or not reachable from the replica |
| 401 on `/v1/assist/turn` | Bearer is missing or not one of the three fixture tokens |
| 400 `chip_invalid` | `chip_id` was not minted on this session, or the session TTL (24h sliding) expired |
| 429 | Shared Redis bucket is empty; wait `Retry-After` seconds |
| Gateway never becomes healthy | Overlay healthcheck probes `/healthz`. Confirm `api` is healthy first (`docker compose ps`) |
| All traffic hits one replica | Overlay not applied, or nginx config is the T03 inline stand-in. Use `-f docker-compose.scale.yml` |
| Embedder build is slow | First `docker compose build embedder` downloads `bge-small-en-v1.5` into the image. Later runs are cached |
| Seed wants an API key | `ENRICH=skip` (the default) imports `data/enriched/titles.jsonl`. Do not set `ENRICH=llm` unless you want to spend |

## 11. Host-side checks (no Docker stack)

```bash
uv sync --group dev
make lint typecheck test
make graph    # regenerates docs/graph.mmd from the compiled graph
```
