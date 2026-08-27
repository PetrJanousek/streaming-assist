.PHONY: lint typecheck test fmt up up-all down logs ps shell seed graph

COMPOSE ?= docker compose

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:
	uv run mypy src tests

test:
	uv run pytest

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests

# Stack. `seed` invokes the jobs CLI (T10); the target exists now so later
# tasks do not have to touch the Makefile to wire it.
#
# `up` starts the stores only. The api container cannot stay up until T13
# adds src/assist/main.py. Use `up-all` for the full stack (api + embedder).
up:
	$(COMPOSE) up -d --wait --wait-timeout 180 postgres elasticsearch redis

up-all:
	$(COMPOSE) up -d

down:
	$(COMPOSE) --profile tools --profile scale down

logs:
	$(COMPOSE) logs -f --tail=200

ps:
	$(COMPOSE) ps

shell:
	$(COMPOSE) exec api bash

seed:
	$(COMPOSE) --profile tools run --rm jobs seed-all

# Architecture diagram from the compiled graph, not a hand-drawn copy.
graph:
	uv run python -c "from assist.graph.build import export_mermaid; export_mermaid()"
