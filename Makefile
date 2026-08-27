.PHONY: lint typecheck test fmt up down logs ps shell seed

COMPOSE ?= docker compose

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:
	uv run mypy src tests

# pytest exits 5 when nothing is collected; T01 ships an empty suite on purpose
test:
	uv run pytest; \
	ec=$$?; \
	if [ $$ec -eq 0 ] || [ $$ec -eq 5 ]; then exit 0; else exit $$ec; fi

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests

# Stack. `seed` invokes the jobs CLI (T10); the target exists now so later
# tasks do not have to touch the Makefile to wire it.
up:
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
