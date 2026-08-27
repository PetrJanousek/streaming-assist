.PHONY: lint typecheck test fmt

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
