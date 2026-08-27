"""Enrichment job: resumable LLM pass, empty-moods on failure, JSONL import."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import subprocess
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import Field
from sqlalchemy import text
from testcontainers.community.postgres import PostgresContainer
from typer.testing import CliRunner

from assist.config import EnrichMode, LLMProvider, Settings
from assist.domain.catalog import Person, Title
from assist.domain.enums import (
    Audience,
    CreditRole,
    GenreId,
    MediaType,
    MoodId,
    Pace,
)
from assist.jobs.enrich import (
    ARTIFACT_NAME,
    EMPTY_ENRICHMENT,
    Enrichment,
    TitleCard,
    default_artifact_path,
    enrich_one,
    enrich_pending,
    estimate_cost,
    format_dry_run_line,
    load_jsonl,
    parse_jsonl_line,
    record_to_line,
    run_async,
    write_enrichment,
    write_jsonl,
)
from assist.llm.cost import CostCallbackHandler, cost_usd
from assist.llm.gateway import LLMUnavailable
from assist.stores.db import CreditRecord, Database

ROOT = Path(__file__).resolve().parents[1]
ENRICH_PATH = ROOT / "src" / "assist" / "jobs" / "enrich.py"
TRUNCATE_SQL = "TRUNCATE titles, people, credits RESTART IDENTITY CASCADE"


def _settings(**kwargs: Any) -> Settings:
    values: dict[str, Any] = {
        "llm_provider": LLMProvider.NONE,
        "anthropic_api_key": None,
        "enrich": EnrichMode.LLM,
        "enrich_limit": 2500,
        "enrich_concurrency": 2,
        "enrich_tokens_in": 350,
        "enrich_tokens_out": 180,
        "llm_timeout_ms": 2500,
    }
    values.update(kwargs)
    return Settings(**values)


def _title(catalog_id: str = "s1", **overrides: object) -> Title:
    payload: dict[str, object] = {
        "catalog_id": catalog_id,
        "media_type": MediaType.FILM,
        "title": "The Matrix",
        "synopsis": "A hacker learns the truth.",
        "release_year": 1999,
        "runtime_min": 136,
        "maturity_rank": 6,
        "origins": ("United States",),
        "genres": (GenreId.SCIFI, GenreId.ACTION),
        "pop_28d": 0.8,
    }
    payload.update(overrides)
    return Title(**payload)  # type: ignore[arg-type]


def _enrichment(**overrides: object) -> Enrichment:
    payload: dict[str, object] = {
        "moods": [MoodId.TENSE, MoodId.DARK],
        "tags": ["dystopia", "slow-burn"],
        "audience": Audience.ADULT,
        "pace": Pace.FAST,
        "era_feel": "90s sci-fi",
        "one_line_hook": "Reality is a lie.",
    }
    payload.update(overrides)
    return Enrichment(**payload)  # type: ignore[arg-type]


def _card(catalog_id: str = "s1") -> TitleCard:
    return TitleCard(
        catalog_id=catalog_id,
        title="The Matrix",
        year="1999",
        media_type="film",
        genres="scifi, action",
        origins="United States",
        cast_head="Keanu Reeves",
        synopsis="A hacker learns the truth.",
    )


class _CannedChat(BaseChatModel):
    canned: Enrichment
    call_log: list[str] = Field(default_factory=list)
    seen_configs: list[RunnableConfig] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "canned-enrich"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise AssertionError("raw generate must not run")

    def with_structured_output(
        self,
        schema: dict[str, Any] | type[Any],
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        assert kwargs.get("method") == "json_schema"

        def _run(_input: Any, config: RunnableConfig | None = None) -> Enrichment:
            self.call_log.append("sync")
            if config is not None:
                self.seen_configs.append(config)
            return self.canned

        async def _arun(_input: Any, config: RunnableConfig | None = None) -> Enrichment:
            self.call_log.append("async")
            if config is not None:
                self.seen_configs.append(config)
            return self.canned

        return RunnableLambda(_run, afunc=_arun)


class _FailChat(BaseChatModel):
    error: Exception
    call_log: list[int] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fail-enrich"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise AssertionError("raw generate must not run")

    def with_structured_output(
        self,
        schema: dict[str, Any] | type[Any],
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        def _raise(_input: Any, config: RunnableConfig | None = None) -> Any:
            self.call_log.append(1)
            raise self.error

        return RunnableLambda(_raise)


class _ResumeChat(BaseChatModel):
    canned: Enrichment
    call_ids: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "resume-enrich"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise AssertionError("raw generate must not run")

    def with_structured_output(
        self,
        schema: dict[str, Any] | type[Any],
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        async def _arun(value: Any, config: RunnableConfig | None = None) -> Enrichment:
            title = ""
            if isinstance(value, dict):
                title = str(value.get("title", ""))
            self.call_ids.append(title)
            return self.canned

        return RunnableLambda(lambda _x: self.canned, afunc=_arun)


def _run_alembic_upgrade(dsn: str) -> None:
    env = os.environ.copy()
    env["POSTGRES_DSN"] = dsn
    proc = subprocess.run(
        ["uv", "run", "alembic", "upgrade", "head"],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        output = (proc.stdout or "") + (proc.stderr or "")
        raise RuntimeError(f"alembic upgrade head failed ({proc.returncode}):\n{output}")


@pytest.fixture(scope="module")
def postgres_dsn() -> Iterator[str]:
    with PostgresContainer(
        "postgres:16-alpine",
        username="assist",
        password="assist",
        dbname="assist",
        driver="asyncpg",
    ) as container:
        dsn = container.get_connection_url()
        _run_alembic_upgrade(dsn)
        yield dsn


@pytest.fixture
async def database(postgres_dsn: str) -> AsyncIterator[Database]:
    db = Database.from_dsn(postgres_dsn, pool_size=2, max_overflow=1)
    async with db.engine.begin() as conn:
        await conn.execute(text(TRUNCATE_SQL))
    try:
        yield db
    finally:
        await db.dispose()


async def _seed_titles(db: Database, catalog_ids: list[str]) -> None:
    async with db.session() as session:
        repo = db.titles(session)
        for catalog_id in catalog_ids:
            await repo.upsert(_title(catalog_id, title=f"Title {catalog_id}"))


def test_enrich_module_uses_gateway_structured_output() -> None:
    source = ENRICH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENRICH_PATH))
    imported_gateway_helper = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "with_structured_output":
            pytest.fail("call assist.llm.gateway.structured_output, not with_structured_output")
        if isinstance(node, ast.ImportFrom) and node.module == "assist.llm.gateway":
            imported_gateway_helper = any(alias.name == "structured_output" for alias in node.names)
    assert imported_gateway_helper
    assert "structured_output(" in source
    assert "CostCallbackHandler" in source
    assert '"callbacks"' in source or "'callbacks'" in source


def test_empty_fallback_has_empty_moods() -> None:
    assert EMPTY_ENRICHMENT.moods == []
    assert EMPTY_ENRICHMENT.tags == []
    assert EMPTY_ENRICHMENT.one_line_hook == ""
    assert EMPTY_ENRICHMENT.era_feel is None


def test_tags_capped_lowercased_and_hook_capped() -> None:
    enrichment = Enrichment(
        moods=[MoodId.FUNNY],
        tags=[" Dystopia ", "dystopia", "A", "B", "C", "D", "E", "F", "G", "H"],
        audience=Audience.ADULT,
        pace=Pace.MEDIUM,
        one_line_hook="x" * 120,
    )
    assert enrichment.tags[0] == "dystopia"
    assert len(enrichment.tags) == 8
    assert len(enrichment.one_line_hook) == 90


def test_estimate_cost_matches_haiku_rates() -> None:
    tokens_in, tokens_out, usd = estimate_cost(2500, tokens_in=350, tokens_out=180)
    assert tokens_in == 2500 * 350
    assert tokens_out == 2500 * 180
    assert usd == pytest.approx(cost_usd(tokens_in, tokens_out))
    # 875_000 in * $1/MTok + 450_000 out * $5/MTok = 0.875 + 2.25 = 3.125
    assert usd == pytest.approx(3.125)


def test_dry_run_prints_token_and_usd_estimate() -> None:
    from assist.config import settings as live_settings
    from assist.jobs.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["enrich", "--dry-run", "--limit", "10"])
    assert result.exit_code == 0, result.output
    tokens_in, tokens_out, usd = estimate_cost(
        10,
        tokens_in=live_settings.enrich_tokens_in,
        tokens_out=live_settings.enrich_tokens_out,
    )
    expected = format_dry_run_line(10, tokens_in, tokens_out, usd)
    assert expected in result.stdout
    assert f"tokens_in={tokens_in}" in result.stdout
    assert "cost_usd=" in result.stdout


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "titles.jsonl"
    rows = [("s1", _enrichment()), ("s2", EMPTY_ENRICHMENT)]
    write_jsonl(path, rows)
    loaded = load_jsonl(path)
    assert [catalog_id for catalog_id, _ in loaded] == ["s1", "s2"]
    assert loaded[0][1] == _enrichment()
    assert loaded[1][1].moods == []
    parsed = parse_jsonl_line(record_to_line("s9", _enrichment()))
    assert parsed is not None
    assert parsed[0] == "s9"


def test_committed_artifact_parses_without_api_key() -> None:
    path = default_artifact_path()
    assert path.name == ARTIFACT_NAME
    assert path.is_file()
    rows = load_jsonl(path)
    assert len(rows) >= 1
    catalog_id, enrichment = rows[0]
    assert catalog_id.startswith("s")
    assert isinstance(enrichment, Enrichment)
    for _catalog_id, item in rows:
        assert len(item.tags) <= 8
        assert len(item.one_line_hook) <= 90


def test_parse_nested_and_invalid_jsonl() -> None:
    nested = json.dumps(
        {
            "catalog_id": "s42",
            "enrichment": {
                "moods": ["funny"],
                "tags": ["stand-up"],
                "audience": "adult",
                "pace": "medium",
                "era_feel": None,
                "one_line_hook": "A joke.",
            },
        }
    )
    parsed = parse_jsonl_line(nested)
    assert parsed is not None
    assert parsed[0] == "s42"
    assert parsed[1].moods == [MoodId.FUNNY]
    assert parse_jsonl_line("{not json") is None
    assert parse_jsonl_line("") is None
    assert parse_jsonl_line(json.dumps({"moods": []})) is None


async def test_failed_call_writes_empty_moods(database: Database) -> None:
    await _seed_titles(database, ["s1"])
    model = _FailChat(error=LLMUnavailable("no key"))
    handler = CostCallbackHandler()
    counts = await enrich_pending(
        database,
        limit=10,
        concurrency=2,
        model=model,
        settings=_settings(),
        cost_handler=handler,
    )
    assert counts["processed"] == 1
    assert counts["failed"] == 1
    async with database.session() as session:
        stored = await database.titles(session).get_stored("s1")
    assert stored is not None
    assert stored.enrichment is not None
    assert stored.enrichment["moods"] == []
    assert stored.title.moods == ()


async def test_schema_failure_writes_empty_moods(database: Database) -> None:
    await _seed_titles(database, ["s1"])
    model = _FailChat(error=OutputParserException("bad json"))
    counts = await enrich_pending(
        database,
        limit=10,
        concurrency=2,
        model=model,
        settings=_settings(llm_provider=LLMProvider.ANTHROPIC, anthropic_api_key="sk-test"),
        cost_handler=CostCallbackHandler(),
    )
    assert counts["failed"] == 1
    async with database.session() as session:
        stored = await database.titles(session).get_stored("s1")
    assert stored is not None
    assert stored.enrichment is not None
    assert stored.enrichment["moods"] == []


async def test_resume_skips_already_enriched(database: Database) -> None:
    await _seed_titles(database, ["s1", "s2", "s3"])
    canned = _enrichment()
    first = _ResumeChat(canned=canned)
    await enrich_pending(
        database,
        limit=2,
        concurrency=2,
        model=first,
        settings=_settings(),
        cost_handler=CostCallbackHandler(),
    )
    assert len(first.call_ids) == 2
    async with database.session() as session:
        remaining = await database.titles(session).count(unenriched_only=True)
    assert remaining == 1
    second = _ResumeChat(canned=canned)
    counts = await enrich_pending(
        database,
        limit=10,
        concurrency=2,
        model=second,
        settings=_settings(),
        cost_handler=CostCallbackHandler(),
    )
    assert counts["processed"] == 1
    assert len(second.call_ids) == 1
    async with database.session() as session:
        assert await database.titles(session).count(unenriched_only=True) == 0
        for catalog_id in ("s1", "s2", "s3"):
            stored = await database.titles(session).get_stored(catalog_id)
            assert stored is not None
            assert stored.enrichment is not None
            assert stored.enrichment["moods"] == ["tense", "dark"]


async def test_cost_callback_is_attached(database: Database) -> None:
    await _seed_titles(database, ["s1"])
    model = _CannedChat(canned=_enrichment())
    handler = CostCallbackHandler()
    await enrich_pending(
        database,
        limit=1,
        concurrency=1,
        model=model,
        settings=_settings(llm_provider=LLMProvider.ANTHROPIC, anthropic_api_key="sk-test"),
        cost_handler=handler,
    )
    assert model.call_log
    assert model.seen_configs
    raw_callbacks = model.seen_configs[0].get("callbacks")
    if isinstance(raw_callbacks, list):
        handlers = list(raw_callbacks)
    else:
        handlers = list(getattr(raw_callbacks, "handlers", []))
    assert any(isinstance(item, CostCallbackHandler) for item in handlers)


async def test_import_jsonl_without_calling_llm(database: Database, tmp_path: Path) -> None:
    await _seed_titles(database, ["s1", "s2"])
    path = tmp_path / "titles.jsonl"
    write_jsonl(path, [("s1", _enrichment()), ("s-missing", _enrichment())])
    counts = await run_async(
        mode=EnrichMode.SKIP,
        artifact=path,
        db=database,
        settings=_settings(enrich=EnrichMode.SKIP, llm_provider=LLMProvider.NONE),
    )
    assert counts["processed"] == 1
    assert counts["skipped"] == 1
    async with database.session() as session:
        stored = await database.titles(session).get_stored("s1")
        missing = await database.titles(session).get_stored("s2")
    assert stored is not None
    assert stored.enrichment is not None
    assert stored.enrichment["one_line_hook"] == "Reality is a lie."
    assert missing is not None
    assert missing.enrichment is None


async def test_cli_import_with_no_api_key(
    postgres_dsn: str, database: Database, tmp_path: Path
) -> None:
    await _seed_titles(database, ["s1"])
    async with database.session() as session:
        person = Person(
            person_id="p1",
            name="Ada Actor",
            name_norm="ada actor",
            roles=(CreditRole.ACTOR,),
            credit_count=1,
        )
        await database.people(session).upsert(person)
        await database.credits(session).upsert(
            CreditRecord(catalog_id="s1", person_id="p1", role=CreditRole.ACTOR)
        )
    path = tmp_path / "titles.jsonl"
    write_jsonl(path, [("s1", _enrichment())])
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    env["LLM_PROVIDER"] = "none"
    env["ENRICH"] = "skip"
    env["POSTGRES_DSN"] = postgres_dsn
    proc = await asyncio.to_thread(
        subprocess.run,
        [
            "uv",
            "run",
            "python",
            "-m",
            "assist.jobs.cli",
            "enrich",
            "--mode",
            "skip",
            "--from-artifact",
            str(path),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    async with database.session() as session:
        stored = await database.titles(session).get_stored("s1")
        names = await database.credits(session).names_for_titles(["s1"])
    assert stored is not None
    assert stored.enrichment is not None
    assert stored.enrichment["moods"] == ["tense", "dark"]
    assert names["s1"] == ["Ada Actor"]


async def test_enrich_one_unavailable_returns_empty() -> None:
    from assist.jobs.enrich import build_chain

    model = _FailChat(error=LLMUnavailable("none"))
    built = build_chain(model=model, settings=_settings())
    result = await enrich_one(_card(), built, CostCallbackHandler())
    assert result == EMPTY_ENRICHMENT
    assert result.moods == []


async def test_export_jsonl(database: Database, tmp_path: Path) -> None:
    await _seed_titles(database, ["s1"])
    await write_enrichment(database, "s1", _enrichment())
    out = tmp_path / "out.jsonl"
    counts = await run_async(
        mode=EnrichMode.NONE,
        export_path=out,
        db=database,
        settings=_settings(enrich=EnrichMode.NONE),
    )
    assert counts["exported"] == 1
    loaded = load_jsonl(out)
    assert loaded[0][0] == "s1"
    assert loaded[0][1].one_line_hook == "Reality is a lie."
