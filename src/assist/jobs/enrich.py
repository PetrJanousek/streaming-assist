"""One-pass LLM enrichment of catalog titles. Resumable, $0 by default.

Writes `titles.enrichment` jsonb. Default `ENRICH=skip` imports the committed
JSONL artifact so `seed-all` spends no API money. `ENRICH=llm` calls Haiku
through the gateway helper (json_schema, never tool-calling). Failures write
empty moods so T12 still indexes the title.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from assist.config import EnrichMode, Settings
from assist.config import settings as default_settings
from assist.domain.enums import Audience, MoodId, Pace
from assist.jobs.fetch import data_dir
from assist.llm.cost import CostCallbackHandler, cost_usd
from assist.llm.gateway import LLMError, structured_output
from assist.obs.logging import get_logger
from assist.stores.db import Database, TitleRecord

log = get_logger("assist.jobs.enrich")

ARTIFACT_NAME = "titles.jsonl"
DEFAULT_LIMIT = 2500
CAST_HEAD_N = 5

# Closed-schema card. Input ~350 / output ~180 tokens per plan §3.3.
ENRICH_PROMPT = """You add catalog metadata for one title. You are not a recommender.

Use only the title card below. Never invent a catalog_id or a person_id.
Never name a title that is not the one in the card.

Moods must be chosen from this closed list: {mood_ids}
Audience is one of: kids, family, teen, adult
Pace is one of: slow, medium, fast
Tags: at most 8 lowercase descriptors for lexical recall.
one_line_hook: at most 90 characters.

Title card:
title: {title}
year: {year}
type: {media_type}
genres: {genres}
country: {origins}
cast: {cast_head}
synopsis: {synopsis}
"""

_PROMPT = ChatPromptTemplate.from_template(ENRICH_PROMPT)


class Enrichment(BaseModel):
    """Plan §3.3 schema. Stored as `titles.enrichment` jsonb."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    moods: list[MoodId] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    audience: Audience
    pace: Pace
    era_feel: str | None = None
    one_line_hook: str = ""

    @field_validator("tags", mode="before")
    @classmethod
    def _clean_tags(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            tag = item.strip().lower()
            if tag and tag not in cleaned:
                cleaned.append(tag)
            if len(cleaned) == 8:
                break
        return cleaned

    @field_validator("one_line_hook", mode="before")
    @classmethod
    def _cap_hook(cls, value: object) -> str:
        if not isinstance(value, str):
            return ""
        return value.strip()[:90]

    @field_validator("era_feel", mode="before")
    @classmethod
    def _empty_era(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        return stripped or None


EMPTY_ENRICHMENT = Enrichment(
    moods=[],
    tags=[],
    audience=Audience.ADULT,
    pace=Pace.MEDIUM,
    era_feel=None,
    one_line_hook="",
)


@dataclass(frozen=True)
class TitleCard:
    catalog_id: str
    title: str
    year: str
    media_type: str
    genres: str
    origins: str
    cast_head: str
    synopsis: str


def enriched_dir() -> Path:
    return data_dir() / "enriched"


def default_artifact_path() -> Path:
    return enriched_dir() / ARTIFACT_NAME


def enrichment_payload(enrichment: Enrichment) -> dict[str, object]:
    return {
        "moods": [mood.value for mood in enrichment.moods],
        "tags": list(enrichment.tags),
        "audience": enrichment.audience.value,
        "pace": enrichment.pace.value,
        "era_feel": enrichment.era_feel,
        "one_line_hook": enrichment.one_line_hook,
    }


def record_to_line(catalog_id: str, enrichment: Enrichment) -> str:
    payload = enrichment_payload(enrichment)
    payload["catalog_id"] = catalog_id
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def parse_jsonl_line(line: str) -> tuple[str, Enrichment] | None:
    raw = line.strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    catalog_id = obj.get("catalog_id")
    if not isinstance(catalog_id, str) or not catalog_id:
        return None
    nested = obj.get("enrichment")
    source = nested if isinstance(nested, dict) else obj
    try:
        enrichment = Enrichment.model_validate(
            {
                "moods": source.get("moods", []),
                "tags": source.get("tags", []),
                "audience": source.get("audience", Audience.ADULT.value),
                "pace": source.get("pace", Pace.MEDIUM.value),
                "era_feel": source.get("era_feel"),
                "one_line_hook": source.get("one_line_hook", ""),
            }
        )
    except ValidationError:
        return None
    return catalog_id, enrichment


def load_jsonl(path: Path) -> list[tuple[str, Enrichment]]:
    records: list[tuple[str, Enrichment]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            parsed = parse_jsonl_line(line)
            if parsed is None:
                if line.strip():
                    log.info("enrich_jsonl_skip", path=str(path), line=line_no)
                continue
            records.append(parsed)
    return records


def write_jsonl(path: Path, rows: Sequence[tuple[str, Enrichment]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for catalog_id, enrichment in rows:
            fh.write(record_to_line(catalog_id, enrichment))
            fh.write("\n")


def estimate_cost(
    title_count: int,
    *,
    tokens_in: int,
    tokens_out: int,
) -> tuple[int, int, float]:
    """Plan §3.3 token envelope times Haiku 4.5 list price."""
    total_in = title_count * tokens_in
    total_out = title_count * tokens_out
    return total_in, total_out, cost_usd(total_in, total_out)


def format_dry_run_line(title_count: int, tokens_in: int, tokens_out: int, usd: float) -> str:
    return (
        f"dry-run titles={title_count} tokens_in={tokens_in} "
        f"tokens_out={tokens_out} cost_usd={usd:.4f}"
    )


def _resolve_mode(mode: EnrichMode | str | None, cfg: Settings) -> EnrichMode:
    if mode is None:
        return cfg.enrich
    if isinstance(mode, EnrichMode):
        return mode
    return EnrichMode(mode.strip().lower())


def build_chain(
    *,
    model: BaseChatModel | None = None,
    settings: Settings | None = None,
) -> Runnable[Any, Enrichment]:
    """Gateway helper with empty-moods fallback. Do not bind tools."""

    def _fallback(_input: Any) -> Enrichment:
        return EMPTY_ENRICHMENT

    return _PROMPT | structured_output(
        Enrichment,
        fallback=RunnableLambda(_fallback),
        model=model,
        settings=settings,
    )


def _card_payload(card: TitleCard) -> dict[str, str]:
    return {
        "mood_ids": ", ".join(item.value for item in MoodId),
        "title": card.title,
        "year": card.year,
        "media_type": card.media_type,
        "genres": card.genres,
        "origins": card.origins,
        "cast_head": card.cast_head,
        "synopsis": card.synopsis,
    }


async def enrich_one(
    card: TitleCard,
    chain: Runnable[Any, Enrichment],
    cost_handler: CostCallbackHandler,
) -> Enrichment:
    config: RunnableConfig = {"callbacks": [cost_handler]}
    try:
        raw = await chain.ainvoke(_card_payload(card), config=config)
    except LLMError:
        log.info("enrich_call_failed", catalog_id=card.catalog_id, reason="gateway")
        return EMPTY_ENRICHMENT
    except Exception:
        log.exception("enrich_call_failed", catalog_id=card.catalog_id, reason="unexpected")
        return EMPTY_ENRICHMENT
    if isinstance(raw, Enrichment):
        return raw
    log.info("enrich_call_failed", catalog_id=card.catalog_id, reason="not_enrichment")
    return EMPTY_ENRICHMENT


async def write_enrichment(db: Database, catalog_id: str, enrichment: Enrichment) -> None:
    # Own transaction per title so a kill keeps every finished write.
    async with db.session() as session:
        await db.titles(session).set_enrichment(catalog_id, enrichment_payload(enrichment))


def _card_from_record(record: TitleRecord, cast_head: str) -> TitleCard:
    title = record.title
    return TitleCard(
        catalog_id=title.catalog_id,
        title=title.title,
        year="" if title.release_year is None else str(title.release_year),
        media_type=title.media_type.value,
        genres=", ".join(genre.value for genre in title.genres),
        origins=", ".join(title.origins),
        cast_head=cast_head,
        synopsis=title.synopsis,
    )


async def load_pending_cards(db: Database, *, limit: int) -> list[TitleCard]:
    async with db.session() as session:
        records = await db.titles(session).list_stored(limit=limit, unenriched_only=True)
        names = await db.credits(session).names_for_titles(
            [record.title.catalog_id for record in records],
            per_title=CAST_HEAD_N,
        )
    return [
        _card_from_record(record, ", ".join(names.get(record.title.catalog_id, ())))
        for record in records
    ]


async def enrich_pending(
    db: Database,
    *,
    limit: int,
    concurrency: int,
    model: BaseChatModel | None,
    settings: Settings,
    cost_handler: CostCallbackHandler,
) -> dict[str, int]:
    cards = await load_pending_cards(db, limit=limit)
    if not cards:
        log.info("enrich_llm_nothing_pending")
        return {"processed": 0, "failed": 0, "skipped": 0}
    chain = build_chain(model=model, settings=settings)
    semaphore = asyncio.Semaphore(concurrency)

    async def _worker(card: TitleCard) -> bool:
        async with semaphore:
            enrichment = await enrich_one(card, chain, cost_handler)
            empty = enrichment == EMPTY_ENRICHMENT
            await write_enrichment(db, card.catalog_id, enrichment)
            log.info(
                "enrich_title_written",
                catalog_id=card.catalog_id,
                empty_moods=not enrichment.moods,
            )
            return empty

    outcomes = await asyncio.gather(*[_worker(card) for card in cards])
    processed = len(cards)
    failed = sum(1 for empty in outcomes if empty)
    log.info("enrich_llm_done", processed=processed, failed=failed)
    return {"processed": processed, "failed": failed, "skipped": 0}


async def import_jsonl(db: Database, path: Path) -> dict[str, int]:
    rows = load_jsonl(path)
    imported = 0
    skipped = 0
    async with db.session() as session:
        repo = db.titles(session)
        for catalog_id, enrichment in rows:
            existing = await repo.get(catalog_id)
            if existing is None:
                skipped += 1
                continue
            await repo.set_enrichment(catalog_id, enrichment_payload(enrichment))
            imported += 1
    log.info("enrich_import_done", path=str(path), imported=imported, skipped=skipped)
    return {"processed": imported, "failed": 0, "skipped": skipped}


async def export_jsonl(db: Database, path: Path) -> int:
    async with db.session() as session:
        records = await db.titles(session).list_stored()
    rows: list[tuple[str, Enrichment]] = []
    for record in records:
        if not record.enrichment:
            continue
        try:
            enrichment = Enrichment.model_validate(record.enrichment)
        except ValidationError:
            continue
        rows.append((record.title.catalog_id, enrichment))
    write_jsonl(path, rows)
    log.info("enrich_export_done", path=str(path), exported=len(rows))
    return len(rows)


def _needs_db(mode: EnrichMode, dry_run: bool, export_path: Path | None) -> bool:
    if export_path is not None:
        return True
    if dry_run:
        return False
    return mode is not EnrichMode.NONE


async def run_async(
    *,
    limit: int | None = None,
    dry_run: bool = False,
    mode: EnrichMode | str | None = None,
    artifact: Path | None = None,
    export_path: Path | None = None,
    db: Database | None = None,
    model: BaseChatModel | None = None,
    settings: Settings | None = None,
    cost_handler: CostCallbackHandler | None = None,
) -> dict[str, int | float]:
    cfg = settings if settings is not None else default_settings
    resolved_mode = _resolve_mode(mode, cfg)
    resolved_limit = cfg.enrich_limit if limit is None else limit
    artifact_path = artifact if artifact is not None else default_artifact_path()
    handler = cost_handler if cost_handler is not None else CostCallbackHandler()

    if dry_run:
        tokens_in, tokens_out, usd = estimate_cost(
            resolved_limit,
            tokens_in=cfg.enrich_tokens_in,
            tokens_out=cfg.enrich_tokens_out,
        )
        line = format_dry_run_line(resolved_limit, tokens_in, tokens_out, usd)
        typer.echo(line)
        log.info(
            "enrich_dry_run",
            titles=resolved_limit,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=usd,
        )
        return {
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": usd,
        }

    own_db = False
    database = db
    if database is None and _needs_db(resolved_mode, dry_run, export_path):
        database = Database.from_settings(cfg)
        own_db = True

    counts: dict[str, int | float] = {
        "processed": 0,
        "failed": 0,
        "skipped": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0.0,
    }
    try:
        if resolved_mode is EnrichMode.NONE and export_path is None:
            log.info("enrich_skipped", reason="ENRICH=none")
            return counts
        if resolved_mode is EnrichMode.SKIP:
            if database is None:
                msg = "enrich import requires a database"
                raise RuntimeError(msg)
            imported = await import_jsonl(database, artifact_path)
            counts.update(imported)
        elif resolved_mode is EnrichMode.LLM:
            if database is None:
                msg = "enrich llm requires a database"
                raise RuntimeError(msg)
            llm_counts = await enrich_pending(
                database,
                limit=resolved_limit,
                concurrency=cfg.enrich_concurrency,
                model=model,
                settings=cfg,
                cost_handler=handler,
            )
            counts.update(llm_counts)
            counts["tokens_in"] = handler.tokens_in
            counts["tokens_out"] = handler.tokens_out
            counts["cost_usd"] = handler.cost_usd
            log.info(
                "enrich_cost",
                tokens_in=handler.tokens_in,
                tokens_out=handler.tokens_out,
                cost_usd=handler.cost_usd,
            )
        if export_path is not None:
            if database is None:
                msg = "enrich export requires a database"
                raise RuntimeError(msg)
            counts["exported"] = await export_jsonl(database, export_path)
        return counts
    finally:
        if own_db and database is not None:
            await database.dispose()


def run(
    *,
    limit: int | None = None,
    dry_run: bool = False,
    mode: EnrichMode | str | None = None,
    artifact: Path | None = None,
    export_path: Path | None = None,
    db: Database | None = None,
    model: BaseChatModel | None = None,
    settings: Settings | None = None,
    cost_handler: CostCallbackHandler | None = None,
) -> dict[str, int | float]:
    """Seed-all entry: default imports the committed JSONL (no API key)."""
    return asyncio.run(
        run_async(
            limit=limit,
            dry_run=dry_run,
            mode=mode,
            artifact=artifact,
            export_path=export_path,
            db=db,
            model=model,
            settings=settings,
            cost_handler=cost_handler,
        )
    )


def register(app: typer.Typer) -> None:
    @app.command("enrich")
    def enrich_cmd(
        limit: Annotated[
            int | None,
            typer.Option("--limit", help="Max titles for llm mode (default ENRICH_LIMIT)"),
        ] = None,
        dry_run: Annotated[
            bool,
            typer.Option("--dry-run", help="Print token and USD estimate; do not call the model"),
        ] = False,
        mode: Annotated[
            str | None,
            typer.Option("--mode", help="skip (import JSONL) | llm | none"),
        ] = None,
        from_artifact: Annotated[
            Path | None,
            typer.Option(
                "--from-artifact",
                help="JSONL to import (default data/enriched/titles.jsonl)",
            ),
        ] = None,
        export: Annotated[
            Path | None,
            typer.Option("--export", help="Write stored enrichments as JSONL"),
        ] = None,
    ) -> None:
        result = run(
            limit=limit,
            dry_run=dry_run,
            mode=mode,
            artifact=from_artifact,
            export_path=export,
        )
        log.info("cli_enrich_done", **result)
