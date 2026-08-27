"""Embed catalog rows and bulk-index them into versioned Elasticsearch indices.

Creates a fresh `titles_vN` / `people_vN`, fills them, then points the aliases
with T06's atomic `swap_alias`. An unaliased latest version is resumed instead
of duplicated. The previous live index is never deleted.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any

import typer
from elasticsearch import AsyncElasticsearch

from assist.config import settings
from assist.domain.catalog import Person, Title
from assist.obs.logging import get_logger
from assist.stores.db import CreditRecord, Database, TitleRecord
from assist.stores.embed_client import EmbedClient
from assist.stores.es import (
    PEOPLE_ALIAS,
    TITLES_ALIAS,
    close_client,
    create_client,
    create_next_index,
    indices_for_alias,
    swap_alias,
    versioned_index_names,
)

log = get_logger("assist.jobs.index")

type JsonScalar = int | str | bool | list[str]


@dataclass(frozen=True)
class IndexResult:
    titles_index: str
    people_index: str
    titles_indexed: int
    people_indexed: int
    titles_skipped: int
    people_skipped: int
    swapped: bool
    titles_previous: tuple[str, ...]
    people_previous: tuple[str, ...]

    def as_dict(self) -> dict[str, JsonScalar]:
        return {
            "titles_index": self.titles_index,
            "people_index": self.people_index,
            "titles_indexed": self.titles_indexed,
            "people_indexed": self.people_indexed,
            "titles_skipped": self.titles_skipped,
            "people_skipped": self.people_skipped,
            "swapped": self.swapped,
            "titles_previous": list(self.titles_previous),
            "people_previous": list(self.people_previous),
        }


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _optional_str(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def build_embedding_text(
    title: Title,
    *,
    tags: list[str],
    people_names: list[str],
    era_feel: str | None,
) -> str:
    """Plain text sent to the embedder. Empty parts are dropped, not as 'None'."""
    parts: list[str] = []
    if title.title:
        parts.append(title.title)
    if title.synopsis:
        parts.append(title.synopsis)
    if tags:
        parts.append(" ".join(tags))
    if people_names:
        parts.append(" ".join(people_names))
    if era_feel:
        parts.append(era_feel)
    return "\n".join(parts)


def title_document(
    record: TitleRecord,
    *,
    people_ids: list[str],
    people_names: list[str],
    embedding: list[float],
) -> dict[str, Any]:
    title = record.title
    enrichment = record.enrichment or {}
    tags = _str_list(enrichment.get("tags"))
    era_feel = _optional_str(enrichment.get("era_feel")) or ""
    doc: dict[str, Any] = {
        "catalog_id": title.catalog_id,
        "media_type": title.media_type.value,
        "genres": [g.value for g in title.genres],
        "moods": [m.value for m in title.moods],
        "origins": list(title.origins),
        "maturity_rank": title.maturity_rank,
        "local_original": title.local_original,
        "release_year": title.release_year,
        "runtime_min": title.runtime_min,
        "people_ids": people_ids,
        "pop_28d": title.pop_28d,
        "title": title.title,
        "synopsis": title.synopsis,
        "tags": tags,
        "people_names": people_names,
        "era_feel": era_feel,
        "embedding": embedding,
    }
    audience = _optional_str(enrichment.get("audience"))
    pace = _optional_str(enrichment.get("pace"))
    if audience is not None:
        doc["audience"] = audience
    if pace is not None:
        doc["pace"] = pace
    return doc


def person_document(person: Person) -> dict[str, Any]:
    return {
        "person_id": person.person_id,
        "name": person.name,
        "name_norm": person.name_norm,
        "roles": [r.value for r in person.roles],
        "active_year_min": person.active_year_min,
        "active_year_max": person.active_year_max,
        "popularity": person.popularity,
        "credit_count": person.credit_count,
    }


def _es_body(resp: object) -> dict[str, Any]:
    body = getattr(resp, "body", resp)
    if isinstance(body, Mapping):
        return dict(body)
    msg = f"unexpected ES response type: {type(resp)!r}"
    raise TypeError(msg)


async def resolve_target_index(
    client: AsyncElasticsearch,
    alias: str,
    *,
    resume: bool = True,
    body: Mapping[str, Any] | None = None,
) -> str:
    """Reuse an unaliased latest `alias_vN`, otherwise create the next version.

    Live aliased indices are left untouched so a failed fill can still roll back.
    `body` is required when `alias` is not the production titles/people name
    (T06's create_next_index only defaults those two).
    """
    live = set(await indices_for_alias(client, alias))
    versions = await versioned_index_names(client, alias)
    if resume and versions:
        latest = versions[-1]
        if latest not in live:
            log.info("index.resume", alias=alias, index=latest)
            return latest
    name = await create_next_index(client, alias, body=body)
    log.info("index.target_created", alias=alias, index=name)
    return name


async def _count_docs(client: AsyncElasticsearch, index: str) -> int:
    resp = await client.count(index=index)
    return int(_es_body(resp)["count"])


async def _existing_ids(
    client: AsyncElasticsearch,
    index: str,
    ids: list[str],
) -> set[str]:
    if not ids:
        return set()
    resp = await client.mget(index=index, ids=ids, source=False)
    docs = _es_body(resp).get("docs", [])
    found: set[str] = set()
    if not isinstance(docs, list):
        return found
    for doc in docs:
        if isinstance(doc, dict) and doc.get("found") is True:
            doc_id = doc.get("_id")
            if isinstance(doc_id, str):
                found.add(doc_id)
    return found


async def _bulk_index(
    client: AsyncElasticsearch,
    index: str,
    docs: list[tuple[str, dict[str, Any]]],
) -> None:
    if not docs:
        return
    operations: list[dict[str, Any]] = []
    for doc_id, source in docs:
        operations.append({"index": {"_index": index, "_id": doc_id}})
        operations.append(source)
    resp = await client.bulk(operations=operations)
    body = _es_body(resp)
    if not body.get("errors"):
        return
    failed: list[dict[str, Any]] = []
    items = body.get("items", [])
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            for part in item.values():
                if isinstance(part, dict) and part.get("error"):
                    failed.append(part)
    log.error("index.bulk_failed", index=index, failed=len(failed), sample=failed[:3])
    msg = f"bulk index into {index} failed for {len(failed)} docs"
    raise RuntimeError(msg)


def _credits_by_title(credits: list[CreditRecord]) -> dict[str, list[CreditRecord]]:
    grouped: dict[str, list[CreditRecord]] = {}
    for credit in credits:
        grouped.setdefault(credit.catalog_id, []).append(credit)
    return grouped


async def _index_people(
    db: Database,
    client: AsyncElasticsearch,
    index: str,
    *,
    batch_size: int,
) -> tuple[int, int]:
    indexed = 0
    skipped = 0
    after_id: str | None = None
    while True:
        async with db.session() as session:
            batch = await db.people(session).scan(after_id=after_id, limit=batch_size)
        if not batch:
            break
        ids = [person.person_id for person in batch]
        already = await _existing_ids(client, index, ids)
        to_write = [person for person in batch if person.person_id not in already]
        skipped += len(batch) - len(to_write)
        await _bulk_index(
            client,
            index,
            [(person.person_id, person_document(person)) for person in to_write],
        )
        indexed += len(to_write)
        after_id = batch[-1].person_id
        log.info(
            "index.people.batch",
            index=index,
            done=indexed + skipped,
            indexed=indexed,
            skipped=skipped,
        )
    return indexed, skipped


async def _index_titles(
    db: Database,
    client: AsyncElasticsearch,
    embedder: EmbedClient,
    index: str,
    *,
    batch_size: int,
    limit: int | None,
) -> tuple[int, int]:
    indexed = 0
    skipped = 0
    after_id: str | None = None
    remaining = limit
    while remaining is None or remaining > 0:
        page_size = batch_size if remaining is None else min(batch_size, remaining)
        async with db.session() as session:
            records = await db.titles(session).scan(after_id=after_id, limit=page_size)
            if not records:
                break
            catalog_ids = [record.title.catalog_id for record in records]
            credits = await db.credits(session).list_for_titles(catalog_ids)
            person_ids = list({credit.person_id for credit in credits})
            people = await db.people(session).get_many(person_ids)
        names_by_id = {person.person_id: person.name for person in people}
        grouped = _credits_by_title(credits)
        already = await _existing_ids(client, index, catalog_ids)
        pending = [record for record in records if record.title.catalog_id not in already]
        skipped += len(records) - len(pending)

        texts: list[str] = []
        pending_meta: list[tuple[TitleRecord, list[str], list[str], list[str], str | None]] = []
        for record in pending:
            enrichment = record.enrichment or {}
            tags = _str_list(enrichment.get("tags"))
            era_feel = _optional_str(enrichment.get("era_feel"))
            title_credits = grouped.get(record.title.catalog_id, [])
            people_ids = [credit.person_id for credit in title_credits]
            people_names = [
                names_by_id[credit.person_id]
                for credit in title_credits
                if credit.person_id in names_by_id
            ]
            texts.append(
                build_embedding_text(
                    record.title,
                    tags=tags,
                    people_names=people_names,
                    era_feel=era_feel,
                )
            )
            pending_meta.append((record, people_ids, people_names, tags, era_feel))

        if pending_meta:
            vectors = await embedder.embed(texts)
            if len(vectors) != len(pending_meta):
                msg = f"embedder returned {len(vectors)} vectors for {len(pending_meta)} titles"
                raise RuntimeError(msg)
            docs = [
                (
                    record.title.catalog_id,
                    title_document(
                        record,
                        people_ids=people_ids,
                        people_names=people_names,
                        embedding=vector,
                    ),
                )
                for (record, people_ids, people_names, _tags, _era), vector in zip(
                    pending_meta, vectors, strict=True
                )
            ]
            await _bulk_index(client, index, docs)
            indexed_ids = [doc_id for doc_id, _source in docs]
            now = datetime.now(UTC)
            async with db.session() as session:
                await db.titles(session).mark_indexed(indexed_ids, now)
            indexed += len(docs)

        after_id = records[-1].title.catalog_id
        if remaining is not None:
            remaining -= len(records)
        log.info(
            "index.titles.batch",
            index=index,
            done=indexed + skipped,
            indexed=indexed,
            skipped=skipped,
        )
    return indexed, skipped


async def index_catalog(
    *,
    db: Database,
    es: AsyncElasticsearch,
    embedder: EmbedClient,
    titles_alias: str = TITLES_ALIAS,
    people_alias: str = PEOPLE_ALIAS,
    titles_body: Mapping[str, Any] | None = None,
    people_body: Mapping[str, Any] | None = None,
    limit: int | None = None,
    resume: bool = True,
    batch_size: int | None = None,
) -> IndexResult:
    page = batch_size if batch_size is not None else settings.index_batch_size
    async with db.session() as session:
        titles_total = await db.titles(session).count()
        people_total = await db.people(session).count()
    expected_titles = titles_total if limit is None else min(limit, titles_total)

    titles_index = await resolve_target_index(es, titles_alias, resume=resume, body=titles_body)
    people_index = await resolve_target_index(es, people_alias, resume=resume, body=people_body)
    log.info(
        "index.start",
        titles_index=titles_index,
        people_index=people_index,
        titles_total=titles_total,
        people_total=people_total,
        expected_titles=expected_titles,
        batch_size=page,
        resume=resume,
    )

    people_indexed, people_skipped = await _index_people(db, es, people_index, batch_size=page)
    titles_indexed, titles_skipped = await _index_titles(
        db, es, embedder, titles_index, batch_size=page, limit=limit
    )

    await es.indices.refresh(index=titles_index)
    await es.indices.refresh(index=people_index)
    titles_count = await _count_docs(es, titles_index)
    people_count = await _count_docs(es, people_index)
    complete = titles_count == expected_titles and people_count == people_total
    titles_previous: tuple[str, ...] = ()
    people_previous: tuple[str, ...] = ()
    if complete:
        # Swap only after both sides are full so a crash leaves both unaliased
        # and the next run resumes instead of building a third version.
        titles_previous = await swap_alias(es, titles_alias, titles_index)
        people_previous = await swap_alias(es, people_alias, people_index)
        log.info(
            "index.swapped",
            titles_index=titles_index,
            people_index=people_index,
            titles_previous=list(titles_previous),
            people_previous=list(people_previous),
        )
    else:
        log.warning(
            "index.incomplete",
            titles_index=titles_index,
            people_index=people_index,
            titles_count=titles_count,
            expected_titles=expected_titles,
            people_count=people_count,
            expected_people=people_total,
        )

    result = IndexResult(
        titles_index=titles_index,
        people_index=people_index,
        titles_indexed=titles_indexed,
        people_indexed=people_indexed,
        titles_skipped=titles_skipped,
        people_skipped=people_skipped,
        swapped=complete,
        titles_previous=titles_previous,
        people_previous=people_previous,
    )
    log.info("index.done", **result.as_dict())
    return result


async def _run_owned(
    *,
    limit: int | None,
    resume: bool,
    batch_size: int | None,
) -> dict[str, JsonScalar]:
    db = Database.from_settings()
    es = create_client()
    embedder = EmbedClient()
    try:
        result = await index_catalog(
            db=db,
            es=es,
            embedder=embedder,
            limit=limit,
            resume=resume,
            batch_size=batch_size,
        )
        return result.as_dict()
    finally:
        await embedder.aclose()
        await close_client(es)
        await db.dispose()


def run(
    *,
    limit: int | None = None,
    resume: bool = True,
    batch_size: int | None = None,
) -> dict[str, JsonScalar]:
    """Entry point for `jobs index` and `seed-all`. Caller owns nothing."""
    return asyncio.run(_run_owned(limit=limit, resume=resume, batch_size=batch_size))


def register(app: typer.Typer) -> None:
    @app.command("index")
    def index_cmd(
        limit: Annotated[
            int | None,
            typer.Option("--limit", help="Index at most this many titles"),
        ] = None,
        resume: Annotated[
            bool,
            typer.Option(
                "--resume/--no-resume",
                help="Reuse an unaliased in-progress versioned index",
            ),
        ] = True,
        batch_size: Annotated[
            int | None,
            typer.Option("--batch-size", help="Titles per embed/bulk page"),
        ] = None,
    ) -> None:
        result = run(limit=limit, resume=resume, batch_size=batch_size)
        log.info("cli_index_done", **result)
