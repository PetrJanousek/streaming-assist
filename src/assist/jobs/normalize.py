"""CSV -> Postgres catalog: taxonomy map, people/credits, synthesized fixtures.

Availability and pop_28d are seeded from `catalog_id` via SHA-256. Python's
`hash()` is process-salted (PYTHONHASHSEED) and must not be used here — a
fixed id has to produce the same windows and popularity across processes.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from assist.config import settings
from assist.domain.catalog import Person, Title
from assist.domain.enums import (
    CreditRole,
    GenreId,
    MaturityRating,
    MediaType,
    Package,
    maturity_rank,
)
from assist.jobs.fetch import data_dir, raw_csv_path, taxonomy_dir
from assist.obs.logging import configure_logging, get_logger
from assist.stores.db import (
    AvailabilityRow,
    AvailabilityWindow,
    CreditRecord,
    CreditRow,
    Database,
    PersonRow,
    TaxonomyEntry,
    TitleRow,
)

log = get_logger("assist.jobs.normalize")

# Rights-territory keys for the synthetic availability fixture. Distinct from
# HOME_COUNTRY (CSV origin names). "US" matches the seeded profile geo.
AVAIL_HOME_GEO = "US"
AVAIL_OTHER_GEO = "XX"

PLAYABLE_SHARE = 0.85
EXPIRED_SHARE = 0.07
PACKAGE_GATED_SHARE = 0.05
# Remaining 0.03 is geo-restricted.

LOCAL_ORIGINAL_SHARE = 0.20

WINDOW_OPEN_START = datetime(2000, 1, 1, tzinfo=UTC)
WINDOW_OPEN_END = datetime(2099, 12, 31, 23, 59, 59, tzinfo=UTC)
WINDOW_EXPIRED_START = datetime(2000, 1, 1, tzinfo=UTC)
WINDOW_EXPIRED_END = datetime(2015, 6, 1, tzinfo=UTC)

_BATCH = 500
_DURATION_LEAK = re.compile(r"^(\d+)\s*(min|seasons?)$", re.IGNORECASE)
_DURATION_FIELD = re.compile(r"^(\d+)\s+(min|Season|Seasons)$")
_TYPE_MAP: Mapping[str, MediaType] = {
    "Movie": MediaType.FILM,
    "TV Show": MediaType.SERIES,
}


class AvailabilityBucket(StrEnum):
    PLAYABLE = "playable"
    EXPIRED = "expired"
    PACKAGE_GATED = "package_gated"
    GEO_RESTRICTED = "geo_restricted"


@dataclass(frozen=True)
class QuarantineRecord:
    catalog_id: str
    title: str
    raw_rating: str
    reason: str
    recovered_duration: str | None = None


@dataclass
class _PersonAccum:
    name: str
    name_norm: str
    roles: set[CreditRole] = field(default_factory=set)
    years: list[int] = field(default_factory=list)
    pops: list[float] = field(default_factory=list)
    credit_count: int = 0


@dataclass(frozen=True)
class NormalizedCatalog:
    titles: tuple[Title, ...]
    people: tuple[Person, ...]
    credits: tuple[CreditRecord, ...]
    availability: tuple[AvailabilityWindow, ...]
    quarantine: tuple[QuarantineRecord, ...]
    taxonomy: tuple[TaxonomyEntry, ...]


def _stable_unit(catalog_id: str, purpose: str) -> float:
    """Uniform [0, 1) from catalog_id. Independent of PYTHONHASHSEED and time."""
    digest = hashlib.sha256(f"{purpose}:{catalog_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def person_id_for(name_norm: str) -> str:
    digest = hashlib.sha256(name_norm.encode()).hexdigest()[:16]
    return f"p_{digest}"


def normalize_person_name(name: str) -> str:
    return " ".join(name.lower().split())


def availability_bucket(catalog_id: str) -> AvailabilityBucket:
    unit = _stable_unit(catalog_id, "avail")
    if unit < PLAYABLE_SHARE:
        return AvailabilityBucket.PLAYABLE
    if unit < PLAYABLE_SHARE + EXPIRED_SHARE:
        return AvailabilityBucket.EXPIRED
    if unit < PLAYABLE_SHARE + EXPIRED_SHARE + PACKAGE_GATED_SHARE:
        return AvailabilityBucket.PACKAGE_GATED
    return AvailabilityBucket.GEO_RESTRICTED


def synthesize_pop_28d(
    catalog_id: str,
    release_year: int | None,
    genres: tuple[GenreId, ...],
) -> float:
    """Skewed [0, 1] popularity: catalog_id base, recency, small genre prior."""
    base = _stable_unit(catalog_id, "pop") ** 2
    year = release_year if release_year is not None else 1970
    recency = min(1.0, max(0.0, (year - 1925) / 96.0))
    prior = 0.0
    if any(g in genres for g in (GenreId.ACTION, GenreId.COMEDY, GenreId.SCIFI)):
        prior = 0.15
    elif GenreId.DRAMA in genres:
        prior = 0.08
    pop = (0.65 * base) + (0.25 * recency) + (0.10 * prior)
    return round(min(1.0, max(0.0, pop)), 6)


def synthesize_local_original(
    catalog_id: str,
    origins: tuple[str, ...],
    home_country: str,
) -> bool:
    if home_country not in origins:
        return False
    return _stable_unit(catalog_id, "local") < LOCAL_ORIGINAL_SHARE


def synthesize_availability(catalog_id: str) -> list[AvailabilityWindow]:
    bucket = availability_bucket(catalog_id)
    windows: list[AvailabilityWindow] = []
    if bucket is AvailabilityBucket.PLAYABLE:
        for package in (Package.BASIC, Package.PREMIUM):
            windows.append(
                AvailabilityWindow(
                    catalog_id=catalog_id,
                    package=package,
                    geo=AVAIL_HOME_GEO,
                    window_start=WINDOW_OPEN_START,
                    window_end=WINDOW_OPEN_END,
                    playable=True,
                )
            )
    elif bucket is AvailabilityBucket.EXPIRED:
        for package in (Package.BASIC, Package.PREMIUM):
            windows.append(
                AvailabilityWindow(
                    catalog_id=catalog_id,
                    package=package,
                    geo=AVAIL_HOME_GEO,
                    window_start=WINDOW_EXPIRED_START,
                    window_end=WINDOW_EXPIRED_END,
                    playable=False,
                )
            )
    elif bucket is AvailabilityBucket.PACKAGE_GATED:
        windows.append(
            AvailabilityWindow(
                catalog_id=catalog_id,
                package=Package.BASIC,
                geo=AVAIL_HOME_GEO,
                window_start=WINDOW_OPEN_START,
                window_end=WINDOW_OPEN_END,
                playable=False,
            )
        )
        windows.append(
            AvailabilityWindow(
                catalog_id=catalog_id,
                package=Package.PREMIUM,
                geo=AVAIL_HOME_GEO,
                window_start=WINDOW_OPEN_START,
                window_end=WINDOW_OPEN_END,
                playable=True,
            )
        )
    else:
        for package in (Package.BASIC, Package.PREMIUM):
            windows.append(
                AvailabilityWindow(
                    catalog_id=catalog_id,
                    package=package,
                    geo=AVAIL_HOME_GEO,
                    window_start=WINDOW_OPEN_START,
                    window_end=WINDOW_OPEN_END,
                    playable=False,
                )
            )
            windows.append(
                AvailabilityWindow(
                    catalog_id=catalog_id,
                    package=package,
                    geo=AVAIL_OTHER_GEO,
                    window_start=WINDOW_OPEN_START,
                    window_end=WINDOW_OPEN_END,
                    playable=True,
                )
            )
    return windows


def load_genre_map(path: Path | None = None) -> dict[str, GenreId | None]:
    map_path = path if path is not None else taxonomy_dir() / "genre_map.json"
    raw = json.loads(map_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"genre map must be a JSON object: {map_path}"
        raise ValueError(msg)
    out: dict[str, GenreId | None] = {}
    for label, target in raw.items():
        if not isinstance(label, str):
            msg = f"genre map keys must be strings: {label!r}"
            raise ValueError(msg)
        if target is None:
            out[label] = None
        elif isinstance(target, str):
            out[label] = GenreId(target)
        else:
            msg = f"genre map value for {label!r} must be a string or null"
            raise ValueError(msg)
    return out


def load_maturity_map(path: Path | None = None) -> dict[str, MaturityRating]:
    map_path = path if path is not None else taxonomy_dir() / "maturity_map.json"
    raw = json.loads(map_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"maturity map must be a JSON object: {map_path}"
        raise ValueError(msg)
    out: dict[str, MaturityRating] = {}
    for label, target in raw.items():
        if not isinstance(label, str) or not isinstance(target, str):
            msg = f"maturity map entries must be strings: {label!r} -> {target!r}"
            raise ValueError(msg)
        out[label] = MaturityRating(target)
    return out


def map_genres(listed_in: str, genre_map: Mapping[str, GenreId | None]) -> tuple[GenreId, ...]:
    seen: list[GenreId] = []
    for raw in listed_in.split(","):
        label = raw.strip()
        if not label:
            continue
        if label not in genre_map:
            log.warning("unmapped_genre_label", label=label)
            continue
        genre = genre_map[label]
        if genre is None:
            continue
        if genre not in seen:
            seen.append(genre)
    return tuple(seen)


def split_people(raw: str) -> tuple[str, ...]:
    names: list[str] = []
    for part in raw.split(","):
        name = " ".join(part.split())
        if name:
            names.append(name)
    return tuple(names)


def parse_origins(raw: str) -> tuple[str, ...]:
    origins: list[str] = []
    for part in raw.split(","):
        country = " ".join(part.split())
        if country and country not in origins:
            origins.append(country)
    return tuple(origins)


def _parse_duration_token(token: str, media_type: MediaType) -> tuple[int | None, int | None]:
    match = _DURATION_FIELD.match(token) or _DURATION_LEAK.match(token)
    if match is None:
        return None, None
    amount = int(match.group(1))
    unit = match.group(2).lower()
    if unit == "min":
        return amount, None
    if media_type is MediaType.SERIES or unit.startswith("season"):
        return None, amount
    return amount, None


def _taxonomy_from_genre_map(genre_map: Mapping[str, GenreId | None]) -> tuple[TaxonomyEntry, ...]:
    by_id: dict[GenreId, list[str]] = {}
    for label, genre in genre_map.items():
        if genre is None:
            continue
        by_id.setdefault(genre, []).append(label)
    entries: list[TaxonomyEntry] = []
    for genre in GenreId:
        synonyms = tuple(by_id.get(genre, ()))
        entries.append(
            TaxonomyEntry(
                kind="genres",
                id=genre.value,
                label=genre.value.replace("_", " "),
                synonyms=synonyms,
            )
        )
    return tuple(entries)


def normalize_csv(
    source: Path,
    *,
    home_country: str | None = None,
    genre_map: Mapping[str, GenreId | None] | None = None,
    maturity_map: Mapping[str, MaturityRating] | None = None,
) -> NormalizedCatalog:
    resolved_home = home_country if home_country is not None else settings.home_country
    genres = dict(genre_map) if genre_map is not None else load_genre_map()
    ratings = dict(maturity_map) if maturity_map is not None else load_maturity_map()

    titles: list[Title] = []
    quarantine: list[QuarantineRecord] = []
    credits: list[CreditRecord] = []
    availability: list[AvailabilityWindow] = []
    people_acc: dict[str, _PersonAccum] = {}
    credit_keys: set[tuple[str, str, str]] = set()

    with source.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            catalog_id = (row.get("show_id") or "").strip()
            title_text = (row.get("title") or "").strip()
            if not catalog_id or not title_text:
                log.warning(
                    "skipped_row_missing_id_or_title",
                    catalog_id=catalog_id,
                    title=title_text,
                )
                continue
            media_type = _TYPE_MAP.get((row.get("type") or "").strip())
            if media_type is None:
                log.warning("skipped_row_unknown_type", catalog_id=catalog_id, type=row.get("type"))
                continue

            raw_rating = (row.get("rating") or "").strip()
            raw_duration = (row.get("duration") or "").strip()
            recovered_duration: str | None = None
            rating = ratings.get(raw_rating)
            leak = _DURATION_LEAK.match(raw_rating)
            if leak is not None:
                recovered_duration = raw_rating
                reason = "duration_in_rating"
                quarantine.append(
                    QuarantineRecord(
                        catalog_id=catalog_id,
                        title=title_text,
                        raw_rating=raw_rating,
                        reason=reason,
                        recovered_duration=recovered_duration,
                    )
                )
                log.warning(
                    "rating_quarantined",
                    catalog_id=catalog_id,
                    title=title_text,
                    raw_rating=raw_rating,
                    reason=reason,
                    recovered_duration=recovered_duration,
                )
                rating = MaturityRating.NR
            elif rating is None:
                reason = "empty_rating" if not raw_rating else "unmapped_rating"
                quarantine.append(
                    QuarantineRecord(
                        catalog_id=catalog_id,
                        title=title_text,
                        raw_rating=raw_rating,
                        reason=reason,
                    )
                )
                log.warning(
                    "rating_quarantined",
                    catalog_id=catalog_id,
                    title=title_text,
                    raw_rating=raw_rating,
                    reason=reason,
                )
                rating = MaturityRating.NR

            duration_token = raw_duration or recovered_duration or ""
            runtime_min, seasons = _parse_duration_token(duration_token, media_type)
            year_raw = (row.get("release_year") or "").strip()
            release_year = int(year_raw) if year_raw.isdigit() else None
            origins = parse_origins(row.get("country") or "")
            title_genres = map_genres(row.get("listed_in") or "", genres)
            pop_28d = synthesize_pop_28d(catalog_id, release_year, title_genres)
            title = Title(
                catalog_id=catalog_id,
                media_type=media_type,
                title=title_text,
                synopsis=(row.get("description") or "").strip(),
                release_year=release_year,
                runtime_min=runtime_min,
                seasons=seasons,
                maturity_rank=maturity_rank(rating),
                origins=origins,
                genres=title_genres,
                local_original=synthesize_local_original(catalog_id, origins, resolved_home),
                pop_28d=pop_28d,
            )
            titles.append(title)
            availability.extend(synthesize_availability(catalog_id))

            for raw_name, role in (
                *((name, CreditRole.DIRECTOR) for name in split_people(row.get("director") or "")),
                *((name, CreditRole.ACTOR) for name in split_people(row.get("cast") or "")),
            ):
                name_norm = normalize_person_name(raw_name)
                pid = person_id_for(name_norm)
                credit_key = (catalog_id, pid, role.value)
                if credit_key in credit_keys:
                    continue
                credit_keys.add(credit_key)
                acc = people_acc.get(pid)
                if acc is None:
                    acc = _PersonAccum(name=raw_name, name_norm=name_norm)
                    people_acc[pid] = acc
                acc.roles.add(role)
                acc.credit_count += 1
                if release_year is not None:
                    acc.years.append(release_year)
                acc.pops.append(pop_28d)
                credits.append(CreditRecord(catalog_id=catalog_id, person_id=pid, role=role))

    people: list[Person] = []
    for pid, acc in people_acc.items():
        mean_pop = sum(acc.pops) / len(acc.pops) if acc.pops else 0.0
        people.append(
            Person(
                person_id=pid,
                name=acc.name,
                name_norm=acc.name_norm,
                roles=tuple(sorted(acc.roles, key=lambda r: r.value)),
                credit_count=acc.credit_count,
                active_year_min=min(acc.years) if acc.years else None,
                active_year_max=max(acc.years) if acc.years else None,
                popularity=round(acc.credit_count * mean_pop, 6),
            )
        )

    return NormalizedCatalog(
        titles=tuple(titles),
        people=tuple(people),
        credits=tuple(credits),
        availability=tuple(availability),
        quarantine=tuple(quarantine),
        taxonomy=_taxonomy_from_genre_map(genres),
    )


def _chunks[T](items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


async def _upsert_titles(session: AsyncSession, titles: Sequence[Title]) -> None:
    if not titles:
        return
    values = [
        {
            "catalog_id": t.catalog_id,
            "media_type": t.media_type.value,
            "title": t.title,
            "synopsis": t.synopsis,
            "release_year": t.release_year,
            "runtime_min": t.runtime_min,
            "seasons": t.seasons,
            "maturity_rank": t.maturity_rank,
            "origins": list(t.origins),
            "genres": [g.value for g in t.genres],
            "local_original": t.local_original,
            "pop_28d": t.pop_28d,
            "enrichment": None,
            "indexed_at": None,
        }
        for t in titles
    ]
    stmt = pg_insert(TitleRow).values(values)
    # Omit enrichment / indexed_at so a re-run cannot wipe T11's JSON.
    stmt = stmt.on_conflict_do_update(
        index_elements=["catalog_id"],
        set_={
            "media_type": stmt.excluded.media_type,
            "title": stmt.excluded.title,
            "synopsis": stmt.excluded.synopsis,
            "release_year": stmt.excluded.release_year,
            "runtime_min": stmt.excluded.runtime_min,
            "seasons": stmt.excluded.seasons,
            "maturity_rank": stmt.excluded.maturity_rank,
            "origins": stmt.excluded.origins,
            "genres": stmt.excluded.genres,
            "local_original": stmt.excluded.local_original,
            "pop_28d": stmt.excluded.pop_28d,
        },
    )
    await session.execute(stmt)


async def _upsert_people(session: AsyncSession, people: Sequence[Person]) -> None:
    if not people:
        return
    values = [
        {
            "person_id": p.person_id,
            "name": p.name,
            "name_norm": p.name_norm,
            "roles": [r.value for r in p.roles],
            "credit_count": p.credit_count,
            "active_year_min": p.active_year_min,
            "active_year_max": p.active_year_max,
            "popularity": p.popularity,
        }
        for p in people
    ]
    stmt = pg_insert(PersonRow).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["person_id"],
        set_={
            "name": stmt.excluded.name,
            "name_norm": stmt.excluded.name_norm,
            "roles": stmt.excluded.roles,
            "credit_count": stmt.excluded.credit_count,
            "active_year_min": stmt.excluded.active_year_min,
            "active_year_max": stmt.excluded.active_year_max,
            "popularity": stmt.excluded.popularity,
        },
    )
    await session.execute(stmt)


async def _upsert_credits(session: AsyncSession, credits: Sequence[CreditRecord]) -> None:
    if not credits:
        return
    values = [
        {
            "catalog_id": c.catalog_id,
            "person_id": c.person_id,
            "role": c.role.value,
        }
        for c in credits
    ]
    stmt = (
        pg_insert(CreditRow)
        .values(values)
        .on_conflict_do_nothing(index_elements=["catalog_id", "person_id", "role"])
    )
    await session.execute(stmt)


async def _upsert_availability(
    session: AsyncSession, windows: Sequence[AvailabilityWindow]
) -> None:
    if not windows:
        return
    values = [
        {
            "catalog_id": w.catalog_id,
            "package": w.package.value,
            "geo": w.geo,
            "window_start": w.window_start,
            "window_end": w.window_end,
            "playable": w.playable,
        }
        for w in windows
    ]
    stmt = pg_insert(AvailabilityRow).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["catalog_id", "package", "geo"],
        set_={
            "window_start": stmt.excluded.window_start,
            "window_end": stmt.excluded.window_end,
            "playable": stmt.excluded.playable,
        },
    )
    await session.execute(stmt)


async def persist_catalog(db: Database, catalog: NormalizedCatalog) -> None:
    async with db.session() as session:
        tax = db.taxonomy(session)
        for entry in catalog.taxonomy:
            await tax.upsert(entry)
        for titles_chunk in _chunks(catalog.titles, _BATCH):
            await _upsert_titles(session, titles_chunk)
        for people_chunk in _chunks(catalog.people, _BATCH):
            await _upsert_people(session, people_chunk)
        for credits_chunk in _chunks(catalog.credits, _BATCH):
            await _upsert_credits(session, credits_chunk)
        for windows_chunk in _chunks(catalog.availability, _BATCH):
            await _upsert_availability(session, windows_chunk)


def _upgrade_head() -> None:
    from alembic import command
    from alembic.config import Config

    root = data_dir().parent
    ini = root / "alembic.ini"
    if not ini.is_file():
        msg = f"alembic.ini not found next to data/: {ini}"
        raise FileNotFoundError(msg)
    command.upgrade(Config(str(ini)), "head")


async def _load_and_count(catalog: NormalizedCatalog) -> dict[str, int]:
    # One event loop owns the engine from persist through dispose. A second
    # asyncio.run() cannot close asyncpg connections opened on the first loop.
    db = Database.from_settings()
    try:
        await persist_catalog(db, catalog)
        async with db.session() as session:
            titles_n = await session.scalar(select(func.count()).select_from(TitleRow))
            people_n = await session.scalar(select(func.count()).select_from(PersonRow))
            credits_n = await session.scalar(select(func.count()).select_from(CreditRow))
            avail_n = await session.scalar(select(func.count()).select_from(AvailabilityRow))
        return {
            "titles": int(titles_n or 0),
            "people": int(people_n or 0),
            "credits": int(credits_n or 0),
            "availability": int(avail_n or 0),
            "quarantined": len(catalog.quarantine),
        }
    finally:
        await db.dispose()


def run(
    csv_path: Path | None = None,
    *,
    migrate: bool = True,
    home_country: str | None = None,
) -> dict[str, int]:
    path = csv_path if csv_path is not None else raw_csv_path()
    if not path.is_file():
        msg = f"catalog CSV not found at {path}; run `jobs fetch` first"
        raise FileNotFoundError(msg)
    if migrate:
        _upgrade_head()
        # alembic fileConfig resets stdlib logging; put JSON structlog back.
        configure_logging(settings.log_level)
    catalog = normalize_csv(path, home_country=home_country)
    playable_basic = sum(
        1
        for window in catalog.availability
        if window.package is Package.BASIC and window.geo == AVAIL_HOME_GEO and window.playable
    )
    playable_frac = playable_basic / len(catalog.titles) if catalog.titles else 0.0
    counts = asyncio.run(_load_and_count(catalog))
    log.info(
        "normalize_done",
        csv=str(path),
        playable_basic_frac=round(playable_frac, 4),
        **counts,
    )
    return counts
