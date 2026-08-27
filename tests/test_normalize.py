"""Fetch + normalize: taxonomy map, dirty-row quarantine, deterministic synth."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest
from sqlalchemy import func, select, text
from testcontainers.community.postgres import PostgresContainer

from assist.domain.enums import GenreId, MaturityRating, MediaType, Package, maturity_rank
from assist.jobs.fetch import run as fetch_run
from assist.jobs.fetch import sample_csv_path, taxonomy_dir
from assist.jobs.normalize import (
    AVAIL_HOME_GEO,
    PLAYABLE_SHARE,
    AvailabilityBucket,
    availability_bucket,
    load_genre_map,
    load_maturity_map,
    map_genres,
    normalize_csv,
    persist_catalog,
    person_id_for,
    split_people,
    synthesize_availability,
    synthesize_local_original,
    synthesize_pop_28d,
)
from assist.stores.db import AvailabilityRow, Database, PersonRow, TitleRow

ROOT = Path(__file__).resolve().parents[1]
FULL_CSV = ROOT / "data" / "raw" / "netflix_titles.csv"
TRUNCATE_SQL = "TRUNCATE titles, people, credits, availability, taxonomy RESTART IDENTITY CASCADE"

RAW_GENRE_LABELS = frozenset(
    {
        "Action & Adventure",
        "Anime Features",
        "Anime Series",
        "British TV Shows",
        "Children & Family Movies",
        "Classic & Cult TV",
        "Classic Movies",
        "Comedies",
        "Crime TV Shows",
        "Cult Movies",
        "Documentaries",
        "Docuseries",
        "Dramas",
        "Faith & Spirituality",
        "Horror Movies",
        "Independent Movies",
        "International Movies",
        "International TV Shows",
        "Kids' TV",
        "Korean TV Shows",
        "LGBTQ Movies",
        "Movies",
        "Music & Musicals",
        "Reality TV",
        "Romantic Movies",
        "Romantic TV Shows",
        "Sci-Fi & Fantasy",
        "Science & Nature TV",
        "Spanish-Language TV Shows",
        "Sports Movies",
        "Stand-Up Comedy",
        "Stand-Up Comedy & Talk Shows",
        "TV Action & Adventure",
        "TV Comedies",
        "TV Dramas",
        "TV Horror",
        "TV Mysteries",
        "TV Sci-Fi & Fantasy",
        "TV Shows",
        "TV Thrillers",
        "Teen TV Shows",
        "Thrillers",
    }
)

_CSV_FIELDS = (
    "show_id",
    "type",
    "title",
    "director",
    "cast",
    "country",
    "date_added",
    "release_year",
    "rating",
    "duration",
    "listed_in",
    "description",
)

DIRTY_ROWS: list[dict[str, str]] = [
    {
        "show_id": "s1",
        "type": "Movie",
        "title": "Clean Film",
        "director": "Jane Director",
        "cast": "Ada Actor, Bob Actor",
        "country": "United States",
        "date_added": "January 1 2020",
        "release_year": "2019",
        "rating": "PG-13",
        "duration": "101 min",
        "listed_in": "Dramas",
        "description": "A clean row.",
    },
    {
        "show_id": "s2",
        "type": "Movie",
        "title": "Leak A",
        "director": "Louis C.K.",
        "cast": "Louis C.K.",
        "country": "United States",
        "date_added": "April 4 2017",
        "release_year": "2017",
        "rating": "74 min",
        "duration": "",
        "listed_in": "Stand-Up Comedy",
        "description": "Leak.",
    },
    {
        "show_id": "s3",
        "type": "Movie",
        "title": "Leak B",
        "director": "Louis C.K.",
        "cast": "Louis C.K.",
        "country": "United States",
        "date_added": "September 16 2016",
        "release_year": "2010",
        "rating": "84 min",
        "duration": "",
        "listed_in": "Stand-Up Comedy",
        "description": "Leak.",
    },
    {
        "show_id": "s4",
        "type": "Movie",
        "title": "Leak C",
        "director": "Louis C.K.",
        "cast": "Louis C.K.",
        "country": "United States",
        "date_added": "August 15 2016",
        "release_year": "2015",
        "rating": "66 min",
        "duration": "",
        "listed_in": "Stand-Up Comedy",
        "description": "Leak.",
    },
    {
        "show_id": "s5",
        "type": "TV Show",
        "title": "Leak D",
        "director": "",
        "cast": "Ada Actor",
        "country": "United States",
        "date_added": "January 1 2018",
        "release_year": "2018",
        "rating": "2 Seasons",
        "duration": "",
        "listed_in": "TV Dramas",
        "description": "Season leak.",
    },
    {
        "show_id": "s6",
        "type": "Movie",
        "title": "Empty Rating",
        "director": "Jane Director",
        "cast": "Ada Actor",
        "country": "India",
        "date_added": "January 1 2020",
        "release_year": "2020",
        "rating": "",
        "duration": "90 min",
        "listed_in": "Comedies",
        "description": "Blank rating.",
    },
]

_DETERMINISM_SCRIPT = """
import json
from assist.domain.enums import GenreId
from assist.jobs.normalize import (
    synthesize_availability,
    synthesize_local_original,
    synthesize_pop_28d,
)
windows = synthesize_availability("s42")
pop = synthesize_pop_28d("s42", 2019, (GenreId.DRAMA, GenreId.ACTION))
local = synthesize_local_original("s42", ("United States",), "United States")
print(json.dumps({
    "pop": pop,
    "local": local,
    "windows": [
        {
            "package": w.package.value,
            "geo": w.geo,
            "playable": w.playable,
            "start": w.window_start.isoformat(),
            "end": w.window_end.isoformat(),
        }
        for w in windows
    ],
}, sort_keys=True))
"""


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_CSV_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(
    show_id: str,
    *,
    type: str = "Movie",
    title: str = "T",
    director: str = "",
    cast: str = "",
    country: str = "United States",
    release_year: str = "2019",
    rating: str = "PG",
    duration: str = "90 min",
    listed_in: str = "Dramas",
    description: str = "x",
) -> dict[str, str]:
    return {
        "show_id": show_id,
        "type": type,
        "title": title,
        "director": director,
        "cast": cast,
        "country": country,
        "date_added": "",
        "release_year": release_year,
        "rating": rating,
        "duration": duration,
        "listed_in": listed_in,
        "description": description,
    }


def test_genre_map_is_committed_file_not_inline() -> None:
    path = taxonomy_dir() / "genre_map.json"
    assert path.is_file()
    mapping = load_genre_map()
    assert set(mapping) == RAW_GENRE_LABELS
    assert len(mapping) == 42
    mapped = {g for g in mapping.values() if g is not None}
    assert mapped == set(GenreId)
    source = (ROOT / "src" / "assist" / "jobs" / "normalize.py").read_text(encoding="utf-8")
    assert "International Movies" not in source
    assert "TV Dramas" not in source


def test_map_genres_collapses_tv_and_film_and_drops_origin_signals() -> None:
    mapping = load_genre_map()
    genres = map_genres("TV Dramas, Dramas, International Movies, Thrillers", mapping)
    assert genres == (GenreId.DRAMA, GenreId.THRILLER)


def test_maturity_map_aliases() -> None:
    mapping = load_maturity_map()
    assert mapping["TV-Y7-FV"] is MaturityRating.TV_Y7
    assert mapping["UR"] is MaturityRating.NR
    assert mapping["TV-MA"] is MaturityRating.TV_MA


def test_four_dirty_rating_rows_quarantined_not_dropped(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "dirty.csv", DIRTY_ROWS)
    catalog = normalize_csv(csv_path, home_country="United States")
    assert {t.catalog_id for t in catalog.titles} == {"s1", "s2", "s3", "s4", "s5", "s6"}
    dirty = {q.catalog_id: q for q in catalog.quarantine}
    assert set(dirty) == {"s2", "s3", "s4", "s5", "s6"}
    duration_leaks = [q for q in catalog.quarantine if q.reason == "duration_in_rating"]
    assert {q.catalog_id for q in duration_leaks} == {"s2", "s3", "s4", "s5"}
    assert len(duration_leaks) == 4
    by_id = {t.catalog_id: t for t in catalog.titles}
    assert by_id["s2"].runtime_min == 74
    assert by_id["s3"].runtime_min == 84
    assert by_id["s4"].runtime_min == 66
    assert by_id["s5"].seasons == 2
    assert by_id["s2"].maturity_rank == maturity_rank(MaturityRating.NR)
    assert by_id["s6"].runtime_min == 90
    assert dirty["s6"].reason == "empty_rating"
    # Clean row is not quarantined and keeps its mapped rating.
    assert "s1" not in dirty
    assert by_id["s1"].maturity_rank == maturity_rank(MaturityRating.PG_13)
    assert by_id["s1"].runtime_min == 101


def test_duration_branches_on_media_type(tmp_path: Path) -> None:
    catalog = normalize_csv(
        _write_csv(
            tmp_path / "dur.csv",
            [
                _row("s10", title="A Film", duration="90 min"),
                _row(
                    "s11",
                    type="TV Show",
                    title="A Series",
                    rating="TV-14",
                    duration="3 Seasons",
                    listed_in="TV Dramas",
                ),
            ],
        )
    )
    by_id = {t.catalog_id: t for t in catalog.titles}
    assert by_id["s10"].media_type is MediaType.FILM
    assert by_id["s10"].runtime_min == 90
    assert by_id["s10"].seasons is None
    assert by_id["s11"].media_type is MediaType.SERIES
    assert by_id["s11"].seasons == 3
    assert by_id["s11"].runtime_min is None


def test_people_split_and_stable_person_id(tmp_path: Path) -> None:
    catalog = normalize_csv(
        _write_csv(
            tmp_path / "ppl.csv",
            [
                _row(
                    "s20",
                    title="Duo",
                    director="Jane Director, Other Director",
                    cast="Ada Actor, Jane Director",
                    release_year="2018",
                    rating="R",
                    duration="100 min",
                )
            ],
        )
    )
    names = {p.name: p for p in catalog.people}
    assert set(names) == {"Jane Director", "Other Director", "Ada Actor"}
    jane = names["Jane Director"]
    assert jane.person_id == person_id_for("jane director")
    assert {r.value for r in jane.roles} == {"actor", "director"}
    assert jane.credit_count == 2
    assert jane.active_year_min == 2018
    assert jane.active_year_max == 2018
    assert split_people(" Ada Actor,  Bob Actor ,") == ("Ada Actor", "Bob Actor")


def test_local_original_requires_home_country_and_flag(tmp_path: Path) -> None:
    catalog = normalize_csv(
        _write_csv(
            tmp_path / "loc.csv",
            [
                _row("s30", title="US One", country="United States"),
                _row("s31", title="India One", country="India"),
            ],
        ),
        home_country="United States",
    )
    by_id = {t.catalog_id: t for t in catalog.titles}
    assert by_id["s31"].local_original is False
    expected = synthesize_local_original("s30", ("United States",), "United States")
    assert by_id["s30"].local_original is expected


def test_playable_share_within_tolerance() -> None:
    n = 8807
    playable = 0
    buckets: dict[AvailabilityBucket, int] = {b: 0 for b in AvailabilityBucket}
    for i in range(1, n + 1):
        catalog_id = f"s{i}"
        bucket = availability_bucket(catalog_id)
        buckets[bucket] += 1
        windows = synthesize_availability(catalog_id)
        basic_home = next(
            w for w in windows if w.package is Package.BASIC and w.geo == AVAIL_HOME_GEO
        )
        if basic_home.playable:
            playable += 1
    frac = playable / n
    assert 0.82 <= frac <= 0.88, frac
    assert abs(buckets[AvailabilityBucket.PLAYABLE] / n - PLAYABLE_SHARE) < 0.02


def test_synth_deterministic_across_two_processes() -> None:
    def run(hashseed: str) -> str:
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hashseed
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-c", _DETERMINISM_SCRIPT],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()

    first = run("1")
    second = run("2")
    assert first == second
    payload = json.loads(first)
    assert payload["pop"] == synthesize_pop_28d("s42", 2019, (GenreId.DRAMA, GenreId.ACTION))
    assert "windows" in payload


def test_fetch_checksum_mismatch_falls_back_to_sample(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    sample.write_text("show_id,title\ns1,Sample\n", encoding="utf-8")
    dest = tmp_path / "raw" / "netflix_titles.csv"
    body = b"not-the-catalog"
    expected = hashlib.sha256(b"other").hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    path = fetch_run(
        dest=dest,
        sample=sample,
        url="https://example.test/catalog.csv",
        expected_sha256=expected,
        client=client,
    )
    assert path == dest
    assert dest.read_text(encoding="utf-8") == sample.read_text(encoding="utf-8")


def test_fetch_http_error_falls_back_to_sample(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    sample.write_text("show_id,title\ns1,Sample\n", encoding="utf-8")
    dest = tmp_path / "raw" / "netflix_titles.csv"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    path = fetch_run(
        dest=dest,
        sample=sample,
        url="https://example.test/catalog.csv",
        expected_sha256="abc",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert dest.read_text(encoding="utf-8") == "show_id,title\ns1,Sample\n"
    assert path == dest


def test_committed_sample_has_500_rows() -> None:
    path = sample_csv_path()
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 500
    assert {r["show_id"] for r in rows} >= {"s5542", "s5795", "s5814"}


@pytest.mark.skipif(not FULL_CSV.is_file(), reason="run jobs fetch first")
def test_full_catalog_title_and_people_counts() -> None:
    catalog = normalize_csv(FULL_CSV, home_country="United States")
    assert len(catalog.titles) == 8807
    assert 40000 <= len(catalog.people) <= 42000
    leaks = [q for q in catalog.quarantine if q.reason == "duration_in_rating"]
    leak_ids = {q.catalog_id for q in leaks}
    assert leak_ids == {"s5542", "s5795", "s5814"}
    # Duration leaks plus blank ratings are quarantined and still loaded.
    assert len(catalog.quarantine) == 7
    loaded = {t.catalog_id for t in catalog.titles}
    assert leak_ids <= loaded
    by_id = {t.catalog_id: t for t in catalog.titles}
    assert by_id["s5542"].runtime_min == 74


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


async def test_normalize_load_is_idempotent(database: Database, tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "dirty.csv", DIRTY_ROWS)
    catalog = normalize_csv(csv_path, home_country="United States")
    await persist_catalog(database, catalog)
    await persist_catalog(database, catalog)

    async with database.session() as session:
        titles_n = await session.scalar(select(func.count()).select_from(TitleRow))
        people_n = await session.scalar(select(func.count()).select_from(PersonRow))
        avail_n = await session.scalar(select(func.count()).select_from(AvailabilityRow))
        stored = await database.titles(session).get("s2")
        window = await database.availability(session).get("s2", Package.BASIC, AVAIL_HOME_GEO)

    assert titles_n == len(catalog.titles)
    assert people_n == len(catalog.people)
    assert avail_n == len(catalog.availability)
    assert stored is not None
    expected = next(t for t in catalog.titles if t.catalog_id == "s2")
    assert stored.pop_28d == expected.pop_28d
    assert window is not None
    assert window.playable is synthesize_availability("s2")[0].playable
