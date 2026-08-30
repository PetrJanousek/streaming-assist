"""Generate `data/taxonomy/genre_frequency.json`: per-genre share of the catalog.

Feeds `assist.domain.genre_frequency`, which ranks `refine_genre` chips by
lift (T38 follow-up) instead of raw pool count -- raw count surfaces whatever
genre is globally common (romance, drama), not what is distinctive to the
retrieved pool.

Offline only: no ES, no Postgres. Reuses `jobs.normalize.normalize_csv`, the
same CSV -> Title parsing the live index is built from, so the table tracks
whatever `genre_map.json` maps titles to. Counts are document frequency (one
count per title per genre, not per occurrence) over whichever CSV `jobs
fetch` last landed -- the real dataset when network access worked, the
committed 500-row sample otherwise. Either way this never touches a live
store, so it runs in CI and in a bare checkout.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from assist.jobs.fetch import raw_csv_path, sample_csv_path
from assist.jobs.normalize import normalize_csv
from assist.obs.logging import get_logger

log = get_logger("assist.jobs.genre_frequency")

OUTPUT_NAME = "genre_frequency.json"


def output_path() -> Path:
    return sample_csv_path().parent / OUTPUT_NAME


def compute(csv_path: Path) -> dict[str, object]:
    """Document-frequency count per genre plus the title total, from one CSV."""
    catalog = normalize_csv(csv_path)
    counts: Counter[str] = Counter()
    for title in catalog.titles:
        for genre in set(title.genres):
            counts[genre.value] += 1
    return {"total_titles": len(catalog.titles), "counts": dict(sorted(counts.items()))}


def run(csv_path: Path | None = None, *, dest: Path | None = None) -> dict[str, object]:
    """Entry point for `jobs genre-frequency` and `seed-all`.

    Prefers the CSV `jobs fetch` landed at `data/raw/`; falls back to the
    committed sample so this always runs without network access.
    """
    source = csv_path
    if source is None:
        source = raw_csv_path() if raw_csv_path().is_file() else sample_csv_path()
    payload = compute(source)
    target = dest if dest is not None else output_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log.info(
        "genre_frequency_done",
        csv=str(source),
        dest=str(target),
        total_titles=payload["total_titles"],
        genres=len(payload["counts"]),  # type: ignore[arg-type]
    )
    return payload


def register(app: typer.Typer) -> None:
    @app.command("genre-frequency")
    def genre_frequency_cmd(
        csv: Annotated[
            Path | None,
            typer.Option(
                "--csv",
                exists=True,
                readable=True,
                help="CSV to count (default data/raw/, else sample)",
            ),
        ] = None,
    ) -> None:
        """Regenerate data/taxonomy/genre_frequency.json from the catalog CSV."""
        run(csv)
