"""Genre frequency generator: pure CSV -> counts, no ES / no Postgres."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from assist.domain.enums import GenreId
from assist.domain.genre_frequency import (
    GenreFrequencyTable,
    default_genre_frequency_path,
    load_genre_frequency,
)
from assist.jobs import genre_frequency as gf
from assist.jobs.fetch import sample_csv_path


def test_compute_counts_genres_as_document_frequency() -> None:
    payload = gf.compute(sample_csv_path())
    total = payload["total_titles"]
    assert isinstance(total, int)
    assert total > 0
    counts = payload["counts"]
    assert isinstance(counts, dict)
    # 21 canonical genres; the sample is small but should hit most of them.
    assert set(counts) <= {g.value for g in GenreId}
    assert all(isinstance(v, int) and v > 0 for v in counts.values())


def test_run_writes_dest_and_is_reproducible(tmp_path: Path) -> None:
    dest = tmp_path / "genre_frequency.json"
    first = gf.run(sample_csv_path(), dest=dest)
    second = gf.run(sample_csv_path(), dest=dest)
    assert first == second
    on_disk = json.loads(dest.read_text(encoding="utf-8"))
    assert on_disk == first


def test_generated_file_loads_into_a_usable_frequency_table(tmp_path: Path) -> None:
    dest = tmp_path / "genre_frequency.json"
    payload = gf.run(sample_csv_path(), dest=dest)
    table = GenreFrequencyTable.from_path(dest)
    total = payload["total_titles"]
    counts = payload["counts"]
    assert isinstance(counts, dict)
    assert isinstance(total, int)
    for label, count in counts.items():
        assert table.share(GenreId(label)) == count / total


def test_default_path_finds_the_committed_table_from_a_repo_checkout() -> None:
    # Regression: parents[N] on the installed module walks into
    # .venv/lib/.../site-packages under Docker, not the repo -- the resolver
    # must search upward instead, and it must land on the real, non-empty
    # committed table when run from an ordinary checkout.
    path = default_genre_frequency_path()
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["total_titles"] > 0
    assert payload["counts"]


def test_default_path_falls_back_to_image_path_when_absent_everywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate the file being absent from every candidate the resolver would
    # walk -- cwd's parents and the module's own parents both come up empty
    # (as they would in a stripped-down image missing the data dir). The
    # resolver must not raise; it hands back the fixed image path instead.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    path = default_genre_frequency_path()
    assert path == Path("/app") / "data" / "taxonomy" / "genre_frequency.json"


def test_missing_file_falls_back_to_uniform_and_ranking_does_not_crash(
    tmp_path: Path,
) -> None:
    table = load_genre_frequency(tmp_path / "does-not-exist.json")
    uniform_share = 1.0 / len(GenreId)
    for genre in GenreId:
        assert table.share(genre) == pytest.approx(uniform_share)
    # Uniform lift(genre) = pool_share(genre) / constant is a rescale by the
    # same factor for every genre, so it must not perturb relative order --
    # ranking degrades to plain pool-frequency, never to an error.
    shares = {genre: table.share(genre) for genre in GenreId}
    assert len(set(shares.values())) == 1
