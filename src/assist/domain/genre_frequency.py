"""Global genre document frequency, used to rank pool genre chips by lift.

Raw frequency within a candidate pool surfaces whatever is globally common in
the whole catalog (romance, drama), not what is thematically distinctive to
this pool -- that is the T38 bug (`horror movies from 90s` minting "More
romance"). Lift corrects for it:

    lift(genre) = pool_share(genre) / catalog_share(genre)

This module is the catalog-share half of that ratio: a static table of one
share per `GenreId`, generated offline by `assist.jobs.genre_frequency` from
the same CSV the live index is built from, and loaded once at node
construction (never per-request I/O -- see `nodes/chips.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

from assist.domain.enums import GenreId
from assist.obs.logging import get_logger

log = get_logger("assist.domain.genre_frequency")

# Fallback when the generated file is absent or malformed. Every genre shares
# the catalog uniformly, so lift(genre) = pool_share(genre) / constant is a
# rescale of pool_share by the same factor for every genre -- ranking
# degrades to plain pool-frequency order, the pre-lift behaviour, rather than
# failing the node.
_UNIFORM_SHARE = 1.0 / len(GenreId)

# Floor so a corrupt/zero entry in the table cannot divide by zero or hand a
# single miscounted genre an unbounded lift.
_MIN_SHARE = 1e-6


class GenreFrequencyTable:
    """Per-genre share of the whole catalog. `share()` never raises."""

    def __init__(self, shares: dict[GenreId, float]) -> None:
        self._shares = dict(shares)

    def share(self, genre: GenreId) -> float:
        return max(self._shares.get(genre, _UNIFORM_SHARE), _MIN_SHARE)

    @classmethod
    def uniform(cls) -> GenreFrequencyTable:
        return cls(dict.fromkeys(GenreId, _UNIFORM_SHARE))

    @classmethod
    def from_counts(cls, counts: dict[str, int], total_titles: int) -> GenreFrequencyTable:
        if total_titles <= 0:
            return cls.uniform()
        shares: dict[GenreId, float] = {}
        for label, count in counts.items():
            try:
                genre = GenreId(label)
            except ValueError:
                # A future genre_map.json can add labels this build's GenreId
                # enum does not know about yet; ignore rather than raise.
                log.info("genre_frequency_unknown_genre_ignored", label=label)
                continue
            shares[genre] = count / total_titles
        return cls(shares)

    @classmethod
    def from_path(cls, path: Path) -> GenreFrequencyTable:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            msg = f"genre frequency table must be a JSON object: {path}"
            raise ValueError(msg)
        counts = raw.get("counts")
        total = raw.get("total_titles")
        if not isinstance(counts, dict) or not isinstance(total, int):
            msg = f"genre frequency table missing counts/total_titles: {path}"
            raise ValueError(msg)
        return cls.from_counts(counts, total)


_GENRE_FREQUENCY_REL = Path("data") / "taxonomy" / "genre_frequency.json"


def default_genre_frequency_path() -> Path:
    """Resolve from cwd, an editable checkout, or the image.

    An installed package sits under `.venv/lib/.../site-packages`, so a fixed
    `parents[N]` walks into the venv instead of the repo -- that is exactly
    the T-series bug that made this table (and the lift ranking built on it)
    silently inert under Docker. Search upward instead. `templates.py`'s
    `default_phrases_path()` has the identical bug; once that fix lands the
    two resolvers should be unified into one helper.
    """
    for start in (Path.cwd(), Path(__file__).resolve()):
        for parent in [start, *start.parents]:
            candidate = parent / _GENRE_FREQUENCY_REL
            if candidate.is_file():
                return candidate
    return Path("/app") / _GENRE_FREQUENCY_REL


def load_genre_frequency(path: Path | None = None) -> GenreFrequencyTable:
    """Load the committed table. A missing or corrupt file falls back to uniform."""
    target = path if path is not None else default_genre_frequency_path()
    try:
        return GenreFrequencyTable.from_path(target)
    except (OSError, json.JSONDecodeError, ValueError, UnicodeDecodeError):
        log.warning("genre_frequency_missing", path=str(target))
        return GenreFrequencyTable.uniform()
