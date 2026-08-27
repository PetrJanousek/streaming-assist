"""Download the Netflix titles CSV with checksum verification.

On any download or checksum failure, copy the committed 500-row sample so
`jobs normalize` can still run. The sample lives under `data/taxonomy/`
because `data/raw/` is gitignored.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import httpx

from assist.config import settings
from assist.obs.logging import get_logger

log = get_logger("assist.jobs.fetch")

RAW_CSV_NAME = "netflix_titles.csv"
SAMPLE_CSV_NAME = "netflix_titles.sample.csv"
GENRE_MAP_NAME = "genre_map.json"

_USER_AGENT = "streaming-assist-fetch/0.1"


def _walk_for_data_dir(start: Path) -> Path | None:
    for parent in [start, *start.parents]:
        candidate = parent / "data"
        if (candidate / "taxonomy" / GENRE_MAP_NAME).is_file():
            return candidate
    return None


def data_dir() -> Path:
    """Resolve `data/` from cwd, this file (editable checkout), or Docker `/app`."""
    found = _walk_for_data_dir(Path.cwd())
    if found is not None:
        return found
    # src/assist/jobs/fetch.py -> repo root in an editable install.
    found = _walk_for_data_dir(Path(__file__).resolve().parents[3])
    if found is not None:
        return found
    docker = Path("/app/data")
    if (docker / "taxonomy" / GENRE_MAP_NAME).is_file():
        return docker
    msg = "data/taxonomy/genre_map.json not found; run from the repo root"
    raise FileNotFoundError(msg)


def taxonomy_dir() -> Path:
    return data_dir() / "taxonomy"


def raw_csv_path() -> Path:
    return data_dir() / "raw" / RAW_CSV_NAME


def sample_csv_path() -> Path:
    return taxonomy_dir() / SAMPLE_CSV_NAME


def _copy_sample(dest: Path, sample: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sample, dest)
    log.warning("fetch_fell_back_to_sample", dest=str(dest), sample=str(sample))
    return dest


def run(
    *,
    dest: Path | None = None,
    sample: Path | None = None,
    url: str | None = None,
    expected_sha256: str | None = None,
    timeout_s: float | None = None,
    client: httpx.Client | None = None,
) -> Path:
    """Fetch the catalog CSV. Always returns a readable path (sample on failure)."""
    dest_path = dest if dest is not None else raw_csv_path()
    sample_path = sample if sample is not None else sample_csv_path()
    dataset_url = url if url is not None else settings.dataset_url
    checksum = (expected_sha256 if expected_sha256 is not None else settings.dataset_sha256).lower()
    timeout = timeout_s if timeout_s is not None else float(settings.dataset_fetch_timeout_s)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".tmp")

    own_client = client is None
    http = client or httpx.Client(
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
    )
    try:
        response = http.get(dataset_url)
        response.raise_for_status()
        payload = response.content
        actual = hashlib.sha256(payload).hexdigest()
        if actual != checksum:
            log.warning(
                "fetch_checksum_mismatch",
                expected=checksum,
                actual=actual,
                url=dataset_url,
            )
            return _copy_sample(dest_path, sample_path)
        tmp_path.write_bytes(payload)
        tmp_path.replace(dest_path)
        log.info("fetch_ok", path=str(dest_path), sha256=actual, bytes=len(payload))
        return dest_path
    except Exception:
        log.exception("fetch_failed", url=dataset_url)
        tmp_path.unlink(missing_ok=True)
        return _copy_sample(dest_path, sample_path)
    finally:
        if own_client:
            http.close()
