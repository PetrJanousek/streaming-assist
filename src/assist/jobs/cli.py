"""Typer entrypoint for offline catalog jobs.

Later tasks (T11 enrich, T12 index, T26 eval) own their own modules. They may
expose `register(app)` or `run()`; this file imports them if present so
`seed-all` grows without those tasks editing T10's CLI.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer

from assist.config import settings
from assist.jobs import fetch as fetch_mod
from assist.jobs import normalize as normalize_mod
from assist.obs.logging import bind_trace_id, configure_logging, get_logger

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Offline catalog jobs")
log = get_logger("assist.jobs.cli")

_OPTIONAL_COMMANDS = (
    "assist.jobs.enrich",
    "assist.jobs.index",
    "assist.jobs.eval",
    "assist.jobs.genre_frequency",
)
_SEED_AFTER_NORMALIZE = (
    "assist.jobs.enrich",
    "assist.jobs.index",
    "assist.jobs.genre_frequency",
)


@app.callback()
def _boot() -> None:
    configure_logging(settings.log_level)
    bind_trace_id(f"job-{uuid4()}")


@app.command()
def fetch() -> None:
    """Download the Netflix titles CSV (checksum, sample fallback)."""
    path = fetch_mod.run()
    log.info("cli_fetch_done", path=str(path))


@app.command()
def normalize(
    csv_path: Annotated[
        Path | None,
        typer.Option("--csv", exists=True, readable=True, help="CSV to load (default data/raw/)"),
    ] = None,
    migrate: Annotated[bool, typer.Option("--migrate/--no-migrate")] = True,
) -> None:
    """Normalize the catalog CSV into Postgres."""
    counts = normalize_mod.run(csv_path, migrate=migrate)
    log.info("cli_normalize_done", **counts)


@app.command(name="seed-all")
def seed_all() -> None:
    """Fetch + normalize, then any later job modules that already exist."""
    fetch_mod.run()
    normalize_mod.run()
    for module_name in _SEED_AFTER_NORMALIZE:
        _run_optional_step(module_name)


def _run_optional_step(module_name: str) -> None:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        log.info("seed_step_skipped", module=module_name, reason="not_present")
        return
    run_fn = getattr(module, "run", None)
    if not callable(run_fn):
        log.info("seed_step_skipped", module=module_name, reason="no_run")
        return
    run_fn()


def _register_optional_commands() -> None:
    for module_name in _OPTIONAL_COMMANDS:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        register = getattr(module, "register", None)
        if callable(register):
            register(app)
            continue
        run_fn = getattr(module, "run", None)
        if callable(run_fn):
            app.command(name=module_name.rsplit(".", 1)[-1])(run_fn)


_register_optional_commands()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
