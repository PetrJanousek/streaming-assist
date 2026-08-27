"""Versioned prompt files. T15/T23 add intent.md and reply.md; load them by name."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

_PROMPTS_DIR = Path(__file__).resolve().parent


def prompt_path(name: str) -> Path:
    """Resolve a prompt basename under this directory. Rejects path traversal."""
    stem = name.removesuffix(".md")
    if not stem or stem != Path(stem).name or stem in {".", ".."}:
        msg = f"prompt name must be a basename, got {name!r}"
        raise ValueError(msg)
    return _PROMPTS_DIR / f"{stem}.md"


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    path = prompt_path(name)
    if not path.is_file():
        msg = f"prompt file not found: {path.name}"
        raise FileNotFoundError(msg)
    return path.read_text(encoding="utf-8")


def chat_prompt_template(name: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(load_prompt(name))
