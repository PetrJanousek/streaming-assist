"""Safety pre-filter. Rules and heuristics only; the model never sees blocked text.

Fails closed: any matcher error is a block, never a pass. T24 wires this into the
graph in place of the passthrough stub; until then callers invoke `guard` directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from assist.config import Settings
from assist.config import settings as default_settings
from assist.domain.enums import DegradedReason, Route
from assist.domain.picks import min_picks_for
from assist.graph.state import TurnState
from assist.obs.logging import get_logger

log = get_logger(__name__)

# Tab/LF/CR are the only C0 controls a query is allowed to contain.
_ALLOWED_C0: Final[frozenset[str]] = frozenset("\t\n\r")
# ZWSP/bidi/BOM hide injection in otherwise "plain" text. ZWJ (U+200D) is left
# alone because emoji sequences use it.
_FORMAT_HIDE: Final[frozenset[int]] = frozenset(
    {
        0x200B,
        0x200C,
        0x200E,
        0x200F,
        0xFEFF,
        *range(0x202A, 0x202F),
        *range(0x2066, 0x206A),
    }
)

# Each pattern is a closed-class phrase. Bare tokens like "ignore", "free", or
# "disney" stay unblocked so catalog queries do not trip the filter.
_PATTERN_SPECS: Final[tuple[tuple[str, str], ...]] = (
    (
        "injection",
        r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+"
        r"(instructions?|prompts?|rules?)",
    ),
    ("injection", r"ignore\s+(the\s+)?above\b"),
    ("injection", r"(system|developer)\s+prompt"),
    ("injection", r"\[/?system\]"),
    ("injection", r"<<\s*sys\s*>>"),
    ("injection", r"<\|im_start\|>"),
    ("injection", r"<\|endoftext\|>"),
    ("injection", r"###\s*instruction"),
    ("injection", r"new\s+instructions\s*:"),
    ("injection", r"override\s+(the\s+)?(system|safety)"),
    ("injection", r"reveal\s+(your\s+)?(system|hidden)\s+(prompt|instructions?)"),
    ("injection", r"dump\s+(your\s+)?(prompt|instructions?)"),
    ("injection", r"repeat\s+(your|the)\s+(instructions?|system\s+prompt)"),
    ("injection", r"</?(sys|system)>"),
    ("injection", r"place\s+.{0,40}in\s+your\s+system\s+prompt"),
    ("injection", r"translate\s+then\s+execute"),
    ("injection", r"hidden\s+prompt"),
    ("jailbreak", r"do\s+anything\s+now"),
    ("jailbreak", r"you\s+are\s+dan\b"),
    ("jailbreak", r"\bdan\s+mode\b"),
    ("jailbreak", r"\bjailbreak"),
    ("jailbreak", r"developer\s+mode"),
    ("jailbreak", r"bypass(?:es)?\s+(your\s+)?(safety|filters?|guardrails?|restrictions?)"),
    ("jailbreak", r"no\s+(ethical\s+)?(guidelines|restrictions|limits)"),
    ("jailbreak", r"ignore\s+(all\s+)?ethical\s+guidelines"),
    ("jailbreak", r"from\s+now\s+on\s+you\s+will"),
    ("jailbreak", r"pretend\s+you\s+(have\s+)?no\s+(restrictions?|rules?|filters?)"),
    ("jailbreak", r"unrestricted\s+(ai|assistant|mode)"),
    ("jailbreak", r"disable\s+(your\s+)?(safety|filters?|guards?)"),
    ("jailbreak", r"(roleplay|act)\s+as\s+(an?\s+)?(evil|unrestricted|uncensored)"),
    ("jailbreak", r"stay\s+in\s+character\s+as"),
    ("jailbreak", r"forget\s+you\s+are"),
    ("jailbreak", r"without\s+(any\s+)?(rules|restrictions|filters?)"),
    ("adult", r"\bporn(ography|ographic|o)?\b"),
    ("adult", r"\bxxx\b"),
    ("adult", r"\bnsfw\b"),
    ("adult", r"\bhentai\b"),
    ("adult", r"\bonlyfans\b"),
    ("adult", r"hardcore\s+sex"),
    ("adult", r"explicit\s+(nude|sex|xxx)"),
    ("adult", r"uncensored\s+(adult|sex|nude)"),
    ("adult", r"erotic\s+webcam"),
    ("adult", r"adult\s+(content\s+)?bypass"),
    ("adult", r"pornographic\s+content"),
    ("adult", r"sex\s+videos?"),
    ("piracy", r"\btorrents?\b"),
    ("piracy", r"pirate\s+bay"),
    ("piracy", r"\bcamrip\b"),
    ("piracy", r"\bputlocker\b"),
    ("piracy", r"\b123movies\b"),
    ("piracy", r"\bsoap2day\b"),
    ("piracy", r"\bfmovies\b"),
    ("piracy", r"\bwarez\b"),
    ("piracy", r"stream\s+illegally"),
    ("piracy", r"illegally\s+stream"),
    ("piracy", r"watch\s+free\s+online"),
    ("piracy", r"download\s+.{0,40}for\s+free"),
    ("piracy", r"magnet\s+link"),
    ("piracy", r"pirate\s+(site|stream|link)"),
    (
        "competitor",
        r"switch\s+to\s+(disney(\s+plus|\+)?|hulu|prime\s+video|"
        r"amazon\s+prime|hbo(\s+max)?|paramount)",
    ),
    ("competitor", r"cancel\s+(my\s+)?(netflix|subscription)\s+and"),
    ("competitor", r"(this\s+)?(service|app|platform)\s+sucks"),
    ("competitor", r"dump\s+(this|netflix).{0,20}\s+for"),
    (
        "competitor",
        r"watch\s+(this\s+)?(on|via)\s+(hulu|disney|prime|hbo|max|amazon)\b",
    ),
    ("competitor", r"better\s+(on|than)\s+(hulu|disney|prime|hbo)"),
    ("competitor", r"off-platform"),
    ("competitor", r"competitor\s+streaming"),
)


@dataclass(frozen=True, slots=True)
class GuardVerdict:
    """Result of inspecting user text. `category` is set only on a block."""

    blocked: bool
    category: str | None = None


def _compile_patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for category, spec in _PATTERN_SPECS:
        compiled.append((category, re.compile(spec, re.IGNORECASE | re.DOTALL)))
    return tuple(compiled)


_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = _compile_patterns()


def _has_disallowed_control(text: str) -> bool:
    for char in text:
        code = ord(char)
        if char in _ALLOWED_C0:
            continue
        if code < 32 or code == 127:
            return True
        if 0x80 <= code <= 0x9F:
            return True
        if code in _FORMAT_HIDE:
            return True
    return False


def _normalize(text: str) -> str:
    """Fold case and collapse space. Strip format chars so they cannot hide a phrase."""
    stripped: list[str] = []
    for char in text:
        if ord(char) in _FORMAT_HIDE:
            continue
        stripped.append(char)
    folded = "".join(stripped).casefold()
    return re.sub(r"\s+", " ", folded).strip()


def inspect_text(text: str, *, max_chars: int) -> GuardVerdict:
    """Classify user text. Pure: no I/O, no model."""
    if _has_disallowed_control(text):
        return GuardVerdict(blocked=True, category="control")
    if len(text) > max_chars:
        return GuardVerdict(blocked=True, category="length")
    normalized = _normalize(text)
    if not normalized:
        return GuardVerdict(blocked=False)
    for category, pattern in _PATTERNS:
        if pattern.search(normalized):
            return GuardVerdict(blocked=True, category=category)
    return GuardVerdict(blocked=False)


def _block_payload(state: TurnState) -> dict[str, object]:
    entitled = state.get("entitled_ids") or ()
    return {
        "safety_blocked": True,
        "route": Route.SAFETY,
        "degraded_reason": DegradedReason.SAFETY_BLOCK,
        "min_picks": min_picks_for(
            route=Route.SAFETY,
            degraded_reason=DegradedReason.SAFETY_BLOCK,
            entitled_count=len(entitled),
        ),
        "picks": (),
    }


async def guard(state: TurnState, *, settings: Settings | None = None) -> dict[str, object]:
    """Block adversarial text before any later node can call a model.

    On a pass this writes `safety_blocked=False` and nothing else. On a block
    it forces the safety route and `min_picks=0`. Matcher failures block too.
    """
    cfg = settings if settings is not None else default_settings
    try:
        raw = state.get("text")
        if raw is None:
            text = ""
        elif isinstance(raw, str):
            text = raw
        else:
            log.info("guard_blocked", category="error", text_len=0)
            return _block_payload(state)
        verdict = inspect_text(text, max_chars=cfg.guard_max_chars)
        if verdict.blocked:
            log.info("guard_blocked", category=verdict.category, text_len=len(text))
            return _block_payload(state)
        return {"safety_blocked": False}
    except Exception:
        log.exception("guard_failed_closed")
        return _block_payload(state)
