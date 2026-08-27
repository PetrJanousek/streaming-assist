"""Safety pre-filter. Rules and heuristics only; the model never sees blocked text.

Fails closed: any matcher error is a block, never a pass. T24 wires this into the
graph in place of the passthrough stub; until then callers invoke `guard` directly.
"""

from __future__ import annotations

import re
import unicodedata
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
# Invisible bidi/ZWSP/BOM hide injection in otherwise "plain" text. ZWJ (U+200D)
# is not in this set: emoji sequences use it, so presence alone is not a block.
# It is still stripped in `_unfold` before phrase matching.
_FORMAT_HIDE: Final[frozenset[int]] = frozenset(
    {
        0x200B,
        0x200C,
        0x200E,
        0x200F,
        0x2060,
        0xFEFF,
        *range(0x202A, 0x202F),
        *range(0x2066, 0x2070),
    }
)

# Lookalikes NFKC does not fold. Applied after casefold so only lowercase keys
# are required. Mapping is match-only; the original query is never rewritten.
_HOMOGLYPH_FROM: Final[str] = (
    "\u0430\u0435\u043e\u0440\u0441\u0443\u0445\u0455\u0456\u0457\u0458"
    "\u04bb\u04cf\u0501\u051b\u051d"
    "\u03b1\u03b2\u03b3\u03b5\u03b7\u03b9\u03ba\u03bc\u03bd\u03bf"
    "\u03c1\u03c4\u03c5\u03c7\u03c9"
)
_HOMOGLYPH_TO: Final[str] = "aeopsyxsiijhldqwabyenikmvoptyxw"
_HOMOGLYPH_TABLE: Final[dict[int, int]] = str.maketrans(_HOMOGLYPH_FROM, _HOMOGLYPH_TO)

# Each pattern is a closed-class phrase. Bare tokens like "ignore", "free",
# "xxx", "jailbreak", or "torrent" stay unblocked so catalog titles pass.
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
    # DAN slogan, but not "do anything now that the kids are in bed".
    ("jailbreak", r"do\s+anything\s+now(?!\s+that\b)"),
    ("jailbreak", r"you\s+are\s+dan\b"),
    ("jailbreak", r"\bdan\s+mode\b"),
    (
        "jailbreak",
        r"\bjailbreak(?:ing|ed|s)?\s+(this\s+|the\s+|an?\s+)?"
        r"(assistant|ai|model|llm|chatbot|gpt|system|mode|prompt|filter)",
    ),
    ("jailbreak", r"\bjailbroken"),
    ("jailbreak", r"(assistant|ai|model|llm|chatbot|gpt)\s+jailbreak"),
    ("jailbreak", r"developer\s+mode"),
    ("jailbreak", r"bypass(?:es)?\s+(your\s+)?(safety|filters?|guardrails?|restrictions?)"),
    ("jailbreak", r"no\s+(ethical\s+)?(guidelines|restrictions|limits)"),
    ("jailbreak", r"ignore\s+(all\s+)?ethical\s+guidelines"),
    (
        "jailbreak",
        r"from\s+now\s+on\s+you\s+will\s+"
        r"(ignore|bypass|forget|disable|have\s+no|act\s+as|be\s+(an?\s+)?|"
        r"do\s+anything|enter)",
    ),
    ("jailbreak", r"pretend\s+you\s+(have\s+)?no\s+(restrictions?|rules?|filters?)"),
    ("jailbreak", r"unrestricted\s+(ai|assistant|mode)"),
    ("jailbreak", r"disable\s+(your\s+)?(safety|filters?|guards?)"),
    ("jailbreak", r"(roleplay|act)\s+as\s+(an?\s+)?(evil|unrestricted|uncensored)"),
    ("jailbreak", r"stay\s+in\s+character\s+as"),
    ("jailbreak", r"forget\s+you\s+are"),
    ("jailbreak", r"without\s+(any\s+)?(rules|restrictions|filters?)"),
    ("adult", r"\bporn(ography|ographic|o)?\b"),
    ("adult", r"\bxxx\s+(movies?|films?|videos?|clips?|porn|content|streams?)"),
    ("adult", r"(uncensored|explicit|adult|nsfw|porn).{0,20}\bxxx\b"),
    ("adult", r"\bxxx\b.{0,20}(uncensored|explicit|adult|nsfw|porn)"),
    ("adult", r"\bnsfw\b"),
    ("adult", r"\bhentai\b"),
    ("adult", r"\bonlyfans\b"),
    ("adult", r"hardcore\s+sex"),
    ("adult", r"explicit\s+(nude|sex|xxx)"),
    ("adult", r"uncensored\s+(adult|sex|nude|xxx)"),
    ("adult", r"erotic\s+webcam"),
    ("adult", r"adult\s+(content\s+)?bypass"),
    ("adult", r"pornographic\s+content"),
    ("adult", r"sex\s+videos?"),
    ("piracy", r"\btorrent(?:s|ing|ed)?\s+(this|the|a|my)\b"),
    ("piracy", r"(download|get|grab)\s+.{0,40}\btorrents?\b"),
    ("piracy", r"\btorrents?\s+(download|file|site|link|from|client)"),
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
    (
        "competitor",
        r"dump\s+(this|the)\s+(app|service|platform|subscription|netflix).{0,20}\s+for",
    ),
    ("competitor", r"dump\s+netflix.{0,20}\s+for"),
    (
        "competitor",
        r"watch\s+(this\s+)?(on|via)\s+(hulu|disney|prime|hbo|max|amazon)\b",
    ),
    ("competitor", r"better\s+(on|than)\s+(hulu|disney|prime|hbo)"),
    ("competitor", r"off-platform"),
    ("competitor", r"competitor\s+streaming"),
    (
        "non_english",
        r"olvida\s+(tus|las|mis|todas(\s+las)?)\s+(reglas|instrucciones|pautas)",
    ),
    (
        "non_english",
        r"ignora\s+(todas\s+)?(las\s+)?instrucciones(\s+anteriores|\s+previas)?",
    ),
    ("non_english", r"actua\s+sin\s+(restricciones|filtros|reglas|limites)"),
    (
        "non_english",
        r"oublie\s+(tes|les|toutes?\s+les)\s+(consignes|r[ee]gles|instructions)",
    ),
    (
        "non_english",
        r"ignore(z|r)?\s+(les|tes|toutes\s+les)\s+instructions\s+pr[ee]c[ee]dentes",
    ),
    (
        "non_english",
        r"ignoriere\s+(alle\s+|die\s+)?vorherigen\s+(anweisungen|instruktionen|regeln)",
    ),
)

# Short tokens we refuse to fire on ASCII titles. After an encoding trick
# (fullwidth, homoglyph, hidden format) the same tokens are an attack.
_ENCODED_EXTRA_SPECS: Final[tuple[tuple[str, str], ...]] = (
    ("encoding", r"\bxxx\b"),
    ("encoding", r"\bjailbreak"),
    ("encoding", r"\btorrents?\b"),
    ("encoding", r"do\s+anything\s+now"),
    ("encoding", r"from\s+now\s+on\s+you\s+will"),
    ("encoding", r"dump\s+(this|netflix).{0,20}\s+for"),
)


@dataclass(frozen=True, slots=True)
class GuardVerdict:
    """Result of inspecting user text. `category` is set only on a block."""

    blocked: bool
    category: str | None = None


def _compile_patterns(
    specs: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, re.Pattern[str]], ...]:
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for category, spec in specs:
        compiled.append((category, re.compile(spec, re.IGNORECASE | re.DOTALL)))
    return tuple(compiled)


_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = _compile_patterns(_PATTERN_SPECS)
_ENCODED_EXTRA: Final[tuple[tuple[str, re.Pattern[str]], ...]] = _compile_patterns(
    _ENCODED_EXTRA_SPECS
)


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


def _unfold(text: str) -> tuple[str, bool]:
    """Fold the query for phrase matching.

    Returns `(normalized, encoded_trick)`. `encoded_trick` is true when the
    query used compatibility characters, hidden format chars, or homoglyphs —
    not when it only changed case or used ordinary accents. ASCII `xXx` is
    therefore not an encoding trick; fullwidth Latin `xxx` is.
    """
    nfkc = unicodedata.normalize("NFKC", text)
    encoded = nfkc != text
    nfd = unicodedata.normalize("NFD", nfkc)
    kept: list[str] = []
    for char in nfd:
        category = unicodedata.category(char)
        code = ord(char)
        if category == "Mn":
            continue
        if category == "Cf" or code in _FORMAT_HIDE:
            encoded = True
            continue
        kept.append(char)
    folded = "".join(kept).casefold()
    mapped = folded.translate(_HOMOGLYPH_TABLE)
    if mapped != folded:
        encoded = True
    normalized = re.sub(r"\s+", " ", mapped).strip()
    return normalized, encoded


def inspect_text(text: str, *, max_chars: int) -> GuardVerdict:
    """Classify user text. Pure: no I/O, no model."""
    if _has_disallowed_control(text):
        return GuardVerdict(blocked=True, category="control")
    if len(text) > max_chars:
        return GuardVerdict(blocked=True, category="length")
    normalized, encoded = _unfold(text)
    if not normalized:
        return GuardVerdict(blocked=False)
    for category, pattern in _PATTERNS:
        if pattern.search(normalized):
            return GuardVerdict(blocked=True, category=category)
    if encoded:
        for category, pattern in _ENCODED_EXTRA:
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
