"""Closed-domain enumerations. New values belong here, not in call-site string literals."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum


class GenreId(StrEnum):
    """Canonical catalog genres. Raw Netflix labels collapse here in T10."""

    ACTION = "action"
    ANIME = "anime"
    CHILDREN = "children"
    CLASSIC = "classic"
    COMEDY = "comedy"
    CRIME = "crime"
    DOCUMENTARY = "documentary"
    DRAMA = "drama"
    FAITH = "faith"
    HORROR = "horror"
    INDEPENDENT = "independent"
    LGBTQ = "lgbtq"
    MUSIC = "music"
    MYSTERY = "mystery"
    REALITY = "reality"
    ROMANCE = "romance"
    SCIFI = "scifi"
    SPORTS = "sports"
    STAND_UP = "stand_up"
    TEEN = "teen"
    THRILLER = "thriller"


class MoodId(StrEnum):
    COZY = "cozy"
    TENSE = "tense"
    FUNNY = "funny"
    BLEAK = "bleak"
    FEELGOOD = "feelgood"
    THOUGHT_PROVOKING = "thought_provoking"
    DARK = "dark"
    UPLIFTING = "uplifting"
    ROMANTIC = "romantic"
    SCARY = "scary"
    BITTERSWEET = "bittersweet"
    ADVENTUROUS = "adventurous"
    MYSTERIOUS = "mysterious"
    NOSTALGIC = "nostalgic"
    WHOLESOME = "wholesome"
    MELANCHOLIC = "melancholic"


class Audience(StrEnum):
    KIDS = "kids"
    FAMILY = "family"
    TEEN = "teen"
    ADULT = "adult"


class Pace(StrEnum):
    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"


class MediaType(StrEnum):
    FILM = "film"
    SERIES = "series"
    ANY = "any"


class RecencyBias(StrEnum):
    TONIGHT = "tonight"
    WEEKEND = "weekend"
    ANY = "any"


class Package(StrEnum):
    BASIC = "basic"
    PREMIUM = "premium"


class DeviceClass(StrEnum):
    TV = "tv"
    MOBILE = "mobile"
    WEB = "web"
    TABLET = "tablet"


class CreditRole(StrEnum):
    ACTOR = "actor"
    DIRECTOR = "director"


class SpeechAct(StrEnum):
    REFINE_MOOD = "refine_mood"
    REFINE_GENRE = "refine_genre"
    REFINE_DURATION = "refine_duration"
    REFINE_ORIGIN = "refine_origin"
    TOGGLE_LOCAL_ORIGINALS = "toggle_local_originals"
    PERSON_DISAMBIGUATE = "person_disambiguate"
    MORE_LIKE_PICK = "more_like_pick"
    RESET_SOFT = "reset_soft"
    CLARIFY_GENRE = "clarify_genre"
    CLARIFY_MEDIA_TYPE = "clarify_media_type"
    SAFE_REFUSE_CONTINUE = "safe_refuse_continue"


class DegradedReason(StrEnum):
    NONE = "none"
    SAFETY_BLOCK = "safety_block"
    GENERATIVE_TIMEOUT = "generative_timeout"
    GENERATIVE_SCHEMA_FAIL = "generative_schema_fail"
    PROVIDER_THROTTLE = "provider_throttle"
    RETRIEVAL_UNAVAILABLE = "retrieval_unavailable"
    SESSION_STORE_UNAVAILABLE = "session_store_unavailable"
    PERSON_AMBIGUOUS = "person_ambiguous"
    EMPTY_CATALOG_MATCH = "empty_catalog_match"
    HARD_TIMEOUT = "hard_timeout"


class Route(StrEnum):
    TEMPLATE = "template"
    GENERATIVE = "generative"
    CLARIFY = "clarify"
    SAFETY = "safety"
    DEGRADED_KEYWORD = "degraded_keyword"


class DeltaSource(StrEnum):
    """Who produced a ConstraintDelta. Callers apply text/rules first, then chip."""

    CHIP = "chip"
    TEXT = "text"
    RULES = "rules"


class MaturityRating(StrEnum):
    TV_Y = "TV-Y"
    TV_Y7 = "TV-Y7"
    G = "G"
    TV_G = "TV-G"
    PG = "PG"
    TV_PG = "TV-PG"
    PG_13 = "PG-13"
    TV_14 = "TV-14"
    R = "R"
    TV_MA = "TV-MA"
    NC_17 = "NC-17"
    NR = "NR"


# Integer ladder used by Title.maturity_rank and the merge clamp.
# Same-rank pairs (G/TV-G, PG/TV-PG, PG-13/TV-14, R/TV-MA) are equivalent ceilings.
MATURITY_RANK: Mapping[MaturityRating, int] = {
    MaturityRating.TV_Y: 1,
    MaturityRating.TV_Y7: 2,
    MaturityRating.G: 3,
    MaturityRating.TV_G: 3,
    MaturityRating.PG: 4,
    MaturityRating.TV_PG: 4,
    MaturityRating.PG_13: 5,
    MaturityRating.TV_14: 5,
    MaturityRating.R: 6,
    MaturityRating.TV_MA: 6,
    MaturityRating.NC_17: 7,
    MaturityRating.NR: 7,
}


def maturity_rank(rating: MaturityRating) -> int:
    return MATURITY_RANK[rating]


def is_stricter_than(requested: MaturityRating, profile_max: MaturityRating) -> bool:
    """True when `requested` is a lower ceiling than the profile allows."""
    return maturity_rank(requested) < maturity_rank(profile_max)
