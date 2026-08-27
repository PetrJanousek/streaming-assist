"""Versioned Elasticsearch index bodies (settings + mappings)."""

from assist.stores.mappings.people import PEOPLE_INDEX_BODY
from assist.stores.mappings.titles import EMBEDDING_DIMS, TITLES_INDEX_BODY

__all__ = [
    "EMBEDDING_DIMS",
    "PEOPLE_INDEX_BODY",
    "TITLES_INDEX_BODY",
]
