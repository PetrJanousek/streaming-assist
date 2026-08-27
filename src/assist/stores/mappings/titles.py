"""titles_vN index settings + mappings (plan §4.2)."""

from __future__ import annotations

from typing import Any

from assist.stores.mappings.synonyms import CATALOG_SYNONYMS

# bge-small-en-v1.5. Changing this requires a new index version (T12).
EMBEDDING_DIMS = 384

_TEXT = {
    "type": "text",
    "analyzer": "english_index",
    "search_analyzer": "english_search",
}

TITLES_INDEX_BODY: dict[str, Any] = {
    "settings": {
        # Single-node compose has no replica to allocate; 0 keeps the index green.
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "filter": {
                "english_possessive": {
                    "type": "stemmer",
                    "language": "possessive_english",
                },
                "english_stop": {"type": "stop", "stopwords": "_english_"},
                "english_stemmer": {"type": "stemmer", "language": "english"},
                "catalog_synonyms": {
                    "type": "synonym_graph",
                    "synonyms": list(CATALOG_SYNONYMS),
                },
            },
            "analyzer": {
                "english_index": {
                    "tokenizer": "standard",
                    "filter": [
                        "english_possessive",
                        "lowercase",
                        "english_stop",
                        "english_stemmer",
                    ],
                },
                "english_search": {
                    "tokenizer": "standard",
                    "filter": [
                        "english_possessive",
                        "lowercase",
                        "catalog_synonyms",
                        "english_stop",
                        "english_stemmer",
                    ],
                },
            },
        },
    },
    "mappings": {
        "properties": {
            "catalog_id": {"type": "keyword"},
            "media_type": {"type": "keyword"},
            "genres": {"type": "keyword"},
            "moods": {"type": "keyword"},
            "origins": {"type": "keyword"},
            "maturity_rank": {"type": "integer"},
            "local_original": {"type": "boolean"},
            "release_year": {"type": "integer"},
            "runtime_min": {"type": "integer"},
            "audience": {"type": "keyword"},
            "pace": {"type": "keyword"},
            "people_ids": {"type": "keyword"},
            "pop_28d": {"type": "float"},
            "title": {
                **_TEXT,
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "synopsis": _TEXT,
            "tags": {
                **_TEXT,
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "people_names": _TEXT,
            "era_feel": _TEXT,
            "embedding": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIMS,
                "index": True,
                "similarity": "cosine",
                "index_options": {
                    "type": "hnsw",
                    "m": 16,
                    "ef_construction": 100,
                },
            },
        }
    },
}
