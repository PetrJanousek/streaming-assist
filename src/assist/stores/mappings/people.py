"""people_vN index settings + mappings (plan §4.2)."""

from __future__ import annotations

from typing import Any

PEOPLE_INDEX_BODY: dict[str, Any] = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "filter": {
                "name_edge_ngram": {
                    "type": "edge_ngram",
                    "min_gram": 2,
                    "max_gram": 12,
                },
            },
            "analyzer": {
                # Prefix index for "nolan" / "the older guy" typeahead-ish match.
                "name_edge": {
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding", "name_edge_ngram"],
                },
                "name_search": {
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding"],
                },
            },
        },
    },
    "mappings": {
        "properties": {
            "person_id": {"type": "keyword"},
            "name": {
                "type": "text",
                "analyzer": "name_edge",
                "search_analyzer": "name_search",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "name_norm": {"type": "keyword"},
            "roles": {"type": "keyword"},
            "active_year_min": {"type": "integer"},
            "active_year_max": {"type": "integer"},
            "popularity": {"type": "float"},
            "credit_count": {"type": "integer"},
        }
    },
}
