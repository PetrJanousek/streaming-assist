"""Search-time synonym pack for the titles english analyzer.

Inline (not synonyms_path) so index creation does not need a file on the ES node.
"""

from __future__ import annotations

# Solr format. synonym_graph runs at search time only — multi-word expansions
# are unsafe at index time.
CATALOG_SYNONYMS: tuple[str, ...] = (
    "sci-fi, scifi, sci fi, science fiction",
    "feel-good, feelgood, feel good",
    "stand-up, standup, stand up, stand-up comedy",
    "kids, children, children's, childrens",
    "lgbtq, lgbt, queer",
    "romcom, rom-com, romantic comedy",
    "indie, independent",
    "docs, documentary, documentaries",
    "thriller, thrillers",
    "comedy, comedies",
    "drama, dramas",
    "horror, horrors",
    "romance, romantic",
    "action, actioner",
    "anime, manga",
    "thought-provoking, thought provoking, thought_provoking",
    "cozy, cosy",
)
