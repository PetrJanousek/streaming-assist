"""T28: intent prompt covers extractable constraints and FieldOp set-vs-clear."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from assist.llm.prompts import load_prompt

_REQUIRED_VARS = {"text", "constraints_json", "genre_ids", "mood_ids"}


def test_intent_prompt_input_variables_unchanged() -> None:
    tmpl = ChatPromptTemplate.from_template(load_prompt("intent"))
    assert set(tmpl.input_variables) == _REQUIRED_VARS


def test_intent_prompt_forbids_titles_and_ids() -> None:
    text = load_prompt("intent")
    assert "Never name a title from memory" in text
    assert "catalog_id" in text
    assert "person_id" in text
    assert "person_ids_from_index as an empty list" in text


def test_intent_prompt_documents_fieldop_set_not_clear() -> None:
    text = load_prompt("intent")
    assert "op=set" in text
    assert "op=add" in text
    assert "op=clear" in text
    assert "Never `op=clear`" in text or "never `op=clear`" in text.lower()
    assert "explicit removal" in text


def test_intent_prompt_lists_extractable_constraint_fields() -> None:
    text = load_prompt("intent")
    for field in (
        "origins",
        "year_min",
        "year_max",
        "duration_max_min",
        "media_type",
        "genres_include",
        "moods",
        "recency_bias",
        "local_originals_only",
    ):
        assert field in text
    assert "Never emit languages" in text or "Never emit languages." in text


def test_intent_prompt_origin_vocabulary_from_index() -> None:
    text = load_prompt("intent")
    for origin in (
        "United States",
        "India",
        "United Kingdom",
        "Canada",
        "France",
        "Japan",
        "Spain",
        "South Korea",
        "Germany",
        "Mexico",
        "China",
        "Australia",
        "Egypt",
        "Turkey",
        "Hong Kong",
        "Czech Republic",
    ):
        assert origin in text


def test_intent_prompt_maps_language_requests_to_origins() -> None:
    text = load_prompt("intent")
    assert "in French → France" in text
    assert "Korean / Korea / in Korean → South Korea" in text
    assert "Czech / Czechia / in Czech → Czech Republic" in text
    assert "do not emit languages" in text.lower()
