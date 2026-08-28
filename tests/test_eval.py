"""Golden-set eval harness: stratification, metrics, determinism, report header."""

from __future__ import annotations

from pathlib import Path

from assist.domain.enums import DegradedReason, Route
from assist.jobs.eval import (
    REQUIRED_SLICES,
    Scorecard,
    aggregate,
    default_report_path,
    golden_dir,
    llm_cache_for,
    load_catalog,
    load_queries,
    person_at_1,
    recall_at_k,
    render_report,
    run,
    run_async,
)
from assist.nodes.intent import IntentClass, normalize_text

ROOT = Path(__file__).resolve().parents[1]
QUERIES_PATH = ROOT / "data" / "golden" / "queries.jsonl"
CATALOG_PATH = ROOT / "data" / "golden" / "catalog.json"


def test_golden_set_size_and_stratification() -> None:
    queries = load_queries(QUERIES_PATH)
    assert len(queries) == 60
    counts = {name: 0 for name in REQUIRED_SLICES}
    for query in queries:
        counts[query.slice] += 1
    for name in REQUIRED_SLICES:
        assert counts[name] > 0, name
    person_share = counts["person_fuzzy"] / len(queries)
    assert person_share >= 0.15
    ids = [query.id for query in queries]
    assert len(ids) == len(set(ids))


def test_recall_at_8_unit() -> None:
    assert recall_at_k(("s1", "s2", "s3"), ("s1", "x", "s3"), k=8) == 2 / 3
    assert recall_at_k(("s1",), ("s2", "s3"), k=8) == 0.0
    miss = tuple(f"x{i}" for i in range(8))
    assert recall_at_k(("a", "b", "c", "d", "e", "f", "g", "h", "i"), miss, k=8) == 0.0
    # Large gold set: 8 hits in top-8 is perfect under the cap.
    gold = tuple(f"s{i}" for i in range(15))
    got = tuple(f"s{i}" for i in range(8))
    assert recall_at_k(gold, got, k=8) == 1.0
    assert recall_at_k((), ("s1",)) is None


def test_person_at_1_unit() -> None:
    assert person_at_1("p1", ("p1", "p2")) is True
    assert person_at_1("p1", ("p2",)) is False
    assert person_at_1("p1", ()) is False
    assert person_at_1(None, ("p1",)) is None


def test_catalog_fixture_covers_expect_ids() -> None:
    queries = load_queries(QUERIES_PATH)
    titles, people = load_catalog(CATALOG_PATH)
    title_ids = {str(item["catalog_id"]) for item in titles}
    person_ids = {person.person_id for person in people}
    for query in queries:
        for catalog_id in query.expect_ids:
            assert catalog_id in title_ids, f"{query.id} missing {catalog_id}"
        if query.expect_person_id:
            assert query.expect_person_id in person_ids, query.id


def test_llm_cache_covers_free_text() -> None:
    queries = load_queries(QUERIES_PATH)
    cache = llm_cache_for(queries)
    for query in queries:
        if query.slice in {"mood_genre", "person_fuzzy", "vague"} and query.expect_class not in {
            IntentClass.PURE_GENRE_FACET.value,
            IntentClass.PURE_DECADE.value,
            IntentClass.DURATION_ONLY.value,
        }:
            assert normalize_text(query.text) in cache, query.id


def test_report_header_names_synthetic_fields() -> None:
    scorecard = Scorecard(
        n_queries=1,
        recall_at_8=0.0,
        person_at_1=0.0,
        schema_failure_rate=0.0,
        degraded_rate=0.0,
        usd_per_turn=0.0,
        route_mix={route.value: 0.0 for route in Route},
        latency_p50_ms={"reply": 0.0},
        latency_p95_ms={"reply": 0.0},
        slice_counts={name: 0 for name in REQUIRED_SLICES},
        mode="fixture",
        seed=26,
    )
    markdown = render_report(scorecard)
    header = markdown.split("## Scorecard", 1)[0].lower()
    assert "synthetic" in header
    assert "availability" in header
    assert "pop_28d" in header
    assert "llm" in header
    assert "latency" in header


def _run_harness(tmp_path: Path) -> Scorecard:
    return run(
        live=False,
        report_path=tmp_path / "eval-report.md",
        queries_path=QUERIES_PATH,
        catalog_path=CATALOG_PATH,
        seed=26,
    )


def test_eval_invokes_graph_and_populates_metrics(tmp_path: Path) -> None:
    scorecard = _run_harness(tmp_path)
    assert scorecard.n_queries == 60
    assert scorecard.mode == "fixture"
    assert scorecard.seed == 26
    assert 0.0 <= scorecard.recall_at_8 <= 1.0
    assert scorecard.recall_at_8 > 0.0
    assert 0.0 <= scorecard.person_at_1 <= 1.0
    assert scorecard.person_at_1 > 0.0
    assert 0.0 <= scorecard.schema_failure_rate <= 1.0
    assert 0.0 <= scorecard.degraded_rate <= 1.0
    assert scorecard.usd_per_turn >= 0.0
    for route in Route:
        assert route.value in scorecard.route_mix
    assert scorecard.route_mix[Route.TEMPLATE.value] > 0.0
    assert scorecard.route_mix[Route.SAFETY.value] > 0.0
    for stage, value in scorecard.latency_p50_ms.items():
        assert stage in scorecard.latency_p95_ms
        assert value >= 0.0
        assert scorecard.latency_p95_ms[stage] >= value
    assert scorecard.latency_p50_ms
    assert scorecard.latency_p95_ms
    report = (tmp_path / "eval-report.md").read_text(encoding="utf-8")
    assert "| recall@8 |" in report
    assert "| person@1 |" in report
    assert "schema-failure rate" in report
    assert "USD per turn" in report
    assert "p50" in report
    assert "Synthetic fixtures" in report


def test_rerun_with_seed_and_cached_llm_is_deterministic(tmp_path: Path) -> None:
    first = run(
        live=False,
        report_path=tmp_path / "a.md",
        queries_path=QUERIES_PATH,
        catalog_path=CATALOG_PATH,
        seed=26,
    )
    second = run(
        live=False,
        report_path=tmp_path / "b.md",
        queries_path=QUERIES_PATH,
        catalog_path=CATALOG_PATH,
        seed=26,
    )
    assert first.canonical_metrics() == second.canonical_metrics()
    assert (tmp_path / "a.md").read_text(encoding="utf-8") == (tmp_path / "b.md").read_text(
        encoding="utf-8"
    )


def test_adversarial_slice_present() -> None:
    queries = [item for item in load_queries(QUERIES_PATH) if item.slice == "adversarial"]
    assert len(queries) >= 8
    empty = aggregate([], mode="fixture", seed=26)
    assert empty.schema_failure_rate == 0.0
    assert empty.recall_at_8 == 0.0


def test_golden_dir_and_default_report_path() -> None:
    assert golden_dir() == ROOT / "data" / "golden"
    assert default_report_path() == ROOT / "docs" / "eval-report.md"


async def test_run_async_writes_report(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "eval-report.md"
    scorecard = await run_async(
        live=False,
        report_path=path,
        queries_path=QUERIES_PATH,
        catalog_path=CATALOG_PATH,
        seed=26,
    )
    assert path.is_file()
    body = path.read_text(encoding="utf-8")
    assert f"{scorecard.n_queries}" in body
    assert DegradedReason.GENERATIVE_SCHEMA_FAIL.value not in body or "schema-failure" in body
