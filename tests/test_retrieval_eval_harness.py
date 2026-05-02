"""
Tests for the retrieval analyzer v0 evaluation harness.

Covers:
- fixture loader works
- harness runs on small fixture
- metrics JSON is created when requested
- full fixture passes all expected labels (regression guard)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the harness script without requiring it to be a package
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
_HARNESS_PATH = _SCRIPTS_DIR / "evaluate_retrieval_analyzers.py"
_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "retrieval_diagnosis_v0.jsonl"


def _load_harness():
    spec = importlib.util.spec_from_file_location("evaluate_retrieval_analyzers", _HARNESS_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_retrieval_analyzers"] = mod
    spec.loader.exec_module(mod)
    return mod


_harness = _load_harness()

load_fixture = _harness.load_fixture
build_run = _harness.build_run
evaluate_case = _harness.evaluate_case
compute_metrics = _harness.compute_metrics
report_to_dict = _harness.report_to_dict
render_summary = _harness.render_summary
main = _harness.main


# ---------------------------------------------------------------------------
# Fixture loader
# ---------------------------------------------------------------------------

class TestFixtureLoader:
    def test_fixture_file_exists(self):
        assert _FIXTURE_PATH.exists(), f"Fixture not found: {_FIXTURE_PATH}"

    def test_loads_12_cases(self):
        cases = load_fixture(_FIXTURE_PATH)
        assert len(cases) == 12

    def test_all_cases_have_required_keys(self):
        cases = load_fixture(_FIXTURE_PATH)
        required = {"case_id", "query", "retrieved_chunks", "cited_doc_ids",
                    "corpus_entries", "expected_profile", "expected_analyzer_results"}
        for case in cases:
            missing = required - case.keys()
            assert not missing, f"{case['case_id']} is missing keys: {missing}"

    def test_all_case_ids_unique(self):
        cases = load_fixture(_FIXTURE_PATH)
        ids = [c["case_id"] for c in cases]
        assert len(ids) == len(set(ids))

    def test_expected_profile_has_required_fields(self):
        cases = load_fixture(_FIXTURE_PATH)
        profile_fields = {
            "overall_retrieval_status", "phantom_citation_doc_ids",
            "stale_doc_ids", "noisy_chunk_ids", "contradictory_pairs",
            "method_type", "calibration_status", "recommended_for_gating",
        }
        for case in cases:
            missing = profile_fields - case["expected_profile"].keys()
            assert not missing, f"{case['case_id']} expected_profile missing: {missing}"

    def test_expected_analyzer_results_cover_all_four_analyzers(self):
        cases = load_fixture(_FIXTURE_PATH)
        expected_analyzers = {
            "CitationMismatchAnalyzer",
            "ScopeViolationAnalyzer",
            "InconsistentChunksAnalyzer",
            "StaleRetrievalAnalyzer",
        }
        for case in cases:
            actual = set(case["expected_analyzer_results"].keys())
            assert actual == expected_analyzers, f"{case['case_id']} missing analyzers"

    def test_invalid_jsonl_raises_value_error(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text("{not valid json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_fixture(bad)

    def test_empty_lines_are_skipped(self, tmp_path):
        source = _FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        with_blanks = "\n".join(source[:2]) + "\n\n" + "\n".join(source[2:4]) + "\n"
        f = tmp_path / "sparse.jsonl"
        f.write_text(with_blanks, encoding="utf-8")
        cases = load_fixture(f)
        assert len(cases) == 4


# ---------------------------------------------------------------------------
# RAGRun construction
# ---------------------------------------------------------------------------

class TestBuildRun:
    def test_build_run_with_chunks_and_corpus(self):
        cases = load_fixture(_FIXTURE_PATH)
        stale_case = next(c for c in cases if c["case_id"] == "case_06_stale_document_by_age")
        run = build_run(stale_case)
        assert len(run.retrieved_chunks) == 1
        assert len(run.corpus_entries) == 1
        assert run.corpus_entries[0].doc_id == "doc-old"
        assert run.corpus_entries[0].timestamp is not None

    def test_build_run_with_empty_chunks(self):
        cases = load_fixture(_FIXTURE_PATH)
        empty_case = next(c for c in cases if c["case_id"] == "case_10_no_retrieved_chunks")
        run = build_run(empty_case)
        assert run.retrieved_chunks == []

    def test_build_run_cited_doc_ids(self):
        cases = load_fixture(_FIXTURE_PATH)
        phantom_case = next(c for c in cases if c["case_id"] == "case_04_phantom_citation")
        run = build_run(phantom_case)
        assert "doc-phantom" in run.cited_doc_ids

    def test_build_run_has_no_profile_attached(self):
        cases = load_fixture(_FIXTURE_PATH)
        run = build_run(cases[0])
        assert run.retrieval_evidence_profile is None


# ---------------------------------------------------------------------------
# Harness runs on small fixture
# ---------------------------------------------------------------------------

class TestHarnessOnSmallFixture:
    def test_evaluate_case_returns_required_keys(self):
        cases = load_fixture(_FIXTURE_PATH)
        result = evaluate_case(cases[0])
        assert "case_id" in result
        assert "profile_matches" in result
        assert "profile_diffs" in result
        assert "analyzer_results" in result

    def test_evaluate_case_has_all_four_analyzers(self):
        cases = load_fixture(_FIXTURE_PATH)
        result = evaluate_case(cases[0])
        assert set(result["analyzer_results"].keys()) == {
            "CitationMismatchAnalyzer",
            "ScopeViolationAnalyzer",
            "InconsistentChunksAnalyzer",
            "StaleRetrievalAnalyzer",
        }

    def test_evaluate_case_analyzer_result_has_verdict(self):
        cases = load_fixture(_FIXTURE_PATH)
        result = evaluate_case(cases[0])
        for name, ar in result["analyzer_results"].items():
            assert "verdict" in ar, f"{name} missing verdict"
            assert ar["verdict"] in {"correct", "false_positive", "false_negative", "status_mismatch"}

    def test_compute_metrics_structure(self):
        cases = load_fixture(_FIXTURE_PATH)[:3]
        results = [evaluate_case(c) for c in cases]
        report = compute_metrics(results)
        assert report.total_cases == 3
        assert set(report.per_analyzer.keys()) == {
            "CitationMismatchAnalyzer",
            "ScopeViolationAnalyzer",
            "InconsistentChunksAnalyzer",
            "StaleRetrievalAnalyzer",
        }

    def test_render_summary_returns_non_empty_string(self):
        cases = load_fixture(_FIXTURE_PATH)[:2]
        results = [evaluate_case(c) for c in cases]
        report = compute_metrics(results)
        summary = render_summary(report)
        assert isinstance(summary, str)
        assert len(summary) > 50


# ---------------------------------------------------------------------------
# Metrics JSON is created when requested
# ---------------------------------------------------------------------------

class TestMetricsJsonOutput:
    def test_json_file_created(self, tmp_path):
        output_path = tmp_path / "eval_report.json"
        main(fixture_path=_FIXTURE_PATH, output_path=output_path)
        assert output_path.exists()

    def test_json_report_has_required_keys(self, tmp_path):
        output_path = tmp_path / "eval_report.json"
        main(fixture_path=_FIXTURE_PATH, output_path=output_path)
        report = json.loads(output_path.read_text(encoding="utf-8"))
        assert "total_cases" in report
        assert "per_analyzer" in report
        assert "disclaimer" in report
        assert "run_timestamp" in report
        assert "case_mismatches" in report

    def test_json_report_has_all_analyzers(self, tmp_path):
        output_path = tmp_path / "eval_report.json"
        main(fixture_path=_FIXTURE_PATH, output_path=output_path)
        report = json.loads(output_path.read_text(encoding="utf-8"))
        assert set(report["per_analyzer"].keys()) == {
            "CitationMismatchAnalyzer",
            "ScopeViolationAnalyzer",
            "InconsistentChunksAnalyzer",
            "StaleRetrievalAnalyzer",
        }

    def test_json_report_per_analyzer_has_metrics_fields(self, tmp_path):
        output_path = tmp_path / "eval_report.json"
        main(fixture_path=_FIXTURE_PATH, output_path=output_path)
        report = json.loads(output_path.read_text(encoding="utf-8"))
        expected_fields = {
            "total", "exact_match", "exact_match_accuracy",
            "false_positives", "false_negatives", "skip_count",
            "profile_used_count", "fallback_count",
        }
        for name, m in report["per_analyzer"].items():
            missing = expected_fields - m.keys()
            assert not missing, f"{name} missing metric fields: {missing}"

    def test_no_output_file_without_output_path(self, tmp_path):
        # Calling main without output_path should not create any JSON file.
        main(fixture_path=_FIXTURE_PATH, output_path=None)
        assert not any(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# Full fixture regression guard
# ---------------------------------------------------------------------------

class TestFullFixtureRegression:
    """Every fixture case must produce exactly the expected labels.

    These are heuristic regression tests — they prove the code behaves
    consistently, not that it is semantically correct.
    """

    def test_all_cases_exact_match(self):
        cases = load_fixture(_FIXTURE_PATH)
        failures: list[str] = []

        for case in cases:
            result = evaluate_case(case)

            if not result["profile_matches"]:
                failures.append(
                    f"{case['case_id']} profile mismatch: {result['profile_diffs']}"
                )

            for name, ar in result["analyzer_results"].items():
                if not ar["exact_match"]:
                    failures.append(
                        f"{case['case_id']} / {name}: "
                        f"actual={ar['status']} expected={ar['expected_status']}"
                    )

        assert not failures, "Fixture regression failures:\n" + "\n".join(failures)

    def test_overall_accuracy_is_100_percent(self):
        cases = load_fixture(_FIXTURE_PATH)
        results = [evaluate_case(c) for c in cases]
        report = compute_metrics(results)
        for name, m in report.per_analyzer.items():
            assert m.exact_match_accuracy == 1.0, (
                f"{name} exact_match_accuracy={m.exact_match_accuracy:.1%} (expected 100%)"
            )

    def test_no_false_positives_or_negatives(self):
        cases = load_fixture(_FIXTURE_PATH)
        results = [evaluate_case(c) for c in cases]
        report = compute_metrics(results)
        for name, m in report.per_analyzer.items():
            assert m.false_positives == 0, f"{name} has {m.false_positives} FP(s)"
            assert m.false_negatives == 0, f"{name} has {m.false_negatives} FN(s)"

    def test_profile_used_for_all_non_skip_results(self):
        cases = load_fixture(_FIXTURE_PATH)
        results = [evaluate_case(c) for c in cases]
        report = compute_metrics(results)
        for name, m in report.per_analyzer.items():
            assert m.fallback_count == 0, (
                f"{name} used legacy fallback {m.fallback_count} time(s) — "
                "harness always attaches a profile so fallback should never fire"
            )
