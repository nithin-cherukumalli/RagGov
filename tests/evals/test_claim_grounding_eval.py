"""
Tests for the claim-grounding evaluation harness (PR 6).

Tests cover:
- Schema validation for valid and invalid records
- load_dataset parses all seed cases without error
- compute_metrics produces correct values on a tiny controlled fixture
- false_pass_rate calculation matches definition exactly
- false_fail_rate calculation matches definition exactly
- evidence_chunk_recall calculation
- contradiction_detection_rate calculation
- fallback_rate calculation
- Overall accuracy
- Edge cases: empty dataset, all-correct, all-wrong
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Resolve paths so tests work from any working directory
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = _REPO_ROOT / "evals" / "claim_grounding"
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from evals.claim_grounding.schema import (  # noqa: E402
    ChunkRecord,
    ClaimGroundingCase,
)
from evals.claim_grounding.run_eval import (  # noqa: E402
    compute_metrics,
    load_dataset,
    render_markdown,
)
from raggov.analyzers.grounding.verifiers import VerificationResult  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_case(
    case_id: str,
    gold_label: str,
    gold_supporting_chunk_ids: list[str] | None = None,
    gold_contradicting_chunk_ids: list[str] | None = None,
    error_type: str | None = None,
) -> ClaimGroundingCase:
    return ClaimGroundingCase(
        case_id=case_id,
        query="test query",
        answer="test answer",
        claim_text="The amount is 100.",
        retrieved_chunks=[
            ChunkRecord(
                chunk_id="chunk-1",
                text="The amount is 100.",
                source_doc_id="doc-1",
                score=0.9,
            )
        ],
        cited_doc_ids=["doc-1"],
        gold_label=gold_label,  # type: ignore[arg-type]
        gold_supporting_chunk_ids=gold_supporting_chunk_ids or [],
        gold_contradicting_chunk_ids=gold_contradicting_chunk_ids or [],
        claim_type="numeric",
        atomicity_status="atomic",
        error_type=error_type,  # type: ignore[arg-type]
    )


def _make_result(
    label: str,
    supporting_chunk_ids: list[str] | None = None,
    fallback_used: bool = False,
) -> VerificationResult:
    return VerificationResult(
        label=label,  # type: ignore[arg-type]
        verifier_name="test_verifier",
        raw_score=0.8,
        evidence_chunk_id=None,
        evidence_span=None,
        rationale="test rationale",
        supporting_chunk_ids=supporting_chunk_ids or [],
        contradicting_chunk_ids=[],
        value_matches=[],
        value_conflicts=[],
        fallback_used=fallback_used,
    )



# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_valid_minimal_case(self) -> None:
        case = _make_case("cgc-test-001", "entailed")
        assert case.case_id == "cgc-test-001"
        assert case.gold_label == "entailed"
        assert case.error_type is None

    def test_valid_unsupported_with_error_type(self) -> None:
        case = _make_case("cgc-test-002", "unsupported", error_type="insufficient_context")
        assert case.gold_label == "unsupported"
        assert case.error_type == "insufficient_context"

    def test_valid_contradicted_with_error_type(self) -> None:
        case = _make_case("cgc-test-003", "contradicted", error_type="value_error")
        assert case.gold_label == "contradicted"
        assert case.error_type == "value_error"

    def test_invalid_gold_label_raises(self) -> None:
        with pytest.raises(Exception):
            ClaimGroundingCase(
                case_id="bad",
                query="q",
                answer="a",
                claim_text="c",
                retrieved_chunks=[],
                gold_label="wrong_value",  # type: ignore[arg-type]
                claim_type="numeric",
                atomicity_status="atomic",
            )

    def test_invalid_claim_type_raises(self) -> None:
        with pytest.raises(Exception):
            ClaimGroundingCase(
                case_id="bad",
                query="q",
                answer="a",
                claim_text="c",
                retrieved_chunks=[],
                gold_label="entailed",
                claim_type="not_a_real_type",  # type: ignore[arg-type]
                atomicity_status="atomic",
            )

    def test_invalid_atomicity_status_raises(self) -> None:
        with pytest.raises(Exception):
            ClaimGroundingCase(
                case_id="bad",
                query="q",
                answer="a",
                claim_text="c",
                retrieved_chunks=[],
                gold_label="entailed",
                claim_type="numeric",
                atomicity_status="not_valid",  # type: ignore[arg-type]
            )

    def test_all_valid_error_types_accepted(self) -> None:
        valid_error_types = [
            "retrieval_miss",
            "context_ignored",
            "value_error",
            "stale_source_error",
            "citation_error",
            "generation_hallucination",
            "insufficient_context",
        ]
        for et in valid_error_types:
            case = _make_case("cgc-x", "unsupported", error_type=et)
            assert case.error_type == et

    def test_all_valid_claim_types_accepted(self) -> None:
        valid_types = [
            "numeric", "date_or_deadline", "go_number",
            "definition", "eligibility", "policy_rule", "general_factual",
        ]
        for ct in valid_types:
            case = ClaimGroundingCase(
                case_id="x",
                query="q",
                answer="a",
                claim_text="c",
                retrieved_chunks=[],
                gold_label="entailed",
                claim_type=ct,  # type: ignore[arg-type]
                atomicity_status="atomic",
            )
            assert case.claim_type == ct


# ---------------------------------------------------------------------------
# Dataset loading tests
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_seed_cases_load_without_error(self) -> None:
        """All 25 seed cases must pass schema validation."""
        cases = load_dataset(_EVAL_DIR / "seed_cases.jsonl")
        assert len(cases) >= 20, f"Expected at least 20 seed cases, got {len(cases)}"

    def test_seed_cases_have_required_fields(self) -> None:
        cases = load_dataset(_EVAL_DIR / "seed_cases.jsonl")
        for case in cases:
            assert case.case_id
            assert case.query
            assert case.claim_text
            assert case.gold_label in ("entailed", "unsupported", "contradicted")

    def test_seed_cases_cover_all_labels(self) -> None:
        cases = load_dataset(_EVAL_DIR / "seed_cases.jsonl")
        labels = {c.gold_label for c in cases}
        assert "entailed" in labels
        assert "unsupported" in labels
        assert "contradicted" in labels

    def test_invalid_jsonl_line_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.jsonl"
        bad_file.write_text('{"not_a_valid_schema": true}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="invalid record"):
            load_dataset(bad_file)

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        seed = _EVAL_DIR / "seed_cases.jsonl"
        lines = seed.read_text(encoding="utf-8").splitlines()
        padded = tmp_path / "padded.jsonl"
        padded.write_text("\n".join([""] + lines + [""]), encoding="utf-8")
        cases = load_dataset(padded)
        assert len(cases) == load_dataset(seed).__len__()


# ---------------------------------------------------------------------------
# Metric computation tests: controlled fixtures
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    """Use fully controlled case+prediction pairs to verify metric formulas."""

    def test_all_correct_gives_perfect_accuracy(self) -> None:
        cases = [
            _make_case("c1", "entailed"),
            _make_case("c2", "unsupported"),
            _make_case("c3", "contradicted"),
        ]
        preds = [
            _make_result("entailed"),
            _make_result("unsupported"),
            _make_result("contradicted"),
        ]
        m = compute_metrics(cases, preds)
        assert m["overall_accuracy"] == 1.0
        assert m["false_pass_rate"] == 0.0
        assert m["false_fail_rate"] == 0.0

    def test_all_wrong_gives_zero_accuracy(self) -> None:
        cases = [
            _make_case("c1", "entailed"),
            _make_case("c2", "unsupported"),
            _make_case("c3", "contradicted"),
        ]
        preds = [
            _make_result("unsupported"),
            _make_result("contradicted"),
            _make_result("entailed"),
        ]
        m = compute_metrics(cases, preds)
        assert m["overall_accuracy"] == 0.0

    def test_empty_dataset_returns_empty_dict(self) -> None:
        m = compute_metrics([], [])
        assert m == {}

    def test_false_pass_rate_definition(self) -> None:
        """
        false_pass_rate = |{cases where gold ∈ {unsupported, contradicted}
                            AND predicted = entailed}|
                        / |{cases where gold ∈ {unsupported, contradicted}}|
        """
        cases = [
            _make_case("c1", "unsupported"),     # false-passable
            _make_case("c2", "contradicted"),    # false-passable
            _make_case("c3", "entailed"),        # not in denominator
        ]
        preds = [
            _make_result("entailed"),   # false pass!
            _make_result("contradicted"),  # correct
            _make_result("entailed"),   # correct
        ]
        m = compute_metrics(cases, preds)
        # Denominator = 2 (c1, c2); numerator = 1 (c1 predicted entailed)
        assert m["false_pass_rate"] == pytest.approx(0.5)
        assert m["raw_counts"]["false_pass"] == 1

    def test_false_pass_rate_all_safe(self) -> None:
        """No false passes when all unsupported/contradicted are caught."""
        cases = [
            _make_case("c1", "unsupported"),
            _make_case("c2", "contradicted"),
        ]
        preds = [
            _make_result("unsupported"),
            _make_result("contradicted"),
        ]
        m = compute_metrics(cases, preds)
        assert m["false_pass_rate"] == 0.0

    def test_false_pass_rate_with_zero_negative_cases(self) -> None:
        """When there are no unsupported/contradicted cases, false_pass_rate = 0."""
        cases = [_make_case("c1", "entailed")]
        preds = [_make_result("entailed")]
        m = compute_metrics(cases, preds)
        assert m["false_pass_rate"] == 0.0

    def test_false_fail_rate_definition(self) -> None:
        """
        false_fail_rate = |{cases where gold = entailed AND predicted ≠ entailed}|
                        / |{cases where gold = entailed}|
        """
        cases = [
            _make_case("c1", "entailed"),   # over-rejected
            _make_case("c2", "entailed"),   # correct
            _make_case("c3", "unsupported"),  # not in denominator
        ]
        preds = [
            _make_result("unsupported"),  # false fail!
            _make_result("entailed"),     # correct
            _make_result("unsupported"),  # correct
        ]
        m = compute_metrics(cases, preds)
        # Denominator = 2 (c1, c2); numerator = 1 (c1 incorrectly rejected)
        assert m["false_fail_rate"] == pytest.approx(0.5)
        assert m["raw_counts"]["false_fail"] == 1

    def test_evidence_chunk_recall_definition(self) -> None:
        """
        For entailed cases with gold_supporting_chunk_ids:
        recall = |predicted ∩ gold| / |gold|
        """
        # Two entailed cases: one perfect, one partial
        cases = [
            _make_case("c1", "entailed", gold_supporting_chunk_ids=["chunk-a", "chunk-b"]),
            _make_case("c2", "entailed", gold_supporting_chunk_ids=["chunk-c"]),
        ]
        preds = [
            _make_result("entailed", supporting_chunk_ids=["chunk-a"]),   # 1/2 recall
            _make_result("entailed", supporting_chunk_ids=["chunk-c"]),   # 1/1 recall
        ]
        m = compute_metrics(cases, preds)
        # Average recall = (0.5 + 1.0) / 2 = 0.75
        assert m["evidence_chunk_recall"] == pytest.approx(0.75)

    def test_evidence_chunk_recall_zero_when_no_match(self) -> None:
        cases = [
            _make_case("c1", "entailed", gold_supporting_chunk_ids=["chunk-gold"]),
        ]
        preds = [
            _make_result("entailed", supporting_chunk_ids=["chunk-wrong"]),
        ]
        m = compute_metrics(cases, preds)
        assert m["evidence_chunk_recall"] == 0.0

    def test_evidence_chunk_recall_skips_non_entailed(self) -> None:
        """Evidence chunk recall only applies to entailed cases."""
        cases = [
            _make_case("c1", "unsupported"),
            _make_case("c2", "contradicted"),
        ]
        preds = [
            _make_result("unsupported"),
            _make_result("contradicted"),
        ]
        m = compute_metrics(cases, preds)
        # No entailed cases → recall denominator = 0 → result = 0.0
        assert m["evidence_chunk_recall"] == 0.0

    def test_contradiction_detection_rate(self) -> None:
        """Rate at which contradicted claims are correctly predicted as contradicted."""
        cases = [
            _make_case("c1", "contradicted"),  # caught
            _make_case("c2", "contradicted"),  # missed (predicted unsupported)
            _make_case("c3", "contradicted"),  # missed (predicted entailed)
        ]
        preds = [
            _make_result("contradicted"),
            _make_result("unsupported"),
            _make_result("entailed"),
        ]
        m = compute_metrics(cases, preds)
        # 1 out of 3 contradictions caught
        assert m["contradiction_detection_rate"] == pytest.approx(1 / 3, abs=1e-4)

    def test_fallback_rate(self) -> None:
        cases = [
            _make_case("c1", "entailed"),
            _make_case("c2", "entailed"),
            _make_case("c3", "unsupported"),
        ]
        preds = [
            _make_result("entailed", fallback_used=True),
            _make_result("entailed", fallback_used=False),
            _make_result("unsupported", fallback_used=True),
        ]
        m = compute_metrics(cases, preds)
        # 2 out of 3 used fallback
        assert m["fallback_rate"] == pytest.approx(2 / 3, abs=1e-4)

    def test_per_label_f1_perfect(self) -> None:
        cases = [
            _make_case("c1", "entailed"),
            _make_case("c2", "unsupported"),
            _make_case("c3", "contradicted"),
        ]
        preds = [
            _make_result("entailed"),
            _make_result("unsupported"),
            _make_result("contradicted"),
        ]
        m = compute_metrics(cases, preds)
        for label in ("entailed", "unsupported", "contradicted"):
            assert m["label_metrics"][label]["f1"] == 1.0

    def test_per_label_precision_recall(self) -> None:
        """
        Entailed: gold=[c1], predicted_as_entailed=[c1, c3]
        c1: gold=entailed, pred=entailed → TP
        c2: gold=unsupported, pred=unsupported → not relevant
        c3: gold=contradicted, pred=entailed → FP (for entailed)
        """
        cases = [
            _make_case("c1", "entailed"),
            _make_case("c2", "unsupported"),
            _make_case("c3", "contradicted"),
        ]
        preds = [
            _make_result("entailed"),     # TP for entailed
            _make_result("unsupported"),  # correct
            _make_result("entailed"),     # FP for entailed
        ]
        m = compute_metrics(cases, preds)
        em = m["label_metrics"]["entailed"]
        # TP=1, FP=1 → precision=0.5; TP=1, FN=0 → recall=1.0; F1=2/3
        assert em["precision"] == pytest.approx(0.5)
        assert em["recall"] == pytest.approx(1.0)
        assert em["f1"] == pytest.approx(2 / 3, abs=1e-3)


# ---------------------------------------------------------------------------
# Markdown rendering test
# ---------------------------------------------------------------------------

class TestRenderMarkdown:
    def test_markdown_contains_required_sections(self) -> None:
        cases = [_make_case("c1", "entailed")]
        preds = [_make_result("entailed")]
        metrics = compute_metrics(cases, preds)
        md = render_markdown(metrics, "HeuristicValueOverlapVerifier", Path("seed_cases.jsonl"))
        assert "# Claim-Grounding Evaluation Report" in md
        assert "false_pass_rate" in md.lower() or "False-pass" in md
        assert "entailed" in md
        assert "unsupported" in md
        assert "contradicted" in md

    def test_markdown_renders_percentages(self) -> None:
        cases = [_make_case("c1", "entailed")]
        preds = [_make_result("entailed")]
        metrics = compute_metrics(cases, preds)
        md = render_markdown(metrics, "HeuristicValueOverlapVerifier", Path("seed_cases.jsonl"))
        assert "100.0%" in md or "100%" in md


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------

class TestRunEvalEndToEnd:
    def test_eval_runs_on_seed_dataset(self) -> None:
        """Smoke test: run_eval should complete without raising."""
        from evals.claim_grounding.run_eval import run_eval
        metrics = run_eval(dataset_path=_EVAL_DIR / "seed_cases.jsonl")
        assert "overall_accuracy" in metrics
        assert "false_pass_rate" in metrics
        assert metrics["total_cases"] >= 20

    def test_eval_writes_json_report(self, tmp_path: Path) -> None:
        from evals.claim_grounding.run_eval import run_eval
        json_path = tmp_path / "report.json"
        run_eval(
            dataset_path=_EVAL_DIR / "seed_cases.jsonl",
            json_out=json_path,
        )
        assert json_path.exists()
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert "overall_accuracy" in payload
        assert "label_metrics" in payload

    def test_eval_writes_markdown_report(self, tmp_path: Path) -> None:
        from evals.claim_grounding.run_eval import run_eval
        md_path = tmp_path / "report.md"
        run_eval(
            dataset_path=_EVAL_DIR / "seed_cases.jsonl",
            md_out=md_path,
        )
        assert md_path.exists()
        content = md_path.read_text(encoding="utf-8")
        assert "# Claim-Grounding Evaluation Report" in content
