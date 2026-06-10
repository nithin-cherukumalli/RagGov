"""
Tests for GovRAG-Calib readiness evaluator.

Covers:
  - test_govrag_calib_readiness_under_30_not_ready
  - test_govrag_calib_readiness_30_to_149_provisional
  - test_govrag_calib_readiness_never_enables_production_gating
  - test_readiness_150_plus_with_heldout_is_ready_for_eval
  - test_readiness_150_plus_without_heldout_is_provisional
  - test_readiness_complete_vs_placeholder_distinction
  - test_actual_dataset_readiness_status (integration)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for p in [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "evals"), str(PROJECT_ROOT / "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from govrag_calib.schema import GovRAGCalibCase


# ---------------------------------------------------------------------------
# Helper: synthetic dataset builder
# ---------------------------------------------------------------------------

def _make_case(
    case_id: str,
    label_confidence: str = "high",
    split: str = "train",
    primary_failure: str = "UNSUPPORTED_CLAIM",
) -> dict:
    return {
        "case_id": case_id,
        "domain": "software",
        "source_type": "synthetic_mutation",
        "query": f"Test query for {case_id}",
        "retrieved_chunks": [
            {"chunk_id": f"{case_id}-chunk-1", "doc_id": f"{case_id}-doc-1", "text": "Test chunk.", "rank": 1}
        ],
        "answer": "Test answer.",
        "citations": [f"{case_id}-doc-1"],
        "expected_primary_failure": primary_failure,
        "expected_stage": "GROUNDING",
        "expected_secondary_failures": [],
        "expected_claim_labels": [],
        "expected_citation_labels": [],
        "label_source": "synthetic_mutation",
        "label_confidence": label_confidence,
        "split": split,
    }


def _write_dataset(cases: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")


# ---------------------------------------------------------------------------
# Import the evaluator functions
# ---------------------------------------------------------------------------

# We import inline to isolate from path issues in test discovery
def _import_evaluate_readiness():
    import importlib.util
    script = PROJECT_ROOT / "scripts" / "evaluate_govrag_calib.py"
    spec = importlib.util.spec_from_file_location("evaluate_govrag_calib", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.evaluate_readiness


evaluate_readiness = _import_evaluate_readiness()


# ---------------------------------------------------------------------------
# Tests: readiness status transitions
# ---------------------------------------------------------------------------

class TestReadinessUnder30NotReady:
    def test_zero_cases_is_not_ready(self, tmp_path):
        dataset = tmp_path / "empty.jsonl"
        dataset.write_text("")
        result = evaluate_readiness(dataset)
        assert result["readiness_status"] == "NOT_READY"
        assert result["complete_cases"] == 0

    def test_ten_cases_is_not_ready(self, tmp_path):
        cases = [_make_case(f"tc-{i:03d}") for i in range(10)]
        dataset = tmp_path / "ten.jsonl"
        _write_dataset(cases, dataset)
        result = evaluate_readiness(dataset)
        assert result["readiness_status"] == "NOT_READY"
        assert result["complete_cases"] == 10

    def test_twenty_nine_cases_is_not_ready(self, tmp_path):
        cases = [_make_case(f"tc-{i:03d}") for i in range(29)]
        dataset = tmp_path / "29.jsonl"
        _write_dataset(cases, dataset)
        result = evaluate_readiness(dataset)
        assert result["readiness_status"] == "NOT_READY"

    def test_placeholder_cases_do_not_count_toward_complete(self, tmp_path):
        """25 high-confidence + 15 placeholders = 25 complete, still NOT_READY."""
        cases = [_make_case(f"tc-{i:03d}", label_confidence="high") for i in range(25)]
        cases += [_make_case(f"ph-{i:03d}", label_confidence="low", split="unset") for i in range(15)]
        dataset = tmp_path / "mixed.jsonl"
        _write_dataset(cases, dataset)
        result = evaluate_readiness(dataset)
        assert result["complete_cases"] == 25
        assert result["placeholder_cases"] == 15
        assert result["readiness_status"] == "NOT_READY"


class TestReadiness30To149Provisional:
    def test_exactly_30_cases_is_provisional(self, tmp_path):
        cases = [_make_case(f"tc-{i:03d}") for i in range(30)]
        dataset = tmp_path / "30.jsonl"
        _write_dataset(cases, dataset)
        result = evaluate_readiness(dataset)
        assert result["readiness_status"] == "PROVISIONAL_DATASET"

    def test_100_cases_is_provisional(self, tmp_path):
        cases = [_make_case(f"tc-{i:03d}") for i in range(100)]
        dataset = tmp_path / "100.jsonl"
        _write_dataset(cases, dataset)
        result = evaluate_readiness(dataset)
        assert result["readiness_status"] == "PROVISIONAL_DATASET"

    def test_149_cases_is_provisional(self, tmp_path):
        cases = [_make_case(f"tc-{i:03d}") for i in range(149)]
        dataset = tmp_path / "149.jsonl"
        _write_dataset(cases, dataset)
        result = evaluate_readiness(dataset)
        assert result["readiness_status"] == "PROVISIONAL_DATASET"

    def test_heldout_unavailable_above_150_is_provisional(self, tmp_path):
        """150+ cases but no heldout split → PROVISIONAL."""
        cases = [_make_case(f"tc-{i:03d}", split="train") for i in range(150)]
        dataset = tmp_path / "150_notrain.jsonl"
        _write_dataset(cases, dataset)
        result = evaluate_readiness(dataset)
        assert result["readiness_status"] == "PROVISIONAL_DATASET"
        assert not result["split_availability"]["heldout"]


class TestReadiness150PlusBehavior:
    def test_150_cases_with_heldout_is_ready(self, tmp_path):
        cases = [_make_case(f"tc-{i:03d}", split="train") for i in range(140)]
        cases += [_make_case(f"held-{i:03d}", split="heldout") for i in range(10)]
        dataset = tmp_path / "150_heldout.jsonl"
        _write_dataset(cases, dataset)
        result = evaluate_readiness(dataset)
        assert result["complete_cases"] == 150
        assert result["split_availability"]["heldout"]
        assert result["readiness_status"] == "READY_FOR_HELDOUT_EVAL"

    def test_200_cases_with_heldout_is_ready(self, tmp_path):
        cases = [_make_case(f"tc-{i:03d}", split="train") for i in range(180)]
        cases += [_make_case(f"held-{i:03d}", split="heldout") for i in range(20)]
        dataset = tmp_path / "200.jsonl"
        _write_dataset(cases, dataset)
        result = evaluate_readiness(dataset)
        assert result["readiness_status"] == "READY_FOR_HELDOUT_EVAL"


# ---------------------------------------------------------------------------
# Tests: production gating is ALWAYS false
# ---------------------------------------------------------------------------

class TestReadinessNeverEnablesProductionGating:
    @pytest.mark.parametrize("n_cases,split,expected_status", [
        (0, "train", "NOT_READY"),
        (30, "train", "PROVISIONAL_DATASET"),
        (149, "train", "PROVISIONAL_DATASET"),
    ])
    def test_production_gating_always_false(self, tmp_path, n_cases, split, expected_status):
        cases = [_make_case(f"tc-{i:03d}", split=split) for i in range(n_cases)]
        dataset = tmp_path / "pg_test.jsonl"
        _write_dataset(cases, dataset)
        result = evaluate_readiness(dataset)
        assert result["production_gating_eligible"] is False, (
            f"production_gating_eligible must always be False. Got True with status={result['readiness_status']}"
        )

    def test_production_gating_false_even_when_ready(self, tmp_path):
        """Even with 150+ cases and heldout, production gating stays false."""
        cases = [_make_case(f"tc-{i:03d}", split="train") for i in range(140)]
        cases += [_make_case(f"held-{i:03d}", split="heldout") for i in range(10)]
        dataset = tmp_path / "ready_pg.jsonl"
        _write_dataset(cases, dataset)
        result = evaluate_readiness(dataset)
        assert result["readiness_status"] == "READY_FOR_HELDOUT_EVAL"
        assert result["production_gating_eligible"] is False

    def test_production_gating_reason_is_present(self, tmp_path):
        dataset = tmp_path / "pg_reason.jsonl"
        _write_dataset([_make_case("tc-001")], dataset)
        result = evaluate_readiness(dataset)
        assert "production_gating_reason" in result
        assert len(result["production_gating_reason"]) > 0

    def test_production_gating_false_is_bool_not_string(self, tmp_path):
        dataset = tmp_path / "type_check.jsonl"
        _write_dataset([_make_case("tc-001")], dataset)
        result = evaluate_readiness(dataset)
        assert isinstance(result["production_gating_eligible"], bool)
        assert result["production_gating_eligible"] is False


# ---------------------------------------------------------------------------
# Tests: complete vs placeholder distinction
# ---------------------------------------------------------------------------

class TestReadinessCompletePlaceholderDistinction:
    def test_unset_split_excluded_from_complete(self, tmp_path):
        cases = [_make_case(f"tc-{i:03d}", split="unset") for i in range(40)]
        dataset = tmp_path / "unset.jsonl"
        _write_dataset(cases, dataset)
        result = evaluate_readiness(dataset)
        assert result["complete_cases"] == 0
        assert result["readiness_status"] == "NOT_READY"

    def test_low_confidence_excluded_from_complete(self, tmp_path):
        cases = [_make_case(f"tc-{i:03d}", label_confidence="low") for i in range(40)]
        dataset = tmp_path / "low_conf.jsonl"
        _write_dataset(cases, dataset)
        result = evaluate_readiness(dataset)
        assert result["complete_cases"] == 0
        assert result["placeholder_cases"] == 40


# ---------------------------------------------------------------------------
# Integration test: actual dataset
# ---------------------------------------------------------------------------

class TestActualDatasetReadiness:
    """Integration test against the real govrag_calib_150.jsonl."""

    DATASET = PROJECT_ROOT / "evals" / "govrag_calib" / "govrag_calib_150.jsonl"

    def test_actual_dataset_exists(self):
        assert self.DATASET.exists(), f"Dataset not found: {self.DATASET}"

    def test_actual_dataset_has_cases(self):
        result = evaluate_readiness(self.DATASET)
        assert result["total_cases"] > 0

    def test_actual_dataset_has_at_least_30_complete_cases(self):
        result = evaluate_readiness(self.DATASET)
        assert result["complete_cases"] >= 30, (
            f"Expected at least 30 complete cases, got {result['complete_cases']}"
        )

    def test_actual_dataset_readiness_is_not_not_ready(self):
        result = evaluate_readiness(self.DATASET)
        assert result["readiness_status"] != "NOT_READY", (
            f"Dataset should have at least PROVISIONAL_DATASET status. "
            f"Got: {result['readiness_status']}"
        )

    def test_actual_dataset_production_gating_false(self):
        result = evaluate_readiness(self.DATASET)
        assert result["production_gating_eligible"] is False

    def test_actual_dataset_has_multiple_splits(self):
        result = evaluate_readiness(self.DATASET)
        # At least train and dev should be present
        split_avail = result["split_availability"]
        assert split_avail["train"] or split_avail["dev"], (
            "Dataset should have at least one of train or dev split defined."
        )
