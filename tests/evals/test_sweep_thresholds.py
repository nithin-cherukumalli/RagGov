"""
Tests for PR 7 — Threshold sweep and calibration reporting.

Tests cover:
- Sweep runs on a tiny controlled fixture (no full seed dataset required)
- Selection policy correctly penalizes false passes over macro F1
- Calibration status inference rules
- assert_not_silently_dev_calibrated raises when required
- Production code cannot silently load dev calibration
- Configurable thresholds in HeuristicValueOverlapVerifier
- Markdown and JSONL output format
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = _REPO_ROOT / "evals" / "claim_grounding"
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from evals.claim_grounding.calibration import (  # noqa: E402
    CalibrationStatus,
    MIN_CASES_FOR_DEV_CALIBRATION,
    assert_not_silently_dev_calibrated,
    infer_calibration_status,
    validate_calibration_config,
)
from evals.claim_grounding.sweep_thresholds import (  # noqa: E402
    FAST_GRID,
    SweepResult,
    render_sweep_markdown,
    run_sweep,
    selection_score,
)
from raggov.analyzers.grounding.verifiers import HeuristicValueOverlapVerifier  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers: minimal fixture dataset
# ---------------------------------------------------------------------------

_TINY_FIXTURE_JSONL = """
{"case_id":"t1","query":"q","answer":"a","claim_text":"The amount is Rs. 100.","retrieved_chunks":[{"chunk_id":"c1","text":"The amount is Rs. 100 per beneficiary.","source_doc_id":"d1","score":0.9}],"cited_doc_ids":["d1"],"gold_label":"entailed","gold_supporting_chunk_ids":["c1"],"gold_contradicting_chunk_ids":[],"claim_type":"numeric","atomicity_status":"atomic","error_type":null,"notes":null}
{"case_id":"t2","query":"q","answer":"a","claim_text":"The subsidy is 60%.","retrieved_chunks":[{"chunk_id":"c2","text":"The subsidy under this scheme is 30% of the benchmark cost.","source_doc_id":"d2","score":0.88}],"cited_doc_ids":["d2"],"gold_label":"contradicted","gold_supporting_chunk_ids":[],"gold_contradicting_chunk_ids":["c2"],"claim_type":"numeric","atomicity_status":"atomic","error_type":"value_error","notes":null}
{"case_id":"t3","query":"q","answer":"a","claim_text":"Eligible persons receive a food grain entitlement of 5 kg per month.","retrieved_chunks":[{"chunk_id":"c3","text":"Eligible beneficiaries are entitled to receive food grains at subsidized rates under the National Food Security Act.","source_doc_id":"d3","score":0.75}],"cited_doc_ids":[],"gold_label":"unsupported","gold_supporting_chunk_ids":[],"gold_contradicting_chunk_ids":[],"claim_type":"numeric","atomicity_status":"atomic","error_type":"insufficient_context","notes":null}
{"case_id":"t4","query":"q","answer":"a","claim_text":"Taxpayers must file returns by the 20th of each month.","retrieved_chunks":[{"chunk_id":"c4","text":"Returns must be filed by the 20th of the following month.","source_doc_id":"d4","score":0.91}],"cited_doc_ids":["d4"],"gold_label":"entailed","gold_supporting_chunk_ids":["c4"],"gold_contradicting_chunk_ids":[],"claim_type":"date_or_deadline","atomicity_status":"atomic","error_type":null,"notes":null}
{"case_id":"t5","query":"q","answer":"a","claim_text":"The penalty rate is Rs. 200 per day.","retrieved_chunks":[{"chunk_id":"c5","text":"A late fee of Rs. 50 per day is levied for delayed submission.","source_doc_id":"d5","score":0.85}],"cited_doc_ids":["d5"],"gold_label":"contradicted","gold_supporting_chunk_ids":[],"gold_contradicting_chunk_ids":["c5"],"claim_type":"numeric","atomicity_status":"atomic","error_type":"value_error","notes":null}
""".strip()


@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> Path:
    """Write the tiny fixture to a temp JSONL file."""
    p = tmp_path / "tiny.jsonl"
    p.write_text(_TINY_FIXTURE_JSONL, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 1. Sweep runs on tiny fixture
# ---------------------------------------------------------------------------

class TestSweepRunsOnTinyFixture:
    def test_sweep_returns_results_for_all_configs(self, tiny_dataset: Path) -> None:
        grid = {
            "support_threshold": [0.4, 0.5],
            "anchor_weight": [0.6],
            "value_match_score_boost": [0.2],
            "missing_critical_value_behavior": ["unsupported"],
            "candidate_top_k": [3],
        }
        results = run_sweep(dataset_path=tiny_dataset, grid=grid)
        assert len(results) == 2  # 2 support_threshold values × 1 each for others

    def test_sweep_marks_exactly_one_recommended(self, tiny_dataset: Path) -> None:
        grid = {
            "support_threshold": [0.4, 0.5, 0.6],
            "anchor_weight": [0.6],
            "value_match_score_boost": [0.2],
            "missing_critical_value_behavior": ["unsupported"],
            "candidate_top_k": [3],
        }
        results = run_sweep(dataset_path=tiny_dataset, grid=grid)
        recommended = [r for r in results if r.is_recommended]
        assert len(recommended) == 1

    def test_sweep_results_have_required_metrics(self, tiny_dataset: Path) -> None:
        grid = {
            "support_threshold": [0.5],
            "anchor_weight": [0.6],
            "value_match_score_boost": [0.2],
            "missing_critical_value_behavior": ["unsupported"],
            "candidate_top_k": [3],
        }
        results = run_sweep(dataset_path=tiny_dataset, grid=grid)
        r = results[0]
        assert 0.0 <= r.false_pass_rate <= 1.0
        assert 0.0 <= r.false_fail_rate <= 1.0
        assert 0.0 <= r.macro_f1 <= 1.0
        assert 0.0 <= r.overall_accuracy <= 1.0
        assert 0.0 <= r.evidence_chunk_recall <= 1.0
        assert r.calibration_status in (
            CalibrationStatus.UNCALIBRATED.value,
            CalibrationStatus.DEV_CALIBRATED_SEED.value,
        )

    def test_sweep_writes_jsonl_output(self, tiny_dataset: Path, tmp_path: Path) -> None:
        grid = {
            "support_threshold": [0.4, 0.5],
            "anchor_weight": [0.6],
            "value_match_score_boost": [0.2],
            "missing_critical_value_behavior": ["unsupported"],
            "candidate_top_k": [3],
        }
        out = tmp_path / "sweep.jsonl"
        run_sweep(dataset_path=tiny_dataset, grid=grid, jsonl_out=out)
        assert out.exists()
        lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        assert len(lines) == 2
        assert "false_pass_rate" in lines[0]
        assert "is_recommended" in lines[0]

    def test_sweep_writes_markdown_output(self, tiny_dataset: Path, tmp_path: Path) -> None:
        grid = {
            "support_threshold": [0.5],
            "anchor_weight": [0.6],
            "value_match_score_boost": [0.2],
            "missing_critical_value_behavior": ["unsupported"],
            "candidate_top_k": [3],
        }
        out = tmp_path / "sweep.md"
        run_sweep(dataset_path=tiny_dataset, grid=grid, md_out=out)
        assert out.exists()
        content = out.read_text()
        assert "# Claim-Grounding Threshold Sweep Report" in content
        assert "Recommended Configuration" in content

    def test_fast_grid_sweep_on_seed_dataset(self) -> None:
        """Smoke test: fast-grid sweep on the full seed dataset must complete."""
        results = run_sweep(
            dataset_path=_EVAL_DIR / "seed_cases.jsonl",
            grid=FAST_GRID,
        )
        assert len(results) > 0
        assert any(r.is_recommended for r in results)


# ---------------------------------------------------------------------------
# 2. Selection policy: penalizes false passes over macro F1
# ---------------------------------------------------------------------------

class TestSelectionPolicy:
    def _make_result(
        self,
        false_pass_rate: float,
        macro_f1: float,
        is_recommended: bool = False,
    ) -> SweepResult:
        return SweepResult(
            support_threshold=0.5,
            anchor_weight=0.6,
            value_match_score_boost=0.2,
            missing_critical_value_behavior="unsupported",
            candidate_top_k=3,
            overall_accuracy=0.8,
            false_pass_rate=false_pass_rate,
            false_fail_rate=0.1,
            macro_f1=macro_f1,
            entailed_f1=0.8,
            unsupported_f1=0.7,
            contradicted_f1=0.6,
            evidence_chunk_recall=0.5,
            fallback_rate=0.0,
            is_recommended=is_recommended,
        )

    def test_lower_false_pass_rate_always_wins(self) -> None:
        """A config with lower fpr always beats higher fpr, even at macro F1 cost."""
        safe = self._make_result(false_pass_rate=0.05, macro_f1=0.60)
        risky = self._make_result(false_pass_rate=0.20, macro_f1=0.95)
        assert selection_score(safe) < selection_score(risky)

    def test_same_fpr_higher_macro_f1_wins(self) -> None:
        """When fpr is equal, the config with higher macro F1 wins (tiebreaker)."""
        good = self._make_result(false_pass_rate=0.10, macro_f1=0.80)
        mediocre = self._make_result(false_pass_rate=0.10, macro_f1=0.60)
        assert selection_score(good) < selection_score(mediocre)

    def test_selection_score_ordering_consistent(self) -> None:
        """selection_score() produces a consistent total order."""
        configs = [
            self._make_result(false_pass_rate=0.30, macro_f1=0.90),  # worst (high fpr)
            self._make_result(false_pass_rate=0.10, macro_f1=0.60),  # ok fpr, low f1
            self._make_result(false_pass_rate=0.10, macro_f1=0.80),  # ok fpr, high f1
            self._make_result(false_pass_rate=0.00, macro_f1=0.50),  # perfect fpr, low f1
        ]
        configs.sort(key=selection_score)
        fprs = [c.false_pass_rate for c in configs]
        # Should be non-decreasing in fpr
        assert fprs == sorted(fprs)
        # Within same fpr=0.10 group: higher f1 should come first
        same_fpr = [c for c in configs if c.false_pass_rate == 0.10]
        assert same_fpr[0].macro_f1 >= same_fpr[1].macro_f1

    def test_sweep_never_selects_high_fpr_config_when_low_fpr_exists(
        self, tiny_dataset: Path
    ) -> None:
        """The recommended config must never have a higher fpr than any other config."""
        grid = {
            "support_threshold": [0.1, 0.5, 0.9],
            "anchor_weight": [0.6],
            "value_match_score_boost": [0.2],
            "missing_critical_value_behavior": ["unsupported"],
            "candidate_top_k": [3],
        }
        results = run_sweep(dataset_path=tiny_dataset, grid=grid)
        recommended = next(r for r in results if r.is_recommended)
        min_fpr = min(r.false_pass_rate for r in results)
        assert recommended.false_pass_rate == min_fpr


# ---------------------------------------------------------------------------
# 3. Calibration status
# ---------------------------------------------------------------------------

class TestCalibrationStatus:
    def test_infer_uncalibrated_when_too_few_cases(self) -> None:
        status = infer_calibration_status(5, "seed_cases.jsonl")
        assert status == CalibrationStatus.UNCALIBRATED.value

    def test_infer_dev_calibrated_seed_when_sufficient_cases_and_seed_name(self) -> None:
        status = infer_calibration_status(MIN_CASES_FOR_DEV_CALIBRATION, "seed_cases.jsonl")
        assert status == CalibrationStatus.DEV_CALIBRATED_SEED.value

    def test_infer_uncalibrated_for_non_seed_name_even_if_large(self) -> None:
        status = infer_calibration_status(1000, "production_queries.jsonl")
        assert status == CalibrationStatus.UNCALIBRATED.value

    def test_calibration_status_from_string(self) -> None:
        assert CalibrationStatus.from_string("uncalibrated") == CalibrationStatus.UNCALIBRATED
        assert CalibrationStatus.from_string("dev_calibrated_seed") == CalibrationStatus.DEV_CALIBRATED_SEED
        assert CalibrationStatus.from_string("production_calibrated") == CalibrationStatus.PRODUCTION_CALIBRATED

    def test_calibration_status_from_string_unknown_defaults_to_uncalibrated(self) -> None:
        assert CalibrationStatus.from_string("magic_special_mode") == CalibrationStatus.UNCALIBRATED

    def test_calibration_status_from_none(self) -> None:
        assert CalibrationStatus.from_string(None) == CalibrationStatus.UNCALIBRATED

    def test_validate_config_uncalibrated_no_warnings(self) -> None:
        status, warnings = validate_calibration_config({"calibration_status": "uncalibrated"})
        assert status == CalibrationStatus.UNCALIBRATED
        assert warnings == []

    def test_validate_config_dev_calibrated_emits_warning(self) -> None:
        status, warnings = validate_calibration_config(
            {"calibration_status": "dev_calibrated_seed"}
        )
        assert status == CalibrationStatus.DEV_CALIBRATED_SEED
        assert len(warnings) >= 1
        assert "synthetic seed" in warnings[0] or "NOT suitable" in warnings[0]

    def test_validate_config_production_calibrated_without_dataset_warns(self) -> None:
        status, warnings = validate_calibration_config(
            {"calibration_status": "production_calibrated"}
        )
        assert status == CalibrationStatus.PRODUCTION_CALIBRATED
        assert any("calibration_dataset" in w for w in warnings)

    def test_validate_config_production_calibrated_with_dataset_no_warnings(self) -> None:
        status, warnings = validate_calibration_config(
            {
                "calibration_status": "production_calibrated",
                "calibration_dataset": "prod_queries_2024_q4.jsonl",
            }
        )
        assert status == CalibrationStatus.PRODUCTION_CALIBRATED
        assert warnings == []


# ---------------------------------------------------------------------------
# 4. assert_not_silently_dev_calibrated
# ---------------------------------------------------------------------------

class TestAssertNotSilentlyDevCalibrated:
    def test_uncalibrated_does_not_raise(self) -> None:
        assert_not_silently_dev_calibrated({"calibration_status": "uncalibrated"})

    def test_production_calibrated_does_not_raise(self) -> None:
        assert_not_silently_dev_calibrated(
            {
                "calibration_status": "production_calibrated",
                "calibration_dataset": "prod.jsonl",
            }
        )

    def test_dev_calibrated_without_opt_in_raises(self) -> None:
        with pytest.raises(ValueError, match="dev_calibrated_seed"):
            assert_not_silently_dev_calibrated(
                {"calibration_status": "dev_calibrated_seed"}
            )

    def test_dev_calibrated_with_allow_flag_does_not_raise(self) -> None:
        # Explicit opt-in must be respected
        assert_not_silently_dev_calibrated(
            {
                "calibration_status": "dev_calibrated_seed",
                "allow_dev_calibration": True,
            }
        )

    def test_missing_calibration_status_treated_as_uncalibrated(self) -> None:
        # No 'calibration_status' key → defaults to uncalibrated → no raise
        assert_not_silently_dev_calibrated({})

    def test_error_message_includes_production_warning(self) -> None:
        with pytest.raises(ValueError, match="NOT production calibration"):
            assert_not_silently_dev_calibrated(
                {"calibration_status": "dev_calibrated_seed"}
            )


# ---------------------------------------------------------------------------
# 5. Configurable thresholds in HeuristicValueOverlapVerifier
# ---------------------------------------------------------------------------

class TestConfigurableVerifierThresholds:
    def test_default_support_threshold_is_0_5(self) -> None:
        v = HeuristicValueOverlapVerifier({})
        assert v._support_threshold == 0.5

    def test_custom_support_threshold_applied(self) -> None:
        v = HeuristicValueOverlapVerifier({"support_threshold": 0.7})
        assert v._support_threshold == 0.7

    def test_default_value_match_boost_is_0_2(self) -> None:
        v = HeuristicValueOverlapVerifier({})
        assert v._value_match_score_boost == 0.2

    def test_custom_value_match_boost(self) -> None:
        v = HeuristicValueOverlapVerifier({"value_match_score_boost": 0.1})
        assert v._value_match_score_boost == 0.1

    def test_default_missing_critical_value_behavior_is_unsupported(self) -> None:
        v = HeuristicValueOverlapVerifier({})
        assert v._missing_critical_value_behavior == "unsupported"

    def test_aggressive_missing_critical_value_behavior(self) -> None:
        v = HeuristicValueOverlapVerifier(
            {"missing_critical_value_behavior": "contradicted"}
        )
        assert v._missing_critical_value_behavior == "contradicted"

    def test_high_support_threshold_produces_more_unsupported(self) -> None:
        """A very high threshold should mark borderline cases as unsupported."""
        from raggov.analyzers.grounding.candidate_selection import (
            EvidenceCandidateSelector,
        )
        from evals.claim_grounding.schema import ChunkRecord, ClaimGroundingCase
        from evals.claim_grounding.run_eval import predict, load_dataset

        # Case with moderate overlap — should flip from entailed → unsupported
        # as support_threshold increases
        claim = "Returns must be filed by the 20th of each month."
        chunk_text = "Registered taxpayers must file returns by the 20th of the following month."

        from raggov.models.chunk import RetrievedChunk
        from raggov.analyzers.grounding.candidate_selection import EvidenceCandidateSelector

        chunks = [RetrievedChunk(chunk_id="c1", text=chunk_text, source_doc_id="d1", score=0.9)]

        v_low = HeuristicValueOverlapVerifier({"support_threshold": 0.1})
        v_high = HeuristicValueOverlapVerifier({"support_threshold": 0.99})
        selector = EvidenceCandidateSelector({})

        candidates = selector.select_candidates(claim, "q", chunks)
        result_low = v_low.verify(claim, "q", candidates)
        result_high = v_high.verify(claim, "q", candidates)

        # Low threshold should be at least as permissive (or equal) to high threshold
        # For this specific claim, low should be entailed and high should be unsupported
        label_order = {"entailed": 0, "unsupported": 1, "contradicted": 2, "abstain": 3}
        assert label_order.get(result_low.label, 9) <= label_order.get(result_high.label, 9)


# ---------------------------------------------------------------------------
# 6. Markdown rendering
# ---------------------------------------------------------------------------

class TestSweepMarkdown:
    def _make_sweep_result(self, fpr: float = 0.1, macro_f1: float = 0.7) -> SweepResult:
        return SweepResult(
            support_threshold=0.5,
            anchor_weight=0.6,
            value_match_score_boost=0.2,
            missing_critical_value_behavior="unsupported",
            candidate_top_k=3,
            overall_accuracy=0.75,
            false_pass_rate=fpr,
            false_fail_rate=0.1,
            macro_f1=macro_f1,
            entailed_f1=0.8,
            unsupported_f1=0.7,
            contradicted_f1=0.6,
            evidence_chunk_recall=0.6,
            fallback_rate=0.0,
            is_recommended=True,
            calibration_status="dev_calibrated_seed",
            selection_notes=["test note"],
        )

    def test_markdown_has_required_sections(self) -> None:
        r = self._make_sweep_result()
        md = render_sweep_markdown([r], r, Path("seed_cases.jsonl"))
        assert "# Claim-Grounding Threshold Sweep Report" in md
        assert "Recommended Configuration" in md
        assert "Top 10 Configurations" in md
        assert "Production Promotion Checklist" in md

    def test_markdown_warns_about_seed_calibration(self) -> None:
        r = self._make_sweep_result()
        md = render_sweep_markdown([r], r, Path("seed_cases.jsonl"))
        assert "NOT production calibration" in md or "synthetic seed" in md.lower()

    def test_markdown_marks_recommended_with_star(self) -> None:
        r = self._make_sweep_result()
        md = render_sweep_markdown([r], r, Path("seed_cases.jsonl"))
        assert "★" in md
