"""Tests for Diagnosis.human_review_required() escalation policy.

Step 2: STALE_RETRIEVAL (version-validity family) must auto-escalate to
human review. Version-validity failures represent freshness/version
mismatches where the system used outdated context — a real engineer
needs to confirm before downstream action.
"""

from __future__ import annotations

from datetime import UTC, datetime

from raggov.models.diagnosis import (
    Diagnosis,
    FailureStage,
    FailureType,
    SecurityRisk,
)


def _diagnosis_with(primary: FailureType, stage: FailureStage) -> Diagnosis:
    return Diagnosis(
        run_id="run-test",
        primary_failure=primary,
        root_cause_stage=stage,
        should_have_answered=True,
        security_risk=SecurityRisk.NONE,
        diagnostic_score=0.5,
        pipeline_health_score=0.5,
        first_failing_node=None,
        citation_faithfulness="genuine",
        recommended_fix="test",
        checks_run=[],
        created_at=datetime(2026, 6, 11, tzinfo=UTC),
    )


def test_stale_retrieval_escalates_to_human_review() -> None:
    """Step 2: version-validity failures must auto-escalate.

    Closes Calib-50 case 007 gap where STALE_RETRIEVAL fired correctly but
    human_review_required stayed False, masking a real freshness defect.
    """
    diagnosis = _diagnosis_with(FailureType.STALE_RETRIEVAL, FailureStage.RETRIEVAL)
    assert diagnosis.human_review_required() is True


def test_clean_does_not_escalate() -> None:
    diagnosis = _diagnosis_with(FailureType.CLEAN, FailureStage.UNKNOWN)
    assert diagnosis.human_review_required() is False


def test_claim_extraction_failed_escalates() -> None:
    """Regression: Step 1's new failure type must remain in escalation set."""
    diagnosis = _diagnosis_with(
        FailureType.CLAIM_EXTRACTION_FAILED, FailureStage.GROUNDING
    )
    assert diagnosis.human_review_required() is True
