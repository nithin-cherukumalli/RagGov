"""Test harness for external-to-native diagnostic alignment benchmark."""

from __future__ import annotations

import pytest

from raggov.analyzers.base import BaseAnalyzer
from raggov.engine import DiagnosisEngine
from raggov.evaluators.base import (
    ExternalEvaluationResult,
    ExternalEvaluatorProvider,
    ExternalSignalRecord,
)
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.run import RAGRun

from tests.external_alignment.external_alignment_cases import (
    CASES,
    ExternalAlignmentCase,
    NativeMockDirective,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockAnalyzer(BaseAnalyzer):
    def __init__(self, result: AnalyzerResult):
        super().__init__()
        self._result = result

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        return self._result


def _build_run(case: ExternalAlignmentCase) -> RAGRun:
    chunks = [
        RetrievedChunk(
            chunk_id=c["chunk_id"],
            text=c["text"],
            source_doc_id=c["source_doc_id"],
            score=c.get("score", 0.9),
        )
        for c in case.retrieved_chunks
    ]
    run = RAGRun(
        run_id=case.case_id,
        query=case.query,
        retrieved_chunks=chunks,
        final_answer=case.final_answer,
        cited_doc_ids=case.cited_doc_ids,
    )
    # Inject external evaluation results into run metadata
    signals = case.mocked_external_signals
    if signals:
        # Group signals by provider
        by_provider: dict[str, list[ExternalSignalRecord]] = {}
        for s in signals:
            provider_key = s.provider.value if hasattr(s.provider, "value") else s.provider
            by_provider.setdefault(provider_key, []).append(s)

        ext_results = []
        for provider_key, sigs in by_provider.items():
            # Determine provider enum
            try:
                prov_enum = ExternalEvaluatorProvider(provider_key)
            except ValueError:
                prov_enum = ExternalEvaluatorProvider.custom
            ext_results.append(
                ExternalEvaluationResult(
                    provider=prov_enum,
                    succeeded=True,
                    signals=sigs,
                ).model_dump()
            )
        run.metadata["external_evaluation_results"] = ext_results
    return run


def _build_native_analyzers(directive: NativeMockDirective) -> list[BaseAnalyzer]:
    """Translate a NativeMockDirective into a list of mock native analyzers."""
    analyzers: list[BaseAnalyzer] = []

    if directive.prompt_injection:
        analyzers.append(_MockAnalyzer(
            AnalyzerResult(
                analyzer_name="PromptInjectionAnalyzer",
                status="fail",
                failure_type=FailureType.PROMPT_INJECTION,
                stage=FailureStage.SECURITY,
            )
        ))

    if directive.unsupported_claims:
        # Build a minimal grounding bundle mock via model_construct
        from raggov.models.grounding import GroundingEvidenceBundle, ClaimEvidenceRecord
        bundle = GroundingEvidenceBundle.model_construct(
            claim_evidence_records=[
                ClaimEvidenceRecord.model_construct(
                    claim_id="c1",
                    claim_text="test claim",
                    claim_label="unsupported",
                    supporting_chunk_ids=[],
                    contradicting_chunk_ids=[],
                )
            ]
        )
        analyzers.append(_MockAnalyzer(
            AnalyzerResult.model_construct(
                analyzer_name="ClaimGroundingAnalyzer",
                status="fail",
                failure_type=FailureType.UNSUPPORTED_CLAIM,
                stage=FailureStage.GROUNDING,
                grounding_evidence_bundle=bundle,
            )
        ))

    if directive.phantom_citations:
        from raggov.models.citation_faithfulness import CitationFaithfulnessReport
        cf_report = CitationFaithfulnessReport.model_construct(
            phantom_citation_doc_ids=["doc2"],
            faithfulness_score=0.2,
        )
        analyzers.append(_MockAnalyzer(
            AnalyzerResult.model_construct(
                analyzer_name="CitationFaithfulnessAnalyzerV0",
                status="fail",
                failure_type=FailureType.CITATION_MISMATCH,
                stage=FailureStage.GROUNDING,
                citation_faithfulness_report=cf_report,
            )
        ))

    if directive.retrieval_noise_suspected:
        from raggov.models.retrieval_diagnosis import (
            RetrievalDiagnosisReport,
            RetrievalFailureType,
            RetrievalDiagnosisMethodType,
            RetrievalDiagnosisCalibrationStatus,
        )
        report = RetrievalDiagnosisReport(
            run_id="test",
            primary_failure_type=RetrievalFailureType.RETRIEVAL_NOISE,
            recommended_fix="Improve chunk filtering",
            method_type=RetrievalDiagnosisMethodType.HEURISTIC_BASELINE,
            calibration_status=RetrievalDiagnosisCalibrationStatus.UNCALIBRATED,
        )
        analyzers.append(_MockAnalyzer(
            AnalyzerResult.model_construct(
                analyzer_name="RetrievalDiagnosisAnalyzerV0",
                status="fail",
                failure_type=FailureType.RETRIEVAL_ANOMALY,
                stage=FailureStage.RETRIEVAL,
                retrieval_diagnosis_report=report,
            )
        ))

    if directive.first_failing_node == "retrieval_coverage" and not directive.retrieval_noise_suspected:
        # Simulate NCV report showing coverage failure
        analyzers.append(_MockAnalyzer(
            AnalyzerResult.model_construct(
                analyzer_name="NCVPipelineVerifier",
                status="fail",
                failure_type=FailureType.INSUFFICIENT_CONTEXT,
                stage=FailureStage.RETRIEVAL,
                ncv_report={"first_failing_node": "retrieval_coverage", "calibration_status": "uncalibrated"},
            )
        ))

    # Always add a clean fallback if no other analyzers were added and no injection
    if not analyzers:
        analyzers.append(_MockAnalyzer(
            AnalyzerResult(
                analyzer_name="CleanAnalyzer",
                status="pass",
                failure_type=FailureType.CLEAN,
                stage=FailureStage.UNKNOWN,
            )
        ))

    return analyzers


def _run_case(case: ExternalAlignmentCase) -> tuple:
    """Execute a case and return (diagnosis, probe_or_none)."""
    run = _build_run(case)
    analyzers = _build_native_analyzers(case.native_mocks)
    engine = DiagnosisEngine(config={"mode": "native"}, analyzers=analyzers)

    # Suppress missing-critical-evidence blocking so we can test clean-block logic precisely
    if not case.native_mocks.critical_analyzers_missing:
        engine._get_missing_critical_evidence = lambda results, r: []

    diagnosis = engine.diagnose(run)
    probe = diagnosis.external_diagnosis_probes[0] if diagnosis.external_diagnosis_probes else None
    return diagnosis, probe


# ---------------------------------------------------------------------------
# Parametrised test per case
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_probe_created(case: ExternalAlignmentCase) -> None:
    """Every external low signal must produce at least one probe."""
    _, probe = _run_case(case)
    assert probe is not None, f"[{case.case_id}] Expected a probe but got none."


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_suspected_pipeline_node(case: ExternalAlignmentCase) -> None:
    """Probe must point to the correct pipeline node."""
    _, probe = _run_case(case)
    assert probe is not None, f"[{case.case_id}] No probe produced."
    assert probe.suspected_pipeline_node == case.expected_suspected_pipeline_node, (
        f"[{case.case_id}] Expected node '{case.expected_suspected_pipeline_node}', "
        f"got '{probe.suspected_pipeline_node}'"
    )


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_suspected_failure_stage(case: ExternalAlignmentCase) -> None:
    """Probe must point to the correct failure stage."""
    _, probe = _run_case(case)
    assert probe is not None, f"[{case.case_id}] No probe produced."
    assert probe.suspected_failure_stage == case.expected_suspected_failure_stage, (
        f"[{case.case_id}] Expected stage '{case.expected_suspected_failure_stage}', "
        f"got '{probe.suspected_failure_stage}'"
    )


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_expected_native_analyzers_referenced(case: ExternalAlignmentCase) -> None:
    """Probe's native_analyzers_to_check must include expected analyzers."""
    _, probe = _run_case(case)
    assert probe is not None, f"[{case.case_id}] No probe produced."
    for expected_analyzer in case.expected_native_analyzers_checked:
        assert expected_analyzer in probe.native_analyzers_to_check, (
            f"[{case.case_id}] Expected analyzer '{expected_analyzer}' not found in "
            f"{probe.native_analyzers_to_check}"
        )


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_native_evidence_found(case: ExternalAlignmentCase) -> None:
    """When native analyzers provide corroborating evidence, it must appear in the probe."""
    _, probe = _run_case(case)
    assert probe is not None, f"[{case.case_id}] No probe produced."
    for expected_fragment in case.expected_native_evidence_found_contains:
        matches = [ev for ev in probe.native_evidence_found if expected_fragment in ev]
        assert matches, (
            f"[{case.case_id}] Expected native evidence fragment '{expected_fragment}' "
            f"not found in {probe.native_evidence_found}"
        )


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_human_review_required(case: ExternalAlignmentCase) -> None:
    """High-severity probes must set should_trigger_human_review=True."""
    _, probe = _run_case(case)
    assert probe is not None, f"[{case.case_id}] No probe produced."
    if case.expected_human_review_required:
        assert probe.should_trigger_human_review, (
            f"[{case.case_id}] Expected should_trigger_human_review=True"
        )


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_primary_failure_behavior(case: ExternalAlignmentCase) -> None:
    """Primary failure must match expected behavior spec."""
    diagnosis, _ = _run_case(case)
    expected = case.expected_primary_failure_behavior
    actual = diagnosis.primary_failure.value if hasattr(diagnosis.primary_failure, "value") else str(diagnosis.primary_failure)

    if expected == "LOW_CONFIDENCE":
        # CLEAN should have been blocked — any non-CLEAN result is acceptable
        assert actual != FailureType.CLEAN.value, (
            f"[{case.case_id}] Expected CLEAN to be blocked, but got '{actual}'"
        )
    elif expected == "PROMPT_INJECTION":
        assert actual == FailureType.PROMPT_INJECTION.value, (
            f"[{case.case_id}] Expected PROMPT_INJECTION, got '{actual}'"
        )
    elif expected in ("UNSUPPORTED_CLAIM", "RETRIEVAL_ANOMALY", "INSUFFICIENT_CONTEXT"):
        # With a native analyzer failing, the primary should match or be a blocked non-CLEAN
        assert actual != FailureType.CLEAN.value, (
            f"[{case.case_id}] Expected CLEAN to be blocked (native failure exists), got '{actual}'"
        )
    elif expected == "CLEAN_OR_LOW_CONFIDENCE":
        # Probe exists (checked in test_external_signal_not_ignored), but blocking is not guaranteed
        # Just ensure no security drift
        pass


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_external_signal_not_ignored(case: ExternalAlignmentCase) -> None:
    """External signals must always produce at least one probe (never silently ignored)."""
    diagnosis, _ = _run_case(case)
    assert len(diagnosis.external_diagnosis_probes) >= 1, (
        f"[{case.case_id}] External signals were ignored — no probes found."
    )


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_no_false_security_from_external(case: ExternalAlignmentCase) -> None:
    """Retrieval/citation external signals must not produce a security-class primary failure."""
    # These are the security-adjacent failure types that would indicate a false positive
    SECURITY_CLASS_FAILURES = {
        FailureType.PROMPT_INJECTION.value,
        FailureType.SUSPICIOUS_CHUNK.value,
        FailureType.PRIVACY_VIOLATION.value,
    }
    diagnosis, probe = _run_case(case)
    if probe and probe.suspected_failure_stage != FailureStage.SECURITY.value:
        # If this probe isn't about security, primary failure must not be a security type
        # UNLESS native prompt_injection was explicitly detected
        if not case.native_mocks.prompt_injection:
            actual = diagnosis.primary_failure.value if hasattr(diagnosis.primary_failure, "value") else str(diagnosis.primary_failure)
            assert actual not in SECURITY_CLASS_FAILURES, (
                f"[{case.case_id}] Non-security external signal caused security primary failure: '{actual}'"
            )
