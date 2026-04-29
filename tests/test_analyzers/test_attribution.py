"""Tests for A2P attribution analyzer."""

from __future__ import annotations

from raggov.analyzers.attribution.a2p import A2PAttributionAnalyzer
from raggov.engine import DiagnosisEngine
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import (
    AnalyzerResult,
    ClaimResult,
    FailureStage,
    FailureType,
    SufficiencyResult,
)
from raggov.models.run import RAGRun


def chunk(chunk_id: str, text: str, score: float | None = None) -> RetrievedChunk:
    """Helper to create a retrieved chunk."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=f"doc-{chunk_id}",
        score=score,
    )


def run_with_chunks(
    chunks: list[RetrievedChunk], query: str = "What is the answer?", answer: str = "Answer."
) -> RAGRun:
    """Helper to create a RAG run with chunks."""
    return RAGRun(query=query, retrieved_chunks=chunks, final_answer=answer)


def test_deterministic_insufficient_context_with_low_scores() -> None:
    """INSUFFICIENT_CONTEXT in prior results + low scores → RETRIEVAL_DEPTH_LIMIT."""
    prior_results = [
        AnalyzerResult(
            analyzer_name="SufficiencyAnalyzer",
            status="fail",
            failure_type=FailureType.INSUFFICIENT_CONTEXT,
            stage=FailureStage.SUFFICIENCY,
            evidence=["Context is insufficient"],
            remediation="Expand retrieval",
        )
    ]

    analyzer = A2PAttributionAnalyzer(
        {"use_llm": False, "prior_results": prior_results}
    )
    result = analyzer.analyze(
        run_with_chunks(
            [
                chunk("chunk-1", "Some text", 0.5),
                chunk("chunk-2", "More text", 0.4),
            ]
        )
    )

    assert result.status == "fail"
    assert result.failure_type == FailureType.RETRIEVAL_DEPTH_LIMIT
    assert result.stage == FailureStage.RETRIEVAL
    assert result.attribution_stage == FailureStage.RETRIEVAL
    assert result.proposed_fix is not None
    assert "top-k" in result.proposed_fix.lower() or "retrieval" in result.proposed_fix.lower()
    assert result.fix_confidence is not None
    assert result.fix_confidence == 0.78
    assert any("confidence basis:" in evidence.lower() for evidence in result.evidence)


def test_deterministic_inconsistent_chunks() -> None:
    """INCONSISTENT_CHUNKS in prior results → CHUNKING_BOUNDARY_ERROR."""
    prior_results = [
        AnalyzerResult(
            analyzer_name="InconsistentChunksAnalyzer",
            status="fail",
            failure_type=FailureType.INCONSISTENT_CHUNKS,
            stage=FailureStage.RETRIEVAL,
            evidence=["Chunks contain inconsistencies"],
            remediation="Review chunks",
        )
    ]

    analyzer = A2PAttributionAnalyzer(
        {"use_llm": False, "prior_results": prior_results}
    )
    result = analyzer.analyze(
        run_with_chunks(
            [
                chunk("chunk-1", "Policy says X", 0.8),
                chunk("chunk-2", "Policy says Y", 0.7),
            ]
        )
    )

    assert result.status == "fail"
    assert result.failure_type == FailureType.CHUNKING_BOUNDARY_ERROR
    assert result.stage == FailureStage.CHUNKING
    assert result.attribution_stage == FailureStage.CHUNKING
    assert result.proposed_fix is not None
    assert "chunk" in result.proposed_fix.lower()
    assert result.fix_confidence is not None


def test_deterministic_parser_structure_loss_preserves_parsing_stage() -> None:
    """Parser failures should not be re-attributed to retrieval."""
    remediation = (
        "Use a structure-preserving parser (unstructured.io, docling, pymupdf4llm) "
        "before chunking. Tables must preserve row-column bindings."
    )
    prior_results = [
        AnalyzerResult(
            analyzer_name="ParserValidationAnalyzer",
            status="fail",
            failure_type=FailureType.TABLE_STRUCTURE_LOSS,
            stage=FailureStage.PARSING,
            evidence=["Table keywords detected but structural separators absent in chunk chunk-1"],
            remediation=remediation,
        )
    ]

    analyzer = A2PAttributionAnalyzer(
        {"use_llm": False, "prior_results": prior_results}
    )
    result = analyzer.analyze(
        run_with_chunks([chunk("chunk-1", "District Vacancies Category Warangal 5 Grade A", 0.8)])
    )

    assert result.status == "fail"
    assert result.failure_type == FailureType.TABLE_STRUCTURE_LOSS
    assert result.stage == FailureStage.PARSING
    assert result.attribution_stage == FailureStage.PARSING
    assert result.proposed_fix == remediation


def test_deterministic_unsupported_claim_with_high_scores() -> None:
    """UNSUPPORTED_CLAIM + high chunk scores → GENERATION_IGNORE."""
    prior_results = [
        AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
            evidence=["Claim not supported"],
            remediation="Verify sources",
        )
    ]

    analyzer = A2PAttributionAnalyzer(
        {"use_llm": False, "prior_results": prior_results}
    )
    result = analyzer.analyze(
        run_with_chunks(
            [
                chunk("chunk-1", "Contains relevant info", 0.9),
                chunk("chunk-2", "More relevant info", 0.85),
            ]
        )
    )

    assert result.status == "fail"
    assert result.failure_type == FailureType.GENERATION_IGNORE
    assert result.stage == FailureStage.GENERATION
    assert result.attribution_stage == FailureStage.GENERATION
    assert result.proposed_fix is not None
    # Check that the fix mentions grounding or model-related improvements
    assert "grounding" in result.proposed_fix.lower() or "model" in result.proposed_fix.lower() or "instruct" in result.proposed_fix.lower()
    assert result.fix_confidence is not None


def test_deterministic_retrieval_anomaly() -> None:
    """RETRIEVAL_ANOMALY in prior results → EMBEDDING_DRIFT."""
    prior_results = [
        AnalyzerResult(
            analyzer_name="RetrievalAnomalyAnalyzer",
            status="fail",
            failure_type=FailureType.RETRIEVAL_ANOMALY,
            stage=FailureStage.SECURITY,
            evidence=["Score anomaly detected"],
            remediation="Investigate",
        )
    ]

    analyzer = A2PAttributionAnalyzer(
        {"use_llm": False, "prior_results": prior_results}
    )
    result = analyzer.analyze(
        run_with_chunks(
            [
                chunk("chunk-1", "Text", 0.99),
                chunk("chunk-2", "Similar text", 0.98),
            ]
        )
    )

    assert result.status == "fail"
    assert result.failure_type == FailureType.EMBEDDING_DRIFT
    assert result.stage == FailureStage.EMBEDDING
    assert result.attribution_stage == FailureStage.EMBEDDING
    assert result.proposed_fix is not None
    assert "embedding" in result.proposed_fix.lower()
    assert result.fix_confidence is not None
    assert result.fix_confidence == 0.82
    assert any("confidence basis:" in evidence.lower() for evidence in result.evidence)


def test_weighted_prior_results_prefer_stronger_generation_signal() -> None:
    """Weighted evidence should beat heuristic check order in deterministic mode."""
    weighted_prior_results = [
        AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
            evidence=["Claim not supported"],
            remediation="Verify sources",
        ),
        AnalyzerResult(
            analyzer_name="InconsistentChunksAnalyzer",
            status="fail",
            failure_type=FailureType.INCONSISTENT_CHUNKS,
            stage=FailureStage.RETRIEVAL,
            evidence=["Chunks contain inconsistencies"],
            remediation="Review chunks",
        ),
    ]

    analyzer = A2PAttributionAnalyzer(
        {
            "use_llm": False,
            "prior_results": list(reversed(weighted_prior_results)),
            "weighted_prior_results": weighted_prior_results,
            "analyzer_weights": {
                "ClaimGroundingAnalyzer": 0.9,
                "InconsistentChunksAnalyzer": 0.5,
            },
        }
    )
    result = analyzer.analyze(
        run_with_chunks(
            [
                chunk("chunk-1", "Contains relevant info", 0.9),
                chunk("chunk-2", "More relevant info", 0.85),
            ]
        )
    )

    assert result.status == "fail"
    assert result.failure_type == FailureType.GENERATION_IGNORE
    assert result.stage == FailureStage.GENERATION


def test_deterministic_default_fallback() -> None:
    """No specific pattern matched → defaults to RETRIEVAL_DEPTH_LIMIT."""
    prior_results = [
        AnalyzerResult(
            analyzer_name="SomeAnalyzer",
            status="fail",
            failure_type=FailureType.STALE_RETRIEVAL,
            stage=FailureStage.RETRIEVAL,
            evidence=["Stale data"],
            remediation="Update index",
        )
    ]

    analyzer = A2PAttributionAnalyzer(
        {"use_llm": False, "prior_results": prior_results}
    )
    result = analyzer.analyze(
        run_with_chunks([chunk("chunk-1", "Some text", 0.6)])
    )

    assert result.status == "fail"
    assert result.failure_type == FailureType.RETRIEVAL_DEPTH_LIMIT
    assert result.stage == FailureStage.RETRIEVAL
    assert result.attribution_stage == FailureStage.RETRIEVAL
    assert result.fix_confidence == 0.45


def test_llm_mode_with_valid_json_response() -> None:
    """LLM mode parses JSON and returns structured attribution."""

    def mock_llm(prompt: str) -> str:
        assert "STEP 1 — ABDUCTION" in prompt
        assert "STEP 2 — ACTION" in prompt
        assert "STEP 3 — PREDICTION" in prompt
        assert "confidence_basis" in prompt
        return """{
            "abduction": "Chunks split logical units incorrectly",
            "root_cause_stage": "CHUNKING",
            "action": "Adjust chunk boundaries to preserve paragraphs",
            "prediction": "Applying the chunking fix would likely preserve context and resolve the observed inconsistency.",
            "confidence": 0.73,
            "confidence_basis": "Medium confidence because inconsistent chunks can also arise from parsing loss, but the evidence points most strongly to boundary errors."
        }"""

    prior_results = [
        AnalyzerResult(
            analyzer_name="test",
            status="fail",
            failure_type=FailureType.INCONSISTENT_CHUNKS,
            stage=FailureStage.RETRIEVAL,
            evidence=["test"],
            remediation="test",
        )
    ]

    analyzer = A2PAttributionAnalyzer(
        {"use_llm": True, "llm_fn": mock_llm, "prior_results": prior_results}
    )
    result = analyzer.analyze(
        run_with_chunks([chunk("chunk-1", "Text", 0.7)])
    )

    assert result.status == "fail"
    assert result.failure_type == FailureType.CHUNKING_BOUNDARY_ERROR
    assert result.stage == FailureStage.CHUNKING
    assert result.attribution_stage == FailureStage.CHUNKING
    assert result.proposed_fix == "Adjust chunk boundaries to preserve paragraphs"
    assert result.fix_confidence == 0.73
    assert "Chunks split logical units incorrectly" in result.evidence
    assert any("Prediction:" in evidence for evidence in result.evidence)
    assert any("Confidence basis:" in evidence for evidence in result.evidence)


def test_llm_mode_falls_back_on_json_parse_error() -> None:
    """LLM mode falls back to deterministic when JSON is invalid."""

    def mock_llm(prompt: str) -> str:
        return "This is not valid JSON"

    prior_results = [
        AnalyzerResult(
            analyzer_name="test",
            status="fail",
            failure_type=FailureType.INSUFFICIENT_CONTEXT,
            stage=FailureStage.SUFFICIENCY,
            evidence=["test"],
            remediation="test",
        )
    ]

    analyzer = A2PAttributionAnalyzer(
        {"use_llm": True, "llm_fn": mock_llm, "prior_results": prior_results}
    )
    result = analyzer.analyze(
        run_with_chunks([chunk("chunk-1", "Text", 0.5)])
    )

    # Should fall back to deterministic mode
    assert result.status == "fail"
    assert result.failure_type == FailureType.RETRIEVAL_DEPTH_LIMIT
    assert result.stage == FailureStage.RETRIEVAL


def test_a2p_runs_last_in_engine_and_overrides_stage() -> None:
    """A2P runs last and its stage overrides primary_failure stage in Diagnosis."""
    # Create a run that will trigger UNSUPPORTED_CLAIM with high scores
    # (should attribute to GENERATION_IGNORE)
    test_run = RAGRun(
        run_id="test-run",
        query="What is the policy?",
        retrieved_chunks=[
            chunk("chunk-1", "The policy covers X", 0.9),
            chunk("chunk-2", "The policy also covers Y", 0.85),
        ],
        final_answer="The policy guarantees 99% reimbursement for Z claims across all regions.",
    )

    # Run full engine with A2P enabled
    engine = DiagnosisEngine(config={"enable_a2p": True, "use_llm": False})
    diagnosis = engine.diagnose(test_run)

    # Check that A2P ran (should be last in analyzer_results)
    a2p_result = next(
        (r for r in diagnosis.analyzer_results if r.analyzer_name == "A2PAttributionAnalyzer"),
        None,
    )
    assert a2p_result is not None, "A2P analyzer should have run"

    # Check that A2P was the last analyzer
    assert diagnosis.analyzer_results[-1].analyzer_name == "A2PAttributionAnalyzer"

    # Check that root_cause_attribution fields are populated
    assert diagnosis.root_cause_attribution is not None
    assert diagnosis.proposed_fix is not None
    assert diagnosis.fix_confidence is not None

    # If A2P detected GENERATION_IGNORE, root_cause_stage should be GENERATION
    # (overriding the original GROUNDING stage from ClaimGroundingAnalyzer)
    if a2p_result.failure_type == FailureType.GENERATION_IGNORE:
        assert diagnosis.root_cause_stage == FailureStage.GENERATION


def test_diagnosis_summary_includes_proposed_fix() -> None:
    """Diagnosis.summary() includes proposed_fix when present."""
    test_run = RAGRun(
        run_id="test-run",
        query="What is X?",
        retrieved_chunks=[chunk("chunk-1", "Info", 0.5)],
        final_answer="Answer",
    )

    engine = DiagnosisEngine(config={"enable_a2p": True, "use_llm": False})
    diagnosis = engine.diagnose(test_run)

    summary = diagnosis.summary()

    # Summary should include fix (proposed or recommended) if A2P ran
    if diagnosis.proposed_fix:
        # New format is "Fix:" or "Fix (XX% confidence):"
        assert "Fix" in summary
        assert diagnosis.proposed_fix in summary


def test_a2p_skips_when_no_prior_failures() -> None:
    """A2P skips analysis when there are no prior failure results."""
    analyzer = A2PAttributionAnalyzer(
        {"use_llm": False, "prior_results": []}
    )
    result = analyzer.analyze(
        run_with_chunks([chunk("chunk-1", "Text", 0.8)])
    )

    assert result.status == "skip"
    assert "no prior failures" in result.evidence[0].lower()


def test_a2p_evidence_structure() -> None:
    """A2P includes abduction reasoning, proposed fix, and prediction in evidence."""
    prior_results = [
        AnalyzerResult(
            analyzer_name="test",
            status="fail",
            failure_type=FailureType.INSUFFICIENT_CONTEXT,
            stage=FailureStage.SUFFICIENCY,
            evidence=["test"],
            remediation="test",
        )
    ]

    analyzer = A2PAttributionAnalyzer(
        {"use_llm": False, "prior_results": prior_results}
    )
    result = analyzer.analyze(
        run_with_chunks([chunk("chunk-1", "Text", 0.5)])
    )

    assert len(result.evidence) == 4
    # Evidence should contain: abduction reasoning, proposed fix, prediction, confidence basis
    # First item is the reasoning (doesn't need specific keywords)
    assert len(result.evidence[0]) > 0  # Has reasoning text
    assert "Proposed fix:" in result.evidence[1]
    assert "Prediction:" in result.evidence[2]
    assert "Confidence basis:" in result.evidence[3]


def test_claim_level_a2p_unsupported_with_insufficient_context() -> None:
    prior_results = [
        AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
            claim_results=[
                ClaimResult(
                    claim_text="Policy includes international on-site repairs.",
                    label="unsupported",
                    supporting_chunk_ids=[],
                )
            ],
        ),
        AnalyzerResult(
            analyzer_name="SufficiencyAnalyzer",
            status="fail",
            failure_type=FailureType.INSUFFICIENT_CONTEXT,
            stage=FailureStage.SUFFICIENCY,
            sufficiency_result=SufficiencyResult(
                sufficient=False,
                missing_evidence=["Policy includes international on-site repairs."],
                affected_claims=["Policy includes international on-site repairs."],
                evidence_chunk_ids=[],
                method="heuristic_claim_aware_v0",
                calibration_status="uncalibrated",
            ),
        ),
    ]
    result = A2PAttributionAnalyzer({"use_llm": False, "prior_results": prior_results}).analyze(
        run_with_chunks([chunk("chunk-1", "Base policy text", 0.7)])
    )

    assert result.claim_attributions is not None
    assert result.claim_attributions[0].primary_cause == "insufficient_context_or_retrieval_miss"
    assert result.claim_attributions[0].attribution_method == "claim_level_a2p_heuristic_v1"
    assert result.claim_attributions[0].calibration_status == "uncalibrated"


def test_claim_level_a2p_contradicted_claim_with_contradicting_chunk() -> None:
    prior_results = [
        AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.CONTRADICTED_CLAIM,
            stage=FailureStage.GROUNDING,
            claim_results=[
                ClaimResult(
                    claim_text="Warranty covers accidental damage.",
                    label="contradicted",
                    supporting_chunk_ids=[],
                    candidate_chunk_ids=["chunk-1"],
                    contradicting_chunk_ids=["chunk-1"],
                )
            ],
        ),
    ]
    result = A2PAttributionAnalyzer({"use_llm": False, "prior_results": prior_results}).analyze(
        run_with_chunks([chunk("chunk-1", "Warranty does not cover accidental damage.", 0.9)])
    )

    assert result.claim_attributions is not None
    assert result.claim_attributions[0].primary_cause == "generation_contradicted_retrieved_evidence"


def test_claim_level_a2p_adds_stale_and_citation_candidate_causes() -> None:
    prior_results = [
        AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
            claim_results=[
                ClaimResult(
                    claim_text="Claim text",
                    label="unsupported",
                    supporting_chunk_ids=[],
                    candidate_chunk_ids=["chunk-1"],
                    evidence_reason="Best chunk score below support threshold.",
                )
            ],
        ),
        AnalyzerResult(
            analyzer_name="CitationMismatchAnalyzer",
            status="fail",
            failure_type=FailureType.CITATION_MISMATCH,
            stage=FailureStage.RETRIEVAL,
        ),
        AnalyzerResult(
            analyzer_name="StaleRetrievalAnalyzer",
            status="fail",
            failure_type=FailureType.STALE_RETRIEVAL,
            stage=FailureStage.RETRIEVAL,
        ),
    ]
    result = A2PAttributionAnalyzer({"use_llm": False, "prior_results": prior_results}).analyze(
        run_with_chunks([chunk("chunk-1", "Some context", 0.8)])
    )

    assert result.claim_attributions is not None
    causes = set(result.claim_attributions[0].candidate_causes)
    assert "citation_mismatch" in causes
    assert "stale_source_usage" in causes


def test_claim_level_a2p_adds_verification_uncertainty_when_claim_fallback_used() -> None:
    prior_results = [
        AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
            claim_results=[
                ClaimResult(
                    claim_text="Claim text",
                    label="unsupported",
                    supporting_chunk_ids=[],
                    fallback_used=True,
                )
            ],
        ),
        AnalyzerResult(
            analyzer_name="SufficiencyAnalyzer",
            status="fail",
            failure_type=FailureType.INSUFFICIENT_CONTEXT,
            stage=FailureStage.SUFFICIENCY,
            sufficiency_result=SufficiencyResult(
                sufficient=False,
                missing_evidence=["Claim text"],
                affected_claims=["Claim text"],
                evidence_chunk_ids=[],
                method="heuristic_claim_aware_v0",
                calibration_status="uncalibrated",
            ),
        ),
    ]
    result = A2PAttributionAnalyzer({"use_llm": False, "prior_results": prior_results}).analyze(
        run_with_chunks([chunk("chunk-1", "Some context", 0.6)])
    )

    assert result.claim_attributions is not None
    assert "verification_uncertainty" in result.claim_attributions[0].candidate_causes


def test_a2p_legacy_fallback_visible_without_typed_claims() -> None:
    prior_results = [
        AnalyzerResult(
            analyzer_name="SufficiencyAnalyzer",
            status="fail",
            failure_type=FailureType.INSUFFICIENT_CONTEXT,
            stage=FailureStage.SUFFICIENCY,
            evidence=["Context is insufficient"],
        )
    ]
    result = A2PAttributionAnalyzer({"use_llm": False, "prior_results": prior_results}).analyze(
        run_with_chunks([chunk("chunk-1", "Some text", 0.4)])
    )

    assert result.claim_attributions is not None
    assert result.claim_attributions[0].fallback_used is True
    assert result.claim_attributions[0].attribution_method == "legacy_failure_level_heuristic"


def test_claim_level_a2p_uses_claim_aware_sufficiency_result_when_available() -> None:
    prior_results = [
        AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
            claim_results=[
                ClaimResult(
                    claim_text="Missing legal section details for export restrictions.",
                    label="unsupported",
                    supporting_chunk_ids=[],
                )
            ],
        ),
        AnalyzerResult(
            analyzer_name="ClaimAwareSufficiencyAnalyzer",
            status="pass",
            sufficiency_result=SufficiencyResult(
                sufficient=False,
                missing_evidence=["Missing legal section details for export restrictions."],
                affected_claims=["Missing legal section details for export restrictions."],
                evidence_chunk_ids=[],
                method="heuristic_claim_aware_v0",
                calibration_status="uncalibrated",
            ),
        ),
    ]

    result = A2PAttributionAnalyzer({"use_llm": False, "prior_results": prior_results}).analyze(
        run_with_chunks([chunk("chunk-1", "Office policy context", 0.6)])
    )

    assert result.claim_attributions is not None
    assert result.claim_attributions[0].primary_cause == "insufficient_context_or_retrieval_miss"


def test_claim_level_a2p_unsupported_with_only_candidate_and_insufficient_is_retrieval_miss() -> None:
    prior_results = [
        AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
            claim_results=[
                ClaimResult(
                    claim_text="Policy includes new emergency reimbursement.",
                    label="unsupported",
                    supporting_chunk_ids=[],
                    candidate_chunk_ids=["chunk-1"],
                    contradicting_chunk_ids=[],
                )
            ],
        ),
        AnalyzerResult(
            analyzer_name="ClaimAwareSufficiencyAnalyzer",
            status="pass",
            sufficiency_result=SufficiencyResult(
                sufficient=False,
                missing_evidence=["Policy includes new emergency reimbursement."],
                affected_claims=["Policy includes new emergency reimbursement."],
                evidence_chunk_ids=["chunk-1"],
                method="heuristic_claim_aware_v0",
                calibration_status="uncalibrated",
            ),
        ),
    ]

    result = A2PAttributionAnalyzer({"use_llm": False, "prior_results": prior_results}).analyze(
        run_with_chunks([chunk("chunk-1", "Policy text with no reimbursement clause", 0.6)])
    )

    assert result.claim_attributions is not None
    assert result.claim_attributions[0].primary_cause == "insufficient_context_or_retrieval_miss"


def test_claim_level_a2p_entailed_stale_claim_maps_to_stale_source_usage_in_v1() -> None:
    prior_results = [
        AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="pass",
            claim_results=[
                ClaimResult(
                    claim_text="Records are retained for twelve months before archival review.",
                    label="entailed",
                    supporting_chunk_ids=["doc7-chunk-1"],
                )
            ],
        ),
        AnalyzerResult(
            analyzer_name="StaleRetrievalAnalyzer",
            status="fail",
            failure_type=FailureType.STALE_RETRIEVAL,
            stage=FailureStage.RETRIEVAL,
            evidence=["doc7 is 2000 days old"],
        ),
    ]

    result = A2PAttributionAnalyzer({"use_llm": False, "prior_results": prior_results}).analyze(
        run_with_chunks([chunk("doc7-chunk-1", "Records are retained for twelve months before archival review.", 0.8)])
    )

    assert result.claim_attributions is not None
    assert result.claim_attributions[0].primary_cause == "stale_source_usage"
    assert result.stage == FailureStage.RETRIEVAL


def test_claim_level_a2p_entailed_citation_invalid_maps_to_citation_mismatch_in_v1() -> None:
    run = RAGRun(
        query="What refund threshold applies?",
        retrieved_chunks=[chunk("chunk-1", "Refund approvals require manager review for requests above one hundred dollars.", 0.82)],
        final_answer="Refund approvals require manager review for requests above one hundred dollars.",
        cited_doc_ids=["doc-phantom-1"],
    )
    prior_results = [
        AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="pass",
            claim_results=[
                ClaimResult(
                    claim_text="Refund approvals require manager review for requests above one hundred dollars.",
                    label="entailed",
                    supporting_chunk_ids=["chunk-1"],
                )
            ],
        ),
        AnalyzerResult(
            analyzer_name="CitationMismatchAnalyzer",
            status="fail",
            failure_type=FailureType.CITATION_MISMATCH,
            stage=FailureStage.RETRIEVAL,
            evidence=["phantom citation: doc-phantom-1"],
        ),
    ]

    result = A2PAttributionAnalyzer({"use_llm": False, "prior_results": prior_results}).analyze(run)

    assert result.claim_attributions is not None
    assert result.claim_attributions[0].primary_cause == "citation_mismatch"
    assert result.stage == FailureStage.RETRIEVAL


def test_claim_level_a2p_entailed_security_risk_maps_to_adversarial_context_in_v1() -> None:
    prior_results = [
        AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="pass",
            claim_results=[
                ClaimResult(
                    claim_text="The refund window is fourteen days.",
                    label="entailed",
                    supporting_chunk_ids=["chunk-1"],
                )
            ],
        ),
        AnalyzerResult(
            analyzer_name="PromptInjectionAnalyzer",
            status="fail",
            failure_type=FailureType.PROMPT_INJECTION,
            stage=FailureStage.SECURITY,
            evidence=["instruction-like content detected in chunk-2"],
        ),
    ]

    result = A2PAttributionAnalyzer({"use_llm": False, "prior_results": prior_results}).analyze(
        run_with_chunks([chunk("chunk-1", "The refund window is fourteen days.", 0.9)])
    )

    assert result.claim_attributions is not None
    assert result.claim_attributions[0].primary_cause == "adversarial_context"
    assert result.stage == FailureStage.SECURITY


def test_claim_level_a2p_entailed_citation_faithfulness_fail_maps_to_post_rationalized_citation() -> None:
    prior_results = [
        AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="pass",
            claim_results=[
                ClaimResult(
                    claim_text="The refund window is fourteen days.",
                    label="entailed",
                    supporting_chunk_ids=["chunk-1"],
                )
            ],
        ),
        AnalyzerResult(
            analyzer_name="CitationFaithfulnessProbe",
            status="fail",
            failure_type=FailureType.POST_RATIONALIZED_CITATION,
            stage=FailureStage.GROUNDING,
            evidence=["citation appears attached after answer generation"],
        ),
    ]

    result = A2PAttributionAnalyzer({"use_llm": False, "prior_results": prior_results}).analyze(
        run_with_chunks([chunk("chunk-1", "The refund window is fourteen days.", 0.9)])
    )

    assert result.claim_attributions is not None
    assert result.claim_attributions[0].primary_cause == "post_rationalized_citation"
    assert result.stage == FailureStage.GROUNDING


def test_claim_level_a2p_v2_entailed_security_risk_maps_to_adversarial_context() -> None:
    prior_results = [
        AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="pass",
            claim_results=[
                ClaimResult(
                    claim_text="The refund window is fourteen days.",
                    label="entailed",
                    supporting_chunk_ids=["chunk-1"],
                )
            ],
        ),
        AnalyzerResult(
            analyzer_name="PromptInjectionAnalyzer",
            status="fail",
            failure_type=FailureType.PROMPT_INJECTION,
            stage=FailureStage.SECURITY,
            evidence=["instruction-like content detected in chunk-2"],
        ),
    ]

    result = A2PAttributionAnalyzer(
        {"use_llm": False, "use_a2p_v2": True, "prior_results": prior_results}
    ).analyze(run_with_chunks([chunk("chunk-1", "The refund window is fourteen days.", 0.9)]))

    assert result.claim_attributions_v2 is not None
    assert result.claim_attributions_v2[0].primary_cause == "adversarial_context"
    assert result.stage == FailureStage.SECURITY
