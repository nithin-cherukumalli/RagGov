"""Tests for A2P attribution analyzer."""

from __future__ import annotations

from raggov.analyzers.attribution.a2p import A2PAttributionAnalyzer
from raggov.engine import DiagnosisEngine
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
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
        final_answer="The policy covers Z which is not in chunks",
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
