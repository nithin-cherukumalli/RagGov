"""Tests for Layer6 taxonomy classifier."""

from __future__ import annotations

import json

from raggov.analyzers.taxonomy_classifier.layer6 import Layer6TaxonomyClassifier
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


def test_scope_violation_maps_to_off_topic_retrieval() -> None:
    """SCOPE_VIOLATION → RETRIEVAL stage, off_topic_retrieval mode."""
    prior_results = [
        AnalyzerResult(
            analyzer_name="ScopeViolationAnalyzer",
            status="fail",
            failure_type=FailureType.SCOPE_VIOLATION,
            stage=FailureStage.RETRIEVAL,
            evidence=["Query and retrieved docs are off-topic"],
            remediation="Review retrieval logic",
        )
    ]

    analyzer = Layer6TaxonomyClassifier({"prior_results": prior_results})
    result = analyzer.analyze(
        run_with_chunks([chunk("chunk-1", "Text", 0.7)])
    )

    assert result.status == "fail"
    assert result.stage == FailureStage.RETRIEVAL

    # Parse the Layer6 report from evidence
    assert len(result.evidence) > 0
    report = json.loads(result.evidence[0])

    assert report["primary_stage"] == "RETRIEVAL"
    assert len(report["stage_failures"]) > 0

    # Find the retrieval failure
    retrieval_failure = next(
        (f for f in report["stage_failures"] if f["stage"] == "RETRIEVAL"), None
    )
    assert retrieval_failure is not None
    assert retrieval_failure["failure_mode"] == "off_topic_retrieval"

    # Check failure chain
    assert "RETRIEVAL" in report["failure_chain"][0]
    assert "off_topic_retrieval" in report["failure_chain"][0]

    # Check engineer action
    assert "embedding model" in report["engineer_action"].lower() or "HyDE" in report["engineer_action"]


def test_parser_structure_loss_maps_to_parsing_not_chunking() -> None:
    """Parser failures should be attributed to PARSING with lost_structure."""
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

    analyzer = Layer6TaxonomyClassifier({"prior_results": prior_results})
    result = analyzer.analyze(run_with_chunks([chunk("chunk-1", "District Vacancies Category", 0.72)]))

    assert result.status == "fail"
    assert result.stage == FailureStage.PARSING

    report = json.loads(result.evidence[0])
    assert report["primary_stage"] == "PARSING"
    assert report["engineer_action"] == remediation

    parsing_failure = next(
        (f for f in report["stage_failures"] if f["stage"] == "PARSING"),
        None,
    )
    assert parsing_failure is not None
    assert parsing_failure["failure_mode"] == "lost_structure"


def test_ncv_report_prefers_earlier_retrieval_stage_as_tiebreaker() -> None:
    """NCV should move a later-stage heuristic failure earlier when appropriate."""
    prior_results = [
        AnalyzerResult(
            analyzer_name="NCVPipelineVerifier",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
            evidence=[
                '{"node_results": [], "first_failing_node": "RETRIEVAL_QUALITY", '
                '"pipeline_health_score": 0.5, '
                '"bottleneck_description": "Pipeline fails at RETRIEVAL_QUALITY: low mean score"}'
            ],
            remediation="Increase retrieval quality.",
        ),
        AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
            evidence=["Claim not grounded in context"],
            remediation="Verify sources",
        ),
    ]
    ncv_report = {
        "node_results": [],
        "first_failing_node": "RETRIEVAL_QUALITY",
        "pipeline_health_score": 0.5,
        "bottleneck_description": "Pipeline fails at RETRIEVAL_QUALITY: low mean score",
    }

    analyzer = Layer6TaxonomyClassifier({"prior_results": prior_results, "ncv_report": ncv_report})
    result = analyzer.analyze(run_with_chunks([chunk("chunk-1", "Low relevance", 0.90)]))

    assert result.status == "fail"
    assert result.stage == FailureStage.RETRIEVAL

    report = json.loads(result.evidence[0])
    assert report["primary_stage"] == "RETRIEVAL"
    retrieval_failure = next(
        (f for f in report["stage_failures"] if f["stage"] == "RETRIEVAL"),
        None,
    )
    assert retrieval_failure is not None
    assert retrieval_failure["failure_mode"] == "missing_relevant_docs"


def test_ncv_report_does_not_override_parsing_signal() -> None:
    """NCV tiebreaker should not override existing parsing failures."""
    prior_results = [
        AnalyzerResult(
            analyzer_name="ParserValidationAnalyzer",
            status="fail",
            failure_type=FailureType.TABLE_STRUCTURE_LOSS,
            stage=FailureStage.PARSING,
            evidence=["Table keywords detected but structural separators absent in chunk chunk-1"],
            remediation="Preserve tables.",
        ),
        AnalyzerResult(
            analyzer_name="NCVPipelineVerifier",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
            evidence=[
                '{"node_results": [], "first_failing_node": "CLAIM_GROUNDING", '
                '"pipeline_health_score": 0.6, '
                '"bottleneck_description": "Pipeline fails at CLAIM_GROUNDING: answer is not anchored"}'
            ],
            remediation="Strengthen grounding.",
        ),
    ]
    ncv_report = {
        "node_results": [],
        "first_failing_node": "CLAIM_GROUNDING",
        "pipeline_health_score": 0.6,
        "bottleneck_description": "Pipeline fails at CLAIM_GROUNDING: answer is not anchored",
    }

    analyzer = Layer6TaxonomyClassifier({"prior_results": prior_results, "ncv_report": ncv_report})
    result = analyzer.analyze(run_with_chunks([chunk("chunk-1", "District Vacancies Category", 0.72)]))

    report = json.loads(result.evidence[0])
    assert report["primary_stage"] == "PARSING"


def test_concurrent_stage_failures_use_earliest_causal_stage_as_primary() -> None:
    """When multiple stages fail, the earliest causal stage should be primary."""
    weighted_prior_results = [
        AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
            evidence=["Claim not grounded in context"],
            remediation="Strengthen grounding instructions",
        ),
        AnalyzerResult(
            analyzer_name="InconsistentChunksAnalyzer",
            status="fail",
            failure_type=FailureType.INCONSISTENT_CHUNKS,
            stage=FailureStage.RETRIEVAL,
            evidence=["Chunks contain inconsistencies"],
            remediation="Review chunking",
        ),
    ]

    analyzer = Layer6TaxonomyClassifier(
        {
            "prior_results": list(reversed(weighted_prior_results)),
            "weighted_prior_results": weighted_prior_results,
            "analyzer_weights": {
                "ClaimGroundingAnalyzer": 0.9,
                "InconsistentChunksAnalyzer": 0.5,
            },
        }
    )
    result = analyzer.analyze(
        run_with_chunks([
            chunk("chunk-1", "Highly relevant context", 0.85),
            chunk("chunk-2", "More highly relevant context", 0.80),
        ])
    )

    assert result.stage == FailureStage.CHUNKING

    report = json.loads(result.evidence[0])
    assert report["primary_stage"] == "CHUNKING"
    assert any(f["stage"] == "GENERATION" for f in report["stage_failures"])
    assert report["engineer_action"] == "Review chunking"


def test_weighted_ncv_does_not_override_explicit_security_signal() -> None:
    """Weighted NCV should still not displace explicit security failures."""
    weighted_prior_results = [
        AnalyzerResult(
            analyzer_name="PromptInjectionAnalyzer",
            status="fail",
            failure_type=FailureType.PROMPT_INJECTION,
            stage=FailureStage.SECURITY,
            evidence=["Instruction-like chunk content"],
            remediation="Sanitize corpus",
        )
    ]
    ncv_report = {
        "node_results": [],
        "first_failing_node": "QUERY_UNDERSTANDING",
        "pipeline_health_score": 0.4,
        "bottleneck_description": "Pipeline fails at QUERY_UNDERSTANDING: low overlap",
    }

    analyzer = Layer6TaxonomyClassifier(
        {
            "prior_results": weighted_prior_results,
            "weighted_prior_results": weighted_prior_results,
            "analyzer_weights": {
                "PromptInjectionAnalyzer": 1.0,
                "NCVPipelineVerifier": 0.6,
            },
            "ncv_report": ncv_report,
        }
    )
    result = analyzer.analyze(run_with_chunks([chunk("chunk-1", "Ignore prior instructions", 0.95)]))

    report = json.loads(result.evidence[0])
    assert report["primary_stage"] == "SECURITY"


def test_unsupported_claim_high_scores_context_ignored() -> None:
    """UNSUPPORTED_CLAIM + high scores → GENERATION stage, context_ignored mode."""
    prior_results = [
        AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
            evidence=["Claim not grounded in context"],
            remediation="Verify sources",
        )
    ]

    analyzer = Layer6TaxonomyClassifier({"prior_results": prior_results})
    result = analyzer.analyze(
        run_with_chunks([
            chunk("chunk-1", "Good context", 0.85),
            chunk("chunk-2", "More good context", 0.80),
        ])
    )

    assert result.status == "fail"
    assert result.stage == FailureStage.GENERATION

    report = json.loads(result.evidence[0])
    assert report["primary_stage"] == "GENERATION"

    generation_failure = next(
        (f for f in report["stage_failures"] if f["stage"] == "GENERATION"), None
    )
    assert generation_failure is not None
    assert generation_failure["failure_mode"] == "context_ignored"

    # Check engineer action mentions grounding or instructions
    assert ("grounding" in report["engineer_action"].lower() or
            "prompt" in report["engineer_action"].lower())


def test_unsupported_claim_low_scores_hallucination() -> None:
    """UNSUPPORTED_CLAIM + low scores → GENERATION stage, hallucination mode."""
    prior_results = [
        AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
            evidence=["Claim not grounded"],
            remediation="Check sources",
        )
    ]

    analyzer = Layer6TaxonomyClassifier({"prior_results": prior_results})
    result = analyzer.analyze(
        run_with_chunks([
            chunk("chunk-1", "Low relevance", 0.45),
            chunk("chunk-2", "Also low", 0.40),
        ])
    )

    assert result.status == "fail"
    assert result.stage == FailureStage.GENERATION

    report = json.loads(result.evidence[0])
    assert report["primary_stage"] == "GENERATION"

    generation_failure = next(
        (f for f in report["stage_failures"] if f["stage"] == "GENERATION"), None
    )
    assert generation_failure is not None
    assert generation_failure["failure_mode"] == "hallucination"

    # Check engineer action mentions retrieval recall
    assert ("retrieval" in report["engineer_action"].lower() or
            "recall" in report["engineer_action"].lower())


def test_unsupported_claim_ambiguous_scores_mixed_quality() -> None:
    """UNSUPPORTED_CLAIM + mid-band scores → GENERATION stage, mixed_quality mode."""
    prior_results = [
        AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
            evidence=["Claim not grounded"],
            remediation="Check sources",
        )
    ]

    analyzer = Layer6TaxonomyClassifier({"prior_results": prior_results})
    result = analyzer.analyze(
        run_with_chunks([
            chunk("chunk-1", "Ambiguous relevance", 0.68),
            chunk("chunk-2", "Also ambiguous", 0.66),
        ])
    )

    assert result.status == "fail"
    assert result.stage == FailureStage.GENERATION

    report = json.loads(result.evidence[0])
    assert report["primary_stage"] == "GENERATION"

    generation_failure = next(
        (f for f in report["stage_failures"] if f["stage"] == "GENERATION"), None
    )
    assert generation_failure is not None
    assert generation_failure["failure_mode"] == "mixed_quality"
    assert "Ambiguous" in report["engineer_action"]


def test_stale_retrieval_maps_correctly() -> None:
    """STALE_RETRIEVAL → RETRIEVAL stage, stale_docs mode."""
    prior_results = [
        AnalyzerResult(
            analyzer_name="StaleRetrievalAnalyzer",
            status="fail",
            failure_type=FailureType.STALE_RETRIEVAL,
            stage=FailureStage.RETRIEVAL,
            evidence=["Documents are outdated"],
            remediation="Re-index corpus",
        )
    ]

    analyzer = Layer6TaxonomyClassifier({"prior_results": prior_results})
    result = analyzer.analyze(
        run_with_chunks([chunk("chunk-1", "Stale doc", 0.7)])
    )

    assert result.status == "fail"
    assert result.stage == FailureStage.RETRIEVAL

    report = json.loads(result.evidence[0])
    assert report["primary_stage"] == "RETRIEVAL"

    retrieval_failure = next(
        (f for f in report["stage_failures"] if f["stage"] == "RETRIEVAL"), None
    )
    assert retrieval_failure is not None
    assert retrieval_failure["failure_mode"] == "stale_docs"

    # Check engineer action mentions re-indexing
    assert "re-index" in report["engineer_action"].lower() or "freshness" in report["engineer_action"].lower()


def test_insufficient_context_low_scores_missing_docs() -> None:
    """INSUFFICIENT_CONTEXT + chunk scores < 0.65 → missing_relevant_docs."""
    prior_results = [
        AnalyzerResult(
            analyzer_name="SufficiencyAnalyzer",
            status="fail",
            failure_type=FailureType.INSUFFICIENT_CONTEXT,
            stage=FailureStage.SUFFICIENCY,
            evidence=["Not enough context"],
            remediation="Expand retrieval",
        )
    ]

    analyzer = Layer6TaxonomyClassifier({"prior_results": prior_results})
    result = analyzer.analyze(
        run_with_chunks([
            chunk("chunk-1", "Low score", 0.50),
            chunk("chunk-2", "Also low", 0.55),
        ])
    )

    assert result.status == "fail"

    report = json.loads(result.evidence[0])
    assert report["primary_stage"] == "RETRIEVAL"

    retrieval_failure = next(
        (f for f in report["stage_failures"] if f["stage"] == "RETRIEVAL"), None
    )
    assert retrieval_failure is not None
    assert retrieval_failure["failure_mode"] == "missing_relevant_docs"


def test_insufficient_context_few_chunks_top_k_too_small() -> None:
    """INSUFFICIENT_CONTEXT + fewer than 3 chunks + varying scores → top_k_too_small."""
    prior_results = [
        AnalyzerResult(
            analyzer_name="SufficiencyAnalyzer",
            status="fail",
            failure_type=FailureType.INSUFFICIENT_CONTEXT,
            stage=FailureStage.SUFFICIENCY,
            evidence=["Not enough context"],
            remediation="Expand retrieval",
        )
    ]

    analyzer = Layer6TaxonomyClassifier({"prior_results": prior_results})
    result = analyzer.analyze(
        run_with_chunks([
            chunk("chunk-1", "High", 0.90),
            chunk("chunk-2", "Low", 0.40),
        ])
    )

    assert result.status == "fail"

    report = json.loads(result.evidence[0])

    retrieval_failure = next(
        (f for f in report["stage_failures"] if f["stage"] == "RETRIEVAL"), None
    )
    assert retrieval_failure is not None
    assert retrieval_failure["failure_mode"] == "top_k_too_small"

    # Check engineer action mentions top-k
    assert "top-k" in report["engineer_action"].lower()


def test_inconsistent_chunks_boundary_errors() -> None:
    """INCONSISTENT_CHUNKS → CHUNKING stage, boundary_errors mode."""
    prior_results = [
        AnalyzerResult(
            analyzer_name="InconsistentChunksAnalyzer",
            status="fail",
            failure_type=FailureType.INCONSISTENT_CHUNKS,
            stage=FailureStage.RETRIEVAL,
            evidence=["Chunks contain inconsistencies"],
            remediation="Review chunking",
        )
    ]

    analyzer = Layer6TaxonomyClassifier({"prior_results": prior_results})
    result = analyzer.analyze(
        run_with_chunks([chunk("chunk-1", "Inconsistent", 0.7)])
    )

    assert result.status == "fail"
    assert result.stage == FailureStage.CHUNKING

    report = json.loads(result.evidence[0])
    assert report["primary_stage"] == "CHUNKING"

    chunking_failure = next(
        (f for f in report["stage_failures"] if f["stage"] == "CHUNKING"), None
    )
    assert chunking_failure is not None
    assert chunking_failure["failure_mode"] == "boundary_errors"


def test_contradicted_claim_over_extraction() -> None:
    """CONTRADICTED_CLAIM → GENERATION stage, over_extraction mode."""
    prior_results = [
        AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.CONTRADICTED_CLAIM,
            stage=FailureStage.GROUNDING,
            evidence=["Claim contradicts context"],
            remediation="Fix extraction",
        )
    ]

    analyzer = Layer6TaxonomyClassifier({"prior_results": prior_results})
    result = analyzer.analyze(
        run_with_chunks([chunk("chunk-1", "Context", 0.8)])
    )

    assert result.status == "fail"
    assert result.stage == FailureStage.GENERATION

    report = json.loads(result.evidence[0])
    assert report["primary_stage"] == "GENERATION"

    generation_failure = next(
        (f for f in report["stage_failures"] if f["stage"] == "GENERATION"), None
    )
    assert generation_failure is not None
    assert generation_failure["failure_mode"] == "over_extraction"


def test_prompt_injection_security_failure() -> None:
    """PROMPT_INJECTION → SECURITY stage."""
    prior_results = [
        AnalyzerResult(
            analyzer_name="PromptInjectionAnalyzer",
            status="fail",
            failure_type=FailureType.PROMPT_INJECTION,
            stage=FailureStage.SECURITY,
            evidence=["Injection detected"],
            remediation="Sanitize input",
        )
    ]

    analyzer = Layer6TaxonomyClassifier({"prior_results": prior_results})
    result = analyzer.analyze(
        run_with_chunks([chunk("chunk-1", "Injection", 0.8)])
    )

    assert result.status == "fail"
    assert result.stage == FailureStage.SECURITY

    report = json.loads(result.evidence[0])
    assert report["primary_stage"] == "SECURITY"

    security_failure = next(
        (f for f in report["stage_failures"] if f["stage"] == "SECURITY"), None
    )
    assert security_failure is not None
    assert security_failure["failure_mode"] == "prompt_injection"


def test_retrieval_anomaly_noisy_results() -> None:
    """RETRIEVAL_ANOMALY → RETRIEVAL stage, noisy_or_misranked_results mode."""
    prior_results = [
        AnalyzerResult(
            analyzer_name="RetrievalAnomalyAnalyzer",
            status="fail",
            failure_type=FailureType.RETRIEVAL_ANOMALY,
            stage=FailureStage.RETRIEVAL,
            evidence=["Anomaly detected"],
            remediation="Audit corpus",
        )
    ]

    analyzer = Layer6TaxonomyClassifier({"prior_results": prior_results})
    result = analyzer.analyze(
        run_with_chunks([chunk("chunk-1", "Anomaly", 0.99)])
    )

    assert result.status == "fail"
    assert result.stage == FailureStage.RETRIEVAL

    report = json.loads(result.evidence[0])

    retrieval_failure = next(
        (f for f in report["stage_failures"] if f["stage"] == "RETRIEVAL"), None
    )
    assert retrieval_failure is not None
    assert retrieval_failure["failure_mode"] == "noisy_or_misranked_results"

def test_suspicious_chunk_still_maps_to_security() -> None:
    """SUSPICIOUS_CHUNK → SECURITY stage, corpus_poisoning mode."""
    prior_results = [
        AnalyzerResult(
            analyzer_name="PoisoningHeuristicAnalyzer",
            status="fail",
            failure_type=FailureType.SUSPICIOUS_CHUNK,
            stage=FailureStage.SECURITY,
            evidence=["Suspicious chunk detected"],
            remediation="Audit corpus",
        )
    ]

    analyzer = Layer6TaxonomyClassifier({"prior_results": prior_results})
    result = analyzer.analyze(
        run_with_chunks([chunk("chunk-1", "Suspicious", 0.99)])
    )

    assert result.status == "fail"
    assert result.stage == FailureStage.SECURITY

    report = json.loads(result.evidence[0])

    security_failure = next(
        (f for f in report["stage_failures"] if f["stage"] == "SECURITY"), None
    )
    assert security_failure is not None
    assert security_failure["failure_mode"] == "corpus_poisoning"


def test_failure_chain_ordering() -> None:
    """Verify failure chain has correct stage ordering."""
    # Multiple failures across stages
    prior_results = [
        AnalyzerResult(
            analyzer_name="ScopeViolationAnalyzer",
            status="fail",
            failure_type=FailureType.SCOPE_VIOLATION,
            stage=FailureStage.RETRIEVAL,
            evidence=["Off topic"],
            remediation="Fix",
        ),
        AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
            evidence=["Not grounded"],
            remediation="Fix",
        ),
    ]

    analyzer = Layer6TaxonomyClassifier({"prior_results": prior_results})
    result = analyzer.analyze(
        run_with_chunks([chunk("chunk-1", "Low", 0.50)])
    )

    assert result.status == "fail"

    report = json.loads(result.evidence[0])

    # Failure chain should show progression
    assert len(report["failure_chain"]) >= 2

    # RETRIEVAL should come before GENERATION in the chain
    retrieval_idx = next(
        (i for i, fc in enumerate(report["failure_chain"]) if "RETRIEVAL" in fc), -1
    )
    generation_idx = next(
        (i for i, fc in enumerate(report["failure_chain"]) if "GENERATION" in fc), -1
    )

    if retrieval_idx != -1 and generation_idx != -1:
        assert retrieval_idx < generation_idx, "RETRIEVAL should come before GENERATION"


def test_engineer_action_present_for_all_paths() -> None:
    """Verify engineer_action is non-empty for all failure modes."""
    failure_scenarios = [
        (FailureType.SCOPE_VIOLATION, FailureStage.RETRIEVAL),
        (FailureType.STALE_RETRIEVAL, FailureStage.RETRIEVAL),
        (FailureType.INSUFFICIENT_CONTEXT, FailureStage.SUFFICIENCY),
        (FailureType.UNSUPPORTED_CLAIM, FailureStage.GROUNDING),
        (FailureType.CONTRADICTED_CLAIM, FailureStage.GROUNDING),
        (FailureType.INCONSISTENT_CHUNKS, FailureStage.RETRIEVAL),
        (FailureType.PROMPT_INJECTION, FailureStage.SECURITY),
    ]

    for failure_type, stage in failure_scenarios:
        prior_results = [
            AnalyzerResult(
                analyzer_name="TestAnalyzer",
                status="fail",
                failure_type=failure_type,
                stage=stage,
                evidence=["Test evidence"],
                remediation="Test fix",
            )
        ]

        analyzer = Layer6TaxonomyClassifier({"prior_results": prior_results})
        result = analyzer.analyze(
            run_with_chunks([chunk("chunk-1", "Text", 0.7)])
        )

        assert result.status == "fail"
        report = json.loads(result.evidence[0])
        assert len(report["engineer_action"]) > 0, f"Engineer action missing for {failure_type}"
        assert report["engineer_action"] != "", f"Engineer action empty for {failure_type}"


def test_no_prior_failures_skips_analysis() -> None:
    """Analyzer skips when no prior failures detected."""
    analyzer = Layer6TaxonomyClassifier({"prior_results": []})
    result = analyzer.analyze(
        run_with_chunks([chunk("chunk-1", "Text", 0.8)])
    )

    assert result.status == "skip"
    assert "no prior failures" in result.evidence[0].lower()


def test_missing_chunk_scores_handled_gracefully() -> None:
    """Analyzer handles missing chunk scores gracefully."""
    prior_results = [
        AnalyzerResult(
            analyzer_name="SufficiencyAnalyzer",
            status="fail",
            failure_type=FailureType.INSUFFICIENT_CONTEXT,
            stage=FailureStage.SUFFICIENCY,
            evidence=["Not enough context"],
            remediation="Expand",
        )
    ]

    analyzer = Layer6TaxonomyClassifier({"prior_results": prior_results})
    # Chunks without scores
    result = analyzer.analyze(
        run_with_chunks([
            chunk("chunk-1", "No score", None),
            chunk("chunk-2", "Also no score", None),
        ])
    )

    # Should still work, just treat as empty scores
    assert result.status == "fail"
    report = json.loads(result.evidence[0])
    assert len(report["stage_failures"]) > 0


def test_integration_with_engine() -> None:
    """Integration test: DiagnosisEngine populates layer6_report in Diagnosis."""
    test_run = RAGRun(
        run_id="test-layer6-integration",
        query="What is the policy?",
        retrieved_chunks=[
            chunk("chunk-1", "Off-topic content", 0.45),
            chunk("chunk-2", "Also off-topic", 0.40),
        ],
        final_answer="The policy says something not in the chunks",
    )

    # Run full engine
    engine = DiagnosisEngine(config={})
    diagnosis = engine.diagnose(test_run)

    # Check that Layer6TaxonomyClassifier ran
    layer6_result = next(
        (r for r in diagnosis.analyzer_results if r.analyzer_name == "Layer6TaxonomyClassifier"),
        None,
    )
    assert layer6_result is not None, "Layer6TaxonomyClassifier should have run"

    # Check that layer6_report is populated in Diagnosis
    assert diagnosis.layer6_report is not None
    assert isinstance(diagnosis.layer6_report, dict)

    # Check that failure_chain is populated
    assert len(diagnosis.failure_chain) > 0

    # Check that summary includes failure chain
    summary = diagnosis.summary()
    if diagnosis.failure_chain:
        assert "Failure chain:" in summary


def test_multiple_failures_same_stage_aggregated() -> None:
    """Multiple failures in same stage should be aggregated properly."""
    prior_results = [
        AnalyzerResult(
            analyzer_name="ScopeViolationAnalyzer",
            status="fail",
            failure_type=FailureType.SCOPE_VIOLATION,
            stage=FailureStage.RETRIEVAL,
            evidence=["Off topic"],
            remediation="Fix",
        ),
        AnalyzerResult(
            analyzer_name="StaleRetrievalAnalyzer",
            status="fail",
            failure_type=FailureType.STALE_RETRIEVAL,
            stage=FailureStage.RETRIEVAL,
            evidence=["Stale"],
            remediation="Fix",
        ),
    ]

    analyzer = Layer6TaxonomyClassifier({"prior_results": prior_results})
    result = analyzer.analyze(
        run_with_chunks([chunk("chunk-1", "Text", 0.5)])
    )

    assert result.status == "fail"

    report = json.loads(result.evidence[0])

    # Should have multiple retrieval failures
    retrieval_failures = [
        f for f in report["stage_failures"] if f["stage"] == "RETRIEVAL"
    ]
    assert len(retrieval_failures) >= 2


def test_primary_stage_is_earliest_in_chain() -> None:
    """Primary stage should be the earliest stage in the failure chain."""
    # Create failures in reverse order: GENERATION, then RETRIEVAL
    prior_results = [
        AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
            evidence=["Not grounded"],
            remediation="Fix",
        ),
        AnalyzerResult(
            analyzer_name="ScopeViolationAnalyzer",
            status="fail",
            failure_type=FailureType.SCOPE_VIOLATION,
            stage=FailureStage.RETRIEVAL,
            evidence=["Off topic"],
            remediation="Fix",
        ),
    ]

    analyzer = Layer6TaxonomyClassifier({"prior_results": prior_results})
    result = analyzer.analyze(
        run_with_chunks([chunk("chunk-1", "Low", 0.50)])
    )

    assert result.status == "fail"

    report = json.loads(result.evidence[0])

    # Primary stage should be RETRIEVAL (earliest)
    # since failures cascade from retrieval → generation
    assert report["primary_stage"] == "RETRIEVAL"
