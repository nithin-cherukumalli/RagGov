from __future__ import annotations

from raggov.analyzers.sufficiency.sufficiency import SufficiencyAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import FailureType
from raggov.models.run import RAGRun


def chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id="c1", text=text, source_doc_id="doc-1", score=None)


def analyze(query: str, context: str, *, min_coverage_ratio: float = 0.8):
    run = RAGRun(query=query, retrieved_chunks=[chunk(context)], final_answer="Answer.")
    return SufficiencyAnalyzer({"min_coverage_ratio": min_coverage_ratio}).analyze(run)


def test_healthcare_missing_dosage_condition() -> None:
    result = analyze(
        "What dose applies if renal impairment is present?",
        "The guideline gives the standard adult dose as 5 mg daily.",
        min_coverage_ratio=0.0,
    )

    assert result.status == "fail"
    assert result.failure_type == FailureType.INSUFFICIENT_CONTEXT
    assert "[sufficiency:partial_requirement_coverage]" in result.evidence[0]


def test_software_missing_version_constraint() -> None:
    result = analyze(
        "How do retries work in SDK version 2.4?",
        "The SDK retries timeout errors by default.",
    )

    assert result.status == "fail"
    assert result.failure_type == FailureType.INSUFFICIENT_CONTEXT
    assert result.sufficiency_result is not None
    assert result.sufficiency_result.required_evidence[0].requirement_type == "required_condition_or_scope"


def test_finance_missing_effective_date() -> None:
    result = analyze(
        "Which advisory fee applies after the 2026 effective date?",
        "The advisory fee is 1.2% for managed accounts.",
    )

    assert result.status == "fail"
    assert result.failure_type == FailureType.INSUFFICIENT_CONTEXT
    assert "[sufficiency:missing_temporal_or_freshness_requirement]" in result.evidence[0]


def test_product_manual_missing_exception() -> None:
    result = analyze(
        "Can every filter be washed?",
        "The washable pre-filter can be rinsed monthly.",
        min_coverage_ratio=0.0,
    )

    assert result.status == "fail"
    assert result.failure_type == FailureType.INSUFFICIENT_CONTEXT
    assert "[sufficiency:missing_exception]" in result.evidence[0]


def test_scientific_answer_missing_comparison_baseline() -> None:
    result = analyze(
        "How much did treatment improve outcomes compared with placebo?",
        "Treatment improved outcomes by 12% in the study cohort.",
    )

    assert result.status == "fail"
    assert result.failure_type == FailureType.INSUFFICIENT_CONTEXT
