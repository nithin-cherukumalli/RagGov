"""Tests for opt-in requirement-aware sufficiency mode."""

from __future__ import annotations

import pytest

from raggov.analyzers.sufficiency.sufficiency import SufficiencyAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import EvidenceRequirement
from raggov.models.run import RAGRun


class ChatClient:
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls = 0

    def chat(self, prompt: str) -> str:
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=f"doc-{chunk_id}",
        score=None,
    )


def run_with_context(query: str, chunks: list[RetrievedChunk]) -> RAGRun:
    return RAGRun(query=query, retrieved_chunks=chunks, final_answer="Answer.")


def test_requirement_extraction_returns_structured_output() -> None:
    client = ChatClient(
        """
        {
          "required_evidence": [
            {"description": "Rules for optional holidays", "type": "rule", "importance": "critical"},
            {"description": "Applicability to AP schools", "type": "scope", "importance": "supporting"}
          ]
        }
        """
    )
    analyzer = SufficiencyAnalyzer({"llm_client": client})

    requirements = analyzer._extract_evidence_requirements(
        "What are the rules for optional holidays in AP schools?"
    )

    assert len(requirements) == 2
    assert {item.requirement_id for item in requirements} == {"req_001", "req_002"}
    assert all(item.verifier == "llm_judge" for item in requirements)


def test_requirement_extraction_handles_llm_failure(caplog: pytest.LogCaptureFixture) -> None:
    analyzer = SufficiencyAnalyzer({"llm_client": ChatClient(RuntimeError("down"))})

    requirements = analyzer._extract_evidence_requirements("What are the rules?")

    assert requirements == []
    assert "Requirement extraction failed" in caplog.text


def test_requirement_extraction_handles_bad_json() -> None:
    analyzer = SufficiencyAnalyzer({"llm_client": ChatClient("not json")})

    requirements = analyzer._extract_evidence_requirements("What are the rules?")

    assert requirements == []


def test_lexical_coverage_detects_covered_requirement() -> None:
    requirement = EvidenceRequirement(
        requirement_id="req_001",
        description="school education department academic calendar 2025",
        requirement_type="scope",
    )

    coverage = SufficiencyAnalyzer()._check_requirement_coverage_lexical(
        [requirement],
        [chunk("c1", "The School Education Department released the academic calendar for 2025.")],
    )

    assert coverage[0].status == "covered"


def test_lexical_coverage_detects_missing_requirement() -> None:
    requirement = EvidenceRequirement(
        requirement_id="req_001",
        description="school education department academic calendar 2025",
        requirement_type="scope",
    )

    coverage = SufficiencyAnalyzer()._check_requirement_coverage_lexical(
        [requirement],
        [chunk("c1", "General Administration Department holiday order for employees")],
    )

    assert coverage[0].status == "missing"


def test_lexical_coverage_detects_partial_requirement() -> None:
    requirement = EvidenceRequirement(
        requirement_id="req_001",
        description="school education department academic calendar 2025",
        requirement_type="scope",
    )

    coverage = SufficiencyAnalyzer()._check_requirement_coverage_lexical(
        [requirement],
        [chunk("c1", "The academic calendar for 2025 lists important dates")],
    )

    assert coverage[0].status == "partial"


def test_requirement_aware_mode_flags_critical_missing() -> None:
    client = ChatClient(
        '{"required_evidence": [{"description": "school education department academic calendar 2025", "type": "scope", "importance": "critical"}]}'
    )
    run = run_with_context(
        "AP school optional holidays 2025 academic calendar",
        [chunk("c1", "teacher transfers seniority roster")],
    )

    result = SufficiencyAnalyzer(
        {"sufficiency_mode": "requirement_aware", "llm_client": client}
    ).analyze(run)

    assert result.sufficiency_result is not None
    assert result.sufficiency_result.sufficiency_label == "insufficient"
    assert result.sufficiency_result.should_abstain is True


def test_requirement_aware_fallback_to_term_coverage() -> None:
    run = run_with_context(
        "AP school optional holidays 2025 academic calendar",
        [chunk("c1", "teacher transfers seniority roster")],
    )

    result = SufficiencyAnalyzer(
        {"sufficiency_mode": "requirement_aware", "llm_client": ChatClient(RuntimeError("down"))}
    ).analyze(run)

    assert result.sufficiency_result is not None
    assert result.sufficiency_result.fallback_used is True
    assert "term_coverage_heuristic_v0" in result.sufficiency_result.method
    assert any("Requirement extraction failed" in item for item in result.sufficiency_result.limitations)


def test_default_mode_is_requirement_aware_when_llm_client_available() -> None:
    client = ChatClient(
        '{"required_evidence": [{"description": "GAD optional holidays 2025", "type": "rule", "importance": "critical"}]}'
    )
    run = run_with_context("GAD optional holidays 2025", [chunk("c1", "GAD optional holidays 2025")])

    result = SufficiencyAnalyzer({"llm_client": client}).analyze(run)

    assert result.sufficiency_result is not None
    assert result.sufficiency_result.method == "requirement_extraction_llm_v0"
    assert client.calls == 1


def test_explicit_term_coverage_mode_does_not_call_llm() -> None:
    client = ChatClient(
        '{"required_evidence": [{"description": "rules", "type": "rule", "importance": "critical"}]}'
    )
    run = run_with_context("GAD optional holidays 2025", [chunk("c1", "GAD optional holidays 2025")])

    result = SufficiencyAnalyzer(
        {"sufficiency_mode": "term_coverage", "llm_client": client}
    ).analyze(run)

    assert result.sufficiency_result is not None
    assert result.sufficiency_result.method == "term_coverage_heuristic_v0"
    assert client.calls == 0


def test_mode_switching_respected() -> None:
    client = ChatClient(
        '{"required_evidence": [{"description": "GAD optional holidays 2025", "type": "rule", "importance": "critical"}]}'
    )
    run = run_with_context("GAD optional holidays 2025", [chunk("c1", "GAD optional holidays 2025")])

    requirement_result = SufficiencyAnalyzer(
        {"sufficiency_mode": "requirement_aware", "llm_client": client}
    ).analyze(run)
    term_result = SufficiencyAnalyzer(
        {"sufficiency_mode": "term_coverage", "llm_client": client}
    ).analyze(run)

    assert requirement_result.sufficiency_result is not None
    assert requirement_result.sufficiency_result.method == "requirement_extraction_llm_v0"
    assert term_result.sufficiency_result is not None
    assert term_result.sufficiency_result.method == "term_coverage_heuristic_v0"
