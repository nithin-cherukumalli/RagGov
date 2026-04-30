"""Tests for parser validation analyzer."""

from __future__ import annotations

from pathlib import Path

from raggov.analyzers.parsing.parser_validation import ParserValidationAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import FailureStage, FailureType
from raggov.models.run import RAGRun
from raggov.parser_validation.models import ChunkingStrategyType, default_chunking_profile
from raggov.parser_validation.profile import ParserValidationProfile


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def chunk(chunk_id: str, text: str, metadata: dict | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=f"doc-{chunk_id}",
        score=None,
        metadata=metadata or {},
    )


def run_with_chunks(
    chunks: list[RetrievedChunk],
    *,
    query: str = "What is the answer?",
    metadata: dict | None = None,
) -> RAGRun:
    return RAGRun(
        query=query,
        retrieved_chunks=chunks,
        final_answer="Answer.",
        metadata=metadata or {},
    )


def profile(strategy: ChunkingStrategyType = ChunkingStrategyType.UNKNOWN) -> ParserValidationProfile:
    return ParserValidationProfile(
        chunking_strategy=default_chunking_profile(strategy),
        infer_from_legacy=True,
    )


def test_clean_table_text_passes() -> None:
    analyzer = ParserValidationAnalyzer(profile=profile())
    result = analyzer.analyze(
        run_with_chunks(
            [
                chunk(
                    "chunk-1",
                    "| District | Vacancies | Category |\n| Warangal | 5 | Grade A |",
                    metadata={
                        "page_start": 1,
                        "source_table_ids": ["table-1"],
                        "section_path": ["Overview"],
                    },
                )
            ],
        )
    )

    assert result.status == "pass"
    assert result.failure_type is None


def test_table_like_numeric_text_without_structure_warns() -> None:
    analyzer = ParserValidationAnalyzer(profile=profile(ChunkingStrategyType.SENTENCE))
    result = analyzer.analyze(
        run_with_chunks(
            [
                chunk(
                    "chunk-1",
                    "A 10 20 30\nB 11 21 31\nC 12 22 32\nD 13 23 33",
                    metadata={
                        "page_start": 1,
                        "source_element_ids": ["e1"],
                    },
                )
            ],
            metadata={"chunking_strategy": "sentence"},
        )
    )

    assert result.status == "warn"
    assert result.failure_type == FailureType.TABLE_STRUCTURE_LOSS
    assert result.stage == FailureStage.PARSING
    assert any("[table_structure_validator]" in evidence for evidence in result.evidence)
    assert any("evidence_type=heuristic" in evidence for evidence in result.evidence)


def test_inline_hierarchy_markers_warn() -> None:
    analyzer = ParserValidationAnalyzer(profile=profile())
    result = analyzer.analyze(
        run_with_chunks(
            [
                chunk(
                    "chunk-1",
                    "Chapter 1 General Provisions Section 1 Scope Rule 1 Applicability Annexure A Format",
                    metadata={
                        "page_start": 1,
                        "source_element_ids": ["e1"],
                    },
                )
            ]
        )
    )

    assert result.status == "warn"
    assert result.failure_type == FailureType.HIERARCHY_FLATTENING
    assert result.stage == FailureStage.PARSING
    assert any("[hierarchy_validator]" in evidence for evidence in result.evidence)


def test_orphaned_fragments_warn() -> None:
    analyzer = ParserValidationAnalyzer(profile=profile(ChunkingStrategyType.SENTENCE))
    result = analyzer.analyze(
        run_with_chunks(
            [
                chunk(
                    "chunk-1",
                    "and the teacher shall be appointed by the committee",
                    metadata={
                        "page_start": 1,
                        "source_element_ids": ["e1"],
                        "section_path": ["Overview"],
                    },
                ),
                chunk(
                    "chunk-2",
                    "thereof subject to the conditions prescribed",
                    metadata={
                        "page_start": 1,
                        "source_element_ids": ["e2"],
                        "section_path": ["Overview"],
                    },
                ),
            ],
            metadata={"chunking_strategy": "sentence"},
        )
    )

    assert result.status == "warn"
    assert result.failure_type == FailureType.HIERARCHY_FLATTENING
    assert result.stage == FailureStage.PARSING
    assert any("[chunk_boundary_validator]" in evidence for evidence in result.evidence)


def test_well_structured_hierarchy_passes() -> None:
    analyzer = ParserValidationAnalyzer(profile=profile())
    result = analyzer.analyze(
        run_with_chunks(
            [
                chunk(
                    "chunk-1",
                    "1.\n  (a) text\n  (b) text\n2.\n  (a) text",
                    metadata={
                        "page_start": 1,
                        "source_element_ids": ["e1"],
                        "section_path": ["Overview"],
                    },
                )
            ]
        )
    )

    assert result.status == "pass"
    assert result.failure_type is None


def test_stale_retrieval_fixture_has_no_structural_false_positive_when_metadata_checks_disabled() -> None:
    analyzer = ParserValidationAnalyzer(
        config={
            "min_metadata_coverage": 0.0,
            "min_provenance_coverage": 0.0,
        },
        profile=profile(),
    )
    fixture_run = RAGRun.model_validate_json((FIXTURES / "stale_retrieval.json").read_text())

    result = analyzer.analyze(fixture_run)

    assert result.status == "pass"
    assert result.failure_type is None
