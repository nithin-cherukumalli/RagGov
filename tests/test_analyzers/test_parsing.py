"""Tests for parser validation analyzer."""

from __future__ import annotations

from pathlib import Path

from raggov.analyzers.parsing.parser_validation import ParserValidationAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import FailureStage, FailureType
from raggov.models.run import RAGRun


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=f"doc-{chunk_id}",
        score=None,
    )


def run_with_chunks(
    chunks: list[RetrievedChunk], *, query: str = "What is the answer?"
) -> RAGRun:
    return RAGRun(
        query=query,
        retrieved_chunks=chunks,
        final_answer="Answer.",
    )


def test_clean_table_text_passes() -> None:
    analyzer = ParserValidationAnalyzer()
    result = analyzer.analyze(
        run_with_chunks(
            [
                chunk(
                    "chunk-1",
                    "| District | Vacancies | Category |\n| Warangal | 5 | Grade A |",
                )
            ]
        )
    )

    assert result.status == "pass"
    assert result.failure_type is None


def test_table_keywords_without_delimiters_fail() -> None:
    analyzer = ParserValidationAnalyzer()
    result = analyzer.analyze(
        run_with_chunks(
            [
                chunk(
                    "chunk-1",
                    "District Vacancies Category Warangal 5 Grade A Khammam 3 Grade B",
                )
            ]
        )
    )

    assert result.status == "fail"
    assert result.failure_type == FailureType.TABLE_STRUCTURE_LOSS
    assert result.stage == FailureStage.PARSING
    assert "structural separators absent" in result.evidence[0]


def test_numbered_items_merged_on_one_line_fail() -> None:
    analyzer = ParserValidationAnalyzer()
    result = analyzer.analyze(
        run_with_chunks(
            [
                chunk(
                    "chunk-1",
                    "Rule 5 (1) Definition (2) Exception (3) Application all on one line",
                )
            ]
        )
    )

    assert result.status == "fail"
    assert result.failure_type == FailureType.HIERARCHY_FLATTENING
    assert result.stage == FailureStage.PARSING


def test_orphaned_fragments_warn() -> None:
    analyzer = ParserValidationAnalyzer()
    result = analyzer.analyze(
        run_with_chunks(
            [
                chunk("chunk-1", "and the teacher shall be appointed by the committee"),
                chunk("chunk-2", "thereof subject to the conditions prescribed"),
            ]
        )
    )

    assert result.status == "warn"
    assert result.failure_type == FailureType.HIERARCHY_FLATTENING
    assert result.stage == FailureStage.PARSING


def test_well_structured_hierarchy_passes() -> None:
    analyzer = ParserValidationAnalyzer()
    result = analyzer.analyze(
        run_with_chunks(
            [
                chunk(
                    "chunk-1",
                    "1.\n  (a) text\n  (b) text\n2.\n  (a) text",
                )
            ]
        )
    )

    assert result.status == "pass"
    assert result.failure_type is None


def test_stale_retrieval_fixture_has_no_false_positive() -> None:
    analyzer = ParserValidationAnalyzer()
    fixture_run = RAGRun.model_validate_json((FIXTURES / "stale_retrieval.json").read_text())

    result = analyzer.analyze(fixture_run)

    assert result.status == "pass"
    assert result.failure_type is None
