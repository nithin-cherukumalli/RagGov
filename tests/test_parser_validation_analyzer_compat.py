from raggov.analyzers.parsing.parser_validation import ParserValidationAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.run import RAGRun
from raggov.parser_validation.models import (
    ChunkingStrategyType,
    ParsedDocumentIR,
    TableIR,
    default_chunking_profile,
)
from raggov.parser_validation.profile import ParserValidationProfile


def chunk(chunk_id: str, text: str, metadata: dict | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=f"doc-{chunk_id}",
        score=None,
        metadata=metadata or {},
    )


def run_with_chunks(chunks: list[RetrievedChunk], metadata: dict | None = None) -> RAGRun:
    return RAGRun(
        query="What is the answer?",
        retrieved_chunks=chunks,
        final_answer="Answer.",
        metadata=metadata or {},
    )


def profile(strategy: ChunkingStrategyType = ChunkingStrategyType.UNKNOWN) -> ParserValidationProfile:
    return ParserValidationProfile(
        chunking_strategy=default_chunking_profile(strategy),
        infer_from_legacy=True,
    )


def test_parser_validation_analyzer_skips_without_chunks():
    run = run_with_chunks([])

    result = ParserValidationAnalyzer().analyze(run)

    assert result.status == "skip"
    assert result.evidence == ["no retrieved chunks available"]


def test_parser_validation_analyzer_returns_parsing_stage_for_table_issue():
    # Numeric multi-line chunk triggers table structure WARN heuristic under UNKNOWN profile.
    run = run_with_chunks(
        [
            chunk(
                "c1",
                "A 10 20 30\nB 11 21 31\nC 12 22 32\nD 13 23 33",
            )
        ]
    )

    result = ParserValidationAnalyzer(profile=profile()).analyze(run)

    assert result.stage == FailureStage.PARSING
    assert result.failure_type in {
        FailureType.TABLE_STRUCTURE_LOSS,
        FailureType.METADATA_LOSS,
        FailureType.HIERARCHY_FLATTENING,
    }
    assert result.status in {"warn", "fail"}
    assert any("evidence_type=" in evidence for evidence in result.evidence)


def test_parser_validation_analyzer_does_not_crash_without_chunk_metadata():
    run = run_with_chunks([chunk("c1", "Plain retrieved text.")])

    result = ParserValidationAnalyzer(profile=profile()).analyze(run)

    assert isinstance(result, AnalyzerResult)


def test_parser_validation_analyzer_evidence_contains_validator_name():
    # Numeric table-like text triggers table_structure_validator WARN under UNKNOWN profile.
    run = run_with_chunks(
        [
            chunk("c1", "A 10 20 30\nB 11 21 31\nC 12 22 32\nD 13 23 33"),
        ]
    )

    result = ParserValidationAnalyzer(profile=profile()).analyze(run)

    assert any(
        validator_name in evidence
        for evidence in result.evidence
        for validator_name in (
            "[table_structure_validator]",
            "[metadata_validator]",
            "[hierarchy_validator]",
            "[chunk_boundary_validator]",
        )
    )


def test_parser_validation_analyzer_config_can_disable_text_fallback():
    run = run_with_chunks(
        [
            chunk("c1", "and continues from the previous sentence", metadata={"page_start": 1}),
            chunk("c2", "which applies to the following clause", metadata={"page_start": 1}),
        ]
    )

    analyzer = ParserValidationAnalyzer(
        config={"enable_text_fallback_heuristics": False},
        profile=profile(ChunkingStrategyType.SENTENCE),
    )
    result = analyzer.analyze(run)

    assert not any("evidence_type=heuristic" in evidence for evidence in result.evidence)


def test_table_failure_outranks_metadata_warning():
    """Table structure FAIL must be the primary finding even when MetadataValidator also warns."""
    parsed_doc = ParsedDocumentIR(
        document_id="doc1",
        tables=(
            TableIR(
                table_id="t1",
                n_rows=3,
                n_cols=2,
                headers=("District", "Vacancies"),
            ),
        ),
    )

    # Chunk references the table (provenance kept) but text is flattened (no separators).
    # page_start is absent so MetadataValidator also emits a WARN under TABLE_AWARE profile.
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            text="District Vacancies Warangal 5 Khammam 3",
            source_doc_id="doc1",
            score=None,
            metadata={"source_table_ids": ["t1"]},
        )
    ]

    run = RAGRun(
        query="How many vacancies?",
        retrieved_chunks=chunks,
        final_answer="Some answer.",
        metadata={
            "parsed_document_ir": parsed_doc,
            "parser_validation_profile": profile(ChunkingStrategyType.TABLE_AWARE),
        },
    )

    result = ParserValidationAnalyzer().analyze(run)

    assert result.failure_type == FailureType.TABLE_STRUCTURE_LOSS
    assert result.status == "fail"
    assert result.stage == FailureStage.PARSING
    assert any("[table_structure_validator]" in evidence for evidence in result.evidence)
