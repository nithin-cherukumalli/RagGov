import pytest

from raggov.analyzers.parsing.parser_validation import ParserValidationAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import FailureStage, FailureType
from raggov.models.run import RAGRun
from raggov.parser_validation.models import (
    ChunkingStrategyProfile,
    ChunkingStrategyType,
    ParsedDocumentIR,
    TableIR,
)
from raggov.parser_validation.profile import (
    CanonicalMetadataMapping,
    MetadataFieldMapping,
    ParserValidationProfile,
)


def _chunk(chunk_id: str, text: str, metadata: dict | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=f"doc-{chunk_id}",
        score=None,
        metadata=metadata or {},
    )


def _run(chunks: list[RetrievedChunk], metadata: dict | None = None) -> RAGRun:
    return RAGRun(
        query="What is the answer?",
        retrieved_chunks=chunks,
        final_answer="Answer.",
        metadata=metadata or {},
    )


def test_parser_validation_analyzer_requires_profile_when_chunks_exist():
    analyzer = ParserValidationAnalyzer()
    run = _run([_chunk("c1", "hello")])

    with pytest.raises(ValueError, match="ParserValidationAnalyzer requires ParserValidationProfile"):
        analyzer.analyze(run)


def test_parser_validation_analyzer_normalizes_profile_mapped_metadata():
    profile = ParserValidationProfile(
        chunking_strategy=ChunkingStrategyProfile(
            strategy_type=ChunkingStrategyType.TABLE_AWARE,
            preserves_table_structure=True,
            preserves_table_headers=True,
            preserves_source_elements=True,
            allows_table_flattening=False,
            requires_page_metadata=True,
            requires_provenance=True,
        ),
        metadata_mapping=CanonicalMetadataMapping(
            page_start=MetadataFieldMapping(aliases=("metadata.pg",), first_transform="int"),
            source_table_ids=MetadataFieldMapping(aliases=("metadata.tbl",)),
            source_element_ids=MetadataFieldMapping(aliases=("metadata.el",)),
            section_path=MetadataFieldMapping(aliases=("metadata.path",)),
        ),
    )
    parsed_doc = ParsedDocumentIR(
        document_id="doc1",
        tables=(TableIR(table_id="t1", headers=("Name", "Score"), n_rows=3, n_cols=2),),
    )
    run = _run(
        [
            _chunk(
                "c1",
                "Name Score A 10 B 20",
                metadata={"pg": "2", "tbl": "t1", "el": ["e1"], "path": ["Results"]},
            )
        ],
        metadata={"parsed_document_ir": parsed_doc},
    )

    result = ParserValidationAnalyzer(profile=profile).analyze(run)

    assert result.status == "fail"
    assert result.failure_type == FailureType.TABLE_STRUCTURE_LOSS
    assert result.stage == FailureStage.PARSING
    assert any("chunking_strategy=table_aware" in evidence for evidence in result.evidence)


def test_parser_validation_analyzer_infer_from_legacy_mode_uses_existing_adapter_aliases():
    profile = ParserValidationProfile(
        chunking_strategy=ChunkingStrategyProfile(
            strategy_type=ChunkingStrategyType.SENTENCE,
            allows_mid_sentence_start=False,
        ),
        infer_from_legacy=True,
    )
    run = _run(
        [
            _chunk("c1", "and continues from the previous sentence", metadata={"page_start": 1}),
            _chunk("c2", "which applies to the following clause", metadata={"page_start": 1}),
        ]
    )

    result = ParserValidationAnalyzer(profile=profile).analyze(run)

    assert result.status == "warn"
    assert result.failure_type == FailureType.HIERARCHY_FLATTENING
    assert any("[chunk_boundary_validator]" in evidence for evidence in result.evidence)
