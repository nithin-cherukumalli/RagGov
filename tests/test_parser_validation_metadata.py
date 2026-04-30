from raggov.parser_validation.engine import ParserValidationEngine
from raggov.parser_validation.models import (
    ChunkIR,
    ChunkingStrategyType,
    ElementIR,
    ParsedDocumentIR,
    ParserFailureType,
    ParserSeverity,
    ParserValidationConfig,
    TableIR,
    default_chunking_profile,
)


def _config(strategy: ChunkingStrategyType) -> ParserValidationConfig:
    return ParserValidationConfig(chunking_profile=default_chunking_profile(strategy))


def test_metadata_validator_detects_missing_page_metadata():
    # HIERARCHICAL requires page metadata, so missing page_start triggers WARN.
    chunks = [
        ChunkIR(chunk_id="c1", text="hello"),
        ChunkIR(chunk_id="c2", text="world"),
    ]

    findings = ParserValidationEngine(
        config=_config(ChunkingStrategyType.HIERARCHICAL)
    ).validate(None, chunks)

    assert any(
        finding.failure_type == ParserFailureType.METADATA_LOSS
        and finding.severity == ParserSeverity.WARN
        and finding.validator_name == "metadata_validator"
        for finding in findings
    )


def test_metadata_validator_passes_when_page_and_provenance_coverage_are_good():
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="hello",
            page_start=1,
            page_end=1,
            source_element_ids=("e1",),
        ),
        ChunkIR(
            chunk_id="c2",
            text="world",
            page_start=2,
            page_end=2,
            source_element_ids=("e2",),
        ),
    ]

    findings = ParserValidationEngine().validate(None, chunks)

    assert not any(
        finding.validator_name == "metadata_validator"
        for finding in findings
    )


def test_metadata_validator_detects_missing_provenance():
    # HIERARCHICAL requires provenance, so chunks with page but no element/table IDs trigger WARN.
    chunks = [
        ChunkIR(chunk_id="c1", text="hello", page_start=1),
        ChunkIR(chunk_id="c2", text="world", page_start=2),
    ]

    findings = ParserValidationEngine(
        config=_config(ChunkingStrategyType.HIERARCHICAL)
    ).validate(None, chunks)

    assert any(
        finding.failure_type == ParserFailureType.PROVENANCE_MISSING
        and finding.severity == ParserSeverity.WARN
        and finding.validator_name == "metadata_validator"
        for finding in findings
    )


def test_metadata_validator_detects_missing_element_ids_when_parsed_doc_has_elements():
    # HIERARCHICAL preserves source elements, so missing source_element_ids triggers FAIL.
    parsed_doc = ParsedDocumentIR(
        document_id="doc1",
        elements=(
            ElementIR(element_id="e1", element_type="Title", text="Chapter 1"),
            ElementIR(element_id="e2", element_type="NarrativeText", text="Body"),
        ),
    )

    chunks = [
        ChunkIR(chunk_id="c1", text="Chapter 1 Body", page_start=1),
    ]

    findings = ParserValidationEngine(
        config=_config(ChunkingStrategyType.HIERARCHICAL)
    ).validate(parsed_doc, chunks)

    assert any(
        finding.failure_type == ParserFailureType.PROVENANCE_MISSING
        and finding.severity == ParserSeverity.FAIL
        and "source_element_ids" in str(finding.evidence[0].expected)
        for finding in findings
    )


def test_metadata_validator_detects_missing_table_ids_when_parsed_doc_has_tables():
    # TABLE_AWARE preserves table structure, so missing source_table_ids triggers FAIL.
    parsed_doc = ParsedDocumentIR(
        document_id="doc1",
        tables=(TableIR(table_id="t1", n_rows=2, n_cols=2),),
    )

    chunks = [
        ChunkIR(chunk_id="c1", text="A B 1 2", page_start=1),
    ]

    findings = ParserValidationEngine(
        config=_config(ChunkingStrategyType.TABLE_AWARE)
    ).validate(parsed_doc, chunks)

    assert any(
        finding.failure_type == ParserFailureType.PROVENANCE_MISSING
        and finding.severity == ParserSeverity.FAIL
        and "source_table_ids" in str(finding.evidence[0].expected)
        for finding in findings
    )
