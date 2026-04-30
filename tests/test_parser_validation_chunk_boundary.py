from raggov.parser_validation.engine import ParserValidationEngine
from raggov.parser_validation.models import (
    ChunkIR,
    ChunkingStrategyType,
    ParserFailureType,
    ParserSeverity,
    ParserValidationConfig,
    default_chunking_profile,
)


def _config(strategy: ChunkingStrategyType) -> ParserValidationConfig:
    return ParserValidationConfig(chunking_profile=default_chunking_profile(strategy))


def test_chunk_boundary_validator_detects_explicit_metadata_flags():
    # HIERARCHICAL forbids cross-section chunks so crosses_section_boundary triggers.
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="Chapter 1 text followed by Chapter 2 text",
            page_start=1,
            source_element_ids=("e1",),
            section_path=("Chapter 1",),
            metadata={"crosses_section_boundary": True},
        ),
        ChunkIR(
            chunk_id="c2",
            text="Normal chunk",
            page_start=1,
            source_element_ids=("e2",),
            section_path=("Chapter 1",),
        ),
    ]

    findings = ParserValidationEngine(
        config=_config(ChunkingStrategyType.HIERARCHICAL)
    ).validate(None, chunks)

    assert any(
        finding.validator_name == "chunk_boundary_validator"
        and finding.failure_type == ParserFailureType.CHUNK_BOUNDARY_DAMAGE
        and finding.severity == ParserSeverity.WARN
        and not finding.is_heuristic
        for finding in findings
    )


def test_chunk_boundary_validator_collects_multiple_metadata_flags():
    # TABLE_AWARE enforces table structure so table-split flags count as damage.
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="row fragment",
            metadata={
                "ends_mid_table_row": True,
                "split_inside_table": True,
            },
        )
    ]

    findings = ParserValidationEngine(
        config=_config(ChunkingStrategyType.TABLE_AWARE)
    ).validate(None, chunks)

    boundary_findings = [
        finding for finding in findings
        if finding.validator_name == "chunk_boundary_validator"
    ]

    assert boundary_findings
    flags = boundary_findings[0].evidence[0].observed["flags"]
    assert flags["ends_mid_table_row"] == 1
    assert flags["split_inside_table"] == 1


def test_chunk_boundary_text_only_smoke_test_is_marked_heuristic():
    # SENTENCE forbids mid-sentence starts so the connector-start heuristic runs.
    chunks = [
        ChunkIR(chunk_id="c1", text="and continues from the previous sentence", page_start=1),
        ChunkIR(chunk_id="c2", text="which applies to the following clause", page_start=1),
        ChunkIR(chunk_id="c3", text="normal clean sentence starts here.", page_start=1),
    ]

    findings = ParserValidationEngine(
        config=_config(ChunkingStrategyType.SENTENCE)
    ).validate(None, chunks)

    assert any(
        finding.validator_name == "chunk_boundary_validator"
        and finding.failure_type == ParserFailureType.CHUNK_BOUNDARY_DAMAGE
        and finding.is_heuristic
        for finding in findings
    )


def test_sentence_strategy_restores_orphaned_fragment_warning():
    chunks = [
        ChunkIR(chunk_id="c1", text="and continues from the previous sentence", page_start=1),
        ChunkIR(chunk_id="c2", text="thereof subject to the conditions prescribed", page_start=1),
    ]

    findings = ParserValidationEngine(
        config=_config(ChunkingStrategyType.SENTENCE)
    ).validate(None, chunks)

    assert any(
        finding.validator_name == "chunk_boundary_validator"
        and finding.failure_type == ParserFailureType.CHUNK_BOUNDARY_DAMAGE
        and finding.severity == ParserSeverity.WARN
        for finding in findings
    )


def test_chunk_boundary_validator_does_not_flag_clean_chunks():
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="This is a complete sentence with clear boundaries.",
            page_start=1,
            source_element_ids=("e1",),
            section_path=("Overview",),
        ),
        ChunkIR(
            chunk_id="c2",
            text="This is another complete sentence.",
            page_start=1,
            source_element_ids=("e2",),
            section_path=("Overview",),
        ),
    ]

    findings = ParserValidationEngine().validate(None, chunks)

    assert not any(
        finding.validator_name == "chunk_boundary_validator"
        for finding in findings
    )


def test_chunk_boundary_validator_respects_disabled_text_fallback():
    chunks = [
        ChunkIR(chunk_id="c1", text="and continues from the previous sentence", page_start=1),
        ChunkIR(chunk_id="c2", text="which applies to the following clause", page_start=1),
    ]

    engine = ParserValidationEngine(
        config=ParserValidationConfig(enable_text_fallback_heuristics=False)
    )

    findings = engine.validate(None, chunks)

    assert not any(
        finding.validator_name == "chunk_boundary_validator"
        and finding.is_heuristic
        for finding in findings
    )
