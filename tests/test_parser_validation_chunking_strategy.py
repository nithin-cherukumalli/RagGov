"""Tests for chunking-strategy-aware parser validation."""

from dataclasses import dataclass

from raggov.parser_validation import (
    ChunkingStrategyType,
    default_chunking_profile,
)
from raggov.parser_validation.adapters import (
    chunking_profile_from_run_metadata,
    chunking_strategy_source_from_run_metadata,
    normalize_chunking_strategy,
)
from raggov.parser_validation.engine import ParserValidationEngine
from raggov.parser_validation.models import (
    ChunkIR,
    ChunkingStrategyProfile,
    ElementIR,
    ParsedDocumentIR,
    ParserFailureType,
    ParserSeverity,
    ParserValidationConfig,
    TableIR,
)


@dataclass
class FakeChunk:
    chunk_id: str
    text: str
    metadata: dict


@dataclass
class FakeRun:
    retrieved_chunks: list
    metadata: dict | None = None


def _engine(strategy: ChunkingStrategyType) -> ParserValidationEngine:
    return ParserValidationEngine(
        config=ParserValidationConfig(
            chunking_profile=default_chunking_profile(strategy)
        )
    )


def _engine_for_profile(profile: ChunkingStrategyProfile) -> ParserValidationEngine:
    return ParserValidationEngine(
        config=ParserValidationConfig(chunking_profile=profile)
    )


# ---------------------------------------------------------------------------
# Strategy alias normalization
# ---------------------------------------------------------------------------


def test_strategy_alias_normalization():
    assert normalize_chunking_strategy("fixed") == ChunkingStrategyType.FIXED_TOKEN
    assert normalize_chunking_strategy("recursive_character_text_splitter") == ChunkingStrategyType.RECURSIVE_TEXT
    assert normalize_chunking_strategy("markdown_headers") == ChunkingStrategyType.MARKDOWN_HEADER
    assert normalize_chunking_strategy("by_title") == ChunkingStrategyType.HIERARCHICAL
    assert normalize_chunking_strategy("table_preserving") == ChunkingStrategyType.TABLE_AWARE
    assert normalize_chunking_strategy("compressed") == ChunkingStrategyType.SUMMARY
    assert normalize_chunking_strategy("unknown-weird") == ChunkingStrategyType.UNKNOWN
    assert normalize_chunking_strategy(None) == ChunkingStrategyType.UNKNOWN
    assert normalize_chunking_strategy(ChunkingStrategyType.SENTENCE) == ChunkingStrategyType.SENTENCE


# ---------------------------------------------------------------------------
# Strategy resolution from run vs chunk metadata
# ---------------------------------------------------------------------------


def test_run_level_strategy_overrides_chunk_level_strategy():
    run = FakeRun(
        retrieved_chunks=[
            FakeChunk(chunk_id="c1", text="hello", metadata={"chunk_strategy": "table_aware"}),
        ],
        metadata={"chunking_strategy": "fixed_token"},
    )

    profile = chunking_profile_from_run_metadata(run)
    source = chunking_strategy_source_from_run_metadata(run)

    assert profile.strategy_type == ChunkingStrategyType.FIXED_TOKEN
    assert source == "run_metadata"


def test_chunk_level_strategy_fallback():
    run = FakeRun(
        retrieved_chunks=[
            FakeChunk(chunk_id="c1", text="hello", metadata={"chunk_strategy": "markdown_header"}),
        ],
        metadata={},
    )

    profile = chunking_profile_from_run_metadata(run)
    source = chunking_strategy_source_from_run_metadata(run)

    assert profile.strategy_type == ChunkingStrategyType.MARKDOWN_HEADER
    assert source == "chunk_metadata"


def test_unknown_strategy_returned_when_no_metadata():
    run = FakeRun(retrieved_chunks=[], metadata={})

    profile = chunking_profile_from_run_metadata(run)
    source = chunking_strategy_source_from_run_metadata(run)

    assert profile.strategy_type == ChunkingStrategyType.UNKNOWN
    assert source == "unknown"


# ---------------------------------------------------------------------------
# UNKNOWN profile is low-noise
# ---------------------------------------------------------------------------


def test_unknown_strategy_profile_is_low_noise():
    # Chunks lacking page metadata and provenance should produce no metadata_validator
    # findings under the UNKNOWN profile.
    chunks = [
        ChunkIR(chunk_id="c1", text="hello"),
        ChunkIR(chunk_id="c2", text="world"),
    ]

    findings = ParserValidationEngine().validate(None, chunks)

    assert not any(f.validator_name == "metadata_validator" for f in findings)


# ---------------------------------------------------------------------------
# FIXED_TOKEN — no mid-sentence false positives
# ---------------------------------------------------------------------------


def test_fixed_token_does_not_flag_connector_start_chunks():
    chunks = [
        ChunkIR(chunk_id="c1", text="and continues from the previous sentence", page_start=1),
        ChunkIR(chunk_id="c2", text="which applies to the following clause", page_start=1),
    ]

    findings = _engine(ChunkingStrategyType.FIXED_TOKEN).validate(None, chunks)

    assert not any(f.validator_name == "chunk_boundary_validator" for f in findings)


# ---------------------------------------------------------------------------
# SENTENCE — catches connector-start boundary issues
# ---------------------------------------------------------------------------


def test_sentence_strategy_flags_many_connector_start_chunks():
    chunks = [
        ChunkIR(chunk_id="c1", text="and continues from the previous sentence", page_start=1),
        ChunkIR(chunk_id="c2", text="which applies to the following clause", page_start=1),
        ChunkIR(chunk_id="c3", text="This is a normal sentence.", page_start=1),
    ]

    findings = _engine(ChunkingStrategyType.SENTENCE).validate(None, chunks)

    assert any(
        f.validator_name == "chunk_boundary_validator"
        and f.failure_type == ParserFailureType.CHUNK_BOUNDARY_DAMAGE
        and f.is_heuristic
        for f in findings
    )


# ---------------------------------------------------------------------------
# HIERARCHICAL — enforces section_path
# ---------------------------------------------------------------------------


def test_hierarchical_strategy_warns_when_section_path_missing():
    parsed_doc = ParsedDocumentIR(
        document_id="doc1",
        elements=(
            ElementIR(
                element_id="e1",
                element_type="Title",
                text="Chapter 1",
                section_path=("Chapter 1",),
            ),
        ),
    )

    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="Chapter 1 Body",
            source_element_ids=("e1",),
            page_start=1,
            section_path=(),
        )
    ]

    findings = _engine(ChunkingStrategyType.HIERARCHICAL).validate(parsed_doc, chunks)

    assert any(
        f.validator_name == "hierarchy_validator"
        and f.failure_type == ParserFailureType.HIERARCHY_FLATTENING
        for f in findings
    )


def test_fixed_token_does_not_warn_solely_for_missing_section_path():
    parsed_doc = ParsedDocumentIR(
        document_id="doc1",
        elements=(
            ElementIR(
                element_id="e1",
                element_type="Title",
                text="Chapter 1",
                section_path=("Chapter 1",),
            ),
        ),
    )

    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="Chapter 1 Body",
            source_element_ids=("e1",),
            page_start=1,
            section_path=(),
        )
    ]

    findings = _engine(ChunkingStrategyType.FIXED_TOKEN).validate(parsed_doc, chunks)

    assert not any(f.validator_name == "hierarchy_validator" for f in findings)


# ---------------------------------------------------------------------------
# TABLE_AWARE — enforces table structure
# ---------------------------------------------------------------------------


def test_table_aware_fails_flattened_table():
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

    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="District Vacancies Warangal 5 Khammam 3",
            source_table_ids=("t1",),
            page_start=1,
        )
    ]

    findings = _engine(ChunkingStrategyType.TABLE_AWARE).validate(parsed_doc, chunks)

    assert any(
        f.validator_name == "table_structure_validator"
        and f.failure_type == ParserFailureType.TABLE_STRUCTURE_LOSS
        and f.severity == ParserSeverity.FAIL
        for f in findings
    )


def test_profile_table_structure_preserve_field_enforces_table_contract():
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
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="District Vacancies Warangal 5 Khammam 3",
            source_table_ids=("t1",),
            page_start=1,
        )
    ]

    findings = _engine_for_profile(
        ChunkingStrategyProfile(table_structure="preserve")
    ).validate(parsed_doc, chunks)

    assert any(
        f.validator_name == "table_structure_validator"
        and f.failure_type == ParserFailureType.TABLE_STRUCTURE_LOSS
        and f.severity == ParserSeverity.FAIL
        for f in findings
    )


# ---------------------------------------------------------------------------
# SUMMARY — suppresses table structure checks
# ---------------------------------------------------------------------------


def test_summary_suppresses_table_structure_checks():
    parsed_doc = ParsedDocumentIR(
        document_id="doc1",
        tables=(
            TableIR(table_id="t1", n_rows=3, n_cols=2, headers=("District", "Vacancies")),
        ),
    )

    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="District Vacancies Warangal 5 Khammam 3",
            source_table_ids=("t1",),
            page_start=1,
            source_element_ids=("e1",),
        )
    ]

    findings = _engine(ChunkingStrategyType.SUMMARY).validate(parsed_doc, chunks)

    assert not any(f.validator_name == "table_structure_validator" for f in findings)


def test_profile_table_structure_summary_field_suppresses_table_checks():
    parsed_doc = ParsedDocumentIR(
        document_id="doc1",
        tables=(
            TableIR(table_id="t1", n_rows=3, n_cols=2, headers=("District", "Vacancies")),
        ),
    )
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="District Vacancies Warangal 5 Khammam 3",
            source_table_ids=("t1",),
            page_start=1,
            source_element_ids=("e1",),
        )
    ]

    findings = _engine_for_profile(
        ChunkingStrategyProfile(table_structure="summary")
    ).validate(parsed_doc, chunks)

    assert not any(f.validator_name == "table_structure_validator" for f in findings)


# ---------------------------------------------------------------------------
# PARENT_CHILD — requires parent metadata
# ---------------------------------------------------------------------------


def test_parent_child_requires_parent_metadata():
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="Section content",
            source_element_ids=("e1",),
            page_start=1,
        ),
        ChunkIR(
            chunk_id="c2",
            text="More content",
            source_element_ids=("e2",),
            page_start=2,
        ),
    ]

    findings = _engine(ChunkingStrategyType.PARENT_CHILD).validate(None, chunks)

    assert any(
        f.validator_name == "metadata_validator"
        and f.failure_type == ParserFailureType.PROVENANCE_MISSING
        and "parent" in f.evidence[0].message.lower()
        for f in findings
    )


def test_parent_child_passes_with_parent_metadata():
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="Section content",
            source_element_ids=("e1",),
            page_start=1,
            metadata={"parent_node_id": "doc-root"},
        ),
        ChunkIR(
            chunk_id="c2",
            text="More content",
            source_element_ids=("e2",),
            page_start=2,
            metadata={"parent_node_id": "doc-root"},
        ),
    ]

    findings = _engine(ChunkingStrategyType.PARENT_CHILD).validate(None, chunks)

    assert not any(
        f.validator_name == "metadata_validator"
        and "parent" in (f.evidence[0].message if f.evidence else "").lower()
        for f in findings
    )


# ---------------------------------------------------------------------------
# MARKDOWN_HEADER — requires section_path
# ---------------------------------------------------------------------------


def test_markdown_header_requires_section_path():
    parsed_doc = ParsedDocumentIR(
        document_id="doc1",
        elements=(
            ElementIR(
                element_id="e1",
                element_type="heading",
                text="Overview",
                section_path=("Overview",),
            ),
        ),
    )

    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="Overview content",
            source_element_ids=("e1",),
            page_start=1,
            section_path=(),
        )
    ]

    findings = _engine(ChunkingStrategyType.MARKDOWN_HEADER).validate(parsed_doc, chunks)

    assert any(
        f.validator_name == "hierarchy_validator"
        and f.failure_type == ParserFailureType.HIERARCHY_FLATTENING
        for f in findings
    )


def test_profile_hierarchy_mode_preserve_field_requires_section_path():
    parsed_doc = ParsedDocumentIR(
        document_id="doc1",
        elements=(
            ElementIR(
                element_id="e1",
                element_type="heading",
                text="Overview",
                section_path=("Overview",),
            ),
        ),
    )
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="Overview content",
            source_element_ids=("e1",),
            page_start=1,
            section_path=(),
        )
    ]

    findings = _engine_for_profile(
        ChunkingStrategyProfile(hierarchy_mode="preserve")
    ).validate(parsed_doc, chunks)

    assert any(
        f.validator_name == "hierarchy_validator"
        and f.failure_type == ParserFailureType.HIERARCHY_FLATTENING
        for f in findings
    )


def test_profile_sentence_boundaries_preserve_field_flags_connector_starts():
    chunks = [
        ChunkIR(chunk_id="c1", text="and continues from the previous sentence", page_start=1),
        ChunkIR(chunk_id="c2", text="which applies to the following clause", page_start=1),
        ChunkIR(chunk_id="c3", text="This is a normal sentence.", page_start=1),
    ]

    findings = _engine_for_profile(
        ChunkingStrategyProfile(sentence_boundaries="preserve")
    ).validate(None, chunks)

    assert any(
        f.validator_name == "chunk_boundary_validator"
        and f.failure_type == ParserFailureType.CHUNK_BOUNDARY_DAMAGE
        and f.is_heuristic
        for f in findings
    )


# ---------------------------------------------------------------------------
# LATE_CHUNKING — allows mid-sentence, requires provenance
# ---------------------------------------------------------------------------


def test_late_chunking_requires_provenance_but_allows_mid_sentence():
    # Connector-start chunks should NOT trigger boundary warnings under LATE_CHUNKING.
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="and continues from the previous sentence",
            source_element_ids=("e1",),
            page_start=1,
        ),
        ChunkIR(
            chunk_id="c2",
            text="which applies to the following clause",
            source_element_ids=("e2",),
            page_start=1,
        ),
    ]

    findings = _engine(ChunkingStrategyType.LATE_CHUNKING).validate(None, chunks)

    assert not any(f.validator_name == "chunk_boundary_validator" for f in findings)


# ---------------------------------------------------------------------------
# Adapter-level analyzer test
# ---------------------------------------------------------------------------


def test_analyzer_uses_run_metadata_chunking_strategy():
    """Adapter resolves TABLE_AWARE from run.metadata and returns correct profile."""
    run = FakeRun(
        retrieved_chunks=[
            FakeChunk(chunk_id="c1", text="hello", metadata={}),
        ],
        metadata={"chunking_strategy": "table_aware"},
    )

    profile = chunking_profile_from_run_metadata(run)

    assert profile.strategy_type == ChunkingStrategyType.TABLE_AWARE
    assert profile.preserves_table_structure is True
    assert profile.allows_table_flattening is False
