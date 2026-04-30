from raggov.parser_validation.engine import ParserValidationEngine
from raggov.parser_validation.models import (
    ChunkIR,
    ChunkingStrategyType,
    ElementIR,
    ParsedDocumentIR,
    ParserFailureType,
    ParserSeverity,
    ParserValidationConfig,
    default_chunking_profile,
)


def _config(strategy: ChunkingStrategyType) -> ParserValidationConfig:
    return ParserValidationConfig(chunking_profile=default_chunking_profile(strategy))


def test_detects_missing_section_path_when_source_document_has_hierarchy():
    # HIERARCHICAL declares section_path preservation, so missing paths trigger WARN.
    parsed_doc = ParsedDocumentIR(
        document_id="doc1",
        elements=(
            ElementIR(
                element_id="e1",
                element_type="Title",
                text="Chapter 1",
                section_path=("Chapter 1",),
            ),
            ElementIR(
                element_id="e2",
                element_type="NarrativeText",
                text="Body text",
                section_path=("Chapter 1",),
            ),
        ),
    )

    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="Chapter 1 Body text",
            source_element_ids=("e1", "e2"),
            page_start=1,
            section_path=(),
        )
    ]

    findings = ParserValidationEngine(
        config=_config(ChunkingStrategyType.HIERARCHICAL)
    ).validate(parsed_doc, chunks)

    assert any(
        finding.validator_name == "hierarchy_validator"
        and finding.failure_type == ParserFailureType.HIERARCHY_FLATTENING
        and finding.severity == ParserSeverity.WARN
        for finding in findings
    )


def test_hierarchy_validator_passes_when_section_path_is_preserved():
    parsed_doc = ParsedDocumentIR(
        document_id="doc1",
        elements=(
            ElementIR(
                element_id="e1",
                element_type="Title",
                text="Chapter 1",
                section_path=("Chapter 1",),
            ),
            ElementIR(
                element_id="e2",
                element_type="NarrativeText",
                text="Body text",
                section_path=("Chapter 1",),
            ),
        ),
    )

    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="Chapter 1 Body text",
            source_element_ids=("e1", "e2"),
            page_start=1,
            section_path=("Chapter 1",),
        )
    ]

    findings = ParserValidationEngine(
        config=_config(ChunkingStrategyType.HIERARCHICAL)
    ).validate(parsed_doc, chunks)

    assert not any(
        finding.validator_name == "hierarchy_validator"
        for finding in findings
    )


def test_hierarchy_validator_detects_explicit_cross_section_boundary_flag():
    # HIERARCHICAL forbids cross-section chunks.
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
            text="Chapter 1 text followed by Chapter 2 text",
            source_element_ids=("e1",),
            page_start=1,
            section_path=("Chapter 1",),
            metadata={"crosses_section_boundary": True},
        )
    ]

    findings = ParserValidationEngine(
        config=_config(ChunkingStrategyType.HIERARCHICAL)
    ).validate(parsed_doc, chunks)

    assert any(
        finding.validator_name == "hierarchy_validator"
        and finding.failure_type == ParserFailureType.HIERARCHY_FLATTENING
        for finding in findings
    )


def test_text_only_inline_hierarchy_smoke_test_is_marked_heuristic():
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text=(
                "Chapter 1 General Provisions Section 1 Scope "
                "Rule 1 Applicability Annexure A Format"
            ),
            page_start=1,
        )
    ]

    findings = ParserValidationEngine().validate(None, chunks)

    assert any(
        finding.validator_name == "hierarchy_validator"
        and finding.failure_type == ParserFailureType.HIERARCHY_FLATTENING
        and finding.is_heuristic
        for finding in findings
    )


def test_unknown_inline_hierarchy_is_warn_not_fail():
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="Chapter 1 Introduction Section 1 Scope Rule 1 Applicability Annexure A Format",
            page_start=1,
        )
    ]

    findings = ParserValidationEngine().validate(None, chunks)
    hierarchy_findings = [
        finding for finding in findings if finding.validator_name == "hierarchy_validator"
    ]

    assert hierarchy_findings
    assert all(finding.severity == ParserSeverity.WARN for finding in hierarchy_findings)
    assert all(finding.is_heuristic for finding in hierarchy_findings)


def test_inline_numbered_rule_clauses_warn_as_heuristic():
    # Fallback hierarchy findings are always WARN (text patterns cannot prove flattening).
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="Rule 5 (1) Definition (2) Exception (3) Application all on one line",
            page_start=1,
        )
    ]

    findings = ParserValidationEngine().validate(None, chunks)

    assert any(
        finding.validator_name == "hierarchy_validator"
        and finding.failure_type == ParserFailureType.HIERARCHY_FLATTENING
        and finding.severity == ParserSeverity.WARN
        and finding.is_heuristic
        for finding in findings
    )


def test_hierarchical_strong_numbered_rule_flattening_fails():
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="Rule 5 (1) Definition (2) Exception (3) Application",
            page_start=1,
        )
    ]

    findings = ParserValidationEngine(
        config=_config(ChunkingStrategyType.HIERARCHICAL)
    ).validate(None, chunks)

    assert any(
        finding.validator_name == "hierarchy_validator"
        and finding.failure_type == ParserFailureType.HIERARCHY_FLATTENING
        and finding.severity == ParserSeverity.FAIL
        and finding.is_heuristic
        and "hierarchy-preserving" in finding.evidence[0].message
        for finding in findings
    )


def test_normal_prose_does_not_trigger_hierarchy_warning():
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="This is normal prose about implementation details.",
            page_start=1,
            source_element_ids=("e1",),
            section_path=("Overview",),
        )
    ]

    findings = ParserValidationEngine().validate(None, chunks)

    assert not any(
        finding.validator_name == "hierarchy_validator"
        for finding in findings
    )
