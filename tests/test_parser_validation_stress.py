from raggov.parser_validation.adapters import chunk_from_rag_chunk
from raggov.parser_validation.engine import ParserValidationEngine
from raggov.parser_validation.models import (
    ChunkIR,
    ChunkingStrategyType,
    ElementIR,
    ParsedDocumentIR,
    ParserEvidence,
    ParserFailureType,
    ParserFinding,
    ParserSeverity,
    ParserValidationConfig,
    TableIR,
    default_chunking_profile,
)


def _config(strategy: ChunkingStrategyType) -> ParserValidationConfig:
    return ParserValidationConfig(chunking_profile=default_chunking_profile(strategy))


def test_table_split_across_chunks_with_header_repeated_does_not_fail_table_structure():
    parsed_doc = ParsedDocumentIR(
        document_id="doc1",
        tables=(TableIR(table_id="t1", n_rows=4, n_cols=2, headers=("Name", "Score")),),
    )
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="| Name | Score |\n|---|---|\n| A | 10 |\n| B | 20 |",
            source_table_ids=("t1",),
            page_start=1,
            source_element_ids=("e1",),
            section_path=("Results",),
        ),
        ChunkIR(
            chunk_id="c2",
            text="| Name | Score |\n|---|---|\n| C | 30 |\n| D | 40 |",
            source_table_ids=("t1",),
            page_start=1,
            source_element_ids=("e2",),
            section_path=("Results",),
        ),
    ]
    findings = ParserValidationEngine(
        config=_config(ChunkingStrategyType.TABLE_AWARE)
    ).validate(parsed_doc, chunks)
    assert not any(
        f.validator_name == "table_structure_validator"
        and f.failure_type == ParserFailureType.TABLE_STRUCTURE_LOSS
        and f.severity == ParserSeverity.FAIL
        for f in findings
    )


def test_table_split_across_chunks_without_header_warns_or_fails():
    # TABLE_AWARE enforces header preservation; c2 lacks headers → WARN/FAIL.
    parsed_doc = ParsedDocumentIR(
        document_id="doc1",
        tables=(TableIR(table_id="t1", n_rows=4, n_cols=2, headers=("Name", "Score")),),
    )
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="| Name | Score |\n|---|---|\n| A | 10 |\n| B | 20 |",
            source_table_ids=("t1",),
            page_start=1,
            source_element_ids=("e1",),
            section_path=("Results",),
        ),
        ChunkIR(
            chunk_id="c2",
            text="| C | 30 |\n| D | 40 |",
            source_table_ids=("t1",),
            page_start=1,
            source_element_ids=("e2",),
            section_path=("Results",),
        ),
    ]

    findings = ParserValidationEngine(
        config=_config(ChunkingStrategyType.TABLE_AWARE)
    ).validate(parsed_doc, chunks)

    assert any(
        f.validator_name == "table_structure_validator"
        and f.failure_type == ParserFailureType.TABLE_STRUCTURE_LOSS
        and f.severity in {ParserSeverity.WARN, ParserSeverity.FAIL}
        and f.evidence
        and f.evidence[0].chunk_id == "c2"
        for f in findings
    )


def test_table_with_html_serialization_does_not_fail():
    parsed_doc = ParsedDocumentIR(
        document_id="doc1",
        tables=(TableIR(table_id="t1", n_rows=2, n_cols=2, headers=("Name", "Score")),),
    )
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="<table><tr><th>Name</th><th>Score</th></tr><tr><td>A</td><td>10</td></tr></table>",
            source_table_ids=("t1",),
            page_start=1,
            source_element_ids=("e1",),
            section_path=("Results",),
        )
    ]

    findings = ParserValidationEngine(
        config=_config(ChunkingStrategyType.TABLE_AWARE)
    ).validate(parsed_doc, chunks)

    assert not any(
        f.validator_name == "table_structure_validator"
        and f.failure_type == ParserFailureType.TABLE_STRUCTURE_LOSS
        and f.severity == ParserSeverity.FAIL
        for f in findings
    )


def test_flattened_table_with_high_camelot_accuracy_fails():
    # TABLE_AWARE enforces structure; flattened text with provenance is FAIL.
    parsed_doc = ParsedDocumentIR(
        document_id="doc1",
        tables=(
            TableIR(
                table_id="t1",
                n_rows=3,
                n_cols=2,
                headers=("Name", "Score"),
                parsing_report={"accuracy": 99.2, "whitespace": 8.0, "order": 1, "page": 2},
            ),
        ),
    )
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="Name Score A 10 B 20 C 30",
            source_table_ids=("t1",),
            page_start=2,
            source_element_ids=("e1",),
            section_path=("Results",),
        )
    ]

    findings = ParserValidationEngine(
        config=_config(ChunkingStrategyType.TABLE_AWARE)
    ).validate(parsed_doc, chunks)

    assert any(
        f.validator_name == "table_structure_validator"
        and f.failure_type == ParserFailureType.TABLE_STRUCTURE_LOSS
        and f.severity == ParserSeverity.FAIL
        for f in findings
    )
    assert any(
        f.evidence and f.evidence[0].expected and f.evidence[0].expected.get("parser_accuracy") == 99.2
        for f in findings
        if f.validator_name == "table_structure_validator"
    )


def test_numeric_prose_should_not_trigger_table_fallback_false_positive():
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text=(
                "In 2024, 35 schools improved by 12 percent. In 2025, 41 schools "
                "reported gains after 6 months of support."
            ),
            page_start=1,
            source_element_ids=("e1",),
            section_path=("Overview",),
        )
    ]

    findings = ParserValidationEngine().validate(None, chunks)

    assert not any(f.failure_type == ParserFailureType.TABLE_STRUCTURE_LOSS for f in findings)


def test_haystack_like_metadata_is_adapted_correctly():
    chunk = {
        "content": "Hello",
        "metadata": {
            "source_id": "docA",
            "split_id": 3,
            "page_number": 5,
            "section_path": ["Chapter 1"],
            "element_ids": ["e1"],
        },
    }

    out = chunk_from_rag_chunk(chunk)

    assert out.page_start == 5
    assert out.source_element_ids == ("e1",)
    assert out.metadata["source_id"] == "docA"
    assert out.metadata["split_id"] == 3


def test_unstructured_like_orig_elements_list_is_adapted_as_source_element_ids():
    chunk = {
        "text": "Intro body",
        "metadata": {
            "orig_elements": ["Title: Intro", "NarrativeText: Body"],
            "page_number": 2,
        },
    }

    out = chunk_from_rag_chunk(chunk)

    assert out.source_element_ids == ("Title: Intro", "NarrativeText: Body")
    assert out.page_start == 2


def test_pymupdf4llm_like_metadata_headers_toc_path_is_adapted():
    chunk = {
        "text": "Chapter text",
        "metadata": {
            "page": 9,
            "toc_path": ["Part I", "Chapter 2"],
        },
    }

    out = chunk_from_rag_chunk(chunk)

    assert out.page_start == 9
    assert out.section_path == ("Part I", "Chapter 2")


def test_metadata_validator_does_not_warn_when_all_chunks_have_page_and_provenance():
    chunks = [
        ChunkIR(chunk_id="c1", text="A", page_start=1, source_element_ids=("e1",)),
        ChunkIR(chunk_id="c2", text="B", page_start=2, source_element_ids=("e2",)),
    ]

    findings = ParserValidationEngine().validate(None, chunks)

    assert not any(f.validator_name == "metadata_validator" for f in findings)


def test_metadata_validator_warns_when_only_half_chunks_have_provenance():
    # HIERARCHICAL requires provenance; 50% coverage is below the 90% threshold.
    chunks = [
        ChunkIR(chunk_id="c1", text="A", page_start=1, source_element_ids=("e1",)),
        ChunkIR(chunk_id="c2", text="B", page_start=1),
        ChunkIR(chunk_id="c3", text="C", page_start=2, source_element_ids=("e3",)),
        ChunkIR(chunk_id="c4", text="D", page_start=2),
    ]

    findings = ParserValidationEngine(
        config=_config(ChunkingStrategyType.HIERARCHICAL)
    ).validate(None, chunks)

    assert any(
        f.validator_name == "metadata_validator"
        and f.failure_type == ParserFailureType.PROVENANCE_MISSING
        and f.severity == ParserSeverity.WARN
        for f in findings
    )


def test_table_of_contents_like_chunk_does_not_create_fail():
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="Chapter 1 Introduction Section 1 Scope Section 2 Definitions Section 3 Powers",
            page_start=1,
        )
    ]

    findings = ParserValidationEngine().validate(None, chunks)

    hierarchy_findings = [f for f in findings if f.validator_name == "hierarchy_validator"]
    assert all(f.severity != ParserSeverity.FAIL for f in hierarchy_findings)
    assert all(f.is_heuristic for f in hierarchy_findings)


def test_preserved_section_path_with_multiple_section_mentions_does_not_trigger_hierarchy_fallback():
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="Chapter 1 Introduction Section 1 Scope Section 2 Definitions Section 3 Powers",
            page_start=1,
            source_element_ids=("e1",),
            section_path=("Chapter 1",),
        )
    ]

    findings = ParserValidationEngine().validate(None, chunks)

    assert not any(f.validator_name == "hierarchy_validator" for f in findings)


def test_parsed_doc_heading_elements_without_section_path_still_count_as_hierarchy():
    # HIERARCHICAL enforces section_path; chunks with provenance but no section_path trigger WARN.
    parsed_doc = ParsedDocumentIR(
        document_id="doc1",
        elements=(ElementIR(element_id="e1", element_type="Heading", text="Chapter 1"),),
    )
    chunks = [
        ChunkIR(chunk_id="c1", text="Chapter 1 Body", page_start=1, source_element_ids=("e1",))
    ]

    findings = ParserValidationEngine(
        config=_config(ChunkingStrategyType.HIERARCHICAL)
    ).validate(parsed_doc, chunks)

    assert any(
        f.validator_name == "hierarchy_validator"
        and f.failure_type == ParserFailureType.HIERARCHY_FLATTENING
        and f.severity == ParserSeverity.WARN
        for f in findings
    )


def test_single_connector_start_chunk_below_threshold_does_not_warn():
    chunks = [
        ChunkIR(
            chunk_id=f"c{i}",
            text="and continues from the previous sentence" if i == 0 else "This is a clean sentence.",
            page_start=1,
            source_element_ids=(f"e{i}",),
            section_path=("Overview",),
        )
        for i in range(20)
    ]

    findings = ParserValidationEngine().validate(None, chunks)

    assert not any(f.validator_name == "chunk_boundary_validator" for f in findings)


def test_many_connector_start_chunks_trigger_heuristic_boundary_warning():
    # SENTENCE forbids mid-sentence starts so the connector heuristic runs.
    chunks = [
        ChunkIR(chunk_id="c1", text="and continues from the previous sentence", page_start=1),
        ChunkIR(chunk_id="c2", text="which applies to the following clause", page_start=1),
        ChunkIR(chunk_id="c3", text="that remains subject to approval", page_start=1),
        ChunkIR(chunk_id="c4", text="This is a clean sentence.", page_start=1),
        ChunkIR(chunk_id="c5", text="This is another clean sentence.", page_start=1),
    ]

    findings = ParserValidationEngine(
        config=_config(ChunkingStrategyType.SENTENCE)
    ).validate(None, chunks)

    assert any(
        f.validator_name == "chunk_boundary_validator"
        and f.failure_type == ParserFailureType.CHUNK_BOUNDARY_DAMAGE
        and f.severity == ParserSeverity.WARN
        and f.is_heuristic
        for f in findings
    )


def test_boundary_metadata_flag_triggers_even_single_chunk():
    # TABLE_AWARE enforces table structure, so split_inside_table triggers WARN.
    chunks = [ChunkIR(chunk_id="c1", text="row fragment", metadata={"split_inside_table": True})]

    findings = ParserValidationEngine(
        config=_config(ChunkingStrategyType.TABLE_AWARE)
    ).validate(None, chunks)

    assert any(
        f.validator_name == "chunk_boundary_validator"
        and f.failure_type == ParserFailureType.CHUNK_BOUNDARY_DAMAGE
        and f.severity == ParserSeverity.WARN
        and not f.is_heuristic
        for f in findings
    )


def test_bullet_lowercase_fragment_not_flagged_as_boundary_damage():
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="- applicable conditions are listed below",
            page_start=1,
            source_element_ids=("e1",),
            section_path=("Overview",),
        ),
        ChunkIR(
            chunk_id="c2",
            text="• applicable conditions are listed below",
            page_start=1,
            source_element_ids=("e2",),
            section_path=("Overview",),
        ),
    ]

    findings = ParserValidationEngine().validate(None, chunks)

    assert not any(f.validator_name == "chunk_boundary_validator" for f in findings)


def test_fail_out_ranks_warn_even_when_warn_has_higher_confidence():
    class FakeValidator:
        name = "fake_validator"

        def validate(self, parsed_doc, chunks, config):
            return [
                ParserFinding(
                    failure_type=ParserFailureType.METADATA_LOSS,
                    severity=ParserSeverity.WARN,
                    confidence=0.99,
                    evidence=(ParserEvidence(message="warn"),),
                    remediation="warn",
                    validator_name=self.name,
                ),
                ParserFinding(
                    failure_type=ParserFailureType.TABLE_STRUCTURE_LOSS,
                    severity=ParserSeverity.FAIL,
                    confidence=0.70,
                    evidence=(ParserEvidence(message="fail"),),
                    remediation="fail",
                    validator_name=self.name,
                ),
            ]

    findings = ParserValidationEngine(validators=[FakeValidator()]).validate(
        None,
        [ChunkIR(chunk_id="c1", text="hello")],
    )

    assert findings[0].severity == ParserSeverity.FAIL
    assert findings[1].severity == ParserSeverity.WARN


def test_table_fail_out_ranks_metadata_warn_in_default_engine():
    # TABLE_AWARE: table FAIL and metadata WARN — table should rank first.
    parsed_doc = ParsedDocumentIR(
        document_id="doc1",
        tables=(TableIR(table_id="t1", n_rows=3, n_cols=2, headers=("Name", "Score")),),
    )
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="Name Score A 10 B 20 C 30",
            source_table_ids=("t1",),
        )
    ]

    findings = ParserValidationEngine(
        config=_config(ChunkingStrategyType.TABLE_AWARE)
    ).validate(parsed_doc, chunks)

    assert findings[0].validator_name == "table_structure_validator"
    assert findings[0].failure_type == ParserFailureType.TABLE_STRUCTURE_LOSS
    assert findings[0].severity == ParserSeverity.FAIL


def test_disabling_text_fallback_removes_all_heuristic_findings_but_keeps_structural_findings():
    # TABLE_AWARE: structural table FAIL survives when fallback is disabled.
    parsed_doc = ParsedDocumentIR(
        document_id="doc1",
        tables=(TableIR(table_id="t1", n_rows=3, n_cols=2, headers=("Name", "Score")),),
    )
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="and Name Score A 10 B 20 C 30",
            source_table_ids=("t1",),
        )
    ]
    engine = ParserValidationEngine(
        config=ParserValidationConfig(
            enable_text_fallback_heuristics=False,
            chunking_profile=default_chunking_profile(ChunkingStrategyType.TABLE_AWARE),
        )
    )

    findings = engine.validate(parsed_doc, chunks)

    assert any(
        f.validator_name == "table_structure_validator"
        and f.failure_type == ParserFailureType.TABLE_STRUCTURE_LOSS
        for f in findings
    )
    assert not any(f.is_heuristic for f in findings)
