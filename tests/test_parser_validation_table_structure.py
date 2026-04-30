from raggov.parser_validation.engine import ParserValidationEngine
from raggov.parser_validation.models import (
    ChunkIR,
    ChunkingStrategyType,
    ParsedDocumentIR,
    ParserFailureType,
    ParserSeverity,
    ParserValidationConfig,
    TableIR,
    default_chunking_profile,
)


def _config(strategy: ChunkingStrategyType) -> ParserValidationConfig:
    return ParserValidationConfig(chunking_profile=default_chunking_profile(strategy))


def test_detects_missing_table_provenance_when_source_table_exists():
    # TABLE_AWARE requires provenance, so a table with no chunk referencing it is FAIL.
    parsed_doc = ParsedDocumentIR(
        document_id="doc1",
        parser_name="test_parser",
        tables=(
            TableIR(
                table_id="table_1",
                n_rows=3,
                n_cols=2,
                headers=("District", "Vacancies"),
                text="District Vacancies A 10 B 20",
            ),
        ),
    )

    chunks = [
        ChunkIR(
            chunk_id="chunk_1",
            text="District Vacancies A 10 B 20",
            source_table_ids=(),
        )
    ]

    findings = ParserValidationEngine(
        config=_config(ChunkingStrategyType.TABLE_AWARE)
    ).validate(parsed_doc, chunks)

    assert any(
        finding.failure_type == ParserFailureType.TABLE_STRUCTURE_LOSS
        and finding.severity == ParserSeverity.FAIL
        and not finding.is_heuristic
        for finding in findings
    )


def test_table_with_markdown_and_provenance_does_not_fail():
    parsed_doc = ParsedDocumentIR(
        document_id="doc1",
        tables=(
            TableIR(
                table_id="table_1",
                n_rows=2,
                n_cols=2,
                headers=("District", "Vacancies"),
            ),
        ),
    )

    chunks = [
        ChunkIR(
            chunk_id="chunk_1",
            text="| District | Vacancies |\n|---|---|\n| A | 10 |\n| B | 20 |",
            source_table_ids=("table_1",),
        )
    ]

    findings = ParserValidationEngine().validate(parsed_doc, chunks)

    assert not any(
        finding.failure_type == ParserFailureType.TABLE_STRUCTURE_LOSS
        and finding.severity == ParserSeverity.FAIL
        for finding in findings
    )


def test_table_with_provenance_but_flattened_text_fails():
    # TABLE_AWARE enforces table structure, so flattened text with source_table_ids is FAIL.
    parsed_doc = ParsedDocumentIR(
        document_id="doc1",
        tables=(
            TableIR(
                table_id="table_1",
                n_rows=3,
                n_cols=3,
                headers=("Name", "Role", "Count"),
                parsing_report={
                    "accuracy": 99.0,
                    "whitespace": 12.0,
                    "order": 1,
                    "page": 1,
                },
            ),
        ),
    )

    chunks = [
        ChunkIR(
            chunk_id="chunk_1",
            text="Name Role Count Alice Teacher 10 Bob Clerk 20",
            source_table_ids=("table_1",),
        )
    ]

    findings = ParserValidationEngine(
        config=_config(ChunkingStrategyType.TABLE_AWARE)
    ).validate(parsed_doc, chunks)

    assert any(
        finding.failure_type == ParserFailureType.TABLE_STRUCTURE_LOSS
        and finding.severity == ParserSeverity.FAIL
        for finding in findings
    )

    assert any(
        finding.evidence
        and finding.evidence[0].expected
        and finding.evidence[0].expected.get("parser_accuracy") == 99.0
        for finding in findings
    )


def test_text_only_numeric_table_smoke_test_is_marked_heuristic():
    # UNKNOWN profile — numeric multi-line chunk triggers WARN heuristic.
    chunks = [
        ChunkIR(
            chunk_id="chunk_1",
            text="A 10 20 30\nB 11 21 31\nC 12 22 32\nD 13 23 33",
        )
    ]

    findings = ParserValidationEngine().validate(None, chunks)

    assert any(
        finding.failure_type == ParserFailureType.TABLE_STRUCTURE_LOSS
        and finding.is_heuristic
        and finding.severity == ParserSeverity.WARN
        for finding in findings
    )


def test_collapsed_header_value_table_text_fails_as_heuristic():
    # TABLE_AWARE disallows table flattening, so the collapsed-table heuristic is FAIL.
    chunks = [
        ChunkIR(
            chunk_id="chunk_1",
            text="District Vacancies Category Warangal 5 Grade A Khammam 3 Grade B",
        )
    ]

    findings = ParserValidationEngine(
        config=_config(ChunkingStrategyType.TABLE_AWARE)
    ).validate(None, chunks)

    assert any(
        finding.failure_type == ParserFailureType.TABLE_STRUCTURE_LOSS
        and finding.is_heuristic
        and finding.severity == ParserSeverity.FAIL
        for finding in findings
    )


def test_collapsed_header_value_table_text_warns_as_heuristic_under_unknown():
    # UNKNOWN allows table flattening, so the collapsed-table heuristic is downgraded to WARN.
    chunks = [
        ChunkIR(
            chunk_id="chunk_1",
            text="District Vacancies Category Warangal 5 Grade A Khammam 3 Grade B",
        )
    ]

    findings = ParserValidationEngine().validate(None, chunks)

    assert any(
        finding.failure_type == ParserFailureType.TABLE_STRUCTURE_LOSS
        and finding.is_heuristic
        and finding.severity == ParserSeverity.WARN
        for finding in findings
    )


def test_coded_prose_does_not_trigger_collapsed_table_heuristic():
    chunks = [
        ChunkIR(
            chunk_id="chunk_1",
            text=(
                "Support portal submission with proof of purchase is required for warranty claims. "
                "Use claim code ORX442 and RMA token ZX9."
            ),
        )
    ]

    findings = ParserValidationEngine().validate(None, chunks)

    assert not any(
        finding.failure_type == ParserFailureType.TABLE_STRUCTURE_LOSS
        for finding in findings
    )


def test_non_numeric_prose_does_not_trigger_table_warning():
    chunks = [
        ChunkIR(
            chunk_id="chunk_1",
            text="This is a normal paragraph about policy implementation and school administration.",
        )
    ]

    findings = ParserValidationEngine().validate(None, chunks)

    assert not any(
        finding.failure_type == ParserFailureType.TABLE_STRUCTURE_LOSS
        for finding in findings
    )
