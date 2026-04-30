from pathlib import Path

from typer.testing import CliRunner

from raggov.cli import app
from raggov.parser_validation.domain_metadata import (
    DomainMetadataGovernanceEngine,
    DomainMetadataRule,
)
from raggov.parser_validation.ingestion import (
    IngestionValidationRequest,
    validate_ingestion,
)
from raggov.parser_validation.metadata_normalizer import MetadataNormalizer
from raggov.parser_validation.models import (
    ChunkIR,
    ChunkingStrategyType,
    ParsedDocumentIR,
    TableIR,
)
from raggov.parser_validation.presets import list_presets, load_preset_profile
from raggov.parser_validation.profile_questionnaire import build_profile_from_answers


runner = CliRunner()


def test_first_wave_presets_load_and_normalize_expected_metadata():
    assert set(list_presets()) >= {
        "unstructured_by_title",
        "haystack_document_splitter",
        "langchain_recursive_character_splitter",
        "pymupdf4llm_page",
    }

    profile = load_preset_profile("haystack_document_splitter")
    normalized = MetadataNormalizer(profile.metadata_mapping).normalize(
        {
            "content": "Hello",
            "metadata": {
                "source_id": "docA",
                "split_id": 3,
                "page_number": "5",
            },
        }
    )

    assert profile.chunking_strategy.strategy_type == ChunkingStrategyType.RECURSIVE_TEXT
    assert normalized.text == "Hello"
    assert normalized.document_id == "docA"
    assert normalized.page_start == 5
    assert normalized.domain_fields["split_id"] == 3


def test_pymupdf4llm_preset_normalizes_page_and_toc_path():
    profile = load_preset_profile("pymupdf4llm_page")
    normalized = MetadataNormalizer(profile.metadata_mapping).normalize(
        {
            "text": "Page text",
            "metadata": {
                "page": 9,
                "toc_path": ["Part I", "Chapter 2"],
            },
        }
    )

    assert normalized.page_start == 9
    assert normalized.section_path == ("Part I", "Chapter 2")


def test_validate_ingestion_returns_document_level_report():
    parsed_doc = ParsedDocumentIR(
        document_id="doc1",
        tables=(TableIR(table_id="t1", headers=("Name", "Score"), n_rows=2, n_cols=2),),
    )
    request = IngestionValidationRequest(
        parsed_doc=parsed_doc,
        chunks=(
            ChunkIR(
                chunk_id="c1",
                text="Name Score A 10 B 20",
                source_table_ids=("t1",),
                page_start=1,
            ),
        ),
        profile=load_preset_profile("unstructured_by_title"),
    )

    report = validate_ingestion(request)

    assert report.document_id == "doc1"
    assert report.total_chunks == 1
    assert report.quality_summary.table_count == 1
    assert report.warning_count >= 0
    assert report.quality_summary.document_quality_score <= 1.0


def test_profile_questionnaire_builds_profile_without_runtime_dependency():
    profile = build_profile_from_answers(
        {
            "name": "custom_sentence",
            "parser_name": "custom",
            "chunking_strategy": "sentence",
            "page_field": "metadata.page_number",
            "provenance_field": "metadata.element_ids",
            "section_field": "metadata.headers",
            "infer_from_legacy": False,
        }
    )

    assert profile.parser.name == "custom"
    assert profile.chunking_strategy.strategy_type == ChunkingStrategyType.SENTENCE
    assert profile.metadata_mapping.page_start.aliases == ("metadata.page_number",)
    assert profile.metadata_mapping.source_element_ids.aliases == ("metadata.element_ids",)


def test_profile_questionnaire_cli_writes_profile_yaml(tmp_path: Path):
    output_path = tmp_path / "profile.yaml"

    result = runner.invoke(
        app,
        [
            "parser-profile-init",
            "--name",
            "custom_sentence",
            "--parser-name",
            "custom",
            "--chunking-strategy",
            "sentence",
            "--page-field",
            "metadata.page_number",
            "--provenance-field",
            "metadata.element_ids",
            "--section-field",
            "metadata.headers",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert "custom_sentence" in output_path.read_text(encoding="utf-8")


def test_domain_metadata_governance_is_separate_from_parser_validators():
    engine = DomainMetadataGovernanceEngine(
        rules=(
            DomainMetadataRule(field="agency", required=True),
            DomainMetadataRule(field="jurisdiction", allowed_values=("federal", "state")),
        )
    )

    report = engine.validate(
        {"jurisdiction": "city"},
        document_id="doc1",
        chunk_id="c1",
    )

    assert report.has_errors is True
    assert {issue.code for issue in report.issues} == {
        "domain_field_missing",
        "domain_field_invalid",
    }
