from raggov.analyzers.parsing.parser_validation import ParserValidationAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import FailureStage, FailureType
from raggov.models.run import RAGRun
from raggov.parser_validation.models import (
    ChunkIR,
    ChunkingStrategyType,
    default_chunking_profile,
)
from raggov.parser_validation.profile import ParserValidationProfile
from raggov.parser_validation.profile_lint import ProfileLintEngine


def _strict_profile(strategy: ChunkingStrategyType) -> ParserValidationProfile:
    return ParserValidationProfile(
        chunking_strategy=default_chunking_profile(strategy),
        infer_from_legacy=False,
    )


def _legacy_profile(strategy: ChunkingStrategyType) -> ParserValidationProfile:
    return ParserValidationProfile(
        chunking_strategy=default_chunking_profile(strategy),
        infer_from_legacy=True,
    )


def test_profile_lint_passes_when_required_metadata_is_present():
    profile = _strict_profile(ChunkingStrategyType.HIERARCHICAL)
    chunks = [
        ChunkIR(
            chunk_id="c1",
            text="Chapter 1 body",
            page_start=1,
            source_element_ids=("e1",),
            section_path=("Chapter 1",),
        ),
        ChunkIR(
            chunk_id="c2",
            text="Chapter 1 more body",
            page_start=1,
            source_element_ids=("e2",),
            section_path=("Chapter 1",),
        ),
    ]

    report = ProfileLintEngine().lint(chunks, profile.chunking_strategy, profile)

    assert report.has_errors is False
    assert report.authority_blocked is False


def test_profile_lint_errors_when_required_metadata_is_missing():
    profile = _strict_profile(ChunkingStrategyType.HIERARCHICAL)
    chunks = [
        ChunkIR(chunk_id="c1", text="Chapter 1 body"),
        ChunkIR(chunk_id="c2", text="Chapter 1 more body"),
    ]

    report = ProfileLintEngine().lint(chunks, profile.chunking_strategy, profile)

    assert report.has_errors is True
    assert report.authority_blocked is True
    assert {issue.code for issue in report.errors} >= {
        "required_page_metadata_missing",
        "required_provenance_missing",
    }


def test_profile_lint_legacy_mode_downgrades_required_metadata_to_warning():
    profile = _legacy_profile(ChunkingStrategyType.HIERARCHICAL)
    chunks = [
        ChunkIR(chunk_id="c1", text="Chapter 1 body"),
        ChunkIR(chunk_id="c2", text="Chapter 1 more body"),
    ]

    report = ProfileLintEngine().lint(chunks, profile.chunking_strategy, profile)

    assert report.has_errors is False
    assert report.authority_blocked is False
    assert {issue.code for issue in report.warnings} >= {
        "required_page_metadata_missing",
        "required_provenance_missing",
    }


def test_profile_lint_warns_when_sentence_boundary_flags_look_missing():
    profile = _strict_profile(ChunkingStrategyType.SENTENCE)
    chunks = [
        ChunkIR(chunk_id="c1", text="and continues from earlier", page_start=1),
        ChunkIR(chunk_id="c2", text="which applies here", page_start=1),
        ChunkIR(chunk_id="c3", text="that remains relevant", page_start=1),
        ChunkIR(chunk_id="c4", text="This chunk starts cleanly.", page_start=1),
    ]

    report = ProfileLintEngine().lint(chunks, profile.chunking_strategy, profile)

    assert any(
        issue.code == "boundary_flags_may_be_missing"
        for issue in report.warnings
    )


def test_analyzer_blocks_diagnostics_when_profile_lint_has_critical_errors():
    profile = _strict_profile(ChunkingStrategyType.HIERARCHICAL)
    run = RAGRun(
        query="What does Chapter 1 say?",
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="c1",
                text="Chapter 1 body",
                source_doc_id="doc1",
                score=None,
            )
        ],
        final_answer="Answer.",
    )

    result = ParserValidationAnalyzer(profile=profile).analyze(run)

    assert result.status == "fail"
    assert result.failure_type == FailureType.METADATA_LOSS
    assert result.stage == FailureStage.PARSING
    assert any("[profile_lint]" in evidence for evidence in result.evidence)
    assert not any("[hierarchy_validator]" in evidence for evidence in result.evidence)


def test_analyzer_legacy_profile_lint_warnings_do_not_block_diagnostics():
    profile = _legacy_profile(ChunkingStrategyType.SENTENCE)
    run = RAGRun(
        query="What applies?",
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="c1",
                text="and continues from the previous sentence",
                source_doc_id="doc1",
                score=None,
            ),
            RetrievedChunk(
                chunk_id="c2",
                text="which applies to the following clause",
                source_doc_id="doc1",
                score=None,
            ),
        ],
        final_answer="Answer.",
    )

    result = ParserValidationAnalyzer(profile=profile).analyze(run)

    assert result.status == "warn"
    assert any("[profile_lint]" in evidence for evidence in result.evidence)
    assert any("[chunk_boundary_validator]" in evidence for evidence in result.evidence)
