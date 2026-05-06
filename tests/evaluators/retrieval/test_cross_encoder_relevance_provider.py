"""Tests for CrossEncoderRetrievalRelevanceProvider and integration points.

No model downloads: sentence_transformers is mocked via sys.modules.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from raggov.analyzers.retrieval.evidence_profile import RetrievalEvidenceProfilerV0
from raggov.analyzers.retrieval_diagnosis import RetrievalDiagnosisAnalyzerV0
from raggov.evaluators.base import ExternalEvaluatorProvider, ExternalSignalType
from raggov.evaluators.retrieval.cross_encoder import CrossEncoderRetrievalRelevanceProvider
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import AnalyzerResult, FailureType, SufficiencyResult
from raggov.models.grounding import ClaimEvidenceRecord, ClaimVerificationLabel
from raggov.models.retrieval_diagnosis import RetrievalFailureType
from raggov.models.retrieval_evidence import (
    ChunkEvidenceProfile,
    EvidenceRole,
    QueryRelevanceLabel,
    RelevanceMethod,
    RetrievalEvidenceProfile,
)
from raggov.models.run import RAGRun


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(chunks: list[RetrievedChunk] | None = None) -> RAGRun:
    return RAGRun(
        query="What are the eligibility criteria for the scheme?",
        retrieved_chunks=chunks or [
            RetrievedChunk(chunk_id="c1", source_doc_id="doc-1", text="Applicants must be over 18.", score=0.9),
            RetrievedChunk(chunk_id="c2", source_doc_id="doc-2", text="Football results from last weekend.", score=0.4),
            RetrievedChunk(chunk_id="c3", source_doc_id="doc-1", text="Income below threshold is required.", score=0.3),
        ],
        final_answer="Applicants over 18 with income below threshold are eligible.",
    )


def _mock_sentence_transformers(scores: list[float]) -> tuple[MagicMock, Any]:
    """Inject a fake sentence_transformers module; return (mock_model, restore_fn)."""
    fake_module = ModuleType("sentence_transformers")
    mock_model = MagicMock()
    mock_model.predict.return_value = scores
    fake_cls = MagicMock(return_value=mock_model)
    fake_module.CrossEncoder = fake_cls  # type: ignore[attr-defined]

    original = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = fake_module

    def restore():
        if original is None:
            sys.modules.pop("sentence_transformers", None)
        else:
            sys.modules["sentence_transformers"] = original

    return mock_model, restore


# ---------------------------------------------------------------------------
# 1. Missing dependency behavior
# ---------------------------------------------------------------------------


def test_missing_dependency_returns_not_succeeded(monkeypatch: pytest.MonkeyPatch) -> None:
    # Remove sentence_transformers from sys.modules to simulate absence.
    original = sys.modules.pop("sentence_transformers", None)
    try:
        monkeypatch.setattr(
            "raggov.evaluators.retrieval.cross_encoder.importlib.util.find_spec",
            lambda name: None if name == "sentence_transformers" else object(),
        )
        provider = CrossEncoderRetrievalRelevanceProvider()
        # Re-create instance after removing the module so _model cache is clear
        result = provider.evaluate(_make_run())
        assert result.succeeded is False
        assert result.missing_dependency is True
        assert result.error is not None
        assert "sentence-transformers" in result.error
    finally:
        if original is not None:
            sys.modules["sentence_transformers"] = original


def test_missing_dependency_is_available_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    original = sys.modules.pop("sentence_transformers", None)
    try:
        monkeypatch.setattr(
            "raggov.evaluators.retrieval.cross_encoder.importlib.util.find_spec",
            lambda name: None if name == "sentence_transformers" else object(),
        )
        provider = CrossEncoderRetrievalRelevanceProvider()
        assert provider.is_available() is False
    finally:
        if original is not None:
            sys.modules["sentence_transformers"] = original


def test_is_available_returns_true_when_dependency_is_installed_but_not_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = sys.modules.pop("sentence_transformers", None)
    try:
        monkeypatch.setattr(
            "raggov.evaluators.retrieval.cross_encoder.importlib.util.find_spec",
            lambda name: object() if name == "sentence_transformers" else None,
        )
        provider = CrossEncoderRetrievalRelevanceProvider()
        assert provider.is_available() is True
    finally:
        if original is not None:
            sys.modules["sentence_transformers"] = original


def test_score_relevance_returns_empty_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """score_relevance() returns [] (not raises) when unavailable — callers check is_available()."""
    original = sys.modules.pop("sentence_transformers", None)
    try:
        monkeypatch.setattr(
            "raggov.evaluators.retrieval.cross_encoder.importlib.util.find_spec",
            lambda name: None if name == "sentence_transformers" else object(),
        )
        provider = CrossEncoderRetrievalRelevanceProvider()
        result = provider.score_relevance("query", ["chunk text"])
        assert result == []
    finally:
        if original is not None:
            sys.modules["sentence_transformers"] = original


def test_missing_cached_model_fails_fast_without_download_attempt() -> None:
    fake_module = ModuleType("sentence_transformers")
    fake_cls = MagicMock(side_effect=OSError("model files not found"))
    fake_module.CrossEncoder = fake_cls  # type: ignore[attr-defined]
    original = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = fake_module
    try:
        provider = CrossEncoderRetrievalRelevanceProvider()
        result = provider.evaluate(_make_run())
        assert result.succeeded is False
        assert "not cached locally" in (result.error or "")
        fake_cls.assert_called_once_with(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            local_files_only=True,
        )
    finally:
        if original is None:
            sys.modules.pop("sentence_transformers", None)
        else:
            sys.modules["sentence_transformers"] = original


# ---------------------------------------------------------------------------
# 2. Mocked scores → correct labels
# ---------------------------------------------------------------------------


def test_relevant_label_when_score_above_relevant_threshold() -> None:
    mock_model, restore = _mock_sentence_transformers([4.5])
    try:
        provider = CrossEncoderRetrievalRelevanceProvider(
            {"relevant_threshold": 3.0, "partial_threshold": 0.0}
        )
        provider._model = mock_model  # skip loading
        run = _make_run([
            RetrievedChunk(chunk_id="c1", source_doc_id="doc-1", text="Relevant text.", score=0.9),
        ])
        result = provider.evaluate(run)
        assert result.succeeded is True
        assert result.signals[0].label == "relevant"
        assert result.signals[0].value == pytest.approx(4.5, abs=0.01)
    finally:
        restore()


def test_partial_label_when_score_between_thresholds() -> None:
    mock_model, restore = _mock_sentence_transformers([1.5])
    try:
        provider = CrossEncoderRetrievalRelevanceProvider(
            {"relevant_threshold": 3.0, "partial_threshold": 0.0}
        )
        provider._model = mock_model
        run = _make_run([
            RetrievedChunk(chunk_id="c1", source_doc_id="doc-1", text="Somewhat relevant.", score=0.5),
        ])
        result = provider.evaluate(run)
        assert result.signals[0].label == "partial"
    finally:
        restore()


def test_irrelevant_label_when_score_below_partial_threshold() -> None:
    mock_model, restore = _mock_sentence_transformers([-2.0])
    try:
        provider = CrossEncoderRetrievalRelevanceProvider(
            {"relevant_threshold": 3.0, "partial_threshold": 0.0}
        )
        provider._model = mock_model
        run = _make_run([
            RetrievedChunk(chunk_id="c1", source_doc_id="doc-1", text="Unrelated.", score=0.1),
        ])
        result = provider.evaluate(run)
        assert result.signals[0].label == "irrelevant"
    finally:
        restore()


def test_custom_thresholds_respected() -> None:
    mock_model, restore = _mock_sentence_transformers([0.6])
    try:
        provider = CrossEncoderRetrievalRelevanceProvider(
            {"relevant_threshold": 0.5, "partial_threshold": 0.2}
        )
        provider._model = mock_model
        run = _make_run([
            RetrievedChunk(chunk_id="c1", source_doc_id="doc-1", text="text", score=0.5),
        ])
        result = provider.evaluate(run)
        # 0.6 >= 0.5 → relevant
        assert result.signals[0].label == "relevant"
    finally:
        restore()


def test_multiple_chunks_multiple_signals() -> None:
    mock_model, restore = _mock_sentence_transformers([4.5, 1.5, -2.0])
    try:
        provider = CrossEncoderRetrievalRelevanceProvider()
        provider._model = mock_model
        result = provider.evaluate(_make_run())
        assert result.succeeded is True
        assert len(result.signals) == 3
        assert result.signals[0].label == "relevant"
        assert result.signals[1].label == "partial"
        assert result.signals[2].label == "irrelevant"
    finally:
        restore()


# ---------------------------------------------------------------------------
# 3. Signals include chunk IDs and doc IDs
# ---------------------------------------------------------------------------


def test_signals_include_chunk_id() -> None:
    mock_model, restore = _mock_sentence_transformers([4.5, -1.0])
    try:
        provider = CrossEncoderRetrievalRelevanceProvider()
        provider._model = mock_model
        run = _make_run([
            RetrievedChunk(chunk_id="chunk-A", source_doc_id="doc-X", text="text A", score=0.9),
            RetrievedChunk(chunk_id="chunk-B", source_doc_id="doc-Y", text="text B", score=0.2),
        ])
        result = provider.evaluate(run)
        assert result.signals[0].affected_chunk_ids == ["chunk-A"]
        assert result.signals[1].affected_chunk_ids == ["chunk-B"]
    finally:
        restore()


def test_signals_include_doc_id() -> None:
    mock_model, restore = _mock_sentence_transformers([4.5, -1.0])
    try:
        provider = CrossEncoderRetrievalRelevanceProvider()
        provider._model = mock_model
        run = _make_run([
            RetrievedChunk(chunk_id="chunk-A", source_doc_id="doc-X", text="text", score=0.9),
            RetrievedChunk(chunk_id="chunk-B", source_doc_id="doc-Y", text="text", score=0.2),
        ])
        result = provider.evaluate(run)
        assert result.signals[0].affected_doc_ids == ["doc-X"]
        assert result.signals[1].affected_doc_ids == ["doc-Y"]
    finally:
        restore()


def test_signals_provider_enum() -> None:
    mock_model, restore = _mock_sentence_transformers([3.5])
    try:
        provider = CrossEncoderRetrievalRelevanceProvider()
        provider._model = mock_model
        run = _make_run([
            RetrievedChunk(chunk_id="c1", source_doc_id="doc-1", text="text", score=0.8),
        ])
        result = provider.evaluate(run)
        assert result.signals[0].provider == ExternalEvaluatorProvider.cross_encoder
        assert result.signals[0].signal_type == ExternalSignalType.retrieval_relevance
        assert result.signals[0].metric_name == "cross_encoder_relevance"
    finally:
        restore()


# ---------------------------------------------------------------------------
# 4. calibration_status = uncalibrated_locally
# ---------------------------------------------------------------------------


def test_calibration_status_uncalibrated_locally() -> None:
    mock_model, restore = _mock_sentence_transformers([3.5])
    try:
        provider = CrossEncoderRetrievalRelevanceProvider()
        provider._model = mock_model
        run = _make_run([RetrievedChunk(chunk_id="c1", source_doc_id="doc-1", text="t", score=0.8)])
        result = provider.evaluate(run)
        for signal in result.signals:
            assert signal.calibration_status == "uncalibrated_locally"
    finally:
        restore()


# ---------------------------------------------------------------------------
# 5. recommended_for_gating = False
# ---------------------------------------------------------------------------


def test_recommended_for_gating_false() -> None:
    mock_model, restore = _mock_sentence_transformers([3.5])
    try:
        provider = CrossEncoderRetrievalRelevanceProvider()
        provider._model = mock_model
        run = _make_run([RetrievedChunk(chunk_id="c1", source_doc_id="doc-1", text="t", score=0.8)])
        result = provider.evaluate(run)
        for signal in result.signals:
            assert signal.recommended_for_gating is False
    finally:
        restore()


# ---------------------------------------------------------------------------
# 6. RetrievalEvidenceProfile can include external relevance signals
# ---------------------------------------------------------------------------


def test_apply_external_signals_updates_chunk_labels() -> None:
    mock_model, restore = _mock_sentence_transformers([4.5, -2.0])
    try:
        provider = CrossEncoderRetrievalRelevanceProvider()
        provider._model = mock_model
        run = _make_run([
            RetrievedChunk(chunk_id="c1", source_doc_id="doc-1", text="Relevant text.", score=0.9),
            RetrievedChunk(chunk_id="c2", source_doc_id="doc-2", text="Irrelevant text.", score=0.2),
        ])
        # Build baseline profile with lexical scorer
        profiler = RetrievalEvidenceProfilerV0()
        profile = profiler.build(run)

        # Apply cross-encoder signals
        result = provider.evaluate(run)
        assert result.succeeded is True
        profiler.apply_external_relevance_signals(profile, result.signals)

        chunk_map = {cp.chunk_id: cp for cp in profile.chunks}
        assert chunk_map["c1"].query_relevance_label == QueryRelevanceLabel.RELEVANT
        assert chunk_map["c1"].relevance_method == RelevanceMethod.CROSS_ENCODER
        assert chunk_map["c2"].query_relevance_label == QueryRelevanceLabel.IRRELEVANT
        assert chunk_map["c2"].relevance_method == RelevanceMethod.CROSS_ENCODER
    finally:
        restore()


def test_apply_external_signals_recomputes_noisy_chunk_ids() -> None:
    mock_model, restore = _mock_sentence_transformers([4.5, -2.0, 1.0])
    try:
        provider = CrossEncoderRetrievalRelevanceProvider()
        provider._model = mock_model
        run = _make_run()
        profiler = RetrievalEvidenceProfilerV0()
        profile = profiler.build(run)

        result = provider.evaluate(run)
        profiler.apply_external_relevance_signals(profile, result.signals)

        # c1 relevant, c2 irrelevant, c3 partial → c2 and c3 should be noisy
        assert "c2" in profile.noisy_chunk_ids
        assert "c3" in profile.noisy_chunk_ids
        assert "c1" not in profile.noisy_chunk_ids
    finally:
        restore()


def test_apply_external_signals_marks_profile_degraded() -> None:
    mock_model, restore = _mock_sentence_transformers([-2.0])
    try:
        provider = CrossEncoderRetrievalRelevanceProvider()
        provider._model = mock_model
        run = _make_run([
            RetrievedChunk(chunk_id="c1", source_doc_id="doc-1", text="Irrelevant.", score=0.1),
        ])
        profiler = RetrievalEvidenceProfilerV0()
        profile = profiler.build(run)

        result = provider.evaluate(run)
        profiler.apply_external_relevance_signals(profile, result.signals)

        assert profile.overall_retrieval_status == "degraded"
    finally:
        restore()


def test_apply_external_signals_adds_limitation_note() -> None:
    mock_model, restore = _mock_sentence_transformers([3.5])
    try:
        provider = CrossEncoderRetrievalRelevanceProvider()
        provider._model = mock_model
        run = _make_run([RetrievedChunk(chunk_id="c1", source_doc_id="doc-1", text="t", score=0.8)])
        profiler = RetrievalEvidenceProfilerV0()
        profile = profiler.build(run)

        result = provider.evaluate(run)
        profiler.apply_external_relevance_signals(profile, result.signals)

        assert any("external_relevance_signals_applied" in lim for lim in profile.limitations)
    finally:
        restore()


def test_apply_external_signals_preserves_signal_provenance_on_profile() -> None:
    mock_model, restore = _mock_sentence_transformers([3.5])
    try:
        provider = CrossEncoderRetrievalRelevanceProvider()
        provider._model = mock_model
        run = _make_run([RetrievedChunk(chunk_id="c1", source_doc_id="doc-1", text="t", score=0.8)])
        profiler = RetrievalEvidenceProfilerV0()
        profile = profiler.build(run)

        result = provider.evaluate(run)
        profiler.apply_external_relevance_signals(profile, result.signals)

        assert profile.external_signals
        signal = profile.external_signals[0]
        assert signal["provider"] == "cross_encoder"
        assert signal["signal_type"] == "retrieval_relevance"
        assert signal["metric_name"] == "cross_encoder_relevance"
        assert signal["affected_chunk_ids"] == ["c1"]
        assert signal["calibration_status"] == "uncalibrated_locally"
        assert signal["recommended_for_gating"] is False
    finally:
        restore()


def test_apply_ignores_non_relevance_signals() -> None:
    """Signals with other signal_types must not touch chunk profiles."""
    from raggov.evaluators.base import ExternalSignalRecord, ExternalSignalType

    run = _make_run([RetrievedChunk(chunk_id="c1", source_doc_id="doc-1", text="t", score=0.8)])
    profiler = RetrievalEvidenceProfilerV0()
    profile = profiler.build(run)
    original_label = profile.chunks[0].query_relevance_label

    non_relevance = ExternalSignalRecord(
        provider=ExternalEvaluatorProvider.cross_encoder,
        signal_type=ExternalSignalType.faithfulness,  # not retrieval_relevance
        metric_name="irrelevant_metric",
        affected_chunk_ids=["c1"],
    )
    profiler.apply_external_relevance_signals(profile, [non_relevance])
    assert profile.chunks[0].query_relevance_label == original_label


# ---------------------------------------------------------------------------
# 7. RetrievalDiagnosisAnalyzer uses external relevance evidence
#    without a hard dependency on sentence-transformers
# ---------------------------------------------------------------------------


def _profile_with_cross_encoder_labels(chunk_configs: list[dict]) -> RetrievalEvidenceProfile:
    """Build a RetrievalEvidenceProfile with cross-encoder method labels (no model needed)."""
    chunks = [
        ChunkEvidenceProfile(
            chunk_id=cfg["chunk_id"],
            source_doc_id=cfg.get("doc_id"),
            query_relevance_label=cfg["label"],
            query_relevance_score=cfg.get("score"),
            relevance_method=cfg.get("method", RelevanceMethod.CROSS_ENCODER),
            evidence_role=cfg.get("role", EvidenceRole.UNKNOWN),
        )
        for cfg in chunk_configs
    ]
    noisy = [
        cp.chunk_id
        for cp in chunks
        if cp.query_relevance_label
        in (QueryRelevanceLabel.IRRELEVANT, QueryRelevanceLabel.PARTIAL)
    ]
    return RetrievalEvidenceProfile(
        run_id="test-run",
        chunks=chunks,
        noisy_chunk_ids=noisy,
        overall_retrieval_status="degraded" if noisy else "ok",
    )


def test_diagnosis_detects_retrieval_noise_from_cross_encoder_labels() -> None:
    """Irrelevant chunks labeled by cross-encoder → RETRIEVAL_NOISE in diagnosis."""
    profile = _profile_with_cross_encoder_labels([
        {"chunk_id": "c1", "doc_id": "doc-1", "label": QueryRelevanceLabel.IRRELEVANT, "score": -2.0},
        {"chunk_id": "c2", "doc_id": "doc-1", "label": QueryRelevanceLabel.IRRELEVANT, "score": -1.5},
        {"chunk_id": "c3", "doc_id": "doc-2", "label": QueryRelevanceLabel.IRRELEVANT, "score": -3.0},
        {"chunk_id": "c4", "doc_id": "doc-2", "label": QueryRelevanceLabel.IRRELEVANT, "score": -2.5},
        {"chunk_id": "c5", "doc_id": "doc-3", "label": QueryRelevanceLabel.IRRELEVANT, "score": -1.8},
    ])
    run = RAGRun(
        run_id="test-run",
        query="eligibility criteria",
        retrieved_chunks=[
            RetrievedChunk(chunk_id=f"c{i}", source_doc_id=f"doc-{i}", text="t", score=0.1)
            for i in range(1, 6)
        ],
        final_answer="answer",
        retrieval_evidence_profile=profile,
    )
    result = RetrievalDiagnosisAnalyzerV0().analyze(run)
    assert result.retrieval_diagnosis_report is not None
    report = result.retrieval_diagnosis_report
    assert report.primary_failure_type == RetrievalFailureType.RETRIEVAL_NOISE
    # Evidence should reference the noisy chunks
    signal_names = [s.signal_name for s in report.evidence_signals]
    assert "noisy_chunk_ids" in signal_names


def test_diagnosis_rank_failure_when_externally_relevant_but_unsupported() -> None:
    """Cross-encoder says chunks are RELEVANT, but claims still unsupported → RANK_FAILURE."""
    profile = _profile_with_cross_encoder_labels([
        {
            "chunk_id": "c1",
            "doc_id": "doc-1",
            "label": QueryRelevanceLabel.RELEVANT,
            "score": 4.5,
            "method": RelevanceMethod.CROSS_ENCODER,
            "role": EvidenceRole.UNKNOWN,
        }
    ])
    unsupported_claim = ClaimEvidenceRecord(
        claim_id="claim-1",
        claim_text="Applicant must be over 18.",
        verification_label=ClaimVerificationLabel.INSUFFICIENT,
        candidate_evidence_chunk_ids=["c1"],
    )
    grounding = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="fail",
        grounding_evidence_bundle={
            "claim_evidence_records": [unsupported_claim]
        },
    )
    sufficiency = AnalyzerResult(
        analyzer_name="SufficiencyAnalyzer",
        status="fail",
        sufficiency_result=SufficiencyResult(
            sufficient=False,
            sufficiency_label="insufficient",
            affected_claims=["claim-1"],
            missing_evidence=["age requirement"],
            method="test",
        ),
    )
    run = RAGRun(
        run_id="test-run",
        query="eligibility criteria",
        retrieved_chunks=[
            RetrievedChunk(chunk_id="c1", source_doc_id="doc-1", text="Applicant must be over 18.", score=0.8)
        ],
        final_answer="answer",
        retrieval_evidence_profile=profile,
    )
    result = RetrievalDiagnosisAnalyzerV0(
        {"prior_results": [grounding, sufficiency]}
    ).analyze(run)
    report = result.retrieval_diagnosis_report
    assert report is not None
    assert report.primary_failure_type == RetrievalFailureType.RANK_FAILURE_UNKNOWN


def test_diagnosis_does_not_import_sentence_transformers() -> None:
    """RetrievalDiagnosisAnalyzerV0 must not hard-import sentence-transformers."""
    import importlib
    import raggov.analyzers.retrieval_diagnosis.retrieval_diagnosis as mod

    source = importlib.util.find_spec(mod.__name__)
    if source and source.origin:
        with open(source.origin) as fh:
            content = fh.read()
        assert "sentence_transformers" not in content, (
            "RetrievalDiagnosisAnalyzerV0 must not import sentence-transformers"
        )


# ---------------------------------------------------------------------------
# 8. No silent fallback
# ---------------------------------------------------------------------------


def test_no_silent_fallback_when_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing dependency must be surfaced; provider must not return empty success."""
    original = sys.modules.pop("sentence_transformers", None)
    try:
        monkeypatch.setattr(
            "raggov.evaluators.retrieval.cross_encoder.importlib.util.find_spec",
            lambda name: None if name == "sentence_transformers" else object(),
        )
        provider = CrossEncoderRetrievalRelevanceProvider()
        result = provider.evaluate(_make_run())
        assert result.succeeded is False
        assert result.missing_dependency is True
        assert result.error is not None and len(result.error) > 0
        assert result.signals == []
    finally:
        if original is not None:
            sys.modules["sentence_transformers"] = original


def test_no_silent_fallback_on_model_error() -> None:
    """Runtime model error must surface in result, not be silently swallowed."""
    mock_module = ModuleType("sentence_transformers")
    mock_model = MagicMock()
    mock_model.predict.side_effect = RuntimeError("GPU out of memory")
    mock_module.CrossEncoder = MagicMock(return_value=mock_model)  # type: ignore[attr-defined]

    original = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = mock_module
    try:
        provider = CrossEncoderRetrievalRelevanceProvider()
        provider._model = mock_model  # bypass _load_model
        result = provider.evaluate(_make_run())
        assert result.succeeded is False
        assert result.missing_dependency is False
        assert result.error is not None
        assert "GPU out of memory" in (result.error or "")
    finally:
        if original is None:
            sys.modules.pop("sentence_transformers", None)
        else:
            sys.modules["sentence_transformers"] = original


# ---------------------------------------------------------------------------
# top_k config respected
# ---------------------------------------------------------------------------


def test_top_k_limits_scored_chunks() -> None:
    mock_model, restore = _mock_sentence_transformers([4.5, -2.0])
    try:
        provider = CrossEncoderRetrievalRelevanceProvider({"top_k": 2})
        provider._model = mock_model
        run = _make_run()  # 3 chunks
        result = provider.evaluate(run)
        assert len(result.signals) == 2
    finally:
        restore()


# ---------------------------------------------------------------------------
# CrossEncoderRelevanceScorer (RetrievalRelevanceScorer protocol)
# ---------------------------------------------------------------------------


def test_scorer_protocol_returns_relevance_score_object() -> None:
    mock_model, restore = _mock_sentence_transformers([4.5])
    try:
        provider = CrossEncoderRetrievalRelevanceProvider()
        provider._model = mock_model
        score_obj = provider.score("test query", "some chunk text")
        assert score_obj.label == QueryRelevanceLabel.RELEVANT
        assert score_obj.method == RelevanceMethod.CROSS_ENCODER.value
        assert score_obj.score == pytest.approx(4.5, abs=0.01)
    finally:
        restore()


def test_scorer_protocol_returns_unknown_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = sys.modules.pop("sentence_transformers", None)
    try:
        monkeypatch.setattr(
            "raggov.evaluators.retrieval.cross_encoder.importlib.util.find_spec",
            lambda name: None if name == "sentence_transformers" else object(),
        )
        provider = CrossEncoderRetrievalRelevanceProvider()
        score_obj = provider.score("test query", "some chunk text")
        assert score_obj.label == QueryRelevanceLabel.UNKNOWN
        assert score_obj.error is not None
    finally:
        if original is not None:
            sys.modules["sentence_transformers"] = original


def test_cross_encoder_scorer_injects_into_profiler() -> None:
    """CrossEncoderRetrievalRelevanceProvider can be used as relevance_scorer in profiler."""
    mock_model, restore = _mock_sentence_transformers([4.5, -2.0])
    try:
        provider = CrossEncoderRetrievalRelevanceProvider()
        provider._model = mock_model

        run = _make_run([
            RetrievedChunk(chunk_id="c1", source_doc_id="doc-1", text="Relevant text.", score=0.9),
            RetrievedChunk(chunk_id="c2", source_doc_id="doc-2", text="Unrelated.", score=0.2),
        ])

        profiler = RetrievalEvidenceProfilerV0(relevance_scorer=provider)
        profile = profiler.build(run)

        chunk_map = {cp.chunk_id: cp for cp in profile.chunks}
        assert chunk_map["c1"].query_relevance_label == QueryRelevanceLabel.RELEVANT
        assert chunk_map["c1"].relevance_method == RelevanceMethod.CROSS_ENCODER
        assert chunk_map["c2"].query_relevance_label == QueryRelevanceLabel.IRRELEVANT
        assert "c2" in profile.noisy_chunk_ids
    finally:
        restore()
