"""Tests for external signal base types and models."""

from __future__ import annotations

import pytest

from raggov.evaluators.base import (
    ExternalEvaluationResult,
    ExternalEvaluatorProvider,
    ExternalSignalRecord,
    ExternalSignalType,
)


# ---------------------------------------------------------------------------
# Enum values
# ---------------------------------------------------------------------------


def test_provider_enum_values() -> None:
    assert ExternalEvaluatorProvider.structured_llm == "structured_llm"
    assert ExternalEvaluatorProvider.ragas == "ragas"
    assert ExternalEvaluatorProvider.deepeval == "deepeval"
    assert ExternalEvaluatorProvider.ragchecker == "ragchecker"
    assert ExternalEvaluatorProvider.refchecker == "refchecker"
    assert ExternalEvaluatorProvider.cross_encoder == "cross_encoder"
    assert ExternalEvaluatorProvider.nli == "nli"
    assert ExternalEvaluatorProvider.presidio == "presidio"
    assert ExternalEvaluatorProvider.custom == "custom"


def test_signal_type_enum_values() -> None:
    assert ExternalSignalType.claim_support == "claim_support"
    assert ExternalSignalType.citation_support == "citation_support"
    assert ExternalSignalType.retrieval_relevance == "retrieval_relevance"
    assert ExternalSignalType.retrieval_context_precision == "retrieval_context_precision"
    assert ExternalSignalType.retrieval_context_recall == "retrieval_context_recall"
    assert ExternalSignalType.retrieval_contextual_relevancy == "retrieval_contextual_relevancy"
    assert ExternalSignalType.retrieval_contextual_precision == "retrieval_contextual_precision"
    assert ExternalSignalType.claim_recall == "claim_recall"
    assert ExternalSignalType.context_utilization == "context_utilization"
    assert ExternalSignalType.faithfulness == "faithfulness"
    assert ExternalSignalType.hallucination == "hallucination"
    assert ExternalSignalType.uncertainty == "uncertainty"
    assert ExternalSignalType.pii == "pii"
    assert ExternalSignalType.prompt_injection == "prompt_injection"
    assert ExternalSignalType.custom == "custom"


# ---------------------------------------------------------------------------
# ExternalSignalRecord defaults
# ---------------------------------------------------------------------------


def test_signal_record_calibration_status_is_always_uncalibrated() -> None:
    record = ExternalSignalRecord(
        provider=ExternalEvaluatorProvider.ragas,
        signal_type=ExternalSignalType.faithfulness,
        metric_name="ragas_faithfulness",
        value=0.85,
    )
    assert record.calibration_status == "uncalibrated_locally"


def test_signal_record_recommended_for_gating_is_false() -> None:
    record = ExternalSignalRecord(
        provider=ExternalEvaluatorProvider.deepeval,
        signal_type=ExternalSignalType.retrieval_relevance,
        metric_name="deepeval_relevance",
    )
    assert record.recommended_for_gating is False


def test_signal_record_method_type_default() -> None:
    record = ExternalSignalRecord(
        provider=ExternalEvaluatorProvider.custom,
        signal_type=ExternalSignalType.custom,
        metric_name="my_custom_metric",
    )
    assert record.method_type == "external_signal_adapter"


def test_signal_record_list_fields_default_empty() -> None:
    record = ExternalSignalRecord(
        provider=ExternalEvaluatorProvider.nli,
        signal_type=ExternalSignalType.claim_support,
        metric_name="nli_entailment",
    )
    assert record.evidence_ids == []
    assert record.affected_claim_ids == []
    assert record.affected_chunk_ids == []
    assert record.affected_doc_ids == []
    assert record.limitations == []


def test_signal_record_raw_payload_preserved() -> None:
    payload = {"score": 0.9, "model": "nli-deberta", "logits": [0.1, 0.9]}
    record = ExternalSignalRecord(
        provider=ExternalEvaluatorProvider.nli,
        signal_type=ExternalSignalType.claim_support,
        metric_name="nli_entailment",
        raw_payload=payload,
    )
    assert record.raw_payload == payload


def test_signal_record_optional_fields_none_by_default() -> None:
    record = ExternalSignalRecord(
        provider=ExternalEvaluatorProvider.ragas,
        signal_type=ExternalSignalType.faithfulness,
        metric_name="ragas_faithfulness",
    )
    assert record.value is None
    assert record.label is None
    assert record.explanation is None
    assert record.raw_payload is None


def test_signal_record_accepts_bool_and_str_values() -> None:
    bool_record = ExternalSignalRecord(
        provider=ExternalEvaluatorProvider.presidio,
        signal_type=ExternalSignalType.pii,
        metric_name="presidio_pii_detected",
        value=True,
    )
    str_record = ExternalSignalRecord(
        provider=ExternalEvaluatorProvider.custom,
        signal_type=ExternalSignalType.custom,
        metric_name="custom_label",
        value="SUPPORTED",
    )
    assert bool_record.value is True
    assert str_record.value == "SUPPORTED"


# ---------------------------------------------------------------------------
# ExternalEvaluationResult defaults
# ---------------------------------------------------------------------------


def test_evaluation_result_succeeded_true() -> None:
    result = ExternalEvaluationResult(
        provider=ExternalEvaluatorProvider.ragas,
        succeeded=True,
    )
    assert result.succeeded is True
    assert result.missing_dependency is False
    assert result.error is None
    assert result.signals == []
    assert result.raw_payload is None
    assert result.latency_ms is None
    assert result.cost_estimate is None


def test_evaluation_result_missing_dependency() -> None:
    result = ExternalEvaluationResult(
        provider=ExternalEvaluatorProvider.deepeval,
        succeeded=False,
        missing_dependency=True,
        error="deepeval not installed.",
    )
    assert result.succeeded is False
    assert result.missing_dependency is True
    assert "deepeval" in (result.error or "")


def test_evaluation_result_raw_payload_preserved() -> None:
    payload = {"raw_scores": [0.7, 0.8], "version": "0.1.0"}
    result = ExternalEvaluationResult(
        provider=ExternalEvaluatorProvider.ragas,
        succeeded=True,
        raw_payload=payload,
    )
    assert result.raw_payload == payload


def test_evaluation_result_signals_attached() -> None:
    signal = ExternalSignalRecord(
        provider=ExternalEvaluatorProvider.ragas,
        signal_type=ExternalSignalType.faithfulness,
        metric_name="ragas_faithfulness",
        value=0.9,
    )
    result = ExternalEvaluationResult(
        provider=ExternalEvaluatorProvider.ragas,
        succeeded=True,
        signals=[signal],
    )
    assert len(result.signals) == 1
    assert result.signals[0].calibration_status == "uncalibrated_locally"
    assert result.signals[0].recommended_for_gating is False
