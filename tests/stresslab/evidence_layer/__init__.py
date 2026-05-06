from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from raggov.engine import DiagnosisEngine
from raggov.evaluators.base import (
    ExternalEvaluationResult,
    ExternalEvaluatorProvider,
)
from raggov.models.corpus import CorpusEntry
from raggov.models.diagnosis import Diagnosis
from raggov.models.run import RAGRun

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "govrag_evidence_30"
DEFAULT_EXTERNAL_PROVIDERS = [
    "ragas",
    "deepeval",
    "refchecker_claim",
    "refchecker_citation",
    "ragchecker",
]


class EvidenceFixtureExpected(BaseModel):
    model_config = ConfigDict(extra="allow")

    primary_failure: str | None = None
    not_clean: bool = True
    required_reports: list[str] = Field(default_factory=list)
    required_evidence_signals: list[str] = Field(default_factory=list)
    expected_missing_external_providers: list[str] = Field(default_factory=list)
    expected_degraded_external_mode: bool = False


class EvidenceFixtureCase(BaseModel):
    model_config = ConfigDict(extra="allow")

    case_id: str
    description: str
    query: str
    retrieved_chunks: list[dict[str, Any]]
    final_answer: str
    citations: list[str] = Field(default_factory=list)
    cited_doc_ids: list[str] = Field(default_factory=list)
    corpus_metadata: dict[str, Any] = Field(default_factory=dict)
    parser_validation_profile: dict[str, Any] | None = None
    mode: str
    expected: EvidenceFixtureExpected
    answer_confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    enabled_external_providers: list[str] | None = None
    retrieval_relevance_provider: str | None = None
    mock_external_results: list[dict[str, Any]] = Field(default_factory=list)

    def to_run(self) -> RAGRun:
        metadata = dict(self.metadata)
        metadata["scenario"] = self.case_id
        metadata["corpus_metadata"] = self.corpus_metadata
        if self.parser_validation_profile is not None:
            metadata["parser_validation_profile"] = self.parser_validation_profile
        query_date = self.corpus_metadata.get("query_date")
        if query_date is not None:
            metadata["query_date"] = query_date
        if self.citations:
            metadata["citations"] = list(self.citations)

        corpus_entries = [
            CorpusEntry.model_validate(entry)
            for entry in self.corpus_metadata.get("entries", [])
        ]

        payload = {
            "run_id": self.case_id,
            "query": self.query,
            "retrieved_chunks": self.retrieved_chunks,
            "final_answer": self.final_answer,
            "cited_doc_ids": self.cited_doc_ids,
            "answer_confidence": self.answer_confidence,
            "corpus_entries": [entry.model_dump(mode="json") for entry in corpus_entries],
            "metadata": metadata,
        }
        return RAGRun.model_validate(payload)


def list_evidence_cases() -> list[str]:
    return sorted(path.stem for path in FIXTURE_DIR.glob("*.json"))


def load_evidence_case(name: str) -> EvidenceFixtureCase:
    fixture_name = name if name.endswith(".json") else f"{name}.json"
    return EvidenceFixtureCase.model_validate_json(
        (FIXTURE_DIR / fixture_name).read_text(encoding="utf-8")
    )


def engine_config_for_case(case: EvidenceFixtureCase) -> dict[str, Any]:
    config: dict[str, Any] = {
        "mode": case.mode,
        "enable_a2p": False,
        "enable_ncv": False,
    }
    if case.mode == "external-enhanced":
        config["enabled_external_providers"] = (
            list(case.enabled_external_providers)
            if case.enabled_external_providers is not None
            else list(DEFAULT_EXTERNAL_PROVIDERS)
        )
        config["retrieval_relevance_provider"] = (
            case.retrieval_relevance_provider or "native"
        )
    return config


def external_results_for_case(
    case: EvidenceFixtureCase,
) -> list[ExternalEvaluationResult]:
    results = [
        ExternalEvaluationResult.model_validate(item)
        for item in case.mock_external_results
    ]
    expected_missing = set(case.expected.expected_missing_external_providers)
    for provider_name in sorted(expected_missing):
        results.append(
            ExternalEvaluationResult(
                provider=_provider_for_name(provider_name),
                adapter_name=provider_name,
                succeeded=False,
                missing_dependency=True,
                error=f"{provider_name}: mocked missing dependency",
            )
        )
    return results


def diagnose_fixture(case: EvidenceFixtureCase) -> Diagnosis:
    run = case.to_run()
    engine = DiagnosisEngine(engine_config_for_case(case))
    mocked_results = external_results_for_case(case)
    engine.external_registry.evaluate_enabled = lambda _run, _enabled, strict_mode=False: mocked_results
    return engine.diagnose(run)


def has_required_report(diagnosis: Diagnosis, report_name: str) -> bool:
    if report_name == "grounding_claim_results":
        return bool(diagnosis.claim_results)
    if report_name == "grounding_evidence_bundle":
        return any(
            result.grounding_evidence_bundle is not None
            for result in diagnosis.analyzer_results
        )
    if report_name == "parser_validation_evidence":
        return any(
            result.analyzer_name == "ParserValidationAnalyzer"
            and bool(result.evidence)
            for result in diagnosis.analyzer_results
        )
    if report_name == "security_evidence":
        return any(
            result.stage is not None and result.stage.value == "SECURITY"
            for result in diagnosis.analyzer_results
        )
    return getattr(diagnosis, report_name, None) is not None


def diagnosis_signal_inventory(diagnosis: Diagnosis) -> set[str]:
    signals: set[str] = set()
    for evidence in diagnosis.evidence:
        signals.add(str(evidence))
    if diagnosis.retrieval_diagnosis_report is not None:
        for signal in diagnosis.retrieval_diagnosis_report.evidence_signals:
            signals.add(signal.signal_name)
        for missing in diagnosis.retrieval_diagnosis_report.missing_reports:
            signals.add(missing)
    if diagnosis.citation_faithfulness_report is not None:
        for record in diagnosis.citation_faithfulness_report.records:
            signals.add(record.citation_support_label.value)
            if record.external_signal_label:
                signals.add(record.external_signal_label)
    if diagnosis.version_validity_report is not None:
        for record in diagnosis.version_validity_report.document_records:
            signals.add(record.validity_status.value)
    return signals


def _provider_for_name(name: str) -> ExternalEvaluatorProvider:
    if "ragas" in name:
        return ExternalEvaluatorProvider.ragas
    if "deepeval" in name:
        return ExternalEvaluatorProvider.deepeval
    if "ragchecker" in name:
        return ExternalEvaluatorProvider.ragchecker
    if "cross_encoder" in name:
        return ExternalEvaluatorProvider.cross_encoder
    if "refchecker" in name:
        return ExternalEvaluatorProvider.refchecker
    if "structured_llm" in name:
        return ExternalEvaluatorProvider.structured_llm
    return ExternalEvaluatorProvider.custom

