"""Analyzer for assessing claim support against retrieved evidence."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

from raggov.analyzers.base import BaseAnalyzer
from raggov.analyzers.grounding.claims import ClaimExtractor
from raggov.analyzers.grounding.evidence_layer import (
    ClaimEvidenceBuilder,
    ClaimEvidenceRecord,
    HeuristicValueOverlapVerifier,
)
from raggov.analyzers.retrieval.scope import STOPWORDS
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import (
    AnalyzerResult,
    ClaimResult,
    FailureStage,
    FailureType,
)
from raggov.models.run import RAGRun


REMEDIATION = (
    "{failed} of {total} claims are unsupported by retrieved context. "
    "Review retrieval quality or add source verification."
)

_CALIBRATION_NOTE = (
    "[Uncalibrated heuristic support score, not calibrated confidence.]"
)


def _record_to_claim_result(record: ClaimEvidenceRecord) -> ClaimResult:
    """Convert a ClaimEvidenceRecord to a ClaimResult for backward compatibility."""
    evidence_reason = f"{record.evidence_reason} {_CALIBRATION_NOTE}"
    return ClaimResult(
        claim_text=record.claim_text,
        label=record.verification_label,
        supporting_chunk_ids=record.supporting_chunk_ids,
        candidate_chunk_ids=[c.chunk_id for c in record.candidate_evidence_chunks],
        contradicting_chunk_ids=record.contradicting_chunk_ids,
        confidence=record.raw_support_score,
        verification_method=record.verification_method,
        evidence_reason=evidence_reason,
        calibration_status=record.calibration_status,
        fallback_used=record.fallback_used,
        value_conflicts=record.value_conflicts,
        value_matches=record.value_matches,
    )


class ClaimGroundingAnalyzer(BaseAnalyzer):
    """Assess whether generated claims are grounded in retrieved chunks."""

    weight = 0.9

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._verifier = HeuristicValueOverlapVerifier(self.config)
        self._builder = ClaimEvidenceBuilder(self._verifier)

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        if not run.retrieved_chunks:
            return self.skip("no retrieved chunks available")

        claim_extractor_client = self.config.get("claim_extractor_client")
        extractor = ClaimExtractor(
            use_llm=claim_extractor_client is not None,
            llm_client=claim_extractor_client,
        )
        claims = extractor.extract(run.final_answer)
        if not claims:
            return self.skip("no claims extracted from final answer")

        claim_results = [
            self._evaluate_claim(claim, index, run.query, run.retrieved_chunks)
            for index, claim in enumerate(claims, start=1)
        ]
        failed_results = [r for r in claim_results if r.label in {"unsupported", "contradicted"}]
        contradicted_results = [r for r in claim_results if r.label == "contradicted"]
        failed_fraction = len(failed_results) / len(claim_results)
        entailed_count = sum(1 for r in claim_results if r.label == "entailed")
        unsupported_count = sum(1 for r in claim_results if r.label == "unsupported")
        contradicted_count = sum(1 for r in claim_results if r.label == "contradicted")
        evidence = [
            (
                "Claim grounding summary: "
                f"total={len(claim_results)}, "
                f"entailed={entailed_count}, "
                f"unsupported={unsupported_count}, "
                f"contradicted={contradicted_count}"
            )
        ]
        fallback_count = sum(1 for r in claim_results if r.fallback_used)
        evidence.append(
            "Claim verification methods: "
            f"structured={len(claim_results) - fallback_count}, "
            f"fallback={fallback_count}"
        )
        remediation = REMEDIATION.format(failed=len(failed_results), total=len(claim_results))

        if failed_fraction >= float(self.config.get("fail_threshold", 0.3)):
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="fail",
                failure_type=FailureType.UNSUPPORTED_CLAIM,
                stage=FailureStage.GROUNDING,
                evidence=evidence,
                claim_results=claim_results,
                remediation=remediation,
            )
        if contradicted_results:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="warn",
                failure_type=FailureType.CONTRADICTED_CLAIM,
                stage=FailureStage.GROUNDING,
                evidence=evidence,
                claim_results=claim_results,
                remediation=remediation,
            )
        if failed_results:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="warn",
                failure_type=FailureType.UNSUPPORTED_CLAIM,
                stage=FailureStage.GROUNDING,
                evidence=evidence,
                claim_results=claim_results,
                remediation=remediation,
            )

        return AnalyzerResult(
            analyzer_name=self.name(),
            status="pass",
            evidence=evidence,
            claim_results=claim_results,
        )

    def _evaluate_claim(
        self,
        claim: str,
        claim_index: int,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> ClaimResult:
        if bool(self.config.get("use_llm", False)) and self.config.get("llm_client") is not None:
            try:
                return self._evaluate_claim_with_llm(claim, chunks)
            except Exception as exc:
                logger.warning(
                    "LLM claim grounding failed, falling back to deterministic: %s", exc
                )
                output = self._verifier._verify_deterministic(claim, chunks)
                evidence_reason = (
                    "LLM verifier unavailable; fell back to deterministic overlap/anchor check "
                    f"{_CALIBRATION_NOTE}"
                )
                return ClaimResult(
                    claim_text=claim,
                    label=output.label,
                    supporting_chunk_ids=output.supporting_chunk_ids,
                    candidate_chunk_ids=output.candidate_chunk_ids,
                    contradicting_chunk_ids=output.contradicting_chunk_ids,
                    confidence=output.raw_support_score,
                    verification_method=output.verification_method,
                    evidence_reason=evidence_reason,
                    calibration_status="uncalibrated",
                    fallback_used=True,
                    value_conflicts=output.value_conflicts,
                    value_matches=output.value_matches,
                )

        record = self._builder._build_single(claim, claim_index, query, chunks)
        return _record_to_claim_result(record)

    def _evaluate_claim_deterministic(
        self, claim: str, chunks: list[RetrievedChunk]
    ) -> ClaimResult:
        """Thin wrapper for backward compatibility. Logic lives in HeuristicValueOverlapVerifier."""
        output = self._verifier._verify_deterministic(claim, chunks)
        evidence_reason = f"{output.evidence_reason} {_CALIBRATION_NOTE}"
        return ClaimResult(
            claim_text=claim,
            label=output.label,
            supporting_chunk_ids=output.supporting_chunk_ids,
            candidate_chunk_ids=output.candidate_chunk_ids,
            contradicting_chunk_ids=output.contradicting_chunk_ids,
            confidence=output.raw_support_score,
            verification_method=output.verification_method,
            evidence_reason=evidence_reason,
            calibration_status="uncalibrated",
            fallback_used=output.fallback_used,
            value_conflicts=output.value_conflicts,
            value_matches=output.value_matches,
        )

    def _evaluate_claim_with_llm(
        self, claim: str, chunks: list[RetrievedChunk]
    ) -> ClaimResult:
        payload = self._call_llm(claim, chunks)
        label = payload.get("label")
        if label not in {"entailed", "unsupported", "contradicted"}:
            raise ValueError("invalid LLM grounding label")
        evidence_chunk_id = payload.get("evidence_chunk_id")
        supporting_chunk_ids = [str(evidence_chunk_id)] if evidence_chunk_id else []
        candidate_chunk_ids = [str(evidence_chunk_id)] if evidence_chunk_id else []
        contradicting_chunk_ids = (
            [str(evidence_chunk_id)]
            if evidence_chunk_id and label == "contradicted"
            else []
        )
        return ClaimResult(
            claim_text=claim,
            label=label,
            supporting_chunk_ids=supporting_chunk_ids if label == "entailed" else [],
            candidate_chunk_ids=candidate_chunk_ids,
            contradicting_chunk_ids=contradicting_chunk_ids,
            confidence=float(payload.get("confidence", 0.0)),
            verification_method="llm_claim_verifier_v1",
            evidence_reason=str(payload.get("rationale", "LLM verifier output")),
            calibration_status="uncalibrated",
            fallback_used=False,
        )

    def _call_llm(self, claim: str, chunks: list[RetrievedChunk]) -> dict[str, Any]:
        client = self.config["llm_client"]
        prompt = self._prompt(claim, chunks)
        if hasattr(client, "chat"):
            response = client.chat(prompt)
        elif hasattr(client, "complete"):
            response = client.complete(prompt)
        else:
            raise TypeError("llm_client must provide chat() or complete()")
        parsed = self._parse_response(response)
        if not isinstance(parsed, dict):
            raise ValueError("LLM grounding response must be a JSON object")
        return parsed

    def _prompt(self, claim: str, chunks: list[RetrievedChunk]) -> str:
        relevant_chunks = "\n\n".join(
            f"[{chunk.chunk_id}] {chunk.text}" for chunk in chunks
        )
        return (
            "Does the following retrieved context support, contradict, or neither "
            "support nor contradict this claim?\n"
            f"Context: {relevant_chunks}\n"
            f"Claim: {claim}\n"
            'Answer with JSON: {"label": "entailed"|"unsupported"|"contradicted", '
            '"confidence": 0.0-1.0, "evidence_chunk_id": "chunk_id or null", '
            '"rationale": "short reason"}'
        )

    def _parse_response(self, response: object) -> Any:
        if isinstance(response, dict):
            if "text" in response:
                response = response["text"]
            elif "content" in response:
                response = response["content"]
            else:
                return response
        if not isinstance(response, str):
            response = str(response)
        return json.loads(response)

    def _tokens(self, text: str) -> list[str]:
        return self._verifier._tokens(text)

    def _terms(self, text: str) -> set[str]:
        return {t for t in self._verifier._tokens(text) if t not in STOPWORDS}
