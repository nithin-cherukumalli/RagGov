"""Analyzer for assessing claim support against retrieved evidence."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

from raggov.analyzers.base import BaseAnalyzer
from raggov.analyzers.grounding.claims import ClaimExtractor
from raggov.analyzers.grounding.diagnostic_rollups import ClaimDiagnosticRollupBuilder, ClaimDiagnosticSummary
from raggov.analyzers.grounding.evidence_layer import ClaimEvidenceBuilder, ClaimEvidenceRecord
from raggov.models.grounding import GroundingEvidenceBundle
from raggov.calibration import ClaimCalibrationLoader
from raggov.analyzers.grounding.triplets import build_triplet_extractor
from raggov.analyzers.grounding.verifiers import (
    EvidenceVerifier,
    HeuristicValueOverlapVerifier,
    StructuredLLMClaimVerifier,
    TripletVerifier,
    LLMTripletVerifierV1,
)
from raggov.analyzers.grounding.candidate_selection import EvidenceCandidateSelector
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
    # Only append the uncalibrated note if status is actually uncalibrated
    evidence_reason = record.evidence_reason
    if record.calibration_status == "uncalibrated":
        evidence_reason = f"{evidence_reason} {_CALIBRATION_NOTE}"
        
    return ClaimResult(
        claim_text=record.claim_text,
        label=record.verification_label,
        supporting_chunk_ids=record.supporting_chunk_ids,
        candidate_chunk_ids=[c.chunk_id for c in record.candidate_evidence_chunks],
        contradicting_chunk_ids=record.contradicting_chunk_ids,
        confidence=record.calibrated_confidence if record.calibrated_confidence is not None else record.verifier_score,
        verification_method=record.verifier_method,
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
        self._selector = EvidenceCandidateSelector(self.config)
        mode = self.config.get("claim_verifier_mode", "heuristic")
        has_llm_client = bool(self.config.get("llm_client"))
        if mode == "structured_llm" and has_llm_client:
            self._verifier = StructuredLLMClaimVerifier(self.config)
        elif self.config.get("use_llm", False) and has_llm_client:
            # Only use LLM verifier when an actual llm_client is provided
            self._verifier = StructuredLLMClaimVerifier(self.config)
        else:
            self._verifier = HeuristicValueOverlapVerifier(self.config)
        
        # Calibration setup
        calib_path = self.config.get("claim_calibration_path")
        self._calibrator = ClaimCalibrationLoader.load(calib_path) if calib_path else None

        self._builder = ClaimEvidenceBuilder(
            self._verifier,
            self._selector,
            triplet_extractor=build_triplet_extractor(self.config),
            calibrator=self._calibrator,
        )
        
        # Triplet verification setup
        if self.config.get("enable_triplet_verification", False):
            # Only LLM triplet verifier is supported for now
            triplet_verifier = LLMTripletVerifierV1(self.config)
            self._builder.set_triplet_verifier(triplet_verifier)
            logger.info("Triplet-level verification enabled.")

        self._rollup_builder = ClaimDiagnosticRollupBuilder(self.config)

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

        # Build ClaimEvidenceRecords (richer than ClaimResult — needed for rollup)
        records: list[ClaimEvidenceRecord] = [
            self._builder._build_single(claim, index, run.query, run.retrieved_chunks)
            for index, claim in enumerate(claims, start=1)
        ]
        claim_results = [_record_to_claim_result(r) for r in records]

        # --- Diagnostic rollup --------------------------------------------
        cited_doc_ids: list[str] = list(getattr(run, "cited_doc_ids", None) or [])
        rollup: ClaimDiagnosticSummary = self._rollup_builder.build(
            records=records,
            retrieved_chunks=run.retrieved_chunks,
            cited_doc_ids=cited_doc_ids,
        )

        # Create the evidence bundle for downstream consumers
        bundle = GroundingEvidenceBundle(
            claim_evidence_records=records,
            diagnostic_rollup=rollup.as_dict(),
            citation_support_summary={
                "citation_support_rate": rollup.citation_support_rate,
                "citation_mismatch_suspected_count": rollup.citation_mismatch_suspected_count,
            },
            calibration_summary={
                "calibration_status": records[0].calibration_status if records else "unavailable",
                "average_score": sum(r.verifier_score for r in records) / len(records) if records else 0.0,
            },
        )

        failed_results = [r for r in claim_results if r.label in {"unsupported", "contradicted"}]
        contradicted_results = [r for r in claim_results if r.label == "contradicted"]
        failed_fraction = len(failed_results) / len(claim_results)
        entailed_count = rollup.entailed_claims
        unsupported_count = rollup.unsupported_claims
        contradicted_count = rollup.contradicted_claims

        evidence = [
            (
                "Claim grounding summary: "
                f"total={rollup.total_claims}, "
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
        # Append RAGChecker-inspired failure pattern summary
        pattern_line = (
            f"Diagnostic patterns [{rollup.diagnostic_version}]: "
            f"{rollup.failure_pattern_summary()}"
        )
        evidence.append(pattern_line)
        if rollup.noisy_context_suspected:
            evidence.append(
                f"Noisy retrieval suspected: evidence_utilization_rate="
                f"{rollup.evidence_utilization_rate:.2f}"
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
                diagnostic_rollup=rollup.as_dict(),
                grounding_evidence_bundle=bundle,
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
                diagnostic_rollup=rollup.as_dict(),
                grounding_evidence_bundle=bundle,
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
                diagnostic_rollup=rollup.as_dict(),
                grounding_evidence_bundle=bundle,
            )

        return AnalyzerResult(
            analyzer_name=self.name(),
            status="pass",
            evidence=evidence,
            claim_results=claim_results,
            diagnostic_rollup=rollup.as_dict(),
            grounding_evidence_bundle=bundle,
        )

    def _evaluate_claim(
        self,
        claim: str,
        claim_index: int,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> ClaimResult:
        record = self._builder._build_single(claim, claim_index, query, chunks)
        return _record_to_claim_result(record)
