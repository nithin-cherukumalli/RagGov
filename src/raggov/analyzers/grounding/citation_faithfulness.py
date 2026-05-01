"""Citation faithfulness probing based on adversarial reliance signals.

This probe evaluates whether the documents cited by an answer genuinely support
its verifiable claims. It utilizes the claim-level evidence builder to assess support,
serving as a practical claim-evidence citation support probe.
"""

from __future__ import annotations

import logging
from typing import Any

from raggov.analyzers.base import BaseAnalyzer
from raggov.analyzers.grounding.claims import ClaimExtractor
from raggov.analyzers.grounding.evidence_layer import ClaimEvidenceBuilder
from raggov.analyzers.grounding.candidate_selection import EvidenceCandidateSelector
from raggov.analyzers.grounding.verifiers import (
    HeuristicValueOverlapVerifier,
    StructuredLLMClaimVerifier,
)
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.run import RAGRun

logger = logging.getLogger(__name__)

REMEDIATION = (
    "Claim-evidence citation support probe failed: cited documents do not support "
    "the claims they are attached to, or citations are missing for supported claims."
)

CONFIDENT_NO_CITATION_THRESHOLD = 0.7


class CitationFaithfulnessProbe(BaseAnalyzer):
    """Probe whether citations reflect genuine document reliance via claim evidence."""

    weight = 0.85

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._selector = EvidenceCandidateSelector(self.config)
        mode = self.config.get("claim_verifier_mode", "heuristic")
        has_llm_client = bool(self.config.get("llm_client"))
        if mode == "structured_llm" and has_llm_client:
            self._verifier = StructuredLLMClaimVerifier(self.config)
        elif self.config.get("use_llm", False) and has_llm_client:
            self._verifier = StructuredLLMClaimVerifier(self.config)
        else:
            self._verifier = HeuristicValueOverlapVerifier(self.config)
        self._builder = ClaimEvidenceBuilder(self._verifier, self._selector)
        
        claim_extractor_client = self.config.get("claim_extractor_client")
        self._extractor = ClaimExtractor(
            use_llm=claim_extractor_client is not None,
            llm_client=claim_extractor_client,
        )

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        if not run.final_answer or not run.final_answer.strip():
            return self.skip("no final answer to probe")
        if not run.retrieved_chunks:
            return self.skip("no retrieved chunks available")

        suspicious_threshold = int(self.config.get("suspicious_threshold", 2))

        claims = self._extractor.extract(run.final_answer)
        if not claims:
            return self.skip("no claims extracted from final answer")

        has_citations = bool(run.cited_doc_ids)
        if not has_citations and not self._is_confident(run):
            return self.skip("no citations to verify and answer confidence is below threshold")

        chunks_by_id = {c.chunk_id: c for c in run.retrieved_chunks}
        probe_results: list[dict[str, Any]] = []
        evidence: list[str] = []
        fail_claims = 0
        warn_claims = 0
        
        for index, claim in enumerate(claims, start=1):
            record = self._builder._build_single(claim, index, run.query, run.retrieved_chunks)
            
            supporting_docs = {
                chunks_by_id[chunk_id].source_doc_id
                for chunk_id in record.supporting_chunk_ids
                if chunk_id in chunks_by_id
            }
            
            status = "unknown"
            reason = record.evidence_reason
            
            if record.verification_label == "entailed":
                if has_citations:
                    if any(doc_id in run.cited_doc_ids for doc_id in supporting_docs):
                        status = "citation_supported"
                    else:
                        status = "citation_mismatch"
                        reason = "Claim is supported by retrieved context, but not by any cited document."
                else:
                    status = "citation_missing"
                    reason = "Claim is confidently supported by retrieved context, but no citations are provided."
            else:
                if has_citations:
                    status = "unsupported_cited_claim"
                    reason = f"Claim is {record.verification_label} but the answer provides citations."
                else:
                    status = "unsupported_no_citation"
                    reason = f"Claim is {record.verification_label} and answer has no citations."
                    
            probe_results.append({
                "claim_text": claim,
                "cited_doc_ids": run.cited_doc_ids,
                "supporting_doc_id": list(supporting_docs)[0] if supporting_docs else None,
                "verification_label": record.verification_label,
                "status": status,
                "reason": reason,
                "fallback_used": record.fallback_used,
            })
            
            if status in ("citation_mismatch", "unsupported_cited_claim"):
                fail_claims += 1
                evidence.append(
                    f"Failed citation: claim='{claim[:50]}...', status={status}, "
                    f"label={record.verification_label}, reason={reason}"
                )
            elif status == "citation_missing":
                warn_claims += 1
                evidence.append(
                    f"Missing citation: claim='{claim[:50]}...', status={status}, "
                    f"label={record.verification_label}"
                )

        if not has_citations and self._is_confident(run) and not fail_claims and not warn_claims and not probe_results:
             return self._fail_post_rationalized(
                [
                    "Answer-without-citation probe failed: "
                    f"confident answer ({run.answer_confidence or 0.0:.2f}) with no citations provided"
                ],
                probe_results=[],
             )

        if fail_claims > 0 or warn_claims >= suspicious_threshold:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="fail",
                failure_type=FailureType.POST_RATIONALIZED_CITATION,
                stage=FailureStage.GROUNDING,
                evidence=evidence,
                remediation=REMEDIATION,
                citation_probe_results=probe_results,
            )
            
        if warn_claims > 0:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="warn",
                failure_type=FailureType.POST_RATIONALIZED_CITATION,
                stage=FailureStage.GROUNDING,
                evidence=evidence,
                remediation=REMEDIATION,
                citation_probe_results=probe_results,
            )

        if not has_citations and all(p["status"] == "unsupported_no_citation" for p in probe_results):
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="pass",
                evidence=["Answer has no citations and claims are unsupported. No post-rationalization found."],
                citation_probe_results=probe_results,
            )

        return AnalyzerResult(
            analyzer_name=self.name(),
            status="pass",
            evidence=["Claim-evidence citation support probe indicates genuine document reliance."],
            citation_probe_results=probe_results,
        )

    def _is_confident(self, run: RAGRun) -> bool:
        return (
            run.answer_confidence is not None
            and run.answer_confidence >= CONFIDENT_NO_CITATION_THRESHOLD
        )

    def _fail_post_rationalized(
        self, evidence: list[str], probe_results: list[dict[str, Any]]
    ) -> AnalyzerResult:
        return AnalyzerResult(
            analyzer_name=self.name(),
            status="fail",
            failure_type=FailureType.POST_RATIONALIZED_CITATION,
            stage=FailureStage.GROUNDING,
            evidence=evidence,
            remediation=REMEDIATION,
            citation_probe_results=probe_results,
        )
