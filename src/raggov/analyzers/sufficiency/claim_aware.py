"""Post-grounding claim-aware sufficiency analyzer."""

from __future__ import annotations

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.diagnosis import AnalyzerResult, ClaimResult, SufficiencyResult
from raggov.models.run import RAGRun


class ClaimAwareSufficiencyAnalyzer(BaseAnalyzer):
    """Compute claim-aware sufficiency from prior grounding claim results."""

    weight = 0.9

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        claim_results = self._prior_claim_results()
        if not claim_results:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="skip",
                evidence=["no grounding claim_results available for claim-aware sufficiency"],
                sufficiency_result=None,
            )

        sufficiency_result = self._claim_aware_sufficiency(claim_results)
        evidence = [
            "Claim-aware sufficiency: "
            f"sufficient={sufficiency_result.sufficient}; "
            f"missing_evidence={len(sufficiency_result.missing_evidence)}; "
            f"affected_claims={len(sufficiency_result.affected_claims)}"
        ]
        return AnalyzerResult(
            analyzer_name=self.name(),
            status="pass",
            evidence=evidence,
            sufficiency_result=sufficiency_result,
        )

    def _prior_claim_results(self) -> list[ClaimResult]:
        prior_results = self.config.get("prior_results", [])
        for result in prior_results:
            if result.analyzer_name != "ClaimGroundingAnalyzer":
                continue
            if result.claim_results:
                return result.claim_results
        return []

    def _claim_aware_sufficiency(self, claim_results: list[ClaimResult]) -> SufficiencyResult:
        missing_evidence: list[str] = []
        affected_claims: list[str] = []
        evidence_chunk_ids: set[str] = set()

        for claim in claim_results:
            if claim.label == "unsupported" and not claim.supporting_chunk_ids:
                missing_evidence.append(claim.claim_text)
                affected_claims.append(claim.claim_text)
            elif claim.label == "contradicted":
                affected_claims.append(claim.claim_text)
            evidence_chunk_ids.update(claim.supporting_chunk_ids)
            evidence_chunk_ids.update(claim.candidate_chunk_ids)
            evidence_chunk_ids.update(claim.contradicting_chunk_ids)

        return SufficiencyResult(
            sufficient=len(missing_evidence) == 0,
            missing_evidence=missing_evidence,
            affected_claims=affected_claims,
            evidence_chunk_ids=sorted(evidence_chunk_ids),
            method="heuristic_claim_aware_v0",
            calibration_status="uncalibrated",
        )
