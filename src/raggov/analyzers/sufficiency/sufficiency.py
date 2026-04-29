"""Analyzer for whether retrieved evidence is sufficient for an answer."""

from __future__ import annotations

import json
import logging
import re
from typing import Any


logger = logging.getLogger(__name__)

from raggov.analyzers.base import BaseAnalyzer
from raggov.analyzers.retrieval.scope import STOPWORDS
from raggov.models.diagnosis import (
    AnalyzerResult,
    ClaimResult,
    FailureStage,
    FailureType,
    SufficiencyResult,
)
from raggov.models.run import RAGRun


REMEDIATION = (
    "Context does not cover key query terms. Consider expanding retrieval "
    "(increase top-k), broadening the query, or abstaining."
)


class SufficiencyAnalyzer(BaseAnalyzer):
    """Determine whether retrieved context is sufficient to answer the query."""

    weight = 0.9

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        if not run.retrieved_chunks:
            return self.skip("no retrieved chunks available")
        if bool(self.config.get("use_llm", False)) and self.config.get("llm_client") is not None:
            return self._analyze_with_llm(run)
        return self._analyze_deterministic(run)

    def _analyze_deterministic(
        self, run: RAGRun, warning_evidence: list[str] | None = None
    ) -> AnalyzerResult:
        query_terms = self._terms(run.query)
        if not query_terms:
            return self.skip("no meaningful query terms available")

        context_terms = self._terms(" ".join(chunk.text for chunk in run.retrieved_chunks))
        covered_terms = query_terms & context_terms
        missing_terms = sorted(query_terms - covered_terms)
        coverage_ratio = len(covered_terms) / len(query_terms)
        coverage_evidence = self._coverage_evidence(coverage_ratio, missing_terms)
        claim_sidecar = self._claim_aware_sufficiency(run)
        evidence = [*(warning_evidence or []), coverage_evidence]
        if claim_sidecar is not None:
            evidence.append(
                "Claim-aware sufficiency: "
                f"sufficient={claim_sidecar.sufficient}; "
                f"missing_evidence={len(claim_sidecar.missing_evidence)}; "
                f"affected_claims={len(claim_sidecar.affected_claims)}"
            )

        if coverage_ratio < float(self.config.get("min_coverage_ratio", 0.3)):
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="fail",
                failure_type=FailureType.INSUFFICIENT_CONTEXT,
                stage=FailureStage.SUFFICIENCY,
                evidence=evidence,
                sufficiency_result=claim_sidecar,
                remediation=self._abstain_remediation(),
            )

        return AnalyzerResult(
            analyzer_name=self.name(),
            status="pass",
            evidence=evidence,
            sufficiency_result=claim_sidecar,
        )

    def _analyze_with_llm(self, run: RAGRun) -> AnalyzerResult:
        try:
            payload = self._call_llm(run)
            sufficient = bool(payload.get("sufficient", False))
            missing = str(payload.get("missing", "")).strip()
            confidence = float(payload.get("confidence", 0.0))
        except Exception as exc:
            logger.warning("LLM sufficiency judge failed, falling back to deterministic: %s", exc)
            return self._analyze_deterministic(
                run,
                [
                    "LLM sufficiency judge failed; fell back to deterministic mode: "
                    f"{exc}"
                ],
            )

        if not sufficient:
            evidence = [f"LLM judge missing: {missing or 'unspecified'}"]
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="fail",
                failure_type=FailureType.INSUFFICIENT_CONTEXT,
                stage=FailureStage.SUFFICIENCY,
                evidence=evidence,
                sufficiency_result=self._claim_aware_sufficiency(run),
                remediation=self._abstain_remediation(),
            )
        if confidence < 0.6:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="warn",
                failure_type=FailureType.INSUFFICIENT_CONTEXT,
                stage=FailureStage.SUFFICIENCY,
                evidence=[f"LLM judge confidence: {confidence:.2f}"],
                sufficiency_result=self._claim_aware_sufficiency(run),
                remediation=REMEDIATION,
            )

        return AnalyzerResult(
            analyzer_name=self.name(),
            status="pass",
            evidence=[f"LLM judge confidence: {confidence:.2f}"],
            sufficiency_result=self._claim_aware_sufficiency(run),
        )

    def _call_llm(self, run: RAGRun) -> dict[str, Any]:
        client = self.config["llm_client"]
        prompt = self._prompt(run)
        if hasattr(client, "chat"):
            response = client.chat(prompt)
        elif hasattr(client, "complete"):
            response = client.complete(prompt)
        else:
            raise TypeError("llm_client must provide chat() or complete()")

        return self._parse_response(response)

    def _parse_response(self, response: object) -> dict[str, Any]:
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

    def _prompt(self, run: RAGRun) -> str:
        chunks = "\n\n".join(
            f"[{chunk.chunk_id} | {chunk.source_doc_id}] {chunk.text}"
            for chunk in run.retrieved_chunks
        )
        return (
            f"Given this query: {run.query}\n"
            f"And these retrieved passages: {chunks}\n"
            "Does the context contain sufficient information to answer the query "
            'accurately? Answer with JSON: {"sufficient": true/false, '
            '"missing": "what is missing if anything", "confidence": 0.0-1.0}'
        )

    def _coverage_evidence(self, coverage_ratio: float, missing_terms: list[str]) -> str:
        missing = ", ".join(missing_terms) if missing_terms else "none"
        return (
            f"Query term coverage: {coverage_ratio:.0%}. "
            f"Terms not found in context: {missing}"
        )

    def _abstain_remediation(self) -> str:
        return f"{REMEDIATION} should_abstain=True"

    def _claim_aware_sufficiency(self, run: RAGRun) -> SufficiencyResult | None:
        claim_results = self._prior_claim_results()
        if not claim_results:
            return None

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

    def _prior_claim_results(self) -> list[ClaimResult]:
        prior_results = self.config.get("prior_results", [])
        for result in prior_results:
            if result.analyzer_name != "ClaimGroundingAnalyzer":
                continue
            if result.claim_results:
                return result.claim_results
        return []

    def _terms(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in STOPWORDS
        }
