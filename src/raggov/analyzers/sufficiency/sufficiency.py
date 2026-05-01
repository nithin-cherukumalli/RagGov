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
    EvidenceCoverage,
    EvidenceRequirement,
    FailureStage,
    FailureType,
    SufficiencyResult,
)
from raggov.models.run import RAGRun


REMEDIATION = (
    "Context does not cover key query terms. Consider expanding retrieval "
    "(increase top-k), broadening the query, or abstaining."
)

TERM_COVERAGE_LIMITATIONS = [
    "Cannot detect missing evidence when query terms overlap with chunks",
    "Cannot identify required evidence types (rule, date, scope, authority)",
    "Cannot distinguish sufficient context from term overlap",
    "Threshold 0.3 is not empirically calibrated",
    "No verification against ground-truth sufficiency labels",
]


class SufficiencyAnalyzer(BaseAnalyzer):
    """Determine whether retrieved context is sufficient to answer the query.

    Config keys
    -----------
    sufficiency_mode : str
        Default "requirement_aware" has false-pass rate 0.000 on the
        15-example mock-LLM gold set. It falls back to term_coverage
        (false-pass 0.818) when no LLM client is available. See
        data/calibration_report_v1.json for full metrics. This is advisory
        only, not a generation gate.
    min_coverage_ratio : float
        Default 0.3 is not calibrated. Calibration on 15-example gold set found
        best threshold of 0.70 (F1 0.706). Change only after re-running
        calibration (python tools/calibrate_sufficiency.py).
    """

    CALIBRATION_REPORT_PATH = "data/calibration_report_v1.json"
    GOLD_SET_SIZE = 15
    REQUIREMENT_AWARE_FALSE_PASS_RATE = 0.0
    TERM_COVERAGE_FALSE_PASS_RATE = 0.818

    weight = 0.9

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        if not run.retrieved_chunks:
            threshold = float(self.config.get("min_coverage_ratio", 0.3))
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="skip",
                evidence=["no retrieved chunks available"],
                sufficiency_result=self._build_v0_structured_result(
                    run=run,
                    coverage_ratio=0.0,
                    missing_terms=[],
                    threshold=threshold,
                    fallback_used=bool(self.config.get("use_llm", False)),
                ),
            )
        if self.config.get("sufficiency_mode", "requirement_aware") == "requirement_aware":
            return self._analyze_requirement_aware(run)
        if bool(self.config.get("use_llm", False)) and self.config.get("llm_client") is not None:
            return self._analyze_with_llm(run)
        return self._analyze_deterministic(run)

    def _analyze_requirement_aware(self, run: RAGRun) -> AnalyzerResult:
        requirements = self._extract_evidence_requirements(
            run.query,
            self.config.get("doc_type_hint"),
        )
        if not requirements:
            logger.warning("Requirement-aware sufficiency failed; falling back to term coverage")
            result = self._analyze_deterministic(run, fallback_used=True)
            if result.sufficiency_result is not None:
                result.sufficiency_result.limitations.append(
                    "Requirement extraction failed; fell back to term coverage heuristic"
                )
            return result

        coverage = self._check_requirement_coverage_lexical(
            requirements,
            run.retrieved_chunks,
        )
        sufficiency_result = self._build_requirement_aware_result(
            requirements,
            coverage,
        )
        self._log_abstention_recommendation(sufficiency_result)

        status = "fail" if sufficiency_result.sufficiency_label == "insufficient" else "pass"
        return AnalyzerResult(
            analyzer_name=self.name(),
            status=status,
            failure_type=(
                FailureType.INSUFFICIENT_CONTEXT
                if sufficiency_result.sufficiency_label == "insufficient"
                else None
            ),
            stage=(
                FailureStage.SUFFICIENCY
                if sufficiency_result.sufficiency_label == "insufficient"
                else None
            ),
            evidence=[
                "Requirement-aware sufficiency: "
                f"requirements={len(requirements)}; "
                f"missing={sum(1 for item in coverage if item.status == 'missing')}; "
                f"partial={sum(1 for item in coverage if item.status == 'partial')}"
            ],
            sufficiency_result=sufficiency_result,
            remediation=(
                self._abstain_remediation()
                if sufficiency_result.sufficiency_label == "insufficient"
                else None
            ),
        )

    def _analyze_deterministic(
        self,
        run: RAGRun,
        warning_evidence: list[str] | None = None,
        fallback_used: bool = False,
    ) -> AnalyzerResult:
        query_terms = self._terms(run.query)
        if not query_terms:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="skip",
                evidence=["no meaningful query terms available"],
                sufficiency_result=SufficiencyResult(
                    sufficient=False,
                    sufficiency_label="unknown",
                    method="term_coverage_heuristic_v0",
                    calibration_status="preliminary_calibrated_v1",
                    fallback_used=fallback_used,
                    limitations=["Skipped because no meaningful query terms were available."],
                ),
            )

        context_terms = self._terms(" ".join(chunk.text for chunk in run.retrieved_chunks))
        covered_terms = query_terms & context_terms
        missing_terms = sorted(query_terms - covered_terms)
        coverage_ratio = len(covered_terms) / len(query_terms)
        coverage_evidence = self._coverage_evidence(coverage_ratio, missing_terms)
        claim_sidecar = self._claim_sidecar(run)
        threshold = float(self.config.get("min_coverage_ratio", 0.3))
        structured_result = self._build_v0_structured_result(
            run=run,
            coverage_ratio=coverage_ratio,
            missing_terms=missing_terms,
            claim_sidecar=claim_sidecar,
            threshold=threshold,
            fallback_used=fallback_used,
        )
        self._log_abstention_recommendation(structured_result)
        evidence = [*(warning_evidence or []), coverage_evidence]
        if claim_sidecar is not None:
            evidence.append(
                "Claim-aware sufficiency: "
                f"sufficient={not claim_sidecar['missing_evidence']}; "
                f"missing_evidence={len(claim_sidecar['missing_evidence'])}; "
                f"affected_claims={len(claim_sidecar['affected_claims'])}"
            )

        if coverage_ratio < float(self.config.get("min_coverage_ratio", 0.3)):
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="fail",
                failure_type=FailureType.INSUFFICIENT_CONTEXT,
                stage=FailureStage.SUFFICIENCY,
                evidence=evidence,
                sufficiency_result=structured_result,
                remediation=self._abstain_remediation(),
            )

        return AnalyzerResult(
            analyzer_name=self.name(),
            status="pass",
            evidence=evidence,
            sufficiency_result=structured_result,
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
                fallback_used=True,
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

    def _call_llm_prompt(self, prompt: str) -> dict[str, Any]:
        client = self.config["llm_client"]
        if hasattr(client, "chat"):
            response = client.chat(prompt)
        elif hasattr(client, "complete"):
            response = client.complete(prompt)
        else:
            raise TypeError("llm_client must provide chat() or complete()")
        return self._parse_response(response)

    def _extract_evidence_requirements(
        self,
        query: str,
        doc_type_hint: str | None = None,
    ) -> list[EvidenceRequirement]:
        """
        Uses LLM to decompose query into required evidence units.
        Returns structured EvidenceRequirement list.
        Falls back to empty list on LLM failure.
        """
        if self.config.get("llm_client") is None:
            logger.warning("Requirement extraction failed: no llm_client configured")
            return []

        prompt = f"""
You are analyzing a query to a government document RAG system.

Your job: identify what specific pieces of evidence are needed to fully
and accurately answer this query.

Query: {query}

For each required piece of evidence, specify:
1. description: What specific information is needed? Be precise.
2. requirement_type: One of:
   - "definition": Definition of a term or concept
   - "rule": A rule, regulation, or policy statement
   - "date": A specific date, effective date, or deadline
   - "authority": Who issued, who has authority, competent authority
   - "scope": Who or what the rule applies to (applicability)
   - "exception": An exception, exemption, or special case
   - "procedure": Steps, process, or how to do something
   - "numeric_value": A specific number, amount, percentage, or threshold
   - "comparison": How something compares (before/after, this vs that)
   - "supersession": Whether something replaces or is replaced by something else
   - "citation": Specific GO number, circular number, order reference
3. importance:
   - "critical": Cannot answer the query without this
   - "supporting": Helps but answer possible without it
   - "optional": Nice to have, not necessary

Return ONLY valid JSON in this exact format:
{{
  "required_evidence": [
    {{
      "description": "...",
      "type": "rule",
      "importance": "critical"
    }}
  ]
}}

If the query cannot be decomposed (e.g., it's a simple factual lookup with
no structural requirements), return an empty list.
"""
        if doc_type_hint:
            prompt = f"{prompt}\nDocument type hint: {doc_type_hint}\n"

        try:
            payload = self._call_llm_prompt(prompt)
            raw_requirements = payload.get("required_evidence", [])
            if not isinstance(raw_requirements, list):
                return []

            requirements: list[EvidenceRequirement] = []
            for index, item in enumerate(raw_requirements, start=1):
                if not isinstance(item, dict):
                    continue
                requirements.append(
                    EvidenceRequirement(
                        requirement_id=f"req_{index:03d}",
                        description=str(item.get("description", "")).strip(),
                        requirement_type=item.get("type", "scope"),
                        importance=item.get("importance", "critical"),
                        verifier="llm_judge",
                    )
                )
            return [item for item in requirements if item.description]
        except Exception as exc:
            logger.warning("Requirement extraction failed: %s", exc)
            return []

    def _check_requirement_coverage_lexical(
        self,
        requirements: list[EvidenceRequirement],
        chunks: list[Any],
    ) -> list[EvidenceCoverage]:
        """
        Checks each evidence requirement against chunk text using lexical overlap.
        This is a heuristic baseline, NOT a validated verifier.
        """
        coverage: list[EvidenceCoverage] = []

        for requirement in requirements:
            tokens = self._terms(requirement.description)
            best_ratio = 0.0
            best_chunk_ids: list[str] = []
            best_matches: set[str] = set()

            for chunk in chunks:
                chunk_terms = self._terms(chunk.text)
                matches = tokens & chunk_terms
                ratio = len(matches) / len(tokens) if tokens else 0.0
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_chunk_ids = [chunk.chunk_id]
                    best_matches = matches
                elif ratio == best_ratio and ratio > 0:
                    best_chunk_ids.append(chunk.chunk_id)
                    best_matches |= matches

            if best_ratio > 0.50:
                status = "covered"
                supporting_chunk_ids = best_chunk_ids
            elif best_ratio >= 0.25:
                status = "partial"
                supporting_chunk_ids = best_chunk_ids
            else:
                status = "missing"
                supporting_chunk_ids = []

            missed = sorted(tokens - best_matches)
            matched = sorted(best_matches)
            coverage.append(
                EvidenceCoverage(
                    requirement_id=requirement.requirement_id,
                    status=status,
                    supporting_chunk_ids=supporting_chunk_ids,
                    rationale=(
                        f"Matched terms: {', '.join(matched) if matched else 'none'}; "
                        f"missed terms: {', '.join(missed) if missed else 'none'}"
                    ),
                    verifier="heuristic",
                    confidence=best_ratio,
                )
            )

        return coverage

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

    def _build_requirement_aware_result(
        self,
        requirements: list[EvidenceRequirement],
        coverage: list[EvidenceCoverage],
    ) -> SufficiencyResult:
        coverage_by_id = {item.requirement_id: item for item in coverage}
        critical = [
            requirement
            for requirement in requirements
            if requirement.importance == "critical"
        ]
        critical_coverage = [
            coverage_by_id[requirement.requirement_id]
            for requirement in critical
            if requirement.requirement_id in coverage_by_id
        ]

        any_critical_missing = any(item.status == "missing" for item in critical_coverage)
        any_partial = any(item.status == "partial" for item in coverage)
        all_critical_covered = bool(critical_coverage) and all(
            item.status == "covered" for item in critical_coverage
        )

        if any_critical_missing:
            sufficiency_label = "insufficient"
        elif all_critical_covered and any_partial:
            sufficiency_label = "partial"
        elif all_critical_covered:
            sufficiency_label = "unknown"
        elif any_partial:
            sufficiency_label = "partial"
        else:
            sufficiency_label = "unknown"

        return SufficiencyResult(
            sufficient=sufficiency_label != "insufficient",
            sufficiency_label=sufficiency_label,
            required_evidence=requirements,
            coverage=coverage,
            should_abstain=any_critical_missing,
            should_expand_retrieval=any(
                item.status in {"partial", "missing"} for item in coverage
            ),
            method="requirement_extraction_llm_v0",
            calibration_status="preliminary_calibrated_v1",
            limitations=[
                "Evidence requirements extracted by uncalibrated LLM judge",
                "Coverage verified by lexical overlap only (not NLI)",
                "Lexical overlap thresholds (0.5, 0.25) are not empirically calibrated",
                "Cannot detect contradictory or stale evidence",
                "No validation against ground-truth sufficiency labels",
            ],
        )

    def _claim_aware_sufficiency(self, run: RAGRun) -> SufficiencyResult | None:
        claim_results = self._prior_claim_results()
        if not claim_results:
            return None

        sidecar = self._claim_sidecar_from_claim_results(claim_results)
        return self._build_claim_sidecar_result(sidecar)

    def _claim_sidecar(self, run: RAGRun) -> dict[str, Any] | None:
        claim_results = self._prior_claim_results()
        if not claim_results:
            return None
        return self._claim_sidecar_from_claim_results(claim_results)

    def _claim_sidecar_from_claim_results(
        self,
        claim_results: list[ClaimResult],
    ) -> dict[str, Any]:
        missing_evidence: list[str] = []
        affected_claims: list[str] = []
        evidence_chunk_ids: set[str] = set()
        coverage: list[EvidenceCoverage] = []

        for index, claim in enumerate(claim_results, start=1):
            if claim.label == "unsupported" and not claim.supporting_chunk_ids:
                missing_evidence.append(claim.claim_text)
                affected_claims.append(claim.claim_text)
            elif claim.label == "contradicted":
                affected_claims.append(claim.claim_text)
            evidence_chunk_ids.update(claim.supporting_chunk_ids)
            evidence_chunk_ids.update(claim.candidate_chunk_ids)
            evidence_chunk_ids.update(claim.contradicting_chunk_ids)
            coverage.append(self._claim_evidence_coverage(index, claim))

        return {
            "missing_evidence": missing_evidence,
            "affected_claims": affected_claims,
            "evidence_chunk_ids": sorted(evidence_chunk_ids),
            "coverage": coverage,
        }

    def _build_claim_sidecar_result(self, sidecar: dict[str, Any]) -> SufficiencyResult:
        return SufficiencyResult(
            sufficient=len(sidecar["missing_evidence"]) == 0,
            sufficiency_label="unknown",
            coverage=sidecar["coverage"],
            missing_evidence=sidecar["missing_evidence"],
            affected_claims=sidecar["affected_claims"],
            evidence_chunk_ids=sidecar["evidence_chunk_ids"],
            method="heuristic_claim_aware_v0",
            calibration_status="preliminary_calibrated_v1",
            limitations=[
                "Claim-aware sufficiency is derived from grounding sidecar labels and does not independently prove retrieval completeness."
            ],
        )

    def _build_v0_structured_result(
        self,
        run: RAGRun,
        coverage_ratio: float,
        missing_terms: list[str],
        claim_sidecar: dict | None = None,
        threshold: float = 0.3,
        fallback_used: bool = False,
    ) -> SufficiencyResult:
        """
        Maps current v0 heuristic outputs into the new structured schema.
        Does NOT change any existing logic. Only remaps outputs.
        """
        claim_missing = list(claim_sidecar["missing_evidence"]) if claim_sidecar else []
        claim_affected = list(claim_sidecar["affected_claims"]) if claim_sidecar else []
        claim_chunk_ids = list(claim_sidecar["evidence_chunk_ids"]) if claim_sidecar else []
        claim_coverage = list(claim_sidecar["coverage"]) if claim_sidecar else []

        if not run.retrieved_chunks:
            sufficiency_label = "unknown"
        elif coverage_ratio < threshold:
            sufficiency_label = "insufficient"
        elif claim_missing or claim_affected:
            sufficiency_label = "partial"
        else:
            sufficiency_label = "unknown"

        term_status = "missing" if coverage_ratio < threshold else "unknown"
        coverage = [
            EvidenceCoverage(
                requirement_id="term_coverage_v0",
                status=term_status,
                supporting_chunk_ids=[
                    chunk.chunk_id for chunk in run.retrieved_chunks
                ],
                rationale=(
                    f"Query term coverage ratio={coverage_ratio:.3f}; "
                    f"missing_terms={', '.join(missing_terms) if missing_terms else 'none'}"
                ),
                verifier="heuristic",
                confidence=coverage_ratio,
            )
        ]
        coverage.extend(claim_coverage)

        limitations = list(TERM_COVERAGE_LIMITATIONS)
        if not run.retrieved_chunks:
            limitations.append("Skipped because no retrieved chunks were available")
        if claim_sidecar is not None:
            limitations.append(
                "Claim-aware analysis is answer-dependent; cannot detect pre-generation sufficiency gaps"
            )
        if fallback_used:
            limitations.append("LLM judge failed; fell back to deterministic term coverage")

        method = "term_coverage_heuristic_v0"
        if claim_sidecar is not None:
            method = f"{method} + claim_grounding_sidecar_heuristic_v0"

        return SufficiencyResult(
            sufficient=sufficiency_label != "insufficient" and not claim_missing,
            sufficiency_label=sufficiency_label,
            required_evidence=[
                EvidenceRequirement(
                    requirement_id="term_coverage_v0",
                    description="Retrieved context should cover meaningful query terms.",
                    requirement_type="scope",
                    importance="critical",
                    query_span=run.query,
                )
            ],
            coverage=coverage,
            should_expand_retrieval=sufficiency_label == "insufficient",
            should_abstain=sufficiency_label == "insufficient",
            threshold_used=threshold,
            fallback_used=fallback_used,
            limitations=limitations,
            missing_evidence=claim_missing,
            affected_claims=claim_affected,
            evidence_chunk_ids=claim_chunk_ids,
            method=method,
            calibration_status="preliminary_calibrated_v1",
        )

    def _claim_evidence_coverage(
        self,
        index: int,
        claim: ClaimResult,
    ) -> EvidenceCoverage:
        if claim.label == "entailed":
            status = "covered"
        elif claim.label == "contradicted":
            status = "contradicted"
        else:
            status = "missing" if not claim.supporting_chunk_ids else "partial"

        return EvidenceCoverage(
            requirement_id=f"claim_{index}",
            status=status,
            supporting_chunk_ids=claim.supporting_chunk_ids or claim.candidate_chunk_ids,
            contradicting_chunk_ids=claim.contradicting_chunk_ids,
            rationale=claim.evidence_reason or f"Claim grounding label: {claim.label}",
            verifier="heuristic",
            confidence=claim.confidence,
        )

    def _log_abstention_recommendation(
        self,
        sufficiency_result: SufficiencyResult,
    ) -> None:
        if sufficiency_result.should_abstain:
            logger.warning(
                "Sufficiency analyzer recommends abstention "
                "(calibration: preliminary, false-pass rate %.3f). "
                "This is advisory, not enforced.",
                self.REQUIREMENT_AWARE_FALSE_PASS_RATE,
            )

    def _prior_claim_results(self) -> list[ClaimResult]:
        prior_results = self.config.get("prior_results", [])
        for result in prior_results:
            if result.analyzer_name != "ClaimGroundingAnalyzer":
                continue
            if result.claim_results:
                return result.claim_results
        return []

    def check_readiness(self) -> dict:
        """
        Returns readiness status for downstream consumers.
        Does NOT prevent use. Informs callers about calibration quality.
        """
        return {
            "schema_locked": True,
            "harness_locked": True,
            "analyzer_locked": True,
            "calibration_status": "preliminary_calibrated_v1",
            "gold_set_size": self.GOLD_SET_SIZE,
            "gold_set_path": "data/sufficiency_gold_set_v1.jsonl",
            "calibration_report_path": self.CALIBRATION_REPORT_PATH,
            "term_coverage_false_pass_rate": self.TERM_COVERAGE_FALSE_PASS_RATE,
            "requirement_aware_false_pass_rate": self.REQUIREMENT_AWARE_FALSE_PASS_RATE,
            "recommended_for_gating": False,
            "recommended_for_advisory": True,
            "advisory_note": (
                "Advisory use means sufficiency signals should inform but not block "
                "generation. Downstream components should weigh sufficiency alongside "
                "other signals."
            ),
            "limitations": [
                "15-example gold set is small; metrics are directional only",
                "Single labeler; no inter-annotator agreement measured",
                "Lexical coverage is heuristic; NLI verification not yet implemented",
                "LLM judge for requirement extraction is uncalibrated",
                "Cannot detect stale/superseded evidence",
                "Cannot resolve contradictory evidence",
            ],
        }

    def _terms(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in STOPWORDS
        }
