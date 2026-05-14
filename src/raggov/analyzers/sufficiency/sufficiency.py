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
    "Cannot identify all required generic evidence types (entity, value, date/time, scope, exception, procedure)",
    "Cannot distinguish sufficient context from term overlap",
    "Threshold 0.3 is not empirically calibrated",
    "No verification against ground-truth sufficiency labels",
]

# --- Structured sufficiency detection constants ---

_CURRENT_QUERY_TERMS = frozenset({
    "current", "latest", "now", "today", "recent", "newest", "present",
})

_TEMPORAL_REQUIREMENT_TERMS = frozenset({
    "current", "latest", "effective", "updated", "version", "versions",
    "before", "after", "since", "until", "during", "when",
})

_STALE_CHUNK_PATTERN = re.compile(
    r"\bas of \d{4}\b|\bsince \d{4}\b|\bdated? \d{4}\b|\b\(\s*\d{4}\s*\)",
    re.IGNORECASE,
)

_TEMPORAL_CONTEXT_PATTERN = re.compile(
    r"\b(?:19|20)\d{2}\b|\b(?:effective|updated|version|rev\.?|revision|release)\b",
    re.IGNORECASE,
)

_CRITICAL_QUERY_TERMS = frozenset({
    "safe", "safety", "dangerous", "danger", "hazard", "hazardous",
    "toxic", "poison", "poisonous", "risk", "risky",
})

_UNIVERSAL_QUERY_PATTERN = re.compile(
    r"\b(?:can|do|does|is|are|must|should)?\s*(all|every|any|everyone)\s+(\w+)\b",
    re.IGNORECASE,
)

_SCOPE_QUALIFIER_TERMS = frozenset({
    "full-time", "fulltime", "full_time", "part-time", "parttime",
    "contract", "permanent", "temporary", "senior", "junior",
    "certain", "specific", "only", "eligible", "qualified", "authorized",
    "washable", "standard", "premium", "legacy", "beta", "experimental",
})

_US_STATES = frozenset({
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming",
})

_CONDITIONAL_PATTERN = re.compile(
    r"\bif\s+(?:(?:i|you|we|they)\s+)?(\w+)\b",
    re.IGNORECASE,
)

_CONDITIONAL_SKIP_TERMS = frozenset({
    "need", "want", "can", "could", "have", "had", "get", "do",
    "would", "should", "must", "might", "may", "will", "shall",
    "be", "is", "are", "was", "were", "has", "not", "no", "any", "more",
}) | STOPWORDS

_SCOPE_PREP_PATTERN = re.compile(r"\bin ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")


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
        # Structured pattern detection takes precedence over term coverage heuristic.
        # These patterns detect specific sufficiency failure modes that the term coverage
        # heuristic cannot distinguish from a general retrieval miss.
        structured = self._detect_structured_sufficiency_type(run)
        if structured is not None:
            sufficiency_result = self._build_structured_sufficiency_result(
                reason=structured["reason"],
                fix_category=structured["fix_category"],
                evidence_markers=structured.get("evidence_markers"),
                missing_requirement_types=structured.get("missing_requirement_types"),
                missing_evidence=structured.get("missing_evidence"),
            )
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="fail",
                failure_type=structured["failure_type"],
                stage=structured["stage"],
                evidence=[structured["evidence"]],
                sufficiency_result=sufficiency_result,
                remediation=structured["remediation"],
            )

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
        missing_anchors = self._missing_query_anchors(run)
        claim_sidecar = self._claim_sidecar(run)
        threshold = float(self.config.get("min_coverage_ratio", 0.3))
        effective_coverage_ratio = 0.0 if missing_anchors else coverage_ratio
        structured_result = self._build_v0_structured_result(
            run=run,
            coverage_ratio=effective_coverage_ratio,
            missing_terms=sorted(set(missing_terms + missing_anchors)),
            claim_sidecar=claim_sidecar,
            threshold=threshold,
            fallback_used=fallback_used,
        )
        self._log_abstention_recommendation(structured_result)
        evidence = [*(warning_evidence or []), coverage_evidence]
        if missing_anchors:
            evidence.append(
                "Critical query anchor missing from retrieved context: "
                f"{', '.join(missing_anchors)}"
            )
        if claim_sidecar is not None:
            evidence.append(
                "Claim-aware sufficiency: "
                f"sufficient={not claim_sidecar['missing_evidence']}; "
                f"missing_evidence={len(claim_sidecar['missing_evidence'])}; "
                f"affected_claims={len(claim_sidecar['affected_claims'])}"
            )

        missing_answer_requirements = self._missing_answer_requirements(run)
        if missing_answer_requirements and not missing_anchors:
            evidence.append(
                "Answer completeness missing retrieved query requirements: "
                f"{', '.join(missing_answer_requirements)}"
            )
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="fail",
                failure_type=FailureType.INSUFFICIENT_CONTEXT,
                stage=FailureStage.GENERATION,
                evidence=evidence,
                sufficiency_result=structured_result,
                remediation="Regenerate the answer to cover all retrieved requirements requested by the query.",
            )

        if missing_anchors or coverage_ratio < float(self.config.get("min_coverage_ratio", 0.3)):
            critical_detection = self._critical_requirement_detection(
                run=run,
                coverage_ratio=coverage_ratio,
                missing_terms=sorted(set(missing_terms + missing_anchors)),
            )
            if critical_detection is not None:
                sufficiency_result = self._build_structured_sufficiency_result(
                    reason=critical_detection["reason"],
                    fix_category=critical_detection["fix_category"],
                    evidence_markers=critical_detection["evidence_markers"],
                    missing_requirement_types=critical_detection["missing_requirement_types"],
                    missing_evidence=critical_detection["missing_evidence"],
                )
                return AnalyzerResult(
                    analyzer_name=self.name(),
                    status="fail",
                    failure_type=critical_detection["failure_type"],
                    stage=critical_detection["stage"],
                    evidence=[critical_detection["evidence"], *evidence],
                    sufficiency_result=sufficiency_result,
                    remediation=critical_detection["remediation"],
                )

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

    def _missing_query_anchors(self, run: RAGRun) -> list[str]:
        context_lower = " ".join(chunk.text for chunk in run.retrieved_chunks).lower()
        anchors: list[str] = []
        for match in re.finditer(r"\b(Project|Tool|Product)\s+[A-Z0-9]\b", run.query):
            anchor = match.group(0).lower()
            if anchor not in context_lower:
                anchors.append(anchor)
        return list(dict.fromkeys(anchors))

    def _missing_answer_requirements(self, run: RAGRun) -> list[str]:
        query_lower = run.query.lower()
        if not re.search(r"\b(list|requirements?|include)\b", query_lower):
            return []

        context_terms = self._terms(" ".join(chunk.text for chunk in run.retrieved_chunks))
        answer_terms = self._terms(run.final_answer)
        candidate_terms: set[str] = set()

        colon_tail = query_lower.split(":", 1)[1] if ":" in query_lower else query_lower
        for token in self._terms(colon_tail):
            if token in {"list", "requirement", "requirements", "include"}:
                continue
            candidate_terms.add(token)

        missing = sorted(
            term
            for term in candidate_terms
            if term in context_terms and term not in answer_terms
        )
        return missing

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
You are analyzing a query to a generic RAG system.

Your job: identify what specific pieces of evidence are needed to fully
and accurately answer this query.

Query: {query}

For each required piece of evidence, specify:
1. description: What specific information is needed? Be precise.
2. requirement_type: One of:
   - "required_entity": A person, product, API, organization, condition group, study, source, or other named entity
   - "required_value": A number, amount, percentage, currency, measurement, version, identifier, quantity, range, or rank
   - "required_date_or_time": A date, deadline, effective date, expiry date, duration, timestamp, or time window
   - "required_condition_or_scope": The population, environment, version, jurisdiction, configuration, scenario, or applicability scope
   - "required_exception_or_limitation": An exception, contraindication, limitation, exclusion, caveat, or special case
   - "required_comparison_baseline": The baseline needed for before/after, alternative, control, benchmark, or relative comparison
   - "required_step_or_procedure": Steps, process, workflow, installation, usage, clinical, operational, or support procedure
   - "required_causal_support": Evidence for a cause, reason, mechanism, effect, or conclusion
   - "required_source_or_citation": A source, citation, document, section, table, figure, record, or source identifier
3. importance:
   - "critical": Cannot answer the query without this
   - "supporting": Helps but answer possible without it
   - "optional": Nice to have, not necessary

Return ONLY valid JSON in this exact format:
{{
  "required_evidence": [
    {{
      "description": "...",
      "type": "required_condition_or_scope",
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
                        requirement_type=self._normalize_requirement_type(item.get("type", "required_condition_or_scope")),
                        importance=item.get("importance", "critical"),
                        verifier="llm_judge",
                    )
                )
            return [item for item in requirements if item.description]
        except Exception as exc:
            logger.warning("Requirement extraction failed: %s", exc)
            return []

    def _normalize_requirement_type(self, raw_type: Any) -> str:
        value = str(raw_type or "").strip()
        legacy_map = {
            "definition": "required_entity",
            "rule": "required_condition_or_scope",
            "date": "required_date_or_time",
            "authority": "required_source_or_citation",
            "scope": "required_condition_or_scope",
            "exception": "required_exception_or_limitation",
            "procedure": "required_step_or_procedure",
            "numeric_value": "required_value",
            "comparison": "required_comparison_baseline",
            "supersession": "required_source_or_citation",
            "citation": "required_source_or_citation",
        }
        allowed = {
            "required_entity",
            "required_value",
            "required_date_or_time",
            "required_condition_or_scope",
            "required_exception_or_limitation",
            "required_comparison_baseline",
            "required_step_or_procedure",
            "required_causal_support",
            "required_source_or_citation",
        }
        return legacy_map.get(value, value if value in allowed else "required_condition_or_scope")

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
                    requirement_type="required_condition_or_scope",
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

    def _detect_structured_sufficiency_type(
        self,
        run: RAGRun,
    ) -> dict[str, Any] | None:
        """Detect specific structured sufficiency failure patterns.

        Returns a dict with reason, failure_type, stage, evidence, remediation
        when a pattern is found. Returns None otherwise.

        Patterns detected (in priority order):
        1. stale_context_mistaken_as_sufficient – query asks for "current" info
           but chunk has a temporal qualifier like "as of 2020".
        2. missing_temporal_or_freshness_requirement – query asks for a temporal,
           effective-date, or version-specific answer but the retrieved context
           lacks temporal qualifiers entirely.
        3. partial_requirement_coverage – query has a conditional clause whose
           condition term is absent from the retrieved context.
        4. missing_exception – query uses a universal quantifier ("all X") but
           the context provides a scoped/qualified answer.
        5. missing_scope_condition – context is scoped to a specific location
           (US state) but the query has no such scope.
        """
        context_text = " ".join(chunk.text for chunk in run.retrieved_chunks)
        query_lower = run.query.lower()
        context_lower = context_text.lower()

        # --- Pattern 1: stale context ---
        query_words = set(re.findall(r"\b\w+\b", query_lower))
        if query_words & _CURRENT_QUERY_TERMS:
            stale_matches = _STALE_CHUNK_PATTERN.findall(context_text)
            if stale_matches:
                return {
                    "reason": "stale_context_mistaken_as_sufficient",
                    "failure_type": FailureType.STALE_RETRIEVAL,
                    "stage": FailureStage.RETRIEVAL,
                    "evidence": (
                        "[sufficiency:stale_context_mistaken_as_sufficient] "
                        "[sufficiency:missing_temporal_or_freshness_requirement] "
                        "Query requests current information but retrieved chunk "
                        f"contains temporal qualifier: {stale_matches[0]!r}"
                    ),
                    "remediation": (
                        "Apply freshness filtering to retrieval; do not use "
                        "outdated sources for time-sensitive queries."
                    ),
                    "fix_category": "FRESHNESS_FILTER",
                    "evidence_markers": [
                        "[sufficiency:stale_context_mistaken_as_sufficient]",
                        "[sufficiency:missing_temporal_or_freshness_requirement]",
                    ],
                    "missing_requirement_types": [
                        "required_condition_or_scope",
                        "required_date_or_time",
                    ],
                    "missing_evidence": [
                        "Current or fresh source evidence required by the query is missing.",
                    ],
                }

        # --- Pattern 2: temporal/freshness requirement missing entirely ---
        query_words = set(re.findall(r"\b\w+\b", query_lower))
        query_has_year = bool(re.search(r"\b(?:19|20)\d{2}\b", run.query))
        query_has_temporal_requirement = bool(query_words & _TEMPORAL_REQUIREMENT_TERMS) or query_has_year
        context_has_temporal_support = bool(_TEMPORAL_CONTEXT_PATTERN.search(context_text))
        if query_has_temporal_requirement and not context_has_temporal_support:
            return {
                "reason": "missing_temporal_or_freshness_requirement",
                "failure_type": FailureType.INSUFFICIENT_CONTEXT,
                "stage": FailureStage.SUFFICIENCY,
                "evidence": (
                    "[sufficiency:missing_temporal_or_freshness_requirement] "
                    "Query requires temporal, effective-date, or version-specific "
                    "support, but retrieved context provides no temporal qualifier "
                    "that can validate the requested time scope."
                ),
                "remediation": (
                    "Expand retrieval to include effective dates, version metadata, "
                    "or current lifecycle state before answering."
                ),
                "fix_category": "COVERAGE_EXPANSION",
                "evidence_markers": [
                    "[sufficiency:missing_temporal_or_freshness_requirement]",
                ],
                "missing_requirement_types": [
                    "required_condition_or_scope",
                    "required_date_or_time",
                ],
                "missing_evidence": [
                    "Temporal or freshness support required by the query is missing.",
                ],
            }

        # --- Pattern 3: conditional clause not addressed ---
        cond_match = _CONDITIONAL_PATTERN.search(run.query)
        if cond_match:
            cond_term = cond_match.group(1).lower()
            if (
                cond_term not in _CONDITIONAL_SKIP_TERMS
                and len(cond_term) > 3
                and cond_term not in context_lower
            ):
                return {
                    "reason": "partial_requirement_coverage",
                    "failure_type": FailureType.INSUFFICIENT_CONTEXT,
                    "stage": FailureStage.SUFFICIENCY,
                    "evidence": (
                        "[sufficiency:partial_requirement_coverage] "
                        f"Query addresses a conditional case ('{cond_term}') that "
                        "is absent from the retrieved context. Only the default "
                        "case is covered."
                    ),
                    "remediation": (
                        "Expand retrieval to cover the conditional scenario; "
                        "consider abstaining if the case is not documented."
                    ),
                    "fix_category": "ABSTENTION_THRESHOLD",
                    "evidence_markers": [
                        "[sufficiency:partial_requirement_coverage]",
                    ],
                    "missing_requirement_types": [
                        "required_condition_or_scope",
                    ],
                    "missing_evidence": [
                        f"Conditional scenario '{cond_term}' is not covered in retrieved context.",
                    ],
                }

        # --- Pattern 4: universal quantifier with scoped context ---
        univ_match = _UNIVERSAL_QUERY_PATTERN.search(run.query)
        if univ_match:
            subject_noun = univ_match.group(2).lower()
            if subject_noun in context_lower:
                context_words = set(re.findall(r"\b[\w-]+\b", context_lower))
                if context_words & _SCOPE_QUALIFIER_TERMS:
                    return {
                        "reason": "missing_exception",
                        "failure_type": FailureType.INSUFFICIENT_CONTEXT,
                        "stage": FailureStage.SUFFICIENCY,
                        "evidence": (
                            "[sufficiency:missing_exception] "
                            f"Query asks about '{univ_match.group(0)}' universally "
                            "but the retrieved context provides a scoped/qualified "
                            "answer. Non-qualifying cases and exceptions are not covered."
                        ),
                        "remediation": (
                            "Expand retrieval to cover all subcategories or confirm "
                            "the universal claim; consider abstaining if the universal "
                            "answer cannot be confirmed."
                        ),
                        "fix_category": "COVERAGE_EXPANSION",
                        "evidence_markers": [
                            "[sufficiency:missing_exception]",
                        ],
                        "missing_requirement_types": [
                            "required_exception_or_limitation",
                            "required_condition_or_scope",
                        ],
                        "missing_evidence": [
                            "Exceptions or non-qualifying cases are not covered in retrieved context.",
                        ],
                    }

        # --- Pattern 5: scope-specific context for scope-general query ---
        scope_matches = _SCOPE_PREP_PATTERN.findall(context_text)
        for location in scope_matches:
            loc_lower = location.lower()
            if loc_lower in _US_STATES and loc_lower not in query_lower:
                return {
                    "reason": "missing_scope_condition",
                    "failure_type": FailureType.INSUFFICIENT_CONTEXT,
                    "stage": FailureStage.SUFFICIENCY,
                    "evidence": (
                        "[sufficiency:missing_scope_condition] "
                        f"Retrieved context is scoped to '{location}' but the query "
                        "has no explicit scope. The answer is only valid for "
                        f"'{location}', not universally."
                    ),
                    "remediation": (
                        "Clarify geographic or organizational scope in the answer, "
                        "or re-query with explicit scope; consider abstaining for "
                        "universal queries."
                    ),
                    "fix_category": "SCOPE_DISAMBIGUATION",
                    "evidence_markers": [
                        "[sufficiency:missing_scope_condition]",
                    ],
                    "missing_requirement_types": [
                        "required_condition_or_scope",
                    ],
                    "missing_evidence": [
                        f"Applicability scope for '{location}' does not justify a scope-free answer.",
                    ],
                }

        return None

    def _critical_requirement_detection(
        self,
        run: RAGRun,
        coverage_ratio: float,
        missing_terms: list[str],
    ) -> dict[str, Any] | None:
        """Return a structured critical-value sufficiency failure when appropriate."""
        query_words = set(re.findall(r"\b\w+\b", run.query.lower()))
        if not (query_words & _CRITICAL_QUERY_TERMS):
            return None

        return {
            "reason": "missing_critical_requirement",
            "failure_type": FailureType.INSUFFICIENT_CONTEXT,
            "stage": FailureStage.SUFFICIENCY,
            "evidence": (
                "[sufficiency:missing_critical_requirement] "
                "Query requests safety-critical guidance, but retrieved context "
                "does not provide enough evidence to support a safe answer. "
                f"Missing support terms: {', '.join(missing_terms) if missing_terms else 'unspecified'}; "
                f"coverage={coverage_ratio:.0%}."
            ),
            "remediation": (
                "Require explicit critical-value verification before answering; "
                "expand retrieval or abstain when safety evidence is incomplete."
            ),
            "fix_category": "CRITICAL_VALUE_CHECK",
            "evidence_markers": [
                "[sufficiency:missing_critical_requirement]",
            ],
            "missing_requirement_types": [
                "required_value",
                "required_condition_or_scope",
            ],
            "missing_evidence": [
                "Critical safety support required by the query is missing.",
            ],
        }

    def _build_structured_sufficiency_result(
        self,
        reason: str,
        fix_category: str,
        *,
        evidence_markers: list[str] | None = None,
        missing_requirement_types: list[str] | None = None,
        missing_evidence: list[str] | None = None,
    ) -> SufficiencyResult:
        """Build a structured SufficiencyResult for explicit root-cause detections."""
        requirement_types = list(missing_requirement_types or [])
        return SufficiencyResult(
            sufficient=False,
            sufficiency_label="insufficient",
            required_evidence=[
                EvidenceRequirement(
                    requirement_id=f"structured_{index:03d}",
                    description=reason.replace("_", " "),
                    requirement_type=requirement_type,
                    importance="critical",
                    verifier="heuristic",
                )
                for index, requirement_type in enumerate(requirement_types, start=1)
            ],
            should_abstain=True,
            should_expand_retrieval=True,
            structured_failure_reason=reason,
            recommended_fix_category=fix_category,
            evidence_markers=list(evidence_markers or []),
            missing_requirement_types=requirement_types,
            missing_evidence=list(missing_evidence or []),
            method="structured_heuristic_v1",
            calibration_status="preliminary_calibrated_v1",
            limitations=[
                "Structured heuristic detection; not NLI-verified",
                "Pattern matching only; may produce false positives on edge cases",
            ],
        )

    def _terms(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in STOPWORDS
        }
