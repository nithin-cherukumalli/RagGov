"""Claim-level diagnostic evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from raggov import diagnose
from raggov.models.chunk import RetrievedChunk
from raggov.models.corpus import CorpusEntry
from raggov.models.diagnosis import AnalyzerResult, Diagnosis
from raggov.models.run import RAGRun

from stresslab.cases import ClaimDiagnosisGoldCase, ClaimDiagnosisGoldSet


EVALUATION_STATUS = "diagnostic_gold_v0_small_unvalidated"


@dataclass(frozen=True)
class ClaimDiagnosisCaseResult:
    """Per-example comparison between expected and observed outputs.

    Tracks separate evaluation axes:
    - Axis A: Claim Support (claim_label_pass)
    - Axis B: Citation Validity (citation_validity_pass)
    - Axis C: Freshness Validity (freshness_validity_pass)
    - Axis D: Context Sufficiency (sufficiency_pass)
    - Axis E: A2P Root Cause (a2p_primary_cause_pass)
    """

    case_id: str
    category: str
    claim_label_pass: bool
    citation_validity_pass: bool
    freshness_validity_pass: bool
    sufficiency_pass: bool
    a2p_primary_cause_pass: bool
    primary_stage_pass: bool
    fix_category_exact_pass: bool
    fix_category_partial_pass: bool
    matched_overall: bool
    expected_non_clean: bool
    observed_primary_failure: str
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ClaimDiagnosisHarnessResult:
    """Aggregate metrics for claim-level diagnostic harness.

    Separate accuracy metrics per evaluation axis:
    - Axis A: claim_label_accuracy
    - Axis B: citation_validity_accuracy
    - Axis C: freshness_validity_accuracy
    - Axis D: sufficiency_accuracy
    - Axis E: a2p_primary_cause_accuracy
    """

    evaluation_status: str
    a2p_mode: str
    total_examples: int
    per_example: list[ClaimDiagnosisCaseResult]
    claim_label_accuracy: float
    citation_validity_accuracy: float
    freshness_validity_accuracy: float
    sufficiency_accuracy: float
    a2p_primary_cause_accuracy: float
    primary_stage_accuracy: float
    fix_category_exact_accuracy: float
    fix_category_partial_accuracy: float
    false_clean_count: int
    claim_label_breakdown: dict[str, dict[str, int]]
    category_metrics: dict[str, dict[str, float | int]]
    mismatches: list[dict[str, Any]]


def run_claim_diagnosis_harness(
    gold_set: ClaimDiagnosisGoldSet,
    engine_config: dict[str, Any] | None = None,
) -> ClaimDiagnosisHarnessResult:
    """Evaluate GovRAG against claim-level gold examples."""
    resolved_engine_config = {
        "enable_a2p": True,
        "use_llm": False,
        **(engine_config or {}),
    }
    a2p_mode = _a2p_mode_from_config(resolved_engine_config)
    per_example: list[ClaimDiagnosisCaseResult] = []
    mismatches: list[dict[str, Any]] = []
    claim_label_breakdown = _empty_claim_label_breakdown()

    for example in gold_set.examples:
        diagnosis = diagnose(_run_from_example(example), config=resolved_engine_config)
        _record_claim_labels(claim_label_breakdown, example, diagnosis)
        case_result = _evaluate_example(example, diagnosis)
        per_example.append(case_result)
        if not case_result.matched_overall:
            expected_actual = _expected_actual_payload(example, diagnosis)
            mismatches.append(
                {
                    "case_id": example.case_id,
                    "category": example.category,
                    "notes": case_result.notes,
                    "expected_primary_stage": example.expected_primary_stage,
                    "observed_primary_stage": diagnosis.root_cause_stage.value,
                    "expected_fix_category": example.expected_fix_category,
                    "observed_fix_category": _fix_category(diagnosis.proposed_fix or diagnosis.recommended_fix),
                    "expected": expected_actual["expected"],
                    "actual": expected_actual["actual"],
                }
            )

    total = len(per_example) or 1
    claim_label_acc = sum(1 for result in per_example if result.claim_label_pass) / total
    citation_validity_acc = sum(1 for result in per_example if result.citation_validity_pass) / total
    freshness_validity_acc = sum(1 for result in per_example if result.freshness_validity_pass) / total
    sufficiency_acc = sum(1 for result in per_example if result.sufficiency_pass) / total
    a2p_cause_acc = sum(1 for result in per_example if result.a2p_primary_cause_pass) / total
    stage_acc = sum(1 for result in per_example if result.primary_stage_pass) / total
    fix_exact_acc = sum(1 for result in per_example if result.fix_category_exact_pass) / total
    fix_partial_acc = sum(1 for result in per_example if result.fix_category_partial_pass) / total
    false_clean_count = sum(
        1
        for result in per_example
        if result.expected_non_clean and result.observed_primary_failure == "CLEAN"
    )
    category_metrics = _category_metrics(per_example)

    return ClaimDiagnosisHarnessResult(
        evaluation_status=gold_set.evaluation_status,
        a2p_mode=a2p_mode,
        total_examples=len(per_example),
        per_example=per_example,
        claim_label_accuracy=claim_label_acc,
        citation_validity_accuracy=citation_validity_acc,
        freshness_validity_accuracy=freshness_validity_acc,
        sufficiency_accuracy=sufficiency_acc,
        a2p_primary_cause_accuracy=a2p_cause_acc,
        primary_stage_accuracy=stage_acc,
        fix_category_exact_accuracy=fix_exact_acc,
        fix_category_partial_accuracy=fix_partial_acc,
        false_clean_count=false_clean_count,
        claim_label_breakdown=claim_label_breakdown,
        category_metrics=category_metrics,
        mismatches=mismatches,
    )


def render_claim_diagnosis_report(result: ClaimDiagnosisHarnessResult) -> str:
    """Render a concise text report with metrics and mismatches."""
    lines = [
        "Claim-Level Diagnostic Evaluation Harness",
        f"evaluation_status={result.evaluation_status}",
        f"a2p_mode={result.a2p_mode}",
        f"total_examples={result.total_examples}",
        "",
        "Aggregate Metrics (by axis):",
        f"  Axis A - Claim Support:       {result.claim_label_accuracy:.2f}",
        f"  Axis B - Citation Validity:   {result.citation_validity_accuracy:.2f}",
        f"  Axis C - Freshness Validity:  {result.freshness_validity_accuracy:.2f}",
        f"  Axis D - Context Sufficiency: {result.sufficiency_accuracy:.2f}",
        f"  Axis E - A2P Root Cause:      {result.a2p_primary_cause_accuracy:.2f}",
        f"  Primary Stage:                {result.primary_stage_accuracy:.2f}",
        f"  Fix Category (exact):         {result.fix_category_exact_accuracy:.2f}",
        f"  Fix Category (partial):       {result.fix_category_partial_accuracy:.2f}",
        f"false_clean_count={result.false_clean_count}",
        f"claim_label_breakdown={result.claim_label_breakdown}",
        "",
        "Per-example results:",
    ]
    for case in result.per_example:
        status = "PASS" if case.matched_overall else "FAIL"
        lines.append(f"- {case.case_id} [{case.category}]: {status}")

    if result.category_metrics:
        lines.append("")
        lines.append("Category Metrics:")
        for category, metrics in sorted(result.category_metrics.items()):
            lines.append(
                f"  - {category}: {int(metrics['total_cases'])} cases, "
                f"claim={metrics['claim_label_accuracy']:.2f}, "
                f"sufficiency={metrics['sufficiency_accuracy']:.2f}, "
                f"overall={metrics['overall_match_rate']:.2f}"
            )

    # Group mismatches by axis
    if result.mismatches:
        lines.append("")
        lines.append("Mismatches by Axis:")
        lines.append("")

        claim_mismatches = [case for case in result.per_example if not case.claim_label_pass]
        if claim_mismatches:
            lines.append(f"Axis A - Claim Support ({len(claim_mismatches)} mismatches):")
            for case in claim_mismatches:
                claim_notes = [note for note in case.notes if "claim label mismatch" in note]
                lines.append(f"  - {case.case_id}: {'; '.join(claim_notes)}")
            lines.append("")

        citation_mismatches = [case for case in result.per_example if not case.citation_validity_pass]
        if citation_mismatches:
            lines.append(f"Axis B - Citation Validity ({len(citation_mismatches)} mismatches):")
            for case in citation_mismatches:
                citation_notes = [note for note in case.notes if "citation validity mismatch" in note]
                lines.append(f"  - {case.case_id}: {'; '.join(citation_notes)}")
            lines.append("")

        freshness_mismatches = [case for case in result.per_example if not case.freshness_validity_pass]
        if freshness_mismatches:
            lines.append(f"Axis C - Freshness Validity ({len(freshness_mismatches)} mismatches):")
            for case in freshness_mismatches:
                freshness_notes = [note for note in case.notes if "freshness validity mismatch" in note]
                lines.append(f"  - {case.case_id}: {'; '.join(freshness_notes)}")
            lines.append("")

        sufficiency_mismatches = [case for case in result.per_example if not case.sufficiency_pass]
        if sufficiency_mismatches:
            lines.append(f"Axis D - Context Sufficiency ({len(sufficiency_mismatches)} mismatches):")
            for case in sufficiency_mismatches:
                sufficiency_notes = [note for note in case.notes if "sufficiency mismatch" in note]
                lines.append(f"  - {case.case_id}: {'; '.join(sufficiency_notes)}")
            lines.append("")

        a2p_mismatches = [case for case in result.per_example if not case.a2p_primary_cause_pass]
        if a2p_mismatches:
            lines.append(f"Axis E - A2P Root Cause ({len(a2p_mismatches)} mismatches):")
            for case in a2p_mismatches:
                a2p_notes = [note for note in case.notes if "a2p cause mismatch" in note]
                lines.append(f"  - {case.case_id}: {'; '.join(a2p_notes)}")
            lines.append("")

        stage_mismatches = [case for case in result.per_example if not case.primary_stage_pass]
        if stage_mismatches:
            lines.append(f"Primary Stage ({len(stage_mismatches)} mismatches):")
            for case in stage_mismatches:
                stage_notes = [note for note in case.notes if "stage mismatch" in note]
                lines.append(f"  - {case.case_id}: {'; '.join(stage_notes)}")
            lines.append("")

        fix_mismatches = [case for case in result.per_example if not case.fix_category_partial_pass]
        if fix_mismatches:
            lines.append(f"Fix Category ({len(fix_mismatches)} mismatches):")
            for case in fix_mismatches:
                fix_notes = [note for note in case.notes if "fix category mismatch" in note]
                lines.append(f"  - {case.case_id}: {'; '.join(fix_notes)}")

    return "\n".join(lines)


def _a2p_mode_from_config(config: dict[str, Any]) -> str:
    if not config.get("enable_a2p", False):
        return "disabled"
    if config.get("use_a2p_v2", False):
        return "v2"
    return "v1"


def _run_from_example(example: ClaimDiagnosisGoldCase) -> RAGRun:
    chunks = [RetrievedChunk.model_validate(chunk) for chunk in example.retrieved_chunks]
    corpus_entries = [CorpusEntry.model_validate(entry) for entry in example.corpus_entries]
    return RAGRun(
        run_id=example.case_id,
        query=example.query,
        retrieved_chunks=chunks,
        final_answer=example.final_answer,
        cited_doc_ids=example.cited_doc_ids,
        corpus_entries=corpus_entries,
        metadata=example.metadata,
    )


def _evaluate_example(example: ClaimDiagnosisGoldCase, diagnosis: Diagnosis) -> ClaimDiagnosisCaseResult:
    notes: list[str] = []

    # Axis A: Claim Support
    observed_claims = {claim.claim_text: claim.label for claim in diagnosis.claim_results}
    claim_label_pass = True
    for expected in example.expected_claims:
        # Support new expected_claim_label field with fallback to expected_label
        expected_label = expected.expected_claim_label or expected.expected_label
        observed_label = observed_claims.get(expected.claim_text)
        if observed_label != expected_label:
            claim_label_pass = False
            notes.append(
                f"claim label mismatch for '{expected.claim_text}': expected {expected_label}, observed {observed_label}"
            )

    # Axis B: Citation Validity
    observed_citation_validity = _observed_citation_validity(diagnosis)
    citation_validity_pass = True
    for expected in example.expected_claims:
        if expected.expected_citation_validity is None:
            continue
        if observed_citation_validity != expected.expected_citation_validity:
            citation_validity_pass = False
            notes.append(
                f"citation validity mismatch for '{expected.claim_text}': expected {expected.expected_citation_validity}, observed {observed_citation_validity}"
            )

    # If no expected citation validity specified, check if it's applicable
    if all(expected.expected_citation_validity is None for expected in example.expected_claims):
        citation_validity_pass = observed_citation_validity in {"valid", "not_applicable"}

    # Axis C: Freshness Validity
    observed_freshness_validity = _observed_freshness_validity(diagnosis)
    freshness_validity_pass = True
    for expected in example.expected_claims:
        if expected.expected_freshness_validity is None:
            continue
        if observed_freshness_validity != expected.expected_freshness_validity:
            freshness_validity_pass = False
            notes.append(
                f"freshness validity mismatch for '{expected.claim_text}': expected {expected.expected_freshness_validity}, observed {observed_freshness_validity}"
            )

    # If no expected freshness validity specified, check if it's applicable
    if all(expected.expected_freshness_validity is None for expected in example.expected_claims):
        freshness_validity_pass = observed_freshness_validity in {"fresh", "unknown"}

    # Axis D: Context Sufficiency
    sufficiency = _claim_aware_or_base_sufficiency(diagnosis)
    observed_sufficient = (
        sufficiency.sufficiency_result.sufficient
        if sufficiency is not None and sufficiency.sufficiency_result is not None
        else None
    )
    # Support both expected_sufficient (old) and expected_sufficiency (new)
    expected_sufficient = example.expected_sufficiency if example.expected_sufficiency is not None else example.expected_sufficient
    sufficiency_pass = observed_sufficient == expected_sufficient
    if not sufficiency_pass:
        notes.append(
            f"sufficiency mismatch: expected {expected_sufficient}, observed {observed_sufficient}"
        )

    # Axis E: A2P Root Cause
    a2p = _analyzer_result(diagnosis, "A2PAttributionAnalyzer")
    claim_attr_by_text: dict[str, str] = {}
    if a2p is not None and a2p.claim_attributions:
        claim_attr_by_text = {
            attribution.claim_text: attribution.primary_cause
            for attribution in a2p.claim_attributions
        }
    a2p_primary_cause_pass = True
    for expected in example.expected_claims:
        if expected.expected_a2p_primary_cause is None:
            continue
        observed_cause = claim_attr_by_text.get(expected.claim_text)
        if observed_cause != expected.expected_a2p_primary_cause:
            a2p_primary_cause_pass = False
            notes.append(
                f"a2p cause mismatch for '{expected.claim_text}': expected {expected.expected_a2p_primary_cause}, observed {observed_cause}"
            )

    primary_stage_pass = diagnosis.root_cause_stage.value == example.expected_primary_stage
    if not primary_stage_pass:
        notes.append(
            f"stage mismatch: expected {example.expected_primary_stage}, observed {diagnosis.root_cause_stage.value}"
        )

    fix_text = diagnosis.proposed_fix or diagnosis.recommended_fix
    observed_fix_category = _fix_category(fix_text)
    fix_category_exact_pass = observed_fix_category == example.expected_fix_category
    fix_category_partial_pass = _fix_category_partial_match(example.expected_fix_category, fix_text)
    if not fix_category_exact_pass:
        notes.append(
            f"fix category mismatch: expected {example.expected_fix_category}, observed {observed_fix_category}"
        )

    matched_overall = (
        claim_label_pass
        and citation_validity_pass
        and freshness_validity_pass
        and sufficiency_pass
        and a2p_primary_cause_pass
        and primary_stage_pass
        and fix_category_partial_pass
    )
    expected_non_clean = _expected_non_clean(example)
    return ClaimDiagnosisCaseResult(
        case_id=example.case_id,
        category=example.category,
        claim_label_pass=claim_label_pass,
        citation_validity_pass=citation_validity_pass,
        freshness_validity_pass=freshness_validity_pass,
        sufficiency_pass=sufficiency_pass,
        a2p_primary_cause_pass=a2p_primary_cause_pass,
        primary_stage_pass=primary_stage_pass,
        fix_category_exact_pass=fix_category_exact_pass,
        fix_category_partial_pass=fix_category_partial_pass,
        matched_overall=matched_overall,
        expected_non_clean=expected_non_clean,
        observed_primary_failure=diagnosis.primary_failure.value,
        notes=notes,
    )


def _expected_actual_payload(
    example: ClaimDiagnosisGoldCase,
    diagnosis: Diagnosis,
) -> dict[str, dict[str, Any]]:
    sufficiency = _claim_aware_or_base_sufficiency(diagnosis)
    observed_sufficient = (
        sufficiency.sufficiency_result.sufficient
        if sufficiency is not None and sufficiency.sufficiency_result is not None
        else None
    )
    expected_sufficient = (
        example.expected_sufficiency
        if example.expected_sufficiency is not None
        else example.expected_sufficient
    )
    return {
        "expected": {
            "primary_stage": example.expected_primary_stage,
            "fix_category": example.expected_fix_category,
            "sufficient": expected_sufficient,
            "claim_labels": {
                claim.claim_text: claim.expected_claim_label or claim.expected_label
                for claim in example.expected_claims
            },
        },
        "actual": {
            "primary_failure": diagnosis.primary_failure.value,
            "primary_stage": diagnosis.root_cause_stage.value,
            "fix_category": _fix_category(diagnosis.proposed_fix or diagnosis.recommended_fix),
            "sufficient": observed_sufficient,
            "claim_labels": {
                claim.claim_text: claim.label
                for claim in diagnosis.claim_results
            },
        },
    }


def _expected_non_clean(example: ClaimDiagnosisGoldCase) -> bool:
    expected_sufficient = (
        example.expected_sufficiency
        if example.expected_sufficiency is not None
        else example.expected_sufficient
    )
    if expected_sufficient is False:
        return True
    if example.expected_primary_stage != "UNKNOWN":
        return True
    for claim in example.expected_claims:
        expected_label = claim.expected_claim_label or claim.expected_label
        if expected_label in {"unsupported", "contradicted", "abstain"}:
            return True
        if claim.expected_citation_validity == "invalid":
            return True
        if claim.expected_freshness_validity == "stale":
            return True
    return False


def _empty_claim_label_breakdown() -> dict[str, dict[str, int]]:
    labels = ["entailed", "unsupported", "contradicted", "abstain", "missing"]
    return {label: {"expected": 0, "observed": 0} for label in labels}


def _record_claim_labels(
    breakdown: dict[str, dict[str, int]],
    example: ClaimDiagnosisGoldCase,
    diagnosis: Diagnosis,
) -> None:
    observed_by_text = {claim.claim_text: claim.label for claim in diagnosis.claim_results}

    for claim in example.expected_claims:
        expected_label = claim.expected_claim_label or claim.expected_label or "missing"
        breakdown.setdefault(expected_label, {"expected": 0, "observed": 0})
        breakdown[expected_label]["expected"] += 1

        observed_label = observed_by_text.get(claim.claim_text, "missing")
        breakdown.setdefault(observed_label, {"expected": 0, "observed": 0})
        breakdown[observed_label]["observed"] += 1


def _category_metrics(
    per_example: list[ClaimDiagnosisCaseResult],
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[ClaimDiagnosisCaseResult]] = {}
    for case in per_example:
        grouped.setdefault(case.category, []).append(case)

    metrics: dict[str, dict[str, float | int]] = {}
    for category, cases in grouped.items():
        total = len(cases) or 1
        metrics[category] = {
            "total_cases": len(cases),
            "claim_label_accuracy": sum(1 for case in cases if case.claim_label_pass) / total,
            "citation_validity_accuracy": sum(1 for case in cases if case.citation_validity_pass) / total,
            "freshness_validity_accuracy": sum(1 for case in cases if case.freshness_validity_pass) / total,
            "sufficiency_accuracy": sum(1 for case in cases if case.sufficiency_pass) / total,
            "a2p_primary_cause_accuracy": sum(1 for case in cases if case.a2p_primary_cause_pass) / total,
            "primary_stage_accuracy": sum(1 for case in cases if case.primary_stage_pass) / total,
            "fix_category_partial_accuracy": sum(1 for case in cases if case.fix_category_partial_pass) / total,
            "overall_match_rate": sum(1 for case in cases if case.matched_overall) / total,
        }
    return metrics


def _analyzer_result(diagnosis: Diagnosis, analyzer_name: str) -> AnalyzerResult | None:
    for result in diagnosis.analyzer_results:
        if result.analyzer_name == analyzer_name:
            return result
    return None


def _claim_aware_or_base_sufficiency(diagnosis: Diagnosis) -> AnalyzerResult | None:
    claim_aware = _analyzer_result(diagnosis, "ClaimAwareSufficiencyAnalyzer")
    if claim_aware is not None and claim_aware.sufficiency_result is not None:
        return claim_aware
    return _analyzer_result(diagnosis, "SufficiencyAnalyzer")


def _fix_category(text: str | None) -> str:
    if not text:
        return "unknown"
    lowered = text.lower()
    if any(term in lowered for term in ("retrieval", "top-k", "query recall", "missing evidence")):
        return "retrieval"
    if any(term in lowered for term in ("generation", "decode", "contradiction", "context-grounded")):
        return "generation"
    if any(term in lowered for term in ("evidence selection", "supported claims", "support checks")):
        return "grounding"
    if any(term in lowered for term in ("security", "prompt injection", "privacy")):
        return "security"
    return "other"


def _fix_category_partial_match(expected_category: str, fix_text: str | None) -> bool:
    if not fix_text:
        return False
    category = _fix_category(fix_text)
    if category == expected_category:
        return True
    lowered = fix_text.lower()
    expected_tokens = expected_category.replace("_", " ").split()
    return any(token in lowered for token in expected_tokens)


def _observed_citation_validity(diagnosis: Diagnosis) -> str:
    """Extract observed citation validity from diagnosis.

    Returns:
    - "invalid" if CitationMismatchAnalyzer failed or CitationFaithfulnessProbe failed/warned
    - "valid" if no citation issues detected
    - "not_applicable" if no citations to check
    """
    citation_mismatch = _analyzer_result(diagnosis, "CitationMismatchAnalyzer")
    citation_faithfulness = _analyzer_result(diagnosis, "CitationFaithfulnessProbe")

    if citation_mismatch is not None and citation_mismatch.status == "fail":
        return "invalid"
    if citation_faithfulness is not None and citation_faithfulness.status in {"fail", "warn"}:
        return "invalid"
    if citation_mismatch is not None and citation_mismatch.status == "skip":
        return "not_applicable"
    if citation_faithfulness is not None and citation_faithfulness.status == "skip":
        return "not_applicable"
    return "valid"


def _observed_freshness_validity(diagnosis: Diagnosis) -> str:
    """Extract observed freshness validity from diagnosis.

    Returns:
    - "stale" if StaleRetrievalAnalyzer failed
    - "fresh" if no stale sources detected
    - "unknown" if no corpus metadata available
    """
    stale_retrieval = _analyzer_result(diagnosis, "StaleRetrievalAnalyzer")

    if stale_retrieval is not None and stale_retrieval.status == "fail":
        return "stale"
    if stale_retrieval is not None and stale_retrieval.status == "skip":
        return "unknown"
    return "fresh"
