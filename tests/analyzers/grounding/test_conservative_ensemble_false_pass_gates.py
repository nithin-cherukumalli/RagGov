from __future__ import annotations

from unittest.mock import MagicMock

from raggov.analyzers.grounding.candidate_selection import EvidenceCandidate
from raggov.analyzers.grounding.verifiers import ConservativeEnsembleVerifier, VerificationResult


def _candidate(chunk_text: str, chunk_id: str = "chunk-1") -> EvidenceCandidate:
    return EvidenceCandidate(
        chunk_id=chunk_id,
        chunk_text=chunk_text,
        source_doc_id="doc-1",
        chunk_text_preview=chunk_text,
        lexical_overlap_score=0.95,
        anchor_overlap_score=0.95,
        value_overlap_score=0.95,
        retrieval_score=0.95,
        rerank_score=None,
        is_best=True,
    )


def _llm_supported(chunk_id: str = "chunk-1") -> VerificationResult:
    return VerificationResult(
        label="entailed",
        support_label="supported",
        raw_score=0.95,
        evidence_chunk_id=chunk_id,
        evidence_span=None,
        rationale="LLM marked the claim as supported.",
        verifier_name="llm_mock",
        supporting_chunk_ids=[chunk_id],
        candidate_chunk_ids=[chunk_id],
        best_candidate_id=chunk_id,
        evidence_mode="single_chunk",
    )


def _heuristic_result(
    support_label: str = "supported",
    label: str | None = None,
    chunk_id: str = "chunk-1",
) -> VerificationResult:
    resolved_label = label
    if resolved_label is None:
        resolved_label = (
            "entailed"
            if support_label == "supported"
            else "contradicted"
            if support_label == "contradicted"
            else "unsupported"
        )
    return VerificationResult(
        label=resolved_label,  # type: ignore[arg-type]
        support_label=support_label,  # type: ignore[arg-type]
        raw_score=0.82,
        evidence_chunk_id=chunk_id,
        evidence_span=None,
        rationale=f"Heuristic judged {support_label}.",
        verifier_name="heuristic_mock",
        supporting_chunk_ids=[chunk_id] if support_label == "supported" else [],
        contradicting_chunk_ids=[chunk_id] if support_label == "contradicted" else [],
        candidate_chunk_ids=[chunk_id],
        best_candidate_id=chunk_id,
        evidence_mode="single_chunk",
    )


def _verifier(
    *,
    llm_result: VerificationResult | None = None,
    heuristic_result: VerificationResult | None = None,
) -> ConservativeEnsembleVerifier:
    verifier = ConservativeEnsembleVerifier({"llm_client": MagicMock()})
    verifier._llm_verifier.verify = MagicMock(return_value=llm_result or _llm_supported())
    verifier._heuristic_verifier.verify = MagicMock(
        return_value=heuristic_result or _heuristic_result()
    )
    return verifier


def test_supported_downgraded_when_claim_value_missing_from_evidence() -> None:
    verifier = _verifier()
    res = verifier.verify(
        "The timeout for standard requests is 30 seconds.",
        "query",
        [_candidate("Standard requests require authentication before sending.")],
        metadata={"numbers": ["30"], "entities": ["standard requests", "timeout"]},
    )

    assert res.support_label == "insufficient_evidence"
    assert res.safety_gate_reason == "critical_value_missing_or_conflicting"
    assert res.safety_gate_category == "value_mismatch_missed"


def test_supported_downgraded_when_evidence_has_conflicting_value() -> None:
    verifier = _verifier()
    res = verifier.verify(
        "The timeout for standard requests is 30 seconds.",
        "query",
        [_candidate("Standard API requests timeout after 5 seconds.")],
        metadata={"numbers": ["30"], "entities": ["standard requests", "timeout"]},
    )

    assert res.support_label == "contradicted"
    assert res.safety_gate_reason == "critical_value_missing_or_conflicting"
    assert res.safety_gate_category == "value_mismatch_missed"


def test_supported_downgraded_when_claim_date_missing_from_evidence() -> None:
    verifier = _verifier()
    res = verifier.verify(
        "The filing deadline is June 30, 2024.",
        "query",
        [_candidate("Applications must be filed before the annual compliance deadline.")],
        metadata={"dates": ["2024-06-30"], "entities": ["filing deadline"]},
    )

    assert res.support_label == "insufficient_evidence"
    assert res.safety_gate_reason == "critical_date_missing_or_conflicting"
    assert res.safety_gate_category == "date_mismatch_missed"


def test_supported_downgraded_when_evidence_has_conflicting_date() -> None:
    verifier = _verifier()
    res = verifier.verify(
        "The filing deadline is June 30, 2024.",
        "query",
        [_candidate("Applications must be filed by July 15, 2024.")],
        metadata={"dates": ["2024-06-30"], "entities": ["filing deadline"]},
    )

    assert res.support_label == "contradicted"
    assert res.safety_gate_reason == "critical_date_missing_or_conflicting"
    assert res.safety_gate_category == "date_mismatch_missed"


def test_supported_downgraded_when_unit_mismatch() -> None:
    verifier = _verifier()
    res = verifier.verify(
        "The maximum dose is 50 mg daily.",
        "query",
        [_candidate("The maximum dose is 50 g daily.")],
        metadata={"numbers": ["50"], "entities": ["maximum dose"]},
    )

    assert res.support_label == "contradicted"
    assert res.safety_gate_reason == "unit_or_magnitude_mismatch"
    assert res.safety_gate_category == "unit_mismatch_missed"


def test_supported_downgraded_when_entity_mismatch() -> None:
    verifier = _verifier()
    res = verifier.verify(
        "The premium savings account offers 4.5% interest.",
        "query",
        [_candidate("The standard savings account offers 4.5% interest.")],
        metadata={"numbers": ["4.5"], "entities": ["premium savings account"]},
    )

    assert res.support_label == "insufficient_evidence"
    assert res.safety_gate_reason == "missing_or_mismatched_critical_entity"
    assert res.safety_gate_category == "entity_mismatch_missed"


def test_supported_downgraded_when_value_binds_to_wrong_entity() -> None:
    verifier = _verifier()
    res = verifier.verify(
        "The premium savings account offers a 4.5% interest rate.",
        "query",
        [_candidate("We offer 4.5% interest on standard savings; premium account holders receive 5.5% interest.")],
        metadata={"numbers": ["4.5"], "entities": ["premium savings"]},
    )

    assert res.support_label == "contradicted"
    assert res.safety_gate_reason == "missing_or_mismatched_critical_entity"
    assert res.safety_gate_category == "entity_mismatch_missed"


def test_supported_downgraded_when_section_number_binds_to_wrong_rule() -> None:
    verifier = _verifier()
    res = verifier.verify(
        "Corporate mergers are governed by Section 12.",
        "query",
        [
            _candidate("Section 12 regulates tax implications of corporate restructuring.", "chunk-a"),
            _candidate("Section 15 regulates terms under which domestic commercial entities merge.", "chunk-b"),
        ],
        metadata={"numbers": ["12"], "entities": ["mergers"]},
    )

    assert res.support_label == "contradicted"
    assert res.safety_gate_reason == "missing_or_mismatched_critical_entity"
    assert res.safety_gate_category == "entity_mismatch_missed"


def test_supported_downgraded_for_related_but_non_supporting_evidence() -> None:
    verifier = _verifier(heuristic_result=_heuristic_result("unsupported"))
    res = verifier.verify(
        "The API endpoint returns JSON by default.",
        "query",
        [_candidate("The API endpoint requires authentication via bearer token.")],
        metadata={"entities": ["API endpoint"]},
    )

    assert res.support_label == "insufficient_evidence"
    assert res.safety_gate_reason == "related_but_non_supporting"
    assert res.safety_gate_category == "related_but_non_supporting"


def test_supported_downgraded_for_compound_claim_partial_support() -> None:
    verifier = _verifier()
    verifier._compound_verifier.verify = MagicMock(
        return_value=VerificationResult(
            label="abstain",
            support_label="skipped",
            raw_score=0.0,
            evidence_chunk_id=None,
            evidence_span=None,
            rationale="decomposition failed",
            verifier_name="compound_mock",
            compound_decomposed=False,
            undecomposed_compound_gate_triggered=True,
        )
    )

    res = verifier.verify(
        "The free tier allows 1000 requests per day and supports custom domains.",
        "query",
        [_candidate("The free tier allows 1000 requests per day.")],
        metadata={"atomicity_status": "compound", "numbers": ["1000"], "entities": ["free tier", "custom domains"]},
    )

    assert res.support_label == "insufficient_evidence"
    assert res.safety_gate_reason == "compound_claim_not_fully_supported"
    assert res.safety_gate_category == "compound_partial_support"


def test_explicit_value_conflict_returns_contradicted_not_supported() -> None:
    verifier = _verifier(heuristic_result=_heuristic_result("contradicted"))
    res = verifier.verify(
        "The annual interest rate is 5.5%.",
        "query",
        [_candidate("The annual interest rate is 3.5%.")],
        metadata={"numbers": ["5.5"], "entities": ["annual interest rate"]},
    )

    assert res.support_label == "contradicted"
    assert res.label == "contradicted"
    assert res.safety_gate_triggered is True


def test_true_paraphrase_support_still_passes() -> None:
    verifier = _verifier(heuristic_result=_heuristic_result("unsupported"))
    res = verifier.verify(
        "Solid water occupies a larger physical volume than its liquid counterpart.",
        "query",
        [_candidate("Liquid water expands and increases in volume upon crystallization into ice.")],
        metadata={"entities": ["solid water", "liquid water"]},
    )

    assert res.support_label == "supported"
    assert res.safety_gate_triggered is False


def test_true_supported_claim_with_matching_values_still_passes() -> None:
    verifier = _verifier()
    res = verifier.verify(
        "The standard API timeout is 5 seconds.",
        "query",
        [_candidate("Standard API requests timeout after 5 seconds.")],
        metadata={"numbers": ["5"], "entities": ["standard API", "timeout"]},
    )

    assert res.support_label == "supported"
    assert res.safety_gate_triggered is False


def test_telemetry_records_gate_category_and_reason() -> None:
    verifier = _verifier()
    res = verifier.verify(
        "The standard API timeout is 30 seconds.",
        "query",
        [_candidate("Standard API requests timeout after 5 seconds.")],
        metadata={"numbers": ["30"], "entities": ["standard API", "timeout"]},
    )

    assert res.safety_gate_triggered is True
    assert res.safety_gate_reason == "critical_value_missing_or_conflicting"
    assert res.safety_gate_category == "value_mismatch_missed"
    assert res.llm_label == "supported"
    assert res.heuristic_label == "supported"
    assert "critical_value_missing_or_conflicting" in res.deterministic_gate_labels
    assert res.critical_fact_check_summary["value_conflict"] is True
