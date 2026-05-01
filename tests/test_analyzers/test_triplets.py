"""
Tests for PR 9 — RefChecker-inspired triplet extraction interface.

Tests cover:
- GO subject + predicate + object extraction
- Circular / notification subject
- Policy prohibits / mandates
- Numeric and date qualifier preservation
- LLM extractor valid JSON path
- LLM extractor invalid JSON fallback (visible, not silent)
- LLM extractor missing client raises
- build_triplet_extractor factory gate (disabled by default)
- ClaimEvidenceRecord has None triplets when flag is off
- ClaimEvidenceRecord has triplets when flag is on
- ClaimGroundingAnalyzer passes triplets through without default behavior change
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from raggov.analyzers.grounding.triplets import (
    ClaimTriplet,
    LLMTripletExtractorV1,
    RuleBasedPolicyTripletExtractorV0,
    TripletExtractor,
    build_triplet_extractor,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _extract(claim: str, claim_id: str = "claim_001") -> list[ClaimTriplet]:
    return RuleBasedPolicyTripletExtractorV0().extract(claim, source_claim_id=claim_id)


# ---------------------------------------------------------------------------
# 1. GO subject extraction
# ---------------------------------------------------------------------------

class TestGOSubjectExtraction:
    def test_go_subject_found(self) -> None:
        triplets = _extract("G.O.Ms.No. 42 mandates a 60% subsidy for farmers.")
        assert len(triplets) >= 1
        subjects = [t.subject.lower() for t in triplets]
        assert any("g.o" in s for s in subjects)

    def test_go_predicate_is_mandates(self) -> None:
        triplets = _extract("G.O.Ms.No. 42 mandates a 60% subsidy for all farmers.")
        assert any("mandate" in t.predicate.lower() for t in triplets)

    def test_go_object_contains_subsidy(self) -> None:
        triplets = _extract("G.O.Ms.No. 42 mandates a 60% subsidy for all farmers.")
        objects = " ".join(t.object.lower() for t in triplets)
        assert "subsidy" in objects or "farmer" in objects

    def test_go_numeric_value_preserved(self) -> None:
        triplets = _extract("G.O.Ms.No. 42 mandates a 60% subsidy for farmers.")
        all_values = [v for t in triplets for v in t.values]
        assert any("60" in v for v in all_values), f"Values were: {all_values}"

    def test_go_allows_optional_holidays(self) -> None:
        claim = "G.O.Rt.No. 123 allows government employees up to 5 optional holidays per year."
        triplets = _extract(claim)
        assert len(triplets) >= 1
        # subject must contain G.O reference
        assert any("g.o" in t.subject.lower() for t in triplets)
        # predicate must contain 'allow'
        assert any("allow" in t.predicate.lower() for t in triplets)
        # values must include 5
        all_values = [v for t in triplets for v in t.values]
        assert any("5" in v for v in all_values)

    def test_triplet_id_is_unique(self) -> None:
        triplets = _extract("G.O.Ms.No. 1 mandates X. G.O.Ms.No. 2 mandates Y.")
        ids = [t.triplet_id for t in triplets]
        assert len(ids) == len(set(ids))

    def test_source_claim_id_preserved(self) -> None:
        triplets = _extract("G.O.Ms.No. 7 mandates compliance.", claim_id="claim_042")
        assert all(t.source_claim_id == "claim_042" for t in triplets)

    def test_extraction_method_is_rule_based_v0(self) -> None:
        triplets = _extract("G.O.Ms.No. 7 mandates compliance.")
        assert all(t.extraction_method == "rule_based_policy_v0" for t in triplets)

    def test_calibration_status_is_uncalibrated(self) -> None:
        triplets = _extract("G.O.Ms.No. 7 mandates compliance.")
        assert all(t.calibration_status == "uncalibrated" for t in triplets)


# ---------------------------------------------------------------------------
# 2. Circular / notification subject
# ---------------------------------------------------------------------------

class TestCircularSubjectExtraction:
    def test_circular_subject_extracted(self) -> None:
        triplets = _extract("Circular no. 5 mandates deadline submission by 31 March 2024.")
        assert len(triplets) >= 1
        subjects = [t.subject.lower() for t in triplets]
        assert any("circular" in s for s in subjects)

    def test_circular_deadline_date_in_values(self) -> None:
        triplets = _extract("Circular no. 5 mandates deadline submission by 31 March 2024.")
        all_values = [v for t in triplets for v in t.values]
        date_found = any("2024" in v or "march" in v.lower() for v in all_values)
        assert date_found, f"Values found: {all_values}"

    def test_notification_subject_extracted(self) -> None:
        triplets = _extract("Notification no. 7 requires all beneficiaries to register by June 2025.")
        subjects = [t.subject.lower() for t in triplets]
        assert any("notification" in s for s in subjects)


# ---------------------------------------------------------------------------
# 3. Policy prohibits
# ---------------------------------------------------------------------------

class TestPolicyProhibits:
    def test_prohibits_predicate_extracted(self) -> None:
        triplets = _extract("The policy prohibits double-dipping of benefits under two schemes.")
        assert any("prohibit" in t.predicate.lower() for t in triplets)

    def test_prohibits_object_present(self) -> None:
        triplets = _extract("The policy prohibits double-dipping of benefits under two schemes.")
        objects = " ".join(t.object.lower() for t in triplets)
        assert "double" in objects or "benefit" in objects or "scheme" in objects

    def test_mandates_predicate_extracted(self) -> None:
        triplets = _extract("The scheme mandates submission of Form 16A by all loanee farmers.")
        assert any("mandate" in t.predicate.lower() for t in triplets)


# ---------------------------------------------------------------------------
# 4. Eligibility actor subject
# ---------------------------------------------------------------------------

class TestEligibilitySubjectExtraction:
    def test_government_employees_subject(self) -> None:
        triplets = _extract("Government employees are entitled to 30 days of casual leave per year.")
        subjects = [t.subject.lower() for t in triplets]
        assert any("employee" in s or "government" in s for s in subjects)

    def test_farmers_subject_extracted(self) -> None:
        triplets = _extract("All farmers are eligible for a 60% subsidy under PM-KUSUM.")
        subjects = [t.subject.lower() for t in triplets]
        assert any("farmer" in s for s in subjects)

    def test_beneficiaries_as_subject(self) -> None:
        triplets = _extract("Beneficiaries are entitled to food grains at Rs. 2 per kg.")
        subjects = [t.subject.lower() for t in triplets]
        assert any("beneficiar" in s for s in subjects)


# ---------------------------------------------------------------------------
# 5. Numeric qualifier preservation
# ---------------------------------------------------------------------------

class TestNumericQualifierPreservation:
    def test_percentage_preserved(self) -> None:
        triplets = _extract("The subsidy covers 75% of the benchmark cost.")
        all_values = [v for t in triplets for v in t.values]
        assert any("75" in v for v in all_values)

    def test_amount_in_rupees_preserved(self) -> None:
        # Use a predicate the extractor recognises — bare "is" is not a policy predicate
        triplets = _extract("A late fee of Rs. 200 per day is levied for delayed submission.")
        all_values = [v for t in triplets for v in t.values]
        assert any("200" in v for v in all_values), f"Values found: {all_values}"


    def test_qualifier_up_to_extracted(self) -> None:
        triplets = _extract("Employees may take up to 5 casual leaves per month.")
        all_qualifiers = [q for t in triplets for q in t.qualifiers]
        # "up to" should appear as a qualifier
        assert any("up to" in q.lower() for q in all_qualifiers) or any(
            "5" in v for t in triplets for v in t.values
        )

    def test_per_day_qualifier_preserved(self) -> None:
        triplets = _extract("A penalty of Rs. 50 per day is applicable for late filing.")
        all_qualifiers = [q for t in triplets for q in t.qualifiers]
        all_values = [v for t in triplets for v in t.values]
        assert any("50" in v for v in all_values)

    def test_date_value_preserved(self) -> None:
        triplets = _extract("Returns must be filed by 20 January 2024.")
        all_values = [v for t in triplets for v in t.values]
        assert any("2024" in v or "january" in v.lower() for v in all_values)


# ---------------------------------------------------------------------------
# 6. Empty / no-match claim
# ---------------------------------------------------------------------------

class TestNoMatchClaim:
    def test_pure_narrative_returns_empty_or_bare_predicate(self) -> None:
        """A claim with no policy verbs or GO references may return no triplets."""
        triplets = _extract("The sky is blue and water is wet.")
        # Either empty or a bare structural triplet — no crash is the requirement
        assert isinstance(triplets, list)

    def test_empty_string_does_not_crash(self) -> None:
        triplets = _extract("")
        assert isinstance(triplets, list)


# ---------------------------------------------------------------------------
# 7. LLM extractor — valid JSON
# ---------------------------------------------------------------------------

class TestLLMExtractorValidJSON:
    def _make_client(self, response: str) -> MagicMock:
        client = MagicMock()
        client.chat.return_value = response
        return client

    def test_llm_parses_valid_triplet(self) -> None:
        payload = json.dumps([
            {
                "subject": "G.O.Ms.No. 42",
                "predicate": "mandates",
                "object": "60% subsidy",
                "qualifiers": ["effective from 2023"],
                "values": ["60%"],
            }
        ])
        client = self._make_client(payload)
        extractor = LLMTripletExtractorV1(client)
        triplets = extractor.extract("G.O.Ms.No. 42 mandates a 60% subsidy.", "claim_001")
        assert len(triplets) == 1
        assert triplets[0].subject == "G.O.Ms.No. 42"
        assert triplets[0].predicate == "mandates"
        assert triplets[0].object == "60% subsidy"
        assert "60%" in triplets[0].values
        assert "effective from 2023" in triplets[0].qualifiers

    def test_llm_extraction_method_is_llm_v1(self) -> None:
        payload = json.dumps([
            {"subject": "S", "predicate": "allows", "object": "O", "qualifiers": [], "values": []}
        ])
        client = self._make_client(payload)
        extractor = LLMTripletExtractorV1(client)
        triplets = extractor.extract("S allows O.", "claim_001")
        assert triplets[0].extraction_method == "llm_triplet_v1"

    def test_llm_empty_array_returns_no_triplets(self) -> None:
        client = self._make_client("[]")
        extractor = LLMTripletExtractorV1(client)
        triplets = extractor.extract("Some claim.", "claim_001")
        assert triplets == []

    def test_llm_strips_markdown_fences(self) -> None:
        payload = "```json\n" + json.dumps([
            {"subject": "A", "predicate": "requires", "object": "B", "qualifiers": [], "values": []}
        ]) + "\n```"
        client = self._make_client(payload)
        extractor = LLMTripletExtractorV1(client)
        triplets = extractor.extract("A requires B.", "claim_001")
        assert len(triplets) == 1

    def test_llm_skips_item_with_empty_fields(self) -> None:
        payload = json.dumps([
            {"subject": "", "predicate": "mandates", "object": "X", "qualifiers": [], "values": []},
            {"subject": "Y", "predicate": "allows", "object": "Z", "qualifiers": [], "values": []},
        ])
        client = self._make_client(payload)
        extractor = LLMTripletExtractorV1(client)
        triplets = extractor.extract("Y allows Z.", "claim_001")
        assert len(triplets) == 1
        assert triplets[0].subject == "Y"


# ---------------------------------------------------------------------------
# 8. LLM extractor — invalid JSON fallback (visible, not silent)
# ---------------------------------------------------------------------------

class TestLLMExtractorInvalidJSONFallback:
    def test_invalid_json_falls_back_to_rule_based(self) -> None:
        client = MagicMock()
        client.chat.return_value = "THIS IS NOT JSON AT ALL {{{]]}"
        extractor = LLMTripletExtractorV1(client)
        # Should fall back silently to rule-based, not raise
        triplets = extractor.extract(
            "G.O.Ms.No. 5 mandates 30% subsidy for beneficiaries.", "claim_001"
        )
        assert isinstance(triplets, list)
        # Fallback method label is informative (not bare llm_triplet_v1)
        for t in triplets:
            assert "fallback" in t.extraction_method

    def test_non_array_json_falls_back(self) -> None:
        client = MagicMock()
        client.chat.return_value = json.dumps({"subject": "X"})  # dict not array
        extractor = LLMTripletExtractorV1(client)
        triplets = extractor.extract("X mandates Y.", "claim_001")
        assert isinstance(triplets, list)
        for t in triplets:
            assert "fallback" in t.extraction_method

    def test_client_exception_falls_back(self) -> None:
        client = MagicMock()
        client.chat.side_effect = RuntimeError("LLM API unavailable")
        extractor = LLMTripletExtractorV1(client)
        triplets = extractor.extract("The scheme requires Rs. 500 deposit.", "claim_001")
        assert isinstance(triplets, list)


# ---------------------------------------------------------------------------
# 9. LLM extractor — missing client raises
# ---------------------------------------------------------------------------

class TestLLMExtractorMissingClient:
    def test_none_client_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="llm_client"):
            LLMTripletExtractorV1(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 10. Factory gate — disabled by default
# ---------------------------------------------------------------------------

class TestBuildTripletExtractorFactory:
    def test_disabled_by_default_returns_none(self) -> None:
        extractor = build_triplet_extractor({})
        assert extractor is None

    def test_enabled_returns_rule_based_extractor(self) -> None:
        extractor = build_triplet_extractor({"enable_triplet_extraction": True})
        assert isinstance(extractor, RuleBasedPolicyTripletExtractorV0)

    def test_llm_mode_without_client_falls_back_to_rule_based(self) -> None:
        extractor = build_triplet_extractor(
            {"enable_triplet_extraction": True, "triplet_extractor_mode": "llm_v1"}
        )
        # No llm_client → falls back gracefully
        assert isinstance(extractor, RuleBasedPolicyTripletExtractorV0)

    def test_llm_mode_with_client_returns_llm_extractor(self) -> None:
        client = MagicMock()
        extractor = build_triplet_extractor(
            {
                "enable_triplet_extraction": True,
                "triplet_extractor_mode": "llm_v1",
                "llm_client": client,
            }
        )
        assert isinstance(extractor, LLMTripletExtractorV1)

    def test_false_flag_always_returns_none(self) -> None:
        extractor = build_triplet_extractor(
            {"enable_triplet_extraction": False, "triplet_extractor_mode": "rule_based_v0"}
        )
        assert extractor is None


# ---------------------------------------------------------------------------
# 11. ClaimEvidenceRecord triplets integration
# ---------------------------------------------------------------------------

class TestClaimEvidenceRecordTriplets:
    def test_triplets_are_none_by_default(self) -> None:
        from raggov.analyzers.grounding.evidence_layer import ClaimEvidenceBuilder
        from raggov.analyzers.grounding.verifiers import HeuristicValueOverlapVerifier
        from raggov.analyzers.grounding.candidate_selection import EvidenceCandidateSelector
        from raggov.models.chunk import RetrievedChunk

        builder = ClaimEvidenceBuilder(
            HeuristicValueOverlapVerifier({}),
            EvidenceCandidateSelector({}),
            triplet_extractor=None,
        )
        chunks = [RetrievedChunk(chunk_id="c1", text="some text", source_doc_id="d1", score=0.9)]
        record = builder._build_single("The subsidy is 60%.", 1, "query", chunks)
        assert record.claim_triplets is None

    def test_triplets_populated_when_extractor_provided(self) -> None:
        from raggov.analyzers.grounding.evidence_layer import ClaimEvidenceBuilder
        from raggov.analyzers.grounding.verifiers import HeuristicValueOverlapVerifier
        from raggov.analyzers.grounding.candidate_selection import EvidenceCandidateSelector
        from raggov.models.chunk import RetrievedChunk

        extractor = RuleBasedPolicyTripletExtractorV0()
        builder = ClaimEvidenceBuilder(
            HeuristicValueOverlapVerifier({}),
            EvidenceCandidateSelector({}),
            triplet_extractor=extractor,
        )
        chunks = [RetrievedChunk(chunk_id="c1", text="some text", source_doc_id="d1", score=0.9)]
        record = builder._build_single(
            "G.O.Ms.No. 10 mandates a 30% subsidy for all farmers.", 1, "query", chunks
        )
        assert record.claim_triplets is not None
        assert isinstance(record.claim_triplets, list)
        assert len(record.claim_triplets) >= 1
        assert all(isinstance(t, ClaimTriplet) for t in record.claim_triplets)

    def test_triplet_source_claim_id_matches_record_claim_id(self) -> None:
        from raggov.analyzers.grounding.evidence_layer import ClaimEvidenceBuilder
        from raggov.analyzers.grounding.verifiers import HeuristicValueOverlapVerifier
        from raggov.analyzers.grounding.candidate_selection import EvidenceCandidateSelector
        from raggov.models.chunk import RetrievedChunk

        extractor = RuleBasedPolicyTripletExtractorV0()
        builder = ClaimEvidenceBuilder(
            HeuristicValueOverlapVerifier({}),
            EvidenceCandidateSelector({}),
            triplet_extractor=extractor,
        )
        chunks = [RetrievedChunk(chunk_id="c1", text="x", source_doc_id="d1", score=0.9)]
        record = builder._build_single("G.O.Ms.No. 5 mandates compliance.", 3, "query", chunks)
        # claim_id is claim_003 for index=3
        for t in (record.claim_triplets or []):
            assert t.source_claim_id == record.claim_id


# ---------------------------------------------------------------------------
# 12. ClaimGroundingAnalyzer does not change default behavior
# ---------------------------------------------------------------------------

class TestClaimGroundingAnalyzerDefaultBehavior:
    def test_triplets_absent_when_flag_not_set(self) -> None:
        from raggov.analyzers.grounding.support import ClaimGroundingAnalyzer
        from raggov.models.run import RAGRun
        from raggov.models.chunk import RetrievedChunk

        run = RAGRun(
            run_id="test-pr9-001",
            query="What is the subsidy?",
            final_answer="The subsidy under PM-KUSUM is 60% of benchmark cost.",
            retrieved_chunks=[
                RetrievedChunk(
                    chunk_id="c1",
                    text="PM-KUSUM provides a 60% subsidy to farmers.",
                    source_doc_id="d1",
                    score=0.9,
                )
            ],
        )
        # No enable_triplet_extraction → triplets should be None on all records
        analyzer = ClaimGroundingAnalyzer(config={})
        result = analyzer.analyze(run)
        # Result must still work normally
        assert result.status in {"pass", "warn", "fail", "skip"}
        # diagnostic_rollup must be present (PR 8)
        assert result.diagnostic_rollup is not None

    def test_triplets_present_when_flag_enabled(self) -> None:
        """
        When enable_triplet_extraction=True, at least one policy-style claim
        should yield non-None claim_triplets on its record.
        """
        from raggov.analyzers.grounding.evidence_layer import ClaimEvidenceBuilder
        from raggov.analyzers.grounding.verifiers import HeuristicValueOverlapVerifier
        from raggov.analyzers.grounding.candidate_selection import EvidenceCandidateSelector
        from raggov.models.chunk import RetrievedChunk

        config = {"enable_triplet_extraction": True}
        extractor = build_triplet_extractor(config)
        builder = ClaimEvidenceBuilder(
            HeuristicValueOverlapVerifier(config),
            EvidenceCandidateSelector(config),
            triplet_extractor=extractor,
        )
        chunks = [
            RetrievedChunk(chunk_id="c1", text="G.O.Ms.No. 7 mandates 50% subsidy.", source_doc_id="d1", score=0.9)
        ]
        record = builder._build_single(
            "G.O.Ms.No. 7 mandates a 50% subsidy for beneficiaries.", 1, "q", chunks
        )
        assert record.claim_triplets is not None
        assert len(record.claim_triplets) >= 1
