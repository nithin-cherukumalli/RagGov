"""Tests for pinpoint data models: serialization, defaults honesty."""

from __future__ import annotations

from raggov.models.pinpoint import (
    CausalChain,
    PinpointEvidence,
    PinpointFinding,
    PinpointLocation,
    TrustDecision,
)


def _make_location(**kwargs) -> PinpointLocation:
    defaults = dict(
        location_id="loc-1",
        localization_method="ncv_first_failing_node_v1",
        method_type="evidence_aggregation",
        calibration_status="uncalibrated",
    )
    defaults.update(kwargs)
    return PinpointLocation(**defaults)


def _make_evidence(**kwargs) -> PinpointEvidence:
    defaults = dict(
        signal_name="test_signal",
        value="does_not_support",
        source_report="CitationFaithfulnessReport",
        interpretation="Citation does not support the claim.",
        method_type="evidence_aggregation",
        calibration_status="uncalibrated",
    )
    defaults.update(kwargs)
    return PinpointEvidence(**defaults)


def _make_finding(**kwargs) -> PinpointFinding:
    defaults = dict(
        finding_id="finding-1",
        location=_make_location(),
    )
    defaults.update(kwargs)
    return PinpointFinding(**defaults)


class TestPinpointLocationSerialization:
    def test_round_trip_preserves_fields(self):
        loc = _make_location(
            ncv_node="citation_support",
            claim_ids=["claim-1"],
            chunk_ids=["chunk-1"],
            doc_ids=["doc-1"],
            limitations=["uncalibrated"],
        )
        dumped = loc.model_dump(mode="json")
        restored = PinpointLocation(**dumped)

        assert restored.location_id == "loc-1"
        assert restored.ncv_node == "citation_support"
        assert restored.claim_ids == ["claim-1"]
        assert restored.chunk_ids == ["chunk-1"]
        assert restored.doc_ids == ["doc-1"]
        assert restored.limitations == ["uncalibrated"]

    def test_optional_fields_default_to_none_or_empty(self):
        loc = _make_location()
        assert loc.pipeline_stage is None
        assert loc.ncv_node is None
        assert loc.failure_type is None
        assert loc.claim_ids == []
        assert loc.citation_ids == []
        assert loc.source_ids == []


class TestPinpointEvidenceSerialization:
    def test_round_trip_preserves_fields(self):
        ev = _make_evidence(
            label="unsupported",
            provider="external_nli_v1",
            affected_claim_ids=["claim-1"],
            affected_chunk_ids=["chunk-1"],
        )
        dumped = ev.model_dump(mode="json")
        restored = PinpointEvidence(**dumped)

        assert restored.signal_name == "test_signal"
        assert restored.label == "unsupported"
        assert restored.provider == "external_nli_v1"
        assert restored.affected_claim_ids == ["claim-1"]
        assert restored.affected_chunk_ids == ["chunk-1"]

    def test_value_accepts_any_type(self):
        assert _make_evidence(value=0.95).value == 0.95
        assert _make_evidence(value=True).value is True
        assert _make_evidence(value="retrieval_miss").value == "retrieval_miss"
        assert _make_evidence(value=None).value is None


class TestPinpointFindingSerialization:
    def test_round_trip_preserves_all_fields(self):
        loc = _make_location(ncv_node="retrieval_coverage")
        ev_for = _make_evidence(signal_name="coverage_signal")
        finding = PinpointFinding(
            finding_id="finding-rt",
            location=loc,
            evidence_for=[ev_for],
            evidence_against=[],
            missing_evidence=["retrieval_diagnosis_report"],
            fallback_heuristics_used=["mean_retrieval_score_threshold"],
            heuristic_score=0.4,
            calibrated_confidence=None,
            calibration_status="uncalibrated",
            human_review_recommended=True,
            recommended_for_gating=False,
        )

        dumped = finding.model_dump(mode="json")
        restored = PinpointFinding(**dumped)

        assert restored.finding_id == "finding-rt"
        assert restored.location.ncv_node == "retrieval_coverage"
        assert restored.evidence_for[0].signal_name == "coverage_signal"
        assert restored.missing_evidence == ["retrieval_diagnosis_report"]
        assert restored.fallback_heuristics_used == ["mean_retrieval_score_threshold"]
        assert restored.heuristic_score == 0.4
        assert restored.calibrated_confidence is None

    def test_serialized_dict_contains_expected_keys(self):
        finding = _make_finding()
        d = finding.model_dump()
        required_keys = {
            "finding_id", "location", "evidence_for", "evidence_against",
            "missing_evidence", "fallback_heuristics_used", "alternative_locations",
            "heuristic_score", "calibrated_confidence", "calibration_status",
            "human_review_recommended", "recommended_for_gating",
        }
        assert required_keys.issubset(d.keys())


class TestPinpointDefaultsHonesty:
    def test_recommended_for_gating_defaults_false(self):
        assert _make_finding().recommended_for_gating is False
        assert _make_location().recommended_for_gating is False

    def test_calibrated_confidence_defaults_none(self):
        assert _make_finding().calibrated_confidence is None

    def test_calibration_status_defaults_uncalibrated(self):
        assert _make_finding().calibration_status == "uncalibrated"

    def test_human_review_recommended_defaults_true(self):
        assert _make_finding().human_review_recommended is True

    def test_evidence_lists_default_empty(self):
        f = _make_finding()
        assert f.evidence_for == []
        assert f.evidence_against == []
        assert f.missing_evidence == []
        assert f.fallback_heuristics_used == []
        assert f.alternative_locations == []

    def test_heuristic_score_defaults_none(self):
        assert _make_finding().heuristic_score is None


class TestCausalChainSerialization:
    def test_defaults_are_honest(self):
        chain = CausalChain(
            chain_id="chain-1",
            root_location=_make_location(),
            causal_hypothesis="Retrieval miss caused unsupported claim.",
            abduct="Claim is unsupported because retrieval missed the source doc.",
            act="Expand retrieval depth.",
            predict="Claim would be supported after retrieval fix.",
        )
        assert chain.calibrated_confidence is None
        assert chain.calibration_status == "uncalibrated"
        assert chain.downstream_locations == []
        assert chain.heuristic_score is None

    def test_round_trip(self):
        chain = CausalChain(
            chain_id="chain-rt",
            root_location=_make_location(ncv_node="retrieval_coverage"),
            causal_hypothesis="Retrieval miss.",
            abduct="A",
            act="B",
            predict="C",
        )
        d = chain.model_dump(mode="json")
        restored = CausalChain(**d)
        assert restored.chain_id == "chain-rt"
        assert restored.root_location.ncv_node == "retrieval_coverage"


class TestTrustDecisionSerialization:
    def test_round_trip(self):
        td = TrustDecision(
            decision="human_review",
            reason="Uncalibrated heuristic only.",
            recommended_for_gating=False,
            human_review_required=True,
            blocking_eligible=False,
            calibration_status="uncalibrated",
        )
        d = td.model_dump(mode="json")
        restored = TrustDecision(**d)
        assert restored.decision == "human_review"
        assert restored.recommended_for_gating is False
        assert restored.human_review_required is True
