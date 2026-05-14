from __future__ import annotations

from raggov.analyzers.grounding.claims import ClaimExtractor
from raggov.analyzers.grounding.evidence_layer import detect_claim_type
from raggov.analyzers.grounding.triplets import (
    GovernmentPolicyTripletExtractorV0,
    GenericRuleTripletExtractorV0,
    build_triplet_extractor,
)
from raggov.analyzers.grounding.value_extraction import extract_value_mentions
from raggov.analyzers.sufficiency.sufficiency import SufficiencyAnalyzer


class CapturingClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def chat(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return '{"required_evidence": []}'


def test_software_version_not_treated_as_go_number() -> None:
    mentions = extract_value_mentions("Go 1.22 deprecates package path v2.0.0.")

    assert all(mention.value_type != "go_number" for mention in mentions)
    assert any(mention.value_type == "version" for mention in mentions)


def test_product_manual_section_not_government_order_section() -> None:
    assert detect_claim_type("Section 4.2 explains filter replacement.") != "go_number"
    assert detect_claim_type("Section 4.2 explains filter replacement.") == "procedural_assertion"


def test_healthcare_effective_date_uses_generic_lifecycle_prompt() -> None:
    client = CapturingClient()

    SufficiencyAnalyzer({"llm_client": client})._extract_evidence_requirements(
        "What dose applies after the guideline effective date?"
    )

    prompt = client.prompts[0].lower()
    assert "generic rag system" in prompt
    assert "government document rag system" not in prompt
    assert "specific go number" not in prompt


def test_finance_expired_disclosure_uses_generic_expiry() -> None:
    analyzer = SufficiencyAnalyzer({"llm_client": CapturingClient()})

    reqs = analyzer._extract_evidence_requirements(
        "Which fee disclosure applies after the expiry date?"
    )

    assert reqs == []


def test_scientific_citation_not_government_citation() -> None:
    assert detect_claim_type("Smith et al. 2024 reported a 12% improvement.") == "value_assertion"


def test_generic_mode_does_not_emit_government_policy_triplets() -> None:
    extractor = build_triplet_extractor({"enable_triplet_extraction": True})

    assert isinstance(extractor, GenericRuleTripletExtractorV0)
    assert not isinstance(extractor, GovernmentPolicyTripletExtractorV0)


def test_generic_mode_does_not_boost_scheme_terms() -> None:
    extractor = build_triplet_extractor({"enable_triplet_extraction": True})
    triplets = extractor.extract("The scheme requires Rs. 500 deposit.", "claim_001")

    assert all(t.extraction_method != "rule_based_policy_v0" for t in triplets)


def test_government_terms_not_required_for_claim_extraction() -> None:
    answer = "The SDK supports retries in version 2.4.1 when timeout errors occur."

    claims = ClaimExtractor().extract(answer)

    assert claims == [answer]


def test_no_default_import_of_government_policy_extractor_in_core_grounding() -> None:
    extractor = build_triplet_extractor({"enable_triplet_extraction": True})

    assert extractor.__class__.__name__ != "GovernmentPolicyTripletExtractorV0"
