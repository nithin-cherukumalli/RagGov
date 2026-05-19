from __future__ import annotations

import json

import pytest

from raggov.analyzers.grounding.claims import (
    ClaimExtractor,
    HeuristicClaimExtractorV0,
    LLMAtomicClaimExtractorV1,
)
from raggov.analyzers.grounding.support import ClaimGroundingAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.run import RAGRun


def chunk(chunk_id: str, text: str, *, doc_id: str | None = None, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=doc_id or f"doc-{chunk_id}",
        score=score,
    )


class StaticLLMClient:
    def __init__(self, response: object) -> None:
        self.response = response

    def complete(self, prompt: str) -> object:
        return self.response


def test_llm_extractor_splits_compound_sentence_into_atomic_claims() -> None:
    answer = "Revenue grew 15% in 2024 and the CEO resigned in March 2024."
    payload = [
        {
            "claim_id": "claim_1",
            "claim_text": "Revenue grew 15% in 2024.",
            "source_sentence": answer,
            "source_start_char": 0,
            "source_end_char": len(answer),
            "atomicity_status": "atomic",
            "claim_type": "comparison",
            "entities": ["Revenue"],
            "dates": ["2024"],
            "numbers": ["15%"],
            "should_verify": True,
            "extraction_method": "llm_atomic_claim_extractor_v1",
            "extraction_confidence": None,
            "extraction_warnings": [],
            "skip_reason": None,
        },
        {
            "claim_id": "claim_2",
            "claim_text": "The CEO resigned in March 2024.",
            "source_sentence": answer,
            "source_start_char": 0,
            "source_end_char": len(answer),
            "atomicity_status": "atomic",
            "claim_type": "entity_attribute",
            "entities": ["CEO"],
            "dates": ["March 2024"],
            "numbers": [],
            "should_verify": True,
            "extraction_method": "llm_atomic_claim_extractor_v1",
            "extraction_confidence": None,
            "extraction_warnings": [],
            "skip_reason": None,
        },
    ]

    extractor = LLMAtomicClaimExtractorV1(llm_client=StaticLLMClient(json.dumps(payload)))
    claims = extractor.extract_structured(answer)

    assert [claim.claim_text for claim in claims] == [
        "Revenue grew 15% in 2024.",
        "The CEO resigned in March 2024.",
    ]
    assert all(claim.atomicity_status == "atomic" for claim in claims)


def test_llm_extractor_preserves_source_sentence() -> None:
    answer = "Paris is the capital of France."
    payload = [
        {
            "claim_id": "claim_1",
            "claim_text": "Paris is the capital of France.",
            "source_sentence": answer,
            "source_start_char": 0,
            "source_end_char": len(answer),
            "atomicity_status": "atomic",
            "claim_type": "entity_attribute",
            "entities": ["Paris", "France"],
            "dates": [],
            "numbers": [],
            "should_verify": True,
            "extraction_method": "llm_atomic_claim_extractor_v1",
            "extraction_confidence": None,
            "extraction_warnings": [],
            "skip_reason": None,
        }
    ]

    extractor = LLMAtomicClaimExtractorV1(llm_client=StaticLLMClient(json.dumps(payload)))
    claim = extractor.extract_structured(answer)[0]

    assert claim.source_sentence == answer
    assert claim.source_start_char == 0
    assert claim.source_end_char == len(answer)


def test_llm_extractor_accepts_object_wrapped_claim_array() -> None:
    answer = "Paris is the capital of France."
    payload = {
        "claims": [
            {
                "claim_id": "claim_1",
                "claim_text": "Paris is the capital of France.",
                "source_sentence": answer,
                "source_start_char": 0,
                "source_end_char": len(answer),
                "atomicity_status": "atomic",
                "claim_type": "entity_attribute",
                "entities": ["Paris", "France"],
                "dates": [],
                "numbers": [],
                "should_verify": True,
                "extraction_method": "llm_atomic_claim_extractor_v1",
                "extraction_confidence": None,
                "extraction_warnings": [],
                "skip_reason": None,
            }
        ]
    }

    extractor = LLMAtomicClaimExtractorV1(llm_client=StaticLLMClient(json.dumps(payload)))
    claims = extractor.extract_structured(answer)

    assert len(claims) == 1
    assert claims[0].claim_text == answer


def test_llm_extractor_does_not_invent_entities_numbers_or_dates() -> None:
    answer = "Revenue grew 15% in 2024."
    payload = [
        {
            "claim_id": "claim_1",
            "claim_text": "Revenue grew 17% in 2025.",
            "source_sentence": answer,
            "source_start_char": 0,
            "source_end_char": len(answer),
            "atomicity_status": "atomic",
            "claim_type": "comparison",
            "entities": ["Revenue"],
            "dates": ["2025"],
            "numbers": ["17%"],
            "should_verify": True,
            "extraction_method": "llm_atomic_claim_extractor_v1",
            "extraction_confidence": None,
            "extraction_warnings": [],
            "skip_reason": None,
        }
    ]

    extractor = LLMAtomicClaimExtractorV1(llm_client=StaticLLMClient(json.dumps(payload)))
    claim = extractor.extract_structured(answer)[0]

    assert claim.should_verify is False
    assert claim.skip_reason == "claim_not_grounded_in_answer"
    assert "possible_extraction_hallucination" in claim.extraction_warnings


def test_malformed_llm_json_triggers_visible_fallback() -> None:
    answer = "Paris is the capital of France."
    extractor = ClaimExtractor(
        llm_client=StaticLLMClient("{not valid json"),
    )

    claims = extractor.extract_structured(answer)

    assert claims
    assert claims[0].extraction_method == "llm_fallback"
    assert claims[0].extraction_warnings


def test_no_llm_configured_triggers_heuristic_fallback_visibly() -> None:
    extractor = HeuristicClaimExtractorV0()
    claim = extractor.extract_structured("Paris is the capital of France.")[0]

    assert claim.extraction_method == "heuristic_atomic_claim_extractor_v0"


def test_structured_claim_metadata_survives_grounding() -> None:
    answer = "Paris is the capital of France."
    payload = [
        {
            "claim_id": "claim_abc",
            "claim_text": "Paris is the capital of France.",
            "source_sentence": answer,
            "source_start_char": 0,
            "source_end_char": len(answer),
            "atomicity_status": "atomic",
            "claim_type": "entity_attribute",
            "entities": ["Paris", "France"],
            "dates": [],
            "numbers": [],
            "should_verify": True,
            "extraction_method": "llm_atomic_claim_extractor_v1",
            "extraction_confidence": None,
            "extraction_warnings": [],
            "skip_reason": None,
        }
    ]
    analyzer = ClaimGroundingAnalyzer(
        {
            "claim_extractor_client": StaticLLMClient(json.dumps(payload)),
        }
    )
    run = RAGRun(
        query="What is the capital of France?",
        retrieved_chunks=[chunk("c1", "Paris is the capital of France.")],
        final_answer=answer,
    )

    result = analyzer.analyze(run)
    record = result.grounding_evidence_bundle.claim_evidence_records[0]

    assert record.claim_id == "claim_abc"
    assert record.source_sentence == answer
    assert record.source_answer_span == (0, len(answer))
    assert record.atomicity_status == "atomic"
    assert record.claim_type == "entity_attribute"
    assert record.extraction_method == "llm_atomic_claim_extractor_v1"


def test_short_factual_claim_is_not_skipped() -> None:
    extractor = HeuristicClaimExtractorV0()
    claim = extractor.extract_structured("Paris is the capital of France.")[0]
    assert claim.should_verify is True


def test_generic_non_government_claim_is_extracted() -> None:
    answer = "The Eiffel Tower is in Paris."
    payload = [
        {
            "claim_id": "claim_1",
            "claim_text": answer,
            "source_sentence": answer,
            "source_start_char": 0,
            "source_end_char": len(answer),
            "atomicity_status": "atomic",
            "claim_type": "entity_attribute",
            "entities": ["Eiffel Tower", "Paris"],
            "dates": [],
            "numbers": [],
            "should_verify": True,
            "extraction_method": "llm_atomic_claim_extractor_v1",
            "extraction_confidence": None,
            "extraction_warnings": [],
            "skip_reason": None,
        }
    ]
    extractor = LLMAtomicClaimExtractorV1(llm_client=StaticLLMClient(json.dumps(payload)))
    claim = extractor.extract_structured(answer)[0]

    assert claim.claim_text == answer
    assert claim.claim_type == "entity_attribute"


def test_top_k_multichunk_evidence_can_support_a_claim() -> None:
    analyzer = ClaimGroundingAnalyzer()
    run = RAGRun(
        query="What are the reimbursement rules?",
        retrieved_chunks=[
            chunk("c1", "Heart surgery reimbursement applies to government employees."),
            chunk("c2", "The reimbursement ceiling is $5,000 for heart surgery."),
            chunk("c3", "Travel reimbursement ceiling is $500."),
        ],
        final_answer="Government employees are eligible for heart surgery reimbursement up to $5,000.",
    )

    result = analyzer.analyze(run)
    record = result.grounding_evidence_bundle.claim_evidence_records[0]

    assert record.support_label == "supported"
    assert record.evidence_mode == "multi_chunk"
    assert set(record.supporting_candidate_ids) >= {"c1", "c2"}


def test_contradiction_not_overridden_by_weak_lexical_support() -> None:
    analyzer = ClaimGroundingAnalyzer()
    run = RAGRun(
        query="What is the interest rate?",
        retrieved_chunks=[
            chunk("c1", "Interest rate is 5%."),
            chunk("c2", "Interest rate is 4% effective immediately."),
        ],
        final_answer="The interest rate is 4%.",
    )

    result = analyzer.analyze(run)
    record = result.grounding_evidence_bundle.claim_evidence_records[0]

    assert record.support_label == "contradicted"
    assert "c1" in record.contradicting_candidate_ids


def test_heuristic_raw_score_is_not_marked_calibrated_confidence() -> None:
    analyzer = ClaimGroundingAnalyzer()
    run = RAGRun(
        query="What is the capital of France?",
        retrieved_chunks=[chunk("c1", "Paris is the capital of France.")],
        final_answer="Paris is the capital of France.",
    )

    result = analyzer.analyze(run)
    record = result.grounding_evidence_bundle.claim_evidence_records[0]

    assert record.calibrated_confidence is None
    assert record.confidence_status == "uncalibrated_heuristic_proxy"


def test_wrong_cited_document_is_exposed_in_claim_evidence() -> None:
    analyzer = ClaimGroundingAnalyzer()
    run = RAGRun(
        query="What is the interest rate?",
        retrieved_chunks=[
            chunk("c1", "Interest rate is 5%.", doc_id="doc-A"),
            chunk("c2", "A tourism brochure.", doc_id="doc-B"),
        ],
        final_answer="The interest rate is 5% [c2].",
    )

    result = analyzer.analyze(run)
    record = result.grounding_evidence_bundle.claim_evidence_records[0]

    assert record.cited_chunk_ids == ["c2"]
    assert record.cited_doc_ids == ["doc-B"]
    assert record.best_supporting_doc_id == "doc-A"
    assert record.support_source_type == "retrieved_uncited_chunk"
