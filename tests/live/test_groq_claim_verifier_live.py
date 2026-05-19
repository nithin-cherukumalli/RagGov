from __future__ import annotations

import json
import os

import pytest

from raggov.analyzers.grounding.claims import LLMAtomicClaimExtractorV1
from raggov.analyzers.grounding.candidate_selection import EvidenceCandidate
from raggov.analyzers.grounding.verifiers import LLMClaimEntailmentVerifierV1
from raggov.connectors.groq_client import build_groq_client_from_env


pytestmark = pytest.mark.live


def _require_groq():
    client, reason = build_groq_client_from_env()
    if client is None:
        pytest.skip(f"Groq live test skipped: {reason}")
    return client


def test_groq_connectivity_json() -> None:
    client = _require_groq()
    response = client.chat('Return exactly this JSON: {"status":"ok","provider":"groq"}')
    payload = json.loads(response)
    assert payload["status"] == "ok"
    assert payload["provider"] == "groq"


def test_groq_claim_extraction_two_atomic_claims() -> None:
    client = _require_groq()
    answer = (
        "The Andhra Pradesh order applies to government schools. "
        "It excludes private unaided schools."
    )
    claims = LLMAtomicClaimExtractorV1(client).extract_structured(answer)
    assert len(claims) >= 2
    texts = [claim.claim_text.lower() for claim in claims]
    assert any("government school" in text for text in texts)
    assert any("private unaided" in text for text in texts)


def test_groq_entailment_does_not_false_support_contradicted_claim() -> None:
    client = _require_groq()
    verifier = LLMClaimEntailmentVerifierV1({"llm_client": client})
    candidates = [
        EvidenceCandidate(
            chunk_id="chunk_1",
            source_doc_id="doc-1",
            chunk_text="The order applies to all government schools in Andhra Pradesh.",
            chunk_text_preview="The order applies to all government schools in Andhra Pradesh.",
            lexical_overlap_score=0.0,
            anchor_overlap_score=0.0,
            value_overlap_score=0.0,
            retrieval_score=0.9,
            rerank_score=None,
            candidate_reason="live test",
        ),
        EvidenceCandidate(
            chunk_id="chunk_2",
            source_doc_id="doc-2",
            chunk_text="Private unaided schools are excluded from this order.",
            chunk_text_preview="Private unaided schools are excluded from this order.",
            lexical_overlap_score=0.0,
            anchor_overlap_score=0.0,
            value_overlap_score=0.0,
            retrieval_score=0.88,
            rerank_score=None,
            candidate_reason="live test",
        ),
    ]
    result = verifier.verify(
        "The order applies to private unaided schools.",
        "What schools are covered?",
        candidates,
        metadata={
            "source_sentence": "The order applies to private unaided schools.",
            "claim_type": "policy_rule",
            "entities": ["private unaided schools"],
            "dates": [],
            "numbers": [],
            "atomicity_status": "atomic",
            "cited_doc_ids": [],
            "cited_chunk_ids": [],
        },
    )
    assert result.support_label in {"contradicted", "insufficient_evidence"}
    assert result.support_label != "supported"
