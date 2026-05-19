from __future__ import annotations

import json
from pathlib import Path

from evals.claim_grounding.run_eval import run_eval
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


class SequenceLLMClient:
    def __init__(self, responses: list[object] | object) -> None:
        if isinstance(responses, list):
            self._responses = list(responses)
        else:
            self._responses = [responses]
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> object:
        self.prompts.append(prompt)
        if not self._responses:
            raise RuntimeError("No more scripted responses available.")
        return self._responses.pop(0)


def test_llm_verifier_marks_fully_supported_claim_as_supported() -> None:
    client = SequenceLLMClient(
        json.dumps(
            {
                "support_label": "supported",
                "support_reason": "Chunk c1 directly states the full claim.",
                "supporting_candidate_ids": ["c1"],
                "contradicting_candidate_ids": [],
                "neutral_candidate_ids": [],
                "verifier_warnings": [],
            }
        )
    )
    analyzer = ClaimGroundingAnalyzer(
        {
            "claim_verifier_mode": "llm_entailment",
            "llm_client": client,
        }
    )
    run = RAGRun(
        query="What is the capital of France?",
        retrieved_chunks=[
            chunk("c1", "Paris is the capital of France."),
            chunk("c2", "France is a country in Europe."),
        ],
        final_answer="Paris is the capital of France.",
    )

    result = analyzer.analyze(run)
    record = result.grounding_evidence_bundle.claim_evidence_records[0]

    assert record.support_label == "supported"
    assert record.verifier_method == "llm_claim_entailment_verifier_v1"
    assert record.supporting_candidate_ids == ["c1"]
    assert record.fallback_used is False


def test_llm_verifier_marks_contradicted_numeric_claim_as_contradicted() -> None:
    client = SequenceLLMClient(
        json.dumps(
            {
                "support_label": "contradicted",
                "support_reason": "Chunk c1 states 5%, which conflicts with the claim's 4%.",
                "supporting_candidate_ids": [],
                "contradicting_candidate_ids": ["c1"],
                "neutral_candidate_ids": [],
                "verifier_warnings": [],
            }
        )
    )
    analyzer = ClaimGroundingAnalyzer(
        {
            "claim_verifier_mode": "llm_entailment",
            "llm_client": client,
        }
    )
    run = RAGRun(
        query="What is the interest rate?",
        retrieved_chunks=[chunk("c1", "The interest rate is 5%.")],
        final_answer="The interest rate is 4%.",
    )

    result = analyzer.analyze(run)
    record = result.grounding_evidence_bundle.claim_evidence_records[0]

    assert record.support_label == "contradicted"
    assert record.contradicting_candidate_ids == ["c1"]


def test_llm_verifier_marks_partial_evidence_as_insufficient() -> None:
    client = SequenceLLMClient(
        json.dumps(
            {
                "support_label": "insufficient_evidence",
                "support_reason": "Chunk c1 covers applicability, but no chunk supports the $5,000 ceiling.",
                "supporting_candidate_ids": [],
                "contradicting_candidate_ids": [],
                "neutral_candidate_ids": ["c1"],
                "verifier_warnings": ["partial_support_detected"],
            }
        )
    )
    analyzer = ClaimGroundingAnalyzer(
        {
            "claim_verifier_mode": "llm_entailment",
            "llm_client": client,
        }
    )
    run = RAGRun(
        query="What is the reimbursement rule?",
        retrieved_chunks=[chunk("c1", "Heart surgery reimbursement applies to government employees.")],
        final_answer="Government employees are eligible for heart surgery reimbursement up to $5,000.",
    )

    result = analyzer.analyze(run)
    record = result.grounding_evidence_bundle.claim_evidence_records[0]

    assert record.support_label == "insufficient_evidence"
    assert record.neutral_candidate_ids == ["c1"]


def test_llm_verifier_returns_supporting_candidate_ids() -> None:
    client = SequenceLLMClient(
        json.dumps(
            {
                "support_label": "supported",
                "support_reason": "Chunks c1 and c2 jointly support the claim.",
                "supporting_candidate_ids": ["c1", "c2"],
                "contradicting_candidate_ids": [],
                "neutral_candidate_ids": [],
                "verifier_warnings": [],
            }
        )
    )
    analyzer = ClaimGroundingAnalyzer(
        {
            "claim_verifier_mode": "llm_entailment",
            "llm_client": client,
        }
    )
    run = RAGRun(
        query="What is the reimbursement rule?",
        retrieved_chunks=[
            chunk("c1", "Heart surgery reimbursement applies to government employees."),
            chunk("c2", "The reimbursement ceiling is $5,000 for heart surgery."),
        ],
        final_answer="Government employees are eligible for heart surgery reimbursement up to $5,000.",
    )

    result = analyzer.analyze(run)
    record = result.grounding_evidence_bundle.claim_evidence_records[0]

    assert record.supporting_candidate_ids == ["c1", "c2"]


def test_malformed_llm_verifier_output_triggers_visible_fallback() -> None:
    client = SequenceLLMClient(["{not json", "{still not json"])
    analyzer = ClaimGroundingAnalyzer(
        {
            "claim_verifier_mode": "llm_entailment",
            "llm_client": client,
        }
    )
    run = RAGRun(
        query="What is the capital of France?",
        retrieved_chunks=[chunk("c1", "Paris is the capital of France.")],
        final_answer="Paris is the capital of France.",
    )

    result = analyzer.analyze(run)
    record = result.grounding_evidence_bundle.claim_evidence_records[0]

    assert record.fallback_used is True
    assert record.fallback_from == "llm_entailment_verifier"
    assert record.fallback_to == "heuristic_top_k_verifier"
    assert record.verifier_method != "llm_claim_entailment_verifier_v1"


def test_heuristic_verifier_is_used_only_when_no_llm_verifier_is_configured() -> None:
    analyzer = ClaimGroundingAnalyzer(
        {
            "claim_verifier_mode": "llm_entailment",
        }
    )
    run = RAGRun(
        query="What is the capital of France?",
        retrieved_chunks=[chunk("c1", "Paris is the capital of France.")],
        final_answer="Paris is the capital of France.",
    )

    result = analyzer.analyze(run)
    record = result.grounding_evidence_bundle.claim_evidence_records[0]

    assert record.verifier_method in {
        "value_aware_structured_claim_verifier_v1",
        "deterministic_overlap_anchor_v0",
    }
    assert record.fallback_used is False


def test_structured_false_pass_rate_improves_over_historical_baseline() -> None:
    metrics = run_eval(
        dataset_path=Path("evals/claim_grounding/structured_cases.jsonl"),
        verifier_mode="heuristic",
    )

    assert metrics["false_pass_rate"] < 0.3333
    assert metrics["contradiction_detection_rate"] >= 1.0
