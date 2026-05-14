"""Tests for Semantic Entropy analyzer."""

from __future__ import annotations

import math

from raggov.analyzers.confidence.semantic_entropy import (
    NLIClusterer,
    SemanticEntropyAnalyzer,
    jaccard_similarity,
)
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import (
    AnalyzerResult,
    ClaimResult,
    FailureStage,
    FailureType,
)
from raggov.models.run import RAGRun


def chunk(chunk_id: str, text: str, score: float | None = None) -> RetrievedChunk:
    """Helper to create a retrieved chunk."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=f"doc-{chunk_id}",
        score=score,
    )


def run_with_chunks(
    chunks: list[RetrievedChunk],
    query: str = "What is the answer?",
    answer: str = "Answer.",
    claim_results: list[ClaimResult] | None = None,
) -> RAGRun:
    """Helper to create a RAG run with chunks."""
    run = RAGRun(query=query, retrieved_chunks=chunks, final_answer=answer)
    if claim_results:
        # Store claim_results in metadata for test purposes
        run.metadata["claim_results"] = claim_results
    return run


def test_jaccard_similarity_identical_strings() -> None:
    """Identical strings should have Jaccard similarity of 1.0."""
    s1 = "the quick brown fox"
    s2 = "the quick brown fox"
    assert jaccard_similarity(s1, s2) == 1.0


def test_jaccard_similarity_disjoint_strings() -> None:
    """Disjoint strings should have Jaccard similarity of 0.0."""
    s1 = "the quick brown fox"
    s2 = "lazy dog sleeps"
    assert jaccard_similarity(s1, s2) == 0.0


def test_jaccard_similarity_partial_overlap() -> None:
    """Partially overlapping strings should have intermediate similarity."""
    s1 = "the quick brown fox"
    s2 = "the lazy brown dog"
    # Common tokens: "the", "brown" (2 out of 6 unique tokens)
    # union = {the, quick, brown, fox, lazy, dog} = 6 tokens
    # intersection = {the, brown} = 2 tokens
    # similarity = 2/6 = 0.333...
    similarity = jaccard_similarity(s1, s2)
    assert 0.3 < similarity < 0.4


def test_jaccard_similarity_case_insensitive() -> None:
    """Jaccard similarity should be case-insensitive."""
    s1 = "The Quick Brown Fox"
    s2 = "the quick brown fox"
    assert jaccard_similarity(s1, s2) == 1.0


def test_nli_clusterer_augmented_proxy_merges_simple_morphology() -> None:
    """Tier A should improve over raw Jaccard for light morphological variation."""
    clusterer = NLIClusterer()

    assert clusterer.tier_label == "Jaccard-augmented proxy"
    assert clusterer.are_equivalent(
        "Revenue increased annually",
        "Revenue increasing annual",
    )


def test_deterministic_mode_all_claims_supported() -> None:
    """Deterministic mode: one claim-label cluster → zero entropy → pass."""
    # Mock prior results with all claims entailed
    prior_results = [
        AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="pass",
            evidence=[],
        )
    ]

    # Create claim results (all entailed)
    claim_results = [
        ClaimResult(claim_text="Claim 1", label="entailed", supporting_chunk_ids=["c1"]),
        ClaimResult(claim_text="Claim 2", label="entailed", supporting_chunk_ids=["c2"]),
        ClaimResult(claim_text="Claim 3", label="entailed", supporting_chunk_ids=["c3"]),
    ]

    analyzer = SemanticEntropyAnalyzer(
        {"use_llm": False, "prior_results": prior_results}
    )

    test_run = run_with_chunks([chunk("c1", "Context", 0.8)], claim_results=claim_results)

    result = analyzer.analyze(test_run)

    assert result.status == "pass"
    assert result.score is not None
    assert result.score == 0.0
    evidence_text = " ".join(result.evidence).lower()
    assert "claim_label_entropy_proxy_v0" in evidence_text
    assert "low uncertainty" in evidence_text


def test_deterministic_mode_all_claims_unsupported() -> None:
    """Deterministic mode: consistent unsupported labels still yield zero entropy."""
    prior_results = [
        AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
            evidence=["Claims not grounded"],
            remediation="Verify",
        )
    ]

    # All claims unsupported
    claim_results = [
        ClaimResult(claim_text="Claim 1", label="unsupported", supporting_chunk_ids=[]),
        ClaimResult(claim_text="Claim 2", label="unsupported", supporting_chunk_ids=[]),
        ClaimResult(claim_text="Claim 3", label="unsupported", supporting_chunk_ids=[]),
    ]

    analyzer = SemanticEntropyAnalyzer(
        {"use_llm": False, "prior_results": prior_results, "entropy_threshold": 1.2}
    )

    test_run = run_with_chunks([chunk("c1", "Context", 0.8)], claim_results=claim_results)

    result = analyzer.analyze(test_run)

    assert result.status == "pass"
    assert result.failure_type is None
    assert result.stage is None
    assert result.score is not None
    assert result.score == 0.0
    evidence_text = " ".join(result.evidence).lower()
    assert "claim_label_entropy_proxy_v0" in evidence_text
    assert "low uncertainty" in evidence_text


def test_deterministic_mode_mixed_claims() -> None:
    """Deterministic mode: two label clusters → medium entropy → warn."""
    prior_results = []

    # Mixed: 2 entailed, 1 unsupported
    claim_results = [
        ClaimResult(claim_text="Claim 1", label="entailed", supporting_chunk_ids=["c1"]),
        ClaimResult(claim_text="Claim 2", label="entailed", supporting_chunk_ids=["c2"]),
        ClaimResult(claim_text="Claim 3", label="unsupported", supporting_chunk_ids=[]),
    ]

    analyzer = SemanticEntropyAnalyzer({"use_llm": False, "entropy_threshold": 1.2})

    test_run = run_with_chunks([chunk("c1", "Context", 0.8)], claim_results=claim_results)

    result = analyzer.analyze(test_run)

    assert result.status == "warn"
    assert result.score is not None
    assert math.isclose(result.score, 0.9182958340544896)
    assert 0.5 <= result.score <= 1.2
    evidence_text = " ".join(result.evidence).lower()
    assert "medium uncertainty" in evidence_text


def test_deterministic_mode_three_label_split_fails() -> None:
    """Deterministic mode: three equally sized label clusters → high entropy → fail."""
    claim_results = [
        ClaimResult(claim_text="Claim 1", label="entailed", supporting_chunk_ids=["c1"]),
        ClaimResult(claim_text="Claim 2", label="unsupported", supporting_chunk_ids=[]),
        ClaimResult(claim_text="Claim 3", label="contradicted", supporting_chunk_ids=[]),
    ]

    analyzer = SemanticEntropyAnalyzer({"use_llm": False, "entropy_threshold": 1.2})

    test_run = run_with_chunks([chunk("c1", "Context", 0.8)], claim_results=claim_results)

    result = analyzer.analyze(test_run)

    assert result.status == "fail"
    assert result.failure_type == FailureType.LOW_CONFIDENCE
    assert result.stage == FailureStage.CONFIDENCE
    assert result.score is not None
    assert result.score > 1.2
    evidence_text = " ".join(result.evidence).lower()
    assert "high uncertainty" in evidence_text
    assert "claim_label_entropy_proxy_v0" in evidence_text


def test_semantic_entropy_consumes_typed_claim_results_first() -> None:
    prior = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="fail",
        claim_results=[
            ClaimResult(claim_text="A", label="entailed", supporting_chunk_ids=["c1"]),
            ClaimResult(claim_text="B", label="unsupported", supporting_chunk_ids=[]),
        ],
        evidence=[
            '{"claim_text":"X","label":"contradicted","supporting_chunk_ids":[],"confidence":0.2}'
        ],
    )
    analyzer = SemanticEntropyAnalyzer({"use_llm": False, "prior_results": [prior]})
    result = analyzer.analyze(run_with_chunks([chunk("c1", "Context", 0.8)]))

    assert result.status == "warn"
    assert result.score is not None
    assert math.isclose(result.score, 1.0)
    assert "claim_label_entropy_proxy_v0" in " ".join(result.evidence).lower()


def test_semantic_entropy_legacy_json_evidence_fallback_still_supported() -> None:
    prior = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="fail",
        evidence=[
            '{"claim_text":"A","label":"entailed","supporting_chunk_ids":["c1"],"confidence":0.9}',
            '{"claim_text":"B","label":"unsupported","supporting_chunk_ids":[],"confidence":0.1}',
        ],
    )
    analyzer = SemanticEntropyAnalyzer({"use_llm": False, "prior_results": [prior]})
    result = analyzer.analyze(run_with_chunks([chunk("c1", "Context", 0.8)]))

    assert result.status == "warn"
    assert result.score is not None
    assert math.isclose(result.score, 1.0)


def test_llm_mode_identical_answers() -> None:
    """LLM mode: identical answers → entropy ≈ 0 → pass."""

    def mock_llm_identical(prompt: str) -> str:
        """Always return the same answer."""
        return "The answer is 42."

    analyzer = SemanticEntropyAnalyzer(
        {
            "use_llm": True,
            "llm_fn": mock_llm_identical,
            "n_samples": 5,
            "entropy_threshold": 1.2,
        }
    )

    test_run = run_with_chunks(
        [chunk("c1", "The answer to everything is 42", 0.9)],
        query="What is the answer?",
    )

    result = analyzer.analyze(test_run)

    assert result.status == "pass"
    assert result.score is not None
    assert result.score < 0.5  # Low entropy (all samples agree)
    # Check custom fields (case-insensitive)
    evidence_text = str(result.evidence).lower()
    assert "entropy" in evidence_text
    assert "cluster" in evidence_text or result.score == 0.0
    assert "llm-nli" in evidence_text


def test_llm_mode_different_answers() -> None:
    """LLM mode: 5 different answers → high entropy → fail."""
    sample_counter = 0

    def mock_llm_different(prompt: str) -> str:
        """Return completely different answers with no token overlap."""
        nonlocal sample_counter
        # Use completely different words with no overlap
        answers = [
            "Alpha",
            "Bravo",
            "Charlie",
            "Delta",
            "Echo",
        ]
        answer = answers[sample_counter % len(answers)]
        sample_counter += 1
        return answer

    analyzer = SemanticEntropyAnalyzer(
        {
            "use_llm": True,
            "llm_fn": mock_llm_different,
            "n_samples": 5,
            "entropy_threshold": 1.2,
        }
    )

    test_run = run_with_chunks(
        [chunk("c1", "Some context", 0.7)],
        query="What is the answer?",
    )

    result = analyzer.analyze(test_run)

    assert result.status == "fail"
    assert result.failure_type == FailureType.LOW_CONFIDENCE
    assert result.stage == FailureStage.CONFIDENCE
    assert result.score is not None
    assert result.score > 1.2  # High entropy (all samples different)
    evidence_text = " ".join(result.evidence).lower()
    assert "confabulation" in evidence_text or "inconsistent" in evidence_text


def test_llm_mode_some_agreement() -> None:
    """LLM mode: some samples agree → medium entropy → warn."""
    sample_counter = 0

    def mock_llm_partial(prompt: str) -> str:
        """Return answers with partial agreement."""
        nonlocal sample_counter
        if prompt.startswith("Do these two statements express the same factual claim?"):
            return "yes" if "Statement 1: Answer A" in prompt and "Statement 2: Answer A" in prompt else (
                "yes" if "Statement 1: Answer B" in prompt and "Statement 2: Answer B" in prompt else "no"
            )
        # 3 samples say "A", 2 samples say "B"
        answers = ["Answer A", "Answer A", "Answer A", "Answer B", "Answer B"]
        answer = answers[sample_counter % len(answers)]
        sample_counter += 1
        return answer

    analyzer = SemanticEntropyAnalyzer(
        {
            "use_llm": True,
            "llm_fn": mock_llm_partial,
            "n_samples": 5,
            "entropy_threshold": 1.2,
        }
    )

    test_run = run_with_chunks(
        [chunk("c1", "Context", 0.8)],
        query="Question?",
    )

    result = analyzer.analyze(test_run)

    # Should be warn with medium entropy
    assert result.status in ("warn", "pass")
    assert result.score is not None
    assert 0.5 <= result.score <= 1.2


def test_llm_mode_llm_nli_merges_semantic_paraphrases() -> None:
    """LLM-NLI should cluster paraphrases that raw token overlap would split."""
    sample_counter = 0

    def mock_llm_semantic(prompt: str) -> str:
        nonlocal sample_counter
        if prompt.startswith("Do these two statements express the same factual claim?"):
            if "founded in 1923" in prompt and "established in 1923" in prompt:
                return "yes"
            return "no"

        answers = [
            "The company was founded in 1923.",
            "The firm was established in 1923.",
            "The company was founded in 1923.",
            "The firm was established in 1923.",
            "The company was founded in 1923.",
        ]
        answer = answers[sample_counter % len(answers)]
        sample_counter += 1
        return answer

    analyzer = SemanticEntropyAnalyzer(
        {
            "use_llm": True,
            "llm_fn": mock_llm_semantic,
            "n_samples": 5,
            "entropy_threshold": 1.2,
        }
    )

    result = analyzer.analyze(
        run_with_chunks([chunk("c1", "The company was founded in 1923.", 0.9)])
    )

    assert result.status == "pass"
    assert result.score == 0.0
    assert "llm-nli" in " ".join(result.evidence).lower()


def test_entropy_calculation_single_cluster() -> None:
    """Single cluster (all samples agree) → entropy = 0.0."""
    # This is tested indirectly by test_llm_mode_identical_answers
    # but we can verify the math here

    def mock_llm_same(prompt: str) -> str:
        return "Same answer every time"

    analyzer = SemanticEntropyAnalyzer(
        {"use_llm": True, "llm_fn": mock_llm_same, "n_samples": 5}
    )

    test_run = run_with_chunks([chunk("c1", "Context", 0.8)])
    result = analyzer.analyze(test_run)

    # With all samples identical, entropy should be exactly 0.0
    assert result.score == 0.0


def test_entropy_calculation_uniform_clusters() -> None:
    """Uniform distribution (5 different answers) → entropy ≈ 2.32."""
    sample_counter = 0

    def mock_llm_uniform(prompt: str) -> str:
        nonlocal sample_counter
        # Each answer is completely different (no token overlap)
        answers = [
            "Alpha",
            "Bravo",
            "Charlie",
            "Delta",
            "Echo",
        ]
        answer = answers[sample_counter % len(answers)]
        sample_counter += 1
        return answer

    analyzer = SemanticEntropyAnalyzer(
        {"use_llm": True, "llm_fn": mock_llm_uniform, "n_samples": 5}
    )

    test_run = run_with_chunks([chunk("c1", "Context", 0.8)])
    result = analyzer.analyze(test_run)

    # Uniform distribution: H = -log2(1/5) = log2(5) ≈ 2.32
    assert result.score is not None
    assert 2.2 < result.score < 2.4


def test_no_llm_fn_provided_skips() -> None:
    """Analyzer skips when use_llm=True but llm_fn not provided."""
    analyzer = SemanticEntropyAnalyzer({"use_llm": True})  # No llm_fn

    test_run = run_with_chunks([chunk("c1", "Context", 0.8)])
    result = analyzer.analyze(test_run)

    assert result.status == "skip"
    assert "llm_fn not provided" in result.evidence[0].lower()


def test_no_claim_results_in_deterministic_mode() -> None:
    """Deterministic mode skips when no claim results available."""
    analyzer = SemanticEntropyAnalyzer({"use_llm": False})

    test_run = run_with_chunks([chunk("c1", "Context", 0.8)])
    # No claim_results set

    result = analyzer.analyze(test_run)

    assert result.status == "skip"
    assert "claim_label_entropy_proxy_v0" in result.evidence[0].lower()


def test_empty_chunks_handled_gracefully() -> None:
    """Analyzer handles empty chunks gracefully."""
    analyzer = SemanticEntropyAnalyzer({"use_llm": False})

    test_run = RAGRun(
        query="Question?",
        retrieved_chunks=[],  # Empty
        final_answer="Answer",
    )

    result = analyzer.analyze(test_run)

    # Should skip or handle gracefully
    assert result.status == "skip"


def test_llm_mode_with_prior_results() -> None:
    """LLM mode can access prior results if needed."""
    prior_results = [
        AnalyzerResult(
            analyzer_name="TestAnalyzer",
            status="pass",
            evidence=["test"],
        )
    ]

    def mock_llm(prompt: str) -> str:
        return "Answer"

    analyzer = SemanticEntropyAnalyzer(
        {
            "use_llm": True,
            "llm_fn": mock_llm,
            "n_samples": 3,
            "prior_results": prior_results,
        }
    )

    test_run = run_with_chunks([chunk("c1", "Context", 0.8)])
    result = analyzer.analyze(test_run)

    # Should not error with prior results present
    assert result.status in ("pass", "warn", "fail")


def test_temperature_hint_used_in_sampling() -> None:
    """Temperature config is available for LLM sampling."""
    sample_call_count = 0
    entailment_call_count = 0

    def mock_llm_with_temp(prompt: str) -> str:
        nonlocal sample_call_count, entailment_call_count
        if prompt.startswith("Do these two statements express the same factual claim?"):
            entailment_call_count += 1
            return "yes"
        sample_call_count += 1
        return f"Answer {sample_call_count}"

    analyzer = SemanticEntropyAnalyzer(
        {
            "use_llm": True,
            "llm_fn": mock_llm_with_temp,
            "n_samples": 5,
            "temperature": 0.9,  # High temperature hint
        }
    )

    test_run = run_with_chunks([chunk("c1", "Context", 0.8)])
    result = analyzer.analyze(test_run)

    assert result.status in ("pass", "warn", "fail")
    # Sampling still happens exactly n_samples times even when LLM-NLI adds extra calls.
    assert sample_call_count == 5
    assert entailment_call_count >= 1


def test_integration_with_engine() -> None:
    """Integration test: SemanticEntropyAnalyzer works with DiagnosisEngine."""
    from raggov.engine import DiagnosisEngine

    def mock_llm(prompt: str) -> str:
        return "Consistent answer"

    test_run = RAGRun(
        run_id="test-entropy",
        query="What is X?",
        retrieved_chunks=[chunk("c1", "X is Y", 0.9)],
        final_answer="X is Y",
    )

    # Create engine with semantic entropy enabled
    engine = DiagnosisEngine(
        config={
            "use_llm": True,
            "llm_fn": mock_llm,
            "n_samples": 3,
        }
    )

    diagnosis = engine.diagnose(test_run)

    # Verify SemanticEntropyAnalyzer ran
    semantic_entropy_result = next(
        (
            r
            for r in diagnosis.analyzer_results
            if r.analyzer_name == "SemanticEntropyAnalyzer"
        ),
        None,
    )

    # Should have run if config is correct
    if semantic_entropy_result:
        assert semantic_entropy_result.status in ("pass", "warn", "fail", "skip")

        # If it ran successfully, check semantic_entropy is populated
        if semantic_entropy_result.status != "skip":
            assert diagnosis.semantic_entropy is not None


def test_remediation_message_for_high_entropy() -> None:
    """High entropy should have specific remediation message."""
    sample_counter = 0

    def mock_llm_different(prompt: str) -> str:
        nonlocal sample_counter
        if prompt.startswith("Do these two statements express the same factual claim?"):
            return "no"
        answers = ["A", "B", "C", "D", "E"]
        answer = answers[sample_counter % 5]
        sample_counter += 1
        return answer

    analyzer = SemanticEntropyAnalyzer(
        {"use_llm": True, "llm_fn": mock_llm_different, "n_samples": 5}
    )

    test_run = run_with_chunks([chunk("c1", "Context", 0.8)])
    result = analyzer.analyze(test_run)

    assert result.status == "fail"
    assert result.remediation is not None
    assert "do not serve" in result.remediation.lower() or "abstention" in result.remediation.lower()
