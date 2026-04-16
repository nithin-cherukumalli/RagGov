"""A2P (Abduct-Act-Predict) counterfactual attribution analyzer."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.run import RAGRun


class A2PAttributionAnalyzer(BaseAnalyzer):
    """
    Implements the Abduct-Act-Predict (A2P) framework for counterfactual failure attribution.

    Given a failed RAG run and prior analyzer results, it reasons through which pipeline
    stage most likely caused the failure using structured counterfactual reasoning.

    A2P works in three steps:
    - Abduction: Infer the hidden root cause behind the observed failure
    - Action: Define the minimal corrective intervention at that stage
    - Prediction: Simulate whether the fix would resolve the failure
    """

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        """Analyze a RAG run and attribute failure to root cause stage."""
        prior_results = self.config.get("prior_results", [])

        # Skip if no prior failures to attribute
        failed_results = [
            r for r in prior_results if r.status == "fail" and r.failure_type is not None
        ]
        if not failed_results:
            return self.skip("no prior failures to attribute")

        use_llm = self.config.get("use_llm", False)
        llm_fn = self.config.get("llm_fn")

        if use_llm and llm_fn:
            return self._llm_mode(run, prior_results, llm_fn)
        else:
            return self._deterministic_mode(run, prior_results)

    def _deterministic_mode(
        self, run: RAGRun, prior_results: list[AnalyzerResult]
    ) -> AnalyzerResult:
        """Deterministic fallback mode using heuristic rules."""
        # Extract failure types from prior results
        failure_types = {r.failure_type for r in prior_results if r.failure_type is not None}

        # Calculate average chunk score if available
        chunk_scores = [c.score for c in run.retrieved_chunks if c.score is not None]
        avg_score = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0

        # Apply heuristic rules
        if FailureType.INSUFFICIENT_CONTEXT in failure_types and avg_score < 0.6:
            # Low scores + insufficient context → retrieval depth issue
            return self._create_attribution_result(
                root_cause_type=FailureType.RETRIEVAL_DEPTH_LIMIT,
                stage=FailureStage.RETRIEVAL,
                abduction="Insufficient context combined with low retrieval scores suggests top-k limit excluded critical chunks",
                action="Increase top-k retrieval parameter to include more candidate chunks",
                prediction="LIKELY",
                prediction_reasoning="More chunks would likely provide sufficient context",
                confidence=0.75,
            )

        if FailureType.INCONSISTENT_CHUNKS in failure_types:
            # Inconsistent chunks → chunking boundary issue
            return self._create_attribution_result(
                root_cause_type=FailureType.CHUNKING_BOUNDARY_ERROR,
                stage=FailureStage.CHUNKING,
                abduction="Inconsistent chunks suggest boundaries split logical units incorrectly",
                action="Adjust chunking boundaries to preserve semantic coherence and logical units",
                prediction="LIKELY",
                prediction_reasoning="Better boundaries would prevent contradictory information across chunks",
                confidence=0.7,
            )

        if FailureType.UNSUPPORTED_CLAIM in failure_types and avg_score > 0.7:
            # Unsupported claims + high scores → generation ignored context
            return self._create_attribution_result(
                root_cause_type=FailureType.GENERATION_IGNORE,
                stage=FailureStage.GENERATION,
                abduction="High-quality chunks with unsupported claims suggests LLM generated from parametric memory instead of context",
                action="Strengthen grounding instructions or use a more context-adherent model",
                prediction="LIKELY",
                prediction_reasoning="Stricter instructions would force LLM to ground in provided context",
                confidence=0.8,
            )

        if FailureType.RETRIEVAL_ANOMALY in failure_types:
            # Retrieval anomalies → embedding drift
            return self._create_attribution_result(
                root_cause_type=FailureType.EMBEDDING_DRIFT,
                stage=FailureStage.EMBEDDING,
                abduction="Statistical retrieval anomalies suggest embedding model collapsed on semantically similar documents",
                action="Review embedding model for semantic drift or switch to a more discriminative model",
                prediction="LIKELY",
                prediction_reasoning="Better embeddings would produce more diverse retrieval results",
                confidence=0.65,
            )

        # Default fallback: assume retrieval depth issue
        return self._create_attribution_result(
            root_cause_type=FailureType.RETRIEVAL_DEPTH_LIMIT,
            stage=FailureStage.RETRIEVAL,
            abduction="No specific pattern matched, defaulting to retrieval depth as most common issue",
            action="Increase top-k retrieval parameter",
            prediction="UNLIKELY",
            prediction_reasoning="Default fallback without strong signal",
            confidence=0.5,
        )

    def _llm_mode(
        self, run: RAGRun, prior_results: list[AnalyzerResult], llm_fn: Callable[[str], str]
    ) -> AnalyzerResult:
        """LLM-based mode using structured A2P prompt."""
        prompt = self._build_a2p_prompt(run, prior_results)

        try:
            response = llm_fn(prompt)
            parsed = json.loads(response)

            # Validate required fields
            root_cause_stage_str = parsed.get("root_cause_stage")
            root_cause_type_str = parsed.get("root_cause_type")

            if not root_cause_stage_str or not root_cause_type_str:
                raise ValueError("Missing required fields in LLM response")

            # Convert strings to enums
            root_cause_stage = FailureStage(root_cause_stage_str)
            root_cause_type = FailureType(root_cause_type_str)

            return self._create_attribution_result(
                root_cause_type=root_cause_type,
                stage=root_cause_stage,
                abduction=parsed.get("abduction_reasoning", ""),
                action=parsed.get("action", ""),
                prediction=parsed.get("prediction", "UNKNOWN"),
                prediction_reasoning=parsed.get("prediction_reasoning", ""),
                confidence=parsed.get("confidence", 0.5),
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Fall back to deterministic mode on parse error
            return self._deterministic_mode(run, prior_results)

    def _build_a2p_prompt(self, run: RAGRun, prior_results: list[AnalyzerResult]) -> str:
        """Build the structured A2P prompt for LLM."""
        # Extract chunk info
        chunks_text = ""
        score_list = []
        for i, chunk in enumerate(run.retrieved_chunks, 1):
            score = chunk.score if chunk.score is not None else "N/A"
            score_list.append(str(score))
            chunks_text += f"\n[Chunk {i}, score={score}]: {chunk.text[:200]}..."

        # Extract symptom list from prior results
        symptom_list = ""
        for result in prior_results:
            if result.status in {"fail", "warn"} and result.failure_type:
                symptom_list += f"\n- {result.failure_type.value}: {', '.join(result.evidence[:2])}"

        prompt = f"""You are diagnosing a RAG pipeline failure. Reason through the following
in three steps.

QUERY: {run.query}

RETRIEVED CHUNKS ({len(run.retrieved_chunks)} chunks, scores: {', '.join(score_list)}):
{chunks_text}

FINAL ANSWER: {run.final_answer}

OBSERVED SYMPTOMS from analyzers:
{symptom_list}

Step 1 — ABDUCTION: What is the most likely root cause?
Consider these possibilities:
- Parser lost document structure (hierarchy, tables) before chunking
- Chunker split content at wrong boundaries, losing logical units
- Embedding model collapsed on near-duplicate docs (semantic drift)
- Retrieval top-k too small, excluded critical chunks
- Reranker demoted the most relevant chunks
- LLM had good context but generated from parametric memory instead

Pick the single most likely cause. Explain your reasoning in 2 sentences.

Step 2 — ACTION: What is the minimal fix?
Describe exactly one concrete change to the pipeline that would most
likely fix this failure. Be specific: e.g. "increase top-k from 3 to 8"
or "add a table-preserving parser before chunking".

Step 3 — PREDICTION: Would that fix resolve the failure?
Given the query and available chunks, predict whether applying your
proposed fix would have produced a grounded answer. Answer: YES / LIKELY / UNLIKELY / NO.
Explain in 1 sentence.

Respond ONLY with valid JSON:
{{
  "root_cause_stage": one of [PARSING, CHUNKING, EMBEDDING, RETRIEVAL, RERANKING, GENERATION],
  "root_cause_type": one of [PARSER_STRUCTURE_LOSS, CHUNKING_BOUNDARY_ERROR, EMBEDDING_DRIFT, RETRIEVAL_DEPTH_LIMIT, RERANKER_FAILURE, GENERATION_IGNORE],
  "abduction_reasoning": "string",
  "action": "string — the specific fix",
  "prediction": one of [YES, LIKELY, UNLIKELY, NO],
  "prediction_reasoning": "string",
  "confidence": 0.0-1.0
}}
"""
        return prompt

    def _create_attribution_result(
        self,
        root_cause_type: FailureType,
        stage: FailureStage,
        abduction: str,
        action: str,
        prediction: str,
        prediction_reasoning: str,
        confidence: float,
    ) -> AnalyzerResult:
        """Create an attribution result with all fields populated."""
        evidence = [
            abduction,
            f"Proposed fix: {action}",
            f"Prediction: {prediction} — {prediction_reasoning}",
        ]

        return AnalyzerResult(
            analyzer_name=self.name(),
            status="fail",
            failure_type=root_cause_type,
            stage=stage,
            evidence=evidence,
            remediation=action,
            attribution_stage=stage,
            proposed_fix=action,
            fix_confidence=confidence,
            score=confidence,
        )
