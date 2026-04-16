"""A2P (Abduct-Act-Predict) counterfactual attribution analyzer."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.run import RAGRun


DEFAULT_LOW_CONFIDENCE = 0.45
DEFAULT_RETRIEVAL_CONFIDENCE = 0.58
RETRIEVAL_ANOMALY_CONFIDENCE = 0.82
LOW_SCORE_FEW_CHUNKS_CONFIDENCE = 0.78
LOW_SCORE_RETRIEVAL_CONFIDENCE = 0.65
INCONSISTENT_CHUNKS_CONFIDENCE = 0.68
PARSER_TABLE_CONFIDENCE = 0.86
PARSER_HIERARCHY_CONFIDENCE = 0.82
GENERATION_HIGH_CONFIDENCE = 0.74
GENERATION_MEDIUM_CONFIDENCE = 0.66

A2P_PROMPT_TEMPLATE = """
You are diagnosing a RAG pipeline failure. Use the A2P (Abduct-Act-Predict) framework:

OBSERVED FAILURES:
{failures}

RAG RUN CONTEXT:
- Query: {query}
- Retrieved chunks: {n_chunks} chunks, avg score: {avg_score:.2f}
- Answer length: {answer_length} chars

STEP 1 — ABDUCTION: What is the most likely hidden root cause in the pipeline
(PARSING/CHUNKING/EMBEDDING/RETRIEVAL/RERANKING/GENERATION/SECURITY) that explains ALL
observed failures? Reason through the causal chain.

STEP 2 — ACTION: What is the minimal corrective intervention at that stage
that would address the root cause?

STEP 3 — PREDICTION: If that intervention were applied, would the observed
failures be resolved? What residual failures might remain?

Respond ONLY with valid JSON:
{{
  "abduction": "<root cause reasoning>",
  "root_cause_stage": "<PARSING|CHUNKING|EMBEDDING|RETRIEVAL|RERANKING|GENERATION|SECURITY>",
  "action": "<specific fix>",
  "prediction": "<expected outcome after fix>",
  "confidence": <0.0-1.0>,
  "confidence_basis": "<why this confidence level>",
  "root_cause_type": "<optional FailureType enum value if you can determine it>"
}}
""".strip()

SECURITY_FAILURE_TYPES = {
    FailureType.PROMPT_INJECTION,
    FailureType.SUSPICIOUS_CHUNK,
    FailureType.RETRIEVAL_ANOMALY,
    FailureType.PRIVACY_VIOLATION,
}


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
        prior_results = self.config.get("weighted_prior_results") or self.config.get("prior_results", [])

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
        """Deterministic A2P approximation using causal rules over the pipeline DAG."""
        analyzer_weights = self.config.get("analyzer_weights", {})
        best_results = self._best_results_by_failure(prior_results, analyzer_weights)
        failure_types = {
            result.failure_type
            for result in prior_results
            if result.status == "fail" and result.failure_type is not None
        }
        chunk_scores = [c.score for c in run.retrieved_chunks if c.score is not None]
        avg_score = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0
        n_chunks = len(chunk_scores)
        candidates: list[tuple[float, int, AnalyzerResult]] = []

        table_result = best_results.get(FailureType.TABLE_STRUCTURE_LOSS)
        if table_result is not None:
            result = self._create_attribution_result(
                root_cause_type=FailureType.TABLE_STRUCTURE_LOSS,
                stage=FailureStage.PARSING,
                abduction=(
                    "Observed table-structure loss is most consistent with a parser-stage "
                    "failure: row and column bindings were stripped before chunking could preserve them."
                ),
                action=table_result.remediation or self._remediation_for(prior_results, FailureType.TABLE_STRUCTURE_LOSS),
                prediction=(
                    "Restoring table-aware parsing should propagate intact cell bindings into chunking "
                    "and prevent the downstream loss of grounded table facts."
                ),
                confidence=PARSER_TABLE_CONFIDENCE,
                confidence_basis=(
                    "High confidence because TABLE_STRUCTURE_LOSS has a direct causal predecessor in the DAG: "
                    "parsing errors occur before chunking and manifest with this exact structure-loss symptom."
                ),
            )
            candidates.append(self._candidate(table_result, result, analyzer_weights, 0))

        hierarchy_result = best_results.get(FailureType.HIERARCHY_FLATTENING)
        if hierarchy_result is not None:
            result = self._create_attribution_result(
                root_cause_type=FailureType.HIERARCHY_FLATTENING,
                stage=FailureStage.PARSING,
                abduction=(
                    "Hierarchy flattening points upstream to parsing because heading and list relationships "
                    "must be lost before chunking or retrieval can act on the document."
                ),
                action=hierarchy_result.remediation or self._remediation_for(prior_results, FailureType.HIERARCHY_FLATTENING),
                prediction=(
                    "Preserving headings and numbered-list structure should keep related content grouped "
                    "and reduce downstream scope and grounding errors."
                ),
                confidence=PARSER_HIERARCHY_CONFIDENCE,
                confidence_basis=(
                    "High confidence because HIERARCHY_FLATTENING is primarily caused by parser output "
                    "that collapses document structure before later stages run."
                ),
            )
            candidates.append(self._candidate(hierarchy_result, result, analyzer_weights, 1))

        anomaly_result = best_results.get(FailureType.RETRIEVAL_ANOMALY)
        if FailureType.RETRIEVAL_ANOMALY in failure_types and anomaly_result is not None:
            result = self._create_attribution_result(
                root_cause_type=FailureType.EMBEDDING_DRIFT,
                stage=FailureStage.EMBEDDING,
                abduction=(
                    "Score anomalies in the retrieved set indicate embedding-space corruption: "
                    "semantically dissimilar documents were mapped into a high-similarity region."
                ),
                action="Re-evaluate the embedding model on-domain and audit the vector index for corruption.",
                prediction=(
                    "Fixing the embedding representation should eliminate anomalous nearest neighbors "
                    "and reduce downstream grounding and retrieval instability."
                ),
                confidence=RETRIEVAL_ANOMALY_CONFIDENCE,
                confidence_basis=(
                    "High confidence because RETRIEVAL_ANOMALY has one dominant causal predecessor in the DAG: "
                    "embedding/index quality, not answer generation or retrieval depth."
                ),
            )
            candidates.append(self._candidate(anomaly_result, result, analyzer_weights, 2))

        insufficient_result = best_results.get(FailureType.INSUFFICIENT_CONTEXT)
        if FailureType.INSUFFICIENT_CONTEXT in failure_types and insufficient_result is not None:
            if avg_score < 0.6 and n_chunks < 4:
                confidence = LOW_SCORE_FEW_CHUNKS_CONFIDENCE
                abduction = (
                    "Insufficient context plus uniformly low retrieval scores and a shallow candidate set "
                    "is best explained by retrieval depth being too small to surface the needed evidence."
                )
                action = "Increase top-k retrieval to at least 5-8 and inspect whether missing supporting chunks appear."
                prediction = (
                    "Expanding retrieval depth should resolve the insufficiency if the missing evidence exists in the corpus; "
                    "residual failure would point to query formulation or indexing."
                )
                confidence_basis = (
                    "Confidence 0.78 because both signals are present: low average similarity and fewer than four scored chunks, "
                    "which strongly narrows the cause to top-k depth rather than generation."
                )
            elif avg_score < 0.6:
                confidence = LOW_SCORE_RETRIEVAL_CONFIDENCE
                abduction = (
                    "Insufficient context with broadly weak retrieval scores suggests the retriever is missing relevant documents "
                    "rather than the generator ignoring good context."
                )
                action = "Audit retrieval recall, expand candidate depth, and verify query rewriting or domain indexing."
                prediction = (
                    "Improving retrieval recall should increase evidence coverage, though residual issues may remain if query scope is mismatched."
                )
                confidence_basis = (
                    "Confidence 0.65 because low retrieval scores support a retrieval-stage cause, but the signal does not isolate "
                    "whether the problem is depth, recall, or query formulation."
                )
            else:
                confidence = DEFAULT_RETRIEVAL_CONFIDENCE
                abduction = (
                    "Insufficient context despite non-trivial retrieval scores suggests the retriever is returning context adjacent to the query "
                    "without capturing the precise scope required to answer it."
                )
                action = "Refine query rewriting and retrieval filters to better match the requested scope before reranking."
                prediction = (
                    "Sharper retrieval scope should reduce insufficiency, but residual ambiguity may remain if the answer is absent from the corpus."
                )
                confidence_basis = (
                    "Confidence 0.58 because insufficient context alone does not isolate a single root cause, but retrieval remains the best fit in the DAG."
                )

            result = self._create_attribution_result(
                root_cause_type=FailureType.RETRIEVAL_DEPTH_LIMIT,
                stage=FailureStage.RETRIEVAL,
                abduction=abduction,
                action=action,
                prediction=prediction,
                confidence=confidence,
                confidence_basis=confidence_basis,
            )
            candidates.append(self._candidate(insufficient_result, result, analyzer_weights, 3))

        inconsistent_result = best_results.get(FailureType.INCONSISTENT_CHUNKS)
        if FailureType.INCONSISTENT_CHUNKS in failure_types and inconsistent_result is not None:
            result = self._create_attribution_result(
                root_cause_type=FailureType.CHUNKING_BOUNDARY_ERROR,
                stage=FailureStage.CHUNKING,
                abduction=(
                    "Conflicting retrieved chunks are most plausibly caused by boundary errors that split or merge logical units "
                    "before retrieval, creating partial context fragments."
                ),
                action="Adjust chunk boundaries to preserve paragraph, section, and list cohesion.",
                prediction=(
                    "Cleaner chunk boundaries should reduce contradictory partial context, though residual issues may remain if parsing already lost structure."
                ),
                confidence=INCONSISTENT_CHUNKS_CONFIDENCE,
                confidence_basis=(
                    "Confidence 0.68 because inconsistent chunks are consistent with chunking errors, but parsing defects can create a similar symptom."
                ),
            )
            candidates.append(self._candidate(inconsistent_result, result, analyzer_weights, 4))

        unsupported_result = best_results.get(FailureType.UNSUPPORTED_CLAIM)
        if FailureType.UNSUPPORTED_CLAIM in failure_types and unsupported_result is not None and avg_score > 0.7:
            confidence = GENERATION_HIGH_CONFIDENCE if avg_score > 0.85 and n_chunks >= 2 else GENERATION_MEDIUM_CONFIDENCE
            confidence_basis = (
                "Confidence 0.74 because unsupported claims appear despite multiple strong retrieval scores, "
                "which is strong evidence that generation ignored available context."
                if confidence == GENERATION_HIGH_CONFIDENCE
                else "Confidence 0.66 because high-scoring context points toward generation-stage drift, but retrieval quality is not overwhelmingly strong."
            )
            result = self._create_attribution_result(
                root_cause_type=FailureType.GENERATION_IGNORE,
                stage=FailureStage.GENERATION,
                abduction=(
                    "Unsupported claims despite strong retrieved evidence indicate the model likely generated from parametric memory "
                    "or ignored the provided context during answer synthesis."
                ),
                action="Strengthen grounding instructions, require chunk-quoted evidence, or switch to a more context-adherent model.",
                prediction=(
                    "A stricter generation policy should resolve unsupported claims if retrieval remains strong; residual errors would suggest grounding heuristics still miss key evidence."
                ),
                confidence=confidence,
                confidence_basis=confidence_basis,
            )
            candidates.append(self._candidate(unsupported_result, result, analyzer_weights, 5))

        if candidates:
            candidates.sort(key=lambda item: (-item[0], item[1]))
            return candidates[0][2]

        return self._create_attribution_result(
            root_cause_type=FailureType.RETRIEVAL_DEPTH_LIMIT,
            stage=FailureStage.RETRIEVAL,
            abduction=(
                "Observed failures do not isolate a single upstream cause, so retrieval remains the lowest-risk default hypothesis "
                "in the causal DAG because many downstream failures originate from missing evidence."
            ),
            action="Inspect retrieval recall first: verify top-k depth, query rewriting, and corpus coverage before changing downstream stages.",
            prediction=(
                "This intervention may surface missing evidence, but the weak signal means residual failures are likely and further diagnosis may still be required."
            ),
            confidence=DEFAULT_LOW_CONFIDENCE,
            confidence_basis=(
                "Confidence 0.45 because the current evidence does not distinguish among multiple upstream causes; this is a low-confidence fallback, not a precise attribution."
            ),
        )

    def _llm_mode(
        self, run: RAGRun, prior_results: list[AnalyzerResult], llm_fn: Callable[[str], str]
    ) -> AnalyzerResult:
        """LLM-based mode using an explicit Abduct-Act-Predict prompt."""
        prompt = self._build_a2p_prompt(run, prior_results)

        try:
            response = llm_fn(prompt)
            parsed = self._parse_llm_payload(response)
            root_cause_stage_str = parsed["root_cause_stage"]
            root_cause_stage = FailureStage(root_cause_stage_str)
            root_cause_type = self._infer_root_cause_type(
                root_cause_stage,
                prior_results,
                parsed.get("root_cause_type"),
            )
            return self._create_attribution_result(
                root_cause_type=root_cause_type,
                stage=root_cause_stage,
                abduction=str(parsed["abduction"]),
                action=str(parsed["action"]),
                prediction=str(parsed["prediction"]),
                confidence=float(parsed["confidence"]),
                confidence_basis=str(parsed["confidence_basis"]),
            )
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            return self._deterministic_mode(run, prior_results)

    def _build_a2p_prompt(self, run: RAGRun, prior_results: list[AnalyzerResult]) -> str:
        """Build the structured A2P prompt for LLM."""
        score_values = [chunk.score for chunk in run.retrieved_chunks if chunk.score is not None]
        avg_score = sum(score_values) / len(score_values) if score_values else 0.0
        return A2P_PROMPT_TEMPLATE.format(
            failures=self._format_failures(prior_results),
            query=run.query,
            n_chunks=len(run.retrieved_chunks),
            avg_score=avg_score,
            answer_length=len(run.final_answer),
        )

    def _remediation_for(
        self,
        prior_results: list[AnalyzerResult],
        failure_type: FailureType,
    ) -> str:
        """Return the concrete remediation attached to a prior result when available."""
        for result in prior_results:
            if result.failure_type == failure_type and result.remediation is not None:
                return result.remediation
        return ""

    def _best_results_by_failure(
        self,
        prior_results: list[AnalyzerResult],
        analyzer_weights: dict[str, float],
    ) -> dict[FailureType, AnalyzerResult]:
        best_results: dict[FailureType, tuple[float, int, AnalyzerResult]] = {}
        for order, result in enumerate(prior_results):
            if result.status != "fail" or result.failure_type is None:
                continue
            weight = self._result_weight(result, analyzer_weights)
            current = best_results.get(result.failure_type)
            if current is None or weight > current[0] or (weight == current[0] and order < current[1]):
                best_results[result.failure_type] = (weight, order, result)
        return {
            failure_type: payload[2]
            for failure_type, payload in best_results.items()
        }

    def _result_weight(
        self,
        result: AnalyzerResult,
        analyzer_weights: dict[str, float],
    ) -> float:
        return float(analyzer_weights.get(result.analyzer_name, 1.0))

    def _create_attribution_result(
        self,
        root_cause_type: FailureType,
        stage: FailureStage,
        abduction: str,
        action: str,
        prediction: str,
        confidence: float,
        confidence_basis: str,
    ) -> AnalyzerResult:
        """Create an attribution result with all fields populated."""
        evidence = [
            abduction,
            f"Proposed fix: {action}",
            f"Prediction: {prediction}",
            f"Confidence basis: {confidence_basis}",
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

    def _candidate(
        self,
        source_result: AnalyzerResult,
        attribution_result: AnalyzerResult,
        analyzer_weights: dict[str, float],
        order: int,
    ) -> tuple[float, int, AnalyzerResult]:
        """Rank a deterministic attribution by evidence strength and analyzer weight."""
        weight = self._result_weight(source_result, analyzer_weights)
        confidence = attribution_result.fix_confidence or 0.0
        return (weight * confidence, order, attribution_result)

    def _format_failures(self, prior_results: list[AnalyzerResult]) -> str:
        """Format observed analyzer failures for the LLM prompt."""
        lines: list[str] = []
        for result in prior_results:
            if result.status not in {"fail", "warn"} or result.failure_type is None:
                continue
            snippet = "; ".join(result.evidence[:2]) if result.evidence else "no evidence provided"
            lines.append(
                f"- {result.failure_type.value} at {result.stage.value if result.stage else 'UNKNOWN'}: {snippet}"
            )
        return "\n".join(lines) if lines else "- No structured failures available"

    def _parse_llm_payload(self, response: Any) -> dict[str, Any]:
        """Normalize and validate an LLM A2P response payload."""
        if isinstance(response, dict):
            payload = response
        else:
            payload = json.loads(str(response))

        required_fields = {
            "abduction",
            "root_cause_stage",
            "action",
            "prediction",
            "confidence",
            "confidence_basis",
        }
        missing = required_fields - set(payload)
        if missing:
            raise ValueError(f"missing required fields: {sorted(missing)}")

        confidence = float(payload["confidence"])
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("confidence must be in [0.0, 1.0]")
        payload["confidence"] = confidence
        return payload

    def _infer_root_cause_type(
        self,
        stage: FailureStage,
        prior_results: list[AnalyzerResult],
        declared_type: str | None = None,
    ) -> FailureType:
        """Infer a canonical root-cause failure type from stage and prior failures."""
        if declared_type:
            return FailureType(declared_type)

        if stage == FailureStage.PARSING:
            failure_types = {
                result.failure_type
                for result in prior_results
                if result.status == "fail" and result.failure_type is not None
            }
            if FailureType.TABLE_STRUCTURE_LOSS in failure_types:
                return FailureType.TABLE_STRUCTURE_LOSS
            if FailureType.HIERARCHY_FLATTENING in failure_types:
                return FailureType.HIERARCHY_FLATTENING
            return FailureType.PARSER_STRUCTURE_LOSS
        if stage == FailureStage.CHUNKING:
            return FailureType.CHUNKING_BOUNDARY_ERROR
        if stage == FailureStage.EMBEDDING:
            return FailureType.EMBEDDING_DRIFT
        if stage == FailureStage.RETRIEVAL:
            return FailureType.RETRIEVAL_DEPTH_LIMIT
        if stage == FailureStage.RERANKING:
            return FailureType.RERANKER_FAILURE
        if stage == FailureStage.GENERATION:
            return FailureType.GENERATION_IGNORE
        if stage == FailureStage.SECURITY:
            for result in prior_results:
                if result.status == "fail" and result.failure_type in SECURITY_FAILURE_TYPES:
                    return result.failure_type
            return FailureType.SUSPICIOUS_CHUNK
        return FailureType.RETRIEVAL_DEPTH_LIMIT
