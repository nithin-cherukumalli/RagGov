"""Semantic Entropy analyzer for detecting model uncertainty.

Based on: "Detecting hallucinations in large language models using semantic entropy"
Farquhar et al., Nature 2024

Samples multiple LLM completions, clusters by semantic equivalence, and computes
entropy over meaning-groups. High entropy indicates the model doesn't know the answer.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.diagnosis import AnalyzerResult, ClaimResult, FailureStage, FailureType
from raggov.models.run import RAGRun


def jaccard_similarity(s1: str, s2: str) -> float:
    """Compute Jaccard similarity between two strings using token overlap.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Jaccard similarity in [0.0, 1.0]
    """
    # Tokenize by splitting on whitespace and converting to lowercase
    tokens1 = set(s1.lower().split())
    tokens2 = set(s2.lower().split())

    if not tokens1 and not tokens2:
        return 1.0  # Both empty = identical

    if not tokens1 or not tokens2:
        return 0.0  # One empty, one not = no similarity

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    return len(intersection) / len(union)


def _cluster_samples(samples: list[str], similarity_threshold: float = 0.5) -> list[list[int]]:
    """Cluster samples by semantic equivalence using Jaccard similarity.

    Args:
        samples: List of sample strings
        similarity_threshold: Minimum similarity to be in same cluster

    Returns:
        List of clusters, where each cluster is a list of sample indices
    """
    n = len(samples)
    if n == 0:
        return []

    # Union-find data structure
    parent = list(range(n))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

    def union(x: int, y: int) -> None:
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_x] = root_y

    # Compare all pairs and merge if similar
    for i in range(n):
        for j in range(i + 1, n):
            if jaccard_similarity(samples[i], samples[j]) > similarity_threshold:
                union(i, j)

    # Build clusters from union-find
    clusters_dict: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in clusters_dict:
            clusters_dict[root] = []
        clusters_dict[root].append(i)

    return list(clusters_dict.values())


def _compute_entropy(clusters: list[list[int]], n_samples: int) -> float:
    """Compute Shannon entropy over cluster distribution.

    Args:
        clusters: List of clusters (each cluster is list of sample indices)
        n_samples: Total number of samples

    Returns:
        Entropy in bits (log base 2)
    """
    if n_samples == 0:
        return 0.0

    entropy = 0.0
    for cluster in clusters:
        p_i = len(cluster) / n_samples
        if p_i > 0:
            entropy -= p_i * math.log2(p_i)

    return entropy


class SemanticEntropyAnalyzer(BaseAnalyzer):
    """Detect model uncertainty using semantic entropy.

    Implements the black-box variant (Farquhar et al., Nature 2024):
    - Sample multiple completions from the LLM
    - Cluster by semantic equivalence
    - Compute entropy over meaning-groups
    - High entropy = confabulation likely
    """

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        """Analyze model uncertainty via semantic entropy."""
        use_llm = bool(self.config.get("use_llm", False))

        if use_llm:
            return self._analyze_with_llm(run)
        else:
            return self._analyze_deterministic(run)

    def _analyze_deterministic(self, run: RAGRun) -> AnalyzerResult:
        """Deterministic fallback using claim results from prior grounding analyzer."""
        # Extract claim results from prior analyzers
        prior_results = self.config.get("prior_results", [])

        # Look for ClaimGroundingAnalyzer in prior results
        claim_results = None
        for result in prior_results:
            if result.analyzer_name == "ClaimGroundingAnalyzer":
                # Try to extract claim results from the result
                # ClaimGroundingAnalyzer stores claim_results in Diagnosis later
                # For now, check if we can get them from evidence (as JSON)
                try:
                    import json
                    if result.evidence:
                        # Try to parse evidence as JSON containing claim results
                        evidence_str = result.evidence[0] if result.evidence else "{}"
                        evidence_data = json.loads(evidence_str)
                        if "claim_results" in evidence_data:
                            from raggov.models.diagnosis import ClaimResult
                            claim_results = [ClaimResult(**cr) for cr in evidence_data["claim_results"]]
                            break
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass

        # If we couldn't extract claim results, check run metadata
        if claim_results is None:
            claim_results = run.metadata.get("claim_results")

        if not claim_results:
            return self.skip("no claim results available for deterministic analysis")

        # Compute disagreement ratio
        total_claims = len(claim_results)
        if total_claims == 0:
            return self.skip("no claims to analyze")

        disagreed_claims = sum(
            1 for claim in claim_results if claim.label in ("unsupported", "contradicted")
        )

        disagreement_ratio = disagreed_claims / total_claims

        # Map to pseudo-entropy: 0 disagreement = 0.0, 100% disagreement = 2.0
        pseudo_entropy = disagreement_ratio * 2.0

        # Get threshold
        entropy_threshold = float(self.config.get("entropy_threshold", 1.2))

        # Build evidence
        evidence = [
            f"Pseudo-entropy: {pseudo_entropy:.2f} (disagreement ratio: {disagreement_ratio:.2%})"
        ]

        # Interpret
        if pseudo_entropy < 0.5:
            evidence.append("LOW uncertainty — claims are well-grounded")
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="pass",
                score=pseudo_entropy,
                evidence=evidence,
            )
        elif pseudo_entropy <= entropy_threshold:
            evidence.append("MEDIUM uncertainty — some claim disagreement")
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="warn",
                score=pseudo_entropy,
                evidence=evidence,
                remediation="Moderate uncertainty detected. Consider additional verification.",
            )
        else:
            evidence.append("HIGH uncertainty — many claims unsupported, confabulation likely")
            result = self._fail(
                failure_type=FailureType.LOW_CONFIDENCE,
                stage=FailureStage.CONFIDENCE,
                evidence=evidence,
                remediation="High response variance detected. Do not serve this answer. Consider retrieval expansion or abstention.",
            )
            result.score = pseudo_entropy
            return result

    def _analyze_with_llm(self, run: RAGRun) -> AnalyzerResult:
        """LLM-based semantic entropy analysis."""
        # Get LLM function
        llm_fn: Callable[[str], str] | None = self.config.get("llm_fn")
        if llm_fn is None:
            return self.skip("llm_fn not provided")

        # Get config
        n_samples = int(self.config.get("n_samples", 5))
        temperature = float(self.config.get("temperature", 0.7))
        entropy_threshold = float(self.config.get("entropy_threshold", 1.2))

        # Check if we have chunks
        if not run.retrieved_chunks:
            return self.skip("no retrieved chunks for LLM sampling")

        # Prepare context from chunks
        chunk_text = "\n\n".join([chunk.text for chunk in run.retrieved_chunks])

        # Step 1: Sample n_samples completions
        samples: list[str] = []
        for i in range(n_samples):
            prompt = (
                f"Answer this question based only on the provided context.\n"
                f"Context: {chunk_text}\n"
                f"Question: {run.query}\n"
                f"Answer:"
            )
            try:
                sample = llm_fn(prompt)
                samples.append(sample)
            except Exception as exc:
                return self.skip(f"LLM sampling failed: {exc}")

        # Step 2: Cluster by semantic equivalence
        clusters = _cluster_samples(samples, similarity_threshold=0.5)
        n_clusters = len(clusters)

        # Step 3: Compute entropy
        entropy = _compute_entropy(clusters, n_samples)

        # Compute sample agreement (fraction in largest cluster)
        largest_cluster_size = max(len(cluster) for cluster in clusters) if clusters else 0
        sample_agreement = largest_cluster_size / n_samples if n_samples > 0 else 0.0

        # Build evidence
        evidence = [
            f"Semantic entropy: {entropy:.2f} — sampled {n_samples} completions, found {n_clusters} meaning-groups",
            f"Sample agreement: {sample_agreement:.2%} (largest cluster: {largest_cluster_size}/{n_samples})",
        ]

        # Step 4: Interpret
        if entropy < 0.5:
            evidence.append("LOW uncertainty — all samples agree")
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="pass",
                score=entropy,
                evidence=evidence,
            )
        elif entropy <= entropy_threshold:
            evidence.append("MEDIUM uncertainty — some response variance")
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="warn",
                score=entropy,
                evidence=evidence,
                remediation="Moderate response variance detected. Consider additional verification.",
            )
        else:
            evidence.append(
                f"HIGH uncertainty (entropy {entropy:.2f}) — model responses are inconsistent, confabulation likely"
            )
            result = self._fail(
                failure_type=FailureType.LOW_CONFIDENCE,
                stage=FailureStage.CONFIDENCE,
                evidence=evidence,
                remediation="High response variance detected. Do not serve this answer. Consider retrieval expansion or abstention.",
            )
            result.score = entropy
            return result
