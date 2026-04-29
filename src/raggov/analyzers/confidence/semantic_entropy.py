"""Semantic Entropy analyzer for detecting model uncertainty.

Based on: "Detecting hallucinations in large language models using semantic entropy"
Farquhar et al., Nature 2024

Samples multiple LLM completions, clusters by semantic equivalence, and computes
entropy over meaning-groups. High entropy indicates the model doesn't know the answer.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.diagnosis import AnalyzerResult, ClaimResult, FailureStage, FailureType
from raggov.models.run import RAGRun

CLAIM_LABELS = ("entailed", "unsupported", "contradicted")
LOW_ENTROPY_THRESHOLD = 0.5
DEFAULT_HIGH_ENTROPY_THRESHOLD = 1.2
JACCARD_EQUIVALENCE_THRESHOLD = 0.5
DEFAULT_NLI_THRESHOLD = 0.7


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


class NLIClusterer:
    """Semantic equivalence clustering for LLM samples.

    Tier A:
        Jaccard-augmented proxy with light stemming and synonym expansion.
        This is still approximate and is labeled honestly in analyzer evidence.

    Tier B:
        Bidirectional NLI with a local cross-encoder if `sentence_transformers`
        and the configured model are available.

    Tier C:
        Bidirectional LLM-based equivalence checks using the configured `llm_fn`.
        This is more expensive but closes the semantic gap when no local NLI
        model is configured.
    """

    SYNONYM_MAP = {
        "firm": "company",
        "company": "company",
        "establish": "found",
        "founded": "found",
        "founded.": "found",
        "established": "found",
        "increase": "grow",
        "increased": "grow",
        "growing": "grow",
        "annually": "annual",
        "yearly": "annual",
        "yoy": "annual",
    }

    def __init__(
        self,
        nli_model: str | None = None,
        llm_fn: Callable[[str], str] | None = None,
        nli_threshold: float = DEFAULT_NLI_THRESHOLD,
    ) -> None:
        self.llm_fn = llm_fn
        self._nli_model: Any | None = None
        self._nli_threshold = nli_threshold

        if nli_model is not None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore

                self._nli_model = CrossEncoder(nli_model, max_length=128)
                self._tier = "B"
            except ImportError:
                self._tier = "A"
        elif llm_fn is not None:
            self._tier = "C"
        else:
            self._tier = "A"

    @property
    def tier_label(self) -> str:
        labels = {
            "A": "Jaccard-augmented proxy",
            "B": "NLI (Farquhar exact)",
            "C": "LLM-NLI",
        }
        return labels[self._tier]

    def are_equivalent(self, s1: str, s2: str) -> bool:
        """Return True when two samples belong to the same meaning-group."""
        if self._tier == "B" and self._nli_model is not None:
            return self._nli_equivalent(s1, s2)
        if self._tier == "C" and self.llm_fn is not None:
            return self._llm_equivalent(s1, s2)
        return self._jaccard_augmented_equivalent(s1, s2)

    def cluster(self, samples: list[str]) -> list[list[int]]:
        """Cluster samples by semantic equivalence using union-find."""
        n = len(samples)
        if n == 0:
            return []

        parent = list(range(n))

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> None:
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_x] = root_y

        for i in range(n):
            for j in range(i + 1, n):
                if self.are_equivalent(samples[i], samples[j]):
                    union(i, j)

        clusters_dict: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            clusters_dict.setdefault(root, []).append(i)
        return list(clusters_dict.values())

    def _nli_equivalent(self, s1: str, s2: str) -> bool:
        """Bidirectional NLI entailment check."""
        score_forward = self._entailment_score(s1, s2)
        score_backward = self._entailment_score(s2, s1)
        return (
            score_forward >= self._nli_threshold
            and score_backward >= self._nli_threshold
        )

    def _entailment_score(self, premise: str, hypothesis: str) -> float:
        """Extract entailment score from cross-encoder output."""
        if self._nli_model is None:
            return 0.0

        raw_score = self._nli_model.predict([(premise, hypothesis)])[0]
        if isinstance(raw_score, (list, tuple)):
            if len(raw_score) >= 3:
                return float(raw_score[2])
            if len(raw_score) == 1:
                return float(raw_score[0])
        if hasattr(raw_score, "tolist"):
            raw_list = raw_score.tolist()
            if isinstance(raw_list, list):
                if len(raw_list) >= 3:
                    return float(raw_list[2])
                if len(raw_list) == 1:
                    return float(raw_list[0])
        return float(raw_score)

    def _llm_equivalent(self, s1: str, s2: str) -> bool:
        """Use the configured LLM as a bidirectional entailment oracle."""
        prompt = (
            'Do these two statements express the same factual claim? '
            'Answer only "yes" or "no".\n'
            f"Statement 1: {s1}\n"
            f"Statement 2: {s2}"
        )
        try:
            response = self.llm_fn(prompt).strip().lower()
        except Exception:
            return self._jaccard_augmented_equivalent(s1, s2)

        if response.startswith("yes"):
            return True
        if response.startswith("no"):
            return False
        return self._jaccard_augmented_equivalent(s1, s2)

    def _jaccard_augmented_equivalent(self, s1: str, s2: str) -> bool:
        """Fallback lexical proxy when no NLI judge is available."""
        tokens1 = self._normalized_tokens(s1)
        tokens2 = self._normalized_tokens(s2)

        if not tokens1 and not tokens2:
            return True
        if not tokens1 or not tokens2:
            return False

        return (
            len(tokens1 & tokens2) / len(tokens1 | tokens2)
            >= JACCARD_EQUIVALENCE_THRESHOLD
        )

    def _normalized_tokens(self, text: str) -> set[str]:
        """Normalize tokens with light stemming and synonym collapsing."""
        normalized: set[str] = set()
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            stemmed = self._stem(token)
            normalized.add(self.SYNONYM_MAP.get(stemmed, stemmed))
        return {token for token in normalized if len(token) > 2}

    def _stem(self, word: str) -> str:
        """Small dependency-free stemmer for proxy clustering."""
        for suffix in ("ingly", "edly", "tion", "ing", "ed", "ly", "er", "est", "ness", "ment", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
        return word


def _cluster_samples(samples: list[str], similarity_threshold: float = 0.5) -> list[list[int]]:
    """Backward-compatible proxy clustering helper."""
    clusterer = NLIClusterer()
    if similarity_threshold != JACCARD_EQUIVALENCE_THRESHOLD:
        clusterer._nli_threshold = similarity_threshold
    return clusterer.cluster(samples)


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
        """Approximate semantic entropy via Shannon entropy over claim-grounding labels."""
        claim_results = self._get_claim_results(run)
        if not claim_results:
            return self.skip("no claim results available for claim-label entropy")

        n_claims = len(claim_results)
        if n_claims == 0:
            return self.skip("no claims to analyze")

        label_counts = {label: 0 for label in CLAIM_LABELS}
        for claim_result in claim_results:
            label_counts[claim_result.label] = label_counts.get(claim_result.label, 0) + 1

        entropy = 0.0
        for count in label_counts.values():
            if count == 0:
                continue
            probability = count / n_claims
            entropy -= probability * math.log2(probability)

        high_entropy_threshold = float(
            self.config.get("entropy_threshold", DEFAULT_HIGH_ENTROPY_THRESHOLD)
        )
        low_entropy_threshold = float(
            self.config.get("low_entropy_threshold", LOW_ENTROPY_THRESHOLD)
        )
        max_entropy = math.log2(len(CLAIM_LABELS))
        evidence = [
            (
                f"Claim-label entropy: {entropy:.2f} / {max_entropy:.2f} "
                f"(entailed={label_counts.get('entailed', 0)}, "
                f"unsupported={label_counts.get('unsupported', 0)}, "
                f"contradicted={label_counts.get('contradicted', 0)})"
            )
        ]

        if entropy < low_entropy_threshold:
            evidence.append("LOW uncertainty — claim labels are semantically consistent")
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="pass",
                score=entropy,
                evidence=evidence,
            )
        if entropy <= high_entropy_threshold:
            evidence.append("MEDIUM uncertainty — claim labels split across meaning-groups")
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="warn",
                score=entropy,
                evidence=evidence,
                remediation="Moderate uncertainty detected. Consider additional verification.",
            )

        evidence.append("HIGH uncertainty — claim labels strongly disagree, confabulation likely")
        result = self._fail(
            failure_type=FailureType.LOW_CONFIDENCE,
            stage=FailureStage.CONFIDENCE,
            evidence=evidence,
            remediation="High response variance detected. Do not serve this answer. Consider retrieval expansion or abstention.",
        )
        result.score = entropy
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
        nli_model = self.config.get("nli_model")

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
        clusterer = NLIClusterer(nli_model=nli_model, llm_fn=llm_fn)
        clusters = clusterer.cluster(samples)
        n_clusters = len(clusters)

        # Step 3: Compute entropy
        entropy = _compute_entropy(clusters, n_samples)

        # Compute sample agreement (fraction in largest cluster)
        largest_cluster_size = max(len(cluster) for cluster in clusters) if clusters else 0
        sample_agreement = largest_cluster_size / n_samples if n_samples > 0 else 0.0

        # Build evidence
        evidence = [
            (
                f"Semantic entropy ({clusterer.tier_label}): {entropy:.2f} — "
                f"sampled {n_samples} completions, found {n_clusters} meaning-groups"
            ),
            (
                f"Sample agreement: {sample_agreement:.2%} "
                f"(largest cluster: {largest_cluster_size}/{n_samples})"
            ),
        ]
        if clusterer.tier_label == "Jaccard-augmented proxy":
            evidence.append(
                "Note: using Jaccard-augmented clustering (approximate). "
                "Install sentence-transformers or provide llm_fn for NLI-based clustering."
            )

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

    def _get_claim_results(self, run: RAGRun) -> list[ClaimResult]:
        """Return normalized claim results from prior grounding output or run metadata."""
        prior_results = self.config.get("prior_results", [])
        for result in prior_results:
            if result.analyzer_name != "ClaimGroundingAnalyzer":
                continue
            typed_claim_results = self._normalize_claim_results(result.claim_results)
            if typed_claim_results:
                return typed_claim_results
            claim_results = self._claim_results_from_evidence(result.evidence)
            if claim_results:
                return claim_results

        return self._normalize_claim_results(run.metadata.get("claim_results"))

    def _claim_results_from_evidence(self, evidence: list[str]) -> list[ClaimResult]:
        """Parse ClaimGroundingAnalyzer evidence into claim results."""
        if not evidence:
            return []

        parsed_results: list[ClaimResult] = []
        for item in evidence:
            try:
                parsed = json.loads(item)
            except json.JSONDecodeError:
                continue

            if isinstance(parsed, dict) and "claim_results" in parsed:
                return self._normalize_claim_results(parsed["claim_results"])
            if isinstance(parsed, dict):
                try:
                    parsed_results.append(ClaimResult(**parsed))
                except (TypeError, ValidationError):
                    continue

        return parsed_results

    def _normalize_claim_results(self, raw_claim_results: Any) -> list[ClaimResult]:
        """Normalize claim results from models or dict payloads."""
        if not raw_claim_results:
            return []

        normalized: list[ClaimResult] = []
        for item in raw_claim_results:
            if isinstance(item, ClaimResult):
                normalized.append(item)
                continue
            if isinstance(item, dict):
                try:
                    normalized.append(ClaimResult(**item))
                except (TypeError, ValidationError):
                    continue
        return normalized
