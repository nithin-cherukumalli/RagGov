"""Citation faithfulness probing based on adversarial reliance signals.

Wallat et al. argue that citation correctness is not the same as citation
faithfulness: a model can generate an answer from parametric memory and attach a
superficially matching citation after the fact. This probe approximates their
counterfactual reliance test without requiring an extra model call by checking
whether cited documents contain the answer's verifiable anchors and uniquely
contribute its content-bearing predicates.
"""

from __future__ import annotations

import re
from typing import Any

from raggov.analyzers.base import BaseAnalyzer
from raggov.analyzers.retrieval.scope import STOPWORDS
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.run import RAGRun


REMEDIATION = (
    "Citations appear post-rationalized: answer anchors or predicates do not "
    "discriminatively match the cited document. Add explicit grounding instructions "
    "requiring the model to quote the specific chunk text it uses."
)

ANCHOR_HIT_THRESHOLD = 0.5
UNIQUE_PREDICATE_THRESHOLD = 0.3
CONFIDENT_NO_CITATION_THRESHOLD = 0.7
CLAIM_COVERAGE_THRESHOLD = 0.4


class CitationFaithfulnessProbe(BaseAnalyzer):
    """Probe whether citations reflect genuine document reliance."""

    weight = 0.85

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        if not run.final_answer or not run.final_answer.strip():
            return self.skip("no final answer to probe")
        if not run.retrieved_chunks:
            return self.skip("no retrieved chunks available")

        suspicious_threshold = int(self.config.get("suspicious_threshold", 2))

        if not run.cited_doc_ids:
            if self._is_confident(run):
                return self._fail_post_rationalized(
                    [
                        "Answer-without-citation probe failed: "
                        f"confident answer ({run.answer_confidence or 0.0:.2f}) with no citations provided"
                    ],
                    probe_results=[],
                )
            return self.skip("no cited_doc_ids provided")

        chunks_by_doc = self._chunks_by_doc_id(run.retrieved_chunks)
        answer = run.final_answer

        probe_results: list[dict[str, Any]] = []
        evidence: list[str] = []
        fail_docs: list[str] = []
        warn_docs: list[str] = []

        for doc_id in run.cited_doc_ids:
            doc_chunks = chunks_by_doc.get(doc_id, [])
            doc_text = " ".join(chunk.text for chunk in doc_chunks) if doc_chunks else ""
            other_texts = [
                " ".join(chunk.text for chunk in chunks)
                for other_id, chunks in chunks_by_doc.items()
                if other_id != doc_id
            ]

            anchor_result = self._anchor_probe(answer, doc_text, other_texts, doc_id)
            predicate_result = self._unique_predicate_probe(answer, doc_text, other_texts, doc_id)
            probe_results.extend([anchor_result, predicate_result])

            anchor_passed = bool(anchor_result["passed"])
            predicate_passed = bool(predicate_result["passed"])

            if not anchor_passed and not predicate_passed:
                fail_docs.append(doc_id)
                evidence.append(
                    f"{doc_id}: both anchor probe and predicate probe failed — "
                    f"anchor_hit={anchor_result['ratio']:.2f} "
                    f"(threshold {ANCHOR_HIT_THRESHOLD:.2f}), "
                    f"unique_predicate={predicate_result['ratio']:.2f} "
                    f"(threshold {UNIQUE_PREDICATE_THRESHOLD:.2f})"
                )
            elif not anchor_passed or not predicate_passed:
                warn_docs.append(doc_id)
                failed_probe = "anchor" if not anchor_passed else "predicate"
                ratio = anchor_result["ratio"] if not anchor_passed else predicate_result["ratio"]
                evidence.append(f"{doc_id}: {failed_probe} probe failed — ratio={ratio:.2f}")

        coverage_result = self._claim_coverage_probe(answer, chunks_by_doc, run.cited_doc_ids)
        probe_results.extend(coverage_result["probe_results"])
        if coverage_result["uncited_fraction"] > 0.4:
            fail_docs.append("_claim_coverage")
            evidence.append(
                f"Claim coverage gap: {coverage_result['uncited_count']} of "
                f"{coverage_result['total_claims']} claims lack anchor support "
                f"in any cited document"
            )

        if fail_docs or len(warn_docs) >= suspicious_threshold:
            return self._fail_post_rationalized(evidence, probe_results)
        if warn_docs:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="warn",
                failure_type=FailureType.POST_RATIONALIZED_CITATION,
                stage=FailureStage.GROUNDING,
                evidence=evidence,
                remediation=REMEDIATION,
                citation_probe_results=probe_results,
            )
        return AnalyzerResult(
            analyzer_name=self.name(),
            status="pass",
            evidence=["Citation faithfulness probes indicate genuine document reliance."],
            citation_probe_results=probe_results,
        )

    def _anchor_probe(
        self,
        answer: str,
        doc_text: str,
        other_texts: list[str],
        doc_id: str,
    ) -> dict[str, Any]:
        """Check whether answer anchors appear in the cited document."""
        anchors = self._extract_anchors(answer)
        if not anchors:
            return {
                "doc_id": doc_id,
                "probe": "anchor",
                "passed": True,
                "ratio": 1.0,
                "anchors_checked": 0,
                "anchors_hit": 0,
            }

        doc_lower = doc_text.lower()
        other_lower = " ".join(other_texts).lower()
        hits = sum(1 for anchor in anchors if anchor in doc_lower)
        unique_hits = sum(1 for anchor in anchors if anchor in doc_lower and anchor not in other_lower)
        ratio = hits / len(anchors)

        return {
            "doc_id": doc_id,
            "probe": "anchor",
            "passed": ratio >= ANCHOR_HIT_THRESHOLD,
            "ratio": ratio,
            "anchors_checked": len(anchors),
            "anchors_hit": hits,
            "anchors_unique_to_doc": unique_hits,
        }

    def _unique_predicate_probe(
        self,
        answer: str,
        doc_text: str,
        other_texts: list[str],
        doc_id: str,
    ) -> dict[str, Any]:
        """Check whether the cited document uniquely contributes answer predicates."""
        answer_predicates = self._predicate_terms(answer)
        doc_lower = doc_text.lower()

        predicates_in_doc = {predicate for predicate in answer_predicates if predicate in doc_lower}
        if not predicates_in_doc:
            return {
                "doc_id": doc_id,
                "probe": "unique_predicate",
                "passed": False,
                "ratio": 0.0,
                "predicates_checked": 0,
                "unique_predicates": 0,
            }

        other_lower = " ".join(other_texts).lower()
        unique_to_doc = {predicate for predicate in predicates_in_doc if predicate not in other_lower}
        ratio = len(unique_to_doc) / len(predicates_in_doc)

        return {
            "doc_id": doc_id,
            "probe": "unique_predicate",
            "passed": ratio >= UNIQUE_PREDICATE_THRESHOLD,
            "ratio": ratio,
            "predicates_checked": len(predicates_in_doc),
            "unique_predicates": len(unique_to_doc),
        }

    def _claim_coverage_probe(
        self,
        answer: str,
        chunks_by_doc: dict[str, list[RetrievedChunk]],
        cited_doc_ids: list[str],
    ) -> dict[str, Any]:
        """Check that each answer claim has support in at least one cited document."""
        claims = self._split_claims(answer)
        probe_results: list[dict[str, Any]] = []
        uncited_count = 0

        for claim in claims:
            anchors = self._extract_anchors(claim)
            content_terms = self._predicate_terms(claim)
            check_terms = anchors if anchors else sorted(content_terms)

            if not check_terms:
                continue

            supported = False
            supporting_doc_id = "none"
            for doc_id in cited_doc_ids:
                doc_text = " ".join(
                    chunk.text for chunk in chunks_by_doc.get(doc_id, [])
                ).lower()
                hits = sum(1 for term in check_terms if term in doc_text)
                if hits / len(check_terms) >= CLAIM_COVERAGE_THRESHOLD:
                    supported = True
                    supporting_doc_id = doc_id
                    break

            probe_results.append(
                {
                    "doc_id": supporting_doc_id,
                    "probe": "claim_coverage",
                    "passed": supported,
                    "claim": claim[:80],
                }
            )

            if not supported:
                uncited_count += 1

        total_claims = len(claims)
        return {
            "probe_results": probe_results,
            "uncited_count": uncited_count,
            "total_claims": total_claims,
            "uncited_fraction": (uncited_count / total_claims) if total_claims else 0.0,
        }

    def _extract_anchors(self, text: str) -> list[str]:
        """Extract verifiable anchors such as numbers and named entities."""
        anchors: set[str] = set()
        anchors.update(
            re.findall(
                r"\b\d+(?:[.,]\d+)?(?:%|k|m|bn|million|billion)?\b",
                text.lower(),
            )
        )
        named_entities = re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b", text)
        anchors.update(
            entity.lower()
            for entity in named_entities
            if entity.lower() not in STOPWORDS
        )
        return sorted(anchors)

    def _predicate_terms(self, text: str) -> set[str]:
        """Return content-bearing terms used to discriminate cited documents."""
        anchor_terms = set(self._extract_anchors(text))
        return {
            token
            for token in re.findall(r"[a-z]{4,}", text.lower())
            if token not in STOPWORDS and token not in anchor_terms
        }

    def _split_claims(self, answer: str) -> list[str]:
        """Split an answer into sentence-level claims."""
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", answer)
            if len(sentence.strip()) > 20
        ]

    def _chunks_by_doc_id(
        self, chunks: list[RetrievedChunk]
    ) -> dict[str, list[RetrievedChunk]]:
        grouped: dict[str, list[RetrievedChunk]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.source_doc_id, []).append(chunk)
        return grouped

    def _is_confident(self, run: RAGRun) -> bool:
        return (
            run.answer_confidence is not None
            and run.answer_confidence >= CONFIDENT_NO_CITATION_THRESHOLD
        )

    def _fail_post_rationalized(
        self, evidence: list[str], probe_results: list[dict[str, Any]]
    ) -> AnalyzerResult:
        return AnalyzerResult(
            analyzer_name=self.name(),
            status="fail",
            failure_type=FailureType.POST_RATIONALIZED_CITATION,
            stage=FailureStage.GROUNDING,
            evidence=evidence,
            remediation=REMEDIATION,
            citation_probe_results=probe_results,
        )
