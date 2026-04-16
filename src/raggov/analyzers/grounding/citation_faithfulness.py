"""Adversarial citation faithfulness probing for RAG answers."""

from __future__ import annotations

import re

from raggov.analyzers.base import BaseAnalyzer
from raggov.analyzers.retrieval.scope import STOPWORDS
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.run import RAGRun


REMEDIATION = (
    "Citations appear superficial. Model may be generating from parametric "
    "knowledge. Add explicit grounding instructions: require the model to quote "
    "the specific chunk text it is using."
)
ANSWER_CONFIDENCE_THRESHOLD = 0.7
CLAIM_SUPPORT_THRESHOLD = 0.4


class CitationFaithfulnessProbe(BaseAnalyzer):
    """Probe whether citations reflect genuine document reliance."""

    weight = 0.85

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        min_anchor_terms = int(self.config.get("min_anchor_terms", 2))
        suspicious_threshold = int(self.config.get("suspicious_threshold", 2))
        cited_chunks = self._chunks_by_doc_id(run.retrieved_chunks)
        answer_terms = self._terms(run.final_answer)

        probe_results: list[dict] = []
        evidence: list[str] = []
        suspicious_anchor_docs: list[str] = []
        partial_anchor_docs: list[str] = []

        for doc_id in run.cited_doc_ids:
            chunks = cited_chunks.get(doc_id, [])
            if not chunks:
                probe_results.append(
                    {
                        "doc_id": doc_id,
                        "probe": "lexical_anchor",
                        "passed": False,
                        "anchor_terms_found": [],
                    }
                )
                suspicious_anchor_docs.append(doc_id)
                continue

            rare_terms = self._rare_terms_for_doc(doc_id, cited_chunks)
            evaluable_rare_terms = rare_terms if len(rare_terms) >= min_anchor_terms else set()
            anchor_terms_found = sorted(answer_terms & evaluable_rare_terms)
            passed = len(anchor_terms_found) >= min_anchor_terms or not evaluable_rare_terms
            probe_results.append(
                {
                    "doc_id": doc_id,
                    "probe": "lexical_anchor",
                    "passed": passed,
                    "anchor_terms_found": anchor_terms_found,
                }
            )

            if evaluable_rare_terms and not anchor_terms_found:
                suspicious_anchor_docs.append(doc_id)
            elif evaluable_rare_terms and len(anchor_terms_found) < min_anchor_terms:
                partial_anchor_docs.append(doc_id)

        if suspicious_anchor_docs:
            evidence.append(
                "Lexical anchor probe failed for cited docs: "
                + ", ".join(sorted(suspicious_anchor_docs))
            )
        elif partial_anchor_docs:
            evidence.append(
                "Lexical anchor probe found weak unique-term usage for cited docs: "
                + ", ".join(sorted(partial_anchor_docs))
            )

        if not run.cited_doc_ids and run.final_answer.strip() and self._is_confident(run):
            evidence.append("Answer-without-citation probe failed: no citations provided")

        uncited_claims = self._uncited_claims(run, cited_chunks, probe_results)
        total_claims = len(self._claims(run.final_answer))
        uncited_fraction = (len(uncited_claims) / total_claims) if total_claims else 0.0

        if uncited_claims:
            evidence.append(
                f"Citation coverage gap: {len(uncited_claims)} of {total_claims} claims lack cited support"
            )

        fail = (
            len(suspicious_anchor_docs) >= suspicious_threshold
            or (not run.cited_doc_ids and run.final_answer.strip() and self._is_confident(run))
            or uncited_fraction > 0.4
        )
        warn = (
            not fail
            and (
                bool(suspicious_anchor_docs)
                or bool(partial_anchor_docs)
                or uncited_fraction > 0
            )
        )

        if fail:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="fail",
                failure_type=FailureType.POST_RATIONALIZED_CITATION,
                stage=FailureStage.GROUNDING,
                evidence=evidence,
                remediation=REMEDIATION,
                citation_probe_results=probe_results,
            )
        if warn:
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
            evidence=["Citation faithfulness probes indicate genuine reliance."],
            citation_probe_results=probe_results,
        )

    def _chunks_by_doc_id(
        self, chunks: list[RetrievedChunk]
    ) -> dict[str, list[RetrievedChunk]]:
        grouped: dict[str, list[RetrievedChunk]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.source_doc_id, []).append(chunk)
        return grouped

    def _rare_terms_for_doc(
        self,
        doc_id: str,
        cited_chunks: dict[str, list[RetrievedChunk]],
    ) -> set[str]:
        doc_terms: set[str] = set()
        other_terms: set[str] = set()

        for current_doc_id, chunks in cited_chunks.items():
            merged_terms: set[str] = set()
            for chunk in chunks:
                merged_terms |= self._terms(chunk.text)
            if current_doc_id == doc_id:
                doc_terms |= merged_terms
            else:
                other_terms |= merged_terms

        return doc_terms - other_terms

    def _uncited_claims(
        self,
        run: RAGRun,
        cited_chunks: dict[str, list[RetrievedChunk]],
        probe_results: list[dict],
    ) -> list[str]:
        uncited: list[str] = []
        for claim in self._claims(run.final_answer):
            claim_terms = self._terms(claim)
            if not claim_terms:
                continue

            supported = False
            supporting_docs: list[str] = []
            for doc_id in run.cited_doc_ids:
                chunks = cited_chunks.get(doc_id, [])
                if not chunks:
                    continue
                best_overlap = 0.0
                for chunk in chunks:
                    chunk_terms = self._terms(chunk.text)
                    overlap = len(claim_terms & chunk_terms) / len(claim_terms)
                    best_overlap = max(best_overlap, overlap)
                if best_overlap >= CLAIM_SUPPORT_THRESHOLD:
                    supported = True
                    supporting_docs.append(doc_id)

            probe_results.append(
                {
                    "doc_id": ",".join(supporting_docs) if supporting_docs else "none",
                    "probe": "citation_coverage",
                    "passed": supported,
                    "anchor_terms_found": sorted(claim_terms)[:5],
                }
            )

            if not supported:
                uncited.append(claim)
        return uncited

    def _claims(self, answer: str) -> list[str]:
        claims = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", answer) if segment.strip()]
        return claims if claims else ([answer.strip()] if answer.strip() else [])

    def _terms(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in STOPWORDS
        }

    def _is_confident(self, run: RAGRun) -> bool:
        return run.answer_confidence is not None and run.answer_confidence >= ANSWER_CONFIDENCE_THRESHOLD
