"""
Candidate Selection Layer for GovRAG grounding analysis.

Separates the candidate evidence retrieval logic from verification logic.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from raggov.analyzers.grounding.claims import ExtractedClaim
from raggov.analyzers.retrieval.scope import STOPWORDS
from raggov.models.chunk import RetrievedChunk


logger = logging.getLogger(__name__)

ANCHOR_WEIGHT = 0.6
CONTENT_TERM_NORMALIZATIONS: dict[str, str] = {
    "grew": "increase",
    "grow": "increase",
    "growing": "increase",
    "increased": "increase",
    "increasing": "increase",
    "increases": "increase",
    "rose": "increase",
    "rising": "increase",
    "declined": "decrease",
    "decline": "decrease",
    "decreasing": "decrease",
    "decreased": "decrease",
    "falls": "decrease",
    "fell": "decrease",
    "annually": "annual",
    "yearly": "annual",
    "yoy": "annual",
    "yearoveryear": "annual",
}


@dataclass
class EvidenceCandidate:
    """A retrieved chunk selected as candidate evidence for a claim."""

    chunk_id: str
    source_doc_id: str | None
    chunk_text: str
    chunk_text_preview: str
    lexical_overlap_score: float
    anchor_overlap_score: float
    value_overlap_score: float
    retrieval_score: float | None
    rerank_score: float | None
    metadata_match_flags: list[str] = field(default_factory=list)
    candidate_reason: str = ""
    is_best: bool = False
    
    @property
    def raw_support_score(self) -> float:
        # Alias for backward compatibility in ClaimEvidenceRecord mapping
        return self.lexical_overlap_score


class EvidenceCandidateSelector:
    """Select candidate chunks for each claim, without deciding entailment."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def select_candidates(
        self,
        claim: ExtractedClaim | str,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[EvidenceCandidate]:
        if not retrieved_chunks:
            return []
            
        claim_text = claim.claim_text if isinstance(claim, ExtractedClaim) else claim
        mode = self.config.get("candidate_mode", "heuristic_top_k_v0")
        top_k = int(self.config.get("candidate_top_k", 3))

        if mode == "all_chunks_debug":
            return self._select_all(claim_text, retrieved_chunks)
        if mode == "cited_only":
            return self._select_cited_only(claim_text, retrieved_chunks)
        if mode == "retrieved_top_k":
            return self._select_retrieved_top_k(claim_text, retrieved_chunks, top_k)
            
        # Default fallback
        return self._select_heuristic_top_k(claim_text, retrieved_chunks, top_k)

    def _select_all(self, claim_text: str, chunks: list[RetrievedChunk]) -> list[EvidenceCandidate]:
        return [self._build_candidate(chunk, 0.0, 0.0) for chunk in chunks]

    def _select_cited_only(self, claim_text: str, chunks: list[RetrievedChunk]) -> list[EvidenceCandidate]:
        candidates = []
        for chunk in chunks:
            if chunk.chunk_id in claim_text or f"[{chunk.chunk_id}]" in claim_text:
                candidates.append(self._build_candidate(chunk, 1.0, 1.0, reason="Cited in claim text"))
        if candidates:
            candidates[0].is_best = True
        return candidates

    def _select_retrieved_top_k(self, claim_text: str, chunks: list[RetrievedChunk], top_k: int) -> list[EvidenceCandidate]:
        candidates = [self._build_candidate(chunk, 0.0, 0.0, reason="Top retrieved chunk") for chunk in chunks[:top_k]]
        if candidates:
            candidates[0].is_best = True
        return candidates

    def _select_heuristic_top_k(
        self, claim_text: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[EvidenceCandidate]:
        claim_terms = self._content_terms(claim_text)
        claim_anchors = self._extract_anchors(claim_text)
        term_weight = 1.0 - float(self.config.get("anchor_weight", ANCHOR_WEIGHT))
        anchor_weight = float(self.config.get("anchor_weight", ANCHOR_WEIGHT))

        scored_candidates = []
        for chunk in chunks:
            chunk_terms = self._content_terms(chunk.text)
            term_coverage = len(claim_terms & chunk_terms) / len(claim_terms) if claim_terms else 0.0
            
            anchor_coverage = 0.0
            if claim_anchors:
                chunk_anchors = set(self._extract_anchors(chunk.text))
                anchor_hits = len(set(claim_anchors) & chunk_anchors)
                anchor_coverage = anchor_hits / len(set(claim_anchors))
                
            combined_score = (term_weight * term_coverage) + (anchor_weight * anchor_coverage) if claim_anchors else term_coverage
            
            candidate = self._build_candidate(
                chunk,
                lexical_overlap_score=combined_score,
                anchor_overlap_score=anchor_coverage,
                reason="Heuristic overlap score",
            )
            scored_candidates.append((combined_score, candidate))
            
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = [c for score, c in scored_candidates[:top_k]]
        if top_candidates:
            top_candidates[0].is_best = True
        return top_candidates

    def _build_candidate(
        self,
        chunk: RetrievedChunk,
        lexical_overlap_score: float,
        anchor_overlap_score: float,
        reason: str = "",
    ) -> EvidenceCandidate:
        return EvidenceCandidate(
            chunk_id=chunk.chunk_id,
            source_doc_id=chunk.source_doc_id,
            chunk_text=chunk.text,
            chunk_text_preview=chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
            lexical_overlap_score=lexical_overlap_score,
            anchor_overlap_score=anchor_overlap_score,
            value_overlap_score=0.0,
            retrieval_score=chunk.score,
            rerank_score=None,
            candidate_reason=reason,
        )

    def _content_terms(self, text: str) -> set[str]:
        content_terms: set[str] = set()
        for token in self._tokens(text):
            if token in STOPWORDS:
                continue
            normalized = self._normalize_content_term(token)
            if not normalized:
                continue
            if normalized.isdigit() or len(normalized) > 2:
                content_terms.add(normalized)
        return content_terms

    def _normalize_content_term(self, token: str) -> str:
        if not token:
            return ""
        normalized = CONTENT_TERM_NORMALIZATIONS.get(token, token)
        if normalized.endswith("ies") and len(normalized) > 4:
            return normalized[:-3] + "y"
        if normalized.endswith("s") and len(normalized) > 4 and not normalized.endswith("ss"):
            return normalized[:-1]
        return normalized

    def _extract_anchors(self, text: str) -> list[str]:
        anchors: list[str] = []
        lowered = text.lower()
        anchors.extend(
            m.group(0) for m in re.finditer(r"(?:[$€£])?\d[\d,]*(?:\.\d+)?%?", lowered)
        )
        for m in re.finditer(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
            value = m.group(0)
            if " " in value or m.start() > 0:
                anchors.append(value.lower())
        for m in re.finditer(r"\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b", text):
            anchors.append(m.group(0).lower())
        deduped: list[str] = []
        seen: set[str] = set()
        for anchor in anchors:
            if anchor in seen:
                continue
            seen.add(anchor)
            deduped.append(anchor)
        return deduped

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())
