"""
RAGChecker-inspired claim-level diagnostic rollup for GovRAG.

Inspired by RAGChecker-style modular diagnostics, not a full RAGChecker
metric implementation.

RAGChecker (Ye et al., 2024) provides fine-grained diagnostic metrics
separating retrieval and generation failure modes.  GovRAG's approximation
works on top of ClaimEvidenceRecords produced by the heuristic evidence layer
and applies lightweight rules to classify suspected failure patterns.

All diagnostic labels are produced by 'diagnostic_rollup_heuristic_v0'.
They are probabilistic indicators, not ground-truth labels.

Failure mode taxonomy (approximation)
--------------------------------------
retrieval_miss_suspected
    Claim is unsupported AND all candidate chunks have very low overlap scores.
    Interpretation: the retriever probably did not find the right document.

context_ignored_suspected
    Claim is unsupported BUT at least one candidate has moderate-to-high
    overlap.  The evidence may exist but the verifier (or the generator)
    failed to use it — either an insufficient_context gap or a generation
    faithfulness issue.

value_error_count
    Claims where value_conflicts is non-empty, regardless of final label.
    These map directly to numeric/date/identifier disagreements.

citation_mismatch_suspected
    Claim is entailed (by retrieved context) but the chunk that supports it
    does not belong to any of the cited doc IDs.
    Wallat et al.: a citation can look correct while being post-rationalised.

noisy_context_suspected
    Many retrieved chunks are NOT referenced as candidates for any claim.
    High noise in the retrieval set can indicate poor retrieval quality.

stale_source_suspected
    Unsupported claim whose best candidate score is above a moderate threshold
    (evidence exists and overlaps, but values do not match) — possibly
    outdated source content.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from raggov.analyzers.grounding.evidence_layer import ClaimEvidenceRecord
from raggov.models.chunk import RetrievedChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (heuristic, not calibrated)
# ---------------------------------------------------------------------------

#: Below this score, assume no usable candidate was found (retrieval miss).
_RETRIEVAL_MISS_MAX_SCORE: float = 0.15

#: Above this score an unsupported claim is classified as context_ignored
#: rather than retrieval_miss — evidence exists but was not used.
_CONTEXT_IGNORED_MIN_SCORE: float = 0.30

#: Above this score for an unsupported claim whose best candidate has values
#: that partially overlap — suspect a stale source rather than a pure miss.
_STALE_SOURCE_MIN_SCORE: float = 0.25

#: Fraction of retrieved chunks not referenced by ANY claim candidate before
#: we flag possible retrieval noise.
_NOISE_FRACTION_THRESHOLD: float = 0.6

#: Minimum absolute number of unused chunks for noise to be flagged.
_NOISE_MIN_UNUSED: int = 2

_DIAGNOSTIC_VERSION = "diagnostic_rollup_heuristic_v0"


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

@dataclass
class ClaimDiagnosticSummary:
    """
    RAGChecker-inspired diagnostic summary for a single RAG run.

    All counts and rates are derived from ClaimEvidenceRecords using
    lightweight heuristic rules (diagnostic_rollup_heuristic_v0).
    They are diagnostic indicators, not calibrated metrics.

    Inspired by RAGChecker-style modular diagnostics, not a full
    RAGChecker metric implementation.
    """

    # ---- Claim counts ------------------------------------------------------
    total_claims: int
    entailed_claims: int
    unsupported_claims: int
    contradicted_claims: int
    abstained_claims: int

    # ---- Aggregate rates (0.0 – 1.0) ---------------------------------------
    claim_support_rate: float
    """Fraction of claims that are entailed."""

    contradiction_rate: float
    """Fraction of claims that are contradicted."""

    unsupported_rate: float
    """Fraction of claims that are unsupported."""

    # ---- Citation coverage -------------------------------------------------
    citation_support_rate: float
    """
    Fraction of entailed claims whose supporting chunk belongs to a cited doc.
    0.0 when cited_doc_ids is empty or there are no entailed claims.
    """

    # ---- Evidence utilisation ----------------------------------------------
    evidence_utilization_rate: float
    """
    Fraction of retrieved chunks referenced as candidates for at least one claim.
    Low values indicate possible retrieval noise.
    """

    # ---- Suspected failure-mode counts ------------------------------------
    retrieval_miss_suspected_count: int
    """
    Unsupported claims where all candidate scores are very low.
    Likely cause: retriever did not return the right chunk.
    """

    context_ignored_suspected_count: int
    """
    Unsupported claims where at least one candidate has a moderate score.
    Likely cause: evidence was retrieved but not used by the generator.
    """

    value_error_count: int
    """Claims with at least one value_conflict (numeric/date/identifier mismatch)."""

    stale_source_suspected_count: int
    """
    Unsupported claims with partial score but no value matches.
    Possibly outdated source content.
    """

    noisy_context_suspected: bool
    """
    True when a large fraction of retrieved chunks were never referenced as
    candidates for any claim.
    """

    citation_mismatch_suspected_count: int
    """
    Entailed claims whose supporting chunk is NOT in any cited document.
    Post-rationalised citation indicator.
    """

    # ---- Metadata ----------------------------------------------------------
    diagnostic_version: str = _DIAGNOSTIC_VERSION
    has_cited_docs: bool = False
    notes: list[str] = field(default_factory=list)

    # ---- Convenience -------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)

    def failure_pattern_summary(self) -> str:
        """
        Return a human-readable one-line summary of the most prominent
        suspected failure patterns for use in analyzer evidence strings.
        """
        parts: list[str] = []
        if self.retrieval_miss_suspected_count:
            parts.append(f"retrieval_miss×{self.retrieval_miss_suspected_count}")
        if self.context_ignored_suspected_count:
            parts.append(f"context_ignored×{self.context_ignored_suspected_count}")
        if self.value_error_count:
            parts.append(f"value_error×{self.value_error_count}")
        if self.citation_mismatch_suspected_count:
            parts.append(f"citation_mismatch×{self.citation_mismatch_suspected_count}")
        if self.stale_source_suspected_count:
            parts.append(f"stale_source×{self.stale_source_suspected_count}")
        if self.noisy_context_suspected:
            parts.append("noisy_retrieval")
        return ", ".join(parts) if parts else "no_suspected_failures"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class ClaimDiagnosticRollupBuilder:
    """
    Builds a ClaimDiagnosticSummary from a list of ClaimEvidenceRecords.

    Inspired by RAGChecker-style modular diagnostics, not a full RAGChecker
    metric implementation.  All diagnostic labels are produced by
    'diagnostic_rollup_heuristic_v0' and are probabilistic indicators.

    Usage::

        builder = ClaimDiagnosticRollupBuilder()
        summary = builder.build(
            records=claim_evidence_records,
            retrieved_chunks=run.retrieved_chunks,
            cited_doc_ids=list(run.cited_doc_ids),
        )
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._retrieval_miss_max_score: float = float(
            cfg.get("diagnostic_retrieval_miss_max_score", _RETRIEVAL_MISS_MAX_SCORE)
        )
        self._context_ignored_min_score: float = float(
            cfg.get("diagnostic_context_ignored_min_score", _CONTEXT_IGNORED_MIN_SCORE)
        )
        self._stale_source_min_score: float = float(
            cfg.get("diagnostic_stale_source_min_score", _STALE_SOURCE_MIN_SCORE)
        )
        self._noise_fraction_threshold: float = float(
            cfg.get("diagnostic_noise_fraction_threshold", _NOISE_FRACTION_THRESHOLD)
        )
        self._noise_min_unused: int = int(
            cfg.get("diagnostic_noise_min_unused", _NOISE_MIN_UNUSED)
        )

    def build(
        self,
        records: list[ClaimEvidenceRecord],
        retrieved_chunks: list[RetrievedChunk],
        cited_doc_ids: list[str] | None = None,
    ) -> ClaimDiagnosticSummary:
        """
        Build the diagnostic summary.

        Args:
            records: ClaimEvidenceRecords produced by ClaimEvidenceBuilder.
            retrieved_chunks: The full list of chunks that were retrieved for
                              the query (used for utilization/noise analysis).
            cited_doc_ids: Document IDs cited in the generated answer.
                           If None or empty, citation analysis is skipped.
        """
        cited: set[str] = set(cited_doc_ids or [])
        total = len(records)

        if total == 0:
            return self._empty_summary(retrieved_chunks, cited)

        # ---- Label counts --------------------------------------------------
        entailed = [r for r in records if r.verification_label == "entailed"]
        unsupported = [
            r
            for r in records
            if r.verification_label in {"unsupported", "insufficient"}
        ]
        contradicted = [r for r in records if r.verification_label == "contradicted"]
        abstained = [
            r for r in records
            if r.verification_label not in {"entailed", "unsupported", "insufficient", "contradicted"}
        ]

        # ---- Failure mode classification ------------------------------------
        retrieval_miss_count = 0
        context_ignored_count = 0
        stale_source_count = 0
        value_error_count = 0
        citation_mismatch_count = 0

        for rec in unsupported:
            best_score = self._best_candidate_score(rec)
            has_value_match = bool(rec.value_matches)

            if best_score <= self._retrieval_miss_max_score:
                retrieval_miss_count += 1
            elif best_score >= self._context_ignored_min_score:
                context_ignored_count += 1
            elif (
                best_score >= self._stale_source_min_score
                and not has_value_match
            ):
                stale_source_count += 1
            # else: borderline — not classified

        for rec in records:
            if rec.value_conflicts:
                value_error_count += 1

        # ---- Citation mismatch (post-rationalisation indicator) -------------
        if cited:
            for rec in entailed:
                if not self._supporting_chunk_is_cited(rec, cited, retrieved_chunks):
                    citation_mismatch_count += 1

        # ---- Evidence utilisation / noise ----------------------------------
        all_candidate_chunk_ids: set[str] = set()
        for rec in records:
            for cand in rec.candidate_evidence_chunks:
                all_candidate_chunk_ids.add(cand.chunk_id)

        retrieved_chunk_ids = {c.chunk_id for c in retrieved_chunks}
        unused_chunk_ids = retrieved_chunk_ids - all_candidate_chunk_ids
        n_retrieved = len(retrieved_chunk_ids)

        utilization_rate = (
            len(all_candidate_chunk_ids & retrieved_chunk_ids) / n_retrieved
            if n_retrieved > 0
            else 0.0
        )
        noise_fraction = len(unused_chunk_ids) / n_retrieved if n_retrieved > 0 else 0.0
        noisy_context = (
            noise_fraction >= self._noise_fraction_threshold
            and len(unused_chunk_ids) >= self._noise_min_unused
        )

        # ---- Citation support rate -----------------------------------------
        citation_support_rate = 0.0
        if cited and entailed:
            citation_supported = sum(
                1
                for rec in entailed
                if self._supporting_chunk_is_cited(rec, cited, retrieved_chunks)
            )
            citation_support_rate = citation_supported / len(entailed)

        # ---- Rates ---------------------------------------------------------
        def _rate(count: int) -> float:
            return round(count / total, 4) if total > 0 else 0.0

        notes: list[str] = []
        if not cited:
            notes.append(
                "cited_doc_ids not provided; citation_support_rate and "
                "citation_mismatch analysis are unavailable."
            )
        if abstained:
            notes.append(
                f"{len(abstained)} claim(s) abstained from verification; "
                "these are excluded from failure-mode classification."
            )

        return ClaimDiagnosticSummary(
            total_claims=total,
            entailed_claims=len(entailed),
            unsupported_claims=len(unsupported),
            contradicted_claims=len(contradicted),
            abstained_claims=len(abstained),
            claim_support_rate=_rate(len(entailed)),
            contradiction_rate=_rate(len(contradicted)),
            unsupported_rate=_rate(len(unsupported)),
            citation_support_rate=round(citation_support_rate, 4),
            evidence_utilization_rate=round(utilization_rate, 4),
            retrieval_miss_suspected_count=retrieval_miss_count,
            context_ignored_suspected_count=context_ignored_count,
            value_error_count=value_error_count,
            stale_source_suspected_count=stale_source_count,
            noisy_context_suspected=noisy_context,
            citation_mismatch_suspected_count=citation_mismatch_count,
            diagnostic_version=_DIAGNOSTIC_VERSION,
            has_cited_docs=bool(cited),
            notes=notes,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _best_candidate_score(self, record: ClaimEvidenceRecord) -> float:
        """Return the highest raw_support_score among candidate chunks, or 0.0."""
        if not record.candidate_evidence_chunks:
            return 0.0
        return max(c.raw_support_score for c in record.candidate_evidence_chunks)

    def _supporting_chunk_is_cited(
        self,
        record: ClaimEvidenceRecord,
        cited_doc_ids: set[str],
        retrieved_chunks: list[RetrievedChunk],
    ) -> bool:
        """
        Return True if at least one of the claim's supporting chunks belongs
        to a cited document.
        """
        chunk_doc_map: dict[str, str | None] = {
            c.chunk_id: c.source_doc_id for c in retrieved_chunks
        }
        # Also include candidate chunk source_doc_ids for richer lookup
        for cand in record.candidate_evidence_chunks:
            if cand.chunk_id not in chunk_doc_map:
                chunk_doc_map[cand.chunk_id] = cand.source_doc_id

        for chunk_id in record.supporting_chunk_ids:
            doc_id = chunk_doc_map.get(chunk_id)
            if doc_id and doc_id in cited_doc_ids:
                return True
        return False

    def _empty_summary(
        self,
        retrieved_chunks: list[RetrievedChunk],
        cited_doc_ids: set[str],
    ) -> ClaimDiagnosticSummary:
        """Return a zero-value summary for runs with no extracted claims."""
        return ClaimDiagnosticSummary(
            total_claims=0,
            entailed_claims=0,
            unsupported_claims=0,
            contradicted_claims=0,
            abstained_claims=0,
            claim_support_rate=0.0,
            contradiction_rate=0.0,
            unsupported_rate=0.0,
            citation_support_rate=0.0,
            evidence_utilization_rate=0.0,
            retrieval_miss_suspected_count=0,
            context_ignored_suspected_count=0,
            value_error_count=0,
            stale_source_suspected_count=0,
            noisy_context_suspected=False,
            citation_mismatch_suspected_count=0,
            diagnostic_version=_DIAGNOSTIC_VERSION,
            has_cited_docs=bool(cited_doc_ids),
            notes=["No claims were extracted; diagnostic rollup is empty."],
        )
