from __future__ import annotations

import re
from typing import Sequence

from raggov.parser_validation.models import (
    ChunkIR,
    ChunkingStrategyProfile,
    ElementIR,
    ParsedDocumentIR,
    ParserEvidence,
    ParserFailureType,
    ParserFinding,
    ParserSeverity,
    ParserValidationConfig,
)


HIERARCHY_REMEDIATION = (
    "Use heading-aware parsing and hierarchical chunking. Preserve section_path "
    "on every chunk and avoid merging unrelated sibling sections."
)

_HIERARCHY_PRESERVE_VALUES = frozenset(
    {"preserve", "preserved", "section_path", "hierarchical", "markdown_header", "headers"}
)
_HIERARCHY_SUPPRESS_VALUES = frozenset(
    {"summary", "summarized", "suppress", "suppressed", "none", "ignore"}
)


class HierarchyValidator:
    """
    Validate whether document hierarchy survived parsing/chunking.

    Strong mode:
    - Enforced when profile.preserves_section_path is True (HIERARCHICAL, MARKDOWN_HEADER).
    - Uses ParsedDocumentIR.elements and chunk.section_path.

    Permissive mode:
    - UNKNOWN and token-based strategies do not declare section_path preservation.
    - Cross-section chunks are allowed when profile.allows_cross_section_chunk is True.

    Fallback mode:
    - Inline heading marker smoke tests run when enable_text_fallback_heuristics is True.
    - Fallback findings are always WARN (never FAIL) and marked is_heuristic=True.
    - Skipped entirely if profile.table_check_suppressed (SUMMARY strategy).
    """

    name = "hierarchy_validator"

    _inline_heading_re = re.compile(
        r"(?i)\b(part\s+[ivxlcdm]+|chapter\s+\d+|section\s+\d+|rule\s+\d+|annexure\s+[a-z0-9]+)\b"
    )

    _numbered_clause_re = re.compile(r"(?m)(\(\d+\)|\b\d+\.\s+[A-Z][A-Za-z]+)")

    def validate(
        self,
        parsed_doc: ParsedDocumentIR | None,
        chunks: Sequence[ChunkIR],
        config: ParserValidationConfig,
    ) -> list[ParserFinding]:
        profile = config.chunking_profile
        findings: list[ParserFinding] = []

        if parsed_doc and self._document_has_hierarchy(parsed_doc):
            findings.extend(
                self._validate_section_path_coverage(parsed_doc, chunks, config, profile)
            )
            findings.extend(self._validate_section_boundary_flags(chunks, profile))

        if config.enable_text_fallback_heuristics:
            findings.extend(self._fallback_inline_hierarchy_smoke_test(chunks, profile))

        return findings

    def _validate_section_path_coverage(
        self,
        parsed_doc: ParsedDocumentIR,
        chunks: Sequence[ChunkIR],
        config: ParserValidationConfig,
        profile: ChunkingStrategyProfile,
    ) -> list[ParserFinding]:
        # Only enforce when the strategy promises to preserve section hierarchy.
        if not self._preserves_section_path(profile):
            return []

        chunks_with_provenance = [
            chunk for chunk in chunks if chunk.source_element_ids or chunk.source_table_ids
        ]

        if not chunks_with_provenance:
            return []

        chunks_missing_section = [
            chunk for chunk in chunks_with_provenance if not chunk.section_path
        ]

        missing_ratio = len(chunks_missing_section) / max(len(chunks_with_provenance), 1)

        if missing_ratio <= (1.0 - config.min_metadata_coverage):
            return []

        return [
            ParserFinding(
                failure_type=ParserFailureType.HIERARCHY_FLATTENING,
                severity=ParserSeverity.WARN,
                confidence=min(0.95, 0.60 + missing_ratio),
                validator_name=self.name,
                evidence=(
                    ParserEvidence(
                        message=(
                            "Source document has hierarchy, but many chunks with provenance "
                            "lack section_path metadata."
                        ),
                        expected={
                            "section_path": "present on chunks with provenance",
                            "hierarchical_element_count": len(
                                [element for element in parsed_doc.elements if element.section_path]
                            ),
                        },
                        observed={
                            "chunks_with_provenance": len(chunks_with_provenance),
                            "chunks_missing_section_path": len(chunks_missing_section),
                            "missing_ratio": round(missing_ratio, 3),
                            "sample_chunk_ids": [
                                chunk.chunk_id for chunk in chunks_missing_section[:10]
                            ],
                        },
                    ),
                ),
                remediation=HIERARCHY_REMEDIATION,
            )
        ]

    def _validate_section_boundary_flags(
        self,
        chunks: Sequence[ChunkIR],
        profile: ChunkingStrategyProfile,
    ) -> list[ParserFinding]:
        # Cross-section chunks are acceptable for strategies that permit them.
        if self._allows_cross_section_chunk(profile):
            return []

        boundary_damaged_chunks = [
            chunk
            for chunk in chunks
            if bool(chunk.metadata.get("crosses_section_boundary"))
        ]

        if not boundary_damaged_chunks:
            return []

        return [
            ParserFinding(
                failure_type=ParserFailureType.HIERARCHY_FLATTENING,
                severity=ParserSeverity.WARN,
                confidence=0.75,
                validator_name=self.name,
                evidence=(
                    ParserEvidence(
                        message=(
                            "Chunks are explicitly marked as crossing section boundaries."
                        ),
                        observed={
                            "cross_section_chunk_count": len(boundary_damaged_chunks),
                            "sample_chunk_ids": [
                                chunk.chunk_id for chunk in boundary_damaged_chunks[:10]
                            ],
                        },
                    ),
                ),
                remediation=HIERARCHY_REMEDIATION,
            )
        ]

    def _fallback_inline_hierarchy_smoke_test(
        self,
        chunks: Sequence[ChunkIR],
        profile: ChunkingStrategyProfile,
    ) -> list[ParserFinding]:
        # Summary chunks may legitimately compress headings; do not flag them.
        if profile.table_check_suppressed or self._hierarchy_checks_suppressed(profile):
            return []

        findings: list[ParserFinding] = []

        for chunk in chunks:
            if chunk.section_path:
                continue

            inline_markers = self._inline_heading_re.findall(chunk.text)
            normalized_markers = sorted({marker.lower() for marker in inline_markers})

            numbered_markers = self._numbered_clause_re.findall(chunk.text)
            flattened_numbered_rule = bool(normalized_markers) and len(numbered_markers) >= 3
            hierarchy_contract_broken = (
                self._preserves_section_path(profile)
                and not chunk.section_path
                and flattened_numbered_rule
            )

            if len(normalized_markers) >= 3 or len(numbered_markers) >= 4 or flattened_numbered_rule:
                confidence = 0.85 if flattened_numbered_rule else 0.60
                if hierarchy_contract_broken:
                    message = (
                        "Text-only smoke test: strong numbered hierarchy markers appear inline "
                        "without section_path metadata, violating the declared hierarchy-preserving "
                        "chunking strategy contract."
                    )
                else:
                    message = (
                        "Text-only smoke test: multiple hierarchy markers appear inline "
                        "without section_path metadata."
                    )
                findings.append(
                    ParserFinding(
                        failure_type=ParserFailureType.HIERARCHY_FLATTENING,
                        severity=(
                            ParserSeverity.FAIL
                            if hierarchy_contract_broken
                            else ParserSeverity.WARN
                        ),
                        confidence=confidence,
                        validator_name=self.name,
                        evidence=(
                            ParserEvidence(
                                message=message,
                                chunk_id=chunk.chunk_id,
                                observed={
                                    "inline_hierarchy_markers": normalized_markers[:10],
                                    "numbered_marker_count": len(numbered_markers),
                                    "chunking_strategy": profile.strategy_type.value,
                                    "preserves_section_path": self._preserves_section_path(profile),
                                },
                            ),
                        ),
                        remediation=HIERARCHY_REMEDIATION,
                        alternative_explanations=(
                            "The source text may legitimately reference multiple sections.",
                            "The chunk may be a table of contents or summary paragraph.",
                        ),
                        is_heuristic=True,
                    )
                )

        return findings

    def _document_has_hierarchy(self, parsed_doc: ParsedDocumentIR) -> bool:
        return any(self._element_is_hierarchical(element) for element in parsed_doc.elements)

    def _element_is_hierarchical(self, element: ElementIR) -> bool:
        if element.section_path:
            return True

        normalized_type = element.element_type.lower()
        return normalized_type in {
            "title",
            "heading",
            "header",
            "section",
            "chapter",
            "part",
            "annexure",
        }

    def _preserves_section_path(self, profile: ChunkingStrategyProfile) -> bool:
        value = self._profile_value(profile.hierarchy_mode)
        return profile.preserves_section_path or value in _HIERARCHY_PRESERVE_VALUES

    def _allows_cross_section_chunk(self, profile: ChunkingStrategyProfile) -> bool:
        value = self._profile_value(profile.hierarchy_mode)
        if value in _HIERARCHY_PRESERVE_VALUES:
            return False
        return profile.allows_cross_section_chunk

    def _hierarchy_checks_suppressed(self, profile: ChunkingStrategyProfile) -> bool:
        return self._profile_value(profile.hierarchy_mode) in _HIERARCHY_SUPPRESS_VALUES

    def _profile_value(self, value: str | None) -> str:
        return (value or "").strip().lower().replace("-", "_")
