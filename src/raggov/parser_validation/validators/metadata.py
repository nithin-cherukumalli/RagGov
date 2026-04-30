from __future__ import annotations

from collections.abc import Callable
from typing import Sequence

from raggov.parser_validation.models import (
    ChunkIR,
    ChunkingStrategyProfile,
    ParsedDocumentIR,
    ParserEvidence,
    ParserFailureType,
    ParserFinding,
    ParserSeverity,
    ParserValidationConfig,
)


METADATA_REMEDIATION = (
    "Preserve document metadata, page spans, source element IDs, table IDs, and section labels "
    "during parsing and chunking. Do not rely only on final chunk text."
)

_PARENT_ID_KEYS = frozenset({"parent_id", "parent_node_id", "parent_document_id", "doc_id"})


class MetadataValidator:
    """
    Validate parser/chunker metadata coverage.

    Checks are gated on ChunkingStrategyProfile fields so that strategies that
    do not promise metadata (UNKNOWN, FIXED_TOKEN, RECURSIVE_TEXT, SENTENCE) do
    not produce spurious noise.
    """

    name = "metadata_validator"

    def validate(
        self,
        parsed_doc: ParsedDocumentIR | None,
        chunks: Sequence[ChunkIR],
        config: ParserValidationConfig,
    ) -> list[ParserFinding]:
        if not chunks:
            return []

        profile = config.chunking_profile
        findings: list[ParserFinding] = []

        # Page metadata — only enforce when the strategy promises it.
        page_coverage = self._coverage(chunks, lambda chunk: chunk.page_start is not None)
        if profile.requires_page_metadata and page_coverage < config.min_metadata_coverage:
            findings.append(
                ParserFinding(
                    failure_type=ParserFailureType.METADATA_LOSS,
                    severity=ParserSeverity.WARN,
                    confidence=min(0.95, 1.0 - page_coverage),
                    validator_name=self.name,
                    evidence=(
                        ParserEvidence(
                            message="Low page metadata coverage across chunks.",
                            expected={">=": config.min_metadata_coverage},
                            observed={
                                "page_metadata_coverage": round(page_coverage, 3),
                                "chunks_with_page_metadata": self._count(
                                    chunks,
                                    lambda chunk: chunk.page_start is not None,
                                ),
                                "total_chunks": len(chunks),
                            },
                        ),
                    ),
                    remediation=METADATA_REMEDIATION,
                )
            )

        # Provenance coverage — only enforce when the strategy promises it.
        provenance_coverage = self._coverage(
            chunks,
            lambda chunk: bool(chunk.source_element_ids or chunk.source_table_ids),
        )
        if profile.requires_provenance and provenance_coverage < config.min_provenance_coverage:
            findings.append(
                ParserFinding(
                    failure_type=ParserFailureType.PROVENANCE_MISSING,
                    severity=ParserSeverity.WARN,
                    confidence=min(0.95, 1.0 - provenance_coverage),
                    validator_name=self.name,
                    evidence=(
                        ParserEvidence(
                            message="Low source element/table provenance coverage across chunks.",
                            expected={">=": config.min_provenance_coverage},
                            observed={
                                "provenance_coverage": round(provenance_coverage, 3),
                                "chunks_with_provenance": self._count(
                                    chunks,
                                    lambda chunk: bool(
                                        chunk.source_element_ids or chunk.source_table_ids
                                    ),
                                ),
                                "total_chunks": len(chunks),
                            },
                        ),
                    ),
                    remediation=METADATA_REMEDIATION,
                )
            )

        # Parent-child strategy requires parent document metadata.
        if profile.requires_parent_id:
            findings.extend(self._validate_parent_id_coverage(chunks))

        # ParsedDocumentIR-aware cross-checks (only when IR is provided).
        if parsed_doc and parsed_doc.has_structural_metadata:
            findings.extend(
                self._validate_expected_metadata_from_parsed_doc(parsed_doc, chunks, profile)
            )

        return findings

    def _validate_expected_metadata_from_parsed_doc(
        self,
        parsed_doc: ParsedDocumentIR,
        chunks: Sequence[ChunkIR],
        profile: ChunkingStrategyProfile,
    ) -> list[ParserFinding]:
        findings: list[ParserFinding] = []

        if parsed_doc.elements:
            chunks_with_element_ids = [
                chunk for chunk in chunks if chunk.source_element_ids
            ]

            if not chunks_with_element_ids and (
                profile.preserves_source_elements or profile.requires_provenance
            ):
                findings.append(
                    ParserFinding(
                        failure_type=ParserFailureType.PROVENANCE_MISSING,
                        severity=ParserSeverity.FAIL,
                        confidence=0.90,
                        validator_name=self.name,
                        evidence=(
                            ParserEvidence(
                                message=(
                                    "ParsedDocumentIR contains source elements, but chunks do not preserve "
                                    "source_element_ids."
                                ),
                                expected={
                                    "source_element_ids": "present on chunks",
                                    "parsed_element_count": len(parsed_doc.elements),
                                },
                                observed="no chunks contain source_element_ids",
                            ),
                        ),
                        remediation=METADATA_REMEDIATION,
                    )
                )

        if parsed_doc.tables:
            chunks_with_table_ids = [
                chunk for chunk in chunks if chunk.source_table_ids
            ]

            if not chunks_with_table_ids and (
                profile.preserves_table_structure or profile.requires_provenance
            ):
                findings.append(
                    ParserFinding(
                        failure_type=ParserFailureType.PROVENANCE_MISSING,
                        severity=ParserSeverity.FAIL,
                        confidence=0.90,
                        validator_name=self.name,
                        evidence=(
                            ParserEvidence(
                                message=(
                                    "ParsedDocumentIR contains tables, but chunks do not preserve "
                                    "source_table_ids."
                                ),
                                expected={
                                    "source_table_ids": "present on chunks",
                                    "parsed_table_count": len(parsed_doc.tables),
                                },
                                observed="no chunks contain source_table_ids",
                            ),
                        ),
                        remediation=METADATA_REMEDIATION,
                    )
                )

        return findings

    def _validate_parent_id_coverage(
        self,
        chunks: Sequence[ChunkIR],
    ) -> list[ParserFinding]:
        chunks_with_parent = [
            chunk for chunk in chunks
            if any(k in chunk.metadata for k in _PARENT_ID_KEYS)
        ]

        if chunks_with_parent:
            return []

        return [
            ParserFinding(
                failure_type=ParserFailureType.PROVENANCE_MISSING,
                severity=ParserSeverity.WARN,
                confidence=0.80,
                validator_name=self.name,
                evidence=(
                    ParserEvidence(
                        message=(
                            "Parent-child strategy requires parent document metadata, "
                            "but no parent_id-like metadata was found."
                        ),
                        observed={
                            "total_chunks": len(chunks),
                            "parent_keys_checked": sorted(_PARENT_ID_KEYS),
                        },
                    ),
                ),
                remediation=METADATA_REMEDIATION,
            )
        ]

    def _coverage(
        self,
        chunks: Sequence[ChunkIR],
        predicate: Callable[[ChunkIR], bool],
    ) -> float:
        if not chunks:
            return 1.0

        return self._count(chunks, predicate) / len(chunks)

    def _count(
        self,
        chunks: Sequence[ChunkIR],
        predicate: Callable[[ChunkIR], bool],
    ) -> int:
        return sum(1 for chunk in chunks if predicate(chunk))
