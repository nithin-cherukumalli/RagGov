"""Profile consistency linting before parser diagnostics."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from raggov.parser_validation.models import ChunkIR, ChunkingStrategyProfile
from raggov.parser_validation.profile import ParserValidationProfile


PROFILE_LINT_REMEDIATION = (
    "Fix the ParserValidationProfile metadata mapping or temporarily enable "
    "infer_from_legacy during migration. Parser diagnostics require trusted "
    "profile-to-metadata alignment before they can claim authority."
)

_PARENT_ID_KEYS = frozenset({"parent_id", "parent_node_id", "parent_document_id", "doc_id"})
_CONNECTOR_START_RE = re.compile(
    r"^(and|or|but|which|that|thereof|herein|aforesaid|the same)\b",
    re.IGNORECASE,
)
_TABLE_STRUCTURE_PRESERVE_VALUES = frozenset(
    {"preserve", "preserved", "structure", "structured", "markdown", "html"}
)
_HIERARCHY_PRESERVE_VALUES = frozenset(
    {"preserve", "preserved", "section_path", "hierarchical", "markdown_header", "headers"}
)
_SENTENCE_BOUNDARY_PRESERVE_VALUES = frozenset(
    {"preserve", "preserved", "strict", "sentence", "sentence_boundary", "sentence_boundaries"}
)


@dataclass(frozen=True)
class ProfileLintIssue:
    """One profile consistency issue."""

    code: str
    message: str
    severity: str
    expected: Any | None = None
    observed: Any | None = None


@dataclass(frozen=True)
class ProfileLintReport:
    """Profile lint report emitted before parser diagnostics."""

    issues: tuple[ProfileLintIssue, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> tuple[ProfileLintIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ProfileLintIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    @property
    def authority_blocked(self) -> bool:
        return self.has_errors


class ProfileLintEngine:
    """Checks whether a declared parser-validation profile matches chunk metadata."""

    def __init__(
        self,
        min_metadata_coverage: float = 0.80,
        min_provenance_coverage: float = 0.90,
    ) -> None:
        self.min_metadata_coverage = min_metadata_coverage
        self.min_provenance_coverage = min_provenance_coverage

    def lint(
        self,
        chunks: Sequence[ChunkIR],
        chunking_profile: ChunkingStrategyProfile,
        validation_profile: ParserValidationProfile,
    ) -> ProfileLintReport:
        if not chunks:
            return ProfileLintReport()

        issues: list[ProfileLintIssue] = []
        migration_mode = validation_profile.infer_from_legacy

        if chunking_profile.requires_page_metadata:
            page_coverage = self._coverage(chunks, lambda chunk: chunk.page_start is not None)
            if page_coverage < self.min_metadata_coverage:
                issues.append(
                    self._issue(
                        code="required_page_metadata_missing",
                        severity="warning" if migration_mode else "error",
                        message="Profile requires page metadata, but normalized chunks do not meet the expected coverage.",
                        expected={">=": self.min_metadata_coverage},
                        observed={"page_metadata_coverage": round(page_coverage, 3)},
                    )
                )

        if chunking_profile.requires_provenance or self._preserves_table_structure(chunking_profile):
            provenance_coverage = self._coverage(
                chunks,
                lambda chunk: bool(chunk.source_element_ids or chunk.source_table_ids),
            )
            if provenance_coverage < self.min_provenance_coverage:
                issues.append(
                    self._issue(
                        code="required_provenance_missing",
                        severity="warning" if migration_mode else "error",
                        message="Profile requires provenance, but normalized chunks do not meet the expected coverage.",
                        expected={">=": self.min_provenance_coverage},
                        observed={"provenance_coverage": round(provenance_coverage, 3)},
                    )
                )

        if self._preserves_section_path(chunking_profile):
            section_coverage = self._coverage(
                chunks,
                lambda chunk: bool(chunk.section_path),
            )
            if section_coverage < self.min_metadata_coverage:
                issues.append(
                    self._issue(
                        code="required_section_path_missing",
                        severity="warning" if migration_mode else "error",
                        message="Profile requires section hierarchy metadata, but normalized chunks lack section_path coverage.",
                        expected={">=": self.min_metadata_coverage},
                        observed={"section_path_coverage": round(section_coverage, 3)},
                    )
                )

        if chunking_profile.requires_parent_id:
            parent_coverage = self._coverage(
                chunks,
                lambda chunk: any(key in chunk.metadata for key in _PARENT_ID_KEYS),
            )
            if parent_coverage < self.min_metadata_coverage:
                issues.append(
                    self._issue(
                        code="required_parent_metadata_missing",
                        severity="warning" if migration_mode else "error",
                        message="Profile requires parent-child metadata, but normalized chunks lack parent_id-like fields.",
                        expected={">=": self.min_metadata_coverage},
                        observed={"parent_metadata_coverage": round(parent_coverage, 3)},
                    )
                )

        issues.extend(self._lint_boundary_flag_plausibility(chunks, chunking_profile))

        return ProfileLintReport(issues=tuple(issues))

    def _lint_boundary_flag_plausibility(
        self,
        chunks: Sequence[ChunkIR],
        profile: ChunkingStrategyProfile,
    ) -> list[ProfileLintIssue]:
        if self._allows_mid_sentence_start(profile):
            return []

        connector_chunks = [
            chunk for chunk in chunks if _CONNECTOR_START_RE.match(chunk.text.strip())
        ]
        if not connector_chunks:
            return []

        chunks_with_boundary_flag = [
            chunk for chunk in chunks
            if chunk.metadata.get("starts_mid_sentence")
            or chunk.metadata.get("split_inside_sentence")
        ]
        connector_ratio = len(connector_chunks) / len(chunks)

        if connector_ratio <= 0.25 or chunks_with_boundary_flag:
            return []

        return [
            self._issue(
                code="boundary_flags_may_be_missing",
                severity="warning",
                message=(
                    "Heuristic sample found connector-start chunks under a sentence-boundary "
                    "preserving profile, but no sentence-boundary flags are present."
                ),
                expected={"boundary_flags": ["starts_mid_sentence", "split_inside_sentence"]},
                observed={
                    "connector_start_ratio": round(connector_ratio, 3),
                    "sample_chunk_ids": [chunk.chunk_id for chunk in connector_chunks[:10]],
                },
            )
        ]

    def _issue(
        self,
        code: str,
        severity: str,
        message: str,
        expected: Any | None = None,
        observed: Any | None = None,
    ) -> ProfileLintIssue:
        return ProfileLintIssue(
            code=code,
            severity=severity,
            message=message,
            expected=expected,
            observed=observed,
        )

    def _coverage(
        self,
        chunks: Sequence[ChunkIR],
        predicate: Any,
    ) -> float:
        return sum(1 for chunk in chunks if predicate(chunk)) / len(chunks)

    def _preserves_table_structure(self, profile: ChunkingStrategyProfile) -> bool:
        return profile.preserves_table_structure or self._profile_value(
            profile.table_structure
        ) in _TABLE_STRUCTURE_PRESERVE_VALUES

    def _preserves_section_path(self, profile: ChunkingStrategyProfile) -> bool:
        return profile.preserves_section_path or self._profile_value(
            profile.hierarchy_mode
        ) in _HIERARCHY_PRESERVE_VALUES

    def _allows_mid_sentence_start(self, profile: ChunkingStrategyProfile) -> bool:
        value = self._profile_value(profile.sentence_boundaries)
        if value in _SENTENCE_BOUNDARY_PRESERVE_VALUES:
            return False
        return profile.allows_mid_sentence_start

    def _profile_value(self, value: str | None) -> str:
        return (value or "").strip().lower().replace("-", "_")
