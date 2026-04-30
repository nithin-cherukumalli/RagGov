from __future__ import annotations

import re
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


CHUNK_BOUNDARY_REMEDIATION = (
    "Use structure-aware chunking. Avoid splitting inside table rows, list items, legal clauses, "
    "sentences, and section boundaries. Preserve parent heading/table context."
)

_BOUNDARY_FLAGS = (
    "crosses_section_boundary",
    "ends_mid_table_row",
    "ends_mid_list_item",
    "starts_mid_sentence",
    "split_inside_sentence",
    "split_inside_table",
    "split_inside_list",
)

_SENTENCE_BOUNDARY_PRESERVE_VALUES = frozenset(
    {"preserve", "preserved", "strict", "sentence", "sentence_boundary", "sentence_boundaries"}
)
_SENTENCE_BOUNDARY_ALLOW_VALUES = frozenset(
    {"allow", "allowed", "permissive", "none", "fixed_token"}
)
_TABLE_STRUCTURE_PRESERVE_VALUES = frozenset(
    {"preserve", "preserved", "structure", "structured", "markdown", "html"}
)
_HIERARCHY_PRESERVE_VALUES = frozenset(
    {"preserve", "preserved", "section_path", "hierarchical", "markdown_header", "headers"}
)


class ChunkBoundaryValidator:
    """
    Validate whether chunk boundaries preserve structural units.

    Strong mode:
    - Uses explicit metadata flags such as crosses_section_boundary,
      ends_mid_table_row, ends_mid_list_item, starts_mid_sentence.
    - Flags are filtered by ChunkingStrategyProfile before counting as damage.

    Fallback mode:
    - Text-only connector/lowercase smoke tests run only when the strategy
      disallows mid-sentence starts (profile.allows_mid_sentence_start=False).
    - Fallback findings are is_heuristic=True.
    """

    name = "chunk_boundary_validator"

    _bad_start_re = re.compile(
        r"^(and|or|but|which|that|thereof|herein|aforesaid|the same)\b",
        re.IGNORECASE,
    )

    def validate(
        self,
        parsed_doc: ParsedDocumentIR | None,
        chunks: Sequence[ChunkIR],
        config: ParserValidationConfig,
    ) -> list[ParserFinding]:
        if not chunks:
            return []

        profile = config.chunking_profile

        metadata_damaged = [
            chunk for chunk in chunks if self._metadata_says_boundary_damaged(chunk, profile)
        ]

        # Connector/lowercase heuristic is only meaningful when the strategy
        # explicitly forbids mid-sentence boundaries.
        if not self._allows_mid_sentence_start(profile) and config.enable_text_fallback_heuristics:
            heuristic_damaged = [
                chunk
                for chunk in chunks
                if not self._metadata_says_boundary_damaged(chunk, profile)
                and self._text_suggests_boundary_damaged(chunk)
            ]
        else:
            heuristic_damaged = []

        findings: list[ParserFinding] = []

        if metadata_damaged:
            findings.append(
                self._build_metadata_finding(metadata_damaged, len(chunks))
            )

        heuristic_ratio = len(heuristic_damaged) / len(chunks)

        if heuristic_ratio > config.max_chunk_boundary_damage_ratio:
            findings.append(
                self._build_heuristic_finding(
                    heuristic_damaged,
                    len(chunks),
                    heuristic_ratio,
                )
            )

        return findings

    def _metadata_says_boundary_damaged(
        self,
        chunk: ChunkIR,
        profile: ChunkingStrategyProfile,
    ) -> bool:
        metadata = chunk.metadata or {}

        for flag in _BOUNDARY_FLAGS:
            if not metadata.get(flag):
                continue

            # Cross-section splits are acceptable for strategies that permit them.
            if flag == "crosses_section_boundary" and self._allows_cross_section_chunk(profile):
                continue

            # Mid-sentence splits are acceptable for strategies that permit them.
            if flag in ("starts_mid_sentence", "split_inside_sentence") and self._allows_mid_sentence_start(profile):
                continue

            # Table-internal splits are irrelevant unless the strategy enforces table structure.
            if flag in ("split_inside_table", "ends_mid_table_row") and not self._enforces_table_structure(profile):
                continue

            return True

        return False

    def _text_suggests_boundary_damaged(self, chunk: ChunkIR) -> bool:
        text = chunk.text.strip()

        if not text:
            return False

        starts_with_connector = bool(self._bad_start_re.match(text))
        starts_lowercase_short = text[0].islower() and len(text.split()) < 80

        # Avoid flagging ordinary lowercase code-like or bullet-like fragments too aggressively.
        starts_lowercase_short = starts_lowercase_short and not text.startswith(("-", "*", "•"))

        return starts_with_connector or starts_lowercase_short

    def _build_metadata_finding(
        self,
        damaged_chunks: Sequence[ChunkIR],
        total_chunks: int,
    ) -> ParserFinding:
        damaged_flags = self._collect_boundary_flags(damaged_chunks)

        return ParserFinding(
            failure_type=ParserFailureType.CHUNK_BOUNDARY_DAMAGE,
            severity=ParserSeverity.WARN,
            confidence=0.80,
            validator_name=self.name,
            evidence=(
                ParserEvidence(
                    message="Chunk metadata explicitly indicates boundary damage.",
                    observed={
                        "damaged_chunk_count": len(damaged_chunks),
                        "total_chunks": total_chunks,
                        "sample_chunk_ids": [chunk.chunk_id for chunk in damaged_chunks[:10]],
                        "flags": damaged_flags,
                    },
                ),
            ),
            remediation=CHUNK_BOUNDARY_REMEDIATION,
        )

    def _build_heuristic_finding(
        self,
        damaged_chunks: Sequence[ChunkIR],
        total_chunks: int,
        damage_ratio: float,
    ) -> ParserFinding:
        return ParserFinding(
            failure_type=ParserFailureType.CHUNK_BOUNDARY_DAMAGE,
            severity=ParserSeverity.WARN,
            confidence=min(0.90, 0.50 + damage_ratio),
            validator_name=self.name,
            evidence=(
                ParserEvidence(
                    message=(
                        "Text-only smoke test: many chunks appear to start mid-sentence "
                        "or with continuation connectors."
                    ),
                    observed={
                        "heuristic_damage_ratio": round(damage_ratio, 3),
                        "damaged_chunk_count": len(damaged_chunks),
                        "total_chunks": total_chunks,
                        "sample_chunk_ids": [chunk.chunk_id for chunk in damaged_chunks[:10]],
                    },
                ),
            ),
            remediation=CHUNK_BOUNDARY_REMEDIATION,
            alternative_explanations=(
                "Some legal or administrative documents naturally contain short fragments.",
                "Some valid bullet/list fragments may begin with lowercase words.",
            ),
            is_heuristic=True,
        )

    def _collect_boundary_flags(
        self,
        chunks: Sequence[ChunkIR],
    ) -> dict[str, int]:
        flag_counts: dict[str, int] = {}

        for chunk in chunks:
            for flag in _BOUNDARY_FLAGS:
                if bool((chunk.metadata or {}).get(flag)):
                    flag_counts[flag] = flag_counts.get(flag, 0) + 1

        return flag_counts

    def _allows_mid_sentence_start(self, profile: ChunkingStrategyProfile) -> bool:
        value = self._profile_value(profile.sentence_boundaries)
        if value in _SENTENCE_BOUNDARY_PRESERVE_VALUES:
            return False
        if value in _SENTENCE_BOUNDARY_ALLOW_VALUES:
            return True
        return profile.allows_mid_sentence_start

    def _allows_cross_section_chunk(self, profile: ChunkingStrategyProfile) -> bool:
        value = self._profile_value(profile.hierarchy_mode)
        if value in _HIERARCHY_PRESERVE_VALUES:
            return False
        return profile.allows_cross_section_chunk

    def _enforces_table_structure(self, profile: ChunkingStrategyProfile) -> bool:
        value = self._profile_value(profile.table_structure)
        return profile.preserves_table_structure or value in _TABLE_STRUCTURE_PRESERVE_VALUES

    def _profile_value(self, value: str | None) -> str:
        return (value or "").strip().lower().replace("-", "_")
