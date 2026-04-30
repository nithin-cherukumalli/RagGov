from __future__ import annotations

import re
from collections import defaultdict
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
    TableIR,
)


TABLE_REMEDIATION = (
    "Use a structure-preserving parser before chunking. Preserve table IDs, "
    "row-column bindings, headers, and table serialization as markdown or HTML. "
    "If a table is split, repeat headers in each table chunk."
)

_TABLE_STRUCTURE_PRESERVE_VALUES = frozenset(
    {"preserve", "preserved", "structure", "structured", "markdown", "html"}
)
_TABLE_STRUCTURE_SUPPRESS_VALUES = frozenset(
    {"summary", "summarized", "suppress", "suppressed", "none", "ignore"}
)
_TABLE_STRUCTURE_FLATTEN_VALUES = frozenset(
    {"flatten", "flattened", "prose", "allow_flattening"}
)


class TableStructureValidator:
    """
    Validate whether source tables survived parsing/chunking.

    Strong mode:
    - Uses ParsedDocumentIR.tables and chunk.source_table_ids.
    - Enforced when profile.preserves_table_structure is True (TABLE_AWARE).

    Permissive mode:
    - UNKNOWN, FIXED_TOKEN, HIERARCHICAL etc. allow table flattening.
    - Only flag missing provenance when the profile requires it.

    Fallback mode:
    - Text-only smoke tests run when enable_text_fallback_heuristics is True.
    - Fallback findings are marked is_heuristic=True.
    - Collapsed-header heuristic severity depends on profile.allows_table_flattening.
    """

    name = "table_structure_validator"

    _markdown_table_re = re.compile(r"(?m)^\s*\|.+\|\s*$")
    _html_table_re = re.compile(r"<\s*(table|tr|td|th)\b", re.IGNORECASE)
    _aligned_col_re = re.compile(r"(?m)^\S+(?:\s{2,}\S+){2,}\s*$")

    def validate(
        self,
        parsed_doc: ParsedDocumentIR | None,
        chunks: Sequence[ChunkIR],
        config: ParserValidationConfig,
    ) -> list[ParserFinding]:
        profile = config.chunking_profile

        if self._table_checks_suppressed(profile):
            return []

        findings: list[ParserFinding] = []

        if parsed_doc and parsed_doc.tables:
            findings.extend(self._validate_known_tables(parsed_doc.tables, chunks, config, profile))

        if config.enable_text_fallback_heuristics:
            findings.extend(self._fallback_numeric_table_smoke_test(chunks, profile))

        return findings

    def _validate_known_tables(
        self,
        tables: Sequence[TableIR],
        chunks: Sequence[ChunkIR],
        config: ParserValidationConfig,
        profile: ChunkingStrategyProfile,
    ) -> list[ParserFinding]:
        findings: list[ParserFinding] = []
        chunks_by_table_id = self._index_chunks_by_table_id(chunks)

        for table in tables:
            related_chunks = chunks_by_table_id.get(table.table_id, [])

            if not related_chunks:
                # Only escalate to FAIL when the declared strategy requires provenance or
                # full table preservation. UNKNOWN and simple text splitters skip this.
                if (
                    profile.requires_provenance
                    or self._preserves_table_structure(profile)
                    or profile.preserves_source_elements
                ):
                    findings.append(
                        ParserFinding(
                            failure_type=ParserFailureType.TABLE_STRUCTURE_LOSS,
                            severity=ParserSeverity.FAIL,
                            confidence=0.90,
                            validator_name=self.name,
                            evidence=(
                                ParserEvidence(
                                    message=(
                                        "Source parser reported a table, but no chunk preserves "
                                        "source_table_ids for it."
                                    ),
                                    table_id=table.table_id,
                                    expected={
                                        "table_id": table.table_id,
                                        "rows": table.n_rows,
                                        "cols": table.n_cols,
                                        "headers": table.headers,
                                    },
                                    observed="no chunk references this table_id",
                                ),
                            ),
                            remediation=TABLE_REMEDIATION,
                            alternative_explanations=(
                                "The upstream parser may have falsely classified a non-table region as a table.",
                            ),
                        )
                    )
                continue

            for chunk in related_chunks:
                structure_score = self._chunk_table_structure_score(chunk.text)
                header_coverage = self._header_coverage(table, chunk.text)

                parser_report = table.parsing_report or {}
                parser_accuracy = parser_report.get("accuracy")
                parser_whitespace = parser_report.get("whitespace")
                parser_order = parser_report.get("order")
                parser_page = parser_report.get("page")

                low_structure = structure_score < config.min_table_structure_score
                missing_headers = bool(table.headers) and header_coverage < 0.50

                finding = self._evaluate_chunk_table_quality(
                    chunk=chunk,
                    table=table,
                    profile=profile,
                    low_structure=low_structure,
                    missing_headers=missing_headers,
                    structure_score=structure_score,
                    header_coverage=header_coverage,
                    parser_accuracy=parser_accuracy,
                    parser_whitespace=parser_whitespace,
                    parser_order=parser_order,
                    parser_page=parser_page,
                )
                if finding is not None:
                    findings.append(finding)

        return findings

    def _evaluate_chunk_table_quality(
        self,
        chunk: ChunkIR,
        table: TableIR,
        profile: ChunkingStrategyProfile,
        low_structure: bool,
        missing_headers: bool,
        structure_score: float,
        header_coverage: float,
        parser_accuracy: object,
        parser_whitespace: object,
        parser_order: object,
        parser_page: object,
    ) -> ParserFinding | None:
        preserves_table_structure = self._preserves_table_structure(profile)
        preserves_table_headers = self._preserves_table_headers(profile)
        allows_table_flattening = self._allows_table_flattening(profile)

        if preserves_table_structure:
            check = low_structure or (preserves_table_headers and missing_headers)
            if not check:
                return None
            confidence = min(0.95, max(0.70, 1.0 - structure_score))
            severity = ParserSeverity.FAIL if low_structure else ParserSeverity.WARN

        elif allows_table_flattening:
            if not (preserves_table_headers and bool(table.headers) and header_coverage < 0.50):
                return None
            confidence = min(0.90, max(0.60, 1.0 - header_coverage))
            severity = ParserSeverity.WARN

        else:
            # Not preserves_table_structure, not allows_table_flattening.
            if not (low_structure or (preserves_table_headers and missing_headers)):
                return None
            confidence = min(0.90, max(0.60, 1.0 - structure_score))
            severity = ParserSeverity.WARN

        return ParserFinding(
            failure_type=ParserFailureType.TABLE_STRUCTURE_LOSS,
            severity=severity,
            confidence=confidence,
            validator_name=self.name,
            evidence=(
                ParserEvidence(
                    message=(
                        "Chunk has table provenance, but its text representation appears "
                        "to have weak row/column/header preservation."
                    ),
                    chunk_id=chunk.chunk_id,
                    table_id=table.table_id,
                    expected={
                        "rows": table.n_rows,
                        "cols": table.n_cols,
                        "headers": table.headers,
                        "parser_accuracy": parser_accuracy,
                        "parser_whitespace": parser_whitespace,
                        "parser_order": parser_order,
                        "parser_page": parser_page,
                    },
                    observed={
                        "structure_score": round(structure_score, 3),
                        "header_coverage": round(header_coverage, 3),
                        "has_markdown_table": bool(self._markdown_table_re.search(chunk.text)),
                        "has_html_table": bool(self._html_table_re.search(chunk.text)),
                        "aligned_column_lines": len(self._aligned_col_re.findall(chunk.text)),
                    },
                ),
            ),
            remediation=TABLE_REMEDIATION,
            alternative_explanations=(
                "The table may have been intentionally summarized as prose.",
                "The source table may be too small for row/column structure to be meaningful.",
            ),
        )

    def _fallback_numeric_table_smoke_test(
        self,
        chunks: Sequence[ChunkIR],
        profile: ChunkingStrategyProfile,
    ) -> list[ParserFinding]:
        findings: list[ParserFinding] = []

        for chunk in chunks:
            # If explicit table provenance exists, do not duplicate with fallback heuristic.
            if chunk.source_table_ids:
                continue

            numeric_ratio = self._numeric_token_ratio(chunk.text)
            line_count = len([line for line in chunk.text.splitlines() if line.strip()])
            structure_score = self._chunk_table_structure_score(chunk.text)
            collapsed_header_values = self._looks_like_collapsed_header_value_table(chunk.text)

            if collapsed_header_values and structure_score < 0.30:
                # When the declared strategy forbids table flattening, this is a hard FAIL.
                # For all other strategies (including UNKNOWN), keep it as WARN.
                severity = (
                    ParserSeverity.FAIL
                    if not self._allows_table_flattening(profile)
                    else ParserSeverity.WARN
                )
                findings.append(
                    ParserFinding(
                        failure_type=ParserFailureType.TABLE_STRUCTURE_LOSS,
                        severity=severity,
                        confidence=0.85,
                        validator_name=self.name,
                        evidence=(
                            ParserEvidence(
                                message=(
                                    "Text-only smoke test: chunk looks like a collapsed header/value table "
                                    "without table separators, HTML tags, or aligned column evidence."
                                ),
                                chunk_id=chunk.chunk_id,
                                observed={
                                    "numeric_token_ratio": round(numeric_ratio, 3),
                                    "line_count": line_count,
                                    "structure_score": round(structure_score, 3),
                                },
                            ),
                        ),
                        remediation=(
                            "Attach parser structural metadata to chunks. Text-only table detection is heuristic "
                            "and cannot reliably prove table degradation."
                        ),
                        alternative_explanations=(
                            "The chunk may be compact prose listing several attributes rather than a table.",
                        ),
                        is_heuristic=True,
                    )
                )
                continue

            if numeric_ratio >= 0.30 and line_count >= 4 and structure_score < 0.30:
                findings.append(
                    ParserFinding(
                        failure_type=ParserFailureType.TABLE_STRUCTURE_LOSS,
                        severity=ParserSeverity.WARN,
                        confidence=0.55,
                        validator_name=self.name,
                        evidence=(
                            ParserEvidence(
                                message=(
                                    "Text-only smoke test: chunk has high numeric density and multiple lines "
                                    "but lacks table separators, HTML tags, or aligned column evidence."
                                ),
                                chunk_id=chunk.chunk_id,
                                observed={
                                    "numeric_token_ratio": round(numeric_ratio, 3),
                                    "line_count": line_count,
                                    "structure_score": round(structure_score, 3),
                                },
                            ),
                        ),
                        remediation=(
                            "Attach parser structural metadata to chunks. Text-only table detection is heuristic "
                            "and cannot reliably prove table degradation."
                        ),
                        alternative_explanations=(
                            "The chunk may be numeric prose rather than a degraded table.",
                        ),
                        is_heuristic=True,
                    )
                )

        return findings

    def _chunk_table_structure_score(self, text: str) -> float:
        if self._markdown_table_re.search(text) or self._html_table_re.search(text):
            return 1.0

        signals = [
            bool(self._markdown_table_re.search(text)),
            bool(self._html_table_re.search(text)),
            "\t" in text,
            len(self._aligned_col_re.findall(text)) >= 2,
            self._estimated_row_consistency(text) >= 0.60,
        ]
        return sum(signals) / len(signals)

    def _estimated_row_consistency(self, text: str) -> float:
        rows = [line.strip() for line in text.splitlines() if line.strip()]
        candidate_rows = [row for row in rows if len(row.split()) >= 3]

        if len(candidate_rows) < 3:
            return 0.0

        widths = [len(row.split()) for row in candidate_rows]
        most_common_width = max(set(widths), key=widths.count)
        return widths.count(most_common_width) / len(widths)

    def _header_coverage(self, table: TableIR, chunk_text: str) -> float:
        if not table.headers:
            return 1.0

        lowered = chunk_text.lower()
        hits = sum(1 for header in table.headers if header.lower() in lowered)
        return hits / len(table.headers)

    def _numeric_token_ratio(self, text: str) -> float:
        tokens = re.findall(r"\S+", text)
        if not tokens:
            return 0.0

        numeric = sum(1 for token in tokens if re.search(r"\d", token))
        return numeric / len(tokens)

    def _looks_like_collapsed_header_value_table(self, text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) != 1:
            return False

        tokens = re.findall(r"[A-Za-z][A-Za-z./()-]*|\d+(?:[.,]\d+)?", lines[0])
        if len(tokens) < 8:
            return False

        numeric_count = sum(1 for token in tokens if re.search(r"\d", token))
        alpha_count = sum(1 for token in tokens if re.search(r"[A-Za-z]", token))
        if numeric_count < 2 or alpha_count < 6:
            return False

        leading_tokens = tokens[:3]
        leading_headers = all(
            re.search(r"[A-Za-z]", token) and not re.search(r"\d", token)
            for token in leading_tokens
        )
        leading_header_case = all(token[:1].isupper() for token in leading_tokens)
        tail = tokens[3:]
        mixed_tail = any(re.search(r"\d", token) for token in tail) and any(
            re.search(r"[A-Za-z]", token) for token in tail
        )

        return leading_headers and leading_header_case and mixed_tail

    def _table_checks_suppressed(self, profile: ChunkingStrategyProfile) -> bool:
        value = self._profile_value(profile.table_structure)
        return profile.table_check_suppressed or value in _TABLE_STRUCTURE_SUPPRESS_VALUES

    def _preserves_table_structure(self, profile: ChunkingStrategyProfile) -> bool:
        value = self._profile_value(profile.table_structure)
        return profile.preserves_table_structure or value in _TABLE_STRUCTURE_PRESERVE_VALUES

    def _preserves_table_headers(self, profile: ChunkingStrategyProfile) -> bool:
        value = self._profile_value(profile.table_structure)
        return profile.preserves_table_headers or value in _TABLE_STRUCTURE_PRESERVE_VALUES

    def _allows_table_flattening(self, profile: ChunkingStrategyProfile) -> bool:
        value = self._profile_value(profile.table_structure)
        if value in _TABLE_STRUCTURE_PRESERVE_VALUES:
            return False
        if value in _TABLE_STRUCTURE_FLATTEN_VALUES | _TABLE_STRUCTURE_SUPPRESS_VALUES:
            return True
        return profile.allows_table_flattening

    def _profile_value(self, value: str | None) -> str:
        return (value or "").strip().lower().replace("-", "_")

    def _index_chunks_by_table_id(self, chunks: Sequence[ChunkIR]) -> dict[str, list[ChunkIR]]:
        indexed: dict[str, list[ChunkIR]] = defaultdict(list)

        for chunk in chunks:
            for table_id in chunk.source_table_ids:
                indexed[table_id].append(chunk)

        return dict(indexed)
