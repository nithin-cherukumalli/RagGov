"""Analyzer for detecting parser-stage structure degradation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.run import RAGRun


TABLE_REMEDIATION = (
    "Use a structure-preserving parser (unstructured.io, docling, pymupdf4llm) "
    "before chunking. Tables must preserve row-column bindings."
)
HIERARCHY_REMEDIATION = (
    "Parser lost document hierarchy. Use heading-aware parsing. Preserve numbered "
    "list structure in chunk boundaries."
)
METADATA_REMEDIATION = (
    "Document identifiers and section labels not found in chunks. Verify parser "
    "preserves metadata. Add metadata injection at chunk level."
)

TABLE_KEYWORDS = (
    "district",
    "category",
    "grade",
    "total",
    "vacancies",
    "sl.no",
    "s.no",
    "sr.no",
    "name",
    "designation",
)
STRONG_TABLE_KEYWORDS = {"vacancies", "sl.no", "s.no", "sr.no", "designation"}
QUERY_METADATA_TERMS = ("order", "rule", "section", "chapter", "g.o.")
ORPHAN_PREFIXES = (
    "and ",
    "or ",
    "but ",
    "which ",
    "that ",
    "thereof",
    "herein",
    "aforesaid",
    "the same",
)
IDENTIFIER_PATTERNS = (
    r"\bG\.O\.\b",
    r"\bG\.O\.Ms\.\b",
    r"\bG\.O\.Rt\.\b",
    r"\bOrder\s+No\.\b",
)
SECTION_LABEL_PATTERNS = (
    r"(?m)^\s*PART\b",
    r"(?m)^\s*CHAPTER\b",
    r"(?m)^\s*SECTION\b",
    r"(?m)^\s*ANNEXURE\b",
)
DATE_NEAR_START_PATTERN = (
    r"(?im)^(?:.{0,80})\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}|[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})\b"
)
DIRECT_HIERARCHY_PATTERNS = (
    r"\(\d+\)[^.]+\(\d+\)",
    r"Rule\s+\d+[^.]*Rule\s+\d+",
    r"\d+\.\s+\w+[^.]*\d+\.\s+\w+",
)
TABLE_ROW_PATTERN = re.compile(
    r"(district|category|grade|total)\s+\w+\s+\w+\s+\w+",
    re.IGNORECASE,
)
TABLE_HEADER_VALUE_PATTERN = re.compile(
    r"\b(?:district|category|grade|total|vacancies|sl\.?no|s\.?no|sr\.?no|name|designation)\b"
    r"(?:\s+[A-Za-z0-9./()-]+){3,}",
    re.IGNORECASE,
)
MARKDOWN_TABLE_PATTERN = re.compile(r"(?m)^\s*\|.+\|\s*$")
HTML_TABLE_PATTERN = re.compile(r"<(?:td|tr|th)\b", re.IGNORECASE)
ALIGNED_COLUMNS_PATTERN = re.compile(r"(?m)^\S+(?:\s{2,}\S+){1,}\s*$")
TOKEN_PATTERN = re.compile(r"\b[a-z][a-z0-9.]*\b", re.IGNORECASE)


@dataclass(frozen=True)
class Finding:
    """Structured parser finding before collapsing to AnalyzerResult."""

    failure_type: FailureType
    status: str
    evidence: list[str]
    remediation: str
    priority: int


class ParserValidationAnalyzer(BaseAnalyzer):
    """Detect parser-stage structural failures directly from chunk text."""

    weight = 0.85

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        if not run.retrieved_chunks:
            return self.skip("no retrieved chunks available")

        findings: list[Finding] = []

        table_finding = self._check_table_degradation(run)
        if table_finding is not None:
            findings.append(table_finding)

        hierarchy_finding = self._check_hierarchy_flattening(run)
        if hierarchy_finding is not None:
            findings.append(hierarchy_finding)

        metadata_finding = self._check_metadata_loss(run)
        if metadata_finding is not None:
            findings.append(metadata_finding)

        if not findings:
            return self._pass()

        primary = min(findings, key=lambda finding: finding.priority)
        evidence = list(primary.evidence)
        for finding in findings:
            if finding is primary:
                continue
            evidence.extend(finding.evidence)

        return AnalyzerResult(
            analyzer_name=self.name(),
            status=primary.status,
            failure_type=primary.failure_type,
            stage=FailureStage.PARSING,
            evidence=evidence,
            remediation=primary.remediation,
        )

    def _check_table_degradation(self, run: RAGRun) -> Finding | None:
        threshold = float(self.config.get("table_score_threshold", 0.2))

        for chunk in run.retrieved_chunks:
            keywords = self._table_keywords(chunk.text)
            has_table_intent = len(keywords) >= 2 or bool(keywords & STRONG_TABLE_KEYWORDS)
            if not has_table_intent:
                continue

            structure_score = self._table_structure_score(chunk.text)
            collapsed_row = self._has_collapsed_table_row(chunk.text)

            if structure_score < threshold and collapsed_row:
                return Finding(
                    failure_type=FailureType.TABLE_STRUCTURE_LOSS,
                    status="fail",
                    evidence=[
                        f"Table keywords detected but structural separators absent in chunk {chunk.chunk_id}"
                    ],
                    remediation=TABLE_REMEDIATION,
                    priority=0,
                )

        return None

    def _check_hierarchy_flattening(self, run: RAGRun) -> Finding | None:
        min_signals = int(self.config.get("min_hierarchy_signals", 2))
        orphaned_chunk_ids: list[str] = []

        for chunk in run.retrieved_chunks:
            signal_count = self._direct_hierarchy_signal_count(chunk.text)
            if signal_count >= min_signals:
                return Finding(
                    failure_type=FailureType.HIERARCHY_FLATTENING,
                    status="fail",
                    evidence=[
                        f"Hierarchy markers merged inline in chunk {chunk.chunk_id}"
                    ],
                    remediation=HIERARCHY_REMEDIATION,
                    priority=1,
                )

            if self._is_orphaned_fragment(chunk.text):
                orphaned_chunk_ids.append(chunk.chunk_id)

        if len(orphaned_chunk_ids) >= 2:
            joined_ids = ", ".join(orphaned_chunk_ids)
            return Finding(
                failure_type=FailureType.HIERARCHY_FLATTENING,
                status="warn",
                evidence=[f"Multiple orphaned fragments detected in chunks {joined_ids}"],
                remediation=HIERARCHY_REMEDIATION,
                priority=1,
            )

        return None

    def _check_metadata_loss(self, run: RAGRun) -> Finding | None:
        query_lower = run.query.lower()
        if not any(term in query_lower for term in QUERY_METADATA_TERMS):
            return None

        combined_text = "\n".join(chunk.text for chunk in run.retrieved_chunks)

        has_identifier = any(
            re.search(pattern, combined_text, flags=re.IGNORECASE)
            for pattern in IDENTIFIER_PATTERNS
        )
        has_section_label = any(
            re.search(pattern, combined_text, flags=re.IGNORECASE)
            for pattern in SECTION_LABEL_PATTERNS
        )
        has_date_near_start = any(
            re.search(DATE_NEAR_START_PATTERN, chunk.text)
            for chunk in run.retrieved_chunks
        )

        if has_identifier or has_section_label or has_date_near_start:
            return None

        return Finding(
            failure_type=FailureType.METADATA_LOSS,
            status="warn",
            evidence=["Document identifiers and section labels not found in retrieved chunks"],
            remediation=METADATA_REMEDIATION,
            priority=2,
        )

    def _table_keywords(self, text: str) -> set[str]:
        tokens = {token.lower() for token in TOKEN_PATTERN.findall(text)}
        return {keyword for keyword in TABLE_KEYWORDS if keyword in tokens}

    def _table_structure_score(self, text: str) -> float:
        lines = [line for line in text.splitlines() if line.strip()]
        signals = [
            "|" in text,
            "\t" in text,
            bool(MARKDOWN_TABLE_PATTERN.search(text)),
            bool(HTML_TABLE_PATTERN.search(text)),
            len([line for line in lines if ALIGNED_COLUMNS_PATTERN.match(line)]) >= 2,
        ]
        return sum(signals) / len(signals)

    def _has_collapsed_table_row(self, text: str) -> bool:
        single_line_text = " ".join(line.strip() for line in text.splitlines() if line.strip())
        return bool(
            TABLE_ROW_PATTERN.search(single_line_text)
            or TABLE_HEADER_VALUE_PATTERN.search(single_line_text)
        )

    def _is_orphaned_fragment(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False

        orphan_signals = 0
        lowered = stripped.lower()

        if lowered.startswith(ORPHAN_PREFIXES):
            orphan_signals += 1

        if re.match(r"^\d+\.\s", stripped):
            orphan_signals += 1

        starts_mid_sentence = stripped[0].islower()
        if len(stripped.split()) < 50 and starts_mid_sentence:
            orphan_signals += 1

        return orphan_signals >= 1

    def _direct_hierarchy_signal_count(self, text: str) -> int:
        """Count direct inline hierarchy degradation signals in a chunk."""
        signal_count = 0

        for pattern in DIRECT_HIERARCHY_PATTERNS:
            signal_count += len(re.findall(pattern, text, flags=re.IGNORECASE))

        # Multiple numbered clauses on one line is an additional strong signal.
        for line in text.splitlines():
            if len(re.findall(r"\(\d+\)", line)) >= 2:
                signal_count += 1

        return signal_count
