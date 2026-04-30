"""Parser-agnostic intermediate representation and validation findings."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict


class ParserSeverity(Enum):
    """Severity assigned to a parser validation finding."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class ParserFailureType(Enum):
    """Parser-stage failure categories."""

    TABLE_STRUCTURE_LOSS = "TABLE_STRUCTURE_LOSS"
    HIERARCHY_FLATTENING = "HIERARCHY_FLATTENING"
    METADATA_LOSS = "METADATA_LOSS"
    OCR_DEGRADATION = "OCR_DEGRADATION"
    CHUNK_BOUNDARY_DAMAGE = "CHUNK_BOUNDARY_DAMAGE"
    PROVENANCE_MISSING = "PROVENANCE_MISSING"


class ChunkingStrategyType(str, Enum):
    """Declared chunking strategies understood by the parser validation layer."""

    UNKNOWN = "unknown"
    FIXED_TOKEN = "fixed_token"
    SENTENCE = "sentence"
    RECURSIVE_TEXT = "recursive_text"
    HIERARCHICAL = "hierarchical"
    MARKDOWN_HEADER = "markdown_header"
    SEMANTIC = "semantic"
    TABLE_AWARE = "table_aware"
    PARENT_CHILD = "parent_child"
    SUMMARY = "summary"
    LATE_CHUNKING = "late_chunking"


class ChunkingStrategyProfile(BaseModel):
    """
    Structural guarantees declared by a chunking strategy.

    Validators consult this profile to decide which checks apply.
    All fields default to the permissive UNKNOWN baseline so that
    undeclared strategies produce minimal false positives.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy_type: ChunkingStrategyType = ChunkingStrategyType.UNKNOWN
    preserves_section_path: bool = False
    preserves_table_structure: bool = False
    preserves_table_headers: bool = False
    preserves_source_elements: bool = False
    allows_mid_sentence_start: bool = True
    allows_cross_section_chunk: bool = True
    allows_table_flattening: bool = True
    requires_page_metadata: bool = False
    requires_provenance: bool = False
    requires_parent_id: bool = False
    table_check_suppressed: bool = False
    sentence_boundaries: str | None = None
    table_structure: str | None = None
    hierarchy_mode: str | None = None


def default_chunking_profile(strategy_type: ChunkingStrategyType) -> ChunkingStrategyProfile:
    """Return v1 structural-guarantee defaults for a given chunking strategy type."""

    if strategy_type == ChunkingStrategyType.FIXED_TOKEN:
        return ChunkingStrategyProfile(
            strategy_type=strategy_type,
            allows_mid_sentence_start=True,
            allows_cross_section_chunk=True,
            allows_table_flattening=True,
        )

    if strategy_type == ChunkingStrategyType.RECURSIVE_TEXT:
        return ChunkingStrategyProfile(
            strategy_type=strategy_type,
            allows_mid_sentence_start=True,
            allows_cross_section_chunk=True,
            allows_table_flattening=True,
        )

    if strategy_type == ChunkingStrategyType.SENTENCE:
        return ChunkingStrategyProfile(
            strategy_type=strategy_type,
            allows_mid_sentence_start=False,
            allows_cross_section_chunk=True,
            allows_table_flattening=True,
        )

    if strategy_type == ChunkingStrategyType.HIERARCHICAL:
        return ChunkingStrategyProfile(
            strategy_type=strategy_type,
            preserves_section_path=True,
            preserves_table_structure=False,
            preserves_table_headers=False,
            preserves_source_elements=True,
            allows_mid_sentence_start=False,
            allows_cross_section_chunk=False,
            allows_table_flattening=True,
            requires_page_metadata=True,
            requires_provenance=True,
        )

    if strategy_type == ChunkingStrategyType.MARKDOWN_HEADER:
        return ChunkingStrategyProfile(
            strategy_type=strategy_type,
            preserves_section_path=True,
            preserves_table_structure=False,
            preserves_table_headers=False,
            preserves_source_elements=True,
            allows_mid_sentence_start=False,
            allows_cross_section_chunk=False,
            allows_table_flattening=True,
            requires_page_metadata=True,
            requires_provenance=True,
        )

    if strategy_type == ChunkingStrategyType.SEMANTIC:
        return ChunkingStrategyProfile(
            strategy_type=strategy_type,
            preserves_source_elements=True,
            allows_mid_sentence_start=False,
            allows_cross_section_chunk=True,
            allows_table_flattening=True,
            requires_page_metadata=True,
            requires_provenance=True,
        )

    if strategy_type == ChunkingStrategyType.TABLE_AWARE:
        return ChunkingStrategyProfile(
            strategy_type=strategy_type,
            preserves_table_structure=True,
            preserves_table_headers=True,
            preserves_source_elements=True,
            allows_mid_sentence_start=False,
            allows_cross_section_chunk=True,
            allows_table_flattening=False,
            requires_page_metadata=True,
            requires_provenance=True,
        )

    if strategy_type == ChunkingStrategyType.PARENT_CHILD:
        return ChunkingStrategyProfile(
            strategy_type=strategy_type,
            preserves_source_elements=True,
            allows_mid_sentence_start=True,
            allows_cross_section_chunk=True,
            allows_table_flattening=True,
            requires_page_metadata=True,
            requires_provenance=True,
            requires_parent_id=True,
        )

    if strategy_type == ChunkingStrategyType.SUMMARY:
        return ChunkingStrategyProfile(
            strategy_type=strategy_type,
            allows_mid_sentence_start=True,
            allows_cross_section_chunk=True,
            allows_table_flattening=True,
            requires_page_metadata=True,
            requires_provenance=True,
            table_check_suppressed=True,
        )

    if strategy_type == ChunkingStrategyType.LATE_CHUNKING:
        return ChunkingStrategyProfile(
            strategy_type=strategy_type,
            preserves_source_elements=True,
            allows_mid_sentence_start=True,
            allows_cross_section_chunk=True,
            allows_table_flattening=True,
            requires_page_metadata=True,
            requires_provenance=True,
        )

    # UNKNOWN: permissive baseline — no structural guarantees declared
    return ChunkingStrategyProfile(strategy_type=strategy_type)


@dataclass(frozen=True)
class LayoutBox:
    page: int | None
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class TableIR:
    table_id: str
    page_start: int | None = None
    page_end: int | None = None
    n_rows: int | None = None
    n_cols: int | None = None
    headers: tuple[str, ...] = ()
    text: str = ""
    html: str | None = None
    markdown: str | None = None
    parser_confidence: float | None = None
    parsing_report: dict[str, Any] = field(default_factory=dict)
    layout_boxes: tuple[LayoutBox, ...] = ()


@dataclass(frozen=True)
class ElementIR:
    element_id: str
    element_type: str
    text: str
    page: int | None = None
    section_path: tuple[str, ...] = ()
    layout_box: LayoutBox | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedDocumentIR:
    document_id: str
    parser_name: str | None = None
    parser_version: str | None = None
    elements: tuple[ElementIR, ...] = ()
    tables: tuple[TableIR, ...] = ()
    document_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_structural_metadata(self) -> bool:
        return bool(
            self.elements
            or self.tables
            or self.document_metadata
            or any(
                element.page is not None
                or element.section_path
                or element.layout_box is not None
                or element.metadata
                for element in self.elements
            )
        )


@dataclass(frozen=True)
class ChunkIR:
    chunk_id: str
    text: str
    source_element_ids: tuple[str, ...] = ()
    source_table_ids: tuple[str, ...] = ()
    page_start: int | None = None
    page_end: int | None = None
    section_path: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParserEvidence:
    message: str
    chunk_id: str | None = None
    element_id: str | None = None
    table_id: str | None = None
    expected: Any | None = None
    observed: Any | None = None


@dataclass(frozen=True)
class ParserFinding:
    failure_type: ParserFailureType
    severity: ParserSeverity
    confidence: float
    evidence: tuple[ParserEvidence, ...]
    remediation: str
    validator_name: str
    alternative_explanations: tuple[str, ...] = ()
    is_heuristic: bool = False


# These are v0/v1 operating thresholds, not research-calibrated constants. They must remain configurable and should be calibrated on a labeled parser-failure dataset.
@dataclass(frozen=True)
class ParserValidationConfig:
    min_table_structure_score: float = 0.70
    min_metadata_coverage: float = 0.80
    min_provenance_coverage: float = 0.90
    max_chunk_boundary_damage_ratio: float = 0.10
    enable_text_fallback_heuristics: bool = True
    chunking_profile: ChunkingStrategyProfile = field(
        default_factory=lambda: default_chunking_profile(ChunkingStrategyType.UNKNOWN)
    )
