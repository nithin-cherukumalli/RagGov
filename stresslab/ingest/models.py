"""Pydantic models for parsed ingest artifacts."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ParsedMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str
    title: str
    abstract: str | None = None
    department: str | None = None
    go_number: str | None = None
    issued_date: date | None = None
    references: list[str] = Field(default_factory=list)
    distribution: list[str] = Field(default_factory=list)


class ParsedNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    label: str
    text: str
    page_start: int = Field(gt=0)
    page_end: int = Field(gt=0)
    parent_node_id: str | None = None
    section_path: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_page_range(self) -> "ParsedNode":
        if self.page_start > self.page_end:
            raise ValueError("page_start must be less than or equal to page_end")
        return self


class ParsedTable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_id: str
    page: int = Field(gt=0)
    title: str | None = None
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    source_path: str
    title: str
    abstract: str | None = None
    department: str | None = None
    go_number: str | None = None
    issued_date: date | None = None
    references: list[str] = Field(default_factory=list)
    nodes: list[ParsedNode] = Field(default_factory=list)
    tables: list[ParsedTable] = Field(default_factory=list)
    distribution: list[str] = Field(default_factory=list)
    metadata: ParsedMetadata | None = None

    @model_validator(mode="after")
    def validate_metadata_consistency(self) -> "ParsedDocument":
        if self.metadata is None:
            return self

        overlapping_fields = (
            "source_path",
            "title",
            "abstract",
            "department",
            "go_number",
            "issued_date",
            "references",
            "distribution",
        )
        conflicts = [
            field_name
            for field_name in overlapping_fields
            if getattr(self, field_name) != getattr(self.metadata, field_name)
        ]
        if conflicts:
            conflict_list = ", ".join(conflicts)
            raise ValueError(f"metadata conflicts with document fields: {conflict_list}")
        return self
