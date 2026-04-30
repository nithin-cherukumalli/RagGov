"""Profile schema models for parser-validation configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from raggov.parser_validation.models import (
    ChunkingStrategyProfile,
    ChunkingStrategyType,
    default_chunking_profile,
)


class MetadataFieldMapping(BaseModel):
    """Canonical metadata field mapping declaration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    aliases: tuple[str, ...] = ()
    fallback: tuple[str, ...] = ()
    first_transform: str | None = None


class CanonicalMetadataMapping(BaseModel):
    """Mappings from raw metadata paths to canonical parser-validation fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: MetadataFieldMapping = Field(default_factory=MetadataFieldMapping)
    text: MetadataFieldMapping = Field(default_factory=MetadataFieldMapping)
    document_id: MetadataFieldMapping = Field(default_factory=MetadataFieldMapping)
    page_start: MetadataFieldMapping = Field(default_factory=MetadataFieldMapping)
    page_end: MetadataFieldMapping = Field(default_factory=MetadataFieldMapping)
    source_element_ids: MetadataFieldMapping = Field(default_factory=MetadataFieldMapping)
    source_table_ids: MetadataFieldMapping = Field(default_factory=MetadataFieldMapping)
    section_path: MetadataFieldMapping = Field(default_factory=MetadataFieldMapping)
    parent_id: MetadataFieldMapping = Field(default_factory=MetadataFieldMapping)
    chunking_strategy: MetadataFieldMapping = Field(default_factory=MetadataFieldMapping)
    boundary_flags: dict[str, MetadataFieldMapping] = Field(default_factory=dict)
    domain_fields: dict[str, MetadataFieldMapping] = Field(default_factory=dict)


class ParserProfile(BaseModel):
    """Parser identity and declared parser capabilities."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str | None = None
    version: str | None = None
    preserves_layout: bool = False
    emits_elements: bool = False
    emits_tables: bool = False
    emits_table_reports: bool = False
    emits_provenance: bool = False
    supports_ocr: bool = False


class ParserValidationProfile(BaseModel):
    """Named parser-validation profile for one parser/chunker pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parser: ParserProfile = Field(default_factory=ParserProfile)
    chunking_strategy: ChunkingStrategyProfile = Field(
        default_factory=lambda: default_chunking_profile(ChunkingStrategyType.UNKNOWN)
    )
    metadata_mapping: CanonicalMetadataMapping = Field(
        default_factory=CanonicalMetadataMapping
    )
    infer_from_legacy: bool = False
    description: str | None = None


class ParserValidationProfileSet(BaseModel):
    """Container for multiple named parser-validation profiles."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    profiles: dict[str, ParserValidationProfile]

    @classmethod
    def from_yaml(cls, text: str) -> ParserValidationProfileSet:
        """Deserialize a profile set from YAML text."""
        data = _load_yaml(text)
        return cls.model_validate(data)

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> ParserValidationProfileSet:
        """Deserialize a profile set from a YAML file."""
        return cls.from_yaml(Path(path).read_text(encoding="utf-8"))

    def to_yaml(self) -> str:
        """Serialize a profile set to YAML text."""
        return _dump_yaml(self.model_dump(mode="json", exclude_none=True))

    def to_yaml_file(self, path: str | Path) -> None:
        """Serialize a profile set to a YAML file."""
        Path(path).write_text(self.to_yaml(), encoding="utf-8")


def _load_yaml(text: str) -> Any:
    try:
        import yaml
    except ImportError:
        return json.loads(text)

    return yaml.safe_load(text)


def _dump_yaml(data: Any) -> str:
    try:
        import yaml
    except ImportError:
        return json.dumps(data, indent=2, sort_keys=True)

    return yaml.safe_dump(data, sort_keys=True)
