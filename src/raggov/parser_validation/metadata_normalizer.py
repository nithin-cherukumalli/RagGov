"""Normalize raw parser/chunker metadata into canonical parser-validation fields."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from raggov.parser_validation.profile import (
    CanonicalMetadataMapping,
    MetadataFieldMapping,
)


class NormalizedMetadata(BaseModel):
    """Canonical metadata plus preserved domain and unmapped fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str | None = None
    text: str | None = None
    document_id: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    source_element_ids: tuple[str, ...] = ()
    source_table_ids: tuple[str, ...] = ()
    section_path: tuple[str, ...] = ()
    parent_id: str | None = None
    chunking_strategy: str | None = None
    boundary_flags: dict[str, bool] = Field(default_factory=dict)
    domain_fields: dict[str, Any] = Field(default_factory=dict)
    unmapped: dict[str, Any] = Field(default_factory=dict)


class MetadataNormalizer:
    """Apply a canonical metadata mapping to one raw metadata dictionary."""

    def __init__(self, mapping: CanonicalMetadataMapping) -> None:
        self.mapping = mapping

    def normalize(self, raw: dict[str, Any]) -> NormalizedMetadata:
        raw_copy = deepcopy(raw)
        consumed_paths: list[tuple[str, ...]] = []

        values: dict[str, Any] = {}
        for field_name in _CANONICAL_FIELDS:
            mapping = getattr(self.mapping, field_name)
            found = self._resolve_mapping(raw, mapping)
            if found is None:
                continue

            path, value = found
            consumed_paths.append(path)
            values[field_name] = self._coerce_canonical_value(
                field_name,
                self._apply_transform(value, mapping.first_transform),
            )

        boundary_flags: dict[str, bool] = {}
        for flag_name, mapping in self.mapping.boundary_flags.items():
            found = self._resolve_mapping(raw, mapping)
            if found is None:
                continue

            path, value = found
            consumed_paths.append(path)
            boundary_flags[flag_name] = self._coerce_bool(
                self._apply_transform(value, mapping.first_transform)
            )

        domain_fields: dict[str, Any] = {}
        for field_name, mapping in self.mapping.domain_fields.items():
            found = self._resolve_mapping(raw, mapping)
            if found is None:
                continue

            path, value = found
            consumed_paths.append(path)
            domain_fields[field_name] = self._apply_transform(
                value,
                mapping.first_transform,
            )

        for path in consumed_paths:
            _remove_path(raw_copy, path)

        return NormalizedMetadata(
            **values,
            boundary_flags=boundary_flags,
            domain_fields=domain_fields,
            unmapped=raw_copy,
        )

    def _resolve_mapping(
        self,
        raw: dict[str, Any],
        mapping: MetadataFieldMapping,
    ) -> tuple[tuple[str, ...], Any] | None:
        for path_text in (*mapping.aliases, *mapping.fallback):
            path = tuple(part for part in path_text.split(".") if part)
            found, value = _get_path(raw, path)
            if found and value is not None:
                return path, value
        return None

    def _apply_transform(self, value: Any, transform: str | None) -> Any:
        if transform is None:
            return value

        if transform == "int":
            return int(value)
        if transform == "float":
            return float(value)
        if transform == "str":
            return str(value)
        if transform == "bool":
            return self._coerce_bool(value)
        if transform in {"tuple", "string_tuple", "str_tuple"}:
            return _string_tuple(value)
        if transform == "identity":
            return value

        raise ValueError(f"Unknown metadata transform: {transform}")

    def _coerce_canonical_value(self, field_name: str, value: Any) -> Any:
        if field_name in {"source_element_ids", "source_table_ids", "section_path"}:
            return _string_tuple(value)
        if field_name in {"page_start", "page_end"}:
            return int(value)
        if field_name in {
            "chunk_id",
            "text",
            "document_id",
            "parent_id",
            "chunking_strategy",
        }:
            return str(value)
        return value

    def _coerce_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off", ""}:
                return False
        return bool(value)


_CANONICAL_FIELDS = (
    "chunk_id",
    "text",
    "document_id",
    "page_start",
    "page_end",
    "source_element_ids",
    "source_table_ids",
    "section_path",
    "parent_id",
    "chunking_strategy",
)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple | list | set):
        return tuple(str(item) for item in value)
    return (str(value),)


def _get_path(data: Any, path: tuple[str, ...]) -> tuple[bool, Any]:
    current = data
    for part in path:
        if isinstance(current, dict):
            if part not in current:
                return False, None
            current = current[part]
            continue

        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return False, None
            current = current[index]
            continue

        return False, None

    return True, current


def _remove_path(data: dict[str, Any], path: tuple[str, ...]) -> None:
    if not path:
        return

    current: Any = data
    parents: list[tuple[dict[str, Any], str]] = []

    for part in path[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        parents.append((current, part))
        current = current[part]

    if isinstance(current, dict):
        current.pop(path[-1], None)

    for parent, key in reversed(parents):
        child = parent.get(key)
        if isinstance(child, dict) and not child:
            parent.pop(key, None)
        else:
            break
