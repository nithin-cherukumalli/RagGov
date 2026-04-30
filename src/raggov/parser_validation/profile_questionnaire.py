"""Dependency-free helpers for generating parser-validation profile YAML."""

from __future__ import annotations

from typing import Any

from raggov.parser_validation.adapters import normalize_chunking_strategy
from raggov.parser_validation.models import default_chunking_profile
from raggov.parser_validation.profile import (
    CanonicalMetadataMapping,
    MetadataFieldMapping,
    ParserProfile,
    ParserValidationProfile,
    ParserValidationProfileSet,
)


def build_profile_from_answers(answers: dict[str, Any]) -> ParserValidationProfile:
    """Build a ParserValidationProfile from questionnaire-style answers."""
    strategy_type = normalize_chunking_strategy(answers.get("chunking_strategy"))

    return ParserValidationProfile(
        parser=ParserProfile(name=_optional_str(answers.get("parser_name"))),
        chunking_strategy=default_chunking_profile(strategy_type),
        metadata_mapping=CanonicalMetadataMapping(
            page_start=_field_mapping(answers.get("page_field"), transform="int"),
            source_element_ids=_field_mapping(
                answers.get("provenance_field"),
                transform="string_tuple",
            ),
            section_path=_field_mapping(
                answers.get("section_field"),
                transform="string_tuple",
            ),
            source_table_ids=_field_mapping(
                answers.get("table_field"),
                transform="string_tuple",
            ),
            parent_id=_field_mapping(answers.get("parent_field")),
        ),
        infer_from_legacy=bool(answers.get("infer_from_legacy", False)),
        description=_optional_str(answers.get("description")),
    )


def build_profile_set_from_answers(answers: dict[str, Any]) -> ParserValidationProfileSet:
    """Build a named profile set from questionnaire-style answers."""
    name = _optional_str(answers.get("name")) or "custom"
    return ParserValidationProfileSet(
        profiles={name: build_profile_from_answers(answers)}
    )


def profile_yaml_from_answers(answers: dict[str, Any]) -> str:
    """Serialize questionnaire answers as parser-validation profile YAML."""
    return build_profile_set_from_answers(answers).to_yaml()


def _field_mapping(value: Any, transform: str | None = None) -> MetadataFieldMapping:
    path = _optional_str(value)
    if path is None:
        return MetadataFieldMapping()
    return MetadataFieldMapping(aliases=(path,), first_transform=transform)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
