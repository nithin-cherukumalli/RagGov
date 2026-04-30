"""First-wave parser-validation profile presets."""

from __future__ import annotations

from importlib import resources

from raggov.parser_validation.profile import (
    ParserValidationProfile,
    ParserValidationProfileSet,
)


_PRESET_NAMES = (
    "unstructured_by_title",
    "haystack_document_splitter",
    "langchain_recursive_character_splitter",
    "pymupdf4llm_page",
)


def list_presets() -> tuple[str, ...]:
    """Return bundled parser-validation preset names."""
    return _PRESET_NAMES


def load_preset_profile(name: str) -> ParserValidationProfile:
    """Load one bundled parser-validation profile preset."""
    if name not in _PRESET_NAMES:
        raise ValueError(f"Unknown parser-validation preset: {name}")

    profile_set = ParserValidationProfileSet.from_yaml(_preset_text(name))
    return profile_set.profiles[name]


def load_preset_profile_set() -> ParserValidationProfileSet:
    """Load all bundled parser-validation presets."""
    return ParserValidationProfileSet(
        profiles={name: load_preset_profile(name) for name in _PRESET_NAMES}
    )


def _preset_text(name: str) -> str:
    return (
        resources.files(__package__)
        .joinpath(f"{name}.yaml")
        .read_text(encoding="utf-8")
    )
