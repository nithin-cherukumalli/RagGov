import pytest
from pydantic import ValidationError

from raggov.parser_validation.models import (
    ChunkingStrategyProfile,
    ChunkingStrategyType,
)
from raggov.parser_validation.profile import (
    CanonicalMetadataMapping,
    MetadataFieldMapping,
    ParserProfile,
    ParserValidationProfile,
    ParserValidationProfileSet,
)


def test_profile_set_loads_multiple_named_profiles_from_yaml():
    payload = """
profiles:
  unstructured_by_title:
    parser:
      name: unstructured
      version: "0.0"
    chunking_strategy:
      strategy_type: hierarchical
      preserves_section_path: true
      requires_provenance: true
    metadata_mapping:
      page_start:
        aliases:
          - metadata.page_number
          - page
        fallback:
          - metadata.page
        first_transform: int
      source_element_ids:
        aliases:
          - metadata.orig_elements
          - metadata.element_ids
    infer_from_legacy: true
  pymupdf4llm_page:
    parser:
      name: pymupdf4llm
    chunking_strategy:
      strategy_type: markdown_header
    metadata_mapping:
      section_path:
        aliases:
          - metadata.toc_path
          - metadata.headers
"""

    profile_set = ParserValidationProfileSet.from_yaml(payload)

    assert set(profile_set.profiles) == {"unstructured_by_title", "pymupdf4llm_page"}
    profile = profile_set.profiles["unstructured_by_title"]
    assert profile.parser.name == "unstructured"
    assert profile.chunking_strategy.strategy_type == ChunkingStrategyType.HIERARCHICAL
    assert profile.chunking_strategy.preserves_section_path is True
    assert profile.metadata_mapping.page_start.aliases == (
        "metadata.page_number",
        "page",
    )
    assert profile.metadata_mapping.page_start.fallback == ("metadata.page",)
    assert profile.metadata_mapping.page_start.first_transform == "int"
    assert profile.infer_from_legacy is True


def test_profile_yaml_roundtrip_preserves_aliases_and_fallbacks():
    profile_set = ParserValidationProfileSet(
        profiles={
            "default": ParserValidationProfile(
                parser=ParserProfile(name="custom_pdf_parser"),
                chunking_strategy=ChunkingStrategyProfile(
                    strategy_type=ChunkingStrategyType.TABLE_AWARE,
                    preserves_table_structure=True,
                    preserves_table_headers=True,
                ),
                metadata_mapping=CanonicalMetadataMapping(
                    page_start=MetadataFieldMapping(
                        aliases=("meta.page",),
                        fallback=("page",),
                        first_transform="int",
                    ),
                    source_table_ids=MetadataFieldMapping(
                        aliases=("meta.table_ids",),
                    ),
                ),
            )
        }
    )

    loaded = ParserValidationProfileSet.from_yaml(profile_set.to_yaml())
    profile = loaded.profiles["default"]

    assert profile.parser.name == "custom_pdf_parser"
    assert profile.chunking_strategy.strategy_type == ChunkingStrategyType.TABLE_AWARE
    assert profile.metadata_mapping.page_start.aliases == ("meta.page",)
    assert profile.metadata_mapping.page_start.fallback == ("page",)
    assert profile.metadata_mapping.page_start.first_transform == "int"
    assert profile.metadata_mapping.source_table_ids.aliases == ("meta.table_ids",)


def test_chunking_strategy_profile_is_strict_pydantic_schema():
    with pytest.raises(ValidationError):
        ChunkingStrategyProfile.model_validate({"unknown_field": True})

    profile = ChunkingStrategyProfile.model_validate(
        {
            "strategy_type": "sentence",
            "sentence_boundaries": "preserved",
            "table_structure": "may_flatten",
            "hierarchy_mode": "none",
        }
    )

    assert profile.strategy_type == ChunkingStrategyType.SENTENCE
    assert profile.sentence_boundaries == "preserved"
