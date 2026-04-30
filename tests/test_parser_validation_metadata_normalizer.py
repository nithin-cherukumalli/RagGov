import pytest

from raggov.parser_validation.metadata_normalizer import (
    MetadataNormalizer,
    NormalizedMetadata,
)
from raggov.parser_validation.profile import (
    CanonicalMetadataMapping,
    MetadataFieldMapping,
)


def test_normalizer_resolves_aliases_before_fallbacks_and_dotted_paths():
    raw = {
        "id": "raw-id",
        "metadata": {
            "chunk_id": "chunk-1",
            "page_number": "7",
            "element_ids": ["e1", "e2"],
            "section": {"path": ["Chapter 1", "Rule 2"]},
        },
    }
    mapping = CanonicalMetadataMapping(
        chunk_id=MetadataFieldMapping(aliases=("metadata.chunk_id",), fallback=("id",)),
        page_start=MetadataFieldMapping(
            aliases=("metadata.page_number",),
            first_transform="int",
        ),
        source_element_ids=MetadataFieldMapping(aliases=("metadata.element_ids",)),
        section_path=MetadataFieldMapping(aliases=("metadata.section.path",)),
    )

    normalized = MetadataNormalizer(mapping).normalize(raw)

    assert normalized.chunk_id == "chunk-1"
    assert normalized.page_start == 7
    assert normalized.source_element_ids == ("e1", "e2")
    assert normalized.section_path == ("Chapter 1", "Rule 2")


def test_normalizer_uses_fallback_when_aliases_are_missing_or_none():
    raw = {
        "metadata": {
            "page_number": None,
            "page": "5",
            "table_id": "t1",
        }
    }
    mapping = CanonicalMetadataMapping(
        page_start=MetadataFieldMapping(
            aliases=("metadata.page_number",),
            fallback=("metadata.page",),
            first_transform="int",
        ),
        source_table_ids=MetadataFieldMapping(fallback=("metadata.table_id",)),
    )

    normalized = MetadataNormalizer(mapping).normalize(raw)

    assert normalized.page_start == 5
    assert normalized.source_table_ids == ("t1",)


def test_normalizer_extracts_domain_fields_and_unmapped_blob():
    raw = {
        "metadata": {
            "page": 2,
            "case_id": "abc",
            "jurisdiction": "AP",
            "unused": "keep",
        },
        "score": 0.91,
    }
    mapping = CanonicalMetadataMapping(
        page_start=MetadataFieldMapping(aliases=("metadata.page",)),
        domain_fields={
            "case_id": MetadataFieldMapping(aliases=("metadata.case_id",)),
            "jurisdiction": MetadataFieldMapping(aliases=("metadata.jurisdiction",)),
        },
    )

    normalized = MetadataNormalizer(mapping).normalize(raw)

    assert normalized.domain_fields == {
        "case_id": "abc",
        "jurisdiction": "AP",
    }
    assert normalized.unmapped == {
        "metadata": {"unused": "keep"},
        "score": 0.91,
    }


def test_normalizer_resolves_boundary_flags_and_bool_transform():
    raw = {
        "metadata": {
            "crosses_section_boundary": "true",
            "split_inside_table": 0,
        }
    }
    mapping = CanonicalMetadataMapping(
        boundary_flags={
            "crosses_section_boundary": MetadataFieldMapping(
                aliases=("metadata.crosses_section_boundary",),
                first_transform="bool",
            ),
            "split_inside_table": MetadataFieldMapping(
                aliases=("metadata.split_inside_table",),
                first_transform="bool",
            ),
        }
    )

    normalized = MetadataNormalizer(mapping).normalize(raw)

    assert normalized.boundary_flags == {
        "crosses_section_boundary": True,
        "split_inside_table": False,
    }


def test_normalizer_can_normalize_from_top_level_metadata_dict():
    raw = {
        "page": "4",
        "toc_path": ("Part I", "Chapter 2"),
        "orig_elements": ["Title: Part I", "Text: Body"],
    }
    mapping = CanonicalMetadataMapping(
        page_start=MetadataFieldMapping(aliases=("page",), first_transform="int"),
        section_path=MetadataFieldMapping(aliases=("toc_path",)),
        source_element_ids=MetadataFieldMapping(aliases=("orig_elements",)),
    )

    normalized = MetadataNormalizer(mapping).normalize(raw)

    assert normalized.page_start == 4
    assert normalized.section_path == ("Part I", "Chapter 2")
    assert normalized.source_element_ids == ("Title: Part I", "Text: Body")


def test_normalizer_rejects_unknown_transform():
    raw = {"metadata": {"page": "4"}}
    mapping = CanonicalMetadataMapping(
        page_start=MetadataFieldMapping(
            aliases=("metadata.page",),
            first_transform="not_a_transform",
        )
    )

    with pytest.raises(ValueError, match="Unknown metadata transform"):
        MetadataNormalizer(mapping).normalize(raw)


def test_normalized_metadata_defaults_are_empty_and_typed():
    normalized = NormalizedMetadata()

    assert normalized.source_element_ids == ()
    assert normalized.source_table_ids == ()
    assert normalized.section_path == ()
    assert normalized.boundary_flags == {}
    assert normalized.domain_fields == {}
    assert normalized.unmapped == {}
