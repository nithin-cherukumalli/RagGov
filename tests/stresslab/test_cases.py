"""Tests for curated stresslab cases."""

from __future__ import annotations

from stresslab.cases import load_case


def test_load_case_reads_curated_fixture() -> None:
    case = load_case("metadata_misread_ms1")

    assert case.case_id == "metadata_misread_ms1"
    assert case.document_set == ["2011SE_MS1.PDF"]
    assert case.expected_primary_failure == "metadata_misread"
    assert case.expected_should_have_answered is True
