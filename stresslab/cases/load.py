"""Load curated stress cases from local fixtures."""

from __future__ import annotations

from pathlib import Path

from .models import ClaimDiagnosisGoldSet, DiagnosisGoldenCase, RAGFailureGoldenCase, StressCase

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
DIAGNOSIS_GOLDEN_DIR = Path(__file__).resolve().parent / "diagnosis_fixtures"
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def load_case(name: str) -> StressCase:
    fixture_name = name if name.endswith(".json") else f"{name}.json"
    fixture_path = FIXTURE_DIR / fixture_name
    return StressCase.model_validate_json(fixture_path.read_text(encoding="utf-8"))


def list_cases() -> list[str]:
    return sorted(path.stem for path in FIXTURE_DIR.glob("*.json"))


def load_diagnosis_golden_case(name: str) -> DiagnosisGoldenCase:
    fixture_name = name if name.endswith(".json") else f"{name}.json"
    fixture_path = DIAGNOSIS_GOLDEN_DIR / fixture_name
    return DiagnosisGoldenCase.model_validate_json(fixture_path.read_text(encoding="utf-8"))


def list_diagnosis_golden_cases() -> list[str]:
    return sorted(path.stem for path in DIAGNOSIS_GOLDEN_DIR.glob("*.json"))


def load_claim_diagnosis_gold_set(name: str = "claim_diagnosis_gold_v1.json") -> ClaimDiagnosisGoldSet:
    fixture_name = name if name.endswith(".json") else f"{name}.json"
    fixture_path = GOLDEN_DIR / fixture_name
    return ClaimDiagnosisGoldSet.model_validate_json(fixture_path.read_text(encoding="utf-8"))


def load_subtle_rag_failures() -> list[RAGFailureGoldenCase]:
    """Load the suite of subtle RAG failures."""
    from .subtle.rag_subtle_failures import SUBTLE_RAG_FAILURES
    return SUBTLE_RAG_FAILURES


def load_common_rag_failures() -> list[RAGFailureGoldenCase]:
    """Load the suite of common RAG failures."""
    from .golden.rag_failures import GOLDEN_RAG_FAILURES
    return GOLDEN_RAG_FAILURES
