"""Stresslab package for curated RAG stress scenarios and runners."""

from .diagnosis_evaluation import DiagnosisGoldenEvaluation, evaluate_diagnosis_case
from .evaluation import CaseEvaluation, evaluate_case

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CaseEvaluation",
    "DiagnosisGoldenEvaluation",
    "evaluate_case",
    "evaluate_diagnosis_case",
]
