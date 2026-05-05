"""External signal adapter layer for GovRAG.

External tools (RAGAS, DeepEval, cross-encoders, etc.) provide advisory
evidence signals. GovRAG owns diagnosis, NCV, A2P, and final reports.
All adapters are optional dependencies; missing adapters are always visible.
"""

from raggov.evaluators.base import (
    CitationVerifierAdapter,
    ClaimVerifierAdapter,
    ExternalEvaluationResult,
    ExternalEvaluatorProvider,
    ExternalSignalProvider,
    ExternalSignalRecord,
    ExternalSignalType,
    RetrievalSignalProvider,
)
from raggov.evaluators.registry import ExternalEvaluatorRegistry

__all__ = [
    "CitationVerifierAdapter",
    "ClaimVerifierAdapter",
    "ExternalEvaluationResult",
    "ExternalEvaluatorProvider",
    "ExternalEvaluatorRegistry",
    "ExternalSignalProvider",
    "ExternalSignalRecord",
    "ExternalSignalType",
    "RetrievalSignalProvider",
]
