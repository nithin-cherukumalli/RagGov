"""Optional RefChecker claim verification signal adapter.

RefChecker is an optional dependency. Install via: pip install refchecker
(or as documented in the project — the package may require manual install).

Input mapping from GovRAG:
  claim / statement     = claim_text
  reference / evidence  = context chunks text
  claim_id              = for affected_claim_ids tracking

Claim label normalization:
  entailed    → claim is supported by evidence
  contradicted → claim conflicts with evidence
  unsupported  → claim has no grounding (hallucination signal)
  unclear      → cannot determine support status

Triplet outputs (if available) are preserved in raw_payload["triplets"].

Signal label thresholds for numeric scores (uncalibrated, configurable):
  >= 0.75 → high
  0.40–0.75 → medium
  < 0.40  → low
  None / error → unavailable
"""

from __future__ import annotations

import importlib
from typing import Any

from raggov.evaluators.base import (
    ExternalEvaluationResult,
    ExternalEvaluatorProvider,
    ExternalSignalRecord,
    ExternalSignalType,
)
from raggov.models.run import RAGRun


# Normalized claim labels from RefChecker-style output.
_VALID_CLAIM_LABELS: frozenset[str] = frozenset(
    {"entailed", "contradicted", "unsupported", "unclear"}
)

# Default derived-label thresholds for numeric scores (uncalibrated).
_DEFAULT_HIGH_THRESHOLD: float = 0.75
_DEFAULT_LOW_THRESHOLD: float = 0.40


def _normalize_claim_label(raw: str | None) -> str:
    """Normalize a raw RefChecker claim label to a canonical GovRAG label."""
    if raw is None:
        return "unclear"
    mapped = {
        "entailment": "entailed",
        "entailed": "entailed",
        "supported": "entailed",
        "contradiction": "contradicted",
        "contradicted": "contradicted",
        "neutral": "unsupported",
        "unsupported": "unsupported",
        "hallucinated": "unsupported",
        "unknown": "unclear",
        "unclear": "unclear",
    }
    return mapped.get(raw.lower().strip(), "unclear")


def _derive_label(value: Any, *, high: float, low: float) -> str:
    """Map a numeric score to a human-readable label."""
    if not isinstance(value, (int, float)):
        return "unavailable"
    if value >= high:
        return "high"
    if value >= low:
        return "medium"
    return "low"


class RefCheckerClaimSignalProvider:
    """Provide advisory RefChecker claim verification signals when available.

    This adapter:
    - Is fully optional; missing refchecker returns succeeded=False, missing_dependency=True.
    - Emits ExternalSignalRecord with calibration_status="uncalibrated_locally"
      and recommended_for_gating=False.
    - Normalizes claim labels (entailed/contradicted/unsupported/unclear).
    - Preserves triplet-level outputs in raw_payload["triplets"].
    - Never directly sets a GovRAG FailureType. GovRAG owns all diagnosis.
    """

    name: str = "refchecker_claim"
    provider: ExternalEvaluatorProvider = ExternalEvaluatorProvider.refchecker

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._high_threshold: float = float(
            self.config.get("refchecker_label_high_threshold", _DEFAULT_HIGH_THRESHOLD)
        )
        self._low_threshold: float = float(
            self.config.get("refchecker_label_low_threshold", _DEFAULT_LOW_THRESHOLD)
        )

    def is_available(self) -> bool:
        try:
            importlib.import_module("refchecker")
            return True
        except ImportError:
            return False

    def evaluate(self, run: RAGRun) -> ExternalEvaluationResult:
        """Evaluate RefChecker claim signals for the given RAGRun."""
        if not self.is_available():
            return ExternalEvaluationResult(
                provider=self.provider,
                succeeded=False,
                missing_dependency=True,
                error=(
                    "refchecker: package not installed. "
                    "Install via `pip install refchecker` or consult project docs."
                ),
            )

        try:
            payload = self._metric_payload(run)
        except Exception as exc:
            return ExternalEvaluationResult(
                provider=self.provider,
                succeeded=False,
                error=f"refchecker claim evaluation failed: {exc}",
            )

        signals = self._signals_from_payload(payload)
        return ExternalEvaluationResult(
            provider=self.provider,
            succeeded=True,
            signals=signals,
            raw_payload=dict(payload),
        )

    def verify_claims(
        self, claims: list[str], context: list[str]
    ) -> list[ExternalSignalRecord]:
        """Verify a list of claim texts against context chunks.

        Returns one ExternalSignalRecord per claim.
        Uses metric_runner if configured; otherwise returns empty list.
        """
        runner = self.config.get("claim_runner")
        if runner is None:
            return []
        results = runner(claims, context)
        signals = []
        for i, item in enumerate(results or []):
            claim_id = item.get("claim_id", f"claim_{i}")
            raw_label = item.get("label")
            label = _normalize_claim_label(raw_label)
            score = item.get("score")
            triplets = item.get("triplets", [])
            signals.append(
                ExternalSignalRecord(
                    provider=self.provider,
                    signal_type=ExternalSignalType.claim_support,
                    metric_name="refchecker_claim_check",
                    value=score,
                    label=label,
                    explanation=(
                        f"RefChecker claim check: label={label}. "
                        "Advisory evidence only — GovRAG owns final claim diagnosis."
                    ),
                    affected_claim_ids=[claim_id],
                    raw_payload={"label": raw_label, "score": score, "triplets": triplets},
                    method_type="external_signal_adapter",
                    calibration_status="uncalibrated_locally",
                    recommended_for_gating=False,
                    limitations=[
                        "refchecker_signal_is_advisory_not_diagnostic",
                        "uncalibrated_locally",
                    ],
                )
            )
        return signals

    def _metric_payload(self, run: RAGRun) -> dict[str, Any]:
        runner = self.config.get("metric_runner")
        if runner is not None:
            return dict(runner(run))
        configured = self.config.get("metric_results")
        if configured is not None:
            return dict(configured)
        return {}

    def _signals_from_payload(self, payload: dict[str, Any]) -> list[ExternalSignalRecord]:
        claims_data = payload.get("claims", [])
        signals = []
        for item in claims_data:
            claim_id = item.get("claim_id", "unknown")
            raw_label = item.get("label")
            label = _normalize_claim_label(raw_label)
            score = item.get("score")
            triplets = item.get("triplets", [])

            # Map label to signal_type
            if label in ("unsupported",):
                sig_type = ExternalSignalType.hallucination
            else:
                sig_type = ExternalSignalType.claim_support

            signals.append(
                ExternalSignalRecord(
                    provider=self.provider,
                    signal_type=sig_type,
                    metric_name="refchecker_claim_check",
                    value=score,
                    label=label,
                    explanation=(
                        f"RefChecker claim={claim_id}: label={label}. "
                        "Advisory evidence only — GovRAG owns final claim diagnosis."
                    ),
                    affected_claim_ids=[claim_id],
                    raw_payload={"label": raw_label, "score": score, "triplets": triplets},
                    method_type="external_signal_adapter",
                    calibration_status="uncalibrated_locally",
                    recommended_for_gating=False,
                    limitations=[
                        "refchecker_signal_is_advisory_not_diagnostic",
                        "uncalibrated_locally",
                    ],
                )
            )
        return signals


# Backward-compatible alias.
RefCheckerClaimAdapter = RefCheckerClaimSignalProvider
