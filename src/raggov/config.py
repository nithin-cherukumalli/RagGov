"""Configuration models and loading utilities for RagGov."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


DiagnosisMode = Literal["external-enhanced", "native", "calibrated"]

class EngineConfig(TypedDict, total=False):
    mode: DiagnosisMode
    enabled_external_providers: list[str]
    strict_external_evaluators: bool
    fallback_policy: str
    # Other config keys as needed
    enable_ncv: bool
    enable_a2p: bool
    use_llm: bool
    use_a2p_v2: bool
    claim_verifier: str
    claim_verifier_mode: str
    llm_client: Any
    llm_fn: Any
    claim_extractor_client: Any
    fail_threshold: float
    claim_calibration_path: str
    calibrator: Any
    enable_triplet_verification: bool
