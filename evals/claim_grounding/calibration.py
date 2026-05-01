"""
Calibration status logic for the GovRAG claim-grounding evaluation harness.

CalibrationStatus vocabulary
-----------------------------
uncalibrated
    The verifier is using default, unvalidated thresholds.
    No sweep has been run, or the sweep was not applied.
    This is the default for all production runs unless explicitly overridden.

dev_calibrated_seed
    A threshold sweep was run on the synthetic seed dataset.
    The result is useful for internal development but is NOT a production
    calibration. The seed dataset is synthetic and does not represent real
    production query distribution.

    A threshold selected on synthetic seed cases is NOT production calibration.

production_calibrated
    A threshold sweep was run on a blind, real-world annotated production
    dataset. The dataset name, annotation date, and annotator information
    must be recorded in the calibration config alongside this status.
    This is the only status that should be used for production deployments.

Rules for production code
--------------------------
- Production code MUST NOT silently load a dev-calibrated config.
- If a config with calibration_status='dev_calibrated_seed' is loaded,
  the engine must emit a WARNING and require explicit opt-in.
- If calibration_status is absent or unknown, treat as 'uncalibrated'.
"""

from __future__ import annotations

from enum import Enum


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class CalibrationStatus(str, Enum):
    """Controlled vocabulary for verifier calibration provenance."""

    UNCALIBRATED = "uncalibrated"
    DEV_CALIBRATED_SEED = "dev_calibrated_seed"
    PRODUCTION_CALIBRATED = "production_calibrated"

    @classmethod
    def from_string(cls, value: str | None) -> "CalibrationStatus":
        """Parse a string into a CalibrationStatus, defaulting to UNCALIBRATED."""
        if value is None:
            return cls.UNCALIBRATED
        try:
            return cls(value)
        except ValueError:
            return cls.UNCALIBRATED


# ---------------------------------------------------------------------------
# Status inference
# ---------------------------------------------------------------------------

#: Minimum number of labeled cases needed to report dev_calibrated_seed status.
#: Below this count the dataset is too small to be informative.
MIN_CASES_FOR_DEV_CALIBRATION: int = 20


def infer_calibration_status(
    n_cases: int,
    dataset_name: str,
) -> str:
    """
    Infer the appropriate calibration_status string for a sweep run.

    Args:
        n_cases: Number of labeled cases in the evaluation dataset.
        dataset_name: Filename of the dataset (used for labelling only).

    Returns:
        A CalibrationStatus value string.

    Notes:
        'dev_calibrated_seed' is returned only when the dataset has enough
        cases to be minimally informative.  Even then, this status does NOT
        imply suitability for production.
    """
    if n_cases >= MIN_CASES_FOR_DEV_CALIBRATION and "seed" in dataset_name.lower():
        return CalibrationStatus.DEV_CALIBRATED_SEED.value
    if n_cases >= MIN_CASES_FOR_DEV_CALIBRATION:
        # Non-seed dataset large enough — could be production, require explicit flag
        return CalibrationStatus.UNCALIBRATED.value
    return CalibrationStatus.UNCALIBRATED.value


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def validate_calibration_config(config: dict) -> tuple[CalibrationStatus, list[str]]:
    """
    Validate a verifier config dict's calibration_status field.

    Returns:
        (status, warnings): The resolved CalibrationStatus and a list of
        warning messages that the caller should log.

    Production code must call this and act on the warnings before loading
    any config other than 'uncalibrated'.
    """
    raw_status = config.get("calibration_status")
    status = CalibrationStatus.from_string(raw_status)
    warnings: list[str] = []

    if status == CalibrationStatus.DEV_CALIBRATED_SEED:
        warnings.append(
            "Verifier config has calibration_status='dev_calibrated_seed'. "
            "This config was tuned on synthetic seed cases and is NOT suitable "
            "for production deployment without further validation. "
            "Set 'allow_dev_calibration=True' in config to suppress this warning."
        )

    if status == CalibrationStatus.PRODUCTION_CALIBRATED:
        dataset = config.get("calibration_dataset")
        if not dataset:
            warnings.append(
                "calibration_status='production_calibrated' but "
                "'calibration_dataset' is not set. Record the dataset name, "
                "annotation date, and annotator before deploying."
            )

    return status, warnings


def assert_not_silently_dev_calibrated(config: dict) -> None:
    """
    Raise ValueError if a dev_calibrated_seed config is loaded without
    explicit opt-in.

    Production analyzers should call this during __init__ when loading
    any external verifier config to prevent silent promotion of seed-tuned
    thresholds to production.

    Args:
        config: The verifier config dict.

    Raises:
        ValueError: If calibration_status='dev_calibrated_seed' and
                    'allow_dev_calibration' is not True.
    """
    status, warnings = validate_calibration_config(config)
    if status == CalibrationStatus.DEV_CALIBRATED_SEED:
        if not config.get("allow_dev_calibration", False):
            raise ValueError(
                "Refusing to silently load dev_calibrated_seed verifier config in "
                "production context. Set allow_dev_calibration=True to acknowledge "
                "that this config was tuned on synthetic seed cases only. "
                "A threshold selected on synthetic seed cases is NOT production calibration."
            )
