"""Calibration package for GovRAG.

Exposes both the ARES PPI statistical calibration for overall metrics
and the claim-level confidence calibration.
"""

from raggov.calibration.core import (
    ARESCalibrator,
    CalibrationSample,
    ConfidenceInterval,
    CalibrationStatus,
)
from raggov.calibration.claim_calibration import (
    ClaimCalibrationModel,
    ClaimCalibrationLoader,
    CalibratedClaimConfidence,
    CalibrationMode,
)

__all__ = [
    "ARESCalibrator",
    "CalibrationSample",
    "ConfidenceInterval",
    "CalibrationStatus",
    "ClaimCalibrationModel",
    "ClaimCalibrationLoader",
    "CalibratedClaimConfidence",
    "CalibrationMode",
]
