"""
Calibrated confidence wrapper for GovRAG claim verification outputs.

Maps raw verifier scores and qualitative signals into calibrated confidence
only when a valid calibration artifact is available.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class CalibrationMode(str, Enum):
    NONE = "none"
    TEMPERATURE_SCALING = "temperature_scaling"
    ISOTONIC_REGRESSION = "isotonic_regression"
    RELIABILITY_TABLE = "reliability_table"


@dataclass
class CalibratedClaimConfidence:
    """Standardized output for calibrated confidence."""
    confidence: float | None
    status: str
    reliability_bucket: str | None = None
    source: str | None = None
    warning: str | None = None


class ClaimCalibrationModel:
    """
    Model wrapper for claim verification calibration.
    
    Supports various calibration strategies to map uncalibrated scores to
    actual probabilities of correctness.
    """

    def __init__(self, mode: CalibrationMode, params: Dict[str, Any], metadata: Dict[str, Any] | None = None):
        self.mode = mode
        self.params = params
        self.metadata = metadata or {}
        self._model = None
        
        if self.mode == CalibrationMode.ISOTONIC_REGRESSION:
            try:
                from sklearn.isotonic import IsotonicRegression
                self._model = IsotonicRegression(out_of_bounds="clip")
                if "y_thresholds" in self.params and "x_thresholds" in self.params:
                    # Pre-trained state loading
                    self._model.fit(self.params["x_thresholds"], self.params["y_thresholds"])
            except ImportError:
                logger.warning("sklearn not available, isotonic_regression will fail.")

    def calibrate(self, features: Dict[str, Any]) -> CalibratedClaimConfidence:
        raw_score = float(features.get("raw_score", 0.0))
        label = features.get("label", "unsupported")
        
        # If label is not 'entailed', confidence is conceptually different
        # (probability of being right about it being unsupported/contradicted).
        # For GovRAG, we primarily calibrate the 'entailed' confidence.
        
        if self.mode == CalibrationMode.NONE:
            return CalibratedClaimConfidence(
                confidence=None,
                status="uncalibrated",
                warning="No calibration model applied."
            )

        if self.mode == CalibrationMode.TEMPERATURE_SCALING:
            temp = float(self.params.get("temperature", 1.0))
            # Simple sigmoid scaling as proxy for temperature scaling on logits
            # assuming raw_score is already in [0,1]
            if raw_score <= 0:
                calibrated = 0.0
            elif raw_score >= 1:
                calibrated = 1.0
            else:
                # logit(p) / T
                logit = np.log(raw_score / (1 - raw_score))
                calibrated = 1 / (1 + np.exp(-logit / temp))
            
            return CalibratedClaimConfidence(
                confidence=float(calibrated),
                status="calibrated",
                reliability_bucket=self._get_bucket(calibrated),
                source=self.metadata.get("source")
            )

        if self.mode == CalibrationMode.RELIABILITY_TABLE:
            table = self.params.get("table", {})
            # Features for table lookup: fallback_used, verifier_mode, label
            fallback = str(features.get("fallback_used", False)).lower()
            mode = features.get("verifier_mode", "unknown")
            
            key = f"{mode}:{fallback}:{label}"
            calibrated = table.get(key)
            if calibrated is None:
                # Try generic fallback
                calibrated = table.get(f"default:{label}")
            
            if calibrated is not None:
                return CalibratedClaimConfidence(
                    confidence=float(calibrated),
                    status="calibrated",
                    reliability_bucket=self._get_bucket(calibrated),
                    source=self.metadata.get("source")
                )
            
        if self.mode == CalibrationMode.ISOTONIC_REGRESSION and self._model:
            try:
                calibrated = self._model.predict([raw_score])[0]
                return CalibratedClaimConfidence(
                    confidence=float(calibrated),
                    status="calibrated",
                    reliability_bucket=self._get_bucket(calibrated),
                    source=self.metadata.get("source")
                )
            except Exception as e:
                logger.error("Isotonic calibration failed: %s", e)

        return CalibratedClaimConfidence(
            confidence=None,
            status="uncalibrated",
            warning=f"Calibration mode {self.mode} failed or not implemented."
        )

    def _get_bucket(self, confidence: float) -> str:
        if confidence >= 0.9:
            return "high"
        if confidence >= 0.7:
            return "medium"
        if confidence >= 0.4:
            return "low"
        return "unreliable"


class ClaimCalibrationLoader:
    """Loads calibration models from disk."""

    @staticmethod
    def load(path: Union[str, Path]) -> ClaimCalibrationModel:
        path = Path(path)
        if not path.exists():
            logger.info("Calibration file not found at %s", path)
            return ClaimCalibrationModel(CalibrationMode.NONE, {})
            
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            mode_str = data.get("mode", "none")
            mode = CalibrationMode(mode_str)
            params = data.get("params", {})
            metadata = data.get("metadata", {})
            metadata["source"] = str(path)
            
            return ClaimCalibrationModel(mode, params, metadata)
        except Exception as e:
            logger.error("Failed to load calibration from %s: %s", path, e)
            return ClaimCalibrationModel(CalibrationMode.NONE, {})
