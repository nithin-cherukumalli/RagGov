"""Statistical calibration using ARES Prediction-Powered Inference (PPI).

This module implements the PPI method from Saad-Falcon et al., "ARES: An Automated
Evaluation Framework for Retrieval Augmented Generation Systems", Stanford, NAACL 2024.

Key capability: Produce statistically valid confidence intervals on diagnostic metrics
using only 150-300 human-labeled examples. For regulated industries (finance, healthcare,
legal), this enables auditable claims like:
    "Faithfulness = 0.72 [95% CI: 0.68–0.76]"

No other RAG diagnostic tool offers this capability.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CalibrationSample(BaseModel):
    """A single calibration sample with automated and gold-standard metrics.

    Used to calibrate GovRAG's diagnostic engine by comparing automated predictions
    against human-labeled ground truth.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    run_id: str
    # Automated metrics from DiagnosisEngine
    automated_faithfulness: float = Field(ge=0.0, le=1.0)
    automated_retrieval_precision: float = Field(ge=0.0, le=1.0)
    automated_answer_correctness: float = Field(ge=0.0, le=1.0)
    # Gold-standard (human-labeled) metrics
    gold_faithfulness: float = Field(ge=0.0, le=1.0)
    gold_retrieval_precision: float = Field(ge=0.0, le=1.0)
    gold_answer_correctness: float = Field(ge=0.0, le=1.0)


class ConfidenceInterval(BaseModel):
    """Confidence interval for a single metric.

    Provides statistically valid bounds on metric accuracy using PPI.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    metric: Literal["faithfulness", "retrieval_precision", "answer_correctness"]
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float = 0.95
    n_labeled: int
    n_total: int

    def __str__(self) -> str:
        """Format as: metric = point [CI: lower–upper] (n_labeled=X, n_total=Y)."""
        return (
            f"{self.metric} = {self.point_estimate:.3f} "
            f"[{self.confidence_level:.0%} CI: {self.lower:.3f}–{self.upper:.3f}] "
            f"(n_labeled={self.n_labeled}, n_total={self.n_total})"
        )


def _ppi_point_estimate(
    automated_values: list[float],
    gold_values: list[float],
    all_automated_values: list[float],
) -> float:
    """Compute PPI point estimate.

    Formula: M_ppi = mean(all_automated) + (1/n) * sum(gold_i - automated_i)

    The correction term uses only the labeled subset to correct systematic bias
    in the automated predictions.

    Args:
        automated_values: Automated predictions for labeled samples
        gold_values: Gold-standard labels for labeled samples
        all_automated_values: Automated predictions for all samples (labeled + unlabeled)

    Returns:
        PPI-corrected point estimate
    """
    n_labeled = len(gold_values)
    mean_automated = sum(all_automated_values) / len(all_automated_values)
    correction = sum(g - a for g, a in zip(gold_values, automated_values)) / n_labeled
    return mean_automated + correction


def _ppi_variance(
    automated_values: list[float],
    gold_values: list[float],
) -> float:
    """Compute variance for PPI confidence interval.

    Uses the residuals (gold - automated) to estimate variance of the correction term.

    Args:
        automated_values: Automated predictions for labeled samples
        gold_values: Gold-standard labels for labeled samples

    Returns:
        Variance estimate (standard error squared)
    """
    n = len(gold_values)
    if n < 2:
        # Cannot compute variance with fewer than 2 samples
        return float("inf")

    residuals = [g - a for g, a in zip(gold_values, automated_values)]
    mean_residual = sum(residuals) / n

    # Sample variance of residuals
    variance = sum((r - mean_residual) ** 2 for r in residuals) / (n - 1)

    # Standard error squared
    return variance / n


def _confidence_interval(
    point_estimate: float,
    variance: float,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Calculate confidence interval using normal approximation.

    Formula: CI = point_estimate +/- z * sqrt(variance)

    Args:
        point_estimate: PPI point estimate
        variance: PPI variance estimate
        confidence_level: Confidence level (0.90, 0.95, or 0.99)

    Returns:
        (lower_bound, upper_bound) tuple
    """
    # Z-scores for common confidence levels (no scipy needed)
    Z_SCORES = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576,
    }
    z = Z_SCORES.get(confidence_level, 1.96)  # Default to 95%

    if math.isinf(variance):
        # Infinite variance - return point estimate as bounds
        return (point_estimate, point_estimate)

    margin = z * math.sqrt(variance)
    return (point_estimate - margin, point_estimate + margin)


class ARESCalibrator:
    """Statistical calibration using Prediction-Powered Inference (PPI).

    Produces confidence intervals on diagnostic metrics using a small set of
    human-labeled samples (minimum 30, recommended 150-300).

    Reference:
        Saad-Falcon et al., "ARES: An Automated Evaluation Framework for
        Retrieval Augmented Generation Systems", Stanford, NAACL 2024.

    Example:
        >>> calibrator = ARESCalibrator()
        >>> for sample in labeled_samples:
        ...     calibrator.add_sample(sample)
        >>> intervals = calibrator.calibrate()
        >>> for ci in intervals:
        ...     print(ci)
        faithfulness = 0.814 [95% CI: 0.771–0.857] (n_labeled=48, n_total=48)
        retrieval_precision = 0.763 [95% CI: 0.721–0.805] (n_labeled=48, n_total=48)
        answer_correctness = 0.812 [95% CI: 0.769–0.855] (n_labeled=48, n_total=48)
    """

    MIN_SAMPLES = 30  # Statistical minimum for valid inference

    def __init__(self, confidence_level: float = 0.95) -> None:
        """Initialize calibrator.

        Args:
            confidence_level: Confidence level for intervals (0.90, 0.95, or 0.99)
        """
        self._samples: list[CalibrationSample] = []
        self._confidence_level = confidence_level

    def add_sample(self, sample: CalibrationSample) -> None:
        """Add a calibration sample with automated and gold metrics.

        Args:
            sample: CalibrationSample with both automated and gold-standard metrics
        """
        self._samples.append(sample)

    def calibrate(self) -> list[ConfidenceInterval]:
        """Compute PPI-corrected confidence intervals for all three metrics.

        Returns:
            List of ConfidenceInterval for faithfulness, retrieval_precision,
            and answer_correctness.

        Raises:
            ValueError: If fewer than MIN_SAMPLES (30) samples provided.
        """
        if len(self._samples) < self.MIN_SAMPLES:
            raise ValueError(
                f"Calibration requires at least {self.MIN_SAMPLES} samples, "
                f"got {len(self._samples)}"
            )

        results = []

        # Calibrate each metric independently
        for metric in ("faithfulness", "retrieval_precision", "answer_correctness"):
            # Extract automated and gold values for this metric
            automated = [
                getattr(sample, f"automated_{metric}") for sample in self._samples
            ]
            gold = [getattr(sample, f"gold_{metric}") for sample in self._samples]

            # Compute PPI estimate
            point_est = _ppi_point_estimate(automated, gold, automated)
            var = _ppi_variance(automated, gold)
            lower, upper = _confidence_interval(point_est, var, self._confidence_level)

            # Clamp to [0, 1] and round
            results.append(
                ConfidenceInterval(
                    metric=metric,
                    point_estimate=round(point_est, 4),
                    lower=round(max(0.0, lower), 4),
                    upper=round(min(1.0, upper), 4),
                    confidence_level=self._confidence_level,
                    n_labeled=len(self._samples),
                    n_total=len(self._samples),
                )
            )

        return results

    def summary(self) -> str:
        """Return a human-readable summary of calibration results.

        Returns:
            Formatted multi-line string with calibration report.
        """
        if len(self._samples) < self.MIN_SAMPLES:
            return (
                f"Insufficient samples: {len(self._samples)}/{self.MIN_SAMPLES} "
                f"(need {self.MIN_SAMPLES - len(self._samples)} more)"
            )

        intervals = self.calibrate()
        lines = [
            f"ARES Calibration Report (n={len(self._samples)}, "
            f"confidence={self._confidence_level:.0%})",
            "-" * 60,
        ]

        for ci in intervals:
            width = ci.upper - ci.lower
            lines.append(
                f"  {ci.metric:20s}: {ci.point_estimate:.2f} "
                f"[{ci.confidence_level:.0%} CI: {ci.lower:.2f}–{ci.upper:.2f}] "
                f"(width: {width:.3f})"
            )

        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Save calibration samples to JSONL file.

        Each line is a JSON-serialized CalibrationSample.

        Args:
            path: Path to output JSONL file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            for sample in self._samples:
                f.write(sample.model_dump_json() + "\n")

    @classmethod
    def load(
        cls, path: str | Path, confidence_level: float = 0.95
    ) -> ARESCalibrator:
        """Load calibration samples from JSONL file.

        Args:
            path: Path to JSONL file with CalibrationSamples
            confidence_level: Confidence level for intervals (default: 0.95)

        Returns:
            ARESCalibrator instance with loaded samples

        Raises:
            OSError: If file cannot be read
            ValidationError: If JSON is malformed or validation fails
        """
        calibrator = cls(confidence_level=confidence_level)

        with Path(path).open() as f:
            for line in f:
                if line.strip():
                    sample = CalibrationSample.model_validate_json(line)
                    calibrator.add_sample(sample)

        return calibrator
