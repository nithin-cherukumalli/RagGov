"""Tests for ARES Prediction-Powered Inference calibration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from raggov.calibration import (
    ARESCalibrator,
    CalibrationSample,
    ConfidenceInterval,
)
from raggov.cli import app


runner = CliRunner()
CALIBRATION_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "stresslab"
    / "archive"
    / "cases"
    / "ares_calibration_samples_v1.jsonl"
)


def _sample(
    run_id: str,
    automated: float = 0.7,
    gold: float = 0.8,
) -> CalibrationSample:
    """Helper to create a calibration sample."""
    return CalibrationSample(
        run_id=run_id,
        automated_faithfulness=automated,
        automated_retrieval_precision=automated,
        automated_answer_correctness=automated,
        gold_faithfulness=gold,
        gold_retrieval_precision=gold,
        gold_answer_correctness=gold,
    )


def _calibrator_with_samples(
    n_samples: int = 50,
    automated_base: float = 0.6,
    gold_base: float = 0.7,
) -> ARESCalibrator:
    """Helper to create a calibrator with n samples."""
    calibrator = ARESCalibrator()
    for i in range(n_samples):
        # Add variance to automated and gold independently
        # This creates variance in the residuals (gold - automated)
        automated = automated_base + (i % 10 - 5) * 0.02
        gold = gold_base + ((i + 3) % 10 - 5) * 0.025  # Different pattern
        calibrator.add_sample(_sample(f"run-{i}", automated, gold))
    return calibrator


def _write_sample_file(path: Path, n_samples: int = 50) -> None:
    """Write sample JSONL file for CLI testing."""
    calibrator = _calibrator_with_samples(n_samples=n_samples)
    calibrator.save(path)


def test_calibration_sample_validation() -> None:
    """CalibrationSample should validate metric bounds [0, 1]."""
    # Valid sample
    sample = CalibrationSample(
        run_id="run-1",
        automated_faithfulness=0.5,
        automated_retrieval_precision=0.6,
        automated_answer_correctness=0.7,
        gold_faithfulness=0.6,
        gold_retrieval_precision=0.7,
        gold_answer_correctness=0.8,
    )
    assert sample.run_id == "run-1"
    assert sample.automated_faithfulness == 0.5

    # Invalid: out of bounds
    with pytest.raises(ValueError):
        CalibrationSample(
            run_id="run-2",
            automated_faithfulness=1.5,  # > 1.0
            automated_retrieval_precision=0.6,
            automated_answer_correctness=0.7,
            gold_faithfulness=0.6,
            gold_retrieval_precision=0.7,
            gold_answer_correctness=0.8,
        )


def test_confidence_interval_model() -> None:
    """ConfidenceInterval should store all required fields."""
    ci = ConfidenceInterval(
        metric="faithfulness",
        point_estimate=0.75,
        lower=0.70,
        upper=0.80,
        confidence_level=0.95,
        n_labeled=50,
        n_total=50,
    )
    assert ci.metric == "faithfulness"
    assert ci.point_estimate == 0.75
    assert ci.lower == 0.70
    assert ci.upper == 0.80


def test_ppi_point_estimate_corrects_bias() -> None:
    """PPI should correct for systematic bias in automated predictions."""
    # Automated underestimates by 0.1 consistently
    calibrator = ARESCalibrator()
    for i in range(50):
        # Automated: 0.6, Gold: 0.7 (consistent +0.1 bias)
        calibrator.add_sample(_sample(f"run-{i}", automated=0.6, gold=0.7))

    intervals = calibrator.calibrate()

    # Point estimate should be corrected upward from 0.6 toward 0.7
    faithfulness_interval = intervals[0]
    assert faithfulness_interval.metric == "faithfulness"
    assert faithfulness_interval.point_estimate > 0.65
    assert faithfulness_interval.point_estimate < 0.75


def test_ppi_correction_when_gold_higher() -> None:
    """When gold > automated, PPI should adjust estimate upward."""
    calibrator = _calibrator_with_samples(
        automated_base=0.5,
        gold_base=0.7,
        n_samples=50,
    )

    intervals = calibrator.calibrate()

    # All three metrics should show upward correction
    for interval in intervals:
        assert interval.point_estimate > 0.5
        assert interval.point_estimate < 0.9


def test_ppi_correction_when_gold_lower() -> None:
    """When gold < automated, PPI should adjust estimate downward."""
    calibrator = _calibrator_with_samples(
        automated_base=0.8,
        gold_base=0.6,
        n_samples=50,
    )

    intervals = calibrator.calibrate()

    # All three metrics should show downward correction
    for interval in intervals:
        assert interval.point_estimate < 0.8
        assert interval.point_estimate > 0.5


def test_ci_width_narrows_with_more_samples() -> None:
    """Confidence interval should be tighter with more samples."""
    small_calibrator = _calibrator_with_samples(n_samples=40)
    large_calibrator = _calibrator_with_samples(n_samples=200)

    small_intervals = small_calibrator.calibrate()
    large_intervals = large_calibrator.calibrate()

    small_width = small_intervals[0].upper - small_intervals[0].lower
    large_width = large_intervals[0].upper - large_intervals[0].lower

    assert large_width < small_width


def test_calibrate_raises_valueerror_with_insufficient_samples() -> None:
    """Calibration should fail with fewer than 30 samples."""
    calibrator = ARESCalibrator()
    for i in range(25):
        calibrator.add_sample(_sample(f"run-{i}"))

    with pytest.raises(ValueError, match="at least 30 samples"):
        calibrator.calibrate()


def test_calibrate_with_exact_minimum_samples() -> None:
    """Calibration should work with exactly 30 samples."""
    calibrator = ARESCalibrator()
    for i in range(30):
        calibrator.add_sample(_sample(f"run-{i}"))

    intervals = calibrator.calibrate()
    assert len(intervals) == 4
    assert all(isinstance(ci, ConfidenceInterval) for ci in intervals)


def test_all_metrics_include_overall_confidence_interval() -> None:
    """Calibration should produce metric intervals plus an overall confidence interval."""
    calibrator = _calibrator_with_samples(n_samples=50)
    intervals = calibrator.calibrate()

    assert len(intervals) == 4
    metrics = {ci.metric for ci in intervals}
    assert metrics == {
        "faithfulness",
        "retrieval_precision",
        "answer_correctness",
        "overall_confidence",
    }


def test_confidence_interval_clamped_to_zero_one() -> None:
    """Confidence intervals should be clamped to [0, 1] range."""
    calibrator = ARESCalibrator()
    # Create scenario that might produce CI outside [0, 1]
    for i in range(50):
        # Very high values with noise
        calibrator.add_sample(_sample(f"run-{i}", automated=0.95, gold=0.98))

    intervals = calibrator.calibrate()

    for interval in intervals:
        assert 0.0 <= interval.lower <= 1.0
        assert 0.0 <= interval.upper <= 1.0
        assert interval.lower <= interval.point_estimate <= interval.upper


def test_golden_calibration_sample_file_loads_and_meets_minimum_size() -> None:
    calibrator = ARESCalibrator.load(CALIBRATION_FIXTURE)

    assert len(calibrator._samples) == 50
    assert len({sample.run_id for sample in calibrator._samples}) == 50


def test_summary_with_sufficient_samples() -> None:
    """Summary should return formatted calibration results."""
    calibrator = _calibrator_with_samples(n_samples=50)
    summary = calibrator.summary()

    assert "ARES Calibration Report" in summary
    assert "n=50" in summary
    assert "95%" in summary
    assert "faithfulness" in summary
    assert "retrieval_precision" in summary
    assert "answer_correctness" in summary


def test_summary_with_insufficient_samples() -> None:
    """Summary should indicate insufficient samples."""
    calibrator = ARESCalibrator()
    for i in range(20):
        calibrator.add_sample(_sample(f"run-{i}"))

    summary = calibrator.summary()
    assert "Insufficient samples" in summary
    assert "20/30" in summary


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """Calibrator should serialize and deserialize without data loss."""
    original = _calibrator_with_samples(n_samples=50)
    samples_file = tmp_path / "samples.jsonl"

    original.save(samples_file)
    loaded = ARESCalibrator.load(samples_file)

    assert len(loaded._samples) == 50

    # Intervals should be identical
    original_intervals = original.calibrate()
    loaded_intervals = loaded.calibrate()

    for orig, load in zip(original_intervals, loaded_intervals):
        assert orig.point_estimate == load.point_estimate
        assert orig.lower == load.lower
        assert orig.upper == load.upper


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    """Save should create parent directories if they don't exist."""
    nested_path = tmp_path / "deep" / "nested" / "dir" / "samples.jsonl"
    calibrator = _calibrator_with_samples(n_samples=30)

    calibrator.save(nested_path)

    assert nested_path.exists()
    assert nested_path.is_file()


def test_load_with_different_confidence_level(tmp_path: Path) -> None:
    """Load should accept custom confidence level."""
    calibrator = _calibrator_with_samples(n_samples=50)
    samples_file = tmp_path / "samples.jsonl"
    calibrator.save(samples_file)

    loaded_95 = ARESCalibrator.load(samples_file, confidence_level=0.95)
    loaded_99 = ARESCalibrator.load(samples_file, confidence_level=0.99)

    intervals_95 = loaded_95.calibrate()
    intervals_99 = loaded_99.calibrate()

    # 99% CI should be wider than 95% CI
    width_95 = intervals_95[0].upper - intervals_95[0].lower
    width_99 = intervals_99[0].upper - intervals_99[0].lower
    assert width_99 > width_95


def test_different_metrics_have_different_values() -> None:
    """Different metrics should have independent corrections."""
    calibrator = ARESCalibrator()
    for i in range(50):
        calibrator.add_sample(
            CalibrationSample(
                run_id=f"run-{i}",
                automated_faithfulness=0.6,
                automated_retrieval_precision=0.7,
                automated_answer_correctness=0.5,
                gold_faithfulness=0.7,
                gold_retrieval_precision=0.8,
                gold_answer_correctness=0.6,
            )
        )

    intervals = calibrator.calibrate()

    # Extract by metric name
    results = {ci.metric: ci.point_estimate for ci in intervals}

    # All should be different (different corrections)
    assert results["faithfulness"] != results["retrieval_precision"]
    assert results["faithfulness"] != results["answer_correctness"]
    assert results["retrieval_precision"] != results["answer_correctness"]


def test_cli_calibrate_command(tmp_path: Path) -> None:
    """CLI calibrate command should produce output and report file."""
    samples_file = tmp_path / "samples.jsonl"
    _write_sample_file(samples_file, n_samples=50)

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["calibrate", str(samples_file)])

        assert result.exit_code == 0
        assert "faithfulness" in result.output.lower() or "ARES" in result.output
        assert Path("calibration_report.json").exists()

        # Verify report content
        report = json.loads(Path("calibration_report.json").read_text())
        assert len(report) == 4
        assert all("metric" in item for item in report)


def test_cli_calibrate_with_custom_confidence(tmp_path: Path) -> None:
    """CLI should accept custom confidence level."""
    samples_file = tmp_path / "samples.jsonl"
    _write_sample_file(samples_file, n_samples=50)

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["calibrate", str(samples_file), "--confidence", "0.99"])

        assert result.exit_code == 0
        assert "99%" in result.output or "0.99" in result.output


def test_cli_calibrate_insufficient_samples_error(tmp_path: Path) -> None:
    """CLI should report clear error when samples are insufficient."""
    samples_file = tmp_path / "samples.jsonl"
    _write_sample_file(samples_file, n_samples=10)

    result = runner.invoke(app, ["calibrate", str(samples_file)])

    assert result.exit_code == 1
    assert "at least 30" in result.output or "Calibration error" in result.output


def test_cli_calibrate_nonexistent_file_error(tmp_path: Path) -> None:
    """CLI should report clear error when file doesn't exist."""
    result = runner.invoke(app, ["calibrate", str(tmp_path / "nonexistent.jsonl")])

    assert result.exit_code == 1
    assert "Error loading samples" in result.output or "No such file" in result.output


def test_add_sample_incremental() -> None:
    """Samples should be addable incrementally."""
    calibrator = ARESCalibrator()
    assert len(calibrator._samples) == 0

    calibrator.add_sample(_sample("run-1"))
    assert len(calibrator._samples) == 1

    calibrator.add_sample(_sample("run-2"))
    assert len(calibrator._samples) == 2


def test_confidence_level_preserved_through_calibration() -> None:
    """Confidence level should be reflected in all intervals."""
    calibrator = ARESCalibrator(confidence_level=0.90)
    for i in range(50):
        calibrator.add_sample(_sample(f"run-{i}"))

    intervals = calibrator.calibrate()

    assert all(ci.confidence_level == 0.90 for ci in intervals)
