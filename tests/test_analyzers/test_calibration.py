"""
Tests for claim verification calibration.
"""

import json
import pytest
import numpy as np
from pathlib import Path
from raggov.calibration.claim_calibration import ClaimCalibrationLoader, CalibrationMode, ClaimCalibrationModel
from evals.claim_grounding.train_calibration import compute_calibration_metrics


def test_ece_calculation_toy_data():
    # perfect calibration
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.1, 0.9, 0.9])
    metrics = compute_calibration_metrics(y_true, y_prob, n_bins=2)
    # Bin 1: [0, 0.5] -> acc=0, conf=0.1 -> |0.1-0| = 0.1
    # Bin 2: [0.5, 1.0] -> acc=1, conf=0.9 -> |0.9-1| = 0.1
    # Weighted avg: 0.5*0.1 + 0.5*0.1 = 0.1
    assert metrics["ece"] == pytest.approx(0.1)


def test_loader_no_file_returns_uncalibrated():
    loader = ClaimCalibrationLoader()
    model = loader.load("non_existent_file.json")
    assert model.mode == CalibrationMode.NONE
    
    res = model.calibrate({"raw_score": 0.8, "label": "entailed"})
    assert res.confidence is None
    assert res.status == "uncalibrated"


def test_reliability_table_calibration():
    params = {
        "table": {
            "mock_verifier:false:entailed": 0.95,
            "mock_verifier:true:entailed": 0.60,
            "default:unsupported": 0.05
        }
    }
    model = ClaimCalibrationModel(CalibrationMode.RELIABILITY_TABLE, params)
    
    # Case 1: normal verifier
    res1 = model.calibrate({
        "raw_score": 0.8, 
        "label": "entailed", 
        "verifier_mode": "mock_verifier",
        "fallback_used": False
    })
    assert res1.confidence == 0.95
    assert res1.status == "calibrated"
    
    # Case 2: fallback used (lower confidence)
    res2 = model.calibrate({
        "raw_score": 0.8, 
        "label": "entailed", 
        "verifier_mode": "mock_verifier",
        "fallback_used": True
    })
    assert res2.confidence == 0.60
    
    # Case 3: default fallback
    res3 = model.calibrate({"label": "unsupported"})
    assert res3.confidence == 0.05


def test_temperature_scaling_calibration():
    # logit(0.8) = ln(4) = 1.386
    # if T=2.0, new_logit = 0.693
    # 1 / (1 + e^-0.693) = 1 / (1 + 0.5) = 0.666
    params = {"temperature": 2.0}
    model = ClaimCalibrationModel(CalibrationMode.TEMPERATURE_SCALING, params)
    
    res = model.calibrate({"raw_score": 0.8, "label": "entailed"})
    assert res.confidence == pytest.approx(0.666, abs=0.01)


def test_calibration_artifact_roundtrip(tmp_path):
    artifact_path = tmp_path / "calib.json"
    artifact = {
        "mode": "temperature_scaling",
        "params": {"temperature": 1.5},
        "metadata": {"version": "v1"}
    }
    artifact_path.write_text(json.dumps(artifact))
    
    model = ClaimCalibrationLoader.load(artifact_path)
    assert model.mode == CalibrationMode.TEMPERATURE_SCALING
    assert model.params["temperature"] == 1.5
    assert "calib.json" in model.metadata["source"]
