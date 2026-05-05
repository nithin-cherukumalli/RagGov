import pytest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from raggov.cli import app

runner = CliRunner()

@patch("raggov.cli._diagnosis_panel")
@patch("raggov.cli.console.print")
@patch("raggov.cli.DiagnosisEngine")
@patch("raggov.cli._load_run")
def test_cli_default_mode(mock_load_run, mock_engine, mock_print, mock_panel, tmp_path):
    run_file = tmp_path / "run.json"
    run_file.write_text("{}")
    
    mock_run_instance = MagicMock()
    mock_load_run.return_value = mock_run_instance
    
    mock_engine_instance = MagicMock()
    mock_engine.return_value = mock_engine_instance
    mock_diagnosis = MagicMock()
    mock_diagnosis.model_dump_json.return_value = "{}"
    mock_engine_instance.diagnose.return_value = mock_diagnosis
    
    result = runner.invoke(app, ["diagnose", str(run_file)])
    assert result.exit_code == 0
    
    # Check that mode defaults to external-enhanced
    called_config = mock_engine.call_args[1].get("config", mock_engine.call_args[0][0] if mock_engine.call_args[0] else {})
    assert called_config.get("mode") == "external-enhanced"

@patch("raggov.cli._diagnosis_panel")
@patch("raggov.cli.console.print")
@patch("raggov.cli.DiagnosisEngine")
@patch("raggov.cli._load_run")
def test_cli_mode_passed_to_engine(mock_load_run, mock_engine, mock_print, mock_panel, tmp_path):
    run_file = tmp_path / "run.json"
    run_file.write_text("{}")
    
    mock_run_instance = MagicMock()
    mock_load_run.return_value = mock_run_instance
    
    mock_engine_instance = MagicMock()
    mock_engine.return_value = mock_engine_instance
    mock_diagnosis = MagicMock()
    mock_diagnosis.model_dump_json.return_value = "{}"
    mock_engine_instance.diagnose.return_value = mock_diagnosis
    
    result = runner.invoke(app, ["diagnose", str(run_file), "--mode", "native"])
    assert result.exit_code == 0
        
    # Check that mode was passed to engine
    called_config = mock_engine.call_args[1].get("config", mock_engine.call_args[0][0] if mock_engine.call_args[0] else {})
    assert called_config.get("mode") == "native"

def test_cli_invalid_mode_rejected(tmp_path):
    run_file = tmp_path / "run.json"
    run_file.write_text("{}")
    
    result = runner.invoke(app, ["diagnose", str(run_file), "--mode", "invalid"])
    assert result.exit_code != 0

def test_cli_calibrated_mode_unimplemented(tmp_path):
    run_file = tmp_path / "run.json"
    run_file.write_text("{}")
    
    result = runner.invoke(app, ["diagnose", str(run_file), "--mode", "calibrated"])
    assert result.exit_code != 0
    assert "Not Implemented" in result.stdout

