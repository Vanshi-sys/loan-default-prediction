from pathlib import Path

import pytest

from src.monitoring import monitored_predict, check_model_health


def test_successful_prediction_logs_latency_without_raw_input(caplog):
    raw_input = {"borrower_secret": "PRIVATE_VALUE"}

    def fake_predict(data):
        assert data is raw_input
        return {"prediction": 1}

    with caplog.at_level("INFO", logger="loan_app"):
        result = monitored_predict(fake_predict, raw_input, "unit_test")

    assert result == {"prediction": 1}
    assert "prediction_success" in caplog.text
    assert "latency_ms=" in caplog.text
    assert "unit_test" in caplog.text
    assert "PRIVATE_VALUE" not in caplog.text


def test_prediction_error_is_logged_and_reraised(caplog):
    def failing_predict(_data):
        raise ValueError("test failure")

    with caplog.at_level("ERROR", logger="loan_app"):
        with pytest.raises(ValueError, match="test failure"):
            monitored_predict(failing_predict, {"x": 1}, "error_test")

    assert "prediction_error" in caplog.text
    assert "error_type=ValueError" in caplog.text
    assert "error_test" in caplog.text


def test_model_health_reports_missing_artifacts(tmp_path):
    health = check_model_health(tmp_path)

    assert health["healthy"] is False
    assert "loan_default_model.pkl" in health["missing_files"]
    assert "model_metadata.json" in health["missing_files"]


def test_model_health_passes_when_required_artifacts_exist(tmp_path):
    (tmp_path / "loan_default_model.pkl").write_bytes(b"test-placeholder")
    (tmp_path / "model_metadata.json").write_text("{}")

    health = check_model_health(tmp_path)

    assert health == {"healthy": True, "missing_files": []}
