from src.prediction_history import (
    record_prediction,
    get_recent_predictions,
    get_history_summary,
)


def test_prediction_history_round_trip(tmp_path, monkeypatch):
    import src.prediction_history as history

    monkeypatch.setattr(history, "DB_PATH", tmp_path / "test_history.db")

    result = {
        "prediction": 1,
        "probability_class_0": 0.2,
        "probability_class_1": 0.8,
    }
    record_prediction(result, "test", 12.5)

    rows = get_recent_predictions()
    summary = get_history_summary()

    assert len(rows) == 1
    assert rows[0]["operation"] == "test"
    assert rows[0]["prediction"] == 1
    assert rows[0]["latency_ms"] == 12.5
    assert summary["total_predictions"] == 1
    assert summary["class_1_count"] == 1
    assert summary["class_0_count"] == 0
