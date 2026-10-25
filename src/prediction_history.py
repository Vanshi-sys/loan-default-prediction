import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "prediction_history.db"


def _connect():
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_history():
    with _connect() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                operation TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                probability_class_0 REAL NOT NULL,
                probability_class_1 REAL NOT NULL,
                latency_ms REAL NOT NULL
            )
        """)


def record_prediction(result, operation, latency_ms):
    """Store prediction metadata only; never store raw borrower inputs."""
    initialize_history()
    with _connect() as connection:
        connection.execute("""
            INSERT INTO prediction_history (
                timestamp_utc, operation, prediction,
                probability_class_0, probability_class_1, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            str(operation),
            int(result["prediction"]),
            float(result["probability_class_0"]),
            float(result["probability_class_1"]),
            max(0.0, float(latency_ms)),
        ))


def predict_and_record(predict_function, raw_input, operation="prediction"):
    """Run existing monitored prediction and record its metadata."""
    from src.monitoring import monitored_predict

    start = time.perf_counter()
    result = monitored_predict(predict_function, raw_input, operation)
    latency_ms = (time.perf_counter() - start) * 1000
    record_prediction(result, operation, latency_ms)
    return result


def get_recent_predictions(limit=100):
    initialize_history()
    limit = max(1, min(int(limit), 1000))
    with _connect() as connection:
        rows = connection.execute("""
            SELECT timestamp_utc, operation, prediction,
                   probability_class_0, probability_class_1, latency_ms
            FROM prediction_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(row) for row in rows]


def get_history_summary():
    initialize_history()
    with _connect() as connection:
        row = connection.execute("""
            SELECT
                COUNT(*) AS total_predictions,
                SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) AS class_1_count,
                SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END) AS class_0_count,
                AVG(latency_ms) AS average_latency_ms,
                AVG(probability_class_1) AS average_class_1_probability
            FROM prediction_history
        """).fetchone()
    return {
        "total_predictions": int(row["total_predictions"] or 0),
        "class_0_count": int(row["class_0_count"] or 0),
        "class_1_count": int(row["class_1_count"] or 0),
        "average_latency_ms": float(row["average_latency_ms"] or 0),
        "average_class_1_probability": float(
            row["average_class_1_probability"] or 0
        ),
    }
