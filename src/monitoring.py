import logging
import time
from pathlib import Path

logger = logging.getLogger("loan_app")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def monitored_predict(predict_function, raw_input, operation="prediction"):
    """Measure prediction latency without logging borrower inputs."""
    start = time.perf_counter()
    try:
        result = predict_function(raw_input)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "event=prediction_success operation=%s latency_ms=%.2f",
            operation, elapsed_ms
        )
        return result
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error(
            "event=prediction_error operation=%s latency_ms=%.2f error_type=%s",
            operation, elapsed_ms, type(exc).__name__
        )
        raise


def check_model_health(project_dir):
    """Check required model artifacts without loading borrower data."""
    project_dir = Path(project_dir)
    required = [
        project_dir / "loan_default_model.pkl",
        project_dir / "model_metadata.json",
    ]
    missing = [str(path.name) for path in required if not path.is_file()]
    return {
        "healthy": not missing,
        "missing_files": missing,
    }
