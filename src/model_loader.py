
import joblib
from sklearn.pipeline import Pipeline

from src.config import MODEL_PATH


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    if not isinstance(model, Pipeline):
        raise TypeError("Loaded model is not a Pipeline.")

    return model
