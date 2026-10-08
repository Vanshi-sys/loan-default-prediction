
from src.model_loader import load_model
from src.features import build_model_input


def predict_loan(raw_input: dict):

    model_input = build_model_input(raw_input)
    model = load_model()

    prediction = model.predict(model_input)[0]
    probabilities = model.predict_proba(model_input)[0]

    return {
        "prediction": int(prediction),
        "probability_class_0": float(probabilities[0]),
        "probability_class_1": float(probabilities[1])
    }
