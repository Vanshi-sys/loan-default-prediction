
import shap
import pandas as pd

from src.model_loader import load_model
from src.features import build_model_input


def explain_prediction(raw_input: dict):

    model = load_model()
    model_input = build_model_input(raw_input)

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed = preprocessor.transform(model_input)
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed_df = pd.DataFrame(
        transformed,
        columns=feature_names
    )

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed_df)

    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0]

    explanation = pd.DataFrame({
        "feature": feature_names,
        "shap_value": values,
        "absolute_shap": abs(values),
        "feature_value": transformed_df.iloc[0].values
    })

    return (
        explanation
        .sort_values("absolute_shap", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
