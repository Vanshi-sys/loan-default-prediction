import logging

import streamlit as st
import sys
from pathlib import Path

DESIGN_SYSTEM_DIR = Path(__file__).resolve().parent / "parallel_work" / "step3_design"
if str(DESIGN_SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(DESIGN_SYSTEM_DIR))
import sys
from loan_design_system import (
    render_sidebar,
    discover_model_status,
    page_header,
    card,
    metric_card,
    callout,
    status_pill,
    surface,
    apply_plotly_theme,
    apply_matplotlib_theme,
)

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

from src.predict import predict_loan
from src.monitoring import monitored_predict, check_model_health
from src.prediction_history import (
    predict_and_record, get_recent_predictions, get_history_summary
)
from src.shap_explainer import explain_prediction
from src.ui_theme import hero_banner, apply_theme
import json
from src.overview_page import render_overview
from src.prediction_page import render_prediction_page
from src.prediction_history import record_prediction

APP_DIR = Path(__file__).resolve().parent
with open(APP_DIR / "model_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)



st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💳",
    layout="wide"
)

apply_theme()

# Step 3 design system: navigation, model status, and theme
active_page = render_sidebar(model_status=discover_model_status(APP_DIR))




st.title("Loan Risk Intelligence")
st.caption("XGBoost · Explainable AI · Model Evaluation")


hero_banner(
    "Explainable credit-risk analytics",
    "Risk, clearly explained.",
    "Machine-learning predictions, SHAP explanations and prediction monitoring in one workspace.",
)




# Redesigned Prediction page callbacks.
_prediction_logger = logging.getLogger("loan_app")

def _record_redesigned_prediction(inputs, out):
    """Record the already-computed result; never run the model twice."""
    normalized = out.get("normalized") or {}
    predicted = normalized.get("predicted_class")
    probabilities = normalized.get("probabilities") or []

    if predicted is None:
        _prediction_logger.warning(
            "Prediction was generated, but history was skipped: "
            "no numeric predicted class was returned."
        )
        return

    try:
        predicted_int = int(predicted)
        if str(predicted_int) != str(predicted).strip():
            raise ValueError("Predicted class is not an integer class label.")
    except (TypeError, ValueError):
        _prediction_logger.warning(
            "Prediction was generated, but history was skipped: "
            "the returned class could not be safely stored."
        )
        return

    found = {}
    for label, value in probabilities:
        key = str(label).strip().lower()
        if key in ("0", "class 0", "probability_class_0"):
            found[0] = value
        elif key in ("1", "class 1", "probability_class_1"):
            found[1] = value
        elif key.endswith("_0") and "probability" in key:
            found[0] = value
        elif key.endswith("_1") and "probability" in key:
            found[1] = value

    try:
        p0 = float(found[0])
        p1 = float(found[1])
        latency = max(0.0, float(out.get("latency_ms", 0.0)))
        if not (0.0 <= p0 <= 1.0 and 0.0 <= p1 <= 1.0):
            raise ValueError("Probabilities are outside the expected range.")
    except (KeyError, TypeError, ValueError):
        _prediction_logger.warning(
            "Prediction was generated, but history was skipped: "
            "both class probabilities were not safely identified."
        )
        return

    record_prediction({
        "prediction": predicted_int,
        "probability_class_0": p0,
        "probability_class_1": p1,
    }, "manual", latency)

def _log_redesigned_prediction_error(inputs, exc):
    _prediction_logger.error(
        "event=prediction_error operation=manual error_type=%s",
        type(exc).__name__,
    )

_prediction_sample_loan = {
    "loan_amnt": 10000.0,
    "funded_amnt": 10000.0,
    "funded_amnt_inv": 10000.0,
    "term": "36 months",
    "int_rate": 12.0,
    "installment": 300.0,
    "sub_grade": "B3",
    "emp_length": "5 years",
    "home_ownership": "RENT",
    "annual_inc": 60000.0,
    "verification_status": "Not Verified",
    "purpose": "debt_consolidation",
    "zip_code": "945",
    "addr_state": "CA",
    "dti": 13.0,
    "delinq_2yrs": 0.0,
    "inq_last_6mths": 0.0,
    "mths_since_last_delinq": 34.0,
    "mths_since_last_record": 90.0,
    "open_acc": 10.0,
    "pub_rec": 0.0,
    "revol_bal": 5000.0,
    "revol_util": 49.1,
    "total_acc": 20.0,
    "pub_rec_bankruptcies": 0.0,
    "issue_d": "Dec-11",
    "earliest_cr_line": "Jan-00",
}


if active_page == "overview":
    render_overview(metadata=metadata)

elif active_page == "insights":
    st.subheader("Model Insights")
    c1, c2 = st.columns(2)
    c1.metric("Validation ROC-AUC", f'{metadata["validation_roc_auc"]:.4f}')
    c2.metric("Validation PR-AUC", f'{metadata["validation_pr_auc"]:.4f}')

    st.write("Model configuration")
    st.dataframe(
        [
            {"Property": "Algorithm", "Value": metadata["model"]},
            {"Property": "Task", "Value": metadata["task"]},
            {"Property": "Input features", "Value": metadata["input_features"]},
            {"Property": "Transformed features", "Value": metadata["transformed_features"]},
        ],
        hide_index=True,
        use_container_width=True
    )

    st.markdown(
        "- ROC-AUC measures ranking ability across classification thresholds.\\n"
        "- PR-AUC summarizes the precision-recall trade-off.\\n"
        "- Validation metrics do not guarantee future performance."
    )
    st.warning(
        "The encoded target mapping has not been independently verified. "
        "Class 1 must not automatically be interpreted as confirmed default."
    )


elif active_page == "monitoring":
    st.subheader("Prediction Monitoring")
    st.caption(
        "Operational history from this app instance. Raw borrower inputs "
        "are not stored. Local history may reset after a restart or redeploy."
    )

    summary = get_history_summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predictions", summary["total_predictions"])
    c2.metric("Class 0 Results", summary["class_0_count"])
    c3.metric("Class 1 Results", summary["class_1_count"])
    c4.metric("Average Latency", f'{summary["average_latency_ms"]:.1f} ms')

    st.metric(
        "Average Class 1 Probability",
        f'{summary["average_class_1_probability"]:.2%}'
    )

    recent = get_recent_predictions(100)
    if recent:
        history_df = pd.DataFrame(recent)
        history_df = history_df.rename(columns={
            "timestamp_utc": "Timestamp (UTC)",
            "operation": "Operation",
            "prediction": "Predicted Class",
            "probability_class_0": "Class 0 Probability",
            "probability_class_1": "Class 1 Probability",
            "latency_ms": "Latency (ms)",
        })
        st.subheader("Recent Predictions")
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        chart_df = history_df[["Operation", "Predicted Class"]].copy()
        chart_df["Predicted Class"] = chart_df["Predicted Class"].astype(str)
        counts = chart_df.groupby(
            ["Operation", "Predicted Class"]
        ).size().unstack(fill_value=0)
        st.subheader("Prediction Class Distribution")
        st.bar_chart(counts)
    else:
        st.info(
            "No predictions recorded yet. Generate a sample or manual "
            "prediction to populate this dashboard."
        )

elif active_page == "prediction":
    render_prediction_page(
        predict_fn=predict_loan,
        shap_fn=explain_prediction,
        sample_loan=_prediction_sample_loan,
        on_result=_record_redesigned_prediction,
        on_error=_log_redesigned_prediction_error,
        show_title=False,
    )
