
import streamlit as st
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
import json

APP_DIR = Path(__file__).resolve().parent
with open(APP_DIR / "model_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)



st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💳",
    layout="wide"
)


st.markdown("""
<style>
/* LOAN_RISK_FUTURISTIC_THEME */
:root {
    --plum: #26162F;
    --rose: #F2A7C6;
    --gold: #E9C58B;
    --teal: #77D5CB;
}
.stApp {
    background:
        radial-gradient(ellipse at 8% 5%, #64314e55, transparent 35%),
        radial-gradient(ellipse at 92% 20%, #167e8050, transparent 30%),
        linear-gradient(145deg, #100f1c 0%, #201529 48%, #102427 100%);
    color: #FAF2F5;
}
[data-testid="stHeader"] {
    background: transparent;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #21152f, #14292d);
    border-right: 1px solid #e9c58b40;
}
[data-testid="stSidebar"] * {
    color: #FAF2F5;
}
h1, h2, h3 {
    color: #FFF0F5 !important;
    letter-spacing: -0.035em;
}
p, label, [data-testid="stCaptionContainer"] {
    color: #DCCBD9;
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff12, #f2a7c60a);
    border: 1px solid #f2a7c655;
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 12px 32px #08061135, inset 0 1px #ffffff12;
    transition: transform .22s ease, border-color .22s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    border-color: #e9c58b;
}
[data-testid="stMetricLabel"] p {
    color: #E9C58B !important;
}
[data-testid="stMetricValue"] {
    color: #FFF5FA !important;
}
div[data-testid="stForm"],
div[data-testid="stDataFrame"],
div[data-testid="stTable"] {
    border: 1px solid #e9c58b35;
    border-radius: 16px;
    overflow: hidden;
}
.stButton > button, .stFormSubmitButton > button {
    border: 1px solid #E9C58B !important;
    border-radius: 12px !important;
    background: linear-gradient(120deg, #713858, #28615F) !important;
    color: white !important;
    font-weight: 650 !important;
    box-shadow: 0 6px 22px #e9c58b20;
    transition: transform .2s ease, box-shadow .2s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px #f2a7c630;
    border-color: #F2A7C6 !important;
}
[data-testid="stRadio"] label {
    border-radius: 10px;
}
hr {
    border-color: #e9c58b35;
}
[data-testid="stAlert"] {
    border-radius: 14px;
}
div[data-testid="stPlotlyChart"],
div[data-testid="stVegaLiteChart"] {
    border-radius: 16px;
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        transition: none !important;
        animation: none !important;
    }
}
</style>
""", unsafe_allow_html=True)


st.title("Loan Risk Intelligence")
st.caption("XGBoost · Explainable AI · Model Evaluation")


st.markdown("""
<!-- LOAN_RISK_FUTURISTIC_HERO -->
<div style="
    position:relative; overflow:hidden; padding:28px 30px;
    margin:10px 0 24px; border-radius:24px;
    border:1px solid #E9C58B66;
    background:linear-gradient(115deg,#39203ddd,#1c3b3bdd 75%);
    box-shadow:0 18px 48px #08061155,inset 0 1px #ffffff20;">
  <div style="position:absolute; width:190px; height:190px; right:5%;
      top:-95px; border-radius:50%;
      background:radial-gradient(circle at 35% 30%,#ffd8e9,#b05b9170 40%,transparent 72%);
      filter:blur(1px); box-shadow:0 0 55px #f2a7c640;"></div>
  <div style="position:absolute; width:115px; height:115px; right:19%;
      bottom:-75px; border-radius:50%;
      border:1px solid #77D5CB88;
      box-shadow:inset -12px -10px 24px #77d5cb20,0 0 35px #77d5cb30;
      transform:rotateX(55deg) rotateZ(-25deg);"></div>
  <div style="position:relative;z-index:1;">
    <div style="font-size:11px;letter-spacing:3px;color:#E9C58B;
        font-weight:700;margin-bottom:12px;">
        ✦ FINANCIAL INTELLIGENCE / INDIA
    </div>
    <div style="font-size:clamp(27px,4vw,43px);font-weight:800;
        line-height:1.1;color:#FFF2F7;max-width:650px;">
        Risk, decoded.
        <span style="color:#F2A7C6;">Decisions, illuminated.</span>
    </div>
    <div style="font-size:14px;color:#E5D7E4;margin-top:14px;
        max-width:610px;line-height:1.7;">
        Explainable machine learning, loan analytics and prediction
        monitoring — bringing a little royal colour to responsible AI.
    </div>
    <div style="display:flex;gap:9px;flex-wrap:wrap;margin-top:19px;">
      <span style="padding:6px 11px;border-radius:30px;
          border:1px solid #F2A7C655;color:#FFE8F1;font-size:11px;">
          ✧ XGBoost
      </span>
      <span style="padding:6px 11px;border-radius:30px;
          border:1px solid #77D5CB66;color:#D8FFF7;font-size:11px;">
          ◈ Explainable AI
      </span>
      <span style="padding:6px 11px;border-radius:30px;
          border:1px solid #E9C58B66;color:#FFE8BB;font-size:11px;">
          ✦ Model monitoring
      </span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.title("Loan Analytics")
    page = st.radio(
        "Navigation",
        ["Overview", "Loan Prediction", "Model Insights", "Prediction Monitoring"],
        index=0
    )

    health = check_model_health(Path(__file__).resolve().parent)
    if health["healthy"]:
        st.success("Model files: available")
    else:
        st.error("Model files missing: " + ", ".join(health["missing_files"]))
    st.divider()
    st.caption("Model: XGBoost")
    st.caption("Validation ROC-AUC: 0.7191")

if page == "Overview":
    st.subheader("Project Overview")
    st.write(
        "An interactive machine-learning application for exploring "
        "loan-related predictions and understanding model behavior."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Algorithm", metadata["model"])
    c2.metric("Input Features", metadata["input_features"])
    c3.metric("Transformed Features", metadata["transformed_features"])
    c4.metric("Validation ROC-AUC", f'{metadata["validation_roc_auc"]:.4f}')

    c1, c2 = st.columns(2)
    c1.metric("Validation PR-AUC", f'{metadata["validation_pr_auc"]:.4f}')
    c2.info("Choose Loan Prediction in the sidebar to test an input.")


    # LOAN_RISK_OVERVIEW_CHARTS
    st.markdown("### Model performance at a glance")
    st.caption(
        "Validation metrics from the saved model metadata. "
        "These are evaluation results, not live prediction counts."
    )
    chart_left, chart_right = st.columns([1.15, 1])

    with chart_left:
        st.markdown("#### Validation metrics")
        metrics_df = pd.DataFrame({
            "Metric": ["ROC-AUC", "PR-AUC"],
            "Score": [
                float(metadata["validation_roc_auc"]),
                float(metadata["validation_pr_auc"])
            ]
        }).set_index("Metric")
        st.bar_chart(metrics_df, y="Score", height=250)

    with chart_right:
        st.markdown("#### Feature pipeline")
        features_df = pd.DataFrame({
            "Stage": ["Input features", "Transformed features"],
            "Count": [
                int(metadata["input_features"]),
                int(metadata["transformed_features"])
            ]
        }).set_index("Stage")
        st.bar_chart(features_df, y="Count", height=250)
        st.caption(
            "Categorical preprocessing expands the 31 raw inputs "
            "into 915 transformed model features."
        )

elif page == "Model Insights":
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


elif page == "Prediction Monitoring":
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

else:
    st.subheader("Loan Prediction")
    st.info(
        "Enter loan and borrower details below. Predictions are educational "
        "model outputs, not lending recommendations."
    )

    if st.button("Try Sample Loan", type="secondary"):
        st.session_state["use_sample_loan"] = True

    if st.session_state.pop("use_sample_loan", False):
        sample_input = {
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
        try:
            demo_result = predict_and_record(predict_loan, sample_input, "sample")
            st.success("Sample prediction generated")
            a, b, c = st.columns(3)
            a.metric("Predicted class", str(demo_result["prediction"]))
            b.metric("Class 0 probability",
                     f'{demo_result["probability_class_0"]:.2%}')
            c.metric("Class 1 probability",
                     f'{demo_result["probability_class_1"]:.2%}')
            demo_explanation = explain_prediction(sample_input)
            st.subheader("Sample SHAP explanation")
            st.dataframe(
                demo_explanation[["feature", "shap_value"]],
                use_container_width=True,
                hide_index=True
            )
            st.caption(
                "Demonstration only. The encoded class mapping has not "
                "been independently verified."
            )
        except Exception as exc:
            st.error(f"Sample prediction failed: {exc}")

    with st.form("loan_form"):

        st.subheader("Loan Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            loan_amnt = st.number_input(
                "Loan Amount",
                min_value=0.0,
                value=10000.0
            )

            funded_amnt = st.number_input(
                "Funded Amount",
                min_value=0.0,
                value=10000.0
            )

            funded_amnt_inv = st.number_input(
                "Funded Amount Inv.",
                min_value=0.0,
                value=10000.0
            )

            term = st.selectbox(
                "Term",
                ["36 months", "60 months"]
            )

            int_rate = st.number_input(
                "Interest Rate (%)",
                min_value=0.0,
                max_value=40.0,
                value=12.0
            )

            installment = st.number_input(
                "Installment",
                min_value=0.0,
                value=300.0
            )

        with col2:

            sub_grade = st.selectbox(
                "Sub Grade",
                [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]
            )

            emp_length = st.selectbox(
                "Employment Length",
                [
                    "< 1 year", "1 year", "2 years", "3 years",
                    "4 years", "5 years", "6 years", "7 years",
                    "8 years", "9 years", "10+ years"
                ]
            )

            home_ownership = st.selectbox(
                "Home Ownership",
                ["RENT", "MORTGAGE", "OWN", "OTHER"]
            )

            annual_inc = st.number_input(
                "Annual Income",
                min_value=0.0,
                value=60000.0
            )

            verification_status = st.selectbox(
                "Verification Status",
                ["Not Verified", "Source Verified", "Verified"]
            )

            purpose = st.selectbox(
                "Purpose",
                [
                    "credit_card",
                    "car",
                    "small_business",
                    "other",
                    "wedding",
                    "debt_consolidation",
                    "home_improvement",
                    "major_purchase",
                    "medical",
                    "moving",
                    "vacation",
                    "house",
                    "renewable_energy"
                ]
            )

        with col3:

            zip_code = st.text_input(
                "ZIP Code Prefix",
                value="945"
            )

            addr_state = st.selectbox(
                "State",
                [
                    "CA","NY","TX","FL","IL","NJ","PA","GA",
                    "OH","MI","NC","VA","AZ","WA","MA","MD",
                    "CO","MN","MO","OR","WI","AL","SC","CT",
                    "KY","OK","LA","UT","IA","NV","AR","KS",
                    "NM","MS","NE","WV","NH","HI","RI","ME",
                    "DE","AK","MT","VT","WY","SD","TN","DC",
                    "ID"
                ]
            )

            dti = st.number_input(
                "Debt-to-Income Ratio",
                min_value=0.0,
                value=13.0
            )

            delinq_2yrs = st.number_input(
                "Delinquencies (2 yrs)",
                min_value=0.0,
                value=0.0
            )

            inq_last_6mths = st.number_input(
                "Inquiries (6 months)",
                min_value=0.0,
                value=0.0
            )

            open_acc = st.number_input(
                "Open Accounts",
                min_value=0.0,
                value=10.0
            )

            pub_rec = st.number_input(
                "Public Records",
                min_value=0.0,
                value=0.0
            )

        st.subheader("Credit History")

        col1, col2, col3 = st.columns(3)

        with col1:
            mths_since_last_delinq = st.number_input(
                "Months Since Last Delinquency",
                min_value=0.0,
                value=34.0
            )

            mths_since_last_record = st.number_input(
                "Months Since Last Record",
                min_value=0.0,
                value=90.0
            )

        with col2:
            revol_bal = st.number_input(
                "Revolving Balance",
                min_value=0.0,
                value=5000.0
            )

            revol_util = st.number_input(
                "Revolving Utilization (%)",
                min_value=0.0,
                max_value=150.0,
                value=49.1
            )

        with col3:
            total_acc = st.number_input(
                "Total Accounts",
                min_value=0.0,
                value=20.0
            )

            pub_rec_bankruptcies = st.number_input(
                "Public Record Bankruptcies",
                min_value=0.0,
                value=0.0
            )

        st.subheader("Dates")

        col1, col2 = st.columns(2)

        with col1:
            issue_date = st.date_input(
                "Loan Issue Date",
                value=date(2011, 12, 1),
                min_value=date(2007, 6, 1),
                max_value=date(2011, 12, 1)
            )

        with col2:
            earliest_credit_date = st.date_input(
                "Earliest Credit Line",
                value=date(2000, 1, 1),
                min_value=date(1950, 1, 1),
                max_value=issue_date
            )

        submit = st.form_submit_button(
            "🔮 Predict Loan Risk",
            use_container_width=True
        )


    if submit:

        raw_input = {
            "loan_amnt": loan_amnt,
            "funded_amnt": funded_amnt,
            "funded_amnt_inv": funded_amnt_inv,
            "term": term,
            "int_rate": int_rate,
            "installment": installment,
            "sub_grade": sub_grade,
            "emp_length": emp_length,
            "home_ownership": home_ownership,
            "annual_inc": annual_inc,
            "verification_status": verification_status,
            "purpose": purpose,
            "zip_code": zip_code,
            "addr_state": addr_state,
            "dti": dti,
            "delinq_2yrs": delinq_2yrs,
            "inq_last_6mths": inq_last_6mths,
            "mths_since_last_delinq": mths_since_last_delinq,
            "mths_since_last_record": mths_since_last_record,
            "open_acc": open_acc,
            "pub_rec": pub_rec,
            "revol_bal": revol_bal,
            "revol_util": revol_util,
            "total_acc": total_acc,
            "pub_rec_bankruptcies": pub_rec_bankruptcies,
            "issue_d": issue_date.strftime("%b-%y"),
            "earliest_cr_line": earliest_credit_date.strftime("%b-%y")
        }

        try:

            import re
            if not re.fullmatch(r"\d{3}", str(zip_code).strip()):
                st.error("Enter a ZIP code prefix containing exactly 3 digits.")
                st.stop()

            result = predict_and_record(predict_loan, raw_input, "manual")

            probability_0 = result["probability_class_0"]
            probability_1 = result["probability_class_1"]

            st.divider()
            st.subheader("Prediction Result")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Prediction Class",
                    str(result["prediction"])
                )

            with col2:
                st.metric(
                    "Class 0 Probability",
                    f"{probability_0:.2%}"
                )

            with col3:
                st.metric(
                    "Class 1 Probability",
                    f"{probability_1:.2%}"
                )

            st.progress(
                probability_1,
                text=f"Class 1 probability: {probability_1:.2%}"
            )

            st.caption(
                "Class labels represent the dataset's encoded target values. "
                "The original mapping of class 1 to confirmed default has not "
                "been independently verified."
            )

            # ----------------------------------------------------
            # SHAP
            # ----------------------------------------------------

            st.divider()
            st.subheader("🔎 Prediction Explanation")

            explanation = explain_prediction(raw_input)

            chart_data = explanation.sort_values(
                "shap_value"
            ).copy()

            chart_data["display_feature"] = (
                chart_data["feature"]
                .str.replace("num__", "", regex=False)
                .str.replace("cat__", "", regex=False)
            )

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.barh(
                chart_data["display_feature"],
                chart_data["shap_value"]
            )

            ax.axvline(0, linewidth=1)
            ax.set_xlabel("SHAP contribution")
            ax.set_ylabel("Feature")
            ax.set_title("Top Feature Contributions")

            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            st.dataframe(
                explanation[
                    ["feature", "shap_value"]
                ],
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:

            st.error(f"Prediction failed: {e}")
