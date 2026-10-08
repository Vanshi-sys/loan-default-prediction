
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

from src.predict import predict_loan
from src.shap_explainer import explain_prediction


st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Loan Default Prediction")
st.caption("XGBoost-based loan risk prediction with SHAP explainability")

st.info(
    "Enter the loan and borrower details below to generate a model prediction."
)

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

        result = predict_loan(raw_input)

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
