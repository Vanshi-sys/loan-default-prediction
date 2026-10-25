import math
import py_compile
from pathlib import Path

from src.predict import predict_loan
from src.shap_explainer import explain_prediction


PROJECT = Path(__file__).resolve().parents[1]

SAMPLE = {
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


def test_app_syntax():
    py_compile.compile(str(PROJECT / "app.py"), doraise=True)


def test_prediction_output_is_valid():
    result = predict_loan(SAMPLE)

    assert result["prediction"] in (0, 1)

    p0 = result["probability_class_0"]
    p1 = result["probability_class_1"]

    assert math.isfinite(p0)
    assert math.isfinite(p1)
    assert 0 <= p0 <= 1
    assert 0 <= p1 <= 1
    assert abs(p0 + p1 - 1) < 1e-5


def test_shap_explanation_schema():
    explanation = explain_prediction(SAMPLE)

    assert not explanation.empty
    assert {"feature", "shap_value"}.issubset(explanation.columns)
    assert explanation["shap_value"].notna().all()
