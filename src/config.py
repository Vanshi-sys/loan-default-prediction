
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "loan_default_model.pkl"

MODEL_INPUT_COLUMNS = [
    "loan_amnt", "funded_amnt", "funded_amnt_inv", "term",
    "int_rate", "installment", "sub_grade", "emp_length",
    "home_ownership", "annual_inc", "verification_status",
    "purpose", "zip_code", "addr_state", "dti", "delinq_2yrs",
    "inq_last_6mths", "mths_since_last_delinq",
    "mths_since_last_record", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "pub_rec_bankruptcies",
    "credit_history_months", "issue_year", "issue_month",
    "mths_since_last_record_missing",
    "mths_since_last_delinq_missing", "emp_length_missing"
]

NUMERIC_FEATURES = [
    "loan_amnt", "funded_amnt", "funded_amnt_inv", "int_rate",
    "installment", "emp_length", "annual_inc", "dti",
    "delinq_2yrs", "inq_last_6mths", "mths_since_last_delinq",
    "mths_since_last_record", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "pub_rec_bankruptcies",
    "credit_history_months", "issue_year", "issue_month",
    "mths_since_last_record_missing",
    "mths_since_last_delinq_missing", "emp_length_missing"
]

CATEGORICAL_FEATURES = [
    "term", "sub_grade", "home_ownership",
    "verification_status", "purpose", "zip_code", "addr_state"
]

MEDIAN_VALUES = {
    "mths_since_last_record": 90.0,
    "mths_since_last_delinq": 34.0,
    "emp_length": 4.0,
    "pub_rec_bankruptcies": 0.0,
    "revol_util": 49.1
}
