from datetime import datetime

import re
import pandas as pd

from src.config import MODEL_INPUT_COLUMNS, MEDIAN_VALUES


EMP_LENGTH_MAP = {
    "< 1 year": 0.5,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 10
}


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except:
        return False


def to_float(value):
    if is_missing(value):
        return None
    return float(value)


def parse_percent(value):
    if is_missing(value):
        return None
    return float(str(value).strip().replace("%", ""))


def normalize_term(value):
    if is_missing(value):
        raise ValueError("term is required.")

    value = str(value).strip()

    if value in {"36", "36 months"}:
        return " 36 months"

    if value in {"60", "60 months"}:
        return " 60 months"

    raise ValueError("term must be 36 months or 60 months.")


def normalize_home_ownership(value):
    if is_missing(value):
        raise ValueError("home_ownership is required.")

    value = str(value).strip().upper()

    if value in {"NONE", "OTHER"}:
        return "OTHER"

    return value


def normalize_zip(value):
    if is_missing(value):
        raise ValueError("zip_code is required.")

    value = str(value).strip().lower()

    if value.endswith("xx"):
        value = value[:-2]

    if not value.isdigit() or len(value) != 3:
        raise ValueError(
            "zip_code must be a 3-digit ZIP prefix, such as 945."
        )

    return value + "xx"


def parse_emp_length(value):
    if is_missing(value):
        return MEDIAN_VALUES["emp_length"]

    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip()

    if value in EMP_LENGTH_MAP:
        return EMP_LENGTH_MAP[value]

    raise ValueError(f"Invalid employment length: {value}")


def value_or_median(value, field):
    if is_missing(value):
        return MEDIAN_VALUES[field]
    return value


def get_date(value, field):
    if is_missing(value):
        raise ValueError(f"{field} is required.")

    # Handle pandas/Python datetime objects directly
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value)

    value = str(value).strip()

    # Expected app format: Dec-11
    try:
        return pd.to_datetime(
            value,
            format="%b-%y",
            errors="raise"
        )
    except Exception:
        # Fallback for standard date strings such as 2011-12-01
        return pd.to_datetime(
            value,
            errors="raise"
        )

def build_model_input(raw_input: dict):

    data = {}

    data["loan_amnt"] = to_float(raw_input.get("loan_amnt"))
    data["funded_amnt"] = to_float(raw_input.get("funded_amnt"))
    data["funded_amnt_inv"] = to_float(raw_input.get("funded_amnt_inv"))

    data["term"] = normalize_term(raw_input.get("term"))

    data["int_rate"] = parse_percent(raw_input.get("int_rate"))
    data["installment"] = to_float(raw_input.get("installment"))

    data["sub_grade"] = str(raw_input.get("sub_grade")).strip()

    data["emp_length"] = parse_emp_length(
        raw_input.get("emp_length")
    )

    data["home_ownership"] = normalize_home_ownership(
        raw_input.get("home_ownership")
    )

    data["annual_inc"] = to_float(raw_input.get("annual_inc"))

    data["verification_status"] = str(
        raw_input.get("verification_status")
    ).strip()

    data["purpose"] = str(
        raw_input.get("purpose")
    ).strip()

    data["zip_code"] = normalize_zip(
        raw_input.get("zip_code")
    )

    data["addr_state"] = str(
        raw_input.get("addr_state")
    ).strip().upper()

    data["dti"] = to_float(raw_input.get("dti"))
    data["delinq_2yrs"] = to_float(raw_input.get("delinq_2yrs"))
    data["inq_last_6mths"] = to_float(raw_input.get("inq_last_6mths"))

    data["mths_since_last_delinq"] = value_or_median(
        to_float(raw_input.get("mths_since_last_delinq")),
        "mths_since_last_delinq"
    )

    data["mths_since_last_record"] = value_or_median(
        to_float(raw_input.get("mths_since_last_record")),
        "mths_since_last_record"
    )

    data["open_acc"] = to_float(raw_input.get("open_acc"))
    data["pub_rec"] = to_float(raw_input.get("pub_rec"))
    data["revol_bal"] = to_float(raw_input.get("revol_bal"))

    data["revol_util"] = value_or_median(
        parse_percent(raw_input.get("revol_util")),
        "revol_util"
    )

    data["total_acc"] = to_float(raw_input.get("total_acc"))

    data["pub_rec_bankruptcies"] = value_or_median(
        to_float(raw_input.get("pub_rec_bankruptcies")),
        "pub_rec_bankruptcies"
    )

    # Dates
    issue_date = get_date(
        raw_input.get("issue_d"),
        "issue_d"
    )

    earliest_date = get_date(
        raw_input.get("earliest_cr_line"),
        "earliest_cr_line"
    )

    data["issue_year"] = issue_date.year
    data["issue_month"] = issue_date.month

    data["credit_history_months"] = (
        (issue_date.year - earliest_date.year) * 12
        + (issue_date.month - earliest_date.month)
    )

    # Missing indicators
    data["mths_since_last_record_missing"] = int(
        is_missing(raw_input.get("mths_since_last_record"))
    )

    data["mths_since_last_delinq_missing"] = int(
        is_missing(raw_input.get("mths_since_last_delinq"))
    )

    data["emp_length_missing"] = int(
        is_missing(raw_input.get("emp_length"))
    )

    return pd.DataFrame([data])[MODEL_INPUT_COLUMNS]
