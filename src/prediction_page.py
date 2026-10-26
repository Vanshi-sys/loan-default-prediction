"""Step 5 - Loan Prediction page (UI layer only).

Self-contained Streamlit page module. It does NOT build a prediction pipeline: it
calls the existing prediction / SHAP functions of the repository (auto-discovered
in src.predict / src.shap_explainer, or injected by the caller) and presents
their output. It never retrains, replaces or re-implements the model.

Public API
----------
render_prediction_page(*, predict_fn=None, shap_fn=None, sample_loan=None, schema=None,
                       class_labels=None, on_result=None, on_error=None, show_title=True)
discover_backend(refresh=False) -> Backend
build_schema(model=None, sample=None, columns=None) -> list[dict]
get_input_schema() -> list[dict]
validate_inputs(values, schema) -> list[str]
soft_warnings(values, schema) -> list[str]
run_prediction(values, predict_fn=None, shap_fn=None, include_shap=True, backend=None, columns=None) -> dict
normalize_prediction(raw, classes=None) -> dict
normalize_shap(raw, predicted_class=None, classes=None) -> pandas.DataFrame | None
format_exact(value) -> str
"""
from __future__ import annotations

import ast
import dataclasses
import html
import importlib
import inspect
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

__all__ = [
    "render_prediction_page", "discover_backend", "build_schema", "get_input_schema",
    "validate_inputs", "soft_warnings", "run_prediction", "normalize_prediction",
    "normalize_shap", "format_exact", "Backend", "PAGE_VERSION",
]

PAGE_VERSION = "1.0.0"
DISCLAIMER = (
    "Predictions are statistical estimates produced by a machine-learning model. They are not "
    "guarantees of what will happen, and probabilities have not been established as calibrated risk "
    "estimates. This tool is for educational and demonstration purposes and must not be the sole "
    "basis for any lending, credit or financial decision."
)
MODEL_FILE = "loan_default_model.pkl"
METADATA_FILE = "model_metadata.json"

PREDICT_NAMES = ("predict_loan", "predict_single", "predict_one", "predict_default",
                 "make_prediction", "run_prediction", "predict")
SHAP_NAMES = ("explain_prediction", "explain_single_prediction", "get_shap_explanation",
              "get_shap_values", "compute_shap_values", "explain_instance", "shap_explanation", "explain")
LOADER_NAMES = ("load_model", "get_model", "load_pipeline", "load_model_artifacts", "load_artifacts")
LABEL_KEYS = ("class_labels", "label_mapping", "label_map", "class_mapping", "class_map",
              "target_mapping", "target_labels", "class_names", "target_names", "labels", "classes")

_CACHE: Dict[str, Any] = {}


# ----------------------------------------------------------------------------
# Repository location
# ----------------------------------------------------------------------------
def find_repo_root() -> Path:
    candidates: List[Path] = []
    env = os.environ.get("LOAN_REPO_ROOT")
    if env:
        candidates.append(Path(env))
    candidates.extend(Path(__file__).resolve().parents)
    candidates.append(Path.cwd())
    for c in candidates:
        if (c / "src").is_dir() and ((c / METADATA_FILE).exists() or (c / "app.py").exists()):
            return c
    return Path.cwd()


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _py(v: Any) -> Any:
    return v.item() if isinstance(v, np.generic) else v


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        r = pd.isna(v)
        return bool(r) if isinstance(r, (bool, np.bool_)) else False
    except Exception:
        return False


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_))


def _is_seq(v: Any) -> bool:
    return isinstance(v, (list, tuple, np.ndarray, pd.Series))


def _safe_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _stretch(fn) -> Dict[str, Any]:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if "use_container_width" in params:
        return {"use_container_width": True}
    if "width" in params:
        return {"width": "stretch"}
    return {}


def _box():
    try:
        if "border" in inspect.signature(st.container).parameters:
            return st.container(border=True)
    except Exception:
        pass
    return st.container()


def format_exact(v: Any) -> str:
    """Render a value the way the model returned it (no rounding, no re-scaling)."""
    if isinstance(v, np.floating):
        return str(v)
    if isinstance(v, np.generic):
        return str(v.item())
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _fraction(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except Exception:
        return None
    return f if 0.0 <= f <= 1.0 else None


_TOKEN_MAP = {
    "amnt": "Amount", "amt": "Amount", "inc": "Income", "int": "Interest", "dti": "DTI", "fico": "FICO",
    "acc": "Accounts", "accts": "Accounts", "mths": "Months", "mo": "Month", "yrs": "Years", "emp": "Employment",
    "delinq": "Delinquencies", "inq": "Inquiries", "pub": "Public", "rec": "Records", "revol": "Revolving",
    "util": "Utilization", "bal": "Balance", "mort": "Mortgage", "num": "Number", "tot": "Total",
    "bc": "Bankcard", "il": "Installment", "avg": "Average", "cur": "Current", "hi": "High", "cred": "Credit",
    "lim": "Limit", "pct": "Percent", "cr": "Credit", "d": "Date", "pymnt": "Payment", "verif": "Verification",
    "coll": "Collections", "sin": "Since", "rcnt": "Recent", "tl": "Trade Lines", "id": "ID", "ln": "Loan",
    "inst": "Installment", "lt": "Less Than", "gt": "Greater Than", "ver": "Verified",
}


def prettify_label(name: str) -> str:
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", str(name))
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", spaced) if t]
    return " ".join(_TOKEN_MAP.get(t.lower(), t.capitalize()) for t in tokens) or str(name)


_HELP_RULES: List[Tuple[str, str]] = [
    (r"loan_?amnt|loan_amount|^amount$", "Total amount of the loan requested."),
    (r"^term$", "Length of the loan, in the form used by the training data."),
    (r"int_rate|interest", "Interest rate on the loan, in the units used by the training data."),
    (r"installment", "Scheduled periodic payment amount."),
    (r"sub_grade", "Detailed loan grade assigned at origination."),
    (r"^grade$", "Loan grade assigned at origination."),
    (r"emp_length", "Length of the borrower's employment."),
    (r"emp_title", "Borrower's job title, as recorded."),
    (r"home_ownership", "Borrower's home-ownership status."),
    (r"annual_inc", "Borrower's annual income."),
    (r"verification", "Whether the borrower's income was verified."),
    (r"purpose", "Stated purpose of the loan."),
    (r"^dti", "Debt-to-income ratio: debt payments relative to income."),
    (r"fico", "FICO credit-score range value."),
    (r"delinq", "Number of delinquency events in the stated period."),
    (r"inq", "Number of credit inquiries."),
    (r"open_acc", "Number of open credit lines."),
    (r"total_acc", "Total number of credit lines on file."),
    (r"pub_rec", "Number of derogatory public records."),
    (r"revol_bal", "Total revolving credit balance."),
    (r"revol_util", "Revolving line utilization rate."),
    (r"mort_acc", "Number of mortgage accounts."),
    (r"addr_state|^state$", "State of residence."),
    (r"earliest_cr", "Date of the earliest reported credit line."),
    (r"application_type", "Whether the application is individual or joint."),
]


def field_help(name: str, kind: str) -> str:
    low = name.lower()
    for pattern, text in _HELP_RULES:
        if re.search(pattern, low):
            return text + "  \nModel field: `" + name + "`"
    base = "Choose the category that applies." if kind == "categorical" else "Enter the value in the same units as the model's training data."
    return base + "  \nModel field: `" + name + "`"


_LOAN_KEYS = {"loan", "amnt", "amount", "funded", "term", "installment", "int", "interest", "purpose",
              "grade", "issue", "initial", "application", "disbursement", "policy", "pymnt", "payment"}
_PROFILE_KEYS = {"emp", "employment", "home", "ownership", "annual", "income", "inc", "verification",
                 "verified", "addr", "zip", "state", "title", "age", "gender", "marital", "education",
                 "region", "occupation"}
_CREDIT_KEYS = {"fico", "credit", "cr", "delinq", "inq", "pub", "rec", "bankruptcies", "collections",
                "chargeoff", "tax", "lien", "acc", "accts", "mort", "open", "revol", "util", "bal", "dti",
                "il", "bc", "num", "mths", "mo", "sin", "pct", "tot", "avg", "max", "lim", "tl", "debt", "total"}
SECTION_ORDER = ["Loan details", "Borrower profile", "Credit & debt history", "Additional details"]


def section_for(name: str) -> str:
    tokens = set(t for t in re.split(r"[^a-z0-9]+", name.lower()) if t)
    if tokens & _LOAN_KEYS:
        return "Loan details"
    if tokens & _PROFILE_KEYS:
        return "Borrower profile"
    if tokens & _CREDIT_KEYS:
        return "Credit & debt history"
    return "Additional details"


# ----------------------------------------------------------------------------
# Model / metadata / backend discovery
# ----------------------------------------------------------------------------
def _extract_model(obj: Any):
    if obj is None:
        return None
    if hasattr(obj, "predict") or hasattr(obj, "predict_proba"):
        return obj
    if isinstance(obj, dict):
        for k in ("model", "pipeline", "clf", "estimator", "classifier"):
            if k in obj and hasattr(obj[k], "predict"):
                return obj[k]
        for v in obj.values():
            if hasattr(v, "predict"):
                return v
    if isinstance(obj, (tuple, list)):
        for v in obj:
            if hasattr(v, "predict"):
                return v
    return None


def _required_params(fn) -> List[str]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return ["?"]
    return [p.name for p in sig.parameters.values()
            if p.default is inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]


def load_model():
    if "model" in _CACHE:
        return _CACHE["model"]
    model, err = None, None
    mod = _safe_import("src.model_loader")
    if mod is not None:
        for n in LOADER_NAMES:
            fn = getattr(mod, n, None)
            if callable(fn) and not _required_params(fn):
                try:
                    model = _extract_model(fn())
                except Exception as exc:
                    err = exc
                if model is not None:
                    break
    if model is None:
        try:
            import joblib
            model = _extract_model(joblib.load(REPO_ROOT / MODEL_FILE))
        except Exception as exc:
            err = exc
    _CACHE["model"] = model
    _CACHE["model_error"] = None if model is not None else repr(err)
    return model


def load_metadata() -> Dict[str, Any]:
    if "metadata" in _CACHE:
        return _CACHE["metadata"]
    meta: Any = {}
    p = REPO_ROOT / METADATA_FILE
    try:
        meta = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        meta = {}
    if not isinstance(meta, dict):
        meta = {"value": meta}
    _CACHE["metadata"] = meta
    return meta


def _find_ct(obj: Any):
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
    except Exception:
        return None
    if isinstance(obj, ColumnTransformer):
        return obj
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            found = _find_ct(step)
            if found is not None:
                return found
    return None


def _iter_estimators(obj: Any):
    yield obj
    if hasattr(obj, "steps"):
        for _, s in obj.steps:
            yield from _iter_estimators(s)


def _model_columns(model: Any, metadata: Dict[str, Any]) -> List[str]:
    cols = getattr(model, "feature_names_in_", None) if model is not None else None
    if cols is None and model is not None:
        ct = _find_ct(model)
        cols = getattr(ct, "feature_names_in_", None) if ct is not None else None
    if cols is not None:
        return [str(c) for c in cols]
    for k in ("feature_names", "features", "raw_features", "input_features", "feature_columns", "columns", "raw_feature_names"):
        v = metadata.get(k)
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            return list(v)
    return []


def _model_classes(model: Any) -> List[Any]:
    if model is None:
        return []
    c = getattr(model, "classes_", None)
    if c is None and hasattr(model, "steps"):
        c = getattr(model.steps[-1][1], "classes_", None)
    return [_py(x) for x in c] if c is not None else []


def _mapping_from(container: Any, classes: Sequence[Any], depth: int = 0) -> Dict[str, str]:
    if not isinstance(container, dict):
        return {}
    lk = {str(k).lower(): k for k in container}
    for key in LABEL_KEYS:
        if key in lk:
            val = container[lk[key]]
            m: Dict[str, str] = {}
            if isinstance(val, dict):
                m = {str(k): str(v) for k, v in val.items() if not isinstance(v, (dict, list))}
            elif isinstance(val, (list, tuple)) and classes and len(val) == len(classes):
                m = {str(c): str(l) for c, l in zip(classes, val)}
            m = {k: v for k, v in m.items() if k != v}
            if m:
                return m
    if depth == 0:
        for v in container.values():
            if isinstance(v, dict):
                m = _mapping_from(v, classes, 1)
                if m:
                    return m
    return {}


def _resolve_labels(classes: Sequence[Any], metadata: Dict[str, Any]) -> Tuple[Dict[str, str], Optional[str]]:
    m = _mapping_from(metadata, classes)
    if m:
        return m, METADATA_FILE
    cfg = _safe_import("src.config")
    if cfg is not None:
        m = _mapping_from({n: v for n, v in vars(cfg).items() if not n.startswith("_")}, classes)
        if m:
            return m, "src.config"
    return {}, None


_SAMPLE_RE = re.compile(r"sample|example|demo|test_input", re.I)


def _clean_record(d: Dict[Any, Any]) -> Dict[str, Any]:
    return {str(k): _py(v) for k, v in d.items()}


def _as_records(obj: Any, base: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(obj, pd.DataFrame):
        for i, (_, row) in enumerate(obj.head(10).iterrows()):
            out["%s #%d" % (base, i + 1)] = _clean_record(row.to_dict())
    elif isinstance(obj, pd.Series):
        out[base] = _clean_record(obj.to_dict())
    elif isinstance(obj, dict):
        if obj and all(isinstance(v, dict) for v in obj.values()):
            for k, v in obj.items():
                out[str(k)] = _clean_record(v)
        else:
            out[base] = _clean_record(obj)
    elif isinstance(obj, (list, tuple)) and obj and all(isinstance(v, dict) for v in obj):
        for i, v in enumerate(obj):
            out["%s #%d" % (base, i + 1)] = _clean_record(v)
    return {k: v for k, v in out.items() if len(v) >= 3 and all(isinstance(c, str) for c in v)}


def _samples_from_app_ast() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    app = REPO_ROOT / "app.py"
    if not app.exists():
        return out
    try:
        tree = ast.parse(app.read_text(encoding="utf-8"))
    except Exception:
        return out
    for node in ast.walk(tree):
        try:
            if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Dict, ast.List)):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if any(_SAMPLE_RE.search(n) for n in names):
                    out.update(_as_records(ast.literal_eval(node.value), "app.py:" + names[0]))
            elif isinstance(node, ast.FunctionDef) and _SAMPLE_RE.search(node.name):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Dict):
                        out.update(_as_records(ast.literal_eval(sub.value), "app.py:" + node.name))
                        break
        except Exception:
            continue
    return out


def _discover_samples(columns: Sequence[str]) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    cands: List[Tuple[str, Dict[str, Dict[str, Any]]]] = []
    for modname in ("src.config", "src.features", "src.predict", "src.model_loader", "src.shap_explainer"):
        mod = _safe_import(modname)
        if mod is None:
            continue
        for n, v in list(vars(mod).items()):
            if n.startswith("__") or not _SAMPLE_RE.search(n):
                continue
            obj = None
            try:
                if isinstance(v, (dict, list, tuple, pd.Series, pd.DataFrame)):
                    obj = v
                elif inspect.isfunction(v) and not _required_params(v):
                    obj = v()
            except Exception:
                obj = None
            recs = _as_records(obj, modname + "." + n)
            if recs:
                cands.append((modname + "." + n, recs))
    app_recs = _samples_from_app_ast()
    if app_recs:
        cands.append(("app.py (static parse)", app_recs))
    colset = set(columns)
    for source, recs in cands:
        if colset:
            recs = {k: v for k, v in recs.items() if sum(1 for c in v if c in colset) >= 0.6 * len(v)}
        if recs:
            return recs, source
    return {}, None


def _pick_callable(module, names: Sequence[str], hints: Sequence[str] = (), exclude: Sequence[str] = ()):
    if module is None:
        return None, None
    for n in names:
        fn = getattr(module, n, None)
        if callable(fn) and not inspect.isclass(fn):
            return fn, n
    for n, fn in vars(module).items():
        if n.startswith("_") or not inspect.isfunction(fn) or getattr(fn, "__module__", None) != module.__name__:
            continue
        low = n.lower()
        if any(h in low for h in hints) and not any(x in low for x in exclude):
            return fn, n
    return None, None


@dataclasses.dataclass
class Backend:
    predict_fn: Optional[Callable[..., Any]] = None
    predict_name: Optional[str] = None
    shap_fn: Optional[Callable[..., Any]] = None
    shap_name: Optional[str] = None
    model: Any = None
    columns: List[str] = dataclasses.field(default_factory=list)
    classes: List[Any] = dataclasses.field(default_factory=list)
    class_labels: Dict[str, str] = dataclasses.field(default_factory=dict)
    label_source: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    samples: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)
    sample_source: Optional[str] = None
    notes: List[str] = dataclasses.field(default_factory=list)


def discover_backend(refresh: bool = False) -> Backend:
    if not refresh and "backend" in _CACHE:
        return _CACHE["backend"]
    if refresh:
        for k in ("model", "metadata", "backend"):
            _CACHE.pop(k, None)
    notes: List[str] = []
    model = load_model()
    meta = load_metadata()
    if model is None:
        notes.append("Model could not be loaded for schema discovery: %s" % _CACHE.get("model_error"))
    columns = _model_columns(model, meta)
    classes = _model_classes(model)
    labels, label_src = _resolve_labels(classes, meta)
    predict_fn = predict_name = None
    for modname in ("src.predict", "src.model_loader", "src.features"):
        fn, name = _pick_callable(_safe_import(modname), PREDICT_NAMES, ("predict",),
                                  ("batch", "history", "plot", "load", "save", "log"))
        if fn is not None:
            predict_fn, predict_name = fn, modname + "." + name
            break
    if predict_fn is None:
        notes.append("No prediction function found in src.predict / src.model_loader / src.features.")
    shap_fn, shap_name = _pick_callable(_safe_import("src.shap_explainer"), SHAP_NAMES, ("shap", "explain"),
                                        ("plot", "load", "save", "log", "summary", "global"))
    if shap_fn is not None:
        shap_name = "src.shap_explainer." + shap_name
    else:
        notes.append("No SHAP function found in src.shap_explainer.")
    samples, sample_src = _discover_samples(columns)
    if not samples:
        notes.append("No sample-loan record found in src.* or app.py.")
    bk = Backend(predict_fn, predict_name, shap_fn, shap_name, model, columns, classes, labels, label_src,
                 meta, samples, sample_src, notes)
    _CACHE["backend"] = bk
    return bk


# ----------------------------------------------------------------------------
# Input schema (derived from the fitted pipeline + the app's own sample record)
# ----------------------------------------------------------------------------
def _clean_options(arr: Any) -> List[Any]:
    out = []
    for x in list(arr):
        x = _py(x)
        if not _is_missing(x):
            out.append(x)
    return out


def _column_info(model: Any) -> Dict[str, Dict[str, Any]]:
    info: Dict[str, Dict[str, Any]] = {}
    ct = _find_ct(model)
    if ct is None:
        return info
    names = [str(n) for n in getattr(ct, "feature_names_in_", [])]
    for _name, trans, cols in getattr(ct, "transformers_", []):
        if isinstance(trans, str) and trans == "drop":
            continue
        if isinstance(cols, slice):
            cols = names[cols]
        elif callable(cols):
            continue
        cols = list(cols)
        cols = [names[c] if isinstance(c, (int, np.integer)) and names else str(c) for c in cols]
        cats = stats = None
        if not isinstance(trans, str):
            for est in _iter_estimators(trans):
                if cats is None and hasattr(est, "categories_"):
                    cats = est.categories_
                if stats is None and hasattr(est, "statistics_"):
                    stats = est.statistics_
        for i, c in enumerate(cols):
            d: Dict[str, Any] = {"kind": "categorical" if cats is not None else "numeric"}
            if cats is not None and i < len(cats):
                d["options"] = _clean_options(cats[i])
            if stats is not None and len(stats) == len(cols):
                s = _py(stats[i])
                if not _is_missing(s):
                    d["default"] = s
            info[c] = d
    return info


def build_schema(model: Any = None, sample: Optional[Dict[str, Any]] = None,
                 columns: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    info = _column_info(model) if model is not None else {}
    if sample:
        names = list(sample.keys())
    elif columns:
        names = list(columns)
    else:
        names = list(info.keys())
    schema: List[Dict[str, Any]] = []
    for name in names:
        ci = info.get(name, {})
        raw_missing = bool(sample) and name in sample and _is_missing(sample[name])
        sv = None if (not sample or raw_missing) else sample.get(name)
        kind = ci.get("kind")
        if isinstance(sv, (bool, str)):
            kind = "categorical"
        elif _is_number(sv) and kind is None:
            kind = "numeric"
        kind = kind or "numeric"
        options = list(ci.get("options") or [])
        if isinstance(sv, bool) and not options:
            options = [False, True]
        if sv is not None and kind == "categorical" and options and sv not in options:
            options.insert(0, sv)
        default = ci.get("default")
        if kind == "categorical":
            if default not in options:
                default = options[0] if options else ""
        elif _is_missing(default) or isinstance(default, str) or not _is_number(default):
            default = 0
        dtype = "str"
        decimals = 2
        if kind == "numeric":
            dtype = "int" if (isinstance(sv, int) and not isinstance(sv, bool)) else "float"
            if isinstance(sv, float):
                s = repr(sv)
                decimals = 6 if "e" in s.lower() else max(2, min(6, len(s.split(".")[1]) if "." in s else 2))
        schema.append({
            "name": name, "label": prettify_label(name), "help": field_help(name, kind),
            "section": section_for(name), "kind": kind, "dtype": dtype, "options": options,
            "default": default, "decimals": decimals,
            "allow_missing": bool(raw_missing and kind == "numeric"), "sample": sv,
        })
    return schema


def get_input_schema() -> List[Dict[str, Any]]:
    bk = discover_backend()
    first = next(iter(bk.samples.values()), None)
    return build_schema(bk.model, first, bk.columns)


_NEG_WARN_TOKENS = {"amnt", "amount", "inc", "income", "rate", "installment", "length", "bal", "util", "dti", "term"}


def validate_inputs(values: Dict[str, Any], schema: Sequence[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    for f in schema:
        n, label, v = f["name"], f["label"], values.get(f["name"])
        if f["kind"] == "numeric":
            if _is_missing(v):
                if not f.get("allow_missing"):
                    errors.append("%s: a numeric value is required." % label)
                continue
            try:
                fv = float(v)
            except Exception:
                errors.append("%s: '%s' is not a number." % (label, v))
                continue
            if not math.isfinite(fv):
                errors.append("%s: the value must be a finite number." % label)
        else:
            if _is_missing(v) or (isinstance(v, str) and not v.strip()):
                errors.append("%s: please choose or enter a value." % label)
            elif f.get("options") and v not in f["options"]:
                errors.append("%s: '%s' is not one of the options the model knows." % (label, v))
    return errors


def soft_warnings(values: Dict[str, Any], schema: Sequence[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for f in schema:
        v = values.get(f["name"])
        tokens = set(re.split(r"[^a-z0-9]+", f["name"].lower()))
        if f["kind"] == "numeric" and _is_number(v) and not _is_missing(v) and v < 0 and tokens & _NEG_WARN_TOKENS:
            out.append("%s is negative - please double-check this value." % f["label"])
    return out


# ----------------------------------------------------------------------------
# Calling the EXISTING prediction / SHAP functions (adapter only)
# ----------------------------------------------------------------------------
def _call_with_input(fn: Callable[..., Any], values: Dict[str, Any], columns: Optional[Sequence[str]] = None):
    cols = list(columns or values.keys())
    ordered = {c: values[c] for c in cols if c in values}
    sig = inspect.signature(fn)
    P = inspect.Parameter
    params = [p for p in sig.parameters.values() if p.kind in (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD, P.KEYWORD_ONLY)]
    has_var_kw = any(p.kind == P.VAR_KEYWORD for p in sig.parameters.values())
    if not params:
        return fn(**ordered) if has_var_kw else fn()
    hits = sum(1 for p in params if p.name in ordered)
    if (len(params) >= 2 and hits >= max(2, int(0.8 * len(ordered)))) or (has_var_kw and not params):
        return fn(**ordered)
    first = params[0]
    ann = str(first.annotation).lower()
    nm = first.name.lower()
    wants_df = "dataframe" in ann or nm in ("df", "data_frame", "frame", "dataframe", "x") or nm.endswith("_df")
    extra: Dict[str, Any] = {}
    for p in params[1:]:
        if p.default is inspect._empty:
            ln = p.name.lower()
            if ln in ("model", "pipeline", "pipe", "clf", "estimator"):
                extra[p.name] = load_model()
            elif ln in ("metadata", "meta"):
                extra[p.name] = load_metadata()
            else:
                raise TypeError("Unsupported required argument '%s' in %s" % (p.name, getattr(fn, "__name__", fn)))
    to_df = lambda: pd.DataFrame([ordered], columns=list(ordered.keys()))
    to_dict = lambda: dict(ordered)
    builders = [to_df, to_dict] if wants_df else [to_dict, to_df]
    first_err: Optional[BaseException] = None
    for build in builders:
        try:
            return fn(build(), **extra)
        except (TypeError, AttributeError, KeyError, ValueError) as exc:
            if first_err is None:
                first_err = exc
    raise first_err  # type: ignore[misc]


# ----------------------------------------------------------------------------
# Output normalisation (presentation only - values are never altered)
# ----------------------------------------------------------------------------
_PRED_KEYS = ("predicted_class", "prediction", "predicted_label", "pred", "predicted", "class", "label", "y_pred", "result", "outcome")
_PROB_KEYS = ("probabilities", "probability", "probs", "proba", "class_probabilities", "predict_proba", "prob", "class_probs")
_LABEL_KEYS = ("predicted_label", "prediction_label", "class_label", "label_name", "predicted_class_label")


def _scalar(x: Any) -> Any:
    if isinstance(x, np.ndarray) and x.size == 1:
        return x.reshape(-1)[0]
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return x[0]
    if isinstance(x, pd.Series) and len(x) == 1:
        return x.iloc[0]
    return x


def _to_mapping(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items()}
    if isinstance(raw, pd.DataFrame):
        return {str(k): v for k, v in raw.iloc[0].to_dict().items()} if len(raw) else {}
    if isinstance(raw, pd.Series):
        return {str(k): v for k, v in raw.to_dict().items()}
    if dataclasses.is_dataclass(raw) and not isinstance(raw, type):
        return dataclasses.asdict(raw)
    if hasattr(raw, "_asdict"):
        return dict(raw._asdict())
    if isinstance(raw, (tuple, list)):
        if len(raw) == 1:
            return {"prediction": raw[0]}
        if len(raw) == 2:
            return {"prediction": raw[0], "probabilities": raw[1]}
        if len(raw) == 3:
            return {"prediction": raw[0], "probabilities": raw[1], "extra": raw[2]}
        return {"result": list(raw)}
    if hasattr(raw, "__dict__") and not isinstance(raw, (str, bytes, np.ndarray, np.generic)):
        d = {k: v for k, v in vars(raw).items() if not k.startswith("_")}
        if d:
            return d
    return {"prediction": raw}


def _probs_to_pairs(val: Any, classes: Sequence[Any]) -> List[Tuple[str, Any]]:
    if isinstance(val, dict):
        return [(str(k), v) for k, v in val.items()]
    if isinstance(val, pd.Series):
        return [(str(k), v) for k, v in val.items()]
    if isinstance(val, pd.DataFrame):
        return [(str(k), v) for k, v in val.iloc[0].items()] if len(val) else []
    if isinstance(val, (list, tuple, np.ndarray)):
        arr = np.asarray(val) if not isinstance(val, np.ndarray) else val
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 1:
            names = [str(c) for c in classes] if classes and len(classes) == len(arr) else ["Output %d" % i for i in range(len(arr))]
            return [(names[i], arr[i]) for i in range(len(arr))]
    if _is_number(val):
        return [("Returned probability", val)]
    return [("Returned value", val)]


def normalize_prediction(raw: Any, classes: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    classes = list(classes or [])
    m = _to_mapping(raw)
    low = {k.lower(): k for k in m}
    used = set()
    label_key = next((low[k] for k in _LABEL_KEYS if k in low), None)
    pred_key = next((low[k] for k in _PRED_KEYS if k in low and low[k] != label_key), None)
    prob_key = next((low[k] for k in _PROB_KEYS if k in low), None)
    predicted = _scalar(m[pred_key]) if pred_key is not None else None
    label = _scalar(m[label_key]) if label_key is not None else None
    probs = None
    if prob_key is not None:
        probs = _probs_to_pairs(m[prob_key], classes)
        used.add(prob_key)
    else:
        pk = [k for k in m if "prob" in k.lower()]
        if pk:
            probs = [(k, m[k]) for k in pk]
            used.update(pk)
    for k in (pred_key, label_key):
        if k is not None:
            used.add(k)
    return {"predicted_class": predicted, "predicted_label": label, "probabilities": probs,
            "extras": {k: v for k, v in m.items() if k not in used}, "raw_type": type(raw).__name__}


_FEATURE_KEYS = ("feature", "feature_name", "features", "name", "variable", "column")
_VALUE_KEYS = ("shap_value", "shap_values", "shap", "contribution", "contributions", "impact", "effect", "importance")
_INPUT_KEYS = ("value", "feature_value", "input_value", "data", "values")


def _pick(lowmap: Dict[str, Any], keys: Sequence[str]):
    for k in keys:
        if k in lowmap:
            return lowmap[k]
    return None


def _class_index(pc: Any, classes: Sequence[Any], n: int) -> int:
    if classes and pc is not None:
        for i, c in enumerate(classes):
            if str(c) == str(pc):
                return min(i, n - 1)
    return n - 1


def _reduce_vals(vals: np.ndarray, n_names: int, pc: Any, classes: Sequence[Any]) -> np.ndarray:
    v = np.asarray(vals, dtype=float)
    if v.ndim == 3:
        v = v[0]
    if v.ndim == 2:
        if n_names and v.shape[0] == n_names and v.shape[1] != n_names:
            v = v[:, _class_index(pc, classes, v.shape[1])]
        elif n_names and v.shape[1] == n_names and v.shape[0] != n_names:
            v = v[_class_index(pc, classes, v.shape[0])]
        else:
            v = v[0]
    return v.reshape(-1)


def _df_from_table(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    lm = {str(c).lower(): c for c in df.columns}
    fcol, vcol = _pick(lm, _FEATURE_KEYS), _pick(lm, _VALUE_KEYS)
    if vcol is None:
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != fcol]
        cand = [c for c in nums if str(c).lower() not in _INPUT_KEYS]
        vcol = (cand or nums or [None])[0]
    if vcol is None:
        return None
    feats = df[fcol].astype(str).values if fcol is not None else df.index.astype(str).values
    out = pd.DataFrame({"feature": feats, "contribution": pd.to_numeric(df[vcol], errors="coerce").values})
    icol = _pick(lm, _INPUT_KEYS)
    if icol is not None and icol not in (vcol, fcol):
        out["value"] = df[icol].astype(str).values
    return out.dropna(subset=["contribution"])


def _shap_to_df(raw: Any, pc: Any, classes: Sequence[Any]) -> Optional[pd.DataFrame]:
    if raw is None:
        return None
    if isinstance(raw, pd.DataFrame):
        return _df_from_table(raw)
    if isinstance(raw, pd.Series):
        return _df_from_table(raw.to_frame("contribution"))
    if hasattr(raw, "values") and hasattr(raw, "feature_names") and not isinstance(raw, dict):
        names = [str(n) for n in (raw.feature_names or [])]
        vals = _reduce_vals(np.asarray(raw.values, dtype=float), len(names), pc, classes)
        names = names[:len(vals)] if names else ["feature_%d" % i for i in range(len(vals))]
        df = pd.DataFrame({"feature": names, "contribution": vals[:len(names)]})
        data = getattr(raw, "data", None)
        if data is not None:
            d = np.asarray(data).reshape(-1) if np.asarray(data).ndim <= 1 else np.asarray(data)[0]
            if len(d) >= len(names):
                df["value"] = [str(x) for x in d[:len(names)]]
        return df
    if isinstance(raw, dict):
        lm = {str(k).lower(): k for k in raw}
        fk = _pick(lm, ("features", "feature_names", "feature", "names"))
        vk = _pick(lm, ("shap_values", "shap_value", "values", "contributions", "shap"))
        if fk is not None and vk is not None and _is_seq(raw[fk]) and _is_seq(raw[vk]):
            names = [str(x) for x in raw[fk]]
            vals = _reduce_vals(np.asarray(raw[vk], dtype=float), len(names), pc, classes)
            n = min(len(names), len(vals))
            df = pd.DataFrame({"feature": names[:n], "contribution": vals[:n]})
            dk = _pick(lm, ("data", "feature_values", "input_values"))
            if dk is not None and _is_seq(raw[dk]) and len(raw[dk]) >= n:
                df["value"] = [str(x) for x in list(raw[dk])[:n]]
            return df
        for k in ("top_features", "contributions", "feature_contributions", "explanation", "shap",
                  "shap_values", "features", "top_contributions", "drivers"):
            if k in lm and isinstance(raw[lm[k]], (dict, list, tuple, pd.DataFrame, pd.Series)):
                r = _shap_to_df(raw[lm[k]], pc, classes)
                if r is not None and not r.empty:
                    return r
        nums = {str(k): v for k, v in raw.items() if _is_number(v)}
        if len(nums) >= 2 and len(nums) >= 0.5 * len(raw):
            return pd.DataFrame({"feature": list(nums.keys()), "contribution": [float(v) for v in nums.values()]})
        return None
    if isinstance(raw, (list, tuple)):
        if not raw:
            return None
        if all(isinstance(x, dict) for x in raw):
            return _df_from_table(pd.DataFrame(list(raw)))
        if all(isinstance(x, (tuple, list)) and len(x) in (2, 3) for x in raw):
            rows = []
            for x in raw:
                row = {"feature": str(x[0]), "contribution": float(x[-1])}
                if len(x) == 3:
                    row["value"] = str(x[1])
                rows.append(row)
            return pd.DataFrame(rows)
        if all(_is_number(x) for x in raw):
            return pd.DataFrame({"feature": ["feature_%d" % i for i in range(len(raw))], "contribution": [float(x) for x in raw]})
        return None
    if isinstance(raw, np.ndarray):
        vals = _reduce_vals(raw, 0, pc, classes)
        return pd.DataFrame({"feature": ["feature_%d" % i for i in range(len(vals))], "contribution": vals})
    return None


def _clean_feature_name(n: str) -> str:
    return re.sub(r"^(num|numeric|cat|categorical|remainder|onehot|ord|ordinal|passthrough)__", "", str(n))


def normalize_shap(raw: Any, predicted_class: Any = None, classes: Optional[Sequence[Any]] = None) -> Optional[pd.DataFrame]:
    df = _shap_to_df(raw, predicted_class, list(classes or []))
    if df is None or df.empty:
        return None
    df = df.copy()
    df["feature"] = df["feature"].map(_clean_feature_name)
    order = df["contribution"].abs().sort_values(ascending=False, kind="stable").index
    return df.loc[order].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Orchestration (no UI)
# ----------------------------------------------------------------------------
def run_prediction(values: Dict[str, Any], predict_fn: Optional[Callable[..., Any]] = None,
                   shap_fn: Optional[Callable[..., Any]] = None, include_shap: bool = True,
                   backend: Optional[Backend] = None, columns: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    bk = backend or discover_backend()
    pf = predict_fn or bk.predict_fn
    if pf is None:
        raise RuntimeError("No prediction function is available. Pass predict_fn=... or check src/predict.py.")
    t0 = time.perf_counter()
    raw = _call_with_input(pf, values, columns)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    norm = normalize_prediction(raw, bk.classes)
    result: Dict[str, Any] = {"raw": raw, "normalized": norm, "latency_ms": latency_ms,
                              "shap": None, "shap_raw": None, "shap_figure": None, "shap_error": None}
    sf = shap_fn or bk.shap_fn
    if include_shap and sf is not None:
        try:
            sraw = _call_with_input(sf, values, columns)
            result["shap_raw"] = sraw
            if hasattr(sraw, "savefig"):
                result["shap_figure"] = sraw
            else:
                result["shap"] = normalize_shap(sraw, norm.get("predicted_class"), bk.classes)
        except Exception as exc:
            result["shap_error"] = "%s: %s" % (type(exc).__name__, exc)
    return result


# ----------------------------------------------------------------------------
# UI pieces
# ----------------------------------------------------------------------------
def _normalize_samples(sample_loan: Any) -> Dict[str, Dict[str, Any]]:
    recs = _as_records(sample_loan, "Sample loan")
    if not recs and isinstance(sample_loan, dict) and sample_loan:
        recs = {"Sample loan": _clean_record(sample_loan)}
    return recs


def _load_sample(record: Dict[str, Any], label: str) -> None:
    ss = st.session_state
    ss["lp_defaults"] = dict(record)
    ss["lp_ver"] = ss.get("lp_ver", 0) + 1
    ss.pop("lp_result", None)
    ss.pop("lp_error", None)
    ss["lp_flash"] = "Loaded '%s'. Review the values, then press Estimate loan outcome." % label


def _reset_form() -> None:
    ss = st.session_state
    ss["lp_defaults"] = {}
    ss["lp_ver"] = ss.get("lp_ver", 0) + 1
    ss.pop("lp_result", None)
    ss.pop("lp_error", None)
    ss["lp_flash"] = "Form reset to its starting values."


def _render_sample_panel(samples: Dict[str, Dict[str, Any]]) -> None:
    with _box():
        st.markdown("#### Try a sample loan")
        if not samples:
            st.caption("No sample loan was found in the application code - fill in the form manually.")
            st.button("Reset form", key="lp_reset_only", on_click=_reset_form)
            return
        st.caption("Fill the whole form with the application's built-in example, then press Estimate loan outcome.")
        label = next(iter(samples))
        if len(samples) > 1:
            label = st.selectbox("Sample", list(samples.keys()), key="lp_sample_choice")
        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            st.button("Load sample loan", key="lp_load_sample", type="primary", on_click=_load_sample,
                      args=(samples[label], label), **_stretch(st.button))
        with c2:
            st.button("Reset form", key="lp_reset", on_click=_reset_form, **_stretch(st.button))


def _coerce_default(f: Dict[str, Any], defaults: Dict[str, Any]) -> Tuple[Any, bool]:
    name = f["name"]
    if name not in defaults:
        return f["default"], False
    v = defaults[name]
    if f["kind"] == "numeric":
        if _is_missing(v):
            return f["default"], bool(f.get("allow_missing"))
        try:
            return (int(round(float(v))) if f["dtype"] == "int" else float(v)), False
        except Exception:
            return f["default"], False
    if f.get("options"):
        if v in f["options"]:
            return v, False
        for o in f["options"]:
            if str(o) == str(v):
                return o, False
        return f["default"], False
    return ("" if _is_missing(v) else str(v)), False


def _render_field(f: Dict[str, Any], defaults: Dict[str, Any], ver: int) -> Any:
    key = "lp_%s_%d" % (f["name"], ver)
    cur, start_missing = _coerce_default(f, defaults)
    if f["kind"] == "numeric":
        base = f["default"] if start_missing else cur
        if f["dtype"] == "int":
            val = st.number_input(f["label"], value=int(round(float(base))), step=1, format="%d", key=key, help=f["help"])
        else:
            dec = int(f.get("decimals", 2))
            step = 1.0 if abs(float(base)) >= 100 else float(10 ** -min(dec, 2))
            val = st.number_input(f["label"], value=float(base), step=step, format="%%.%df" % dec, key=key, help=f["help"])
        if f.get("allow_missing"):
            na = st.checkbox("%s - not provided" % f["label"], value=start_missing, key=key + "_na",
                             help="Leave this value missing; the model's own preprocessing handles it.")
            if na:
                return float("nan")
        return val
    if f.get("options"):
        idx = f["options"].index(cur) if cur in f["options"] else 0
        return st.selectbox(f["label"], f["options"], index=idx, key=key, help=f["help"], format_func=str)
    return st.text_input(f["label"], value=str(cur or ""), key=key, help=f["help"])


def _render_form(schema: List[Dict[str, Any]], shap_available: bool, ver: int, defaults: Dict[str, Any]):
    values: Dict[str, Any] = {}
    sections: Dict[str, List[Dict[str, Any]]] = {}
    for f in schema:
        sections.setdefault(f["section"], []).append(f)
    with st.form("lp_form"):
        for sec in SECTION_ORDER:
            fields = sections.get(sec)
            if not fields:
                continue
            with _box():
                st.markdown("##### " + sec)
                ncols = 3 if len(fields) >= 6 else 2
                cols = st.columns(ncols)
                for i, f in enumerate(fields):
                    with cols[i % ncols]:
                        values[f["name"]] = _render_field(f, defaults, ver)
        include_shap = st.checkbox("Include SHAP explanation", value=shap_available, disabled=not shap_available,
                                   key="lp_include_shap_%d" % ver,
                                   help="Shows which inputs contributed most to this specific result.")
        if not shap_available:
            st.caption("No SHAP explainer was found in the application code.")
        submitted = st.form_submit_button("Estimate loan outcome", type="primary", **_stretch(st.form_submit_button))
    ordered = {f["name"]: values[f["name"]] for f in schema if f["name"] in values}
    return ordered, submitted, include_shap


def _class_display(name: Any, labels: Dict[str, str]) -> str:
    s = str(name)
    if s in labels:
        return "%s (encoded class %s)" % (labels[s], s)
    if re.fullmatch(r"-?\d+(\.0+)?", s):
        return "Encoded class %s" % s
    return s


def _render_card(norm: Dict[str, Any], labels: Dict[str, str], label_src: Optional[str], latency_ms: float) -> None:
    pc, pl = norm.get("predicted_class"), norm.get("predicted_label")
    if pl is not None:
        headline = html.escape(format_exact(pl))
        sub = "Label returned by the prediction code"
    elif pc is not None:
        headline = html.escape(_class_display(format_exact(pc), labels))
        sub = "Model output for this loan"
    else:
        headline = "Class probabilities returned"
        sub = "The prediction function did not return a single class"
    rows = ""
    for name, val in (norm.get("probabilities") or []):
        shown = html.escape(format_exact(val))
        frac = _fraction(val)
        bar = ""
        if frac is not None:
            bar = ("<div style='height:8px;border-radius:4px;background:rgba(128,128,128,0.25);'>"
                   "<div style='height:8px;border-radius:4px;width:%.4f%%;background:#3b82f6;'></div></div>" % (frac * 100.0))
        rows += ("<div style='margin:0.65rem 0;'><div style='display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;'>"
                 "<span>%s</span><span style='font-weight:600;font-variant-numeric:tabular-nums;word-break:break-all;'>%s</span></div>%s</div>"
                 % (html.escape(_class_display(name, labels)), shown, bar))
    prob_block = ""
    if rows:
        prob_block = ("<div style='margin-top:1rem;font-size:0.8rem;letter-spacing:0.04em;text-transform:uppercase;opacity:0.7;'>"
                      "Model-returned probabilities</div>" + rows)
    card = ("<div class='lp-result-card' style='border:1px solid rgba(59,130,246,0.45);border-left:8px solid #3b82f6;"
            "border-radius:14px;padding:1.2rem 1.4rem;margin:0.5rem 0 0.8rem 0;background:rgba(59,130,246,0.08);'>"
            "<div style='font-size:0.8rem;letter-spacing:0.04em;text-transform:uppercase;opacity:0.7;'>%s</div>"
            "<div style='font-size:1.9rem;font-weight:700;line-height:1.25;margin-top:0.2rem;'>%s</div>%s"
            "<div style='margin-top:1rem;font-size:0.85rem;opacity:0.85;'>This is an estimate, not a guarantee.</div></div>"
            % (html.escape(sub), headline, prob_block))
    st.markdown(card, unsafe_allow_html=True)
    if labels:
        st.caption("Class label mapping taken from `%s`." % label_src)
    else:
        st.caption("Class labels are shown as the model's encoded target values. This page does not assume which "
                   "encoded class corresponds to loan default - no mapping was found in the application code or metadata.")
    st.caption("Probabilities are shown exactly as returned by the model and are not calibrated risk estimates. "
               "Computed in %.0f ms." % latency_ms)


def _render_shap(out: Dict[str, Any]) -> None:
    st.markdown("### Why did the model return this?")
    if out.get("shap_figure") is not None:
        try:
            st.pyplot(out["shap_figure"])
        except Exception as exc:
            st.warning("The SHAP figure could not be displayed (%s)." % type(exc).__name__)
        return
    df = out.get("shap")
    if out.get("shap_error"):
        st.warning("The SHAP explanation could not be generated, but the prediction above is unaffected. Details: " + out["shap_error"])
        return
    if df is None or df.empty:
        if out.get("shap_raw") is not None:
            st.info("The explainer returned an output in a format this page does not recognise; showing it as returned.")
            try:
                st.write(out["shap_raw"])
            except Exception:
                st.text(str(out["shap_raw"])[:2000])
        else:
            st.caption("No SHAP explanation was requested or available.")
        return
    st.caption("Each bar shows how much a feature moved the model's output for this one loan, away from its average. "
               "Positive values raise the explained output, negative values lower it. SHAP describes the model's behaviour; "
               "it does not show cause and effect.")
    n_max = min(30, len(df))
    n = min(10, n_max)
    if n_max > 3:
        n = st.slider("Number of features to show", 3, n_max, n, key="lp_shap_n")
    top = df.head(n).copy()
    top["effect"] = np.where(top["contribution"] >= 0, "Raises output", "Lowers output")
    labels = []
    seen: Dict[str, int] = {}
    for f in top["feature"]:
        seen[f] = seen.get(f, 0) + 1
        labels.append(f if seen[f] == 1 else "%s (%d)" % (f, seen[f]))
    top["feature"] = labels
    try:
        import altair as alt
        chart = (alt.Chart(top).mark_bar().encode(
            x=alt.X("contribution:Q", title="SHAP contribution"),
            y=alt.Y("feature:N", sort=list(top["feature"]), title=None),
            color=alt.condition(alt.datum.contribution > 0, alt.value("#2563eb"), alt.value("#f59e0b")),
            tooltip=["feature", "contribution", "effect"]).properties(height=max(180, 28 * len(top))))
        st.altair_chart(chart, **_stretch(st.altair_chart))
    except Exception:
        st.bar_chart(top.set_index("feature")["contribution"])
    show_cols = [c for c in ("feature", "value", "contribution", "effect") if c in top.columns]
    try:
        st.dataframe(top[show_cols], hide_index=True, **_stretch(st.dataframe))
    except Exception:
        st.dataframe(top[show_cols])


def _render_outputs(schema: List[Dict[str, Any]]) -> None:
    ss = st.session_state
    err, res = ss.get("lp_error"), ss.get("lp_result")
    if err:
        if err["kind"] == "validation":
            st.error("Please fix the following before estimating:\n\n" + "\n".join("- " + m for m in err["messages"]))
        else:
            st.error("The model could not produce an estimate for these inputs. Please review the values and try again. "
                     "If this keeps happening, the technical details below will help a developer diagnose it.")
            with st.expander("Technical details"):
                st.code("%s: %s" % (err["type"], err["message"]))
                st.code(err["trace"])
    if not res:
        return
    out, norm = res["out"], res["out"]["normalized"]
    st.markdown("### Result")
    for w in res.get("warnings", []):
        st.warning(w)
    _render_card(norm, res["labels"], res["label_src"], out["latency_ms"])
    st.info(DISCLAIMER)
    if res.get("callback_warning"):
        st.warning(res["callback_warning"])
    _render_shap(out)
    with st.expander("Inputs used for this estimate"):
        st.dataframe(pd.DataFrame({"field": list(res["inputs"].keys()),
                                   "value": [str(v) for v in res["inputs"].values()]}), hide_index=True)
    with st.expander("Raw model output"):
        try:
            st.json(json.loads(json.dumps(norm, default=lambda o: format_exact(o) if isinstance(o, np.generic) else str(o))))
        except Exception:
            st.text(str(norm))
    st.caption("Estimated at %s." % res["stamp"])


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
def render_prediction_page(*, predict_fn: Optional[Callable[..., Any]] = None,
                           shap_fn: Optional[Callable[..., Any]] = None,
                           sample_loan: Any = None,
                           schema: Optional[List[Dict[str, Any]]] = None,
                           class_labels: Optional[Dict[Any, str]] = None,
                           on_result: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
                           on_error: Optional[Callable[[Dict[str, Any], BaseException], None]] = None,
                           show_title: bool = True) -> None:
    """Render the Loan Prediction page. Does not call st.set_page_config."""
    bk = discover_backend()
    pf = predict_fn or bk.predict_fn
    sf = shap_fn or bk.shap_fn
    samples = _normalize_samples(sample_loan) if sample_loan is not None else bk.samples
    first = next(iter(samples.values()), None) if samples else None
    sch = schema or build_schema(bk.model, first, bk.columns)
    labels = {str(k): str(v) for k, v in class_labels.items()} if class_labels else dict(bk.class_labels)
    label_src = "the caller" if class_labels else bk.label_source
    ss = st.session_state
    ss.setdefault("lp_ver", 0)
    ss.setdefault("lp_defaults", {})

    if show_title:
        st.title("Loan Prediction")
        st.write("Enter the loan and borrower details, then run the model to see its estimate and which inputs influenced it.")
    if not sch:
        st.error("The input schema could not be determined from the model or the sample loan, so the form cannot be shown.")
        with st.expander("What was checked"):
            for n in bk.notes:
                st.write(n)
        return
    if pf is None:
        st.error("No prediction function was found, so predictions are unavailable on this page.")
        with st.expander("What was checked"):
            for n in bk.notes:
                st.write(n)
        return

    st.info("Predictions are estimates, not guarantees. Review every field before you run the model.")
    flash = ss.pop("lp_flash", None)
    _render_sample_panel(samples)
    if flash:
        st.success(flash)
    values, submitted, include_shap = _render_form(sch, sf is not None, ss["lp_ver"], ss["lp_defaults"])

    if submitted:
        ss.pop("lp_result", None)
        ss.pop("lp_error", None)
        errors = validate_inputs(values, sch)
        if errors:
            ss["lp_error"] = {"kind": "validation", "messages": errors}
        else:
            try:
                with st.spinner("Running the model..."):
                    out = run_prediction(values, predict_fn=pf, shap_fn=sf, include_shap=include_shap,
                                         backend=bk, columns=[f["name"] for f in sch])
                ss["lp_result"] = {"inputs": values, "out": out, "labels": labels, "label_src": label_src,
                                   "stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                   "warnings": soft_warnings(values, sch)}
                if on_result is not None:
                    try:
                        on_result(values, out)
                    except Exception as cb_exc:
                        ss["lp_result"]["callback_warning"] = "The result was produced, but logging it failed (%s)." % type(cb_exc).__name__
            except Exception as exc:
                ss["lp_error"] = {"kind": "exception", "type": type(exc).__name__, "message": str(exc),
                                  "trace": traceback.format_exc()}
                if on_error is not None:
                    try:
                        on_error(values, exc)
                    except Exception:
                        pass
    _render_outputs(sch)
