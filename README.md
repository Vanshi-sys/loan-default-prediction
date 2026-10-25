# Loan Default Prediction

An end-to-end machine learning application for loan-risk prediction using **XGBoost**, **scikit-learn**, **SHAP**, and **Streamlit**. The project combines a trained classification pipeline with an interactive interface, model explanations, prediction history, operational monitoring, and automated tests.

[![Loan Prediction CI](https://github.com/Vanshi-sys/loan-default-prediction/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/Vanshi-sys/loan-default-prediction/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)

## Live Demo

[Open the Streamlit application](https://loan-default-prediction-bhybshy7n3bv75w3ssltf8.streamlit.app/)

*Note: hosted availability may vary while the application starts or restarts.*

## Model Performance

Reported validation metrics for the saved XGBoost model:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.7191 |
| PR-AUC | 0.3265 |
| Raw input features | 31 |
| Transformed features | 915 |

ROC-AUC measures ranking performance across classification thresholds. PR-AUC summarizes the precision–recall trade-off and is particularly informative when classes are imbalanced.

These are validation results, not a guarantee of performance on future data. Threshold selection, probability calibration, and external validation would be needed before operational use.

## Features

- **Loan prediction:** generate a prediction and class probabilities from user-provided fields.
- **Sample prediction:** explore the application using an example input.
- **Model insights:** view model metadata and reported validation metrics.
- **SHAP explanations:** inspect feature contributions for an individual prediction.
- **Prediction history:** record timestamps, operation type, predicted class, class probabilities, and latency.
- **Monitoring dashboard:** view prediction counts, class distribution, recent history, and average latency.
- **Model health check:** verify that required model artifacts are present.
- **Automated tests:** validate prediction output, SHAP output structure, application syntax, monitoring behaviour, and prediction-history functionality.
- **GitHub Actions CI:** run automated tests on repository pushes.

## Architecture

```text
User input
    |
    v
Streamlit interface
    |
    v
Feature preparation
    |
    v
Saved scikit-learn / XGBoost pipeline
    |
    +----> Predicted class and probabilities
    |
    +----> SHAP feature explanation
    |
    +----> Operational logs and prediction history
                         |
                         v
                Monitoring dashboard
```

The saved model pipeline includes preprocessing and classification. The application uses the model artifact and metadata file at runtime.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application and ML code |
| Pandas / NumPy | Data manipulation |
| scikit-learn | Preprocessing and ML pipeline |
| XGBoost | Binary classification model |
| SHAP | Model prediction explanations |
| Streamlit | Interactive web interface |
| SQLite | Local prediction-history storage |
| pytest | Automated testing |
| GitHub Actions | Continuous integration |

## Repository Structure

```text
loan-default-prediction/
├── .github/
│   └── workflows/
│       └── tests.yml
├── src/
│   ├── config.py
│   ├── features.py
│   ├── model_loader.py
│   ├── monitoring.py
│   ├── predict.py
│   ├── prediction_history.py
│   └── shap_explainer.py
├── tests/
│   ├── test_prediction.py
│   ├── test_prediction_history.py
│   └── test_monitoring.py
├── app.py
├── loan_default_model.pkl
├── model_metadata.json
├── requirements.txt
└── README.md
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Vanshi-sys/loan-default-prediction.git
cd loan-default-prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The scikit-learn version is pinned to `1.6.1` to match the version used to serialize the current model artifact.

### 4. Start the application

```bash
streamlit run app.py
```

### 5. Run automated tests

```bash
python -m pytest tests -v
```

## Monitoring and Data Storage

Prediction history stores operational metadata rather than raw borrower inputs. The current SQLite database is local to the application filesystem.

**Persistence limitation:** local files on a hosted Streamlit environment may not survive restarts, redeployments, or changes in the hosting instance. The current history implementation is intended for demonstration; durable multi-user history would require an appropriately configured persistent database.

Monitoring logs include operation type, prediction success or failure, latency, and error type. They are designed not to log raw borrower inputs.

## Responsible Use and Limitations

- This project is for educational and demonstration purposes only. It must not be used as the sole basis for financial, credit, or lending decisions.
- The reported validation ROC-AUC and PR-AUC do not establish fairness, regulatory compliance, or suitability for real lending.
- The original mapping of encoded class `1` to confirmed loan default has not been independently verified. Class labels are therefore shown as encoded target values.
- Predicted probabilities have not been established as calibrated risk estimates.
- SHAP values describe model contributions; they do not establish causal relationships.
- The model may reflect historical-data biases and may perform differently on new populations or over time.
- A production deployment would need additional work on calibration, fairness evaluation, data and model drift, access control, persistent storage, security, and ongoing validation.

## Future Improvements

- Add probability calibration and decision-threshold evaluation.
- Evaluate performance by relevant subgroups and assess potential bias.
- Implement drift monitoring with reference data and defined alert thresholds.
- Add durable database storage for prediction history.
- Expand integration tests and deployment health checks.
- Document model versioning and reproducible training.

## Disclaimer

This project is intended for educational and demonstration purposes only and should not be treated as financial advice or as an automated lending decision system.
