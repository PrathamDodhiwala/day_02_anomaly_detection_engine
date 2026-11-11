# Day 02 — Unsupervised Anomaly Detection Engine

A Streamlit application that flags unusual observations in tabular datasets using Isolation Forest and DBSCAN.

## Features
- Upload CSV data or generate a built-in demo dataset.
- Inspect dataset size, data types, missing values and unique values.
- Select numeric features and standardize them before modelling.
- Run Isolation Forest, DBSCAN, or both.
- Compare detector outputs, inspect flagged records and export CSV results.

## Windows / VS Code setup (PowerShell)

```powershell
py -3.10 --version
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run
```powershell
streamlit run app.py
```
Open the local URL printed in the terminal, normally http://localhost:8501.

## Test
```powershell
pytest -q
```

## How to use
1. Open Dataset and upload a CSV or click Use generated demo dataset.
2. Open Detection Lab and select numeric features.
3. Choose Isolation Forest, DBSCAN, or both.
4. Run anomaly detection.
5. Open Results & Export to investigate flags and download CSVs.

## Algorithms
**Isolation Forest** isolates observations through random feature splits. Its contamination parameter controls the expected outlier proportion.

**DBSCAN** groups observations in dense neighbourhoods. Points labelled -1 are treated as noise. It is sensitive to feature scaling, eps and min_samples.

Selected numeric features are standardized before detection. Anomaly flags are investigative signals, not proof of fraud, defects or malicious activity. Review results with domain knowledge.

## Structure
- app.py — Streamlit interface
- src/data_utils.py — loading and preprocessing
- src/detectors.py — Isolation Forest and DBSCAN
- src/metrics.py — result summary
- src/ui.py — theme
- tests/test_core.py — basic tests
