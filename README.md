# IEEE SB GEHU ML Challenge - Fault Detection

## Overview
This project solves a binary fault-detection task using 47 sensor features (`F01`-`F47`).
- `Class = 0`: Normal device behavior
- `Class = 1`: Faulty device behavior

The full workflow is in `notebook.ipynb`: data cleaning, feature selection, model training, cross-validation, final retraining, and `FINAL.csv` generation.

## Evaluator Quickstart (Windows PowerShell)
From the repository root, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1
.\.venv\Scripts\python.exe .\predict.py --model trained_model.json --test TEST.csv --output FINAL.csv
```

This creates `.venv`, installs all dependencies from `requirements.txt`, and generates `FINAL.csv` from the provided [`trained_model.json`](./trained_model.json).

## Manual Venv Setup

### PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python .\predict.py --model trained_model.json --test TEST.csv --output FINAL.csv
deactivate
```

### Command Prompt (cmd.exe)
```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python predict.py --model trained_model.json --test TEST.csv --output FINAL.csv
deactivate
```

## Final Performance
| Metric | Value |
|---|---|
| CV Strategy | 5-fold Stratified CV |
| Final CV AUC | `0.9992 +- 0.0002` |
| Final Model | XGBoost (Optuna-tuned parameters) |

## Approach

### 1. Data Quality and Cleaning
- Removed 1 corrupted target row (`Class = 2.799`) from training data.
- Removed 738 duplicate rows to avoid CV leakage/inflated metrics.
- Used deterministic processing to keep the run reproducible.

### 2. Feature Selection
A preliminary XGBoost model is used to compute normalized mean absolute SHAP
importance for each feature.

Features with normalized SHAP importance below `0.010` are treated as weak and
dropped.

The notebook prints the dropped-feature list and resulting feature count for the
current run.

### 3. Train/Test Alignment
- Preserved `ID` from `TEST.csv` for submission output.
- Built `x_test_reduced` with exact same columns/order as reduced train features.
- Added row-count checks before saving final submission.

### 4. Model Choice
XGBoost was selected because the feature-target relationship is strongly nonlinear and tree ensembles handled this structure better than linear baselines.

### 5. Hyperparameter Tuning and Validation
- Hyperparameters were tuned with Optuna (TPE sampler).
- Best configuration validated with 5-fold Stratified CV.
- Per-fold AUC and standard deviation are reported to confirm stability, not just peak score.

## Final Model Parameters
```python
{
    "n_estimators": 914,
    "learning_rate": 0.0286,
    "max_depth": 9,
    "min_child_weight": 1,
    "gamma": 0.3764,
    "subsample": 0.7999,
    "colsample_bytree": 0.6223,
    "reg_alpha": 1.44e-07,
    "reg_lambda": 1.6079
}
```

## Setup Guide

### 1. Prerequisites
- Python 3.9+ recommended
- Jupyter Notebook (only needed if you want to re-run full training notebook)

### 2. One-command setup (recommended)
```bash
powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1
```
This single script creates `.venv` and installs all dependencies.

### 3. Manual install (alternative)
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Required files in project root
- `TRAIN.csv`
- `TEST.csv`
- `notebook.ipynb`
- [`trained_model.json`](./trained_model.json)

### 5. Run the full pipeline (optional retraining)
```bash
jupyter notebook notebook.ipynb
```
In VS Code/Jupyter, select kernel: `Python (.venv ML Challenge)`.

Run all cells from top to bottom. The notebook will:
1. Clean training data
2. Drop weak features
3. Train and validate XGBoost
4. Save `trained_model.json`
5. Generate `FINAL.csv`

## Predict Using Saved Model
Use the provided script to generate predictions directly from `trained_model.json` and `TEST.csv`:

```bash
python predict.py --model trained_model.json --test TEST.csv --output FINAL.csv
```

## Model Save/Load Verification
```python
import numpy as np
import xgboost as xgb

# Save trained model
model.save_model("trained_model.json")
print("Model saved as trained_model.json")

# Reload and verify probability predictions match
reloaded = xgb.XGBClassifier()
reloaded.load_model("trained_model.json")

probs_verify = reloaded.predict_proba(x_test_reduced)[:, 1]
print("Model loads correctly:", np.allclose(probs, probs_verify))
```

## Output Files
After a successful run you should have:
- [`trained_model.json`](./trained_model.json) - serialized trained XGBoost model
- `FINAL.csv` - final predictions in required format (`ID`, `CLASS`)

## Repository Structure
```text
your-repo/
|- README.md
|- setup_venv.ps1
|- notebook.ipynb
|- predict.py
|- requirements.txt
|- TRAIN.csv
|- TEST.csv
|- trained_model.json
`- FINAL.csv
```

## Reproducibility Notes
- Fixed random seed (`42`) across pipeline components.
- Stratified folds are used to preserve class distribution.
- Submission generation includes sanity checks for row-count consistency.

## Competition Evaluation Alignment
This repository is organized for all major evaluation dimensions:
- Metric performance (CV AUC)
- Code quality (clear staged pipeline)
- Reproducibility (fixed seeds, saved model)
- Explanation (documented modeling decisions)
