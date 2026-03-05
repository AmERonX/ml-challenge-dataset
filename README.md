# IEEE SB GEHU ML Challenge - Ensemble Pipeline

## Overview
This repository uses a three-model ensemble for binary fault detection:
- XGBoost
- LightGBM
- CatBoost

Final predictions are generated in `FINAL.csv` with columns `ID, CLASS`.

## Quickstart (Windows PowerShell)
```powershell
powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1
.\.venv\Scripts\python.exe .\model\predict.py --config .\model\ensemble_config.json --test .\dataset\TEST.csv --output .\FINAL.csv
```

## Manual Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python .\model\predict.py --config .\model\ensemble_config.json --test .\dataset\TEST.csv --output .\FINAL.csv
deactivate
```

## Re-run Training
Run `notebook.ipynb` top to bottom to retrain all three models and regenerate:
- `model/trained_model_xgboost.json`
- `model/trained_model_lgboost.json`
- `model/trained_model_catboost.json`
- `model/ensemble_config.json`
- `FINAL.csv`

## Required Files
- `dataset/TRAIN.csv`
- `dataset/TEST.csv`
- `notebook.ipynb`
- `model/predict.py`
- `model/ensemble_config.json`
- `model/trained_model_xgboost.json`
- `model/trained_model_lgboost.json`
- `model/trained_model_catboost.json`

## Minimal Structure
```text
ML Challenge Dataset/
|- README.md
|- setup_venv.ps1
|- requirements.txt
|- notebook.ipynb
|- model/
:  |- predict.py
:  |- ensemble_config.json
:  |- trained_model_xgboost.json
:  |- trained_model_lgboost.json
:  `- trained_model_catboost.json
|- dataset/
:  |- TRAIN.csv
:  `- TEST.csv
`- FINAL.csv
```
