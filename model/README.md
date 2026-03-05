# Model Folder

- `ensemble_config.json`
  - Contains ensemble model configurations like  weights, threshold, and model file mapping.
  - Used during inference to blend model probabilities.

- `trained_model_xgboost.json`
  - Saved XGBoost model.
  - Provides XGBoost prediction probabilities.

- `trained_model_lgboost.json`
  - Saved LightGBM model.
  - Provides LightGBM prediction probabilities.

- `trained_model_catboost.json`
  - Saved CatBoost model.
  - Provides CatBoost prediction probabilities.

- `predict.py`
  - Loads ensemble configurations + all three models.
  - Reads `dataset/TEST.csv` and writes final predictions to  `FINAL.csv`.
