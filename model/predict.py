import argparse
import json
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import pandas as pd
import xgboost as xgb


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_model_meta(config: dict, name: str) -> dict:
    for item in config.get("models", []):
        if item.get("name") == name:
            return item
    raise ValueError(f"Missing model entry for '{name}' in ensemble config.")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Generate FINAL.csv using the three-model ensemble."
    )
    parser.add_argument(
        "--config",
        default=str(script_dir / "ensemble_config.json"),
        help="Path to ensemble_config.json",
    )
    parser.add_argument(
        "--test",
        default=str(repo_root / "dataset" / "TEST.csv"),
        help="Path to TEST.csv",
    )
    parser.add_argument(
        "--output",
        default=str(repo_root / "FINAL.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    test_path = Path(args.test)
    output_path = Path(args.output)

    config = load_config(config_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    xgb_meta = get_model_meta(config, "XGBoost")
    lgb_meta = get_model_meta(config, "LightGBM")
    cb_meta = get_model_meta(config, "CatBoost")

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str((config_path.parent / xgb_meta["file"]).resolve()))

    lgb_model = lgb.Booster(model_file=str((config_path.parent / lgb_meta["file"]).resolve()))

    cb_model = cb.CatBoostClassifier()
    cb_model.load_model(str((config_path.parent / cb_meta["file"]).resolve()), format="json")

    test_df = pd.read_csv(test_path)
    if "ID" not in test_df.columns:
        raise ValueError("TEST.csv must contain an 'ID' column.")

    feature_cols = xgb_model.get_booster().feature_names
    if not feature_cols:
        raise ValueError("XGBoost model does not expose feature names.")

    missing_cols = [col for col in feature_cols if col not in test_df.columns]
    if missing_cols:
        raise ValueError(f"TEST.csv is missing expected feature columns: {missing_cols}")

    x_test = test_df[feature_cols].copy()

    p_xgb = xgb_model.predict_proba(x_test)[:, 1]
    p_lgb = lgb_model.predict(x_test)
    p_cb = cb_model.predict_proba(x_test)[:, 1]

    blend = (
        float(xgb_meta["weight"]) * p_xgb
        + float(lgb_meta["weight"]) * p_lgb
        + float(cb_meta["weight"]) * p_cb
    )
    threshold = float(config["threshold"])
    preds = (blend >= threshold).astype(int)

    final_df = pd.DataFrame({"ID": test_df["ID"], "CLASS": preds})
    if len(final_df) != len(test_df):
        raise RuntimeError("Prediction row count does not match TEST.csv row count.")

    final_df.to_csv(output_path, index=False)
    print(f"Saved {len(final_df)} predictions to {output_path}")


if __name__ == "__main__":
    main()

