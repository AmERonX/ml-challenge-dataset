import argparse
from pathlib import Path

import pandas as pd
import xgboost as xgb


WEAK_FEATURES = ["F20", "F40", "F45", "F11", "F44", "F43", "F41"]
EXPECTED_FEATURES = [f"F{i:02d}" for i in range(1, 48) if f"F{i:02d}" not in WEAK_FEATURES]


def build_test_features(test_df: pd.DataFrame) -> pd.DataFrame:
    if "ID" not in test_df.columns:
        raise ValueError("TEST.csv must contain an 'ID' column.")

    x_test = test_df.drop(columns=["ID"]).copy()

    missing_expected = [col for col in EXPECTED_FEATURES if col not in x_test.columns]
    if missing_expected:
        raise ValueError(f"TEST.csv is missing expected feature columns: {missing_expected}")

    # Match the same reduced feature space and order used in training.
    x_test = x_test[EXPECTED_FEATURES]
    return x_test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load trained_model.json and generate FINAL.csv predictions from TEST.csv"
    )
    parser.add_argument("--model", default="trained_model.json", help="Path to trained XGBoost model JSON")
    parser.add_argument("--test", default="TEST.csv", help="Path to TEST.csv")
    parser.add_argument("--output", default="FINAL.csv", help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    args = parser.parse_args()

    model_path = Path(args.model)
    test_path = Path(args.test)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    test_df = pd.read_csv(test_path)
    x_test = build_test_features(test_df)

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    probs = model.predict_proba(x_test)[:, 1]
    preds = (probs >= args.threshold).astype(int)

    final_df = pd.DataFrame({"ID": test_df["ID"], "CLASS": preds})
    if len(final_df) != len(test_df):
        raise RuntimeError("Prediction row count does not match TEST.csv row count.")

    final_df.to_csv(args.output, index=False)
    print(f"Saved {len(final_df)} predictions to {args.output}")


if __name__ == "__main__":
    main()

