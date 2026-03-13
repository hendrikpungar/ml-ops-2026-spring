from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
HW1_DIR = ROOT / "homeworks" / "hw1"
HW2_DIR = ROOT / "homeworks" / "hw2"

DATA_V1 = HW1_DIR / "green_tripdata_2021-01.parquet"
DATA_V2 = HW2_DIR / "green_tripdata_2021-02.parquet"
MODEL_V1 = HW2_DIR / "regression_model_v1.joblib"
MODEL_V3 = HW2_DIR / "regression_model_v3.joblib"
METRICS_JSON = HW2_DIR / "version_metrics.json"

TARGET = "total_amount"


def prepare_data(features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    jan = pd.read_parquet(DATA_V1)
    feb = pd.read_parquet(DATA_V2)

    combined = pd.concat([jan, feb], ignore_index=True)
    cols = features + [TARGET]
    df = combined[cols].dropna().copy()

    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 100)]
    df = df[(df[TARGET] > 0) & (df[TARGET] < 200)]

    X = df[features]
    y = df[TARGET]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    pred = model.predict(X_test)
    return {
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    }


def main() -> None:
    v1_model = joblib.load(MODEL_V1)
    features = list(v1_model.feature_names_in_)

    X_train, X_test, y_train, y_test = prepare_data(features)
    v2_metrics = evaluate(v1_model, X_test, y_test)

    v3_model = LinearRegression()
    v3_model.fit(X_train, y_train)
    v3_metrics = evaluate(v3_model, X_test, y_test)

    joblib.dump(v3_model, MODEL_V3)

    payload = {
        "version_1": {
            "description": "Existing model from Practice 1 with original data/model/code baseline.",
            "data": "January 2021",
            "model": "regression_model_v1.joblib",
            "code": "Practice 1 training pipeline",
        },
        "version_2": {
            "description": "Existing trained model evaluated on new combined-data split.",
            "data": "January + February 2021",
            "model": "regression_model_v1.joblib",
            "code": "Same training model, updated evaluation data/split",
            "metrics": v2_metrics,
        },
        "version_3": {
            "description": "Model retrained on the new combined-data training set.",
            "data": "January + February 2021",
            "model": "regression_model_v3.joblib",
            "code": "Same model family retrained on updated train set",
            "metrics": v3_metrics,
        },
        "delta_v3_minus_v2": {
            "r2": v3_metrics["r2"] - v2_metrics["r2"],
            "mae": v3_metrics["mae"] - v2_metrics["mae"],
            "rmse": v3_metrics["rmse"] - v2_metrics["rmse"],
        },
        "features": features,
        "target": TARGET,
        "split": "train_test_split(test_size=0.2, random_state=42)",
    }

    METRICS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Version 2 metrics (old model on new test set):", v2_metrics)
    print("Version 3 metrics (retrained model):", v3_metrics)
    print("Saved:")
    print(f"- {MODEL_V3}")
    print(f"- {METRICS_JSON}")


if __name__ == "__main__":
    main()
