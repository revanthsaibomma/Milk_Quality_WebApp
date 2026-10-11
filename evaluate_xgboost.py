
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Milk_Grading__1_.csv"

FEATURES = [
    "pH", "Temprature", "Taste", "Odor",
    "Fat", "Turbidity", "Colour"
]

LABELS = ["Low", "Medium", "High"]


def main():
    df = pd.read_csv(CSV_PATH)

    if len(df.columns) == 1:
        df = pd.read_csv(CSV_PATH, sep=r"\s+")

    df.columns = df.columns.astype(str).str.strip()

    aliases = {
        "ph": "pH",
        "temperature": "Temprature",
        "temprature": "Temprature",
        "taste": "Taste",
        "odor": "Odor",
        "fat": "Fat",
        "turbidity": "Turbidity",
        "colour": "Colour",
        "color": "Colour",
        "grade": "Grade",
    }

    df = df.rename(columns={
        col: aliases[col.lower()]
        for col in df.columns if col.lower() in aliases
    })

    df = df[FEATURES + ["Grade"]].copy()

    for col in FEATURES + ["Grade"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().drop_duplicates().reset_index(drop=True)

    grade_map = {0.0: 0, 0.5: 1, 1.0: 2}
    y_series = df["Grade"].map(grade_map)

    if y_series.isna().any():
        raise ValueError("Unexpected Grade values.")

    y = y_series.astype(int).to_numpy()

    # Match feature engineering in train_model.py.
    X_df = df[FEATURES].copy()
    X_df["pH_deviation"] = (df["pH"] - 6.7).abs()
    X_df["temp_risk"] = (df["Temprature"] > 40).astype(int)
    X_df["sensory_score"] = (
        df["Taste"] + df["Odor"] + (1 - df["Turbidity"])
    )
    X_df["colour_norm"] = df["Colour"] / 255.0
    X_df["pH_colour"] = (
        X_df["pH_deviation"] * X_df["colour_norm"]
    )
    X_df["turb_temp"] = (
        df["Turbidity"] * X_df["temp_risk"]
    )

    X = X_df.to_numpy(dtype=np.float32)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    pipeline = make_pipeline(StandardScaler(), model)

    # Every prediction comes from a model that did not train on that row.
    predictions = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv,
        method="predict",
    )

    print("\nUNIQUE-RECORD XGBOOST EVALUATION")
    print("Number of unique records:", len(df))
    print(f"Accuracy: {accuracy_score(y, predictions) * 100:.2f}%")

    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(
        pd.DataFrame(
            confusion_matrix(y, predictions, labels=[0, 1, 2]),
            index=[f"Actual {label}" for label in LABELS],
            columns=[f"Predicted {label}" for label in LABELS],
        )
    )

    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            labels=[0, 1, 2],
            target_names=LABELS,
            digits=3,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
