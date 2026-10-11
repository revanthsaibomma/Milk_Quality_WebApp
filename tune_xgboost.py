
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Milk_Grading__1_.csv"

FEATURES = [
    "pH", "Temprature", "Taste", "Odor",
    "Fat", "Turbidity", "Colour"
]


def load_data():
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
        c: aliases[c.lower()]
        for c in df.columns if c.lower() in aliases
    })

    df = df[FEATURES + ["Grade"]].copy()

    for col in FEATURES + ["Grade"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Deduplicate before evaluation.
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    y = df["Grade"].map({0.0: 0, 0.5: 1, 1.0: 2})

    if y.isna().any():
        raise ValueError("Unexpected Grade values.")

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

    return X_df.to_numpy(dtype=np.float32), y.astype(int).to_numpy()


def main():
    X, y = load_data()

    print("Unique records:", len(y))
    print("Class counts:", dict(zip(*np.unique(y, return_counts=True))))

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []

    # Small, deliberately limited grid for 83 records.
    configurations = product(
        [2, 3, 4],           # max_depth
        [0.03, 0.05, 0.1],  # learning_rate
        [100, 200],          # n_estimators
        [1, 3],               # min_child_weight
    )

    for depth, rate, trees, child_weight in configurations:
        model = XGBClassifier(
            n_estimators=trees,
            max_depth=depth,
            learning_rate=rate,
            min_child_weight=child_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        pipeline = make_pipeline(StandardScaler(), model)

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "f1_macro": "f1_macro",
                "recall_macro": "recall_macro",
            },
            n_jobs=1,
            error_score="raise",
        )

        results.append({
            "max_depth": depth,
            "learning_rate": rate,
            "n_estimators": trees,
            "min_child_weight": child_weight,
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "macro_f1": scores["test_f1_macro"].mean(),
            "macro_recall": scores["test_recall_macro"].mean(),
        })

    results_df = pd.DataFrame(results).sort_values(
        ["macro_f1", "accuracy_mean"],
        ascending=False,
    )

    print("\nTOP 10 CONFIGURATIONS")
    print(results_df.head(10).round(4).to_string(index=False))

    output_path = BASE_DIR / "xgboost_tuning_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nAll results saved to:", output_path)


if __name__ == "__main__":
    main()
