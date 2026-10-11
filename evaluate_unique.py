
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Milk_Grading__1_.csv"

FEATURES = [
    "pH", "Temprature", "Taste", "Odor",
    "Fat", "Turbidity", "Colour"
]


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
        c: aliases[c.lower()]
        for c in df.columns if c.lower() in aliases
    })

    required = FEATURES + ["Grade"]
    df = df[required].copy()

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().drop_duplicates().reset_index(drop=True)

    grade_map = {0.0: 0, 0.5: 1, 1.0: 2}
    y_series = df["Grade"].map(grade_map)

    if y_series.isna().any():
        raise ValueError("Unexpected Grade values.")

    y = y_series.astype(int).to_numpy()

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

    print("Unique rows:", len(df))
    print("Grade counts:", dict(zip(*np.unique(y, return_counts=True))))

    counts = np.bincount(y, minlength=3)
    n_splits = min(5, int(counts.min()))

    if n_splits < 2:
        raise ValueError(
            "Not enough unique records in every grade for cross-validation."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    xgb = XGBClassifier(
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

    svm = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            random_state=42,
        ),
    )

    ensemble = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("boost", xgb),
            ("svm", svm),
        ],
        voting="soft",
        weights=[1, 1.5, 1],
    )

    models = {
        "Random Forest": rf,
        "XGBoost": xgb,
        "Voting Ensemble": ensemble,
        "SVM": svm,
    }

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "recall_macro": "recall_macro",
    }

    results = []

    for name, model in models.items():
        pipeline = make_pipeline(StandardScaler(), model)

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        results.append({
            "Model": name,
            "Accuracy mean (%)":
                scores["test_accuracy"].mean() * 100,
            "Accuracy std (%)":
                scores["test_accuracy"].std() * 100,
            "Macro F1 (%)":
                scores["test_f1_macro"].mean() * 100,
            "Macro Recall (%)":
                scores["test_recall_macro"].mean() * 100,
        })

    result_df = pd.DataFrame(results).sort_values(
        "Macro F1 (%)", ascending=False
    )

    print("\nRESULTS ON UNIQUE RECORDS")
    print(result_df.round(2).to_string(index=False))

    result_df.to_csv(
        BASE_DIR / "unique_validation_results.csv",
        index=False,
    )

    print("\nSaved: unique_validation_results.csv")


if __name__ == "__main__":
    main()
