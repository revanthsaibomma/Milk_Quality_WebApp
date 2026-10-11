
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC

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
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required].copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    # Match the grade encoding used by your training script.
    grade_map = {0.0: 0, 0.5: 1, 1.0: 2}
    y_series = df["Grade"].map(grade_map)

    if y_series.isna().any():
        raise ValueError(
            "Unexpected Grade values. Expected 0.0, 0.5, and 1.0."
        )

    y = y_series.astype(int).to_numpy()

    # Group identical feature combinations so copies stay together.
    groups = pd.MultiIndex.from_frame(df[FEATURES]).factorize()[0]
    X_df = df[FEATURES].copy()

    # Apply the same feature engineering as the original model.
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

    print("Rows:", len(df))
    print("Unique feature groups:", len(np.unique(groups)))
    print("Grade counts:", dict(zip(*np.unique(y, return_counts=True))))

    if len(np.unique(groups)) < 5:
        raise ValueError("At least five unique feature groups are needed.")

    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=42
    )

    # Confirm that each validation fold contains all three classes.
    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups), start=1
    ):
        if len(np.unique(y[test_idx])) < 3:
            raise ValueError(
                f"Fold {fold} lacks one or more grade classes. "
                "Use fewer folds or revise the evaluation design."
            )

        overlap = set(groups[train_idx]) & set(groups[test_idx])
        if overlap:
            raise RuntimeError("A feature group leaked across folds.")

    try:
        from xgboost import XGBClassifier

        boost = XGBClassifier(
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
        print("Boosting model: XGBoost")

    except ImportError:
        boost = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
        )
        print("Boosting model: Gradient Boosting")

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    svm = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf", C=10, gamma="scale",
            probability=True, random_state=42
        ),
    )
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("boost", boost), ("svm", svm)],
        voting="soft",
        weights=[1, 1.5, 1],
    )

    models = {
        "Random Forest": rf,
        "Boosting": boost,
        "SVM": svm,
        "Voting Ensemble": ensemble,
    }

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "recall_macro": "recall_macro",
    }

    results = []

    for name, model in models.items():
        # Scaling is fitted inside each training fold.
        pipeline = make_pipeline(StandardScaler(), model)

        scores = cross_validate(
            pipeline,
            X,
            y,
            groups=groups,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        results.append({
            "Model": name,
            "Accuracy mean (%)": scores["test_accuracy"].mean() * 100,
            "Accuracy std (%)": scores["test_accuracy"].std() * 100,
            "Macro F1 (%)": scores["test_f1_macro"].mean() * 100,
            "Macro Recall (%)": scores["test_recall_macro"].mean() * 100,
        })

    result_df = pd.DataFrame(results).sort_values(
        "Macro F1 (%)", ascending=False
    )

    print("\nGROUPED CROSS-VALIDATION RESULTS")
    print(result_df.round(2).to_string(index=False))

    result_df.to_csv(BASE_DIR / "grouped_validation_results.csv", index=False)
    print("\nSaved: grouped_validation_results.csv")


if __name__ == "__main__":
    main()
