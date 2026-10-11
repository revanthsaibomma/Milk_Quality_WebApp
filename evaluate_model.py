
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Milk_Grading__1_.csv"

FEATURE_COLS = [
    "pH", "Temprature", "Taste", "Odor",
    "Fat", "Turbidity", "Colour"
]


def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

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

    df = df.rename(
        columns={
            col: aliases[col.lower()]
            for col in df.columns
            if col.lower() in aliases
        }
    )

    required = FEATURE_COLS + ["Grade"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            f"Found: {df.columns.tolist()}"
        )

    df = df[required].copy()

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)

    # Dataset's documented numeric labels:
    # 0 = Low, 0.5 = Medium, 1 = High.
    # Also supports 2 as High for datasets using 0/1/2.
    numeric_grade_map = {
        0.0: 0,
        0.5: 1,
        1.0: 2,
        2.0: 2,
    }

    y = df["Grade"].map(numeric_grade_map)

    if y.isna().any():
        raise ValueError(
            "Unexpected Grade values. Check the dataset's label encoding."
        )

    y = y.astype(int).to_numpy()

    # Feature engineering matching train_model.py.
    df["pH_deviation"] = (df["pH"] - 6.7).abs()
    df["temp_risk"] = (df["Temprature"] > 40).astype(int)
    df["sensory_score"] = (
        df["Taste"] + df["Odor"] + (1 - df["Turbidity"])
    )
    df["colour_norm"] = df["Colour"] / 255.0
    df["pH_colour"] = df["pH_deviation"] * df["colour_norm"]
    df["turb_temp"] = df["Turbidity"] * df["temp_risk"]

    feature_names = FEATURE_COLS + [
        "pH_deviation",
        "temp_risk",
        "sensory_score",
        "colour_norm",
        "pH_colour",
        "turb_temp",
    ]

    X = df[feature_names].to_numpy(dtype=np.float32)

    return X, y


def main():
    X, y = load_data()

    classes, counts = np.unique(y, return_counts=True)

    print("Total valid samples:", len(y))
    print("Class counts:", dict(zip(classes, counts)))

    if len(classes) < 2:
        raise ValueError("At least two grade classes are required.")

    # Every class must have at least five examples for 5-fold CV.
    if counts.min() < 5:
        raise ValueError(
            "Some classes have fewer than five examples. "
            "Use fewer CV folds or collect more samples."
        )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            random_state=42,
        )),
    ])

    try:
        from xgboost import XGBClassifier

        boost_model = XGBClassifier(
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
        print("\nUsing XGBoost.")

    except ImportError:
        boost_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
        )
        print("\nUsing Gradient Boosting.")

    boost = Pipeline([
        ("scaler", StandardScaler()),
        ("model", boost_model),
    ])

    ensemble = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("boost", boost),
            ("svm", svm),
        ],
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
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
    }

    results = []

    print("\nRunning stratified 5-fold cross-validation...")

    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        result = {
            "Model": name,
            "Accuracy mean (%)": scores["test_accuracy"].mean() * 100,
            "Accuracy std (%)": scores["test_accuracy"].std() * 100,
            "Macro F1 (%)": scores["test_f1_macro"].mean() * 100,
            "Macro Precision (%)": (
                scores["test_precision_macro"].mean() * 100
            ),
            "Macro Recall (%)": (
                scores["test_recall_macro"].mean() * 100
            ),
        }

        results.append(result)

    results_df = pd.DataFrame(results).sort_values(
        by="Macro F1 (%)",
        ascending=False,
    )

    print("\nCROSS-VALIDATION RESULTS")
    print(results_df.round(2).to_string(index=False))

    output_path = BASE_DIR / "validation_results.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")
    print(
        "\nUse the mean and variation across folds to compare models. "
        "Cross-validation is not a substitute for an independent "
        "external test set."
    )


if __name__ == "__main__":
    main()
