
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Milk_Grading__1_.csv"
MODEL_PATH = BASE_DIR / "milk_model.joblib"

FEATURE_COLS = [
    "pH", "Temprature", "Taste", "Odor",
    "Fat", "Turbidity", "Colour",
]

GRADE_LABEL = {0.0: "Low", 0.5: "Medium", 1.0: "High"}
GRADE_TO_IDX = {"Low": 0, "Medium": 1, "High": 2}


def main():
    try:
        if not CSV_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found: {CSV_PATH}"
            )

        # Load dataset and normalize column names.
        df = pd.read_csv(CSV_PATH, sep=None, engine="python")
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.replace(r"\s+", "", regex=True)
        )

        # Normalize supported column-name variations.
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
            for col in df.columns
            if col.lower() in aliases
        })

        required = FEATURE_COLS + ["Grade"]
        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}. "
                f"Found: {df.columns.tolist()}"
            )

        df = df[required].copy()

        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=required).drop_duplicates()
        df = df.reset_index(drop=True)

        # Map original grades to class indices.
        grade_names = df["Grade"].map(GRADE_LABEL)

        if grade_names.isna().any():
            raise ValueError(
                "Unsupported Grade values: "
                f"{df.loc[grade_names.isna(), 'Grade'].unique()}"
            )

        y = grade_names.map(GRADE_TO_IDX).to_numpy(dtype=int)

        if len(np.unique(y)) != 3:
            raise ValueError(
                "The dataset must contain Low, Medium, and High grades."
            )

        # Feature engineering: keep consistent with app.py.
        df["pH_deviation"] = (df["pH"] - 6.7).abs()
        df["temp_risk"] = (df["Temprature"] > 40).astype(int)

        df["sensory_score"] = (
            df["Taste"] + df["Odor"] + (1 - df["Turbidity"])
        )

        df["colour_norm"] = df["Colour"] / 255.0

        df["pH_colour"] = (
            df["pH_deviation"] * df["colour_norm"]
        )

        df["turb_temp"] = (
            df["Turbidity"] * df["temp_risk"]
        )

        feature_names = FEATURE_COLS + [
            "pH_deviation",
            "temp_risk",
            "sensory_score",
            "colour_norm",
            "pH_colour",
            "turb_temp",
        ]

        X = df[feature_names].to_numpy(dtype=np.float32)

        # Hold out a test set for a basic evaluation.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Tuned XGBoost candidate.
        model = XGBClassifier(
            n_estimators=200,
            max_depth=2,
            learning_rate=0.03,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        print("Training tuned XGBoost...")
        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)
        test_accuracy = float(np.mean(predictions == y_test))

        print("\nClassification report:")
        print(classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2],
            target_names=["Low", "Medium", "High"],
            zero_division=0,
        ))

        print("Confusion matrix (actual rows, predicted columns):")
        print(confusion_matrix(
            y_test, predictions, labels=[0, 1, 2]
        ))

        # Keep bundle keys compatible with the existing Flask app.
        bundle = {
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "class_map": {
                0: "Low",
                1: "Medium",
                2: "High",
            },
            "test_accuracy": test_accuracy,
        }

        joblib.dump(bundle, MODEL_PATH)

        print("\nTraining completed successfully.")
        print("Unique records used:", len(df))
        print("Training records:", len(X_train))
        print("Test records:", len(X_test))
        print(f"Holdout accuracy: {test_accuracy * 100:.2f}%")
        print("Saved model:", MODEL_PATH)

    except Exception as exc:
        print(f"\nERROR: {exc}")
        raise


if __name__ == "__main__":
    main()
