import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score, recall_score
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("Milk_Grading__1_.csv", sep=None, engine="python")

# Remove leading/trailing spaces and normalize column names
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(r"\s+", "", regex=True)

print("Columns:", df.columns.tolist())

# Remove incomplete rows and duplicate records
df = df.dropna().drop_duplicates().reset_index(drop=True)

print("Unique dataset shape:", df.shape)
print("Grade distribution:")
print(df["Grade"].value_counts())

# Features and target
X = df[
    ["pH", "Temprature", "Taste", "Odor",
     "Fat", "Turbidity", "Colour"]
].copy()

y = df["Grade"].map({0: 0, 0.5: 1, 1: 2})

if y.isna().any():
    raise ValueError(
        "Unexpected Grade values. Check the dataset's Grade column."
    )

# Feature engineering: must match the model's existing training code
X["pH_deviation"] = abs(X["pH"] - 6.7)
X["temp_risk"] = (X["Temprature"] > 40).astype(int)
X["sensory_score"] = (
    X["Taste"] + X["Odor"] + (1 - X["Turbidity"])
)
X["colour_norm"] = X["Colour"] / 255
X["pH_deviation"] = abs(X["pH"] - 6.7)

X["temp_risk"] = (X["Temprature"] > 40).astype(int)

X["sensory_score"] = (
    X["Taste"] + X["Odor"] + (1 - X["Turbidity"])
)

X["colour_norm"] = X["Colour"] / 255.0

X["pH_colour"] = X["pH_deviation"] * X["colour_norm"]

X["turb_temp"] = X["Turbidity"] * X["temp_risk"]

# Use identical CV splits for both models
cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "macro_f1": make_scorer(
        f1_score, average="macro", zero_division=0
    ),
    "macro_recall": make_scorer(
        recall_score, average="macro", zero_division=0
    )
}

models = {
    "Original XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    ),

    "Tuned XGBoost": XGBClassifier(
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
        n_jobs=-1
    )
}

results = []

for name, model in models.items():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y.astype(int),
        cv=cv,
        scoring=scoring,
        n_jobs=1
    )

    row = {"Model": name}

    for metric in scoring:
        values = scores[f"test_{metric}"]
        row[f"{metric}_mean"] = values.mean() * 100
        row[f"{metric}_std"] = values.std() * 100

    results.append(row)

results_df = pd.DataFrame(results)

print("\nRepeated 5-fold CV results (50 folds per model):")
print(results_df.round(2).to_string(index=False))

results_df.to_csv("xgboost_comparison_results.csv", index=False)
print("\nSaved results to xgboost_comparison_results.csv")
