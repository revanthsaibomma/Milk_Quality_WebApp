
from pathlib import Path

import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Milk_Grading__1_.csv"

FEATURES = [
    "pH", "Temprature", "Taste", "Odor",
    "Fat", "Turbidity", "Colour"
]


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

    # Load comma-separated or whitespace-separated data.
    df = pd.read_csv(CSV_PATH)

    if len(df.columns) == 1:
        df = pd.read_csv(CSV_PATH, sep=r"\s+")

    df.columns = df.columns.astype(str).str.strip()

    # Standardize column names.
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

    required = FEATURES + ["Grade"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Detected columns: {df.columns.tolist()}"
        )

    df = df[required].copy()

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    original_count = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)

    print("\n========== DATASET SUMMARY ==========")
    print("Dataset:", CSV_PATH)
    print("Original rows:", original_count)
    print("Complete numeric rows:", len(df))
    print("Rows removed due to missing/invalid values:",
          original_count - len(df))

    # 1. Exact duplicate rows, including the Grade column.
    exact_duplicates = int(df.duplicated().sum())

    print("\n========== DUPLICATE CHECK ==========")
    print("Exact duplicate rows:", exact_duplicates)
    print(
        "Percentage of rows duplicated:",
        round(100 * exact_duplicates / len(df), 2)
        if len(df) else 0,
        "%"
    )

    # 2. Duplicate input features, regardless of Grade.
    feature_duplicates = df.duplicated(subset=FEATURES, keep=False)
    repeated_features = df.loc[feature_duplicates]

    print(
        "\nRows whose seven input features repeat:",
        int(feature_duplicates.sum())
    )

    # Same inputs but different grades may indicate inconsistent labels.
    if not repeated_features.empty:
        grade_counts = (
            repeated_features.groupby(FEATURES)["Grade"]
            .nunique()
        )
        conflicting_groups = int((grade_counts > 1).sum())

        print(
            "Repeated feature groups with conflicting grades:",
            conflicting_groups
        )

        if conflicting_groups:
            print("\nExamples of conflicting labels:")
            print(
                repeated_features.groupby(FEATURES)["Grade"]
                .agg(["count", "nunique", "min", "max"])
                .query("nunique > 1")
                .head(10)
            )

    # 3. Class distribution.
    print("\n========== GRADE DISTRIBUTION ==========")
    print(df["Grade"].value_counts(dropna=False).sort_index())
    print("\nGrade percentages:")
    print(
        (df["Grade"].value_counts(normalize=True) * 100)
        .sort_index()
        .round(2)
    )

    # 4. Check feature/target relationships.
    print("\n========== FEATURE CORRELATIONS ==========")
    print(
        df.corr(numeric_only=True)["Grade"]
        .drop("Grade")
        .sort_values(key=abs, ascending=False)
        .round(3)
    )

    # 5. Check whether the Grade can be predicted by a single feature.
    print("\n========== GRADE BY FEATURE ==========")

    for feature in FEATURES:
        grouped = df.groupby(feature)["Grade"].nunique()
        single_grade_values = int((grouped == 1).sum())
        total_values = len(grouped)

        print(
            f"{feature}: {single_grade_values}/{total_values} "
            "distinct feature values have only one observed grade."
        )

    # 6. Save repeated records for manual review.
    output_path = BASE_DIR / "duplicate_rows_review.csv"

    if not repeated_features.empty:
        repeated_features.sort_values(FEATURES).to_csv(
            output_path, index=False
        )
        print("\nRepeated-feature rows saved to:", output_path)
    else:
        print("\nNo repeated input-feature rows found.")

    print("\n========== INTERPRETATION ==========")
    print(
        "Duplicates do not automatically prove data leakage. "
        "Investigate whether records represent the same physical sample, "
        "repeated measurements, or separate valid samples."
    )
    print(
        "Feature correlations do not prove target leakage. "
        "Confirm how Grade was assigned and whether any input feature "
        "was derived from the target."
    )


if __name__ == "__main__":
    main()
