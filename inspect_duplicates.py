
from pathlib import Path
import pandas as pd

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

    df = df[FEATURES + ["Grade"]].copy()

    print("\n1. Most frequent complete records")
    print(df.value_counts().head(15).to_string())

    print("\n2. Number of unique values per input feature")
    print(df[FEATURES].nunique().to_string())

    print("\n3. Number of unique full records")
    print(df.drop_duplicates().shape[0], "of", len(df))

    print("\n4. Grade distribution for each feature combination")
    grade_counts = (
        df.groupby(FEATURES, dropna=False)["Grade"]
        .agg(["count", "nunique", "min", "max"])
        .sort_values("count", ascending=False)
    )

    print(grade_counts.head(15).to_string())

    print("\n5. Example repeated records")
    print(
        df[df.duplicated(subset=FEATURES, keep=False)]
        .sort_values(FEATURES)
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
