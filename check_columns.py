
import pandas as pd

df = pd.read_csv("Milk_Grading__1_.csv", sep=None, engine="python")

print("Column names in your dataset:")
for col in df.columns:
    print(repr(col))
