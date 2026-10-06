import pandas as pd
import numpy as np

input_path = "data/raw/diabetes/diabetes_raw.csv"
output_path = "data/raw/diabetes/diabetes_clean.csv"

df = pd.read_csv(input_path)

# Replace suspicious filler values with missing values
df["Insulin"] = df["Insulin"].replace(102.5, np.nan)
df["Insulin"] = df["Insulin"].replace(169.5, np.nan)

df["SkinThickness"] = df["SkinThickness"].replace(27, np.nan)
df["SkinThickness"] = df["SkinThickness"].replace(32, np.nan)

df.to_csv(output_path, index=False)

print("Clean diabetes dataset created successfully!")
print("Saved to:", output_path)
print("Shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())