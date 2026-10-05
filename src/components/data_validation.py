"""
Data Validation component (Stage 2).

This file only LOOKS at a dataset and prints a report. It never changes the
data and never trains a model.

Run from the project root:
    python -m src.components.data_validation --module diabetes
    python -m src.components.data_validation --module stroke
    python -m src.components.data_validation --module mental_health
"""

import argparse
import os

import pandas as pd

# Where each raw file should be, and what its target column is called.
# These match MODULE_CONFIG in data_ingestion.py.
DATASETS = {
    "diabetes": {
        "path": os.path.join("data", "raw", "diabetes", "diabetes_raw.csv"),
        "target": "Outcome",
        "type": "tabular",
        # In this medical data, 0 is NOT a possible real value for these columns.
        "zero_means_missing": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
    },
    "stroke": {
        "path": os.path.join("data", "raw", "stroke", "stroke_raw.csv"),
        "target": "stroke",
        "type": "tabular",
        "zero_means_missing": [],
    },
    "mental_health": {
        "path": os.path.join("data", "raw", "mental_health", "mental_health_raw.csv"),
        "target": "label",
        "type": "text",
        "text_column": "text",
    },
}


def show(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def check_tabular(df, target, zero_means_missing):
    show("Columns and data types")
    print(df.dtypes.to_string())

    show("Missing values (NaN)")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string() if missing.sum() > 0 else "No NaN values")

    show("Strange placeholder text such as 'Unknown', 'N/A', '?'")
    found = False
    for col in df.select_dtypes(include="object").columns:
        counts = df[col].value_counts()
        for bad in ["Unknown", "unknown", "N/A", "NA", "?", "", " "]:
            if bad in counts.index:
                print(f"{col}: '{bad}' appears {counts[bad]} times")
                found = True
    if not found:
        print("None found")

    show("Duplicate rows")
    print("Exact duplicate rows:", df.duplicated().sum())

    show("Target distribution")
    print(df[target].value_counts().to_string())
    print((df[target].value_counts(normalize=True) * 100).round(2).to_string())

    show("Zeros in columns where zero is clinically impossible")
    if zero_means_missing:
        for col in zero_means_missing:
            print(f"{col}: {(df[col] == 0).sum()} zeros")
    else:
        print("(none configured for this module)")

    show("Numeric summary (min / max help you spot impossible values)")
    print(df.describe().T.round(3).to_string())

    show("Values repeated suspiciously often (possible filled-in values)")
    for col in df.select_dtypes(include="number").columns:
        if col == target or df[col].nunique() < 15:
            continue
        # look at the 3 most common values of this column
        for top_value, top_count in df[col].value_counts().head(3).items():
            if top_count / len(df) > 0.10:
                split = df[df[col] == top_value][target].value_counts().to_dict()
                print(f"{col}: value {top_value} appears {top_count} times, target split = {split}")


def check_text(df, target, text_column):
    show("Columns and data types")
    print(df.dtypes.to_string())

    show("Missing values")
    print(df.isna().sum().to_string())

    show("Class distribution")
    print(df[target].value_counts().to_string())
    print((df[target].value_counts(normalize=True) * 100).round(2).to_string())

    show("Duplicates")
    print("Duplicate texts:", df[text_column].duplicated().sum())
    same_text = df[df[text_column].duplicated(keep=False)]
    conflict = same_text.groupby(text_column)[target].nunique()
    print("Same text with DIFFERENT labels:", (conflict > 1).sum())

    show("Text quality")
    text = df[text_column].fillna("").astype(str)
    words = text.str.split().str.len()
    print("Empty texts:", (text.str.strip() == "").sum())
    print("Very short texts (< 3 words):", (words < 3).sum())
    print("Word count summary:")
    print(words.describe().round(1).to_string())
    print("Texts containing a URL:", text.str.contains(r"http|www", case=False).sum())


def validate(module):
    cfg = DATASETS[module]
    show(f"DATASET VALIDATION: {module}")
    if not os.path.exists(cfg["path"]):
        print(f"FILE NOT FOUND: {cfg['path']}")
        print("Put the dataset there first, then run this again.")
        return

    df = pd.read_csv(cfg["path"])
    print("File   :", cfg["path"])
    print("Rows   :", df.shape[0])
    print("Columns:", df.shape[1])
    print("Names  :", list(df.columns))

    if cfg["target"] not in df.columns:
        print(f"\nTarget column '{cfg['target']}' is NOT in the file.")
        print("Either rename the column or change DATASETS in this file.")
        return

    if cfg["type"] == "tabular":
        check_tabular(df, cfg["target"], cfg["zero_means_missing"])
    else:
        if cfg["text_column"] not in df.columns:
            print(f"\nText column '{cfg['text_column']}' is NOT in the file.")
            return
        check_text(df, cfg["target"], cfg["text_column"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True, choices=list(DATASETS))
    args = parser.parse_args()
    validate(args.module)