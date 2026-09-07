from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data")
PROCESSED_DIR = Path("processed")

PROCESSED_DIR.mkdir(exist_ok=True)


# ============================================================
# SAFE CSV READER
# ============================================================

def read_csv_safely(file_path):

    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1"
    ]

    for encoding in encodings:

        try:

            df = pd.read_csv(
                file_path,
                encoding=encoding,
                low_memory=False
            )

            print(
                f"Successfully loaded {file_path.name} "
                f"using {encoding}"
            )

            return df

        except UnicodeDecodeError:

            continue

    raise ValueError(
        f"Unable to read {file_path.name}"
    )


# ============================================================
# LOAD DATA
# ============================================================

file_path = DATA_DIR / "combined_wind_experiments.csv"

df = read_csv_safely(file_path)


print("\n==========================================")
print("RAW DATASET")
print("==========================================")

print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# COLUMN CLEANING
# ============================================================

# Remove accidental unnamed columns
unnamed_columns = [
    column
    for column in df.columns
    if column.startswith("Unnamed:")
]

print("\nUnnamed columns:")
print(unnamed_columns)


# Rename first unnamed column to timestamp
if "Unnamed: 0" in df.columns:

    df.rename(
        columns={
            "Unnamed: 0": "timestamp"
        },
        inplace=True
    )


# Remove the mostly empty Unnamed:31 column
if "Unnamed: 31" in df.columns:

    df.drop(
        columns=["Unnamed: 31"],
        inplace=True
    )


# ============================================================
# TIMESTAMP PROCESSING
# ============================================================

if "timestamp" in df.columns:

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    print(
        "\nInvalid timestamps:",
        df["timestamp"].isna().sum()
    )

    # Time-based features

    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["hour"] = df["timestamp"].dt.hour


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

duplicates_before = df.duplicated().sum()

print(
    "\nDuplicate rows before cleaning:",
    duplicates_before
)

df = df.drop_duplicates()

print(
    "Duplicate rows after cleaning:",
    df.duplicated().sum()
)


# ============================================================
# INFINITE VALUES
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns

df[numeric_columns] = df[numeric_columns].replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# MISSING VALUE REPORT
# ============================================================

missing_report = pd.DataFrame({

    "column":
        df.columns,

    "missing_count":
        df.isna().sum(),

    "missing_percentage":
        (
            df.isna().mean() * 100
        ).round(2)

})


missing_report = missing_report.sort_values(
    "missing_percentage",
    ascending=False
)


missing_report.to_csv(
    PROCESSED_DIR / "missing_value_report.csv",
    index=False
)


print("\n==========================================")
print("MISSING VALUE REPORT")
print("==========================================")

print(
    missing_report[
        missing_report["missing_count"] > 0
    ]
)


# ============================================================
# NUMERICAL SUMMARY
# ============================================================

summary = df.describe().T

summary.to_csv(
    PROCESSED_DIR / "numerical_summary.csv"
)


# ============================================================
# EXPERIMENT SUMMARY
# ============================================================

if "Experiment" in df.columns:

    experiment_summary = (
        df["Experiment"]
        .value_counts()
        .reset_index()
    )

    experiment_summary.columns = [
        "Experiment",
        "Row_Count"
    ]

    experiment_summary.to_csv(
        PROCESSED_DIR /
        "experiment_summary.csv",
        index=False
    )

    print("\n==========================================")
    print("EXPERIMENT COUNTS")
    print("==========================================")

    print(experiment_summary)


# ============================================================
# SAVE CLEANED DATA
# ============================================================

output_file = (
    PROCESSED_DIR /
    "cleaned_combined_wind_experiments.csv"
)

df.to_csv(
    output_file,
    index=False
)


print("\n==========================================")
print("DATA ENGINEERING COMPLETE")
print("==========================================")

print("Final rows:", len(df))
print("Final columns:", len(df.columns))

print(
    "\nCleaned dataset saved to:"
)

print(output_file)