from pathlib import Path
import pandas as pd


DATA_DIR = Path("data")


def read_csv_safely(file_path):
    """
    Try common encodings until the CSV can be read.
    """

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

            print(f"Successfully read using encoding: {encoding}")
            return df

        except UnicodeDecodeError:
            continue

    raise ValueError(
        f"Could not read {file_path.name} using the available encodings."
    )


def inspect_dataset(file_path):

    print("\n" + "=" * 90)
    print(f"DATASET: {file_path.name}")
    print("=" * 90)

    df = read_csv_safely(file_path)

    print("\n========== BASIC INFORMATION ==========")

    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nColumn Names:")

    for i, column in enumerate(df.columns, start=1):
        print(f"{i}. {column}")

    print("\n========== FIRST 5 ROWS ==========")

    print(df.head())

    print("\n========== DATA TYPES ==========")

    print(df.dtypes)

    print("\n========== MISSING VALUES ==========")

    missing = df.isnull().sum()

    missing_percentage = (
        df.isnull().mean() * 100
    ).round(2)

    missing_table = pd.DataFrame({
        "Missing_Count": missing,
        "Missing_Percentage": missing_percentage
    })

    print(
        missing_table[
            missing_table["Missing_Count"] > 0
        ].sort_values(
            "Missing_Count",
            ascending=False
        )
    )

    print("\n========== DUPLICATES ==========")

    print(
        "Duplicate rows:",
        df.duplicated().sum()
    )

    print("\n========== NUMERICAL COLUMNS ==========")

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    print(numerical_columns)

    print("\n========== CATEGORICAL COLUMNS ==========")

    categorical_columns = df.select_dtypes(
        exclude="number"
    ).columns.tolist()

    print(categorical_columns)

    print("\n========== STATISTICS ==========")

    print(df.describe(include="all").T)

    return df


# ==========================================================
# DATASET 1
# ==========================================================

electrolyzer_file = (
    DATA_DIR / "combined_wind_experiments.csv"
)

if electrolyzer_file.exists():

    electrolyzer_df = inspect_dataset(
        electrolyzer_file
    )

else:

    print(
        f"File not found: {electrolyzer_file}"
    )


# ==========================================================
# DATASET 2
# ==========================================================

scada_file = (
    DATA_DIR / "Aventa_AV7_IET_OST_SCADA.csv"
)

if scada_file.exists():

    scada_df = inspect_dataset(
        scada_file
    )

else:

    print(
        f"File not found: {scada_file}"
    )