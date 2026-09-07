import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "processed",
    "cleaned_combined_wind_experiments.csv"
)

EDA_FOLDER = os.path.join(
    PROJECT_ROOT,
    "processed",
    "eda"
)

os.makedirs(EDA_FOLDER, exist_ok=True)

print("=" * 70)
print("AeroHydro-PINN - Exploratory Data Analysis")
print("=" * 70)

# ============================================================
# 2. LOAD CLEANED DATA
# ============================================================

print("\nLoading processed dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("BASIC INFORMATION")
print("=" * 70)

print("\nNumber of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nData types:")
print(df.dtypes)

# ============================================================
# 4. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

missing = df.isnull().sum()

missing = missing[missing > 0].sort_values(ascending=False)

print("\nMissing values:")
print(missing)

if len(missing) > 0:

    plt.figure(figsize=(12, 7))

    missing.plot(kind="bar")

    plt.title("Missing Values by Feature")
    plt.xlabel("Feature")
    plt.ylabel("Number of Missing Values")
    plt.xticks(rotation=90)
    plt.tight_layout()

    output = os.path.join(
        EDA_FOLDER,
        "missing_values.png"
    )

    plt.savefig(output, dpi=300)
    plt.close()

    print("\nSaved:", output)

# ============================================================
# 5. NUMERICAL FEATURES
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

print("\n" + "=" * 70)
print("NUMERICAL FEATURES")
print("=" * 70)

print("\nNumber of numerical features:", len(numeric_columns))

print("\nNumerical features:")
for column in numeric_columns:
    print("-", column)

# ============================================================
# 6. WIND TURBINE POWER DISTRIBUTION
# ============================================================

power_column = "Wind Turbine Power (kW)"

if power_column in df.columns:

    plt.figure(figsize=(10, 6))

    df[power_column].dropna().plot(
        kind="hist",
        bins=50
    )

    plt.title("Wind Turbine Power Distribution")
    plt.xlabel("Wind Turbine Power (kW)")
    plt.ylabel("Frequency")
    plt.tight_layout()

    output = os.path.join(
        EDA_FOLDER,
        "wind_turbine_power_distribution.png"
    )

    plt.savefig(output, dpi=300)
    plt.close()

    print("\nSaved:", output)

# ============================================================
# 7. ELECTROLYZER CURRENT DISTRIBUTION
# ============================================================

current_column = "H2E_f_NLR_CurrentCmd (A)"

if current_column in df.columns:

    plt.figure(figsize=(10, 6))

    df[current_column].dropna().plot(
        kind="hist",
        bins=50
    )

    plt.title("Electrolyzer Current Distribution")
    plt.xlabel("Electrolyzer Current (A)")
    plt.ylabel("Frequency")
    plt.tight_layout()

    output = os.path.join(
        EDA_FOLDER,
        "electrolyzer_current_distribution.png"
    )

    plt.savefig(output, dpi=300)
    plt.close()

    print("\nSaved:", output)

# ============================================================
# 8. HYDROGEN PRODUCTION DISTRIBUTION
# ============================================================

h2_column = "H2E_f_Elec_CalcProdRate"

if h2_column in df.columns:

    plt.figure(figsize=(10, 6))

    df[h2_column].dropna().plot(
        kind="hist",
        bins=50
    )

    plt.title("Hydrogen Production Rate Distribution")
    plt.xlabel("Hydrogen Production Rate")
    plt.ylabel("Frequency")
    plt.tight_layout()

    output = os.path.join(
        EDA_FOLDER,
        "hydrogen_production_distribution.png"
    )

    plt.savefig(output, dpi=300)
    plt.close()

    print("\nSaved:", output)

# ============================================================
# 9. EFFICIENCY DISTRIBUTION
# ============================================================

efficiency_column = "Efficiency (kWh/kg)"

if efficiency_column in df.columns:

    efficiency_data = df[efficiency_column].dropna()

    # Remove extreme values only for visualization.
    # Original data remains unchanged.
    upper_limit = efficiency_data.quantile(0.99)

    filtered_efficiency = efficiency_data[
        efficiency_data <= upper_limit
    ]

    plt.figure(figsize=(10, 6))

    filtered_efficiency.plot(
        kind="hist",
        bins=50
    )

    plt.title(
        "Electrolyzer Efficiency Distribution "
        "(99th Percentile View)"
    )

    plt.xlabel("Efficiency (kWh/kg)")
    plt.ylabel("Frequency")

    plt.tight_layout()

    output = os.path.join(
        EDA_FOLDER,
        "efficiency_distribution.png"
    )

    plt.savefig(output, dpi=300)
    plt.close()

    print("\nSaved:", output)

# ============================================================
# 10. WIND POWER vs ELECTROLYZER CURRENT
# ============================================================

if (
    power_column in df.columns
    and current_column in df.columns
):

    plot_df = df[
        [power_column, current_column]
    ].dropna()

    plt.figure(figsize=(10, 6))

    plt.scatter(
        plot_df[power_column],
        plot_df[current_column],
        alpha=0.3
    )

    plt.title(
        "Wind Turbine Power vs Electrolyzer Current"
    )

    plt.xlabel("Wind Turbine Power (kW)")
    plt.ylabel("Electrolyzer Current (A)")

    plt.tight_layout()

    output = os.path.join(
        EDA_FOLDER,
        "wind_power_vs_electrolyzer_current.png"
    )

    plt.savefig(output, dpi=300)
    plt.close()

    print("\nSaved:", output)

# ============================================================
# 11. WIND POWER vs HYDROGEN PRODUCTION
# ============================================================

if (
    power_column in df.columns
    and h2_column in df.columns
):

    plot_df = df[
        [power_column, h2_column]
    ].dropna()

    plt.figure(figsize=(10, 6))

    plt.scatter(
        plot_df[power_column],
        plot_df[h2_column],
        alpha=0.3
    )

    plt.title(
        "Wind Turbine Power vs Hydrogen Production"
    )

    plt.xlabel("Wind Turbine Power (kW)")
    plt.ylabel("Hydrogen Production Rate")

    plt.tight_layout()

    output = os.path.join(
        EDA_FOLDER,
        "wind_power_vs_hydrogen.png"
    )

    plt.savefig(output, dpi=300)
    plt.close()

    print("\nSaved:", output)

# ============================================================
# 12. ELECTROLYZER CURRENT vs HYDROGEN PRODUCTION
# ============================================================

if (
    current_column in df.columns
    and h2_column in df.columns
):

    plot_df = df[
        [current_column, h2_column]
    ].dropna()

    plt.figure(figsize=(10, 6))

    plt.scatter(
        plot_df[current_column],
        plot_df[h2_column],
        alpha=0.3
    )

    plt.title(
        "Electrolyzer Current vs Hydrogen Production"
    )

    plt.xlabel("Electrolyzer Current (A)")
    plt.ylabel("Hydrogen Production Rate")

    plt.tight_layout()

    output = os.path.join(
        EDA_FOLDER,
        "current_vs_hydrogen.png"
    )

    plt.savefig(output, dpi=300)
    plt.close()

    print("\nSaved:", output)

# ============================================================
# 13. CORRELATION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

correlation = df[numeric_columns].corr()

print("\nCorrelation matrix:")
print(correlation.round(3))

# Save correlation matrix as CSV

correlation_file = os.path.join(
    EDA_FOLDER,
    "correlation_matrix.csv"
)

correlation.to_csv(correlation_file)

print("\nSaved:", correlation_file)

# Plot heatmap manually using matplotlib

plt.figure(figsize=(18, 14))

plt.imshow(
    correlation,
    aspect="auto"
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Feature Correlation Matrix")

plt.tight_layout()

output = os.path.join(
    EDA_FOLDER,
    "correlation_heatmap.png"
)

plt.savefig(output, dpi=300)
plt.close()

print("\nSaved:", output)

# ============================================================
# 14. EXPERIMENT ANALYSIS
# ============================================================

experiment_column = "Experiment"

if experiment_column in df.columns:

    print("\n" + "=" * 70)
    print("EXPERIMENT ANALYSIS")
    print("=" * 70)

    experiment_counts = (
        df[experiment_column]
        .value_counts()
    )

    print("\nNumber of experiments:")
    print(len(experiment_counts))

    print("\nRows per experiment:")
    print(experiment_counts)

    plt.figure(figsize=(12, 7))

    experiment_counts.plot(
        kind="bar"
    )

    plt.title("Number of Observations per Experiment")
    plt.xlabel("Experiment")
    plt.ylabel("Number of Observations")

    plt.xticks(rotation=90)

    plt.tight_layout()

    output = os.path.join(
        EDA_FOLDER,
        "experiment_comparison.png"
    )

    plt.savefig(output, dpi=300)
    plt.close()

    print("\nSaved:", output)

# ============================================================
# 15. OUTLIER ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("OUTLIER ANALYSIS")
print("=" * 70)

outlier_results = []

for column in numeric_columns:

    values = df[column].dropna()

    if len(values) == 0:
        continue

    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outlier_count = (
        (values < lower) |
        (values > upper)
    ).sum()

    outlier_percentage = (
        outlier_count / len(values)
    ) * 100

    outlier_results.append({
        "Feature": column,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower_Bound": lower,
        "Upper_Bound": upper,
        "Outlier_Count": outlier_count,
        "Outlier_Percentage": outlier_percentage
    })

outlier_df = pd.DataFrame(
    outlier_results
)

outlier_file = os.path.join(
    EDA_FOLDER,
    "outlier_analysis.csv"
)

outlier_df.to_csv(
    outlier_file,
    index=False
)

print("\nOutlier analysis saved:")
print(outlier_file)

# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print("\nEDA files have been saved inside:")

print(EDA_FOLDER)

print("\nGenerated files:")

for file in sorted(os.listdir(EDA_FOLDER)):
    print("-", file)

print("\nNext stage:")
print("Feature Engineering + FLORIS Wake Data Generation")

print("=" * 70)