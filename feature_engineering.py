import numpy as np
import pandas as pd

INPUT_FILE = r"C:\Users\T9981\AeroHydro-PINN\processed\cleaned_combined_wind_experiments.csv"
OUTPUT_FILE = r"C:\Users\T9981\AeroHydro-PINN\processed\engineered_features.csv"

print("Starting feature engineering...")
print("Loading dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Dataset loaded.")
print("Rows:", len(df))
print("Columns:", len(df.columns))

df = df.drop_duplicates()

df.columns = df.columns.str.strip()

df = df.drop(columns=["Unnamed: 31"], errors="ignore")

df = df.rename(columns={"Unnamed: 0": "timestamp"})

df["timestamp"] = pd.to_datetime(
df["timestamp"],
errors="coerce"
)

df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["day_of_week"] = df["timestamp"].dt.dayofweek

df["wind_power_available"] = (
df["Wind Turbine Power (kW)"].notna().astype(int)
)

df["wind_power_filled"] = (
df["Wind Turbine Power (kW)"].fillna(0)
)

df["electrolyzer_power_W"] = (
df["Power Supply Average Voltage (Vdc)"]
*
df["H2E_f_PSU_A_Current"]
)

df["electrolyzer_power_kW"] = (
df["electrolyzer_power_W"] / 1000
)

df["current_difference_A"] = (
df["H2E_f_NLR_CurrentCmd (A)"]
-
df["H2E_f_PSU_A_Current"]
)

df["current_ratio"] = (
df["H2E_f_PSU_A_Current"]
/
df["H2E_f_NLR_CurrentCmd (A)"].replace(0, np.nan)
)

df["hydrogen_available"] = (
df["H2E_f_Elec_CalcProdRate"].notna().astype(int)
)

df["hydrogen_active"] = (
df["H2E_f_Elec_CalcProdRate"] > 0
).astype(int)

df["power_per_hydrogen"] = (
df["electrolyzer_power_kW"]
/
df["H2E_f_Elec_CalcProdRate"].replace(0, np.nan)
)

temperature_columns = [
"H2E_f_TE218_Temp",
"H2E_f_TE219_Temp",
"H2E_f_TE338_Temp",
"H2E_f_TE601_Temp",
"H2E_f_TT641_Temp",
"H2E_f_TT645_Temp",
"H2E_f_TT646_Temp",
"H2E_f_TT647_Temp"
]

df["mean_temperature"] = (
df[temperature_columns].mean(axis=1)
)

df["min_temperature"] = (
df[temperature_columns].min(axis=1)
)

df["max_temperature"] = (
df[temperature_columns].max(axis=1)
)

df["temperature_range"] = (
df["max_temperature"]
-
df["min_temperature"]
)

pressure_columns = [
"H2E_f_PT264_Pressure",
"H2E_f_PT307_Pressure",
"H2E_f_PT312_Pressure",
"H2E_f_PT604_Pressure"
]

df["mean_pressure"] = (
df[pressure_columns].mean(axis=1)
)

df["min_pressure"] = (
df[pressure_columns].min(axis=1)
)

df["max_pressure"] = (
df[pressure_columns].max(axis=1)
)

df["pressure_range"] = (
df["max_pressure"]
-
df["min_pressure"]
)

water_columns = [
"H2E_f_LS201_H2OLevel",
"H2E_f_LS301_H2OLevel"
]

df["mean_water_level"] = (
df[water_columns].mean(axis=1)
)

df["flow_available"] = (
df["IVAL_f_FM011_Flow"].notna().astype(int)
)

df["efficiency_suspect"] = (
df["Efficiency (kWh/kg)"] > 100
).astype(int)

df["experiment_id"] = (
df["Experiment"].astype("category").cat.codes
)

numeric_columns = df.select_dtypes(
include=np.number
).columns

df[numeric_columns] = (
df[numeric_columns]
.replace([np.inf, -np.inf], np.nan)
)

df = df.reset_index(drop=True)

df.to_csv(
OUTPUT_FILE,
index=False
)

print("")
print("==========================================")
print("FEATURE ENGINEERING COMPLETE")
print("==========================================")
print("Final rows:", len(df))
print("Final columns:", len(df.columns))
print("Output file:")
print(OUTPUT_FILE)
print("==========================================")