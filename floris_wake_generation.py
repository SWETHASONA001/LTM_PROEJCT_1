"""
AeroHydro-PINN
FLORIS Wake Feature Generation
Compatible with FLORIS 4.6.6

FLORIS source:
    C:/Users/T9981/AeroHydro-PINN/floris

Output:
    processed/floris_wake_features.csv

IMPORTANT:
    FLORIS returns power in watts.
    This script converts power to kW.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FLORIS_ROOT = PROJECT_ROOT / "floris"

OUTPUT_DIR = PROJECT_ROOT / "processed"

OUTPUT_FILE = (
    OUTPUT_DIR /
    "floris_wake_features.csv"
)


# ============================================================
# MAKE CLONED FLORIS AVAILABLE
# ============================================================

if FLORIS_ROOT.exists():
    sys.path.insert(
        0,
        str(FLORIS_ROOT)
    )


# ============================================================
# IMPORT FLORIS
# ============================================================

try:

    from floris import FlorisModel

except Exception as exc:

    print("\nERROR: FLORIS could not be imported.")
    print(exc)

    sys.exit(1)


# ============================================================
# FIND FLORIS CONFIGURATION
# ============================================================

def find_config():

    preferred = (
        FLORIS_ROOT
        / "examples"
        / "inputs"
        / "gch.yaml"
    )

    if preferred.exists():
        return preferred

    matches = list(
        FLORIS_ROOT.rglob("gch.yaml")
    )

    if matches:
        return matches[0]

    yaml_files = list(
        FLORIS_ROOT.rglob("*.yaml")
    )

    if yaml_files:
        return yaml_files[0]

    yml_files = list(
        FLORIS_ROOT.rglob("*.yml")
    )

    if yml_files:
        return yml_files[0]

    return None


# ============================================================
# CREATE WIND INPUT
# ============================================================

def create_wind_data():

    rows = []

    # Wind speed sweep
    wind_speeds = np.arange(
        4.0,
        16.1,
        0.5
    )

    # Wind direction
    wind_directions = [
        270.0
    ]

    # Turbulence intensity
    turbulence_values = [
        0.10
    ]

    for speed in wind_speeds:

        for direction in wind_directions:

            for turbulence in turbulence_values:

                rows.append(
                    {
                        "wind_speed_ms": float(speed),
                        "wind_direction_deg": float(direction),
                        "turbulence_intensity": float(turbulence)
                    }
                )

    return pd.DataFrame(rows)


# ============================================================
# RUN FLORIS
# ============================================================

def run_floris(
    config_file,
    wind_data
):

    print("\nLoading FLORIS model:")
    print(config_file)

    fmodel = FlorisModel(
        str(config_file)
    )

    print(
        "FLORIS model loaded successfully."
    )

    # --------------------------------------------------------
    # Wind inputs
    # --------------------------------------------------------

    wind_speeds = (
        wind_data[
            "wind_speed_ms"
        ]
        .to_numpy(
            dtype=float
        )
    )

    wind_directions = (
        wind_data[
            "wind_direction_deg"
        ]
        .to_numpy(
            dtype=float
        )
    )

    turbulence = (
        wind_data[
            "turbulence_intensity"
        ]
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # Set FLORIS conditions
    # --------------------------------------------------------

    print(
        "\nSetting wind conditions..."
    )

    fmodel.set(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        turbulence_intensities=turbulence
    )

    # --------------------------------------------------------
    # Run wake simulation
    # --------------------------------------------------------

    print(
        "Running FLORIS wake simulation..."
    )

    fmodel.run()

    print(
        "Wake simulation completed."
    )

    return fmodel


# ============================================================
# EXTRACT POWER
# ============================================================

def extract_power(
    fmodel
):

    # --------------------------------------------------------
    # Farm power
    # --------------------------------------------------------

    farm_power_w = np.asarray(
        fmodel.get_farm_power(),
        dtype=float
    )

    farm_power_w = np.squeeze(
        farm_power_w
    )

    # FLORIS returns W
    # Convert W -> kW

    farm_power_kw = (
        farm_power_w / 1000.0
    )

    # --------------------------------------------------------
    # Turbine power
    # --------------------------------------------------------

    turbine_power_w = np.asarray(
        fmodel.get_turbine_powers(),
        dtype=float
    )

    if turbine_power_w.ndim == 1:

        turbine_power_w = (
            turbine_power_w.reshape(
                -1,
                1
            )
        )

    # W -> kW

    turbine_power_kw = (
        turbine_power_w / 1000.0
    )

    return (
        farm_power_kw,
        turbine_power_kw
    )


# ============================================================
# CREATE FEATURES
# ============================================================

def create_features(
    wind_data,
    farm_power_kw,
    turbine_power_kw
):

    result = wind_data.copy()

    # --------------------------------------------------------
    # Number of turbines
    # --------------------------------------------------------

    number_of_turbines = (
        turbine_power_kw.shape[1]
    )

    result[
        "number_of_turbines"
    ] = number_of_turbines

    # --------------------------------------------------------
    # Farm power
    # --------------------------------------------------------

    result[
        "floris_farm_power_kw"
    ] = farm_power_kw

    # --------------------------------------------------------
    # Individual turbines
    # --------------------------------------------------------

    for i in range(
        number_of_turbines
    ):

        result[
            f"floris_turbine_{i + 1}_power_kw"
        ] = turbine_power_kw[
            :,
            i
        ]

    # --------------------------------------------------------
    # Installed capacity
    # --------------------------------------------------------
    #
    # gch.yaml has 3 x 5 MW turbines.
    # We calculate capacity from the simulated maximum.
    #

    installed_capacity_kw = (
        5000.0 *
        number_of_turbines
    )

    result[
        "installed_capacity_kw"
    ] = installed_capacity_kw

    # --------------------------------------------------------
    # Capacity factor
    # --------------------------------------------------------

    result[
        "farm_capacity_factor"
    ] = (
        result[
            "floris_farm_power_kw"
        ]
        /
        installed_capacity_kw
    )

    result[
        "farm_capacity_factor"
    ] = result[
        "farm_capacity_factor"
    ].clip(
        lower=0.0,
        upper=1.0
    )

    # --------------------------------------------------------
    # Power in MW
    # --------------------------------------------------------

    result[
        "floris_farm_power_mw"
    ] = (
        result[
            "floris_farm_power_kw"
        ]
        / 1000.0
    )

    # --------------------------------------------------------
    # Total turbine power
    # --------------------------------------------------------

    turbine_columns = [
        column
        for column in result.columns
        if column.startswith(
            "floris_turbine_"
        )
        and column.endswith(
            "_power_kw"
        )
    ]

    # --------------------------------------------------------
    # Turbine power imbalance
    # --------------------------------------------------------

    if len(turbine_columns) >= 2:

        turbine_values = (
            result[
                turbine_columns
            ]
            .to_numpy(
                dtype=float
            )
        )

        result[
            "turbine_power_std_kw"
        ] = np.std(
            turbine_values,
            axis=1
        )

        result[
            "turbine_power_min_kw"
        ] = np.min(
            turbine_values,
            axis=1
        )

        result[
            "turbine_power_max_kw"
        ] = np.max(
            turbine_values,
            axis=1
        )

        result[
            "turbine_power_range_kw"
        ] = (
            result[
                "turbine_power_max_kw"
            ]
            -
            result[
                "turbine_power_min_kw"
            ]
        )

    else:

        result[
            "turbine_power_std_kw"
        ] = 0.0

        result[
            "turbine_power_min_kw"
        ] = result[
            turbine_columns[0]
        ]

        result[
            "turbine_power_max_kw"
        ] = result[
            turbine_columns[0]
        ]

        result[
            "turbine_power_range_kw"
        ] = 0.0

    # --------------------------------------------------------
    # NOTE
    # --------------------------------------------------------
    #
    # We intentionally DO NOT call the incoming wind speed
    # "effective wind speed".
    #
    # Actual wake velocity should later be extracted from
    # the FLORIS flow field.
    #

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AeroHydro-PINN")
    print("FLORIS Wake Generation")
    print("FLORIS 4.6.6")
    print("=" * 70)

    # --------------------------------------------------------
    # Check repository
    # --------------------------------------------------------

    print(
        "\nChecking cloned FLORIS repository..."
    )

    if not FLORIS_ROOT.exists():

        print(
            "\nERROR: FLORIS repository not found:"
        )

        print(
            FLORIS_ROOT
        )

        sys.exit(1)

    print(
        "FLORIS repository found:"
    )

    print(
        FLORIS_ROOT
    )

    # --------------------------------------------------------
    # Find config
    # --------------------------------------------------------

    print(
        "\nSearching for FLORIS configuration..."
    )

    config_file = find_config()

    if config_file is None:

        print(
            "\nERROR: No FLORIS YAML configuration found."
        )

        sys.exit(1)

    print(
        "\nUsing FLORIS configuration:"
    )

    print(
        config_file
    )

    # --------------------------------------------------------
    # Wind data
    # --------------------------------------------------------

    print(
        "\nCreating FLORIS wind conditions..."
    )

    wind_data = (
        create_wind_data()
    )

    print(
        f"Wind conditions generated: "
        f"{len(wind_data)}"
    )

    print(
        f"Wind-speed range: "
        f"{wind_data['wind_speed_ms'].min():.1f}"
        f" to "
        f"{wind_data['wind_speed_ms'].max():.1f} m/s"
    )

    print(
        f"Wind direction: "
        f"{wind_data['wind_direction_deg'].iloc[0]:.1f}°"
    )

    print(
        f"Turbulence intensity: "
        f"{wind_data['turbulence_intensity'].iloc[0]:.2f}"
    )

    # --------------------------------------------------------
    # Run FLORIS
    # --------------------------------------------------------

    fmodel = run_floris(
        config_file,
        wind_data
    )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    print(
        "\nExtracting FLORIS results..."
    )

    (
        farm_power_kw,
        turbine_power_kw
    ) = extract_power(
        fmodel
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if len(farm_power_kw) != len(
        wind_data
    ):

        print(
            "\nERROR: Output length mismatch."
        )

        print(
            "Wind rows:",
            len(wind_data)
        )

        print(
            "Power rows:",
            len(farm_power_kw)
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Create dataset
    # --------------------------------------------------------

    result = create_features(
        wind_data,
        farm_power_kw,
        turbine_power_kw
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "FLORIS WAKE GENERATION COMPLETE"
    )
    print("=" * 70)

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        f"\nRows generated: "
        f"{len(result)}"
    )

    print(
        f"Columns generated: "
        f"{len(result.columns)}"
    )

    print(
        "\nPower units: kW"
    )

    print(
        "\nFarm power statistics:"
    )

    print(
        f"Minimum: "
        f"{result['floris_farm_power_kw'].min():.2f} kW"
    )

    print(
        f"Maximum: "
        f"{result['floris_farm_power_kw'].max():.2f} kW"
    )

    print(
        f"Mean: "
        f"{result['floris_farm_power_kw'].mean():.2f} kW"
    )

    print(
        "\nMaximum farm power in MW:"
    )

    print(
        f"{result['floris_farm_power_mw'].max():.3f} MW"
    )

    print(
        "\nNext step:"
    )

    print(
        "Validate FLORIS wake behaviour, then merge "
        "these simulation features with the engineered "
        "SCADA/hydrogen dataset."
    )

    print(
        "=" * 70
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()