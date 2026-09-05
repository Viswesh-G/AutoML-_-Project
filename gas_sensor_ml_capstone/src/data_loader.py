
"""
Data loader for the UCI Gas Sensor Array Drift at Different Concentrations
dataset (UCI Dataset ID: 270).

Raw format:
    class;concentration 1:value 2:value ... 128:value

Example:
    1;10.000000 1:15596.162100 2:1.868245 ... 128:-2.654529
"""

from pathlib import Path
import pandas as pd


# Mapping from UCI gas class number to gas name
GAS_NAMES = {
    1: "Ethanol",
    2: "Ethylene",
    3: "Ammonia",
    4: "Acetaldehyde",
    5: "Acetone",
    6: "Toluene"
}


def parse_dat_file(file_path):
    """
    Parse one UCI Gas Sensor .dat file.

    Returns a DataFrame containing:
        gas_class
        concentration
        Feature1 ... Feature128
    """

    file_path = Path(file_path)

    rows = []

    with open(file_path, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            # Ignore empty lines
            if not line:
                continue

            parts = line.split()

            # First item contains class and concentration
            # Example: 1;10.000000
            class_concentration = parts[0]

            try:
                gas_class_str, concentration_str = (
                    class_concentration.split(";")
                )
            except ValueError:
                raise ValueError(
                    f"Invalid class/concentration format in "
                    f"{file_path} at line {line_number}: "
                    f"{parts[0]}"
                )

            gas_class = int(gas_class_str)
            concentration = float(concentration_str)

            # Store one measurement
            row = {
                "gas_class": gas_class,
                "concentration": concentration
            }

            # Parse Feature1:xxx through Feature128:xxx
            for feature_pair in parts[1:]:

                try:
                    feature_number, feature_value = feature_pair.split(":")

                    feature_number = int(feature_number)
                    feature_value = float(feature_value)

                    row[f"Feature{feature_number}"] = feature_value

                except ValueError:
                    raise ValueError(
                        f"Invalid feature format in "
                        f"{file_path} at line {line_number}: "
                        f"{feature_pair}"
                    )

            rows.append(row)

    df = pd.DataFrame(rows)

    # Expected 128 sensor features
    feature_columns = [
        f"Feature{i}" for i in range(1, 129)
    ]

    # Make sure all feature columns exist
    for column in feature_columns:
        if column not in df.columns:
            df[column] = pd.NA

    # Fixed column order
    df = df[
        ["gas_class", "concentration"] + feature_columns
    ]

    return df


def load_dataset(data_dir):
    """
    Load all .dat batch files from the specified data directory.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the batch .dat files.

    Returns
    -------
    pandas.DataFrame
        Combined dataset.
    """

    data_dir = Path(data_dir)

    print("Looking for dataset in:")
    print(data_dir)
    print()

    # Check that the directory exists
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory does not exist:\n{data_dir}"
        )

    # Find .dat files
    dat_files = sorted(data_dir.glob("*.dat"))

    if not dat_files:
        raise FileNotFoundError(
            f"No .dat files found in:\n{data_dir}\n\n"
            "Make sure the UCI batch .dat files are directly "
            "inside the data folder."
        )

    print(f"Found {len(dat_files)} .dat files.")
    print()

    dataframes = []

    # Load each batch
    for file_path in dat_files:

        print(f"Loading: {file_path.name}")

        df_batch = parse_dat_file(file_path)

        # Store batch name
        df_batch["batch"] = file_path.stem

        dataframes.append(df_batch)

        print(f"  Samples loaded: {len(df_batch)}")

    # Combine all batches
    df = pd.concat(
        dataframes,
        ignore_index=True
    )

    # Feature columns
    feature_columns = [
        f"Feature{i}" for i in range(1, 129)
    ]

    # Fixed column order
    df = df[
        ["gas_class", "concentration", "batch"]
        + feature_columns
    ]

    return df


def add_gas_names(df):
    """
    Add readable gas names based on gas_class.
    """

    df = df.copy()

    df["gas_name"] = df["gas_class"].map(GAS_NAMES)

    return df


# ============================================================
# TEST THE LOADER
# ============================================================

if __name__ == "__main__":

    # data_loader.py is located in:
    #
    # gas_sensor_ml_capstone/
    #     src/
    #         data_loader.py
    #
    # Therefore parents[1] gives:
    #
    # gas_sensor_ml_capstone/

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # Dataset location:
    # gas_sensor_ml_capstone/data/
    DATA_DIR = PROJECT_ROOT / "data"

    print("=" * 60)
    print("UCI GAS SENSOR DATASET LOADER")
    print("=" * 60)
    print()

    # Load dataset
    df = load_dataset(DATA_DIR)

    print()
    print("=" * 60)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 60)

    print("\nDataset shape:")
    print(df.shape)

    print("\nFirst five rows:")
    print(df.head())

    print("\nGas class distribution:")
    print(df["gas_class"].value_counts().sort_index())

    print("\nMissing values:")
    print(df.isnull().sum().sum())
