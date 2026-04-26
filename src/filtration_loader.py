import os
import re
from typing import Optional

import pandas as pd


RAW_TO_STANDARD_COLUMNS = {
    "Time(s)": "time",
    "Flow_Rate(ml/m)": "flow_rate",
    "Upstream_Pressure(psi)": "pup",
    "Downstream_Pressure(psi)": "pdown",
}


def sample_number_from_name(name: str) -> int:
    """
    Extract sample number from names like:
      Sample33.csv
      Sample09.csv
    """
    base = os.path.splitext(os.path.basename(name))[0]
    m = re.search(r"sample(\d+)", base, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not parse sample number from: {name}")
    return int(m.group(1))


def particle_size_range_to_numeric(text: str) -> float:
    """
    Convert particle size range strings like:
      '45-53' -> 49.0
      '63-75' -> 69.0
    by taking the midpoint.
    """
    s = str(text).strip()
    m = re.match(r"(\d+)\s*-\s*(\d+)", s)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        return 0.5 * (a + b)

    # fallback if already numeric-like
    return float(s)


def load_profile_excel(excel_path: str) -> pd.DataFrame:
    """
    Load the operation-profile Excel file and keep only the non-empty first sheet.

    Expected columns:
      Sample
      Particle Size (micron)
      Solid Ratio(%)
    """
    xls = pd.ExcelFile(excel_path)
    df = pd.read_excel(excel_path, sheet_name="Sheet1").copy()

    expected = ["Sample", "Particle Size (micron)", "Solid Ratio(%)"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns in {excel_path}: {missing}. "
            f"Found: {df.columns.tolist()}"
        )

    df = df[expected].copy()
    df["Sample"] = df["Sample"].astype(int)
    df["psize"] = df["Particle Size (micron)"].apply(particle_size_range_to_numeric)
    df["sratio"] = df["Solid Ratio(%)"].astype(float)

    return df[["Sample", "psize", "sratio"]]


def load_single_filtration_csv(
    csv_path: str,
    split: str,
    particle_label: str,
    psize: float,
    sratio: float,
) -> pd.DataFrame:
    """
    Load one sample CSV and standardize columns.
    """
    df = pd.read_csv(csv_path).copy()

    missing = [c for c in RAW_TO_STANDARD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV {csv_path} is missing columns {missing}. "
            f"Found: {df.columns.tolist()}"
        )

    df = df.rename(columns=RAW_TO_STANDARD_COLUMNS)

    df = df[["time", "flow_rate", "pup", "pdown"]].copy()

    sample_num = sample_number_from_name(csv_path)
    run_name = f"Sample{sample_num:02d}"
    run_id = f"{split}_{particle_label}_{run_name}"

    df["sample"] = sample_num
    df["run_name"] = run_name
    df["run_id"] = run_id
    df["split"] = split
    df["particle_label"] = particle_label
    df["psize"] = float(psize)
    df["sratio"] = float(sratio)

    return df


def load_filtration_split(
    split_dir: str,
    split_name: str,
) -> pd.DataFrame:
    """
    Load one split, e.g.:
      src/Training
      src/Validation

    Expected structure:
      split_dir/
        Large/
          Sample33.csv ...
        Small/
          Sample01.csv ...
        <split_name> Operation Profiles of Samples.xlsx
    """
    excel_name = f"{split_name} Operation Profiles of Samples.xlsx"
    excel_path = os.path.join(split_dir, excel_name)

    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Could not find profile Excel: {excel_path}")

    profiles = load_profile_excel(excel_path)
    profile_map = {
        int(row["Sample"]): {"psize": float(row["psize"]), "sratio": float(row["sratio"])}
        for _, row in profiles.iterrows()
    }

    frames = []

    for particle_label in ["Small", "Large"]:
        folder = os.path.join(split_dir, particle_label)
        if not os.path.isdir(folder):
            continue

        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(".csv"):
                continue

            csv_path = os.path.join(folder, fname)
            sample_num = sample_number_from_name(fname)

            if sample_num not in profile_map:
                raise ValueError(
                    f"Sample {sample_num} from {csv_path} not found in {excel_name}"
                )

            meta = profile_map[sample_num]
            df_run = load_single_filtration_csv(
                csv_path=csv_path,
                split=split_name,
                particle_label=particle_label,
                psize=meta["psize"],
                sratio=meta["sratio"],
            )
            frames.append(df_run)

    if not frames:
        raise ValueError(f"No CSV files found under {split_dir}")

    return pd.concat(frames, ignore_index=True)


def load_filtration_dataset(root_dir: str) -> pd.DataFrame:
    """
    Load full filtration dataset from root_dir, which should contain:
      root_dir/
        Training/
        Validation/
    """
    frames = []

    training_dir = os.path.join(root_dir, "Training")
    validation_dir = os.path.join(root_dir, "Validation")

    if os.path.isdir(training_dir):
        frames.append(load_filtration_split(training_dir, "Training"))

    if os.path.isdir(validation_dir):
        frames.append(load_filtration_split(validation_dir, "Validation"))

    if not frames:
        raise ValueError(f"No Training/Validation folders found under {root_dir}")

    return pd.concat(frames, ignore_index=True)


def print_filtration_dataset_summary(df: pd.DataFrame) -> None:
    print("Filtration dataset summary")
    print("-------------------------")
    print("Rows:", len(df))
    print("Runs:", df["run_id"].nunique())
    print("Splits:", sorted(df["split"].dropna().unique().tolist()))
    print("Particle labels:", sorted(df["particle_label"].dropna().unique().tolist()))
    print("Particle sizes:", sorted(df["psize"].dropna().unique().tolist()))
    print("Solid ratios:", sorted(df["sratio"].dropna().unique().tolist()))