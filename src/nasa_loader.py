import pandas as pd
import numpy as np


NASA_COLUMNS = (
    ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_fd001_train(filepath: str) -> pd.DataFrame:
    """
    Load NASA C-MAPSS FD001 training file.

    Expected format:
    - space separated
    - 26 columns:
        unit_id, cycle, 3 operational settings, 21 sensor measurements

    Returns:
        DataFrame with clean column names.
    """
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        engine="python",
    )

    # Some versions may have extra empty columns at the end.
    if df.shape[1] > 26:
        df = df.iloc[:, :26]

    if df.shape[1] != 26:
        raise ValueError(
            f"Expected 26 columns in NASA FD001 file, but found {df.shape[1]}"
        )

    df.columns = NASA_COLUMNS
    return df


def compute_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute true Remaining Useful Life (RUL) for each engine cycle
    in the training set.

    RUL_t = max_cycle_for_engine - current_cycle
    """
    out = df.copy()
    max_cycles = out.groupby("unit_id")["cycle"].transform("max")
    out["RUL"] = max_cycles - out["cycle"]
    return out


def add_normalized_degradation_target(
    df: pd.DataFrame,
    rul_cap: int = 125
) -> pd.DataFrame:
    """
    Add a normalized degradation target in [0,1].

    We first cap the RUL to reduce the dominance of very early cycles:
        RUL_capped = min(RUL, rul_cap)

    Then define:
        y_deg = 1 - RUL_capped / rul_cap

    Interpretation:
        y_deg ~ 0  -> healthy / early life
        y_deg ~ 1  -> near failure
    """
    out = df.copy()

    if "RUL" not in out.columns:
        raise ValueError("RUL column not found. Run compute_rul(df) first.")

    out["RUL_capped"] = np.minimum(out["RUL"].values, rul_cap)
    out["y_deg"] = 1.0 - (out["RUL_capped"].values / float(rul_cap))
    out["y_deg"] = out["y_deg"].clip(0.0, 1.0)

    return out


def split_by_engine(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    random_seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataframe by engine id, not by rows, to avoid leakage.

    Returns:
        train_df, cal_df
    """
    rng = np.random.default_rng(random_seed)

    engine_ids = df["unit_id"].unique()
    engine_ids = np.array(sorted(engine_ids))
    rng.shuffle(engine_ids)

    n_train = int(len(engine_ids) * train_ratio)
    train_ids = set(engine_ids[:n_train])
    cal_ids = set(engine_ids[n_train:])

    train_df = df[df["unit_id"].isin(train_ids)].copy()
    cal_df = df[df["unit_id"].isin(cal_ids)].copy()

    return train_df, cal_df


def get_default_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return a simple default feature set for FD001.

    For the easiest first version, we keep:
    - 3 operational settings
    - all 21 sensors

    Later, you may reduce this set if needed.
    """
    feature_cols = ["op_setting_1", "op_setting_2", "op_setting_3"] + [
        c for c in df.columns if c.startswith("sensor_")
    ]
    return feature_cols


def print_fd001_summary(df: pd.DataFrame) -> None:
    """
    Print a quick summary for debugging.
    """
    print("NASA FD001 summary")
    print("------------------")
    print("Rows:", len(df))
    print("Engines:", df["unit_id"].nunique())
    print("Cycle range:", int(df["cycle"].min()), "to", int(df["cycle"].max()))

    if "RUL" in df.columns:
        print("RUL range:", int(df["RUL"].min()), "to", int(df["RUL"].max()))

    if "y_deg" in df.columns:
        print("Normalized degradation target range:",
              f"{df['y_deg'].min():.4f} to {df['y_deg'].max():.4f}")