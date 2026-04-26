import math
from typing import Optional

import numpy as np
import pandas as pd
import pywt
from scipy import stats
from scipy.fftpack import fft
from scipy.signal import hilbert


def compute_pressure_drop(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["drop"] = out["pup"] - out["pdown"]
    return out


def mark_clogging(df: pd.DataFrame, clog_threshold: float = 20.0) -> pd.DataFrame:
    out = df.copy()
    out["clogged"] = (out["drop"] >= clog_threshold).astype(int)
    return out


def truncate_at_first_clogging(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each run, keep samples only up to the first clogged point, inclusive.
    If a run never clogs, keep the entire run.
    """
    out_frames = []

    for run_id, g in df.groupby("run_id", sort=False):
        g = g.sort_values("time").copy()
        idx = np.where(g["clogged"].values == 1)[0]

        if len(idx) > 0:
            first_pos = idx[0]
            g = g.iloc[: first_pos + 1].copy()

        out_frames.append(g)

    return pd.concat(out_frames, ignore_index=True)


def compute_rul_per_run(df: pd.DataFrame) -> pd.DataFrame:
    """
    RUL at time t = time_of_failure - current_time
    """
    out_frames = []

    for run_id, g in df.groupby("run_id", sort=False):
        g = g.sort_values("time").copy()
        t_fail = float(g["time"].max())
        g["rul"] = t_fail - g["time"]
        out_frames.append(g)

    return pd.concat(out_frames, ignore_index=True)


def add_normalized_degradation_target(
    df: pd.DataFrame,
    rul_cap: Optional[float] = None,
) -> pd.DataFrame:
    """
    Add y_deg in [0,1]:
      y_deg = 1 - min(rul, cap) / cap

    If rul_cap is None, use the max RUL of each run.
    """
    out_frames = []

    for run_id, g in df.groupby("run_id", sort=False):
        g = g.sort_values("time").copy()

        max_rul = float(g["rul"].max()) if rul_cap is None else float(rul_cap)
        if max_rul <= 0:
            max_rul = 1.0

        g["RUL_capped"] = np.minimum(g["rul"].values, max_rul)
        g["y_deg"] = 1.0 - (g["RUL_capped"].values / max_rul)
        g["y_deg"] = g["y_deg"].clip(0.0, 1.0)

        out_frames.append(g)

    return pd.concat(out_frames, ignore_index=True)


def eps(frame: np.ndarray) -> np.ndarray:
    h = hilbert(frame)
    amp = np.abs(h)
    return np.abs(fft(amp)) ** 2


def mpr(frame: np.ndarray) -> float:
    env = np.sqrt(eps(frame))
    end = len(env) // 2
    env = env[1:end]
    if len(env) == 0:
        return 0.0
    peak = np.max(env)
    denom = np.mean(env)
    if denom == 0:
        return 0.0
    return float(math.log10(peak / denom) * 20.0)


def rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(frame))))


def rmsf(frame: np.ndarray) -> float:
    fft_df = np.abs(fft(frame))
    end = len(fft_df) // 2
    fft_df = fft_df[1:end]
    if len(fft_df) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(fft_df))))


def wse(frame: np.ndarray) -> float:
    dcoff = []
    maxlev = 2
    coeffs = pywt.wavedec(frame, "db5", level=maxlev)

    for i in range(1, len(coeffs)):
        temp = np.abs(fft(coeffs[i]))
        half = len(temp) // 2
        dcoff.append(np.sum(temp[1:half]))

    if not dcoff:
        return 0.0

    dcoff_max = max(dcoff)
    if dcoff_max <= 0:
        return 0.0

    return float(math.log10(dcoff_max))


def get_slope(frame: np.ndarray) -> float:
    x = np.arange(len(frame))
    s, _, _, _, _ = stats.linregress(x, frame)
    return float(s)


def extract_rolling_features_for_run(
    df_run: pd.DataFrame,
    window_size: int = 100,
) -> pd.DataFrame:
    """
    Extract rolling features for one run.

    Assumes input already contains:
      time, flow_rate, pup, pdown, psize, sratio, drop, rul, y_deg
    """
    g = df_run.sort_values("time").copy()

    drop = g["drop"]

    g["slope"] = drop.rolling(window_size).apply(get_slope, raw=True)
    g["mpr"] = drop.rolling(window_size).apply(mpr, raw=True)
    g["rms"] = drop.rolling(window_size).apply(rms, raw=True)
    g["rmsf"] = drop.rolling(window_size).apply(rmsf, raw=True)
    g["wse"] = drop.rolling(window_size).apply(wse, raw=True)
    g["kurt"] = drop.rolling(window_size).kurt()
    g["skew"] = drop.rolling(window_size).skew()
    g["var"] = drop.rolling(window_size).var()
    g["std"] = drop.rolling(window_size).std()
    g["cov"] = drop.rolling(window_size).cov()

    # cumulative dirt proxy from your paper workflow
    g["clog_am"] = np.cumsum(g["flow_rate"] * g["psize"] * g["sratio"])

    g = g.dropna().reset_index(drop=True)
    return g


def extract_rolling_features_dataset(
    df: pd.DataFrame,
    window_size: int = 100,
) -> pd.DataFrame:
    out_frames = []
    for run_id, g in df.groupby("run_id", sort=False):
        out_frames.append(extract_rolling_features_for_run(g, window_size=window_size))
    return pd.concat(out_frames, ignore_index=True)


def get_selected_feature_columns() -> list[str]:
    """
    Selected features consistent with your PHME2020 paper.
    """
    return [
        "flow_rate",
        "pup",
        "drop",
        "slope",
        "rms",
        "rmsf",
        "var",
        "std",
        "cov",
        "clog_am",
        "psize",
        "sratio",
    ]


def make_filtration_stage_labels(
    y_deg: np.ndarray,
    q1: float = 0.33,
    q2: float = 0.66,
) -> np.ndarray:
    labels = np.zeros(len(y_deg), dtype=int)
    labels[y_deg >= q1] = 1
    labels[y_deg >= q2] = 2
    return labels


def build_feature_matrix(
    df_feat: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    target_col: str = "y_deg",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if feature_cols is None:
        feature_cols = get_selected_feature_columns()

    X = df_feat[feature_cols].values.astype(np.float32)
    y = df_feat[target_col].values.astype(np.float32)
    run_ids = df_feat["run_id"].values

    return X, y, run_ids


def print_filtration_feature_summary(df_feat: pd.DataFrame) -> None:
    print("Filtration feature dataset summary")
    print("---------------------------------")
    print("Rows:", len(df_feat))
    print("Runs:", df_feat["run_id"].nunique())

    if "rul" in df_feat.columns:
        print("RUL range:", float(df_feat["rul"].min()), "to", float(df_feat["rul"].max()))

    if "y_deg" in df_feat.columns:
        print(
            "Normalized degradation target range:",
            f"{df_feat['y_deg'].min():.4f} to {df_feat['y_deg'].max():.4f}"
        )