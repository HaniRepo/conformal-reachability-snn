import numpy as np
import pandas as pd


def create_engine_windows(
    engine_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "y_deg",
    window_size: int = 30,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create windows for a single engine trajectory.

    Each window has shape [window_size, num_features].
    It is flattened to [window_size * num_features] so it can be fed
    into the current SNN implementation.

    The target for each window is taken as the target value at the
    last time step of the window.
    """
    engine_df = engine_df.sort_values("cycle").reset_index(drop=True)

    X = engine_df[feature_cols].values.astype(np.float32)
    y = engine_df[target_col].values.astype(np.float32)

    windows = []
    targets = []

    n = len(engine_df)
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size

        window = X[start:end]                  # shape [window_size, num_features]
        target = y[end - 1]                   # target at last step of window

        windows.append(window.reshape(-1))    # flatten
        targets.append(target)

    if len(windows) == 0:
        return np.empty((0, window_size * len(feature_cols)), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32)


def build_nasa_window_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "y_deg",
    window_size: int = 30,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a full NASA window dataset across all engines.

    Returns:
        X_all: [num_windows, window_size * num_features]
        y_all: [num_windows]
        engine_ids: [num_windows]
    """
    all_windows = []
    all_targets = []
    all_engine_ids = []

    for engine_id, engine_df in df.groupby("unit_id"):
        X_eng, y_eng = create_engine_windows(
            engine_df=engine_df,
            feature_cols=feature_cols,
            target_col=target_col,
            window_size=window_size,
            stride=stride,
        )

        if len(X_eng) == 0:
            continue

        all_windows.append(X_eng)
        all_targets.append(y_eng)
        all_engine_ids.append(np.full(len(y_eng), engine_id, dtype=np.int32))

    if len(all_windows) == 0:
        raise ValueError("No windows were created. Check window_size and input dataframe.")

    X_all = np.vstack(all_windows).astype(np.float32)
    y_all = np.concatenate(all_targets).astype(np.float32)
    engine_ids = np.concatenate(all_engine_ids).astype(np.int32)

    return X_all, y_all, engine_ids


def normalize_window_dataset(X: np.ndarray) -> np.ndarray:
    """
    Normalize each flattened window independently.

    Input:
        X shape [num_windows, num_features_flat]

    Output:
        normalized X of same shape
    """
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


def print_window_dataset_summary(
    X: np.ndarray,
    y: np.ndarray,
    engine_ids: np.ndarray,
) -> None:
    """
    Print a quick summary for debugging.
    """
    print("NASA window dataset summary")
    print("---------------------------")
    print("Windows shape:", X.shape)
    print("Targets shape:", y.shape)
    print("Engine ids shape:", engine_ids.shape)
    print("Number of engines in windows:", len(np.unique(engine_ids)))
    print("Target range:", f"{y.min():.4f} to {y.max():.4f}")

    if len(X) > 0:
        print("First window mean/std before optional normalization:",
              f"{np.mean(X[0]):.4f}", f"{np.std(X[0]):.4f}")