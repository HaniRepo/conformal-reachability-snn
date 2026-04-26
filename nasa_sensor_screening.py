import os
import numpy as np
import matplotlib.pyplot as plt

from src.nasa_loader import (
    load_fd001_train,
    compute_rul,
    add_normalized_degradation_target,
)
from src.snn_score_nasa_fusion import (
    make_nasa_train_cal_split,
    train_snn_and_get_nasa_scores_fixed_split,
)
from src.conformal_score import split_conformal_score_interval, apply_score_interval
from src.reachability import finite_horizon_reachability


# =========================================================
# CONFIG
# =========================================================
FILEPATH = r"data/data_nasa/train_FD001.txt"
OUT_DIR = "results_nasa_single_sensor_screening_improved"

SENSORS_TO_TEST = [
    "sensor_11",
    "sensor_12",
    "sensor_7",
    "sensor_4",
    "sensor_15",
    "sensor_20",
    "sensor_21",
]

WINDOW_SIZE = 30
STRIDE = 1
SMOOTHING_WINDOW = 7
HORIZON = 10
ALPHA = 0.1

HIDDEN_SIZE = 64
EPOCHS = 40
LR = 1e-3
BATCH_SIZE = 64
RANDOM_SEED = 42

MIN_CYCLES = 180
MAX_ENGINES = 25

USE_MONOTONE_ENFORCEMENT = False
USE_PER_ENGINE_DTMC = True
VERBOSE_TRAINING = False


# =========================================================
# PLOT STYLE
# =========================================================
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# =========================================================
# HELPERS
# =========================================================
def moving_average_edge(x: np.ndarray, window: int = 7) -> np.ndarray:
    if window <= 1:
        return x.copy()
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    x_pad = np.pad(x, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x_pad, kernel, mode="valid")


def detect_id_and_cycle_columns(df):
    possible_engine_cols = ["unit_nr", "unit_number", "engine_id", "unit_id", "id"]
    possible_cycle_cols = ["time_cycles", "cycle", "cycles"]

    engine_col = None
    cycle_col = None

    for c in possible_engine_cols:
        if c in df.columns:
            engine_col = c
            break

    for c in possible_cycle_cols:
        if c in df.columns:
            cycle_col = c
            break

    if engine_col is None:
        raise KeyError(f"Could not find engine id column. Available columns: {list(df.columns)}")
    if cycle_col is None:
        raise KeyError(f"Could not find cycle column. Available columns: {list(df.columns)}")

    return engine_col, cycle_col


def filter_to_clean_engines(df, min_cycles: int = 180, max_engines: int = 25):
    engine_col, cycle_col = detect_id_and_cycle_columns(df)
    engine_lengths = df.groupby(engine_col)[cycle_col].max().sort_values(ascending=False)
    selected = engine_lengths[engine_lengths >= min_cycles].index.tolist()

    if len(selected) == 0:
        selected = engine_lengths.index.tolist()[:max_engines]
    else:
        selected = selected[:max_engines]

    df_small = df[df[engine_col].isin(selected)].copy()
    return df_small, selected, engine_lengths, engine_col, cycle_col


def normalize_1d_per_engine(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    mn = np.min(x)
    mx = np.max(x)
    return (x - mn) / (mx - mn + 1e-8)


def normalize_signed_per_engine(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    mn = np.min(x)
    mx = np.max(x)
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn + 1e-8)


def build_progression_features(raw_signal: np.ndarray, smooth_window: int = 7):
    """
    Improved feature pipeline for your method:
      1) smoothed signal
      2) slope of smoothed signal
      3) deviation from initial smoothed value
    """
    smooth = moving_average_edge(raw_signal, window=smooth_window)
    slope = np.diff(smooth, prepend=smooth[0])
    deviation = smooth - smooth[0]

    smooth_n = normalize_1d_per_engine(smooth)
    slope_n = normalize_signed_per_engine(slope)
    deviation_n = normalize_signed_per_engine(deviation)

    return smooth_n, slope_n, deviation_n


def build_window_dataset_from_selected_sensor_improved(
    df,
    sensor_col: str,
    engine_col: str,
    cycle_col: str,
    target_col: str,
    window_size: int,
    stride: int,
    smooth_window: int,
):
    X_list = []
    y_list = []
    engine_ids = []

    for eng in df[engine_col].unique():
        run_df = df[df[engine_col] == eng].sort_values(cycle_col).copy()

        raw_signal = run_df[sensor_col].values.astype(float)
        y_run = run_df[target_col].values.astype(float)

        smooth_n, slope_n, deviation_n = build_progression_features(
            raw_signal,
            smooth_window=smooth_window,
        )

        feature_mat = np.column_stack([smooth_n, slope_n, deviation_n])  # [T,3]

        T = len(run_df)
        for start in range(0, T - window_size + 1, stride):
            end = start + window_size
            window = feature_mat[start:end]
            target = y_run[end - 1]

            X_list.append(window.reshape(-1))
            y_list.append(target)
            engine_ids.append(eng)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    engine_ids = np.asarray(engine_ids)

    return X, y, engine_ids


def normalize_features_global(X: np.ndarray):
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    std[std == 0] = 1.0
    Xn = (X - mean) / std
    return Xn.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def make_stage_labels(y: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(y), dtype=int)
    labels[y >= 0.33] = 1
    labels[y >= 0.66] = 2
    return labels


def learn_state_thresholds_from_calibration(y_cal: np.ndarray, score_cal: np.ndarray):
    h = score_cal[y_cal == 0]
    d = score_cal[y_cal == 1]
    c = score_cal[y_cal == 2]

    if len(h) == 0 or len(d) == 0 or len(c) == 0:
        return 0.33, 0.66

    q1 = 0.5 * (np.mean(h) + np.mean(d))
    q2 = 0.5 * (np.mean(d) + np.mean(c))

    q1 = float(np.clip(q1, 0.05, 0.90))
    q2 = float(np.clip(q2, q1 + 0.05, 0.95))
    return q1, q2


def smooth_scores_per_engine(scores: np.ndarray, engine_ids: np.ndarray, window: int = 7):
    out = np.zeros_like(scores)
    for eng in np.unique(engine_ids):
        mask = engine_ids == eng
        out[mask] = moving_average_edge(scores[mask], window=window)
    return out


def enforce_monotone_per_engine(scores: np.ndarray, engine_ids: np.ndarray):
    out = np.zeros_like(scores)
    for eng in np.unique(engine_ids):
        mask = engine_ids == eng
        out[mask] = np.maximum.accumulate(scores[mask])
    return out


def score_to_state(score: np.ndarray, q1: float, q2: float) -> np.ndarray:
    s = np.zeros(len(score), dtype=int)
    s[score >= q1] = 1
    s[score >= q2] = 2
    return s


def estimate_transition_matrix(states: np.ndarray, n_states: int = 3):
    counts = np.zeros((n_states, n_states), dtype=np.int64)
    for i in range(len(states) - 1):
        counts[states[i], states[i + 1]] += 1

    P = counts.astype(float)
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = P / row_sums
    return counts, P


def estimate_transition_matrix_per_engine(states: np.ndarray, engine_ids: np.ndarray, n_states: int = 3):
    counts = np.zeros((n_states, n_states), dtype=np.int64)

    for eng in np.unique(engine_ids):
        mask = engine_ids == eng
        s = states[mask]
        if len(s) < 2:
            continue
        for i in range(len(s) - 1):
            counts[s[i], s[i + 1]] += 1

    P = counts.astype(float)
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = P / row_sums
    return counts, P


def choose_representative_engine_from_scores(engine_ids: np.ndarray, scores: np.ndarray):
    best_engine = None
    best_cost = float("inf")

    unique_engines = np.unique(engine_ids)
    counts = {eng: np.sum(engine_ids == eng) for eng in unique_engines}
    median_len = np.median(list(counts.values()))

    for eng in unique_engines:
        mask = engine_ids == eng
        s = scores[mask]
        if len(s) < 40:
            continue

        s_sm = moving_average_edge(s, window=7)
        diffs = np.diff(s_sm)
        downward = np.sum(np.clip(-diffs, 0, None))
        late_level_penalty = abs(np.mean(s_sm[-10:]) - 0.9)
        length_penalty = 0.001 * abs(len(s) - median_len)
        cost = 2.0 * downward + 0.25 * late_level_penalty + length_penalty

        if cost < best_cost:
            best_cost = cost
            best_engine = int(eng)

    if best_engine is None:
        best_engine = int(unique_engines[0])

    return best_engine


def plot_score_with_conformal_single(
    scores: np.ndarray,
    score_lo: np.ndarray,
    score_hi: np.ndarray,
    engine_ids: np.ndarray,
    out_path: str,
    q1: float,
    q2: float,
    sensor_name: str,
):
    eng = choose_representative_engine_from_scores(engine_ids, scores)
    mask = engine_ids == eng

    idx = np.arange(np.sum(mask))
    s_eng = moving_average_edge(scores[mask], window=7)
    lo_eng = moving_average_edge(score_lo[mask], window=7)
    hi_eng = moving_average_edge(score_hi[mask], window=7)

    if USE_MONOTONE_ENFORCEMENT:
        s_eng = np.maximum.accumulate(s_eng)
        lo_eng = np.maximum.accumulate(lo_eng)
        hi_eng = np.maximum.accumulate(hi_eng)

    lo_eng = np.minimum(lo_eng, hi_eng)
    lo_eng = np.clip(lo_eng, 0.0, 1.0)
    hi_eng = np.clip(hi_eng, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.axhspan(0.00, q1, alpha=0.05)
    ax.axhspan(q1, q2, alpha=0.05)
    ax.axhspan(q2, 1.00, alpha=0.05)

    ax.fill_between(idx, lo_eng, hi_eng, alpha=0.14, label="Conformal interval")
    ax.plot(idx, s_eng, linewidth=2.8, label="SNN score")
    ax.axhline(q1, linestyle=":", linewidth=1.4, label="State thresholds")
    ax.axhline(q2, linestyle=":", linewidth=1.4)

    ax.set_title(f"{sensor_name}: degradation score with conformal uncertainty (engine {eng})")
    ax.set_xlabel("Window index within engine")
    ax.set_ylabel("Degradation score")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def matrix_cleanliness_score(P: np.ndarray) -> float:
    """
    Larger is cleaner:
    rewards self-loops and forward progression,
    penalizes backward transitions.
    """
    score = 0.0
    score += P[0, 0] + P[1, 1] + P[2, 2]
    score += P[0, 1] + P[1, 2]
    score -= P[1, 0] + P[2, 1] + P[2, 0] + P[0, 2]
    return float(score)


# =========================================================
# MAIN
# =========================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_fd001_train(FILEPATH)
    df = compute_rul(df)
    df = add_normalized_degradation_target(df, rul_cap=125)

    engine_col, cycle_col = detect_id_and_cycle_columns(df)

    df, selected_engines, _, _, _ = filter_to_clean_engines(
        df,
        min_cycles=MIN_CYCLES,
        max_engines=MAX_ENGINES,
    )

    print("Selected engines:")
    print(selected_engines)

    results = []

    for sensor_name in SENSORS_TO_TEST:
        print("\n" + "=" * 70)
        print(f"Running single-sensor screening for {sensor_name}")
        print("=" * 70)

        sensor_dir = os.path.join(OUT_DIR, sensor_name)
        os.makedirs(sensor_dir, exist_ok=True)

        X, y, engine_ids = build_window_dataset_from_selected_sensor_improved(
            df=df,
            sensor_col=sensor_name,
            engine_col=engine_col,
            cycle_col=cycle_col,
            target_col="y_deg",
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            smooth_window=SMOOTHING_WINDOW,
        )

        X, _, _ = normalize_features_global(X)
        labels = make_stage_labels(y)

        split_indices = make_nasa_train_cal_split(
            labels=labels,
            test_size=0.3,
            random_seed=RANDOM_SEED,
        )

        result = train_snn_and_get_nasa_scores_fixed_split(
            windows=X,
            labels=labels,
            split_indices=split_indices,
            hidden_size=HIDDEN_SIZE,
            epochs=EPOCHS,
            lr=LR,
            batch_size=BATCH_SIZE,
            random_seed=RANDOM_SEED,
            verbose=VERBOSE_TRAINING,
        )

        scores = result["score_all"]
        y_cal = result["y_cal"]
        score_cal = result["score_cal"]

        y_cal_score = y_cal.astype(np.float32) / 2.0
        qhat = split_conformal_score_interval(y_cal_score, score_cal, alpha=ALPHA)
        score_lo, score_hi = apply_score_interval(scores, qhat, min_score=0.0, max_score=1.0)
        mean_width = float(np.mean(score_hi - score_lo))

        q1, q2 = learn_state_thresholds_from_calibration(y_cal, score_cal)
        threshold_gap = float(q2 - q1)

        scores_smoothed = smooth_scores_per_engine(scores, engine_ids, window=SMOOTHING_WINDOW)

        if USE_MONOTONE_ENFORCEMENT:
            scores_for_states = enforce_monotone_per_engine(scores_smoothed, engine_ids)
        else:
            scores_for_states = scores_smoothed.copy()

        states = score_to_state(scores_for_states, q1=q1, q2=q2)

        if USE_PER_ENGINE_DTMC:
            counts, P = estimate_transition_matrix_per_engine(states, engine_ids, n_states=3)
        else:
            counts, P = estimate_transition_matrix(states, n_states=3)

        p_H = finite_horizon_reachability(P, target_state=2, horizon=HORIZON)
        clean_score = matrix_cleanliness_score(P)

        plot_score_with_conformal_single(
            scores=scores_for_states,
            score_lo=score_lo,
            score_hi=score_hi,
            engine_ids=engine_ids,
            out_path=os.path.join(sensor_dir, f"{sensor_name}_score_with_conformal.png"),
            q1=q1,
            q2=q2,
            sensor_name=sensor_name,
        )

        np.savetxt(os.path.join(sensor_dir, "transition_matrix.csv"), P, delimiter=",", fmt="%.10f")

        with open(os.path.join(sensor_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"Single-sensor screening: {sensor_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"qhat: {qhat:.4f}\n")
            f.write(f"mean_width: {mean_width:.4f}\n")
            f.write(f"q1: {q1:.4f}\n")
            f.write(f"q2: {q2:.4f}\n")
            f.write(f"threshold_gap: {threshold_gap:.4f}\n")
            f.write(f"matrix_cleanliness: {clean_score:.4f}\n")
            f.write("\nTransition matrix P:\n")
            for row in P:
                f.write("  " + "  ".join(f"{x:.6f}" for x in row) + "\n")
            f.write("\nFinite-horizon reachability to critical state:\n")
            f.write(f"  From healthy:   {p_H[0]:.4f}\n")
            f.write(f"  From degrading: {p_H[1]:.4f}\n")
            f.write(f"  From critical:  {p_H[2]:.4f}\n")

        row = {
            "sensor": sensor_name,
            "qhat": float(qhat),
            "width": float(mean_width),
            "q1": float(q1),
            "q2": float(q2),
            "gap": float(threshold_gap),
            "clean": float(clean_score),
            "H": float(p_H[0]),
            "D": float(p_H[1]),
            "C": float(p_H[2]),
        }
        results.append(row)

        print(
            f"qhat={row['qhat']:.4f}, "
            f"width={row['width']:.4f}, "
            f"gap={row['gap']:.4f}, "
            f"clean={row['clean']:.4f}, "
            f"H={row['H']:.4f}, "
            f"D={row['D']:.4f}"
        )

    # sort mainly by width and qhat, then prefer larger threshold gap
    results_sorted = sorted(
        results,
        key=lambda r: (r["width"], r["qhat"], -r["gap"], -r["clean"])
    )

    print("\n" + "=" * 90)
    print("FINAL SCREENING SUMMARY")
    print("=" * 90)
    for r in results_sorted:
        print(
            f"{r['sensor']:10s} | "
            f"qhat={r['qhat']:.3f} | "
            f"width={r['width']:.3f} | "
            f"gap={r['gap']:.3f} | "
            f"clean={r['clean']:.3f} | "
            f"H={r['H']:.3f} | "
            f"D={r['D']:.3f}"
        )

    with open(os.path.join(OUT_DIR, "final_ranking.txt"), "w", encoding="utf-8") as f:
        f.write("FINAL SCREENING SUMMARY\n")
        f.write("=" * 90 + "\n")
        for r in results_sorted:
            f.write(
                f"{r['sensor']:10s} | "
                f"qhat={r['qhat']:.4f} | "
                f"width={r['width']:.4f} | "
                f"gap={r['gap']:.4f} | "
                f"clean={r['clean']:.4f} | "
                f"H={r['H']:.4f} | "
                f"D={r['D']:.4f}\n"
            )


if __name__ == "__main__":
    main()