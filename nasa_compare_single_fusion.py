import os
import numpy as np
import matplotlib.pyplot as plt

from src.nasa_loader import (
    load_fd001_train,
    compute_rul,
    add_normalized_degradation_target,
    print_fd001_summary,
)
from src.snn_score_nasa import train_snn_and_get_nasa_scores
from src.snn_score_nasa_fusion import train_two_branch_late_fusion
from src.conformal_score import split_conformal_score_interval, apply_score_interval
from src.reachability import finite_horizon_reachability


# =========================================================
# CONFIG
# =========================================================
FILEPATH = r"data/data_nasa/train_FD001.txt"
OUT_DIR = "results_nasa_single_vs_fusion"

USE_LIGHT_PREPROCESSING = True
USE_PER_ENGINE_DTMC = True
USE_SELECTED_ENGINES_ONLY = True

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

W1 = 0.5
W2 = 0.5

VERBOSE_TRAINING = True


# =========================================================
# BASIC HELPERS
# =========================================================
def moving_average_edge(x: np.ndarray, window: int = 7) -> np.ndarray:
    if window <= 1:
        return x.copy()

    pad_left = window // 2
    pad_right = window - 1 - pad_left
    x_pad = np.pad(x, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x_pad, kernel, mode="valid")


def normalize_1d_per_engine(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    mn = np.min(x)
    mx = np.max(x)

    if mx - mn < 1e-8:
        return np.zeros_like(x)

    return (x - mn) / (mx - mn + 1e-8)


def normalize_signed_per_engine(x: np.ndarray) -> np.ndarray:
    return normalize_1d_per_engine(x)


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


def build_processed_signal(raw_signal: np.ndarray, smooth_window: int = 7):
    """
    Progression-oriented features:
      1. smoothed signal
      2. slope of smoothed signal
      3. deviation from initial smoothed value
      4. rolling local standard deviation
    """
    smooth = moving_average_edge(raw_signal, window=smooth_window)
    slope = np.diff(smooth, prepend=smooth[0])
    deviation = smooth - smooth[0]

    roll_window = 7
    rolling_std = np.array([
        np.std(smooth[max(0, i - roll_window): i + 1])
        for i in range(len(smooth))
    ])

    smooth_n = normalize_1d_per_engine(smooth)
    slope_n = normalize_signed_per_engine(slope)
    deviation_n = normalize_signed_per_engine(deviation)
    rolling_std_n = normalize_1d_per_engine(rolling_std)

    return smooth_n, slope_n, deviation_n, rolling_std_n


def build_window_dataset_from_selected_sensor(
    df,
    sensor_col: str,
    engine_col: str,
    cycle_col: str,
    target_col: str,
    window_size: int,
    stride: int,
    use_light_preprocessing: bool,
    smooth_window: int,
):
    X_list = []
    y_list = []
    engine_ids = []

    for eng in df[engine_col].unique():
        run_df = df[df[engine_col] == eng].sort_values(cycle_col).copy()

        raw_signal = run_df[sensor_col].values.astype(float)
        y_run = run_df[target_col].values.astype(float)

        if use_light_preprocessing:
            smooth_n, slope_n, deviation_n, rolling_std_n = build_processed_signal(
                raw_signal,
                smooth_window=smooth_window,
            )

            feature_mat = np.column_stack([
                smooth_n,
                slope_n,
                deviation_n,
                rolling_std_n,
            ])
        else:
            raw_n = normalize_1d_per_engine(raw_signal)
            feature_mat = raw_n[:, None]

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


def score_to_state(score: np.ndarray, q1: float, q2: float) -> np.ndarray:
    s = np.zeros(len(score), dtype=int)
    s[score >= q1] = 1
    s[score >= q2] = 2
    return s


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


def degradation_indicator(scores: np.ndarray, engine_ids: np.ndarray) -> float:
    values = []

    for eng in np.unique(engine_ids):
        mask = engine_ids == eng
        s = scores[mask]

        if len(s) < 10:
            continue

        k = max(3, len(s) // 5)
        early = np.mean(s[:k])
        late = np.mean(s[-k:])
        values.append(late - early)

    if len(values) == 0:
        return 0.0

    return float(np.mean(values))


# =========================================================
# EXPERIMENT
# =========================================================
def run_one_experiment(
    df,
    engine_col,
    cycle_col,
    sensors,
    use_monotone: bool,
):
    if len(sensors) == 1:
        sensor = sensors[0]

        X, y, engine_ids = build_window_dataset_from_selected_sensor(
            df=df,
            sensor_col=sensor,
            engine_col=engine_col,
            cycle_col=cycle_col,
            target_col="y_deg",
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            use_light_preprocessing=USE_LIGHT_PREPROCESSING,
            smooth_window=SMOOTHING_WINDOW,
        )

        X, _, _ = normalize_features_global(X)
        labels = make_stage_labels(y)

        result = train_snn_and_get_nasa_scores(
            X,
            labels,
            hidden_size=HIDDEN_SIZE,
            epochs=EPOCHS,
            lr=LR,
            batch_size=BATCH_SIZE,
            random_seed=RANDOM_SEED,
        )

        score_all = result["score_all"]
        score_cal = result["score_cal"]
        y_cal = result["y_cal"]

    elif len(sensors) == 2:
        sensor_1, sensor_2 = sensors

        X1, y1, engine_ids_1 = build_window_dataset_from_selected_sensor(
            df=df,
            sensor_col=sensor_1,
            engine_col=engine_col,
            cycle_col=cycle_col,
            target_col="y_deg",
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            use_light_preprocessing=USE_LIGHT_PREPROCESSING,
            smooth_window=SMOOTHING_WINDOW,
        )

        X2, y2, engine_ids_2 = build_window_dataset_from_selected_sensor(
            df=df,
            sensor_col=sensor_2,
            engine_col=engine_col,
            cycle_col=cycle_col,
            target_col="y_deg",
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            use_light_preprocessing=USE_LIGHT_PREPROCESSING,
            smooth_window=SMOOTHING_WINDOW,
        )

        if not np.allclose(y1, y2):
            raise ValueError("Fusion targets are not aligned.")

        if not np.array_equal(engine_ids_1, engine_ids_2):
            raise ValueError("Fusion engine ids are not aligned.")

        X1, _, _ = normalize_features_global(X1)
        X2, _, _ = normalize_features_global(X2)

        y = y1
        engine_ids = engine_ids_1
        labels = make_stage_labels(y)

        fusion_result = train_two_branch_late_fusion(
            windows_1=X1,
            windows_2=X2,
            labels=labels,
            hidden_size=HIDDEN_SIZE,
            epochs=EPOCHS,
            lr=LR,
            batch_size=BATCH_SIZE,
            random_seed=RANDOM_SEED,
            test_size=0.3,
            w1=W1,
            w2=W2,
            verbose=VERBOSE_TRAINING,
        )

        score_all = fusion_result["score_all_fused"]
        score_cal = fusion_result["score_cal_fused"]
        y_cal = fusion_result["y_cal"]

    else:
        raise ValueError("Only one-sensor or two-sensor experiments are supported.")

    # conformal
    y_cal_score = y_cal.astype(np.float32) / 2.0
    qhat = split_conformal_score_interval(y_cal_score, score_cal, alpha=ALPHA)
    score_lo, score_hi = apply_score_interval(score_all, qhat, min_score=0.0, max_score=1.0)
    mean_width = float(np.mean(score_hi - score_lo))

    # thresholds
    q1, q2 = learn_state_thresholds_from_calibration(y_cal, score_cal)

    # score smoothing and optional monotone enforcement
    scores_smoothed = smooth_scores_per_engine(
        score_all,
        engine_ids,
        window=SMOOTHING_WINDOW,
    )

    if use_monotone:
        scores_for_states = enforce_monotone_per_engine(scores_smoothed, engine_ids)
    else:
        scores_for_states = scores_smoothed.copy()

    states = score_to_state(scores_for_states, q1=q1, q2=q2)

    # DTMC
    if USE_PER_ENGINE_DTMC:
        counts, P = estimate_transition_matrix_per_engine(states, engine_ids, n_states=3)
    else:
        counts, P = estimate_transition_matrix(states, n_states=3)

    p_H = finite_horizon_reachability(P, target_state=2, horizon=HORIZON)

    D = degradation_indicator(scores_for_states, engine_ids)

    return {
        "qhat": float(qhat),
        "width": float(mean_width),
        "PH_crit": float(p_H[0]),
        "D": float(D),
        "P": P,
        "counts": counts,
        "q1": q1,
        "q2": q2,
    }


# =========================================================
# OUTPUT HELPERS
# =========================================================
def print_console_table(rows):
    print("\n")
    print("=" * 86)
    print("Comparison of single-sensor and fusion configurations")
    print("=" * 86)
    print(f"{'Method':<20} {'Monotone':<10} {'qhat':>10} {'Width':>10} {'P_H(crit)':>12} {'D':>10}")
    print("-" * 86)

    for r in rows:
        print(
            f"{r['Method']:<20} "
            f"{r['Monotone']:<10} "
            f"{r['qhat']:>10.4f} "
            f"{r['Width']:>10.4f} "
            f"{r['PH_crit']:>12.4f} "
            f"{r['D']:>10.4f}"
        )

    print("=" * 86)


def save_csv(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Method,Monotone,qhat,Width,PH_crit,D\n")

        for r in rows:
            f.write(
                f"{r['Method']},{r['Monotone']},"
                f"{r['qhat']:.4f},{r['Width']:.4f},"
                f"{r['PH_crit']:.4f},{r['D']:.4f}\n"
            )


def save_latex_table(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparison of single-sensor and fusion configurations.}\n")
        f.write("\\label{tab:single_fusion_comparison}\n")
        f.write("\\begin{tabular}{llcccc}\n")
        f.write("\\toprule\n")
        f.write("Method & Monotone & $\\hat{q}$ & Width & $P_H(\\mathrm{crit})$ & $D$ \\\\\n")
        f.write("\\midrule\n")

        for r in rows:
            f.write(
                f"{r['Method']} & {r['Monotone']} & "
                f"{r['qhat']:.4f} & {r['Width']:.4f} & "
                f"{r['PH_crit']:.4f} & {r['D']:.4f} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def safe_name(text):
    return (
        text.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("+", "_")
        .replace("/", "_")
    )
def print_saved_sensor_screening(screening_path):
    print("\n")
    print("=" * 90)
    print("FINAL SENSOR SCREENING SUMMARY")
    print("=" * 90)

    if not os.path.exists(screening_path):
        print(f"Screening file not found: {screening_path}")
        print("Run nasa_sensor_screening.py first.")
        return

    with open(screening_path, "r", encoding="utf-8") as f:
        print(f.read().strip())


# =========================================================
# MAIN
# =========================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_fd001_train(FILEPATH)
    df = compute_rul(df)
    df = add_normalized_degradation_target(df, rul_cap=125)

    print_fd001_summary(df)

    engine_col, cycle_col = detect_id_and_cycle_columns(df)

    if USE_SELECTED_ENGINES_ONLY:
        df, selected_engines, _, _, _ = filter_to_clean_engines(
            df,
            min_cycles=MIN_CYCLES,
            max_engines=MAX_ENGINES,
        )

        print("\nSelected engines:")
        print(selected_engines)

    experiments = [
        ("Single (S11)", ["sensor_11"], False),
        ("Single (S11)", ["sensor_11"], True),
        ("Fusion (S11+S15)", ["sensor_11", "sensor_15"], False),
        ("Fusion (S11+S15)", ["sensor_11", "sensor_15"], True),
    ]

    rows = []

    for method, sensors, monotone in experiments:
        print("\n" + "=" * 70)
        print(f"Running: {method}, Monotone={'Yes' if monotone else 'No'}")
        print("=" * 70)

        res = run_one_experiment(
            df=df,
            engine_col=engine_col,
            cycle_col=cycle_col,
            sensors=sensors,
            use_monotone=monotone,
        )

        rows.append({
            "Method": method,
            "Monotone": "Yes" if monotone else "No",
            "qhat": res["qhat"],
            "Width": res["width"],
            "PH_crit": res["PH_crit"],
            "D": res["D"],
        })

        exp_dir = os.path.join(
            OUT_DIR,
            f"{safe_name(method)}_{'monotone' if monotone else 'no_monotone'}",
        )
        os.makedirs(exp_dir, exist_ok=True)

        np.savetxt(
            os.path.join(exp_dir, "transition_matrix.csv"),
            res["P"],
            delimiter=",",
            fmt="%.10f",
        )

        np.savetxt(
            os.path.join(exp_dir, "transition_counts.csv"),
            res["counts"],
            delimiter=",",
            fmt="%d",
        )

    csv_path = os.path.join(OUT_DIR, "single_vs_fusion_comparison.csv")
    tex_path = os.path.join(OUT_DIR, "single_vs_fusion_comparison_table.tex")

    save_csv(rows, csv_path)
    save_latex_table(rows, tex_path)

    screening_path = os.path.join(
        "results_nasa_single_sensor_screening_improved",
        "final_ranking.txt"
    )

    print_saved_sensor_screening(screening_path)
    print_console_table(rows)

    

    print("\nSaved files:")
    print(" -", csv_path)
    print(" -", tex_path)
    print(" - transition matrices saved inside each experiment folder")


if __name__ == "__main__":
    main()