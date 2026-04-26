import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.nasa_loader import (
    load_fd001_train,
    compute_rul,
    add_normalized_degradation_target,
    print_fd001_summary,
)
from src.snn_score_nasa import train_snn_and_get_nasa_scores
from src.conformal_score import split_conformal_score_interval, apply_score_interval
from src.reachability import finite_horizon_reachability





# =========================================================
# CONFIG SWITCHES
# =========================================================
FILEPATH = r"data/data_nasa/train_FD001.txt"   # change if needed
OUT_DIR = "results_nasa"

SELECTED_SENSOR = "sensor_11"             # try "sensor_7" too

USE_LIGHT_PREPROCESSING = True            # True = raw + smooth + slope
USE_MONOTONE_ENFORCEMENT = True           # True = main paper version
USE_PER_ENGINE_DTMC = True                # keep True
USE_SELECTED_ENGINES_ONLY = True          # use a cleaner subset for figures/experiment

WINDOW_SIZE = 30
STRIDE = 1
SMOOTHING_WINDOW = 3
HORIZON = 10
ALPHA = 0.1

HIDDEN_SIZE = 64
EPOCHS = 40
LR = 1e-3
BATCH_SIZE = 64
RANDOM_SEED = 42

MIN_CYCLES = 180
MAX_ENGINES = 25
# =========================================================
# HELPER PRISM
# =========================================================

def build_prism_dtmc_text(P: np.ndarray, init_state: int, horizon: int = 10) -> str:
    n_states = P.shape[0]
    critical_state = n_states - 1

    lines = []
    lines.append("dtmc\n")
    lines.append(f"const int H = {horizon};\n\n")

    lines.append("module health_model\n")
    lines.append(f"    s : [0..{n_states - 1}] init {init_state};\n")
    lines.append("    t : [0..H] init 0;\n\n")

    for i in range(n_states):
        terms = []
        for j in range(n_states):
            terms.append(f"{P[i, j]:.10f} : (s'={j}) & (t'=t+1)")
        lines.append(
            f"    [] s={i} & t<H -> " + " + ".join(terms) + ";\n"
        )

    lines.append("    [] t=H -> 1.0 : (s'=s) & (t'=t);\n")
    lines.append("endmodule\n\n")
    lines.append(f'label "critical" = s={critical_state};\n\n')
    lines.append("// Example property to check in Storm:\n")
    lines.append('// P=? [ F<=H "critical" ]\n')

    return "".join(lines)


def save_prism_models(P: np.ndarray, out_dir: str, horizon: int, prefix: str = "health_init"):
    for init_state in range(P.shape[0]):
        prism_text = build_prism_dtmc_text(
            P,
            init_state=init_state,
            horizon=horizon,
        )
        out_path = os.path.join(out_dir, f"{prefix}{init_state}.pm")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(prism_text)


def save_storm_commands(out_dir: str, horizon: int, prefix: str = "health_init"):
    """
    Save example Storm commands for all initial states.
    """
    cmd_path = os.path.join(out_dir, "storm_commands.txt")
    with open(cmd_path, "w", encoding="utf-8") as f:
        for init_state in range(3):
            model_name = f"{prefix}{init_state}.pm"
            f.write(
                f'storm --prism "{model_name}" --prop \'P=? [ F<={horizon} "critical" ]\'\n'
            )


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


def choose_representative_engine(df, engine_col: str, cycle_col: str) -> int:
    lengths = df.groupby(engine_col)[cycle_col].max().sort_values()
    return int(lengths.index[len(lengths) // 2])


def normalize_1d_per_engine(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    mn = np.min(x)
    mx = np.max(x)
    return (x - mn) / (mx - mn + 1e-8)


def build_processed_signal(raw_signal: np.ndarray, smooth_window: int = 9):
    smooth = moving_average_edge(raw_signal, window=smooth_window)
    slope = np.diff(smooth, prepend=smooth[0])

    raw_n = normalize_1d_per_engine(raw_signal)
    smooth_n = normalize_1d_per_engine(smooth)

    slope_min = np.min(slope)
    slope_max = np.max(slope)
    slope_n = (slope - slope_min) / (slope_max - slope_min + 1e-8)

    return raw_n, smooth_n, slope_n


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
            raw_n, smooth_n, slope_n = build_processed_signal(raw_signal, smooth_window=smooth_window)
            feature_mat = np.column_stack([raw_n, smooth_n, slope_n])   # [T, 3]
        else:
            raw_n = normalize_1d_per_engine(raw_signal)
            feature_mat = raw_n[:, None]                                # [T, 1]

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


def _resample_to_fixed_length(values: np.ndarray, target_len: int = 100):
    if len(values) == 1:
        return np.full(target_len, values[0], dtype=float)
    x_old = np.linspace(0.0, 1.0, len(values))
    x_new = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_new, x_old, values)


# =========================================================
# PLOTS FOR DATA / PROBLEM
# =========================================================
def plot_raw_signal_representative_engine(
    df,
    sensor_col: str,
    engine_col: str,
    cycle_col: str,
    out_path: str,
):
    rep_eng = choose_representative_engine(df, engine_col, cycle_col)
    run_df = df[df[engine_col] == rep_eng].sort_values(cycle_col).copy()

    sensor_num = sensor_col.split("_")[-1]

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.plot(run_df[cycle_col].values, run_df[sensor_col].values, linewidth=2.2)

    ax.set_title(f"Representative FD001 engine (Sensor {sensor_num})")
    ax.set_xlabel("Cycle")
    ax.set_ylabel(f"HPC outlet static pressure (Sensor {sensor_num})")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_processed_signal_representative_engine(
    df,
    sensor_col: str,
    engine_col: str,
    cycle_col: str,
    out_path: str,
    smooth_window: int = 9,
):
    rep_eng = choose_representative_engine(df, engine_col, cycle_col)
    run_df = df[df[engine_col] == rep_eng].sort_values(cycle_col).copy()

    cycles = run_df[cycle_col].values
    raw_signal = run_df[sensor_col].values.astype(float)
    raw_n, smooth_n, slope_n = build_processed_signal(raw_signal, smooth_window=smooth_window)

    sensor_num = sensor_col.split("_")[-1]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(cycles, raw_n, linewidth=1.5, alpha=0.45, label="Raw normalized signal")
    ax.plot(cycles, smooth_n, linewidth=2.5, label="Smoothed signal")
    #ax.plot(cycles, slope_n, linewidth=1.8, alpha=0.85, label="Local trend")

    ax.set_title(f"Processed input signal for representative engine (Sensor {sensor_num})")
    ax.set_xlabel("Cycle")
    ax.set_ylabel(f"Normalized Sensor ({sensor_num})")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", frameon=True)
    ax.plot(cycles, raw_n, linewidth=1.5, alpha=0.25, label="Raw normalized signal")  # lower alpha
    #ax.plot(cycles, slope_n, linewidth=1.6, alpha=0.6, label="Local trend")  # reduce dominance
    
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_mean_degradation_trend(df, sensor_col: str, engine_col: str, out_path: str):
    all_runs = []
    max_len = 0

    for _, run_df in df.groupby(engine_col):
        vals = run_df[sensor_col].values.astype(float)

        vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)

        all_runs.append(vals)
        max_len = max(max_len, len(vals))

    aligned = np.full((len(all_runs), max_len), np.nan)

    for i, vals in enumerate(all_runs):
        aligned[i, :len(vals)] = vals

    mean_trend = np.nanmean(aligned, axis=0)
    std_trend = np.nanstd(aligned, axis=0)

    x = np.arange(max_len)
    sensor_num = sensor_col.split("_")[-1]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, mean_trend, linewidth=2.8, label="Mean trend")
    ax.fill_between(
        x,
        mean_trend - std_trend,
        mean_trend + std_trend,
        alpha=0.25,
        label=r"$\pm 1$ std",
    )

    ax.set_xlabel("Cycle")
    ax.set_ylabel(f"Normalized Sensor {sensor_num}")
    ax.set_title(f"Average FD001 degradation trend (Sensor {sensor_num})")
    ax.set_xlim(0, 300)
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

# =========================================================
# PLOTS FOR RESULTS
# =========================================================
def plot_dtmc_graph_3state(P: np.ndarray, out_path: str):
    labels = ["Healthy", "Degrading", "Critical"]

    pos = {
        0: (0.0, 0.0),
        1: (3.0, 0.0),
        2: (6.0, 0.0),
    }

    fig, ax = plt.subplots(figsize=(9.5, 3.8))

    for i, label in enumerate(labels):
        x, y = pos[i]
        circle = plt.Circle((x, y), 0.50, fill=False, linewidth=2.2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center", fontsize=12)

    def forward_arrow(i, j, y_offset=0.18):
        x1, y1 = pos[i]
        x2, y2 = pos[j]

        ax.annotate(
            "",
            xy=(x2 - 0.55, y2 + y_offset),
            xytext=(x1 + 0.55, y1 + y_offset),
            arrowprops=dict(arrowstyle="->", linewidth=1.8),
        )

        ax.text(
            (x1 + x2) / 2,
            y_offset + 0.12,
            f"{P[i, j]:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    def backward_arrow(i, j, y_offset=-0.18):
        x1, y1 = pos[i]
        x2, y2 = pos[j]

        ax.annotate(
            "",
            xy=(x2 + 0.55, y2 + y_offset),
            xytext=(x1 - 0.55, y1 + y_offset),
            arrowprops=dict(arrowstyle="->", linewidth=1.8),
        )

        ax.text(
            (x1 + x2) / 2,
            y_offset - 0.18,
            f"{P[i, j]:.4f}",
            ha="center",
            va="top",
            fontsize=10,
        )

    def self_loop(i):
        x, y = pos[i]

        ax.annotate(
            "",
            xy=(x + 0.35, y + 0.55),
            xytext=(x - 0.35, y + 0.55),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=1.8,
                connectionstyle="arc3,rad=-1.9",
            ),
        )

        ax.text(
            x,
            y + 1.18,
            f"{P[i, i]:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    for i in range(3):
        self_loop(i)

    # adjacent transitions only
    forward_arrow(0, 1)
    backward_arrow(1, 0)

    forward_arrow(1, 2)
    backward_arrow(2, 1)

    ax.set_xlim(-0.9, 6.9)
    ax.set_ylim(-1.05, 1.55)
    ax.axis("off")
    ax.set_title("Learned 3-State DTMC", fontsize=15)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

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


def plot_snn_score_with_conformal(
    scores: np.ndarray,
    score_lo: np.ndarray,
    score_hi: np.ndarray,
    engine_ids: np.ndarray,
    out_path: str,
    q1: float,
    q2: float,
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

    ax.set_title(f"SNN degradation score with conformal uncertainty (engine {eng})")
    ax.set_xlabel("Window index within engine")
    ax.set_ylabel("Degradation score")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_transition_matrix_heatmap(P: np.ndarray, out_path: str):
    labels = ["Healthy", "Degrading", "Critical"]

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    im = ax.imshow(P, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Transition probability")

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(labels, rotation=18)
    ax.set_yticklabels(labels)

    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            val = P[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color)

    ax.set_title("FD001 DTMC transition matrix")
    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_reachability_barchart(p_H: np.ndarray, out_path: str):
    labels = ["Healthy", "Degrading", "Critical"]
    x = np.arange(len(p_H))

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    bars = ax.bar(x, p_H, width=0.62)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Reachability probability")
    ax.set_title("Finite-horizon reachability to the critical state")

    for rect, val in zip(bars, p_H):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_nasa_storm_models(P: np.ndarray, out_dir: str, horizon_list=None) -> None:
    if horizon_list is None:
        horizon_list = [1, 2, 5, 10, 15, 20, 25]

    storm_dir = os.path.join(out_dir, "storm_models")
    os.makedirs(storm_dir, exist_ok=True)

    np.savetxt(
        os.path.join(storm_dir, "transition_matrix.csv"),
        P,
        delimiter=",",
        fmt="%.10f",
    )

    save_prism_models(
        P=P,
        out_dir=storm_dir,
        horizon=HORIZON,
        prefix="nasa_init",
    )

    with open(os.path.join(storm_dir, "storm_commands.txt"), "w", encoding="utf-8") as f:
        for init_state in [0, 1]:
            for h in horizon_list:
                f.write(
                    f"storm --prism nasa_init{init_state}.pm "
                    f"--prop 'P=? [F<={h} \"critical\"]'\n"
                )
            f.write("\n")

    with open(os.path.join(storm_dir, "storm_summary.txt"), "w", encoding="utf-8") as f:
        f.write("NASA FD001 Storm Export Summary\n")
        f.write("===============================\n\n")
        f.write("Model type: bounded DTMC with explicit time counter t\n")
        f.write(f"Internal model horizon H: {HORIZON}\n")
        f.write("Initial states in commands: 0, 1\n")
        f.write("Critical state: 2\n")
        f.write("Command horizons: " + ", ".join(str(h) for h in horizon_list) + "\n")
        f.write(
            "Note: for command horizons larger than H, the reachability "
            "probability remains constant because the model self-loops at t=H.\n\n"
        )
        f.write("Transition matrix P:\n")
        for row in P:
            f.write("  " + "  ".join(f"{x:.10f}" for x in row) + "\n")

# =========================================================
# MAIN
# =========================================================
def main():
    mode_name = "monotone" if USE_MONOTONE_ENFORCEMENT else "raw"
    prep_name = "lightprep" if USE_LIGHT_PREPROCESSING else "rawsignal"
    out_dir = os.path.join(OUT_DIR, f"{prep_name}_{mode_name}")
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------
    # Load NASA FD001
    # ---------------------------------------
    df = load_fd001_train(FILEPATH)
    df = compute_rul(df)
    df = add_normalized_degradation_target(df, rul_cap=125)

    print_fd001_summary(df)
    print("\nColumns:")
    print(df.columns.tolist())

    engine_col, cycle_col = detect_id_and_cycle_columns(df)

    if USE_SELECTED_ENGINES_ONLY:
        df, selected_engines, _, _, _ = filter_to_clean_engines(
            df,
            min_cycles=MIN_CYCLES,
            max_engines=MAX_ENGINES,
        )
        print("\nSelected engines for cleaner experiment:")
        print(selected_engines)

    # ---------------------------------------
    # Problem/setup figures
    # ---------------------------------------
    plot_raw_signal_representative_engine(
        df=df,
        sensor_col=SELECTED_SENSOR,
        engine_col=engine_col,
        cycle_col=cycle_col,
        out_path=os.path.join(out_dir, "nasa_raw_signal_representative_engine.png"),
    )

    plot_processed_signal_representative_engine(
        df=df,
        sensor_col=SELECTED_SENSOR,
        engine_col=engine_col,
        cycle_col=cycle_col,
        out_path=os.path.join(out_dir, "nasa_processed_signal_representative_engine.png"),
        smooth_window=SMOOTHING_WINDOW,
    )
    
    problem_dir = os.path.join(out_dir, "problem")
    os.makedirs(problem_dir, exist_ok=True)

    plot_mean_degradation_trend(
        df=df,
        sensor_col=SELECTED_SENSOR,
        engine_col=engine_col,
        out_path=os.path.join(problem_dir, "nasa_fd001_mean_trend.png"),
)

    

    # ---------------------------------------
    # Build window dataset
    # ---------------------------------------
    X, y, engine_ids = build_window_dataset_from_selected_sensor(
        df=df,
        sensor_col=SELECTED_SENSOR,
        engine_col=engine_col,
        cycle_col=cycle_col,
        target_col="y_deg",
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        use_light_preprocessing=USE_LIGHT_PREPROCESSING,
        smooth_window=SMOOTHING_WINDOW,
    )

    print("\nWindow dataset summary:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("num engines:", len(np.unique(engine_ids)))
    print("input mode:", "raw + smooth + slope" if USE_LIGHT_PREPROCESSING else "raw only")

    X, X_mean, X_std = normalize_features_global(X)
    labels = make_stage_labels(y)

    # ---------------------------------------
    # Train SNN
    # ---------------------------------------
    result = train_snn_and_get_nasa_scores(
        X,
        labels,
        hidden_size=HIDDEN_SIZE,
        epochs=EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        random_seed=RANDOM_SEED,
    )

    scores = result["score_all"]

    # ---------------------------------------
    # Conformal
    # ---------------------------------------
    y_cal_score = result["y_cal"].astype(np.float32) / 2.0
    qhat = split_conformal_score_interval(y_cal_score, result["score_cal"], alpha=ALPHA)
    score_lo, score_hi = apply_score_interval(scores, qhat, min_score=0.0, max_score=1.0)
    mean_width = float(np.mean(score_hi - score_lo))

    # ---------------------------------------
    # Threshold learning
    # ---------------------------------------
    q1, q2 = learn_state_thresholds_from_calibration(result["y_cal"], result["score_cal"])
    print(f"\nLearned thresholds from calibration: q1={q1:.4f}, q2={q2:.4f}")

    # ---------------------------------------
    # Raw vs monotone state abstraction
    # ---------------------------------------
    scores_smoothed = smooth_scores_per_engine(scores, engine_ids, window=SMOOTHING_WINDOW)

    if USE_MONOTONE_ENFORCEMENT:
        scores_for_states = enforce_monotone_per_engine(scores_smoothed, engine_ids)
    else:
        scores_for_states = scores_smoothed.copy()

    states = score_to_state(scores_for_states, q1=q1, q2=q2)

    # ---------------------------------------
    # DTMC
    # ---------------------------------------
    if USE_PER_ENGINE_DTMC:
        counts, P = estimate_transition_matrix_per_engine(states, engine_ids, n_states=3)
    else:
        counts, P = estimate_transition_matrix(states, n_states=3)

    p_H = finite_horizon_reachability(P, target_state=2, horizon=HORIZON)
    #save_nasa_storm_models(P, out_dir=out_dir)

        # ---------------------------------------
    # Save PRISM / Storm models
    # ---------------------------------------
    save_nasa_storm_models(P, out_dir=out_dir)

    # ---------------------------------------
    # Output/result plots
    # ---------------------------------------
    plot_snn_score_with_conformal(
        scores=scores_for_states,
        score_lo=score_lo,
        score_hi=score_hi,
        engine_ids=engine_ids,
        out_path=os.path.join(out_dir, "nasa_snn_score_with_conformal.png"),
        q1=q1,
        q2=q2,
    )

    plot_transition_matrix_heatmap(
        P=P,
        out_path=os.path.join(out_dir, "nasa_transition_matrix_heatmap.png"),
    )
    plot_dtmc_graph_3state(
        P=P,
        out_path=os.path.join(out_dir, "nasa_dtmc_graph_3state.png"),
    )
    '''
    plot_reachability_barchart(
        p_H=p_H,
        out_path=os.path.join(out_dir, "nasa_reachability_barplot.png"),
    )
    '''

    # ---------------------------------------
    # Save summary
    # ---------------------------------------
    with open(os.path.join(out_dir, "results_summary.txt"), "w", encoding="utf-8") as f:
        f.write("NASA FD001 all-in-one simplified one-signal results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Selected sensor: {SELECTED_SENSOR}\n")
        f.write(f"Use light preprocessing: {USE_LIGHT_PREPROCESSING}\n")
        f.write(f"Use monotone enforcement: {USE_MONOTONE_ENFORCEMENT}\n")
        f.write(f"Use per-engine DTMC counting: {USE_PER_ENGINE_DTMC}\n")
        f.write(f"Window size: {WINDOW_SIZE}\n")
        f.write(f"Stride: {STRIDE}\n")
        f.write(f"Smoothing window: {SMOOTHING_WINDOW}\n")
        f.write(f"Conformal qhat: {qhat:.4f}\n")
        f.write(f"Mean conformal width: {mean_width:.4f}\n")
        f.write(f"Learned thresholds: q1={q1:.4f}, q2={q2:.4f}\n")
        f.write(f"Score range: min={np.min(scores_for_states):.4f}, max={np.max(scores_for_states):.4f}\n\n")
        f.write("Transition matrix P:\n")
        for row in P:
            f.write("  " + "  ".join(f"{x:.6f}" for x in row) + "\n")
        f.write("\nFinite-horizon reachability to critical state:\n")
        for i, prob in enumerate(p_H):
            f.write(f"  From state {i}: {prob:.4f}\n")

    np.savetxt(os.path.join(out_dir, "transition_matrix.csv"), P, delimiter=",", fmt="%.10f")

    print("\nSaved outputs under:", out_dir)
    print("Created:")
    print(" - nasa_raw_signal_representative_engine.png")
    print(" - nasa_processed_signal_representative_engine.png")
    print(" - nasa_snn_score_with_conformal.png")
    print(" - nasa_transition_matrix_heatmap.png")
    print(" - nasa_reachability_barplot.png")
    print(" - nasa_dtmc_graph_3state.png")
    print(" - results_summary.txt")
    print(" - health_init0.pm")
    print(" - health_init1.pm")
    print(" - health_init2.pm")
    print(" - storm_commands.txt")


if __name__ == "__main__":
    main()