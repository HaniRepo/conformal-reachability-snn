import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch

from src.filtration_loader import (
    load_filtration_dataset,
    print_filtration_dataset_summary,
)
from src.filtration_features import (
    compute_pressure_drop,
    mark_clogging,
    truncate_at_first_clogging,
    compute_rul_per_run,
    add_normalized_degradation_target,
    extract_rolling_features_dataset,
    build_feature_matrix,
    make_filtration_stage_labels,
    get_selected_feature_columns,
    print_filtration_feature_summary,
)
from src.snn_score_nasa import train_snn_and_get_nasa_scores
from src.conformal_score import split_conformal_score_interval, apply_score_interval
from src.reachability import finite_horizon_reachability


plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.4,
})


ROOT_DIR = "data/data_filter"
OUT_DIR = "results_extended_filtration"
CLOG_THRESHOLD = 20.0
WINDOW_SIZE = 100
ALPHA = 0.1
HORIZON = 25
SMOOTH_WINDOW = 15

T1 = 0.0092
T2 = 0.5001
T3 = 0.9999

STATE_NAMES_4 = ["Healthy", "Early degr.", "Late degr.", "Critical"]
STATE_NAMES_3 = ["Healthy", "Degrading", "Critical"]

def make_state_thresholds(scores: np.ndarray, n_states: int = 4, mode: str = "quantile") -> np.ndarray:
    if n_states == 3:
        if mode == "fixed":
            return np.array([0.33, 0.66], dtype=float)
        return np.quantile(scores, [1 / 3, 2 / 3])

    if n_states == 4:
        if mode == "fixed":
            return np.array([0.20, 0.50, 0.80], dtype=float)
        return np.quantile(scores, [0.25, 0.50, 0.75])

    qs = np.linspace(0, 1, n_states + 1)[1:-1]
    return np.quantile(scores, qs)


def discretize_scores(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    states = np.zeros(len(scores), dtype=int)
    for i, thr in enumerate(thresholds):
        states[scores >= thr] = i + 1
    return states


def estimate_transition_matrix(states: np.ndarray, n_states: int = 4):
    counts = np.zeros((n_states, n_states), dtype=np.int64)

    for i in range(len(states) - 1):
        counts[states[i], states[i + 1]] += 1

    P = counts.astype(float)
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = P / row_sums

    return counts, P

def fit_feature_normalizer(X_train: np.ndarray):
    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_feature_normalizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((X - mean) / std).astype(np.float32)


def detect_run_column(df):
    for c in ["run_id", "run_name", "run", "sample", "id"]:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find run identifier column. Available: {list(df.columns)}")


def choose_middle_run(df, run_col: str):
    if "split" in df.columns:
        df_sub = df[df["split"] == "Validation"].copy()
        if len(df_sub) == 0:
            df_sub = df.copy()
    else:
        df_sub = df.copy()

    lengths = df_sub.groupby(run_col).size().sort_values()
    return lengths.index[len(lengths) // 2]


def moving_average_edge(x: np.ndarray, window: int = 9):
    if window <= 1:
        return x.copy()
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    x_pad = np.pad(x, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x_pad, kernel, mode="valid")


def _resample_to_fixed_length(values: np.ndarray, target_len: int = 100):
    if len(values) == 1:
        return np.full(target_len, values[0], dtype=float)
    x_old = np.linspace(0.0, 1.0, len(values))
    x_new = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_new, x_old, values)


def score_to_state_4(scores: np.ndarray, t1: float, t2: float, t3: float):
    states = np.zeros(len(scores), dtype=int)
    states[scores >= t1] = 1
    states[scores >= t2] = 2
    states[scores >= t3] = 3
    return states


def estimate_transition_matrix_per_run(states: np.ndarray, run_ids: np.ndarray, n_states: int = 4):
    counts = np.zeros((n_states, n_states), dtype=np.int64)

    for rid in np.unique(run_ids):
        mask = run_ids == rid
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


def state_distribution(states: np.ndarray):
    total = len(states)
    out = {}
    for s, name in enumerate(STATE_NAMES_4):
        c = int(np.sum(states == s))
        out[s] = (name, c, c / total if total > 0 else 0.0)
    return out


def label_distribution(labels: np.ndarray):
    total = len(labels)
    out = {}
    for s, name in enumerate(STATE_NAMES_3):
        c = int(np.sum(labels == s))
        out[s] = (name, c, c / total if total > 0 else 0.0)
    return out


def choose_representative_run_df(df, run_col: str):
    if "split" in df.columns:
        df_sub = df[df["split"] == "Validation"].copy()
        if len(df_sub) == 0:
            df_sub = df.copy()
    else:
        df_sub = df.copy()

    lengths = df_sub.groupby(run_col).size().sort_values()
    return lengths.index[len(lengths) // 2]


def choose_representative_run(score_val: np.ndarray, y_val: np.ndarray, run_val: np.ndarray):
    run_errors = []
    for rid in np.unique(run_val):
        mask = run_val == rid
        mae = float(np.mean(np.abs(score_val[mask] - y_val[mask])))
        run_errors.append((rid, mae, np.sum(mask)))

    run_errors.sort(key=lambda x: x[1])
    return run_errors[len(run_errors) // 2][0]


def save_transition_matrix_csv(P: np.ndarray, out_path: str):
    np.savetxt(out_path, P, delimiter=",", fmt="%.10f")


def export_dtmc_to_prism(
    P: np.ndarray,
    init_state: int,
    out_path: str,
    critical_state: int,
    absorb_critical: bool = False,
    reward_type: str | None = None,
):
    n_states = P.shape[0]
    P_export = P.copy()

    if absorb_critical:
        P_export[critical_state, :] = 0.0
        P_export[critical_state, critical_state] = 1.0

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("dtmc\n\n")
        f.write("module health\n")
        f.write(f"    s : [0..{n_states - 1}] init {init_state};\n\n")

        for i in range(n_states):
            terms = []
            for j in range(n_states):
                terms.append(f"{P_export[i, j]:.10f} : (s'={j})")
            f.write(f"    [] s={i} -> " + " + ".join(terms) + ";\n")

        f.write("endmodule\n\n")
        f.write(f'label "critical" = s={critical_state};\n\n')

        if reward_type == "time":
            f.write('rewards "time"\n')
            f.write("    true : 1;\n")
            f.write("endrewards\n")

        elif reward_type == "degradation_occupancy":
            f.write('rewards "degradation_occupancy"\n')
            f.write("    s=1 : 1;\n")
            f.write("    s=2 : 2;\n")
            f.write("endrewards\n")


def save_filtration_storm_models(P: np.ndarray, out_dir: str):
    storm_dir = os.path.join(out_dir, "storm_models")
    os.makedirs(storm_dir, exist_ok=True)

    n_states = P.shape[0]
    critical_state = n_states - 1

    np.savetxt(
        os.path.join(storm_dir, "transition_matrix.csv"),
        P,
        delimiter=",",
        fmt="%.10f",
    )

    for init_state in range(n_states - 1):
        export_dtmc_to_prism(
            P=P,
            init_state=init_state,
            out_path=os.path.join(storm_dir, f"health_init{init_state}_absorb.pm"),
            critical_state=critical_state,
            absorb_critical=True,
            reward_type="time",
        )

    for init_state in range(n_states - 1):
        export_dtmc_to_prism(
            P=P,
            init_state=init_state,
            out_path=os.path.join(storm_dir, f"health_init{init_state}_occ.pm"),
            critical_state=critical_state,
            absorb_critical=False,
            reward_type="degradation_occupancy",
        )

    with open(os.path.join(storm_dir, "storm_commands_expected_time.txt"), "w", encoding="utf-8") as f:
        for init_state in range(n_states - 1):
            f.write(
                f"storm --prism health_init{init_state}_absorb.pm "
                f"--prop 'R=? [F \"critical\"]'\n"
            )

    with open(os.path.join(storm_dir, "storm_commands_occupancy.txt"), "w", encoding="utf-8") as f:
        for init_state in range(n_states - 1):
            for h in range(1, 26):
                f.write(
                    f"storm --prism health_init{init_state}_occ.pm "
                    f"--prop 'R{{\"degradation_occupancy\"}}=? [ C<={h} ]'\n"
                )
            f.write("\n")

    with open(os.path.join(storm_dir, "storm_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Filtration Storm Export Summary\n")
        f.write("===============================\n\n")
        f.write("Generated model sets:\n")
        f.write("1) health_init*_absorb.pm for expected time to failure\n")
        f.write("2) health_init*_occ.pm for degradation occupancy\n\n")
        f.write("Initial states: 0, 1, 2\n")
        f.write(f"Critical/failure state: {critical_state}\n\n")
        f.write("Transition matrix P:\n")
        for row in P:
            f.write("  " + "  ".join(f"{x:.10f}" for x in row) + "\n")


def save_pup_only_plot(df, base_out_dir: str, window_size: int = 100, smooth_window: int = 9):
    pup_dir = os.path.join(base_out_dir, "pup_only")
    os.makedirs(pup_dir, exist_ok=True)

    df_feat = extract_rolling_features_dataset(df, window_size=window_size)
    run_col = detect_run_column(df_feat)
    rep_run = choose_middle_run(df_feat, run_col)

    if "split" in df_feat.columns:
        df_sub = df_feat[df_feat["split"] == "Validation"].copy()
        if len(df_sub) == 0:
            df_sub = df_feat.copy()
    else:
        df_sub = df_feat.copy()

    run_df = df_sub[df_sub[run_col] == rep_run].copy().reset_index(drop=True)

    x = np.linspace(0, 100, len(run_df))
    pup_raw = run_df["pup"].values.astype(float)
    pup_raw = (pup_raw - np.mean(pup_raw)) / (np.std(pup_raw) + 1e-8)
    pup_smooth = moving_average_edge(pup_raw, window=smooth_window)

    plt.figure(figsize=(7.0, 4.8))
    plt.plot(x, pup_smooth, label="pup")
    plt.title("Representative Processed PUP Evolution")
    plt.xlabel("Progress through run toward clogging (%)")
    plt.ylabel("Normalized PUP value")
    plt.xlim(0, 100)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(pup_dir, "pup_only.png"), bbox_inches="tight")
    plt.close()


def plot_representative_pressure_drop_run(df, out_path: str, clog_threshold: float):
    run_col = detect_run_column(df)
    rep_run = choose_representative_run_df(df, run_col)
    run_df = df[df[run_col] == rep_run].copy().reset_index(drop=True)
    x = np.arange(len(run_df))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(x, run_df["drop"].values, linewidth=2.4, label="Pressure drop")
    ax.axhline(clog_threshold, linestyle="--", linewidth=1.5, label="Clogging threshold")

    if "clogged" in run_df.columns and np.any(run_df["clogged"].values == 1):
        clog_idx = int(np.argmax(run_df["clogged"].values == 1))
        ax.axvline(clog_idx, linestyle=":", linewidth=1.5, label="Clogging point")

    ax.set_title("Representative Filtration Run with Progressive Clogging")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Pressure drop")
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_best_validation_run(score_val, score_lo, score_hi, run_val, y_val, out_path, t1, t2, t3, smooth_window=15):
    rep_run = choose_representative_run(score_val, y_val, run_val)
    mask = run_val == rep_run

    progress = np.linspace(0, 100, np.sum(mask))
    s_run = moving_average_edge(score_val[mask], window=smooth_window)
    lo_run = moving_average_edge(score_lo[mask], window=smooth_window)
    hi_run = moving_average_edge(score_hi[mask], window=smooth_window)

    width_mean = float(np.mean(hi_run - lo_run))

    fig, ax = plt.subplots(figsize=(12, 5.3))
    ax.fill_between(progress, lo_run, hi_run, alpha=0.18, label="Conformal interval")
    ax.plot(progress, s_run, linewidth=2.8, label="Smoothed SNN score")

    ax.axhline(t1, linestyle="--", linewidth=1.2)
    ax.axhline(t2, linestyle="--", linewidth=1.2)
    ax.axhline(t3, linestyle="--", linewidth=1.2)

    ax.text(3, min(t1 + 0.02, 0.08), "Healthy", fontsize=10)
    ax.text(3, min(t2 - 0.04, 0.25), "Early degr.", fontsize=10)
    ax.text(3, min(t3 - 0.06, 0.75), "Late degr.", fontsize=10)
    ax.text(84, 0.95, "Critical", fontsize=10)

    ax.set_title(f"Representative SNN Score with Conformal Uncertainty (mean width={width_mean:.3f})")
    ax.set_xlabel("Progress through run toward clogging (%)")
    ax.set_ylabel("SNN degradation score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, 100)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix_from_probs(y_true: np.ndarray, probs: np.ndarray, out_path: str):
    pred = np.argmax(probs, axis=1)
    cm = confusion_matrix(y_true, pred, labels=[0, 1, 2]).astype(float)

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized fraction")

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(STATE_NAMES_3, rotation=20)
    ax.set_yticklabels(STATE_NAMES_3)

    for i in range(3):
        for j in range(3):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color)

    ax.set_title("Filtration Validation Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_transition_matrix_heatmap(P: np.ndarray, out_path: str):
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(P, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Transition probability")

    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xticklabels(STATE_NAMES_4, rotation=22)
    ax.set_yticklabels(STATE_NAMES_4)

    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            val = P[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color)

    ax.set_title("Filtration DTMC Transition Matrix")
    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_dtmc_graph_4state(P: np.ndarray, out_path: str):
    labels = STATE_NAMES_4
    pos = {0: (0.0, 0.0), 1: (2.7, 0.0), 2: (5.4, 0.0), 3: (8.1, 0.0)}

    fig, ax = plt.subplots(figsize=(12.0, 3.8))

    for i, label in enumerate(labels):
        x, y = pos[i]
        circle = plt.Circle((x, y), 0.50, fill=False, linewidth=2.2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center", fontsize=11)

    def forward_arrow(i, j, y_offset=0.18):
        x1, y1 = pos[i]
        x2, y2 = pos[j]
        ax.annotate("", xy=(x2 - 0.55, y2 + y_offset), xytext=(x1 + 0.55, y1 + y_offset),
                    arrowprops=dict(arrowstyle="->", linewidth=1.8))
        ax.text((x1 + x2) / 2, y_offset + 0.12, f"{P[i, j]:.4f}",
                ha="center", va="bottom", fontsize=10)

    def backward_arrow(i, j, y_offset=-0.18):
        x1, y1 = pos[i]
        x2, y2 = pos[j]
        ax.annotate("", xy=(x2 + 0.55, y2 + y_offset), xytext=(x1 - 0.55, y1 + y_offset),
                    arrowprops=dict(arrowstyle="->", linewidth=1.8))
        ax.text((x1 + x2) / 2, y_offset - 0.18, f"{P[i, j]:.4f}",
                ha="center", va="top", fontsize=10)

    def self_loop(i):
        x, y = pos[i]
        ax.annotate("", xy=(x + 0.35, y + 0.55), xytext=(x - 0.35, y + 0.55),
                    arrowprops=dict(arrowstyle="->", linewidth=1.8, connectionstyle="arc3,rad=-1.9"))
        ax.text(x, y + 1.18, f"{P[i, i]:.4f}", ha="center", va="bottom", fontsize=11)

    for i in range(4):
        self_loop(i)

    forward_arrow(0, 1)
    backward_arrow(1, 0)
    forward_arrow(1, 2)
    backward_arrow(2, 1)
    forward_arrow(2, 3)
    backward_arrow(3, 2)

    ax.set_xlim(-0.9, 9.0)
    ax.set_ylim(-1.05, 1.55)
    ax.axis("off")
    ax.set_title("Learned 4-State DTMC", fontsize=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_reachability_barchart(p_H: np.ndarray, out_path: str):
    x = np.arange(len(p_H))
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    bars = ax.bar(x, p_H, width=0.62)

    ax.set_xticks(x)
    ax.set_xticklabels(STATE_NAMES_4)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Reachability probability")
    ax.set_title(f"Filtration Finite-Horizon Reachability to Critical State (H={HORIZON})")

    for rect, val in zip(bars, p_H):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_results_summary(filepath, qhat, mean_width, t1, t2, t3, label_stats, abstract_stats,
                         P, p_H, acc_val, X_train_shape, X_val_shape, scores_shape,
                         states_shape, score_min, score_max):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("Filtration Results Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Official validation classification accuracy: {acc_val:.4f}\n")
        f.write(f"Conformal score radius qhat: {qhat:.4f}\n")
        f.write(f"Mean conformal interval width: {mean_width:.4f}\n")
        f.write(f"State thresholds: {t1:.4f}, {t2:.4f}, {t3:.4f}\n")
        f.write(f"Reachability horizon: H={HORIZON}\n")
        f.write(f"Validation score range: min={score_min:.4f}, max={score_max:.4f}\n\n")

        f.write(f"Training feature matrix shape: {X_train_shape}\n")
        f.write(f"Validation feature matrix shape: {X_val_shape}\n")
        f.write(f"Validation score shape: {scores_shape}\n")
        f.write(f"Validation states shape: {states_shape}\n\n")

        f.write("Validation 3-Class Label Distribution\n")
        for s in [0, 1, 2]:
            name, c, p = label_stats[s]
            f.write(f"  {name} ({s}): {c} ({p:.2%})\n")

        f.write("\nAbstracted State Distribution\n")
        for s in [0, 1, 2, 3]:
            name, c, p = abstract_stats[s]
            f.write(f"  {name} ({s}): {c} ({p:.2%})\n")

        f.write("\nTransition Matrix P\n")
        for row in P:
            f.write("  " + "  ".join(f"{x:.10f}" for x in row) + "\n")

        f.write(f"\nFinite-Horizon Reachability to Final State within H={HORIZON}\n")
        for i, prob in enumerate(p_H):
            f.write(f"  From state {i}: {prob:.4f}\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_filtration_dataset(ROOT_DIR)
    print_filtration_dataset_summary(df)

    df = compute_pressure_drop(df)
    df = mark_clogging(df, clog_threshold=CLOG_THRESHOLD)
    df = truncate_at_first_clogging(df)
    df = compute_rul_per_run(df)
    df = add_normalized_degradation_target(df, rul_cap=None)
    '''
    problem_dir = os.path.join(OUT_DIR, "problem")
    os.makedirs(problem_dir, exist_ok=True)

    plot_representative_pressure_drop_run(
        df,
        out_path=os.path.join(problem_dir, "representative_pressure_drop_run.png"),
        clog_threshold=CLOG_THRESHOLD,
    )
    

    save_pup_only_plot(df, base_out_dir=OUT_DIR, window_size=WINDOW_SIZE)
    '''
    
    df_feat = extract_rolling_features_dataset(df, window_size=WINDOW_SIZE)
    print_filtration_feature_summary(df_feat)

    df_train = df_feat[df_feat["split"] == "Training"].copy()
    df_val = df_feat[df_feat["split"] == "Validation"].copy()

    feature_cols = get_selected_feature_columns()

    X_train, y_train, run_train = build_feature_matrix(df_train, feature_cols=feature_cols)
    X_val, y_val, run_val = build_feature_matrix(df_val, feature_cols=feature_cols)

    X_mean, X_std = fit_feature_normalizer(X_train)
    X_train = apply_feature_normalizer(X_train, X_mean, X_std)
    X_val = apply_feature_normalizer(X_val, X_mean, X_std)

    labels_train = make_filtration_stage_labels(y_train, q1=0.33, q2=0.66)
    labels_val = make_filtration_stage_labels(y_val, q1=0.33, q2=0.66)

    result_train = train_snn_and_get_nasa_scores(
        X_train,
        labels_train,
        hidden_size=32,
        epochs=25,
        lr=1e-3,
        batch_size=64,
        random_seed=42,
    )

    model = result_train["model"]
    model.eval()

    class_values = np.array([0.0, 0.5, 1.0], dtype=np.float32)

    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        logits_val = model(X_val_tensor)
        probs_val = torch.softmax(logits_val, dim=1).cpu().numpy()

    pred_val = np.argmax(probs_val, axis=1)
    acc_val = float(np.mean(pred_val == labels_val))

    score_val = np.sum(probs_val * class_values, axis=1)

    y_val_score = labels_val.astype(np.float32) / 2.0
    qhat = split_conformal_score_interval(y_val_score, score_val, alpha=ALPHA)
    score_lo, score_hi = apply_score_interval(score_val, qhat, min_score=0.0, max_score=1.0)
    mean_width = float(np.mean(score_hi - score_lo))
    # To keep it consistant 
    #states_4 = score_to_state_4(score_val, T1, T2, T3)
    #counts, P = estimate_transition_matrix_per_run(states_4, run_val, n_states=4)
    thresholds = make_state_thresholds(score_val, n_states=4, mode="quantile")
    states_4 = discretize_scores(score_val, thresholds)
    counts, P = estimate_transition_matrix(states_4, n_states=4)
    
    
    p_H = finite_horizon_reachability(P, target_state=3, horizon=HORIZON)

    save_filtration_storm_models(P, out_dir=OUT_DIR)

    label_stats = label_distribution(labels_val)
    abstract_stats = state_distribution(states_4)
    '''
    plot_best_validation_run(
        score_val=score_val,
        score_lo=score_lo,
        score_hi=score_hi,
        run_val=run_val,
        y_val=y_val,
        out_path=os.path.join(OUT_DIR, "best_validation_run.png"),
        t1=T1,
        t2=T2,
        t3=T3,
        smooth_window=SMOOTH_WINDOW,
    )
    plot_dtmc_graph_4state(
        P=P,
        out_path=os.path.join(OUT_DIR, "dtmc_graph_4state.png"),
    )

    plot_reachability_barchart(
        p_H=p_H,
        out_path=os.path.join(OUT_DIR, "reachability_barplot.png"),
    )

    '''
    plot_confusion_matrix_from_probs(
        y_true=labels_val,
        probs=probs_val,
        out_path=os.path.join(OUT_DIR, "confusion_matrix.png"),
    )

    plot_transition_matrix_heatmap(
        P=P,
        out_path=os.path.join(OUT_DIR, "transition_matrix_heatmap.png"),
    )

    

    save_transition_matrix_csv(P, os.path.join(OUT_DIR, "transition_matrix.csv"))

    save_results_summary(
        filepath=os.path.join(OUT_DIR, "results_summary.txt"),
        qhat=qhat,
        mean_width=mean_width,
        t1=float(thresholds[0]),
        t2=float(thresholds[1]),
        t3=float(thresholds[2]),
        label_stats=label_stats,
        abstract_stats=abstract_stats,
        P=P,
        p_H=p_H,
        acc_val=acc_val,
        X_train_shape=X_train.shape,
        X_val_shape=X_val.shape,
        scores_shape=score_val.shape,
        states_shape=states_4.shape,
        score_min=float(np.min(score_val)),
        score_max=float(np.max(score_val)),
    )

    print("\nSaved filtration results under:", OUT_DIR)


if __name__ == "__main__":
    main()