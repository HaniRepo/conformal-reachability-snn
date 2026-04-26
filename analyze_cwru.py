import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.data_loader import load_severity_signals
from src.state_discretization import discretize_states
from src.dtmc import estimate_transition_matrix
from src.reachability import finite_horizon_reachability
from src.snn_score import train_snn_and_get_scores
from src.conformal_score import split_conformal_score_interval, apply_score_interval


def normalize(signal: np.ndarray) -> np.ndarray:
    std = np.std(signal)
    if std == 0:
        raise ValueError("Signal standard deviation is zero; cannot normalize.")
    return (signal - np.mean(signal)) / std


def combine_severity_signals(
    normal_signal: np.ndarray,
    mild_signal: np.ndarray,
    medium_signal: np.ndarray,
    severe_signal: np.ndarray,
) -> tuple[np.ndarray, int]:
    n = min(len(normal_signal), len(mild_signal), len(medium_signal), len(severe_signal))
    combined = np.concatenate([
        normal_signal[:n],
        mild_signal[:n],
        medium_signal[:n],
        severe_signal[:n],
    ])
    return combined, n


def create_windows_with_positions(
    signal: np.ndarray,
    window_size: int = 200,
    stride: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    windows = []
    starts = []
    for i in range(0, len(signal) - window_size + 1, stride):
        windows.append(signal[i:i + window_size])
        starts.append(i)
    return np.array(windows), np.array(starts)


def normalize_windows(windows: np.ndarray) -> np.ndarray:
    mean = np.mean(windows, axis=1, keepdims=True)
    std = np.std(windows, axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (windows - mean) / std


def create_severity_labels(window_starts: np.ndarray, segment_length: int) -> np.ndarray:
    labels = np.zeros(len(window_starts), dtype=int)
    for idx, start in enumerate(window_starts):
        if start < segment_length:
            labels[idx] = 0
        elif start < 2 * segment_length:
            labels[idx] = 1
        elif start < 3 * segment_length:
            labels[idx] = 2
        else:
            labels[idx] = 3
    return labels


def severity_distribution(labels: np.ndarray):
    names = {0: "Healthy", 1: "Mild", 2: "Medium", 3: "Severe"}
    total = len(labels)
    out = {}
    for s in [0, 1, 2, 3]:
        c = int(np.sum(labels == s))
        out[s] = (names[s], c, c / total if total > 0 else 0.0)
    return out


def state_distribution(states: np.ndarray):
    names = {0: "Healthy", 1: "Degrading", 2: "Critical"}
    total = len(states)
    out = {}
    for s in [0, 1, 2]:
        c = int(np.sum(states == s))
        out[s] = (names[s], c, c / total if total > 0 else 0.0)
    return out


def save_transition_matrix_csv(P: np.ndarray, out_path: str):
    np.savetxt(out_path, P, delimiter=",", fmt="%.6f")


def save_results_summary(
    filepath: str,
    rpm_folder: str,
    qhat: float,
    mean_width: float,
    q1: float,
    q2: float,
    severity_stats,
    abstract_stats,
    P: np.ndarray,
    p_H: np.ndarray,
    acc_cal: float,
    windows_shape,
    scores_shape,
    states_shape,
    score_min: float,
    score_max: float,
):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"QEST SNN Results Summary - {rpm_folder}\n")
        f.write("=" * 55 + "\n\n")

        f.write(f"Calibration accuracy: {acc_cal:.4f}\n")
        f.write(f"Conformal score radius qhat: {qhat:.4f}\n")
        f.write(f"Mean conformal interval width: {mean_width:.4f}\n")
        f.write(f"State thresholds: q1={q1:.4f}, q2={q2:.4f}\n")
        f.write(f"Score range: min={score_min:.4f}, max={score_max:.4f}\n\n")

        f.write(f"Windows shape: {windows_shape}\n")
        f.write(f"SNN score shape: {scores_shape}\n")
        f.write(f"States shape: {states_shape}\n\n")

        f.write("4-Class Severity Distribution\n")
        for s in [0, 1, 2, 3]:
            name, c, p = severity_stats[s]
            f.write(f"  {name} ({s}): {c} ({p:.2%})\n")

        f.write("\nAbstracted 3-State Distribution\n")
        for s in [0, 1, 2]:
            name, c, p = abstract_stats[s]
            f.write(f"  {name} ({s}): {c} ({p:.2%})\n")

        f.write("\nTransition Matrix P\n")
        for row in P:
            f.write("  " + "  ".join(f"{x:.6f}" for x in row) + "\n")

        f.write("\nFinite-Horizon Reachability to State 2\n")
        for i, prob in enumerate(p_H):
            f.write(f"  From state {i}: {prob:.4f}\n")

def moving_average_edge(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()

    pad_left = window // 2
    pad_right = window - 1 - pad_left
    x_pad = np.pad(x, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x_pad, kernel, mode="valid")


def rolling_rms(signal: np.ndarray, window: int = 2000) -> np.ndarray:
    sq = signal.astype(float) ** 2
    mean_sq = moving_average_edge(sq, window=window)
    return np.sqrt(np.maximum(mean_sq, 0.0))


def normalize_feature(x: np.ndarray) -> np.ndarray:
    std = np.std(x)
    if std == 0:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def save_preprocessing_plot(
    data_path: str,
    rpm_folder: str,
    base_out_dir: str,
    rms_window: int = 2000,
):
    preprocessing_dir = os.path.join(base_out_dir, "preprocessing")
    os.makedirs(preprocessing_dir, exist_ok=True)

    normal_signal, mild_signal, medium_signal, severe_signal = load_severity_signals(
        data_path,
        rpm_folder=rpm_folder,
        fault_type="IR",
        severities=(7, 14, 21),
        end_tag="DE12",
    )

    signal, segment_length = combine_severity_signals(
        normal_signal, mild_signal, medium_signal, severe_signal
    )

    signal = normalize(signal)

    rms_feature = rolling_rms(signal, window=rms_window)
    rms_feature = normalize_feature(rms_feature)

    x = np.arange(len(rms_feature))
    x1 = segment_length
    x2 = 2 * segment_length
    x3 = 3 * segment_length

    plt.figure(figsize=(12, 4.6))
    plt.plot(x, rms_feature, linewidth=2.0, label="Rolling RMS feature")

    for xx in [x1, x2, x3]:
        plt.axvline(xx, color="black", linestyle="--", linewidth=1.2)

    ymin, ymax = float(np.min(rms_feature)), float(np.max(rms_feature))
    yr = ymax - ymin if ymax > ymin else 1.0
    ypos = ymax - 0.08 * yr

    plt.text(segment_length * 0.5, ypos, "Healthy", ha="center", va="center")
    plt.text(segment_length * 1.5, ypos, "Mild", ha="center", va="center")
    plt.text(segment_length * 2.5, ypos, "Medium", ha="center", va="center")
    plt.text(segment_length * 3.5, ypos, "Severe", ha="center", va="center")

    plt.title(f"CWRU Rolling-RMS Feature Progression - {rpm_folder}")
    plt.xlabel("Sample index")
    plt.ylabel("Normalized feature value")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.2, linestyle="--")
    plt.tight_layout()

    out_path = os.path.join(preprocessing_dir, "preprocessing_plot.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Created preprocessing plot:", out_path)


def plot_transition_matrix_heatmap(P: np.ndarray, out_path: str):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(P, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    labels = ["Healthy", "Degrading", "Critical"]
    plt.xticks([0, 1, 2], labels, rotation=20)
    plt.yticks([0, 1, 2], labels)

    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            val = P[i, j]
            color = "white" if val > 0.5 else "black"
            plt.text(j, i, f"{val:.3f}", ha="center", va="center", color=color)

    plt.title("CWRU DTMC Transition Matrix")
    plt.xlabel("Next state")
    plt.ylabel("Current state")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_reachability_barchart(p_H: np.ndarray, out_path: str):
    labels = ["Healthy", "Degrading", "Critical"]
    x = np.arange(len(p_H))

    plt.figure(figsize=(7, 4.5))
    plt.bar(x, p_H)
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Reachability probability")
    plt.title("CWRU Finite-Horizon Reachability to Critical State")

    for i, val in enumerate(p_H):
        plt.text(i, val + 0.02, f"{val:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_confusion_matrix_from_probs(
    y_true: np.ndarray,
    probs: np.ndarray,
    out_path: str,
):
    pred = np.argmax(probs, axis=1)
    cm = confusion_matrix(y_true, pred, labels=[0, 1, 2, 3]).astype(float)

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm / row_sums

    labels = ["Healthy", "Mild", "Medium", "Severe"]

    plt.figure(figsize=(6.8, 5.8))
    im = plt.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks([0, 1, 2, 3], labels, rotation=20)
    plt.yticks([0, 1, 2, 3], labels)

    for i in range(4):
        for j in range(4):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=color)

    plt.title("CWRU Calibration Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_score_by_class_boxplot(
    labels: np.ndarray,
    scores: np.ndarray,
    out_path: str,
):
    class_names = ["Healthy", "Mild", "Medium", "Severe"]
    data = [scores[labels == k] for k in range(4)]

    plt.figure(figsize=(8, 4.8))
    plt.boxplot(data, labels=class_names, showfliers=False)
    plt.ylabel("SNN severity score")
    plt.title("CWRU Score Distribution by True Severity Class")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_score_vs_true_class_scatter(
    labels: np.ndarray,
    scores: np.ndarray,
    out_path: str,
    max_points: int = 3000,
    random_seed: int = 42,
):
    rng = np.random.default_rng(random_seed)
    n = len(labels)

    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        labels_plot = labels[idx]
        scores_plot = scores[idx]
    else:
        labels_plot = labels
        scores_plot = scores

    jitter = rng.uniform(-0.08, 0.08, size=len(labels_plot))

    plt.figure(figsize=(7.5, 4.8))
    plt.scatter(labels_plot + jitter, scores_plot, s=10, alpha=0.35)
    plt.xticks([0, 1, 2, 3], ["Healthy", "Mild", "Medium", "Severe"])
    plt.ylabel("SNN severity score")
    plt.xlabel("True severity class")
    plt.title("CWRU Score vs True Severity Class")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_mean_score_per_segment(
    labels: np.ndarray,
    scores: np.ndarray,
    out_path: str,
):
    class_names = ["Healthy", "Mild", "Medium", "Severe"]
    means = [float(np.mean(scores[labels == k])) for k in range(4)]
    stds = [float(np.std(scores[labels == k])) for k in range(4)]

    x = np.arange(4)

    plt.figure(figsize=(7.5, 4.8))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, class_names)
    plt.ylabel("Mean SNN severity score")
    plt.title("CWRU Mean Score by Severity Segment")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_dtmc_graph(P: np.ndarray, out_path: str):
    labels = ["Healthy", "Degrading", "Critical"]

    pos = {
        0: (0.0, 0.0),
        1: (3.0, 0.0),
        2: (6.0, 0.0),
    }

    fig, ax = plt.subplots(figsize=(9.5, 3.8))

    # nodes
    for i, label in enumerate(labels):
        x, y = pos[i]
        circle = plt.Circle((x, y), 0.48, fill=False, linewidth=2.2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center", fontsize=12)

    def forward_arrow(i, j, y_offset=0.18):
        x1, y1 = pos[i]
        x2, y2 = pos[j]

        ax.annotate(
            "",
            xy=(x2 - 0.52, y2 + y_offset),
            xytext=(x1 + 0.52, y1 + y_offset),
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
        # arrow from i to j, usually right-to-left
        x1, y1 = pos[i]
        x2, y2 = pos[j]

        ax.annotate(
            "",
            xy=(x2 + 0.52, y2 + y_offset),
            xytext=(x1 - 0.52, y1 + y_offset),
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
            xy=(x + 0.33, y + 0.48),        # arrow head near top-right of node
            xytext=(x - 0.33, y + 0.48),    # start near top-left of node
            arrowprops=dict(
                arrowstyle="->",
                linewidth=1.8,
                connectionstyle="arc3,rad=-1.9",  # inverted loop above node
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

    # self loops
    for i in range(3):
        self_loop(i)

    # Healthy <-> Degrading
    forward_arrow(0, 1, y_offset=0.18)
    backward_arrow(1, 0, y_offset=-0.18)

    # Degrading <-> Critical
    forward_arrow(1, 2, y_offset=0.18)
    backward_arrow(2, 1, y_offset=-0.18)

    ax.set_xlim(-0.9, 6.9)
    ax.set_ylim(-1.05, 1.55)
    ax.axis("off")
    ax.set_title("Learned 3-State DTMC", fontsize=15)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def run_case_and_save(
    data_path: str,
    rpm_folder: str,
    base_out_dir: str,
    horizon: int = 10,
    alpha: float = 0.1,
):
    safe_name = rpm_folder.replace(" ", "_")
    out_dir = os.path.join(base_out_dir, safe_name)
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"RUNNING RESULTS FOR: {rpm_folder}")
    print("=" * 60)

    normal_signal, mild_signal, medium_signal, severe_signal = load_severity_signals(
        data_path,
        rpm_folder=rpm_folder,
        fault_type="IR",
        severities=(7, 14, 21),
        end_tag="DE12",
    )

    

    signal, segment_length = combine_severity_signals(
        normal_signal, mild_signal, medium_signal, severe_signal
    )
    signal = normalize(signal)

    windows, window_starts = create_windows_with_positions(signal, window_size=200, stride=100)
    windows = normalize_windows(windows)
    labels = create_severity_labels(window_starts, segment_length)

    result = train_snn_and_get_scores(
        windows,
        labels,
        hidden_size=64,
        epochs=35,
        lr=1e-3,
        batch_size=64,
        random_seed=42,
    )

    s_t = result["score_all"]
    y_cal = result["y_cal"]
    score_cal = result["score_cal"]
    probs_cal = result["probs_cal"]

    pred_cal = np.argmax(probs_cal, axis=1)
    acc_cal = float(np.mean(pred_cal == y_cal))

    qhat = split_conformal_score_interval(y_cal, score_cal, alpha=alpha)
    score_lo, score_hi = apply_score_interval(s_t, qhat, min_score=0.0, max_score=3.0)
    mean_width = float(np.mean(score_hi - score_lo))

    states, q1, q2 = discretize_states(s_t)
    counts, P = estimate_transition_matrix(states, n_states=3)
    plot_dtmc_graph(
    P=P,
    out_path=os.path.join(out_dir, "dtmc_graph.png"),
)
    p_H = finite_horizon_reachability(P, target_state=2, horizon=horizon)
    case_name = f"cwru_{safe_name}"
    save_storm_models(P, out_dir=out_dir, case_name=case_name)
    #save_storm_models(P, out_dir=out_dir, horizon=horizon)
    case_name = f"cwru_{safe_name}"
    save_storm_models(P, out_dir=out_dir, case_name=case_name)

    sev_stats = severity_distribution(labels)
    abs_stats = state_distribution(states)

    # segment boundaries in sample index
    x1 = segment_length
    x2 = 2 * segment_length
    x3 = 3 * segment_length

    # segment boundaries in window index
    w1 = np.sum(window_starts < segment_length)
    w2 = np.sum(window_starts < 2 * segment_length)
    w3 = np.sum(window_starts < 3 * segment_length)

    # Figure 1: full severity progression signal
    plt.figure(figsize=(12, 4))
    plt.plot(signal, linewidth=0.7)
    for x in [x1, x2, x3]:
        plt.axvline(x=x, color="black", linestyle="--", linewidth=1)
    ymax = np.max(signal)
    ymin = np.min(signal)
    ypos = ymax - 0.08 * (ymax - ymin)
    plt.text(segment_length * 0.5, ypos, "Healthy", ha="center")
    plt.text(segment_length * 1.5, ypos, "Mild", ha="center")
    plt.text(segment_length * 2.5, ypos, "Medium", ha="center")
    plt.text(segment_length * 3.5, ypos, "Severe", ha="center")
    plt.title(f"CWRU Severity Progression Signal - {rpm_folder}")
    plt.xlabel("Sample index")
    plt.ylabel("Normalized amplitude")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "severity_signal_plot.png"), dpi=300)
    plt.close()

    # Figure 2: SNN score
    plt.figure(figsize=(12, 4))
    plt.plot(s_t, linewidth=1.0, label="SNN severity score")
    for x in [w1, w2, w3]:
        plt.axvline(x=x, color="black", linestyle="--", linewidth=1)
    plt.title(f"SNN Severity Score - {rpm_folder}")
    plt.xlabel("Window index")
    plt.ylabel("Severity score")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "snn_severity_score_plot.png"), dpi=300)
    plt.close()

    # Figure 3: conformal interval
    x = np.arange(len(s_t))
    plt.figure(figsize=(12, 4))
    plt.plot(x, s_t, linewidth=1.0, label="SNN severity score")
    plt.fill_between(x, score_lo, score_hi, alpha=0.25, label="Conformal interval")
    for xx in [w1, w2, w3]:
        plt.axvline(x=xx, color="black", linestyle="--", linewidth=1)
    plt.title(f"SNN Severity Score with Conformal Interval - {rpm_folder}")
    plt.xlabel("Window index")
    plt.ylabel("Severity score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "score_interval_plot.png"), dpi=300)
    plt.close()

    # Figure 4: abstracted states
    plt.figure(figsize=(12, 4))
    plt.plot(states, drawstyle="steps-post", linewidth=1.0)
    for xx in [w1, w2, w3]:
        plt.axvline(x=xx, color="black", linestyle="--", linewidth=1)
    plt.title(f"Abstracted 3-State Trajectory - {rpm_folder}")
    plt.xlabel("Window index")
    plt.ylabel("State")
    plt.yticks([0, 1, 2])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "states_plot.png"), dpi=300)
    plt.close()

    # New stronger plots
    plot_transition_matrix_heatmap(
        P=P,
        out_path=os.path.join(out_dir, "transition_matrix_heatmap.png"),
    )

    plot_reachability_barchart(
        p_H=p_H,
        out_path=os.path.join(out_dir, "reachability_barplot.png"),
    )

    plot_confusion_matrix_from_probs(
        y_true=y_cal,
        probs=probs_cal,
        out_path=os.path.join(out_dir, "confusion_matrix.png"),
    )
    ''' 
    plot_score_by_class_boxplot(
        labels=labels,
        scores=s_t,
        out_path=os.path.join(out_dir, "score_by_class_boxplot.png"),
    )
    
    #More analysis
    plot_score_vs_true_class_scatter(
        labels=labels,
        scores=s_t,
        out_path=os.path.join(out_dir, "score_vs_true_class_scatter.png"),
        max_points=3000,
        random_seed=42,
    )
    #More analysis
    plot_mean_score_per_segment(
        labels=labels,
        scores=s_t,
        out_path=os.path.join(out_dir, "mean_score_per_segment.png"),
    )
    '''
    save_transition_matrix_csv(P, os.path.join(out_dir, "transition_matrix.csv"))

    save_results_summary(
        filepath=os.path.join(out_dir, "results_summary.txt"),
        rpm_folder=rpm_folder,
        qhat=qhat,
        mean_width=mean_width,
        q1=q1,
        q2=q2,
        severity_stats=sev_stats,
        abstract_stats=abs_stats,
        P=P,
        p_H=p_H,
        acc_cal=acc_cal,
        windows_shape=windows.shape,
        scores_shape=s_t.shape,
        states_shape=states.shape,
        score_min=float(np.min(s_t)),
        score_max=float(np.max(s_t)),
    )

    return {
        "rpm_folder": rpm_folder,
        "qhat": qhat,
        "mean_width": mean_width,
        "acc_cal": acc_cal,
        "q1": q1,
        "q2": q2,
        "p_H": p_H,
        "severity_stats": sev_stats,
        "abstract_stats": abs_stats,
        "P": P,
        "counts": counts,
        "windows_shape": windows.shape,
        "scores_shape": s_t.shape,
        "states_shape": states.shape,
        "score_min": float(np.min(s_t)),
        "score_max": float(np.max(s_t)),
    }


def save_comparison_summary(filepath: str, case_results: list[dict]):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("Comparison Summary Across Case Studies\n")
        f.write("=" * 45 + "\n\n")

        for res in case_results:
            f.write(f"{res['rpm_folder']}\n")
            f.write("-" * len(res["rpm_folder"]) + "\n")
            f.write(f"Calibration accuracy: {res['acc_cal']:.4f}\n")
            f.write(f"Conformal qhat: {res['qhat']:.4f}\n")
            f.write(f"Mean interval width: {res['mean_width']:.4f}\n")
            f.write(f"Thresholds: q1={res['q1']:.4f}, q2={res['q2']:.4f}\n")
            f.write(f"Score range: min={res['score_min']:.4f}, max={res['score_max']:.4f}\n")
            f.write(
                "Reachability: "
                + ", ".join(f"state {i} -> {p:.4f}" for i, p in enumerate(res["p_H"]))
                + "\n"
            )

            f.write("\n4-Class Severity Distribution\n")
            for s in [0, 1, 2, 3]:
                name, c, p = res["severity_stats"][s]
                f.write(f"  {name} ({s}): {c} ({p:.2%})\n")

            f.write("\nAbstracted 3-State Distribution\n")
            for s in [0, 1, 2]:
                name, c, p = res["abstract_stats"][s]
                f.write(f"  {name} ({s}): {c} ({p:.2%})\n")

            f.write("\nTransition Matrix P\n")
            for row in res["P"]:
                f.write("  " + "  ".join(f"{x:.6f}" for x in row) + "\n")

            f.write("\n" + "=" * 45 + "\n\n")


def export_dtmc_to_prism(P: np.ndarray, init_state: int, out_path: str) -> None:
    n_states = P.shape[0]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("dtmc\n\n")
        f.write("module health\n")
        f.write(f"    s : [0..{n_states - 1}] init {init_state};\n\n")

        for i in range(n_states):
            terms = []
            for j in range(n_states):
                terms.append(f"{P[i, j]:.10f} : (s'={j})")
            f.write(f"    [] s={i} -> " + " + ".join(terms) + ";\n")

        f.write("endmodule\n\n")
        f.write(f'label "critical" = s={n_states - 1};\n')


def save_storm_models(P: np.ndarray, out_dir: str, case_name: str, horizon_list=None) -> None:
    if horizon_list is None:
        horizon_list = [1, 2, 5, 10, 15, 20, 25]

    storm_dir = os.path.join(out_dir, "storm_models")
    os.makedirs(storm_dir, exist_ok=True)

    np.savetxt(
        os.path.join(storm_dir, f"{case_name}_transition_matrix.csv"),
        P,
        delimiter=",",
        fmt="%.10f",
    )

    for init_state in [0, 1, 2]:
        export_dtmc_to_prism(
            P=P,
            init_state=init_state,
            out_path=os.path.join(storm_dir, f"{case_name}_init{init_state}.pm"),
        )

    with open(os.path.join(storm_dir, f"{case_name}_storm_commands.txt"), "w", encoding="utf-8") as f:
        for init_state in [0, 1]:  # only meaningful initial states
            for h in horizon_list:
                f.write(
                    f'storm --prism {case_name}_init{init_state}.pm '
                    f'--prop \'P=? [F<={h} "critical"]\'\n'
                )
            f.write("\n")

def main():
    base_out_dir = "results_cwru"
    os.makedirs(base_out_dir, exist_ok=True)

    data_path = "data/data_cwru"
    rpm_folders = ["1797 RPM", "1730 RPM"]

    save_preprocessing_plot(
    data_path=data_path,
    rpm_folder="1730 RPM",
    base_out_dir=base_out_dir,
)

    all_results = []

    for rpm_folder in rpm_folders:
        res = run_case_and_save(
            data_path=data_path,
            rpm_folder=rpm_folder,
            base_out_dir=base_out_dir,
            #horizon=10,
            alpha=0.1,
        )
        all_results.append(res)

    save_comparison_summary(
        filepath=os.path.join(base_out_dir, "comparison_summary.txt"),
        case_results=all_results,
    )

    print("\nSaved all results under:", base_out_dir)
    print("Created comparison summary:", os.path.join(base_out_dir, "comparison_summary.txt"))
    print("Please Note that you can find all detailed figures in the related folders,However, here is the figures addressed in the paper: ")
    print("Figure 3b: results_cwru \\ 1730_RPM \\ severity_signal_plot.jpg ")
    print("Figure 6a: results_cwru \\ preprocessing \\ preprocessing_plot.jpg")
    print("Figure 6b: results_cwru \\ 1730_RPM \\ score_interval_plot.jpg ")
    print("Figure 7a: results_cwru \\ 1730_RPM \\ dtmc_graph.png ")




if __name__ == "__main__":
    main()