import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.snn_model import TinySNN


def make_nasa_train_cal_split(
    labels: np.ndarray,
    test_size: float = 0.3,
    random_seed: int = 42,
):
    """
    Create one fixed train/calibration split to reuse across multiple sensor branches.
    """
    y = labels.astype(np.int64)
    all_idx = np.arange(len(y))

    train_idx, cal_idx = train_test_split(
        all_idx,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    return {
        "train_idx": np.asarray(train_idx, dtype=np.int64),
        "cal_idx": np.asarray(cal_idx, dtype=np.int64),
    }


def train_snn_and_get_nasa_scores_fixed_split(
    windows: np.ndarray,
    labels: np.ndarray,
    split_indices: dict,
    hidden_size: int = 64,
    epochs: int = 25,
    lr: float = 1e-3,
    batch_size: int = 64,
    random_seed: int = 42,
    verbose: bool = True,
):
    """
    Train a TinySNN using a fixed train/calibration split.
    This is useful when multiple sensor branches must share the same calibration set.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    X = windows.astype(np.float32)
    y = labels.astype(np.int64)

    train_idx = np.asarray(split_indices["train_idx"], dtype=np.int64)
    cal_idx = np.asarray(split_indices["cal_idx"], dtype=np.int64)

    if np.max(train_idx) >= len(X) or np.max(cal_idx) >= len(X):
        raise ValueError("Split indices exceed dataset size.")

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_cal = X[cal_idx]
    y_cal = y[cal_idx]

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_cal_t = torch.tensor(X_cal)

    model = TinySNN(input_size=X.shape[1], hidden_size=hidden_size, output_size=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    n_train = X_train_t.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n_train)
        epoch_loss = 0.0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if verbose:
            num_batches = int(np.ceil(n_train / batch_size))
            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs}, avg_loss={avg_epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        logits_all = model(torch.tensor(X))
        probs_all = torch.softmax(logits_all, dim=1).cpu().numpy()

        logits_cal = model(X_cal_t)
        probs_cal = torch.softmax(logits_cal, dim=1).cpu().numpy()

    severity_levels = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    score_all = probs_all @ severity_levels
    score_cal = probs_cal @ severity_levels

    return {
        "model": model,
        "score_all": score_all,
        "X_cal": X_cal,
        "y_cal": y_cal,
        "score_cal": score_cal,
        "probs_all": probs_all,
        "probs_cal": probs_cal,
        "train_idx": train_idx,
        "cal_idx": cal_idx,
    }


def train_two_branch_late_fusion(
    windows_1: np.ndarray,
    windows_2: np.ndarray,
    labels: np.ndarray,
    hidden_size: int = 64,
    epochs: int = 25,
    lr: float = 1e-3,
    batch_size: int = 64,
    random_seed: int = 42,
    test_size: float = 0.3,
    w1: float = 0.5,
    w2: float = 0.5,
    verbose: bool = True,
):
    """
    Train two independent SNN branches with a shared fixed split, then fuse scores.

    Fusion:
        fused_score = w1 * score_1 + w2 * score_2
    """
    if abs((w1 + w2) - 1.0) > 1e-8:
        raise ValueError(f"Fusion weights must sum to 1. Got w1+w2={w1+w2}")

    X1 = windows_1.astype(np.float32)
    X2 = windows_2.astype(np.float32)
    y = labels.astype(np.int64)

    if len(X1) != len(X2) or len(X1) != len(y):
        raise ValueError(
            f"Input length mismatch: len(X1)={len(X1)}, len(X2)={len(X2)}, len(y)={len(y)}"
        )

    split_indices = make_nasa_train_cal_split(
        labels=y,
        test_size=test_size,
        random_seed=random_seed,
    )

    if verbose:
        print("\nTraining fusion branch 1 ...")
    result_1 = train_snn_and_get_nasa_scores_fixed_split(
        windows=X1,
        labels=y,
        split_indices=split_indices,
        hidden_size=hidden_size,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        random_seed=random_seed,
        verbose=verbose,
    )

    if verbose:
        print("\nTraining fusion branch 2 ...")
    result_2 = train_snn_and_get_nasa_scores_fixed_split(
        windows=X2,
        labels=y,
        split_indices=split_indices,
        hidden_size=hidden_size,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        random_seed=random_seed + 1,
        verbose=verbose,
    )

    if not np.array_equal(result_1["y_cal"], result_2["y_cal"]):
        raise ValueError("Calibration labels are not aligned across the two branches.")

    score_all_fused = w1 * result_1["score_all"] + w2 * result_2["score_all"]
    score_cal_fused = w1 * result_1["score_cal"] + w2 * result_2["score_cal"]

    return {
        "split_indices": split_indices,
        "branch_1": result_1,
        "branch_2": result_2,
        "score_all_fused": score_all_fused,
        "score_cal_fused": score_cal_fused,
        "y_cal": result_1["y_cal"],
        "w1": w1,
        "w2": w2,
    }