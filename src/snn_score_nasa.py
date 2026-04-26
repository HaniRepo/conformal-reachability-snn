import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.snn_model import TinySNN


def train_snn_and_get_nasa_scores(
    windows: np.ndarray,
    labels: np.ndarray,
    hidden_size: int = 64,
    epochs: int = 25,
    lr: float = 1e-3,
    batch_size: int = 64,
    random_seed: int = 42,
):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    X = windows.astype(np.float32)
    y = labels.astype(np.int64)

    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, test_size=0.3, random_state=random_seed, stratify=y
    )

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_cal_t = torch.tensor(X_cal)
    y_cal_t = torch.tensor(y_cal)

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
        #commented to see better learning status
        #print(f"Epoch {epoch+1}/{epochs}, loss={epoch_loss:.4f}")

        num_batches = int(np.ceil(n_train / batch_size))
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, avg_loss={avg_epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        logits_all = model(torch.tensor(X))
        probs_all = torch.softmax(logits_all, dim=1).cpu().numpy()

        logits_cal = model(X_cal_t)
        probs_cal = torch.softmax(logits_cal, dim=1).cpu().numpy()

    # map 3 classes to scalar score in [0,1]
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
    }