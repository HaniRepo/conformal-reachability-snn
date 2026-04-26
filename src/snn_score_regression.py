import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.snn_model import TinySNN


def train_snn_and_get_scores_regression(
    windows: np.ndarray,
    targets: np.ndarray,
    hidden_size: int = 64,
    epochs: int = 25,
    lr: float = 1e-3,
    batch_size: int = 64,
    random_seed: int = 42,
):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    X = windows.astype(np.float32)
    y = targets.astype(np.float32)

    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, test_size=0.3, random_state=random_seed
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    X_cal_t = torch.tensor(X_cal, dtype=torch.float32)
    y_cal_t = torch.tensor(y_cal, dtype=torch.float32).unsqueeze(1)

    model = TinySNN(input_size=X.shape[1], hidden_size=hidden_size, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

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
            preds = torch.sigmoid(logits)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, loss={epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        logits_all = model(torch.tensor(X, dtype=torch.float32))
        score_all = torch.sigmoid(logits_all).squeeze(1).cpu().numpy()

        logits_cal = model(X_cal_t)
        score_cal = torch.sigmoid(logits_cal).squeeze(1).cpu().numpy()

    return {
        "model": model,
        "score_all": score_all,
        "X_cal": X_cal,
        "y_cal": y_cal,
        "score_cal": score_cal,
    }