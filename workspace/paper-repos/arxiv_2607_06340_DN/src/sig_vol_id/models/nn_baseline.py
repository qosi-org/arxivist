"""
Fully-connected neural network baseline (Section 6.7 robustness check).

ASSUMED hyperparameters (SIR ambiguities[3], confidence 0.4): dropout rate,
batch size, and Adam learning rate are not given numerically in the paper;
literature-standard defaults are used (see configs/config.yaml
`nn_baseline`).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SignatureMLP(nn.Module):
    """3-hidden-layer MLP (128-64-32) matching the paper's described architecture.

    Args:
        input_dim: Number of signature features (d_N).
        n_classes: Number of output classes.
        dropout: Dropout probability (ASSUMED, paper does not specify).
    """

    def __init__(self, input_dim: int, n_classes: int, dropout: float = 0.2):
        super().__init__()
        layers = []
        dims = [input_dim, 128, 64, 32]
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(dims[-1], n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    model: SignatureMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
    device: str = "cpu",
) -> SignatureMLP:
    """Train with Adam + categorical cross-entropy, early stopping on val loss.

    Standardizes inputs internally is NOT done here -- caller should pass
    already-standardized features (mean 0, std 1), per the paper's description.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)

    n = X_train_t.shape[0]
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            out = model(X_train_t[idx])
            loss = criterion(out, y_train_t[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_loss = criterion(val_out, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
