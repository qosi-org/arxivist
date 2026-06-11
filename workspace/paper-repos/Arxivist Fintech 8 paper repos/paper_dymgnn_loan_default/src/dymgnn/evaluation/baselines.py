"""
evaluation/baselines.py — Baseline Models (Section 5.1).

Implements all baseline models benchmarked in the paper:
  GNN-based:     Static GCN, Static GAT (Table 3)
  Non-GNN-based: Logistic Regression, XGBoost, DNN (Table 4)

For GNN baselines: uses last snapshot of each window; behavioural features
averaged across 6 snapshots (Section 5.1).

Paper: Zandi et al. (EJOR 2025). DOI: 10.1016/j.ejor.2024.09.025
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray


class StaticGNNBaseline(nn.Module):
    """Static GCN or GAT baseline (Section 5.1, Table 3).

    Uses only the last snapshot (no temporal dynamics).
    Behavioural feature values are averaged over 6 snapshots.

    Args:
        gnn_type: 'GCN' or 'GAT'.
        num_features: Node feature dimension d.
        embedding_dim: GNN embedding dimension D.
        num_gat_heads: GAT heads (if gnn_type='GAT').
        decoder_hidden1: First decoder layer size (20, Fig. 5).
        decoder_hidden2: Second decoder layer size (10, Fig. 5).
    """

    def __init__(
        self,
        gnn_type: str,
        num_features: int,
        embedding_dim: int,
        num_gat_heads: int = 4,
        decoder_hidden1: int = 20,
        decoder_hidden2: int = 10,
    ) -> None:
        super().__init__()
        from dymgnn.models.gcn_layer import GCNLayer
        from dymgnn.models.gat_layer import GATLayer
        from dymgnn.models.decoder import Decoder

        if gnn_type == "GCN":
            self.gnn = GCNLayer(num_features, embedding_dim)
        else:
            self.gnn = GATLayer(num_features, embedding_dim, num_heads=num_gat_heads)

        self.decoder = Decoder(embedding_dim, decoder_hidden1, decoder_hidden2)
        self.gnn_type = gnn_type

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Static GNN forward on one snapshot."""
        z = self.gnn(x, adj)
        return self.decoder(z)

    def __repr__(self) -> str:
        return f"StaticGNNBaseline({self.gnn_type})"


def train_lr_baseline(
    X_train: NDArray, y_train: NDArray, cfg: dict[str, Any]
) -> Any:
    """Train Logistic Regression baseline (Section 5.1, Table 4).

    Uses saga solver with L1/L2 penalty grid search on validation set.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    param_grid = {"penalty": ["l1", "l2"], "C": [0.01, 0.1, 1.0, 10.0]}
    lr = GridSearchCV(
        LogisticRegression(solver="saga", max_iter=1000, random_state=42),
        param_grid, cv=3, scoring="roc_auc", n_jobs=-1
    )
    lr.fit(X_train, y_train)
    return lr.best_estimator_


def train_xgb_baseline(
    X_train: NDArray, y_train: NDArray, cfg: dict[str, Any]
) -> Any:
    """Train XGBoost baseline (Section 5.1, Table 4).

    Grid search over: lr {0.001,0.01,0.1}, max_depth {2,3,4},
    n_estimators {50,100,250,500}, alpha {0.1,...,0.9}.
    """
    try:
        from xgboost import XGBClassifier
        from sklearn.model_selection import GridSearchCV
    except ImportError:
        raise ImportError("xgboost required: pip install xgboost")

    param_grid = {
        "learning_rate": [0.001, 0.01, 0.1],
        "max_depth": [2, 3, 4],
        "n_estimators": [50, 100, 250, 500],
        "reg_alpha": [0.1, 0.3, 0.5, 0.7, 0.9],
    }
    # Reduced grid for speed — full search as in paper
    fast_grid = {
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 4],
        "n_estimators": [100, 250],
        "reg_alpha": [0.1, 0.5, 0.9],
    }
    xgb = GridSearchCV(
        XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        fast_grid, cv=3, scoring="roc_auc", n_jobs=1
    )
    xgb.fit(X_train, y_train)
    return xgb.best_estimator_


class DNNBaseline(nn.Module):
    """Deep Neural Network baseline (Section 5.1, Table 4, Fig. B.1).

    Architecture from Fig. B.1:
        Dense(16→30) → ReLU → Dropout(0.5)
        → Dense(30→50) → ReLU → Dropout(0.5)
        → Dense(50→20) → ReLU → Dropout(0.5)
        → Dense(20→1) → Sigmoid
    """

    def __init__(self, num_features: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 30),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(30, 50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def __repr__(self) -> str:
        return "DNNBaseline(16→30→50→20→1)"
