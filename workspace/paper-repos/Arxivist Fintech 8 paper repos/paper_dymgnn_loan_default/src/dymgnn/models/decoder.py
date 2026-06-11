"""
models/decoder.py — Feed-Forward Decoder for Node Classification.

Implements the decoder architecture from Figure 5 (Section 3.6):
    Dense(D → 20) → ReLU → Dropout(0.5) → Dense(20 → 10) → ReLU →
    Dropout(0.5) → Dense(10 → 1) → Sigmoid

Output is the probability of loan default for each node.

Paper: Zandi et al. (EJOR 2025). DOI: 10.1016/j.ejor.2024.09.025
Section 3.6, Figure 5. Architecture explicitly shown in Fig. 5.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class Decoder(nn.Module):
    """Feed-forward decoder for binary node classification (Fig. 5).

    Takes node embeddings H_att (or H^(τ) without attention) and predicts
    the probability of default for each node.

    Decoder architecture (Fig. 5 — explicitly stated):
        Dense(in_dim → hidden1=20) → ReLU → Dropout(p=0.5)
        → Dense(20 → hidden2=10) → ReLU → Dropout(p=0.5)
        → Dense(10 → 1) → Sigmoid

    Args:
        in_dim: Input dimension D (GNN/RNN embedding size).
        hidden1: First hidden layer size. Default 20 (from Fig. 5).
        hidden2: Second hidden layer size. Default 10 (from Fig. 5).
        dropout: Dropout probability. Default 0.5 (from Fig. 5).
    """

    def __init__(
        self,
        in_dim: int,
        hidden1: int = 20,       # [HIGH] Fig. 5: Dense Layer (20, 10)
        hidden2: int = 10,       # [HIGH] Fig. 5: Dense Layer (10, 1)
        dropout: float = 0.5,   # [HIGH] Fig. 5: Dropout (p=0.5)
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2

        # Fig. 5 architecture
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )

    def forward(self, h: Tensor) -> Tensor:
        """Map node embeddings to default probability.

        Args:
            h: Node embedding matrix [nl, D] (H_att or H^(τ)).

        Returns:
            Y_hat: Default probability for each node [nl, 1].
        """
        assert h.dim() == 2, f"Expected [nl, D], got {h.shape}"
        assert h.shape[1] == self.in_dim, (
            f"in_dim mismatch: expected {self.in_dim}, got {h.shape[1]}"
        )
        return self.net(h)  # [nl, 1]

    def __repr__(self) -> str:
        return (
            f"Decoder(in_dim={self.in_dim}, "
            f"hidden=[{self.hidden1},{self.hidden2}], "
            f"out=1)"
        )
