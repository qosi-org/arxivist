"""
models/baselines/lstm.py
==========================
LSTM baseline (Section 5.1): 2-layer stacked LSTM, 128 hidden units,
dropout 0.2 between layers, linear output head. Missing values handled via
element-wise mask multiplication before the LSTM (Section 5.1: "the input
tensor is multiplied by the mask before being fed to the LSTM").
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """
    Args:
        d_x: input feature dimension (dataset-dependent).
        hidden_dim: LSTM hidden size. Explicitly stated in paper: 128.
        num_layers: explicitly stated in paper: 2.
        dropout: explicitly stated in paper: 0.2.
    """

    def __init__(self, d_x: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_x,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_x] input window.
            mask: [B, L, d_x] binary mask (1 = observed, 0 = missing).
        Returns:
            [B] scalar predictions.
        """
        x_masked = x * mask
        out, (h_n, _) = self.lstm(x_masked)
        final_hidden = h_n[-1]  # [B, hidden_dim], last layer's final hidden state
        return self.head(final_hidden).squeeze(-1)
