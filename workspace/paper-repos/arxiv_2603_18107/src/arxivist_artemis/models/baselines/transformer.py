"""
models/baselines/transformer.py
==================================
Vanilla Transformer baseline (Section 5.2): encoder-only, adapted for
single-step forecasting. Input projection to d_model=128, positional
encoding, 3 encoder layers (8 heads, FFN dim 256, dropout 0.1), final
timestep's representation -> linear head.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


class TransformerBaseline(nn.Module):
    """
    Args:
        d_x: input feature dimension.
        d_model: explicitly stated in paper: 128.
        n_heads: explicitly stated in paper: 8.
        n_layers: explicitly stated in paper: 3.
        ffn_dim: explicitly stated in paper: 256.
        dropout: explicitly stated in paper: 0.1.
    """

    def __init__(
        self, d_x: int, d_model: int = 128, n_heads: int = 8, n_layers: int = 3,
        ffn_dim: int = 256, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(d_x, d_model)
        self.pos_encoding = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_x], mask: [B, L, d_x] (1=observed, 0=missing).
        Returns:
            [B] scalar predictions.
        """
        x_masked = x * mask
        h = self.input_proj(x_masked)
        h = self.pos_encoding(h)
        h = self.encoder(h)
        final_step = h[:, -1, :]  # last time step's representation
        return self.head(final_step).squeeze(-1)
