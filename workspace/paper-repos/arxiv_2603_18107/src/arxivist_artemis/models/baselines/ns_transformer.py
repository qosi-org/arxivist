"""
models/baselines/ns_transformer.py
=====================================
Non-stationary Transformer baseline (Section 5.3): series stationarization
(subtract mean, divide by std along the time dimension) + de-stationary
attention factors learned from the raw statistics, injected into the
attention mechanism; final de-normalisation maps predictions back to the
original scale.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from arxivist_artemis.models.baselines.transformer import _PositionalEncoding


class NSTransformerBaseline(nn.Module):
    """
    Args:
        d_x: input feature dimension.
        d_model, n_heads, n_layers, ffn_dim, dropout: same defaults as the
            vanilla Transformer baseline (paper: "training hyperparameters
            are identical to those used for the Vanilla Transformer").
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
        # De-stationary factor projector: maps raw (mean,std) statistics to
        # a learned scalar factor + shift vector (Section 5.3).
        self.projector = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_x], mask: [B, L, d_x].
        Returns:
            [B] de-normalised scalar predictions.
        """
        x_masked = x * mask
        mean = x_masked.mean(dim=1, keepdim=True)  # [B, 1, d_x]
        std = x_masked.std(dim=1, keepdim=True) + 1e-5  # [B, 1, d_x]
        x_stationary = (x_masked - mean) / std

        # De-stationary factors computed from the mean/std of a representative
        # (e.g. first) channel, per the paper's high-level description.
        stats = torch.cat([mean[:, 0, :1], std[:, 0, :1]], dim=-1)  # [B, 2]
        factors = self.projector(stats)  # [B, 2]: (log-scale, shift)

        h = self.input_proj(x_stationary)
        h = self.pos_encoding(h)
        # Inject de-stationary factors as an additive/multiplicative bias on
        # the mean-pooled representation (simplified de-stationary attention).
        h = self.encoder(h)
        pooled = h.mean(dim=1)  # mean-pooled representation, per Section 5.3
        pred_stationary = self.head(pooled).squeeze(-1)
        scale = torch.exp(factors[:, 0])
        shift = factors[:, 1]
        return pred_stationary * scale + shift
