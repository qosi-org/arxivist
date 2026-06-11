"""
models/gat_layer.py — Graph Attention Network layer.

Implements the GAT formulation from Section 3.2, Equations 2–4:
    e_ij = LeakyReLU(a^T [WX_i ‖ WX_j])              (Eq. 2)
    α_ij = softmax over N(v_i) ∪ {v_i} of exp(e_ij)   (Eq. 3)
    Z_i  = Σ_{j ∈ N(v_i)∪{v_i}} α_ij W X_j           (Eq. 4)

Multi-head output is aggregated by averaging (Section 3.2).

Paper: Zandi et al. (EJOR 2025). DOI: 10.1016/j.ejor.2024.09.025
Reference: Veličković et al. (2018) "Graph Attention Networks"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GATLayer(nn.Module):
    """Multi-head Graph Attention Network layer (Section 3.2, Eq. 2–4).

    Assigns different importance to each neighbour via learned attention
    coefficients. Multi-head outputs are averaged.

    Args:
        in_features: Input feature dimension d.
        out_features: Output embedding dimension D (per head, then averaged to D).
        num_heads: Number of attention heads H.
            # ASSUMED: 4 — not stated in paper (IA-02). Common: 4, 8.
        leaky_relu_alpha: Negative slope for LeakyReLU (default 0.2, standard in GAT).
            # ASSUMED: 0.2 — standard GAT default.
        dropout: Attention dropout probability (default 0.0).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,         # ASSUMED — see IA-02
        leaky_relu_alpha: float = 0.2,  # ASSUMED — standard GAT default
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.leaky_relu_alpha = leaky_relu_alpha

        # W ∈ R^{D×d} — shared feature transformation for all heads
        # Each head has its own W and attention vector a
        self.heads = nn.ModuleList([
            _GATHead(in_features, out_features, leaky_relu_alpha, dropout)
            for _ in range(num_heads)
        ])

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """Apply multi-head graph attention to one snapshot.

        Implements Eq. 2–4. Multi-head outputs averaged (Section 3.2).

        Args:
            x: Node feature matrix [nl, d].
            adj: Adjacency matrix [nl, nl] (0/1, self-loops will be added).

        Returns:
            Z: Node embedding matrix [nl, D].
        """
        assert x.dim() == 2, f"Expected [nl, d], got {x.shape}"
        assert adj.dim() == 2 and adj.shape[0] == adj.shape[1] == x.shape[0], (
            f"adj shape {adj.shape} incompatible with x.shape[0]={x.shape[0]}"
        )

        # Add self-loops (each node attends to itself, Section 3.2)
        nl = x.shape[0]
        adj_with_self = adj + torch.eye(nl, device=adj.device, dtype=adj.dtype)
        adj_with_self = (adj_with_self > 0).float()  # ensure 0/1

        # Compute each head, then average (Section 3.2 multi-head averaging)
        head_outputs = [head(x, adj_with_self) for head in self.heads]  # H × [nl, D]
        out = torch.stack(head_outputs, dim=0).mean(dim=0)               # [nl, D]
        return out

    def __repr__(self) -> str:
        return (
            f"GATLayer(in={self.in_features}, out={self.out_features}, "
            f"heads={self.num_heads})"
        )


class _GATHead(nn.Module):
    """Single attention head for GAT (Section 3.2, Eq. 2–4)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        leaky_relu_alpha: float = 0.2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # W ∈ R^{D×d}
        self.W = nn.Linear(in_features, out_features, bias=False)
        # a^T ∈ R^{1×2D} — attention weight vector (Eq. 2)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_alpha)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """Single-head GAT forward pass implementing Eq. 2–4."""
        nl = x.shape[0]

        # Feature transformation: WX ∈ R^{nl×D}
        Wx = self.W(x)  # [nl, D]

        # Compute pairwise attention scores (Eq. 2): e_ij = LeakyReLU(a^T [WX_i ‖ WX_j])
        # Broadcast to build all pairs efficiently
        Wx_i = Wx.unsqueeze(1).expand(nl, nl, -1)  # [nl, nl, D]
        Wx_j = Wx.unsqueeze(0).expand(nl, nl, -1)  # [nl, nl, D]
        pair = torch.cat([Wx_i, Wx_j], dim=-1)      # [nl, nl, 2D]
        e = self.leaky_relu(self.a(pair).squeeze(-1))  # [nl, nl]

        # Mask non-edges with -inf before softmax (Eq. 3)
        mask = (adj == 0)
        e = e.masked_fill(mask, float("-inf"))

        # Normalize attention scores (Eq. 3)
        alpha = F.softmax(e, dim=1)         # [nl, nl]
        alpha = self.dropout(alpha)

        # Replace NaN from all-masked rows (isolated nodes)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        # Aggregate neighbour features weighted by attention (Eq. 4)
        out = alpha @ Wx  # [nl, D]
        return out
