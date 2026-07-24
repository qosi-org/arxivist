"""
models/baselines/informer.py
===============================
Informer baseline (Section 5.4): encoder-only with a simplified ProbSparse
attention mechanism (selects top-scoring queries by a sparsity measure) and
a linear head replacing the generative decoder. Input projection to
d_model=64, 2 encoder layers, per paper.

NOTE: this is a simplified ProbSparse approximation for reproducibility
purposes (full KL-divergence-based query sampling from the original Informer
paper is approximated here by a top-k query-norm heuristic); the paper does
not give implementation-level detail beyond "for each query, only a subset
of keys are used ... queries with the highest sparsity scores are selected".
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from arxivist_artemis.models.baselines.transformer import _PositionalEncoding


class _ProbSparseAttention(nn.Module):
    """Simplified ProbSparse self-attention: top-u queries (by max-key-score
    sparsity measure) get full attention; others receive the mean value."""

    def __init__(self, d_model: int, n_heads: int, factor: int = 5) -> None:
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.factor = factor
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,L,d]
        K = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        u = min(L, max(1, int(self.factor * math.log(max(L, 2)))))
        scores_full = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,L,L]
        sparsity = scores_full.max(dim=-1).values - scores_full.mean(dim=-1)  # [B,H,L]
        _, top_idx = torch.topk(sparsity, u, dim=-1)  # [B,H,u]

        attn = torch.softmax(scores_full, dim=-1)
        out_full = torch.matmul(attn, V)  # [B,H,L,d]
        # Non-top queries get the mean value (Informer's default fill for un-selected queries).
        mean_v = V.mean(dim=2, keepdim=True).expand_as(out_full)
        mask = torch.zeros(B, self.n_heads, L, 1, device=x.device)
        mask.scatter_(2, top_idx.unsqueeze(-1), 1.0)
        out = mask * out_full + (1 - mask) * mean_v

        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)


class _InformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn = _ProbSparseAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attn(x)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class InformerBaseline(nn.Module):
    """
    Args:
        d_x: input feature dimension.
        d_model: explicitly stated in paper: 64.
        n_layers: explicitly stated in paper: 2.
        n_heads, ffn_dim, dropout: not individually specified for Informer;
            ASSUMED consistent with the encoder-layer conventions used
            elsewhere in the paper (8 heads, dim 256, 0.1).
    """

    def __init__(
        self, d_x: int, d_model: int = 64, n_layers: int = 2, n_heads: int = 8,
        ffn_dim: int = 256, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(d_x, d_model)
        self.pos_encoding = _PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [_InformerEncoderLayer(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_masked = x * mask
        h = self.input_proj(x_masked)
        h = self.pos_encoding(h)
        for layer in self.layers:
            h = layer(h)
        return self.head(h[:, -1, :]).squeeze(-1)
