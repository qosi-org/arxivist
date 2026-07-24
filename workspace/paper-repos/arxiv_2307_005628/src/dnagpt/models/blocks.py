"""Building blocks for DNAGPT (Sec 2.1, Fig 1c/d).

A classic GPT decoder block (causal masked multi-head self-attention + FFN +
LayerNorm), plus the two DNAGPT-specific pieces that let one model handle both
sequence and numerical data:

* ``NumericalEmbedding`` — an MLP that maps raw scalar values into the D-dim
  token space, co-trained with the DNA sequence embeddings.
* ``RegressionHead`` — an MLP that decodes number tokens back to scalars.

The ``ClassificationHead`` is a plain linear map over the token vocabulary.
Everything is plain PyTorch and runs on CPU.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequentialEmbedding(nn.Module):
    """Embed DNA-sequence / special tokens into D channels (Fig 1d, left)."""

    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb(ids)


class NumericalEmbedding(nn.Module):
    """Map raw scalar values -> D-dim embeddings (Fig 1d, right).

    The paper feeds numbers "directly into a Numerical Embedding Layer" (MLP)
    that is co-trained with the DNA embeddings. Shapes are unspecified in the
    paper (# ASSUMED: 1 -> D -> D, GELU).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, numbers: torch.Tensor) -> torch.Tensor:
        # numbers: [B, M] -> [B, M, 1] -> [B, M, D]
        return self.mlp(numbers.unsqueeze(-1))


class RegressionHead(nn.Module):
    """Decode number tokens back to scalars (Fig 1d, MSE target). # ASSUMED MLP D->D->1."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)


class CausalSelfAttention(nn.Module):
    """Masked (unidirectional) multi-head self-attention — the GPT attention."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each [B, H, T, hd]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, T, T]
        # causal mask: position t attends only to <= t
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal, float("-inf"))
        if attn_mask is not None:  # [B, T] padding mask (1 = keep)
            pad = (~attn_mask.bool()).view(b, 1, 1, t)
            att = att.masked_fill(pad, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)
        out = (att @ v).transpose(1, 2).reshape(b, t, d)
        return self.proj(out)


class GPTBlock(nn.Module):
    """A GPT decoder block: LN -> causal attn -> residual; LN -> FFN -> residual.

    Bias-free (paper Fig S3 "Bias: False"). Layout mirrors the official DNAGPT
    checkpoint (GPT-2-style ``ln_1/attn/ln_2/mlp``) so its weights can be
    key-mapped in ``DNAGPT.from_pretrained``.
    """

    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, bias=False)
        self.attn = CausalSelfAttention(dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=False),
            nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x
