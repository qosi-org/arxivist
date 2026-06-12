"""
gmlp/models/gmlp_block.py
-------------------------
Single gMLP residual block.

Paper Section 2 + Figure 1: "Pay Attention to MLPs" (arXiv:2105.08050)

Each block implements:
    Z  = GeLU(X_pre @ U)           channel expand
    Z̃  = SGU(Z)   or aMLP_SGU(Z)  spatial gating
    Y  = Z̃ @ V                     channel contract
    out = Y + shortcut              residual add

SIR Eq. 1: Z = σ(XU),  Z̃ = s(Z),  Y = Z̃V

Stochastic depth (DropPath) is applied in vision models only.
NLP models use survival_prob=1.0 (disabled).

Paper ref: Section 2, Fig 1, Table 1 (vision), Table 5 (NLP)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .sgu import SpatialGatingUnit, aMLP_SGU


# ---------------------------------------------------------------------------
# Stochastic depth (DropPath) — vision only
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """
    Stochastic depth: randomly drop entire residual branch during training.
    Introduced by Huang et al. 2016, used in DeiT and here for vision gMLP.
    Applied per-sample in the batch (not per-channel).

    Paper Table 1: survival_prob ∈ {1.00, 0.95, 0.80} for Ti/S/B.

    Args:
        survival_prob: Probability of keeping the branch (1.0 = no drop).
    """

    def __init__(self, survival_prob: float = 1.0) -> None:
        super().__init__()
        self.survival_prob = survival_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.survival_prob == 1.0:
            return x
        B = x.shape[0]
        # Bernoulli mask: shape [B, 1, 1] → broadcasts over [B, n, d]
        keep = torch.rand(B, 1, 1, device=x.device, dtype=x.dtype) < self.survival_prob
        # Scale kept samples to maintain expected value
        return x * keep / self.survival_prob

    def __repr__(self) -> str:
        return f"DropPath(survival_prob={self.survival_prob})"


# ---------------------------------------------------------------------------
# gMLPBlock
# ---------------------------------------------------------------------------

class gMLPBlock(nn.Module):
    """
    Single gMLP residual block.

    Forward pass (SIR Eq. 1):
        shortcut = x
        x_pre    = LayerNorm(x)                  pre-norm
        x        = GeLU(x_pre @ U)               channel expand: d_model → d_ffn
        x        = SGU(x)  [or aMLP_SGU(x, x_pre)]  spatial gating → d_ffn/2
        x        = x @ V                          channel contract: d_ffn/2 → d_model
        x        = DropPath(x) + shortcut         residual (DropPath for vision)

    Args:
        d_model:       Token hidden dimension.
        d_ffn:         FFN expansion dim (≈ 4–6× d_model).
        seq_len:       Sequence length n (needed for W shape).
        use_toeplitz:  Toeplitz constraint on spatial W (True for NLP).
        use_tiny_attn: If True, use aMLP_SGU instead of SGU.
        d_attn:        Tiny attention dim (aMLP only).
        w_init_std:    W near-zero init std.
                       # ASSUMED: 0.002 (SIR ambiguity_002, conf=0.65)
        attn_fusion_mode: Gate fusion for aMLP.
                       # ASSUMED: 'add' (SIR ambiguity_001, conf=0.75)
        survival_prob: DropPath survival probability (1.0 = disabled).

    Paper ref: Section 2, Figure 1, SIR architecture.modules[0]
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        seq_len: int,
        use_toeplitz: bool = True,
        use_tiny_attn: bool = False,
        d_attn: int = 64,
        w_init_std: float = 0.002,         # ASSUMED — SIR ambiguity_002
        attn_fusion_mode: str = "add",     # ASSUMED — SIR ambiguity_001
        survival_prob: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.use_tiny_attn = use_tiny_attn

        # Pre-norm (applied before channel expansion)
        self.norm = nn.LayerNorm(d_model)

        # Channel expansion: d_model → d_ffn  (U matrix in Eq. 1)
        self.channel_expand = nn.Linear(d_model, d_ffn)

        # GeLU activation (σ in Eq. 1)
        self.act = nn.GELU()

        # Spatial gating unit (s(·) in Eq. 1)
        if use_tiny_attn:
            self.sgu = aMLP_SGU(
                d_ffn=d_ffn,
                d_model=d_model,
                seq_len=seq_len,
                d_attn=d_attn,
                use_toeplitz=use_toeplitz,
                w_init_std=w_init_std,
                fusion_mode=attn_fusion_mode,
            )
        else:
            self.sgu = SpatialGatingUnit(
                d_ffn=d_ffn,
                seq_len=seq_len,
                use_toeplitz=use_toeplitz,
                w_init_std=w_init_std,
            )

        # Channel contraction: d_ffn/2 → d_model  (V matrix in Eq. 1)
        self.channel_contract = nn.Linear(d_ffn // 2, d_model)

        # Stochastic depth (vision only; NLP uses survival_prob=1.0)
        self.drop_path = DropPath(survival_prob)

    def forward(self, x: Tensor) -> Tensor:
        """
        Implements SIR Eq. 1: Z = σ(XU), Z̃ = s(Z), Y = Z̃V, out = Y + x

        Args:
            x: Tensor of shape [B, n, d_model]

        Returns:
            Tensor of shape [B, n, d_model]
        """
        assert x.dim() == 3, f"[gMLPBlock] Expected [B, n, d_model], got {x.shape}"
        assert x.shape[-1] == self.d_model, (
            f"[gMLPBlock] d_model mismatch: expected {self.d_model}, got {x.shape[-1]}"
        )

        shortcut = x

        # Pre-LayerNorm (Figure 1 pseudocode: x = norm(x, axis="channel"))
        x_pre = self.norm(x)                                    # [B, n, d_model]

        # Channel expand + GeLU  (Eq. 1: Z = σ(XU))
        x = self.channel_expand(x_pre)                         # [B, n, d_ffn]
        x = self.act(x)                                        # [B, n, d_ffn]

        # Spatial gating (Eq. 1: Z̃ = s(Z))
        if self.use_tiny_attn:
            # aMLP: pass both Z and X_pre (attention needs pre-block normalized input)
            x = self.sgu(x, x_pre)                            # [B, n, d_ffn/2]
        else:
            x = self.sgu(x)                                   # [B, n, d_ffn/2]

        # Channel contract (Eq. 1: Y = Z̃V)
        x = self.channel_contract(x)                          # [B, n, d_model]

        # Residual + optional DropPath
        return self.drop_path(x) + shortcut                   # [B, n, d_model]

    def __repr__(self) -> str:
        return (
            f"gMLPBlock(d_model={self.d_model}, d_ffn={self.d_ffn}, "
            f"use_tiny_attn={self.use_tiny_attn}, "
            f"sgu={self.sgu.__class__.__name__})"
        )
