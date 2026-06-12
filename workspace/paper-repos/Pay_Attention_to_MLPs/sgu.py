"""
gmlp/models/sgu.py
------------------
Spatial Gating Unit (SGU) and aMLP hybrid SGU.

Paper Section 2.1 + Figure 1: "Pay Attention to MLPs" (arXiv:2105.08050)

The SGU is the central innovation of gMLP. It replaces self-attention for
cross-token communication using a *static* spatial projection W ∈ R^{n×n}
combined with multiplicative gating.

Key design choices (Section 2.1):
1. Channel split: Z is divided into Z1 (bypass) and Z2 (spatial branch)
   → each carries different information, reduces interaction order
2. Near-zero W init + ones bias → s(Z) ≈ Z at init (each block = plain FFN)
3. LayerNorm on Z2 before spatial projection → training stability for large NLP

SIR Eq. 3 (simple): s(Z) = Z ⊙ f_{W,b}(Z)
SIR Eq. 4 (SGU):    s(Z) = Z1 ⊙ f_{W,b}(Z2),  Z = Z1 || Z2

The aMLP variant (Section 4.3) adds a tiny self-attention to the gate:
  gate = spatial_proj(Z2) + tiny_attn(X_pre)       ← additive fusion (ASSUMED)
  # TODO: verify fusion mechanism from paper — SIR ambiguity_001 (conf=0.75)

Paper ref: Section 2.1, Fig 1, Fig 6, SIR equations 3-4
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

from .toeplitz import ToeplitzLinear
from .tiny_attn import TinyAttention


class SpatialGatingUnit(nn.Module):
    """
    Spatial Gating Unit (SGU) — the core cross-token mixing layer in gMLP.

    Splits input Z into two halves along the channel axis:
      - Z1: multiplicative bypass (gate input) — passes through unmodified
      - Z2: spatial branch — mixed across tokens via W ∈ R^{n×n}

    Returns Z1 ⊙ (W @ Z2 + b) — element-wise product.

    The interaction order of the output is 2nd-order (z_i * z_j terms),
    compared to 3rd-order for self-attention (Section 2.1).

    Args:
        d_ffn:          Total channel dimension of input Z (will be split to d_ffn//2).
        seq_len:        Sequence length n. Determines W shape.
        use_toeplitz:   Toeplitz constraint on W (True for NLP, False for vision).
        w_init_std:     Near-zero init std for W.
                        # ASSUMED: 0.002 (SIR ambiguity_002, conf=0.65) TODO:verify

    Paper ref: Section 2.1, Figure 1, SIR eq4
    """

    def __init__(
        self,
        d_ffn: int,
        seq_len: int,
        use_toeplitz: bool = True,
        w_init_std: float = 0.002,   # ASSUMED — SIR ambiguity_002
    ) -> None:
        super().__init__()
        assert d_ffn % 2 == 0, f"d_ffn must be even for channel split, got {d_ffn}"
        self.d_ffn = d_ffn
        self.d_half = d_ffn // 2

        # LayerNorm applied to Z2 before spatial projection (stability for large NLP)
        # Paper Figure 1 pseudocode: v = norm(v, axis="channel")
        self.norm = nn.LayerNorm(self.d_half)

        # Spatial projection: W ∈ R^{n×n}, b ∈ R^{n}
        # Implements f_{W,b}(Z2) from SIR Eq. 2
        self.spatial_proj = ToeplitzLinear(
            seq_len=seq_len,
            use_toeplitz=use_toeplitz,
            w_init_std=w_init_std,
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Implements SIR Eq. 4: s(Z) = Z1 ⊙ f_{W,b}(Z2)

        Args:
            z: Tensor of shape [B, n, d_ffn]

        Returns:
            Tensor of shape [B, n, d_ffn//2]
        """
        assert z.dim() == 3, f"[SGU] Expected [B, n, d_ffn], got {z.shape}"
        assert z.shape[-1] == self.d_ffn, (
            f"[SGU] Channel dim mismatch: expected {self.d_ffn}, got {z.shape[-1]}"
        )

        # SIR Eq. 4: split Z into Z1 || Z2 along channel axis
        z1, z2 = z.chunk(2, dim=-1)        # each [B, n, d_ffn/2]

        # Normalise Z2 before spatial projection (stability)
        z2 = self.norm(z2)                 # [B, n, d_ffn/2]

        # Apply spatial projection: W @ z2 + b
        z2 = self.spatial_proj(z2)         # [B, n, d_ffn/2]

        # Multiplicative gate: Z1 ⊙ f_{W,b}(Z2)
        return z1 * z2                     # [B, n, d_ffn/2]

    def __repr__(self) -> str:
        return (
            f"SpatialGatingUnit(d_ffn={self.d_ffn}, "
            f"use_toeplitz={self.spatial_proj.use_toeplitz})"
        )


class aMLP_SGU(nn.Module):
    """
    Hybrid Spatial Gating Unit for aMLP (gMLP + Tiny Attention).

    Extends SGU by adding a TinyAttention branch to the gating signal.
    The spatial gate is augmented with attention output:
        spatial_gate = ToeplitzLinear(Z2)         [B, n, d_ffn/2]
        attn_gate    = TinyAttention(X_pre)       [B, n, d_ffn/2]
        gate         = spatial_gate + attn_gate   ← fusion_mode='add'
        output       = Z1 * gate

    ⚠ WARNING: fusion mechanism is ASSUMED based on Fig 6 diagram.
    # TODO: verify from paper — SIR ambiguity_001 (confidence=0.75)
    The `fusion_mode` config parameter allows switching to 'concat' or 'replace'
    if the additive assumption proves incorrect.

    Args:
        d_ffn:          Total channel dim of input Z.
        d_model:        Model hidden dim (input to TinyAttention from X_pre).
        seq_len:        Sequence length n.
        d_attn:         Tiny attention hidden dim. Paper uses 64 or 128.
        use_toeplitz:   Toeplitz constraint on spatial W.
        w_init_std:     W init std. # ASSUMED: 0.002
        fusion_mode:    How to combine spatial and attention gates.
                        'add' (default/assumed), 'concat', 'replace'.
                        # ASSUMED: 'add' — SIR ambiguity_001, conf=0.75

    Paper ref: Section 4.3, Figure 6, SIR architecture.modules[3]
    """

    def __init__(
        self,
        d_ffn: int,
        d_model: int,
        seq_len: int,
        d_attn: int = 64,
        use_toeplitz: bool = True,
        w_init_std: float = 0.002,       # ASSUMED — SIR ambiguity_002
        fusion_mode: str = "add",        # ASSUMED — SIR ambiguity_001 TODO:verify
    ) -> None:
        super().__init__()
        assert d_ffn % 2 == 0, f"d_ffn must be even for channel split, got {d_ffn}"
        assert fusion_mode in ("add", "concat", "replace"), (
            f"fusion_mode must be 'add'/'concat'/'replace', got '{fusion_mode}'"
        )
        self.d_ffn = d_ffn
        self.d_half = d_ffn // 2
        self.fusion_mode = fusion_mode

        self.norm = nn.LayerNorm(self.d_half)

        self.spatial_proj = ToeplitzLinear(
            seq_len=seq_len,
            use_toeplitz=use_toeplitz,
            w_init_std=w_init_std,
        )

        # TinyAttention projects d_model → d_out (= d_half for 'add'/'replace')
        # For 'concat': d_out = d_half, then a fusion projection halves back
        self.tiny_attn = TinyAttention(
            d_model=d_model,
            d_attn=d_attn,
            d_out=self.d_half,
        )

        # Fusion projection only needed for 'concat' mode
        if fusion_mode == "concat":
            self.fusion_proj = nn.Linear(2 * self.d_half, self.d_half, bias=False)

    def forward(self, z: Tensor, x_pre: Tensor) -> Tensor:
        """
        Compute aMLP hybrid gate.

        Args:
            z:     Tensor [B, n, d_ffn]   — post-GeLU expanded activations
            x_pre: Tensor [B, n, d_model] — pre-block LayerNorm output (input to TinyAttn)
                   Paper Fig.6: "normalized input of the gMLP block... right before channel expansion"

        Returns:
            Tensor [B, n, d_ffn//2]
        """
        assert z.dim() == 3, f"[aMLP_SGU] Expected z: [B, n, d_ffn], got {z.shape}"
        assert x_pre.dim() == 3, f"[aMLP_SGU] Expected x_pre: [B, n, d_model], got {x_pre.shape}"

        z1, z2 = z.chunk(2, dim=-1)    # each [B, n, d_half]

        # Spatial gate branch
        z2_normed = self.norm(z2)
        spatial_gate = self.spatial_proj(z2_normed)    # [B, n, d_half]

        # Tiny attention gate branch (from pre-block normalized input)
        attn_gate = self.tiny_attn(x_pre)              # [B, n, d_half]

        # Fuse spatial and attention gates
        # WARNING: fusion_mode='add' is ASSUMED — SIR ambiguity_001 (conf=0.75)
        if self.fusion_mode == "add":
            gate = spatial_gate + attn_gate            # [B, n, d_half]
        elif self.fusion_mode == "replace":
            # Pure attention gate — no spatial projection in gate
            gate = attn_gate
        else:  # concat
            gate = self.fusion_proj(
                torch.cat([spatial_gate, attn_gate], dim=-1)
            )                                          # [B, n, d_half]

        return z1 * gate                               # [B, n, d_half]

    def __repr__(self) -> str:
        return (
            f"aMLP_SGU(d_ffn={self.d_ffn}, "
            f"d_attn={self.tiny_attn.d_attn}, "
            f"fusion_mode='{self.fusion_mode}')"
        )
