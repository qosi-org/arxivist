"""
gmlp/models/tiny_attn.py
------------------------
Tiny single-head self-attention for the aMLP hybrid architecture.

Paper Section 4.3 + Figure 6: "Pay Attention to MLPs" (arXiv:2105.08050)

In aMLP, a *tiny* single-head self-attention is attached to the gating
function of the SGU. The design hypothesis is that gMLP already handles
most spatial interactions; the tiny attention only needs to model
cross-sentence alignment (Appendix D shows it attends across [SEP] tokens).

Key contrast with BERT's self-attention:
  - BERT: 12 heads × 64 dim = 768-dim attention
  - aMLP:  1 head  × 64 dim =  64-dim attention (12× smaller)

Implements SIR Eq. 7: A = softmax(QK^T / sqrt(d_attn)), out = AV

Paper ref: Section 4.3, Figure 6, Table 5/6
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TinyAttention(nn.Module):
    """
    Single-head scaled dot-product attention used in aMLP.

    Takes the pre-block LayerNorm output (x_pre) as input — NOT the
    post-expansion Z. This is explicitly shown in Figure 6 caption:
    "We use the normalized input of the gMLP block (endpoint after the
    input normalization and right before the channel expansion) as the
    input to the tiny self-attention."

    The output is projected to d_out = d_ffn // 2, matching the SGU
    gate branch shape, so it can be fused additively.

    Args:
        d_model:  Input dimension (pre-block norm output dim).
        d_attn:   Hidden dim for Q/K/V projections. Paper: 64 or 128.
        d_out:    Output projection dim. Must equal d_ffn // 2 for fusion.
                  Defaults to d_attn if None.

    Paper ref: Section 4.3, Figure 6, SIR eq7
    """

    def __init__(
        self,
        d_model: int,
        d_attn: int = 64,
        d_out: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.d_attn = d_attn
        self.d_out = d_out if d_out is not None else d_attn
        self.scale = math.sqrt(d_attn)

        # Project input to Q, K, V simultaneously (single projection)
        # Paper pseudocode Fig 6: qkv = proj(x, 3 * d_attn, axis="channel")
        self.qkv_proj = nn.Linear(d_model, 3 * d_attn, bias=False)

        # Output projection back to d_out (= d_ffn // 2 for SGU fusion)
        # Paper pseudocode Fig 6: return proj(x, d_out, axis="channel")
        self.out_proj = nn.Linear(d_attn, self.d_out, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute single-head attention.

        SIR Eq. 7:
            A = softmax(QK^T / sqrt(d_attn))
            out = AV

        Args:
            x: Tensor of shape [B, n, d_model] — pre-block LayerNorm output

        Returns:
            Tensor of shape [B, n, d_out]
        """
        assert x.dim() == 3, f"[TinyAttention] Expected [B, n, d_model], got {x.shape}"
        B, n, _ = x.shape

        # Project to QKV: [B, n, 3*d_attn]
        qkv = self.qkv_proj(x)
        # Split along channel dim: each [B, n, d_attn]
        q, k, v = qkv.chunk(3, dim=-1)

        # Scaled dot-product attention: [B, n, n]
        # SIR Eq. 7: w = einsum("bnd,bmd->bnm", q, k) → softmax(w * rsqrt(d_attn))
        attn_scores = torch.bmm(q, k.transpose(-1, -2)) / self.scale   # [B, n, n]
        attn_weights = F.softmax(attn_scores, dim=-1)                   # [B, n, n]

        # Weighted sum of values: [B, n, d_attn]
        attn_out = torch.bmm(attn_weights, v)

        # Project to output dim: [B, n, d_out]
        return self.out_proj(attn_out)

    def __repr__(self) -> str:
        return (
            f"TinyAttention(d_attn={self.d_attn}, d_out={self.d_out})"
        )


# Avoid NameError for Optional in older Python
from typing import Optional
