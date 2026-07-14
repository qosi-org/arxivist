"""Reference implementation of the order-2 Hyena operator.

Implements HyenaDNA Sec 3 / Hyena (Poli et al. 2023): a data-controlled
gating of implicit long convolutions evaluated via FFT in O(L log L).

WARNING: low-confidence implementation. The implicit-filter MLP internals
(SIR confidence 0.55) are inferred from the Hyena reference, NOT stated in this
paper. This module is a *reference / fallback* path — the reproduction critical
path uses official pretrained weights (see pretrained.py). Numerics here will
NOT exactly match the published checkpoints.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def fft_conv(u: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Causal long convolution of signal u with filter k via FFT.

    Args:
        u: [B, D, L] input signal.
        k: [D, L] filter (one per channel).
    Returns:
        [B, D, L] convolved signal (causal, same length).
    """
    L = u.shape[-1]
    fft_len = 2 * L
    u_f = torch.fft.rfft(u.float(), n=fft_len)
    k_f = torch.fft.rfft(k.float(), n=fft_len)
    y = torch.fft.irfft(u_f * k_f, n=fft_len)[..., :L]
    return y.to(u.dtype)


class HyenaFilter(nn.Module):
    """Implicit long filter parameterized by an MLP over positional features.

    ASSUMED structure (SIR ambiguity, conf 0.55): sinusoidal positional
    embedding -> small MLP -> per-channel filter, modulated by an exponential
    decay window.
    """

    def __init__(self, d_model: int, filter_order: int = 64, n_bands: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_bands = n_bands
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_bands + 1, filter_order),
            nn.GELU(),
            nn.Linear(filter_order, d_model),
        )
        # Learnable per-channel decay for the window.
        self.decay = nn.Parameter(torch.linspace(0.5, 3.0, d_model))

    def __repr__(self) -> str:  # noqa: D105
        return f"HyenaFilter(d_model={self.d_model}, n_bands={self.n_bands})"

    def positional_features(self, L: int, device: torch.device) -> torch.Tensor:
        t = torch.linspace(0, 1, L, device=device).unsqueeze(-1)  # [L, 1]
        bands = 2.0 ** torch.arange(self.n_bands, device=device) * math.pi
        feats = torch.cat([t, torch.sin(t * bands), torch.cos(t * bands)], dim=-1)
        return feats  # [L, 2*n_bands+1]

    def forward(self, L: int, device: torch.device) -> torch.Tensor:
        feats = self.positional_features(L, device)          # [L, F]
        h = self.mlp(feats).transpose(0, 1)                  # [D, L]
        t = torch.linspace(0, 1, L, device=device)           # [L]
        window = torch.exp(-self.decay.unsqueeze(-1) * t.unsqueeze(0))  # [D, L]
        return h * window                                    # [D, L]


class HyenaOperator(nn.Module):
    """Order-2 Hyena operator.

    Eq. (Hyena): y = x2 * FFTConv(h, x1 * FFTConv(h0, x0)), with x0,x1,x2 from
    a short-conv projection of the input.
    """

    def __init__(self, d_model: int, short_filter_len: int = 3) -> None:
        super().__init__()
        self.d_model = d_model
        self.order = 2  # SIR: order 2
        self.in_proj = nn.Linear(d_model, 3 * d_model)
        # Short depthwise conv on the projected features (causal).
        self.short_conv = nn.Conv1d(
            3 * d_model, 3 * d_model, kernel_size=short_filter_len,
            groups=3 * d_model, padding=short_filter_len - 1,
        )
        self.filter1 = HyenaFilter(d_model)
        self.filter2 = HyenaFilter(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def __repr__(self) -> str:  # noqa: D105
        return f"HyenaOperator(d_model={self.d_model}, order={self.order})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, f"Expected [B, L, D], got {tuple(x.shape)}"
        B, L, D = x.shape
        u = self.in_proj(x).transpose(1, 2)          # [B, 3D, L]
        u = self.short_conv(u)[..., :L]              # causal short conv -> [B, 3D, L]
        x0, x1, x2 = u.chunk(3, dim=1)               # each [B, D, L]

        h0 = self.filter1(L, x.device)              # [D, L]
        h1 = self.filter2(L, x.device)              # [D, L]
        v = x1 * fft_conv(x0, h0)                     # gated long conv (order 1)
        y = x2 * fft_conv(v, h1)                      # gated long conv (order 2)
        y = y.transpose(1, 2)                         # [B, L, D]
        return self.out_proj(y)


class HyenaBlock(nn.Module):
    """Pre-norm Hyena block: LN -> Hyena -> residual -> LN -> FFN -> residual.

    ASSUMED pre-norm residual (SIR conf 0.7); FFN 4x reverse bottleneck (conf 0.75).
    """

    def __init__(self, d_model: int, ffn_mult: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.hyena = HyenaOperator(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def __repr__(self) -> str:  # noqa: D105
        return f"HyenaBlock(d_model={self.norm1.normalized_shape[0]})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.hyena(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
