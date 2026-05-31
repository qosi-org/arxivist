"""
Dilated Causal Convolution residual block.
Implements Eq. 24-26 from Section 4.2 (Moreno-Pino & Zohren 2024).
Architecture inspired by WaveNet (Oord et al. 2016).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedCausalConv1d(nn.Module):
    """
    Single dilated causal convolution.
    Implements Eq. 24 (layer 1) and Eq. 25 (layer l):
        F^(l)(t) = sum_{tau=0}^{s-1} k^(l)_tau * F^(l-1)_{t - d*tau}
    Causality enforced by left-padding only.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation  # causal: pad left only
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)

    def __repr__(self):
        return f"DilatedCausalConv1d(in={self.conv.in_channels}, out={self.conv.out_channels}, dilation={self.conv.dilation})"


class DCCBlock(nn.Module):
    """
    WaveNet-style residual block with gated activation (ASSUMED: tanh*sigmoid),
    residual connection, and skip output.
    Implements the hierarchical DCC structure from Section 4.2 / Eq. 25-26.

    Args:
        residual_channels: Width of residual stream (paper Table 1: 32)
        dilation_channels: Internal DCC width (paper Table 1: 64)
        skip_channels: Skip connection width (paper Table 1: 128)
        kernel_size: Convolutional kernel size (paper Table 1: 3)
        dilation: Dilation factor for this layer (2^l)
    """
    def __init__(
        self,
        residual_channels: int,
        dilation_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
    ):
        super().__init__()
        # Dilated causal conv projecting to 2x dilation_channels for gated activation
        # ASSUMED: gated tanh*sigmoid (WaveNet style); confidence=0.78
        self.dilated_conv = DilatedCausalConv1d(
            residual_channels, 2 * dilation_channels, kernel_size, dilation
        )
        # 1x1 convolutions for residual and skip projections (Eq. 26)
        self.residual_proj = nn.Conv1d(dilation_channels, residual_channels, kernel_size=1)
        self.skip_proj = nn.Conv1d(dilation_channels, skip_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, residual_channels, L]
        Returns:
            residual: [B, residual_channels, L]
            skip:     [B, skip_channels, L]
        """
        assert x.dim() == 3, f"Expected [B, C, L], got {x.shape}"
        h = self.dilated_conv(x)                    # [B, 2*dilation_channels, L]
        # Gated activation: tanh(h1) * sigmoid(h2) — ASSUMED: WaveNet style
        h_tanh, h_sig = h.chunk(2, dim=1)           # each [B, dilation_channels, L]
        h = torch.tanh(h_tanh) * torch.sigmoid(h_sig)  # [B, dilation_channels, L]
        residual = self.residual_proj(h) + x        # [B, residual_channels, L]
        skip = self.skip_proj(h)                    # [B, skip_channels, L]
        return residual, skip

    def __repr__(self):
        return (f"DCCBlock(residual_ch={self.residual_proj.out_channels}, "
                f"dilation={self.dilated_conv.conv.dilation})")
